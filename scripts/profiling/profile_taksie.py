"""
"""Profile TaKSIE on CALVIN.

Measures per-module: params, MACs, latency, VRAM.
Resolution: 256×256 (32×32 latent), matching cfg.yaml.

Usage:
    cd /home/chal2525/mal_prototype
    source .venv/bin/activate
    PATH=/usr/local/cuda-12/bin:$PATH \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    CUDA_VISIBLE_DEVICES=1 python scripts/profiling/profile_taksie.py \
        --n_reps 50 --device cuda:0
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "repos" / "TaKSIE"))
sys.path.insert(0, str(ROOT / "repos" / "calvin" / "calvin_models"))

# TaKSIE generates at 256×256 (cfg.yaml: resolution=256)
# → latent = 32×32×4 (SD 1.5 VAE factor = 8)
IMG_RES = 256
LATENT_RES = IMG_RES // 8  # 32
POLICY_RES = 200  # GC policy input resolution
REGEN_INTERVAL = 20  # max_per_frame from cfg.yaml

# ── Helpers ──────────────────────────────────────────────────────────────

def count_params(model) -> int:
    if isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters())
    return 0

def try_fvcore_macs(model, inputs) -> int | None:
    try:
        from fvcore.nn import FlopCountAnalysis
        a = FlopCountAnalysis(model, inputs)
        a.unsupported_ops_warnings(False)
        a.uncalled_modules_warnings(False)
        return int(a.total())
    except Exception as e:
        print(f"  [fvcore] MACs failed: {e}")
        return None

def latency_stats(times: list[float]) -> dict:
    a = np.array(times) * 1000
    return {
        "mean_ms": round(float(np.mean(a)), 3),
        "std_ms":  round(float(np.std(a)), 3),
        "min_ms":  round(float(np.min(a)), 3),
        "max_ms":  round(float(np.max(a)), 3),
        "p50_ms":  round(float(np.percentile(a, 50)), 3),
        "p95_ms":  round(float(np.percentile(a, 95)), 3),
    }

def benchmark_torch(fn, warmup=5, n_reps=50, dev_idx=0):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev_idx)
    times = []
    for _ in range(n_reps):
        torch.cuda.synchronize(dev_idx)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(dev_idx)
        times.append(time.perf_counter() - t0)
    return times

def benchmark_jax(fn, warmup=5, n_reps=30):
    import jax
    for _ in range(warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t0)
    return times

def vram_mb(dev_idx=0) -> float:
    return torch.cuda.max_memory_allocated(dev_idx) / 1e6

def fmt(macs):
    if macs is None: return "N/A"
    if macs >= 1e12: return f"{macs/1e12:.2f}T"
    if macs >= 1e9:  return f"{macs/1e9:.2f}G"
    if macs >= 1e6:  return f"{macs/1e6:.2f}M"
    return str(macs)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--running_config", type=str,
                        default=str(ROOT / "repos" / "TaKSIE" / "cfg" / "cfg.yaml"))
    parser.add_argument("--output", type=str,
                        default=str(ROOT / "results" / "profiling" / "profile_taksie.json"))
    args = parser.parse_args()

    import yaml
    device = torch.device(args.device)
    dev_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0

    with open(args.running_config) as f:
        cfg = yaml.safe_load(f)

    # Fix paths
    if not os.path.isabs(cfg.get("dgcbc_path", "")):
        cfg["dgcbc_path"] = os.path.join(str(ROOT), "models", "taksie_gcbc", cfg["dgcbc_path"])
    if not os.path.isabs(cfg.get("negative_prompt_yaml_path", "")):
        cfg["negative_prompt_yaml_path"] = os.path.join(
            str(ROOT), "repos", "TaKSIE", cfg["negative_prompt_yaml_path"])

    print(f"Resolution: {IMG_RES}×{IMG_RES}, Latent: {LATENT_RES}×{LATENT_RES}")
    print(f"Device: {args.device}")

    # Load TaKSIE wrapper (loads all models)
    from taksie.taksie_wrapper import taksie_wrapper
    print("\nLoading TaKSIE model...")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(dev_idx)
    model = taksie_wrapper(cfg, device=args.device)
    torch.cuda.synchronize(dev_idx)
    vram_after_load = vram_mb(dev_idx)
    print(f"  VRAM after load: {vram_after_load:.0f} MB")

    results = {"system": "taksie", "device": args.device,
               "resolution": IMG_RES, "n_benchmark_reps": args.n_reps}
    modules = {}

    # ── 1. CLIP Vision Encoder (ViT-L/14) ────────────────────────────
    print("\n=== CLIP ViT-L/14 Vision Encoder ===")
    clip_model = model.visionModel.eval()
    clip_params = count_params(clip_model)
    dummy_pixel = torch.randn(1, 3, 224, 224, device=device)
    clip_macs = try_fvcore_macs(clip_model, (dummy_pixel,))

    torch.cuda.reset_peak_memory_stats(dev_idx)
    clip_times = benchmark_torch(
        lambda: clip_model(pixel_values=dummy_pixel), n_reps=args.n_reps, dev_idx=dev_idx)
    modules["clip_vision_encoder"] = {
        "params": clip_params, "macs": clip_macs,
        "latency": latency_stats(clip_times),
        "vram_peak_mb": round(vram_mb(dev_idx), 1),
        "schedule": "1x per regen",
    }
    print(f"  Params: {clip_params:,}, MACs: {fmt(clip_macs)}")
    print(f"  Latency: {modules['clip_vision_encoder']['latency']['mean_ms']:.2f} ms")

    # ── 2. LIV (ResNet-50, 224×224) ──────────────────────────────────
    print("\n=== LIV ResNet-50 ===")
    liv_model = model.liv.eval()
    liv_params = count_params(liv_model)
    dummy_liv = torch.randn(1, 3, 224, 224, device=device)
    liv_macs = try_fvcore_macs(liv_model, (dummy_liv,))
    if liv_macs is None:
        liv_macs = int(4.1e9)

    torch.cuda.reset_peak_memory_stats(dev_idx)
    liv_times = benchmark_torch(
        lambda: liv_model(dummy_liv), n_reps=args.n_reps, dev_idx=dev_idx)
    modules["liv_resnet50"] = {
        "params": liv_params, "macs": liv_macs,
        "latency": latency_stats(liv_times),
        "vram_peak_mb": round(vram_mb(dev_idx), 1),
        "schedule": "1x per step (progress eval)",
    }
    print(f"  Params: {liv_params:,}, MACs: {fmt(liv_macs)}")
    print(f"  Latency: {modules['liv_resnet50']['latency']['mean_ms']:.2f} ms")

    # ── 3. SD Text Encoder ───────────────────────────────────────────
    print("\n=== SD Text Encoder ===")
    text_enc = model.pipe.text_encoder.eval()
    te_params = count_params(text_enc)
    dummy_tok = torch.randint(0, 49408, (1, 77), device=device)
    te_macs = try_fvcore_macs(text_enc, (dummy_tok,))

    torch.cuda.reset_peak_memory_stats(dev_idx)
    te_times = benchmark_torch(
        lambda: text_enc(dummy_tok), n_reps=args.n_reps, dev_idx=dev_idx)
    modules["sd_text_encoder"] = {
        "params": te_params, "macs": te_macs,
        "latency": latency_stats(te_times),
        "vram_peak_mb": round(vram_mb(dev_idx), 1),
        "schedule": "1x per regen",
    }
    print(f"  Params: {te_params:,}, MACs: {fmt(te_macs)}")
    print(f"  Latency: {modules['sd_text_encoder']['latency']['mean_ms']:.2f} ms")

    # ── 4. SD UNet (single fwd, batch=1, 32×32 latent) ──────────────
    print(f"\n=== SD UNet (single fwd, {LATENT_RES}×{LATENT_RES} latent) ===")
    unet = model.pipe.unet.eval()
    unet_params = count_params(unet)
    dummy_lat = torch.randn(1, 4, LATENT_RES, LATENT_RES, device=device)
    dummy_t = torch.tensor([500], device=device)
    dummy_enc = torch.randn(1, 77, 768, device=device)
    unet_macs = try_fvcore_macs(unet, (dummy_lat, dummy_t, dummy_enc))

    torch.cuda.reset_peak_memory_stats(dev_idx)
    unet_times = benchmark_torch(
        lambda: unet(dummy_lat, dummy_t, encoder_hidden_states=dummy_enc),
        n_reps=args.n_reps, dev_idx=dev_idx)
    modules["sd_unet_single_fwd"] = {
        "params": unet_params, "macs": unet_macs,
        "latency": latency_stats(unet_times),
        "vram_peak_mb": round(vram_mb(dev_idx), 1),
        "note": f"single fwd at {LATENT_RES}×{LATENT_RES}; gen = 50 steps × batch=3 (triple CFG)",
    }
    print(f"  Params: {unet_params:,}, MACs: {fmt(unet_macs)}")
    print(f"  Latency: {modules['sd_unet_single_fwd']['latency']['mean_ms']:.2f} ms")

    # ── 5. ControlNet (single fwd, batch=3 triple CFG) ──────────────
    print(f"\n=== ControlNet (single fwd, batch=3, {LATENT_RES}×{LATENT_RES}) ===")
    controlnet = model.pipe.controlnet.eval()
    cn_params = count_params(controlnet)

    B_cfg = 3
    dummy_lat_b3 = torch.randn(B_cfg, 4, LATENT_RES, LATENT_RES, device=device)
    dummy_t_b3 = torch.full((B_cfg,), 500, device=device, dtype=torch.long)
    dummy_enc_b3 = torch.randn(B_cfg, 77, 768, device=device)
    dummy_cond_b3 = torch.randn(B_cfg, 3, IMG_RES, IMG_RES, device=device)
    # class_labels: (seq_len, batch, 768) — trajectory CLIP features
    dummy_cls = torch.randn(2, B_cfg, 768, device=device)

    cn_macs_single = None
    try:
        # Provide class_labels for ControlNet MACs estimation
        dummy_lat_b1 = torch.randn(1, 4, LATENT_RES, LATENT_RES, device=device)
        dummy_t_b1 = torch.tensor([500], device=device, dtype=torch.long)
        dummy_enc_b1 = torch.randn(1, 77, 768, device=device)
        dummy_cond_b1 = torch.randn(1, 3, IMG_RES, IMG_RES, device=device)
        dummy_cls_b1 = torch.randn(2, 1, 768, device=device)
        controlnet.h = None
        cn_macs_single = try_fvcore_macs(
            controlnet, (dummy_lat_b1, dummy_t_b1, dummy_enc_b1, dummy_cond_b1, dummy_cls_b1))
    except Exception as e:
        print(f"  fvcore MACs estimation failed: {e}")
    if cn_macs_single is None:
        # Fallback: ControlNet ≈ 43% of UNet (encoder-only portion)
        cn_macs_single = int((unet_macs or 0) * 0.43)
        print(f"  Using fallback estimate: {fmt(cn_macs_single)}")

    torch.cuda.reset_peak_memory_stats(dev_idx)
    def cn_fn():
        controlnet.h = None
        controlnet(dummy_lat_b3, dummy_t_b3,
                    encoder_hidden_states=dummy_enc_b3,
                    controlnet_cond=dummy_cond_b3,
                    class_labels=dummy_cls)

    cn_times = benchmark_torch(cn_fn, n_reps=args.n_reps, dev_idx=dev_idx)
    modules["controlnet_single_fwd"] = {
        "params": cn_params,
        "macs_batch1": cn_macs_single,
        "latency_batch3": latency_stats(cn_times),
        "vram_peak_mb": round(vram_mb(dev_idx), 1),
        "note": f"batch=3 (triple CFG), {LATENT_RES}×{LATENT_RES} latent",
    }
    print(f"  Params: {cn_params:,}, MACs (batch=1): {fmt(cn_macs_single)}")
    print(f"  Latency (batch=3): {modules['controlnet_single_fwd']['latency_batch3']['mean_ms']:.2f} ms")

    # ── 6. VAE Decoder (32×32 → 256×256) ────────────────────────────
    print(f"\n=== SD VAE Decoder ({LATENT_RES}×{LATENT_RES} → {IMG_RES}×{IMG_RES}) ===")
    vae = model.pipe.vae.eval()
    vae_params = count_params(vae)
    vae_dec_macs = try_fvcore_macs(vae.decoder, (dummy_lat.narrow(1, 0, 4)[:1],))

    torch.cuda.reset_peak_memory_stats(dev_idx)
    dummy_lat_dec = torch.randn(1, 4, LATENT_RES, LATENT_RES, device=device)
    vae_times = benchmark_torch(
        lambda: vae.decode(dummy_lat_dec), n_reps=args.n_reps, dev_idx=dev_idx)
    modules["vae_decoder"] = {
        "params": vae_params,
        "macs": vae_dec_macs,
        "latency": latency_stats(vae_times),
        "vram_peak_mb": round(vram_mb(dev_idx), 1),
        "schedule": "1x per regen",
    }
    print(f"  Params: {vae_params:,}, MACs: {fmt(vae_dec_macs)}")
    print(f"  Latency: {modules['vae_decoder']['latency']['mean_ms']:.2f} ms")

    # ── 7. Full Subgoal Generation Pipeline ──────────────────────────
    print("\n=== Full Subgoal Generation (50 UniPC steps, triple CFG) ===")
    # Time the actual pipeline call with class_labels (CLIP trajectory features)
    from PIL import Image
    dummy_pil = Image.fromarray(np.random.randint(0, 255, (IMG_RES, IMG_RES, 3), dtype=np.uint8))
    prompt = "open the drawer"
    neg_prompts = getattr(model, 'neg_prompts', None)
    if neg_prompts is None:
        neg_prompts = ["low quality, blurry"]

    # Build a dummy CLIP trajectory feature (simulating 2 keyframes seen so far)
    # TaKSIE does: feature_list.append(get_img_feature(obs)), then cat, then triple
    with torch.no_grad():
        dummy_clip_feat = model.visionModel(
            pixel_values=torch.randn(1, 3, 224, 224, device=device)
        ).image_embeds.unsqueeze(0)  # (1, 1, 768)
    # Simulate 2 keyframes: (2, 1, 768) → tripled to (2, 3, 768)
    dummy_traj = dummy_clip_feat.repeat(2, 1, 1)  # (2, 1, 768)
    dummy_traj_tripled = torch.cat([dummy_traj, dummy_traj, dummy_traj], dim=1)  # (2, 3, 768)

    guidance_scale = getattr(model, 'guidance_scale', 2.5)
    image_guidance_scale = getattr(model, 'image_guidance_scale', 2.5)
    num_inference_steps = getattr(model, 'num_inference_steps', 50)

    def gen_fn():
        with torch.no_grad():
            model.pipe(
                prompt=prompt,
                image=dummy_pil.resize((IMG_RES, IMG_RES)),
                num_inference_steps=num_inference_steps,
                negative_prompt=neg_prompts[0] if isinstance(neg_prompts, list) else neg_prompts,
                class_label=dummy_traj_tripled,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
            )

    print("  Warming up...")
    gen_fn()
    gen_fn()

    torch.cuda.reset_peak_memory_stats(dev_idx)
    gen_times = benchmark_torch(gen_fn, warmup=2, n_reps=min(args.n_reps, 15), dev_idx=dev_idx)

    # Total MACs for generation: 50 steps × 3 batch × (UNet + ControlNet) + VAE dec + text enc + CLIP vis
    num_steps = 50
    gen_total_macs = (
        num_steps * 3 * ((unet_macs or 0) + (cn_macs_single or 0)) +  # 50×3×(UNet+CN)
        (vae_dec_macs or 0) +
        (te_macs or 0) +
        (clip_macs or 0) +
        (liv_macs or 0)  # LIV encodes the generated goal
    )
    modules["full_generation"] = {
        "total_macs": gen_total_macs,
        "latency": latency_stats(gen_times),
        "vram_peak_mb": round(vram_mb(dev_idx), 1),
        "detail": f"50×3×(UNet+CN) + VAE_dec + Text_enc + CLIP_vis + LIV(goal)",
    }
    print(f"  Total MACs: {fmt(gen_total_macs)}")
    print(f"  Latency: {modules['full_generation']['latency']['mean_ms']:.1f} ms")

    # ── 8. GCBC JAX Diffusion Policy (ResNet-50-bridge, 20 DDPM) ─────
    print("\n=== GCBC Diffusion Policy (ResNet-50-bridge, 20 DDPM steps) ===")
    import jax
    import jax.numpy as jnp

    gcbc_params = sum(x.size for x in jax.tree_util.tree_leaves(model.agent.state.params))
    print(f"  Params: {gcbc_params:,}")

    # Compute MACs for ResNet-50 bridge (6ch, 200×200)
    r50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
    r50.conv1 = torch.nn.Conv2d(6, 64, 7, stride=2, padding=3, bias=False)
    r50.eval()
    r50_macs = try_fvcore_macs(r50, (torch.randn(1, 6, POLICY_RES, POLICY_RES),))
    del r50
    print(f"  ResNet-50-bridge MACs (6ch, {POLICY_RES}×{POLICY_RES}): {fmt(r50_macs)}")

    # MLP score net: small (~1M MACs)
    mlp_macs = 256 * 256 * 3 * 2  # estimate
    gcbc_total_macs = 20 * ((r50_macs or 0) + mlp_macs)
    print(f"  Total MACs (20 DDPM steps): {fmt(gcbc_total_macs)}")

    dummy_obs_jax = {"image": jnp.zeros((1, 1, POLICY_RES, POLICY_RES, 3), dtype=jnp.uint8)}
    dummy_goal_jax = {"image": jnp.zeros((1, POLICY_RES, POLICY_RES, 3), dtype=jnp.uint8)}

    print("  Warming up JAX...")
    _ = model.agent.sample_actions(dummy_obs_jax, dummy_goal_jax, seed=jax.random.PRNGKey(0))

    gcbc_times = benchmark_jax(
        lambda: model.agent.sample_actions(dummy_obs_jax, dummy_goal_jax, seed=jax.random.PRNGKey(0)),
        n_reps=min(args.n_reps, 30))

    modules["gcbc_diffusion_policy"] = {
        "params": int(gcbc_params),
        "macs_total_20_steps": gcbc_total_macs,
        "macs_encoder_single": r50_macs,
        "latency": latency_stats(gcbc_times),
        "schedule": "1x per step (20 DDPM denoising internally)",
        "encoder": f"ResNet-50-bridge (6ch early_goal_concat, {POLICY_RES}×{POLICY_RES})",
        "diffusion_steps": 20,
        "action_horizon": 4,
        "action_dim": 7,
    }
    print(f"  Latency: {modules['gcbc_diffusion_policy']['latency']['mean_ms']:.2f} ms")

    # ── Amortized Per-Step ───────────────────────────────────────────
    print("\n=== Amortized Per-Step Summary ===")
    policy_macs = gcbc_total_macs
    policy_lat = modules["gcbc_diffusion_policy"]["latency"]["mean_ms"]
    liv_per_step_macs = liv_macs or 0
    liv_per_step_lat = modules["liv_resnet50"]["latency"]["mean_ms"]
    gen_lat = modules["full_generation"]["latency"]["mean_ms"]

    per_step_fixed_macs = policy_macs + liv_per_step_macs
    per_step_fixed_lat = policy_lat + liv_per_step_lat
    amortized_macs = per_step_fixed_macs + gen_total_macs / REGEN_INTERVAL
    amortized_lat = per_step_fixed_lat + gen_lat / REGEN_INTERVAL

    total_params = clip_params + liv_params + unet_params + cn_params + vae_params + te_params + int(gcbc_params)
    peak_vram = max(modules["full_generation"]["vram_peak_mb"], vram_after_load)

    results["modules"] = modules
    results["amortized_per_step"] = {
        "macs": int(amortized_macs),
        "macs_formatted": fmt(int(amortized_macs)),
        "latency_ms": round(amortized_lat, 2),
        "vram_peak_mb": round(peak_vram, 1),
        "total_params": total_params,
        "breakdown": (
            f"policy({fmt(policy_macs)}) + LIV({fmt(liv_per_step_macs)}) "
            f"+ gen({fmt(gen_total_macs)})/{REGEN_INTERVAL}"
        ),
        "note": f"Adaptive regen: max {REGEN_INTERVAL} steps or LIV distance < 0.04",
    }

    print(f"  MACs:    {fmt(int(amortized_macs))}")
    print(f"  Latency: {amortized_lat:.2f} ms")
    print(f"  VRAM:    {peak_vram:.0f} MB")
    print(f"  Params:  {total_params:,}")

    # ── Save ─────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
