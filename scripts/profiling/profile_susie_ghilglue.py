"""
Profile SuSIE (pure) and GHIL-Glue (with VF filtering) on CALVIN.

Measures per-module: params, MACs, latency, VRAM.
Outputs separate JSONs for SuSIE (batch=1) and GHIL-Glue (batch=4 + VF).

Usage:
    cd /home/chal2525/mal_prototype/repos/ghil-glue
    source .venv/bin/activate
    PTXAS=$(dirname $(find .venv -path '*/nvidia/cuda_nvcc/bin/ptxas' -print -quit))
    PATH=$PTXAS:$PATH XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1" \
    CUDA_VISIBLE_DEVICES=0 python /home/chal2525/mal_prototype/scripts/profiling/profile_susie_ghilglue.py \
        --n_reps 50 --output_dir /home/chal2525/mal_prototype/results/profiling
"""
from __future__ import annotations
import argparse, json, os, sys, time
from functools import partial
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".45")

import numpy as np

GHIL_ROOT = Path("/home/chal2525/mal_prototype/repos/ghil-glue")
MAL_ROOT  = Path("/home/chal2525/mal_prototype")
CKPT_DIR  = MAL_ROOT / "models" / "ghil-glue"

sys.path.insert(0, str(GHIL_ROOT / "external" / "susie"))
sys.path.insert(0, str(GHIL_ROOT / "external" / "jaxrl_m"))

# ── Helpers ──────────────────────────────────────────────────────────────

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

def benchmark_jax(fn, warmup=5, n_reps=50):
    import jax
    for _ in range(warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t0)
    return times

def benchmark_tf_cpu(fn, warmup=5, n_reps=50):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times

def count_jax_params(params) -> int:
    import jax
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

def gpu_vram_mb() -> float:
    import subprocess
    pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader,nounits"], text=True, timeout=5
        ).strip()
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2 and int(parts[0]) == pid:
                return float(parts[1])
    except Exception:
        pass
    return 0.0

def fmt(macs):
    if macs is None: return "N/A"
    if macs >= 1e12: return f"{macs/1e12:.2f}T"
    if macs >= 1e9:  return f"{macs/1e9:.2f}G"
    if macs >= 1e6:  return f"{macs/1e6:.2f}M"
    return str(macs)


# ── Phase 1: MACs via fvcore (CPU, PyTorch) ──────────────────────────────

def compute_macs():
    """Compute MACs for all components using fvcore on PyTorch equivalents."""
    import torch
    from fvcore.nn import FlopCountAnalysis

    def _macs(model, inputs):
        a = FlopCountAnalysis(model, inputs)
        a.unsupported_ops_warnings(False)
        a.uncalled_modules_warnings(False)
        return int(a.total())

    macs = {}
    pretrained = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    # SuSIE UNet: instruct-pix2pix, 8-channel input, 32×32 latent (256px)
    print("[MACs] SuSIE UNet (8ch, 32×32)...")
    from diffusers import UNet2DConditionModel
    cfg = UNet2DConditionModel.load_config(pretrained, subfolder="unet")
    cfg["in_channels"] = 8
    unet = UNet2DConditionModel.from_config(cfg).eval()
    macs["unet_single"] = _macs(unet, (
        torch.randn(1, 8, 32, 32), torch.tensor([500]), torch.randn(1, 77, 768)
    ))
    del unet
    print(f"  UNet single fwd: {fmt(macs['unet_single'])}")

    # VAE encoder (256×256 → 32×32×4)
    print("[MACs] VAE encoder (256×256)...")
    from diffusers import AutoencoderKL
    vae_cfg = AutoencoderKL.load_config(pretrained, subfolder="vae")
    vae = AutoencoderKL.from_config(vae_cfg).eval()
    macs["vae_enc"] = _macs(vae.encoder, (torch.randn(1, 3, 256, 256),))
    macs["vae_dec"] = _macs(vae.decoder, (torch.randn(1, 4, 32, 32),))
    vae_params = sum(p.numel() for p in vae.parameters())
    macs["vae_enc_params"] = sum(p.numel() for p in vae.encoder.parameters())
    macs["vae_dec_params"] = sum(p.numel() for p in vae.decoder.parameters())
    macs["vae_total_params"] = vae_params
    del vae
    print(f"  VAE enc: {fmt(macs['vae_enc'])}")
    print(f"  VAE dec: {fmt(macs['vae_dec'])}")

    # CLIP text encoder
    print("[MACs] CLIP text encoder...")
    from transformers import CLIPTextModel, CLIPTextConfig
    clip_cfg = CLIPTextConfig.from_pretrained(pretrained, subfolder="text_encoder")
    clip = CLIPTextModel(clip_cfg).eval()
    macs["clip_text"] = _macs(clip, (torch.randint(0, 49408, (1, 77)),))
    macs["clip_text_params"] = sum(p.numel() for p in clip.parameters())
    del clip
    print(f"  CLIP text: {fmt(macs['clip_text'])}")

    # ResNet-34 bridge (6ch, 200×200) — GC policy encoder
    print("[MACs] ResNet-34-bridge (6ch, 200×200)...")
    import torchvision.models as tvm
    r34 = tvm.resnet34(weights=None)
    r34.conv1 = torch.nn.Conv2d(6, 64, 7, stride=2, padding=3, bias=False)
    r34.eval()
    macs["resnet34_6ch"] = _macs(r34, (torch.randn(1, 6, 200, 200),))
    del r34
    print(f"  ResNet-34 (6ch, 200×200): {fmt(macs['resnet34_6ch'])}")

    # MLP score network estimate: 3 ResNet blocks, hidden_dim=256, input ~512+128+28=~668
    # Very small compared to ResNet encoder
    macs["mlp_score_single"] = 256 * 256 * 3 * 2  # ~393K per block × 3 blocks
    print(f"  MLP score (per step, est.): {fmt(macs['mlp_score_single'])}")

    # ResNet-34 bridge (3ch, 200×200) — for VF encoders (×2)
    print("[MACs] ResNet-34-bridge (3ch, 200×200) for VF...")
    r34_3 = tvm.resnet34(weights=None).eval()
    r34_3.conv1 = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
    macs["resnet34_3ch"] = _macs(r34_3, (torch.randn(1, 3, 200, 200),))
    del r34_3
    print(f"  ResNet-34 (3ch, 200×200): {fmt(macs['resnet34_3ch'])}")

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return macs


# ── Phase 2: JAX component profiling ────────────────────────────────────

def profile_susie_components(n_reps, macs):
    """Profile SuSIE subgoal generation components (batch=1)."""
    import jax
    import jax.numpy as jnp
    from susie.jax_utils import initialize_compilation_cache
    initialize_compilation_cache()

    results = {}
    ckpt = str(CKPT_DIR / "SuSIE" / "calvin" / "params_ema")
    sd_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    # ── CLIP Text Encoder ─────────────────────────────────────────
    print("\n=== CLIP Text Encoder ===")
    from susie.model import load_text_encoder
    tokenize, _, text_encode = load_text_encoder(sd_path)
    tokens = tokenize(["open the drawer"])
    _ = text_encode(tokens); jax.block_until_ready(_)
    clip_times = benchmark_jax(lambda: text_encode(tokens), n_reps=n_reps)
    results["clip_text_encoder"] = {
        "params": macs["clip_text_params"],
        "macs": macs["clip_text"],
        "latency": latency_stats(clip_times),
        "vram_peak_mb": round(gpu_vram_mb(), 1),
        "schedule": "1x per regen",
    }
    print(f"  Latency: {results['clip_text_encoder']['latency']['mean_ms']:.2f} ms")

    # ── VAE Encoder ───────────────────────────────────────────────
    print("\n=== VAE Encoder ===")
    from susie.model import load_vae
    vae_encode, vae_decode = load_vae(sd_path)
    rng = jax.random.PRNGKey(0)
    dummy_img = jnp.ones((1, 256, 256, 3))
    _ = vae_encode(rng, dummy_img, scale=False); jax.block_until_ready(_)
    vae_enc_times = benchmark_jax(lambda: vae_encode(rng, dummy_img, scale=False), n_reps=n_reps)
    results["vae_encoder"] = {
        "params": macs["vae_enc_params"],
        "macs": macs["vae_enc"],
        "latency": latency_stats(vae_enc_times),
        "vram_peak_mb": round(gpu_vram_mb(), 1),
        "schedule": "1x per regen",
    }
    print(f"  Latency: {results['vae_encoder']['latency']['mean_ms']:.2f} ms")

    # ── SuSIE UNet (single forward) ──────────────────────────────
    print("\n=== SuSIE UNet (single fwd) ===")
    from susie.model import _load_orbax_model_and_config
    model_def, params, _ = _load_orbax_model_and_config(ckpt, None)
    unet_params = count_jax_params(params)

    dummy_lat = jnp.ones((1, 32, 32, 8))
    dummy_t = jnp.array([500])
    dummy_enc = jnp.ones((1, 77, 768))

    @jax.jit
    def unet_fwd(lat, t, enc, p):
        return model_def.apply({"params": p}, lat, t, enc)

    _ = unet_fwd(dummy_lat, dummy_t, dummy_enc, params); jax.block_until_ready(_)
    unet_times = benchmark_jax(lambda: unet_fwd(dummy_lat, dummy_t, dummy_enc, params), n_reps=n_reps)
    results["susie_unet_single_fwd"] = {
        "params": unet_params,
        "macs": macs["unet_single"],
        "latency": latency_stats(unet_times),
        "vram_peak_mb": round(gpu_vram_mb(), 1),
        "note": "single fwd; generation = 50 steps × 3 CFG = 150 calls",
    }
    print(f"  Params: {unet_params:,}")
    print(f"  Latency: {results['susie_unet_single_fwd']['latency']['mean_ms']:.2f} ms")

    # ── VAE Decoder ───────────────────────────────────────────────
    print("\n=== VAE Decoder ===")
    dummy_lat4 = jnp.ones((1, 32, 32, 4))
    _ = vae_decode(dummy_lat4); jax.block_until_ready(_)
    vae_dec_times = benchmark_jax(lambda: vae_decode(dummy_lat4), n_reps=n_reps)
    results["vae_decoder"] = {
        "params": macs["vae_dec_params"],
        "macs": macs["vae_dec"],
        "latency": latency_stats(vae_dec_times),
        "vram_peak_mb": round(gpu_vram_mb(), 1),
        "schedule": "1x per regen",
    }
    print(f"  Latency: {results['vae_decoder']['latency']['mean_ms']:.2f} ms")

    # ── Full SuSIE generation (batch=1) ──────────────────────────
    print("\n=== Full SuSIE Generation (batch=1, 50 DDIM steps) ===")
    from susie.model import create_sample_fn
    sample_fn_b1 = create_sample_fn(
        ckpt, "kvablack/dlimp-diffusion/9n9ped8m",
        num_timesteps=50, prompt_w=7.5, context_w=1.5, eta=0.0,
        pretrained_path=sd_path, num_samples=1,
    )
    dummy_obs = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    prompt = "open the drawer"
    print("  Warming up (JIT)...")
    _ = sample_fn_b1(dummy_obs, prompt)
    _ = sample_fn_b1(dummy_obs, prompt)

    gen_b1_times = benchmark_jax(lambda: sample_fn_b1(dummy_obs, prompt), warmup=2, n_reps=min(n_reps, 20))
    gen_b1_total_macs = (
        macs["clip_text"] +
        macs["vae_enc"] +
        150 * macs["unet_single"] +  # 50 steps × 3 CFG
        macs["vae_dec"]
    )
    results["susie_full_generation_b1"] = {
        "total_macs": gen_b1_total_macs,
        "latency": latency_stats(gen_b1_times),
        "vram_peak_mb": round(gpu_vram_mb(), 1),
        "detail": "CLIP_text + VAE_enc + 150×UNet + VAE_dec",
    }
    print(f"  Total MACs: {fmt(gen_b1_total_macs)}")
    print(f"  Latency: {results['susie_full_generation_b1']['latency']['mean_ms']:.1f} ms")

    # ── Full GHIL-Glue generation (batch=4) ──────────────────────
    print("\n=== Full GHIL-Glue Generation (batch=4, 50 DDIM steps) ===")
    sample_fn_b4 = create_sample_fn(
        ckpt, "kvablack/dlimp-diffusion/9n9ped8m",
        num_timesteps=50, prompt_w=7.5, context_w=1.5, eta=0.0,
        pretrained_path=sd_path, num_samples=4,
    )
    print("  Warming up (JIT)...")
    _ = sample_fn_b4(dummy_obs, prompt)
    _ = sample_fn_b4(dummy_obs, prompt)

    gen_b4_times = benchmark_jax(lambda: sample_fn_b4(dummy_obs, prompt), warmup=2, n_reps=min(n_reps, 15))
    gen_b4_total_macs = (
        4 * macs["clip_text"] +
        4 * macs["vae_enc"] +
        4 * 150 * macs["unet_single"] +  # 50 steps × 3 CFG × 4 samples
        4 * macs["vae_dec"]
    )
    results["ghilglue_full_generation_b4"] = {
        "total_macs": gen_b4_total_macs,
        "latency": latency_stats(gen_b4_times),
        "vram_peak_mb": round(gpu_vram_mb(), 1),
        "detail": "4×(CLIP_text + VAE_enc + 150×UNet + VAE_dec)",
    }
    print(f"  Total MACs: {fmt(gen_b4_total_macs)}")
    print(f"  Latency: {results['ghilglue_full_generation_b4']['latency']['mean_ms']:.1f} ms")

    return results


def profile_gc_policy(n_reps, macs, config_str="calvin_gcdiffusion_noactnorm-sagemaker-auggoaldiff"):
    """Profile the GC diffusion policy (ResNet-34 + DDPM 20 steps)."""
    import jax
    import jax.numpy as jnp
    sys.path.insert(0, str(GHIL_ROOT / "calvin_models"))
    from calvin_agent.evaluation.gcbc_train_config import get_config
    from jaxrl_m.vision import encoders
    from jaxrl_m.agents import agents
    import orbax.checkpoint

    agent_config, _, _, _ = get_config(config_str)
    ckpt = str(CKPT_DIR / "susie_low_level" / "calvin" / "gcdiffusion" /
               "auggoaldiff" / "seed_0" / "20240227_194024" / "checkpoint_150000")

    print("\n=== GC Diffusion Policy (ResNet-34-bridge + DDPM 20 steps) ===")

    example_batch = {
        "observations": {"image": np.zeros((1, 1, 200, 200, 3), dtype=np.uint8)},
        "actions": np.zeros((1, 4, 7), dtype=np.float32),
        "goals": {"image": np.zeros((1, 200, 200, 3), dtype=np.uint8),
                  "language": np.zeros((1, 512), dtype=np.float32)},
    }
    encoder_def = encoders[agent_config.encoder](**agent_config.encoder_kwargs)
    rng = jax.random.PRNGKey(42)
    agent = agents[agent_config.agent].create(
        rng=rng, observations=example_batch["observations"],
        goals=example_batch["goals"], actions=example_batch["actions"],
        encoder_def=encoder_def, **agent_config.agent_kwargs,
    )
    agent = orbax.checkpoint.PyTreeCheckpointer().restore(ckpt, item=agent)

    gc_params = count_jax_params(agent.state.params)
    print(f"  Params: {gc_params:,}")

    dummy_obs = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    dummy_goal = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    print("  Warming up...")
    _ = agent.sample_actions(
        {"image": dummy_obs[np.newaxis, ...]}, {"image": dummy_goal},
        seed=jax.random.PRNGKey(0), temperature=0.0)

    gc_times = benchmark_jax(
        lambda: agent.sample_actions(
            {"image": dummy_obs[np.newaxis, ...]}, {"image": dummy_goal},
            seed=jax.random.PRNGKey(0), temperature=0.0),
        n_reps=n_reps)

    # MACs: 20 DDPM steps, each calls full model (encoder + MLP)
    encoder_macs = macs["resnet34_6ch"]
    mlp_macs = macs["mlp_score_single"]
    total_macs = 20 * (encoder_macs + mlp_macs)

    result = {
        "params": gc_params,
        "macs_total_20_steps": total_macs,
        "macs_encoder_single": encoder_macs,
        "macs_mlp_single": mlp_macs,
        "latency": latency_stats(gc_times),
        "vram_peak_mb": round(gpu_vram_mb(), 1),
        "schedule": "1x per step (20 DDPM denoising internally)",
        "encoder": "ResNet-34-bridge (6ch early_goal_concat, 200×200)",
        "diffusion_steps": 20,
        "action_horizon": 4,
        "action_dim": 7,
    }
    print(f"  Total MACs (20 steps): {fmt(total_macs)}")
    print(f"  Latency: {result['latency']['mean_ms']:.2f} ms")
    return result


def profile_progress_vf(n_reps, macs):
    """Profile the high-level progress VF for GHIL-Glue filtering."""
    import jax
    import jax.numpy as jnp
    sys.path.insert(0, str(GHIL_ROOT / "calvin_models"))
    from calvin_agent.evaluation.gcbc_train_config import get_config
    from jaxrl_m.vision import encoders
    from jaxrl_m.agents import agents
    from jaxrl_m.data.text_processing import text_processors
    import orbax.checkpoint

    vf_config_str = "calvinlcbc_lcgcprogressvf_noactnorm-auggoaldiff"
    vf_config, _, _, _ = get_config(vf_config_str)
    ckpt = str(CKPT_DIR / "susie_low_level" / "calvinlcbc" / "lcgcprogressvf" /
               "auggoaldiff" / "seed_0" / "20240510_005751" / "checkpoint_100000")

    print("\n=== Progress VF (2× ResNet-34-FiLM + MLP) ===")

    example_batch = {
        "observations": {"image": np.zeros((1, 200, 200, 3), dtype=np.uint8)},
        "actions": np.zeros((1, 4, 7), dtype=np.float32),
        "goals": {"image": np.zeros((1, 200, 200, 3), dtype=np.uint8),
                  "language": np.zeros((1, 512), dtype=np.float32)},
    }
    encoder_def = encoders[vf_config.encoder](**vf_config.encoder_kwargs)
    rng = jax.random.PRNGKey(42)
    vf_agent = agents[vf_config.agent].create(
        rng=rng, observations=example_batch["observations"],
        goals=example_batch["goals"], actions=example_batch["actions"],
        encoder_def=encoder_def, **vf_config.agent_kwargs,
    )
    vf_agent = orbax.checkpoint.PyTreeCheckpointer().restore(ckpt, item=vf_agent)
    vf_params = count_jax_params(vf_agent.state.params)
    print(f"  Params: {vf_params:,}")

    # VF filtering: evaluate 4 candidate goals (batch=4)
    dummy_obs4 = np.random.randint(0, 255, (4, 200, 200, 3), dtype=np.uint8)
    dummy_goal4 = np.random.randint(0, 255, (4, 200, 200, 3), dtype=np.uint8)
    dummy_lang4 = np.zeros((4, 512), dtype=np.float32)

    print("  Warming up...")
    _ = vf_agent.value_function(
        {"image": dummy_obs4}, {"image": dummy_goal4, "language": dummy_lang4})

    vf_times = benchmark_jax(
        lambda: vf_agent.value_function(
            {"image": dummy_obs4}, {"image": dummy_goal4, "language": dummy_lang4}),
        n_reps=n_reps)

    # MACs: 2 ResNet-34 encoders (obs + goal) × 4 batch + MLP head
    vf_macs_per_sample = 2 * macs["resnet34_3ch"]  # obs encoder + goal encoder
    vf_total_macs = 4 * vf_macs_per_sample  # batch of 4

    result = {
        "params": vf_params,
        "macs_batch4": vf_total_macs,
        "macs_per_sample": vf_macs_per_sample,
        "latency": latency_stats(vf_times),
        "vram_peak_mb": round(gpu_vram_mb(), 1),
        "schedule": "1x per regen (batch=4)",
        "encoder": "2× ResNet-34-bridge-FiLM + MLP value head",
    }
    print(f"  Total MACs (batch=4): {fmt(vf_total_macs)}")
    print(f"  Latency: {result['latency']['mean_ms']:.2f} ms")
    return result


def profile_muse_text(n_reps):
    """Profile the MUSE text encoder (TF Hub, CPU)."""
    print("\n=== MUSE Text Encoder (TF Hub, CPU) ===")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU for TF
    import tensorflow as tf
    import tensorflow_hub as hub
    # Restore CUDA visibility after
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("_ORIG_CUDA_VISIBLE_DEVICES", "0")

    # Actually, for the MUSE encoder we need to be careful not to break JAX.
    # Let's load it the same way the eval code does.
    sys.path.insert(0, str(GHIL_ROOT / "calvin_models"))
    from jaxrl_m.data.text_processing import text_processors

    tp = text_processors["muse_embedding"]()
    dummy_text = "open the drawer"

    print("  Warming up...")
    _ = tp.encode(dummy_text)
    _ = tp.encode(dummy_text)

    times = benchmark_tf_cpu(lambda: tp.encode(dummy_text), n_reps=n_reps)

    result = {
        "params": "N/A (TF Hub USE-Multilingual-v3)",
        "macs": "N/A",
        "latency": latency_stats(times),
        "vram_peak_mb": 0,
        "schedule": "1x per regen (CPU)",
        "note": "TF Hub Universal Sentence Encoder, runs on CPU",
    }
    print(f"  Latency: {result['latency']['mean_ms']:.2f} ms")
    return result


# ── Phase 3: Assemble final JSONs ────────────────────────────────────────

def build_susie_json(components, gc, gen_macs):
    """Build the SuSIE profile JSON."""
    regen_interval = 20
    gen_latency = components["susie_full_generation_b1"]["latency"]["mean_ms"]
    policy_latency = gc["latency"]["mean_ms"]
    gen_total_macs = components["susie_full_generation_b1"]["total_macs"]
    policy_macs = gc["macs_total_20_steps"]

    amortized_latency = policy_latency + gen_latency / regen_interval
    amortized_macs = policy_macs + gen_total_macs / regen_interval

    # Total params: unique models loaded
    unet_params = components["susie_unet_single_fwd"]["params"]
    clip_params = components["clip_text_encoder"]["params"]
    vae_params = gen_macs["vae_total_params"]
    gc_params = gc["params"]
    total_params = unet_params + clip_params + vae_params + gc_params

    vram_peak = max(
        components["susie_full_generation_b1"]["vram_peak_mb"],
        gc["vram_peak_mb"],
    )

    return {
        "system": "susie_pure",
        "description": "Vanilla SuSIE: SD-based subgoal generation + GC diffusion policy. Regen every 20 steps.",
        "modules": {
            "clip_text_encoder": components["clip_text_encoder"],
            "vae_encoder": components["vae_encoder"],
            "susie_unet": components["susie_unet_single_fwd"],
            "vae_decoder": components["vae_decoder"],
            "gc_diffusion_policy": gc,
        },
        "full_generation": {
            "macs": gen_total_macs,
            "latency_ms": gen_latency,
            "regen_interval": regen_interval,
            "components": "CLIP_text + VAE_enc + 150×UNet(50 steps × 3 CFG) + VAE_dec",
        },
        "amortized_per_step": {
            "macs": int(amortized_macs),
            "macs_formatted": fmt(int(amortized_macs)),
            "latency_ms": round(amortized_latency, 2),
            "vram_peak_mb": vram_peak,
            "total_params": total_params,
            "breakdown": f"policy({fmt(policy_macs)}) + gen({fmt(gen_total_macs)})/{regen_interval}",
        },
    }


def build_ghilglue_json(components, gc, vf, muse, gen_macs):
    """Build the GHIL-Glue w/ filtering profile JSON."""
    regen_interval = 20
    gen_latency = components["ghilglue_full_generation_b4"]["latency"]["mean_ms"]
    vf_latency = vf["latency"]["mean_ms"]
    muse_latency = muse["latency"]["mean_ms"]
    policy_latency = gc["latency"]["mean_ms"]

    gen_total_macs = components["ghilglue_full_generation_b4"]["total_macs"]
    vf_macs = vf["macs_batch4"]
    policy_macs = gc["macs_total_20_steps"]

    regen_total_macs = gen_total_macs + vf_macs  # MUSE MACs negligible
    regen_total_latency = gen_latency + vf_latency + muse_latency

    amortized_latency = policy_latency + regen_total_latency / regen_interval
    amortized_macs = policy_macs + regen_total_macs / regen_interval

    unet_params = components["susie_unet_single_fwd"]["params"]
    clip_params = components["clip_text_encoder"]["params"]
    vae_params = gen_macs["vae_total_params"]
    gc_params = gc["params"]
    vf_params = vf["params"]
    total_params = unet_params + clip_params + vae_params + gc_params + vf_params

    vram_peak = max(
        components["ghilglue_full_generation_b4"]["vram_peak_mb"],
        gc["vram_peak_mb"],
        vf["vram_peak_mb"],
    )

    return {
        "system": "ghilglue_full",
        "description": "GHIL-Glue (SuSIE): 4-sample subgoal gen + VF filtering + GC policy. Regen every 20 steps.",
        "modules": {
            "clip_text_encoder": components["clip_text_encoder"],
            "vae_encoder": components["vae_encoder"],
            "susie_unet": components["susie_unet_single_fwd"],
            "vae_decoder": components["vae_decoder"],
            "progress_vf": vf,
            "muse_text_encoder": muse,
            "gc_diffusion_policy": gc,
        },
        "full_generation": {
            "macs": regen_total_macs,
            "latency_ms": round(regen_total_latency, 2),
            "regen_interval": regen_interval,
            "components": "4×(CLIP_text + VAE_enc + 150×UNet + VAE_dec) + MUSE + VF(batch=4)",
        },
        "amortized_per_step": {
            "macs": int(amortized_macs),
            "macs_formatted": fmt(int(amortized_macs)),
            "latency_ms": round(amortized_latency, 2),
            "vram_peak_mb": vram_peak,
            "total_params": total_params,
            "breakdown": f"policy({fmt(policy_macs)}) + regen({fmt(regen_total_macs)})/{regen_interval}",
        },
    }


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reps", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=str(MAL_ROOT / "results" / "profiling"))
    args = parser.parse_args()

    # Save original CUDA env
    os.environ["_ORIG_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    print("=" * 70)
    print("Phase 1: Computing MACs via fvcore (CPU)")
    print("=" * 70)
    mac_data = compute_macs()

    print("\n" + "=" * 70)
    print("Phase 2: Profiling SuSIE/GHIL-Glue components (GPU)")
    print("=" * 70)

    # Profile MUSE first (CPU, before JAX takes over GPU)
    muse_result = profile_muse_text(args.n_reps)

    # Profile SuSIE generation components + GHIL-Glue batch=4
    comp_results = profile_susie_components(args.n_reps, mac_data)

    # Profile GC policy
    gc_result = profile_gc_policy(args.n_reps, mac_data)

    # Profile Progress VF
    vf_result = profile_progress_vf(args.n_reps, mac_data)

    print("\n" + "=" * 70)
    print("Phase 3: Building output JSONs")
    print("=" * 70)

    susie_json = build_susie_json(comp_results, gc_result, mac_data)
    ghilglue_json = build_ghilglue_json(comp_results, gc_result, vf_result, muse_result, mac_data)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "profile_susie.json", "w") as f:
        json.dump(susie_json, f, indent=2, default=str)
    print(f"\nSaved: {out_dir / 'profile_susie.json'}")

    with open(out_dir / "profile_ghilglue.json", "w") as f:
        json.dump(ghilglue_json, f, indent=2, default=str)
    print(f"Saved: {out_dir / 'profile_ghilglue.json'}")

    # Print summary tables
    print("\n" + "=" * 70)
    print("SuSIE Amortized Per-Step Summary")
    print("=" * 70)
    s = susie_json["amortized_per_step"]
    print(f"  MACs:    {s['macs_formatted']}")
    print(f"  Latency: {s['latency_ms']} ms")
    print(f"  VRAM:    {s['vram_peak_mb']} MB")
    print(f"  Params:  {s['total_params']:,}")
    print(f"  {s['breakdown']}")

    print("\n" + "=" * 70)
    print("GHIL-Glue Amortized Per-Step Summary")
    print("=" * 70)
    g = ghilglue_json["amortized_per_step"]
    print(f"  MACs:    {g['macs_formatted']}")
    print(f"  Latency: {g['latency_ms']} ms")
    print(f"  VRAM:    {g['vram_peak_mb']} MB")
    print(f"  Params:  {g['total_params']:,}")
    print(f"  {g['breakdown']}")


if __name__ == "__main__":
    main()
