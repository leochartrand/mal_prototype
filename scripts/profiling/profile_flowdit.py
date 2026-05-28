"""
Profile DiT end-to-end on CALVIN — adaptive subgoal variant.

Three execution scopes
    Initial (1x per sub-task):  MiniLM text encoder (cached across regens; auto-
                                invalidates when the instruction changes).
    Regen   (every K steps):    DiT-Air full generation (EULER_STEPS × 1 fwd, distilled).
    Every step:                 Theia encoder + Theia cosine progress evaluator +
                                GCBC policy (4 Euler ODE steps).

Mean regen interval K is measured from a real 1000-sequence eval of the
distilled student (see ``results/baselines/distill_v3_e10_min5_1000_regen_stats.json``):
K ≈ 8.3 with adaptive regen (δ_high=0.90, min_per_frame=5, max_per_frame=20).
74% of regens fire at K=5 (min-clamp), 17% at K=20 (max-clamp).

Note: Theia z_obs is shared — computed once per step and reused as z_init for
DiT at regen time. No double-counting of Theia at regen.

Usage:
    cd /home/chal2525/mal_prototype
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python scripts/profiling/profile_flowdit.py \\
        --n_reps 100 --device cuda:0 --output_dir results/profiling
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

THEIA_PATH    = ROOT / "models" / "theia_small_cdiv"
MINILM_PATH   = ROOT / "models" / "all-MiniLM-L6-v2"
FLOWDIT_CKPT  = ROOT / "models" / "calvin_ft_subgoal_rae_distill" / "model.pt"
GCBC_CKPT     = ROOT / "models" / "gcbc_abcd"             / "gcbc_step400000.pt"
REGEN_STATS   = ROOT / "results" / "baselines" / "distill_v3_e10_min5_1000_regen_stats.json"

FLOWDIT_CFG = dict(
    latent_dim=384, num_patches=196, hidden_dim=896, depth=18, num_heads=14,
    text_dim=384, pooled_text_dim=384, max_text_len=25, mlp_ratio=4.0,
    dropout=0.1, use_pooled_text=False,
    cfg_drop_prompt=0.05, cfg_drop_context=0.05, cfg_drop_both=0.05,
)
GCBC_CFG = dict(
    action_dim=7, act_pred_horizon=5,
    theia_dim=384, num_sampling_steps=4,
    time_dim=32, hidden_dim=256, num_blocks=3, dropout=0.0,
)

EULER_STEPS    = 4
CONTEXT_CFG    = 2.5
PROMPT_CFG     = 6.0
MAX_TEXT_LEN   = 25


# ── helpers ──────────────────────────────────────────────────────────────

def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def try_fvcore(model, inputs) -> int | None:
    try:
        from fvcore.nn import FlopCountAnalysis
        a = FlopCountAnalysis(model, inputs)
        a.unsupported_ops_warnings(False)
        a.uncalled_modules_warnings(False)
        return int(a.total())
    except Exception as e:
        print(f"  [fvcore] {e}")
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

def benchmark(fn, warmup=5, n_reps=100, dev_idx=0) -> list[float]:
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

def vram_mb(dev_idx=0) -> float:
    return torch.cuda.max_memory_allocated(dev_idx) / 1e6

def fmt(macs) -> str:
    if macs is None: return "N/A"
    if macs >= 1e12: return f"{macs/1e12:.2f}T"
    if macs >= 1e9:  return f"{macs/1e9:.2f}G"
    if macs >= 1e6:  return f"{macs/1e6:.2f}M"
    return str(macs)


# ── module profilers ──────────────────────────────────────────────────────

def profile_theia(device, dev_idx, n_reps):
    """Profile Theia encoder-only — pure DeiT-Small backbone, no translator/neck/wrappers.

    The full Theia model has translator heads (per target VFM), a neck, and HF
    preprocessing wrappers around the backbone. FlowDiT only consumes the backbone's
    patch features; the rest is dead weight at inference. This measures the floor:
    encoder forward on a pre-normalized (1,3,224,224) tensor.
    """
    from transformers import AutoModel
    print("\n=== Theia-Small backbone (encoder only) ===")
    torch.cuda.reset_peak_memory_stats(dev_idx)
    model = AutoModel.from_pretrained(str(THEIA_PATH), trust_remote_code=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Strip down to the DeiT ViT (drop translator/neck/loss heads, drop HF preprocessor)
    backbone = model.backbone.model if hasattr(model.backbone, "model") else model.backbone
    params = count_params(backbone)

    dummy_px = torch.randn(1, 3, 224, 224, device=device)
    macs = try_fvcore(backbone, (dummy_px,))
    if macs is None:
        macs = int(4.61e9)  # DeiT-Small/16 known value

    @torch.no_grad()
    def fn():
        backbone(dummy_px)

    torch.cuda.reset_peak_memory_stats(dev_idx)
    times = benchmark(fn, n_reps=n_reps, dev_idx=dev_idx)
    peak = vram_mb(dev_idx)

    print(f"  Params: {params:,}  MACs: {fmt(macs)}  Latency: {np.mean(times)*1000:.2f} ms  VRAM: {peak:.0f} MB")
    return model, params, macs, latency_stats(times), round(peak, 1)


def profile_minilm(device, dev_idx, n_reps):
    """Profile MiniLM encoder-only — skips the pooler that FlowDiT doesn't use."""
    from transformers import AutoTokenizer, AutoModel as HFModel
    print("\n=== MiniLM-L6-v2 (encoder only, no pooler) ===")
    tok = AutoTokenizer.from_pretrained(str(MINILM_PATH))
    # add_pooling_layer=False drops the [CLS]-pooler head — we never read pooler_output
    model = HFModel.from_pretrained(str(MINILM_PATH), add_pooling_layer=False).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    params = count_params(model)

    dummy_enc = tok(["open the drawer"], return_tensors="pt",
                    padding="max_length", max_length=MAX_TEXT_LEN, truncation=True)
    input_ids  = dummy_enc["input_ids"].to(device)
    attn_mask  = dummy_enc["attention_mask"].to(device)

    macs = try_fvcore(model, (input_ids, attn_mask))
    if macs is None:
        macs = int(2.69e8)  # known value from earlier measurement

    @torch.no_grad()
    def fn():
        model(input_ids=input_ids, attention_mask=attn_mask)

    torch.cuda.reset_peak_memory_stats(dev_idx)
    times = benchmark(fn, n_reps=n_reps, dev_idx=dev_idx)
    peak = vram_mb(dev_idx)

    print(f"  Params: {params:,}  MACs: {fmt(macs)}  Latency: {np.mean(times)*1000:.2f} ms  VRAM: {peak:.0f} MB")
    return model, tok, params, macs, latency_stats(times), round(peak, 1)


def profile_flowdit_modules(device, dev_idx, n_reps):
    from models.flowdit import DiTAir
    print("\n=== DiT-Air ===")
    model = DiTAir(**FLOWDIT_CFG).to(device).eval()
    if FLOWDIT_CKPT.exists():
        ckpt = torch.load(str(FLOWDIT_CKPT), map_location=device)
        sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(sd, strict=False)
        print(f"  Loaded: {FLOWDIT_CKPT.name}")
    else:
        print("  No checkpoint found — using random weights")
    for p in model.parameters():
        p.requires_grad_(False)
    params = count_params(model)

    # Dummy inputs for a single forward
    B = 1
    dummy_zt    = torch.randn(B, 196, 384, device=device)
    dummy_zinit = torch.randn(B, 196, 384, device=device)
    dummy_t     = torch.rand(B, device=device)
    dummy_txt   = torch.randn(B, MAX_TEXT_LEN, 384, device=device)
    dummy_mask  = torch.ones(B, MAX_TEXT_LEN, device=device)
    # Mask-weighted mean pool of hidden states — matches real inference (wrapper line 278)
    dummy_pool  = dummy_txt.mean(dim=1)  # [B, 384]

    # MACs for a single forward pass (no pooled text — use_pooled_text=False)
    macs_single = try_fvcore(model, (dummy_zt, dummy_t, dummy_zinit, dummy_txt, dummy_mask))
    if macs_single is None:
        macs_single = int(4.85e10)  # fallback

    # Single forward latency — wrapped in bf16 autocast to match eval-time inference
    torch.cuda.reset_peak_memory_stats(dev_idx)
    @torch.no_grad()
    def single_fn():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(dummy_zt, dummy_t, dummy_zinit, dummy_txt, dummy_mask)

    times_single = benchmark(single_fn, n_reps=n_reps, dev_idx=dev_idx)
    vram_single = vram_mb(dev_idx)
    lat_single = latency_stats(times_single)
    print(f"  Params: {params:,}  Single fwd MACs: {fmt(macs_single)}  Latency: {lat_single['mean_ms']:.2f} ms (bf16)")

    # Full generation: distilled student → 1 fwd per Euler step, no CFG passes
    print(f"\n=== DiT full generation ({EULER_STEPS} Euler steps, distilled student, bf16) ===")
    @torch.no_grad()
    def gen_fn():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.sample_euler(
                dummy_zinit, dummy_txt,
                text_mask=dummy_mask,
                pooled_text_emb=dummy_pool,
                num_steps=EULER_STEPS,
                # No CFG scales → 1 fwd/step (distilled student bakes CFG into a single output)
            )

    # Warmup
    gen_fn(); gen_fn()
    torch.cuda.reset_peak_memory_stats(dev_idx)
    times_gen = benchmark(gen_fn, warmup=2, n_reps=min(n_reps, 30), dev_idx=dev_idx)
    vram_gen = vram_mb(dev_idx)
    lat_gen = latency_stats(times_gen)

    # 1 fwd per step (distilled, no CFG), EULER_STEPS steps
    macs_gen = macs_single * EULER_STEPS
    print(f"  Total MACs: {fmt(macs_gen)} ({EULER_STEPS} steps × 1 fwd × {fmt(macs_single)}/fwd)")
    print(f"  Latency: {lat_gen['mean_ms']:.1f} ms  VRAM: {vram_gen:.0f} MB")

    return model, params, macs_single, lat_single, round(vram_single, 1), macs_gen, lat_gen, round(vram_gen, 1)


def profile_progress_eval(device, dev_idx, n_reps):
    from models.progress_evaluator import TheiaProgressEvaluator
    print("\n=== Theia Progress Evaluator (patch-mean cosine, every step) ===")
    evaluator = TheiaProgressEvaluator(delta_high=0.90, max_steps=20)

    dummy_obs     = torch.randn(1, 196, 384, device=device)
    dummy_subgoal = torch.randn(1, 196, 384, device=device)

    # MACs: per-patch cosine = (2D-1) flops/patch ≈ 2*384*196 multiplies + reductions.
    # F.cosine_similarity does 2 dot products (numerator + 2 norms) per patch.
    # Rough estimate: 3 × 196 × 384 = 226K MACs.
    macs = 3 * 196 * 384

    @torch.no_grad()
    def fn():
        evaluator.step_count = 0  # avoid max_steps trip in benchmark
        evaluator.should_advance(dummy_obs, dummy_subgoal)

    torch.cuda.reset_peak_memory_stats(dev_idx)
    times = benchmark(fn, n_reps=n_reps, dev_idx=dev_idx)
    peak = vram_mb(dev_idx)
    lat = latency_stats(times)

    print(f"  MACs: {fmt(macs)}  Latency: {lat['mean_ms']:.3f} ms  VRAM: {peak:.0f} MB")
    return macs, lat, round(peak, 1)


def profile_gcbc(device, dev_idx, n_reps):
    from models.diffusion_policy import FlowMatchingGCBCPolicy
    print("\n=== GCBC policy (flow matching, 4 Euler steps) ===")
    model = FlowMatchingGCBCPolicy(**GCBC_CFG).to(device).eval()
    if GCBC_CKPT.exists():
        ckpt = torch.load(str(GCBC_CKPT), map_location=device)
        sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(sd, strict=False)
        print(f"  Loaded: {GCBC_CKPT.name}")
    else:
        print("  No checkpoint found — using random weights")
    for p in model.parameters():
        p.requires_grad_(False)
    params = count_params(model)

    dummy_obs  = torch.randn(1, 196, 384, device=device)
    dummy_goal = torch.randn(1, 196, 384, device=device)

    # Projector MACs (obs) + projector MACs (goal)
    proj_macs = try_fvcore(model.projector, (dummy_obs,))
    if proj_macs is None: proj_macs = 0

    # Velocity net MACs (single step)
    with torch.no_grad():
        z_obs_pool  = model.projector(dummy_obs)
        z_goal_pool = model.projector(dummy_goal)
    flat_action = torch.randn(1, GCBC_CFG["act_pred_horizon"] * GCBC_CFG["action_dim"], device=device)
    dummy_t_v = torch.rand(1, device=device)
    vnet_macs = try_fvcore(model.velocity_net, (flat_action, z_obs_pool, z_goal_pool, dummy_t_v))
    if vnet_macs is None: vnet_macs = 0

    # Total: 2 projector calls (obs + goal) + 4 velocity net calls
    n_sampling = GCBC_CFG["num_sampling_steps"]
    total_macs = 2 * proj_macs + n_sampling * vnet_macs

    @torch.no_grad()
    def fn():
        model.sample_actions(dummy_obs, dummy_goal)

    torch.cuda.reset_peak_memory_stats(dev_idx)
    times = benchmark(fn, n_reps=n_reps, dev_idx=dev_idx)
    peak = vram_mb(dev_idx)
    lat = latency_stats(times)

    print(f"  Params: {params:,}  MACs: {fmt(total_macs)}  Latency: {lat['mean_ms']:.2f} ms  VRAM: {peak:.0f} MB")
    print(f"  Breakdown: 2×projector({fmt(proj_macs)}) + {n_sampling}×vnet({fmt(vnet_macs)})")
    return model, params, proj_macs, vnet_macs, total_macs, lat, round(peak, 1)


# ── assemble output JSONs ─────────────────────────────────────────────────

def build_json(
    system_name, description, regen_interval, avg_steps_per_task,
    theia_params, theia_macs, theia_lat, theia_vram,
    minilm_params, minilm_macs, minilm_lat, minilm_vram,
    flowdit_params, flowdit_macs_single, flowdit_lat_single, flowdit_vram_single,
    flowdit_macs_gen, flowdit_lat_gen, flowdit_vram_gen,
    gcbc_params, gcbc_proj_macs, gcbc_vnet_macs, gcbc_total_macs, gcbc_lat, gcbc_vram,
    progress_macs, progress_lat, progress_vram,
    regen_stats=None,
    include_progress_eval=True,
):
    # 3 execution scopes:
    #   initial  (1x per sub-task):       MiniLM text encoder (cached after that)
    #   regen    (every regen_interval):  DiT generation  (Theia z_obs shared)
    #   per_step (every step):            Theia + GCBC [+ Progress Evaluator]
    pe_macs_used = progress_macs if include_progress_eval else 0
    pe_lat_used  = progress_lat["mean_ms"] if include_progress_eval else 0.0
    per_step_macs   = theia_macs + gcbc_total_macs + pe_macs_used
    per_step_lat_ms = theia_lat["mean_ms"] + gcbc_lat["mean_ms"] + pe_lat_used

    regen_macs   = flowdit_macs_gen
    regen_lat_ms = flowdit_lat_gen["mean_ms"]

    initial_macs   = minilm_macs
    initial_lat_ms = minilm_lat["mean_ms"]

    amortized_macs = (
        per_step_macs
        + regen_macs / regen_interval
        + initial_macs / avg_steps_per_task
    )
    amortized_lat_ms = (
        per_step_lat_ms
        + regen_lat_ms / regen_interval
        + initial_lat_ms / avg_steps_per_task
    )

    total_params = theia_params + minilm_params + flowdit_params + gcbc_params
    peak_vram    = max(theia_vram, minilm_vram, flowdit_vram_gen, gcbc_vram, progress_vram)

    return {
        "system": system_name,
        "description": description,
        "regen_interval": regen_interval,
        "avg_steps_per_task": avg_steps_per_task,
        "regen_stats": regen_stats,
        "modules": {
            "theia_encoder": {
                "params": theia_params,
                "macs": theia_macs,
                "latency": theia_lat,
                "vram_peak_mb": theia_vram,
                "schedule": "every step",
                "note": "z_obs reused as z_init for DiT at regen time (no double-count)",
            },
            "minilm_text_encoder": {
                "params": minilm_params,
                "macs": minilm_macs,
                "latency": minilm_lat,
                "vram_peak_mb": minilm_vram,
                "schedule": "1x per sub-task",
                "note": f"max_length={MAX_TEXT_LEN}; wrapper caches hidden/mask/pooled until instruction changes",
            },
            "flowdit_air_single_fwd": {
                "params": flowdit_params,
                "macs": flowdit_macs_single,
                "latency": flowdit_lat_single,
                "vram_peak_mb": flowdit_vram_single,
                "note": f"reference; generation = {EULER_STEPS} steps × 1 fwd = {EULER_STEPS} fwds (distilled)",
            },
            "flowdit_full_generation": {
                "total_macs": flowdit_macs_gen,
                "latency": flowdit_lat_gen,
                "vram_peak_mb": flowdit_vram_gen,
                "schedule": "1x per regen",
                "detail": (
                    f"{EULER_STEPS} Euler steps × 1 fwd (distilled, no CFG: "
                    f"context_w={CONTEXT_CFG}, prompt_w={PROMPT_CFG})"
                ),
            },
            "progress_evaluator": {
                "macs": progress_macs,
                "latency": progress_lat,
                "vram_peak_mb": progress_vram,
                "schedule": "every step" if include_progress_eval else "disabled",
                "included_in_per_step": include_progress_eval,
                "detail": "Theia patch-mean cosine sim; triggers regen when sim >= delta_high (0.90) or steps >= max_steps (20)",
            },
            "gcbc_policy": {
                "params": gcbc_params,
                "macs_total": gcbc_total_macs,
                "macs_projector_single": gcbc_proj_macs,
                "macs_vnet_single": gcbc_vnet_macs,
                "latency": gcbc_lat,
                "vram_peak_mb": gcbc_vram,
                "schedule": f"every step ({GCBC_CFG['num_sampling_steps']} Euler ODE steps internally)",
                "action_horizon": GCBC_CFG["act_pred_horizon"],
                "action_dim": GCBC_CFG["action_dim"],
            },
        },
        "initial_per_task": {
            "macs": int(initial_macs),
            "macs_formatted": fmt(int(initial_macs)),
            "latency_ms": round(initial_lat_ms, 2),
            "components": "MiniLM",
            "schedule": f"1x per sub-task (amortized over {avg_steps_per_task:.1f} steps)",
        },
        "per_regen": {
            "macs": int(regen_macs),
            "macs_formatted": fmt(int(regen_macs)),
            "latency_ms": round(regen_lat_ms, 2),
            "regen_interval": regen_interval,
            "components": f"DiT ({EULER_STEPS} steps × 1 fwd, distilled); Theia z_obs shared with per-step",
        },
        "per_step": {
            "macs": int(per_step_macs),
            "macs_formatted": fmt(int(per_step_macs)),
            "latency_ms": round(per_step_lat_ms, 2),
            "components": (
                "Theia + Progress Evaluator + GCBC"
                if include_progress_eval else "Theia + GCBC (no progress eval)"
            ),
        },
        "amortized_per_step": {
            "macs": int(amortized_macs),
            "macs_formatted": fmt(int(amortized_macs)),
            "latency_ms": round(amortized_lat_ms, 2),
            "vram_peak_mb": round(peak_vram, 1),
            "total_params": total_params,
            "breakdown": (
                f"per_step({fmt(per_step_macs)}) "
                f"+ regen({fmt(regen_macs)})/{regen_interval:.2f} "
                f"+ initial({fmt(initial_macs)})/{avg_steps_per_task:.1f}"
            ),
        },
    }


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reps",      type=int, default=100)
    parser.add_argument("--device",      type=str, default="cuda:0")
    parser.add_argument("--output_dir",  type=str, default=str(ROOT / "results" / "profiling"))
    args = parser.parse_args()

    device  = torch.device(args.device)
    dev_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0

    torch.cuda.init()
    print(f"Device: {args.device}  |  n_reps: {args.n_reps}")
    print(f"DiT generation: {EULER_STEPS} Euler steps, distilled student (1 fwd/step)")

    # ── Load measured regen stats ───────────────────────────────────
    with open(REGEN_STATS) as f:
        regen_stats = json.load(f)
    regen_interval = float(regen_stats["mean_regen_interval"])
    # avg_steps_per_task = total_steps / (num_sequences × 5 sub-tasks)
    # 1000-seq eval: 443915 / 5000 = 88.78
    avg_steps_per_task = regen_stats["total_steps"] / (1000 * 5)
    print(f"\nMeasured regen interval: {regen_interval:.2f} steps "
          f"(median={regen_stats['median_regen_interval']}, "
          f"hist[1]={regen_stats['regen_interval_histogram'].get('1', 0)})")
    print(f"Avg steps per sub-task: {avg_steps_per_task:.2f}")

    # ── Profile modules ──────────────────────────────────────────────
    _, th_params, th_macs, th_lat, th_vram = profile_theia(device, dev_idx, args.n_reps)
    _, _, ml_params, ml_macs, ml_lat, ml_vram = profile_minilm(device, dev_idx, args.n_reps)
    (_, fd_params, fd_macs_single, fd_lat_single, fd_vram_single,
     fd_macs_gen, fd_lat_gen, fd_vram_gen) = profile_flowdit_modules(device, dev_idx, args.n_reps)
    pe_macs, pe_lat, pe_vram = profile_progress_eval(device, dev_idx, args.n_reps)
    (_, gc_params, gc_proj_macs, gc_vnet_macs, gc_total_macs,
     gc_lat, gc_vram) = profile_gcbc(device, dev_idx, args.n_reps)

    # ── Print summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Per-Module Summary")
    print("=" * 70)
    rows = [
        ("Module",                "Schedule",       "Params",                      "MACs",                  "Latency (ms)",              "VRAM (MB)"),
        ("minilm_text_encoder",   "1x per task",    f"{ml_params/1e6:.1f}M",       fmt(ml_macs),            f"{ml_lat['mean_ms']:.2f}",   f"{ml_vram:.0f}"),
        ("flowdit_air (×1 fwd)",  "ref",            f"{fd_params/1e6:.1f}M",       fmt(fd_macs_single),     f"{fd_lat_single['mean_ms']:.2f}", f"{fd_vram_single:.0f}"),
        (f"flowdit_full_gen (×{EULER_STEPS*3})","1x per regen",   "—",                           fmt(fd_macs_gen),        f"{fd_lat_gen['mean_ms']:.1f}",f"{fd_vram_gen:.0f}"),
        ("theia_encoder",         "every step",     f"{th_params/1e6:.1f}M",       fmt(th_macs),            f"{th_lat['mean_ms']:.2f}",   f"{th_vram:.0f}"),
        ("progress_evaluator",    "every step",     "—",                           fmt(pe_macs),            f"{pe_lat['mean_ms']:.3f}",   f"{pe_vram:.0f}"),
        ("gcbc_policy",           "every step",     f"{gc_params/1e6:.1f}M",       fmt(gc_total_macs),      f"{gc_lat['mean_ms']:.2f}",   f"{gc_vram:.0f}"),
    ]
    col_w = [max(len(r[i]) for r in rows) + 2 for i in range(6)]
    for i, row in enumerate(rows):
        line = "| " + " | ".join(c.ljust(w) for c, w in zip(row, col_w)) + " |"
        print(line)
        if i == 0:
            print("|-" + "-|-".join("-" * w for w in col_w) + "-|")

    # ── Build and save JSONs (3 variants) ────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    regen_stats_summary = {
        "mean_regen_interval": regen_stats["mean_regen_interval"],
        "median_regen_interval": regen_stats["median_regen_interval"],
        "std_regen_interval": regen_stats["std_regen_interval"],
        "total_steps": regen_stats["total_steps"],
        "subgoal_gen_count": regen_stats["subgoal_gen_count"],
    }

    # (name, description, K, include_progress_eval, regen_stats_to_attach)
    VARIANTS = [
        (
            "flowdit_subgoal_adaptive",
            "DiT subgoal-RAE with Theia cosine progress eval. Adaptive regen "
            "(delta_high=0.90, max_steps=20). K = measured mean from 1000-seq CALVIN ABC→D eval.",
            regen_interval,
            True,
            regen_stats_summary,
        ),
        (
            "flowdit_fixed20",
            "DiT subgoal-RAE with SuSIE-style fixed regen every 20 steps. "
            "No progress evaluator overhead.",
            20.0,
            False,
            None,
        ),
        (
            "flowdit_endgoal",
            "DiT subgoal-RAE in endgoal mode: regen once per ~30-step task. "
            "No progress evaluator overhead.",
            30.0,
            False,
            None,
        ),
    ]

    for name, desc, K, with_pe, stats_attach in VARIANTS:
        data = build_json(
            name, desc, K, avg_steps_per_task,
            th_params, th_macs, th_lat, th_vram,
            ml_params, ml_macs, ml_lat, ml_vram,
            fd_params, fd_macs_single, fd_lat_single, fd_vram_single,
            fd_macs_gen, fd_lat_gen, fd_vram_gen,
            gc_params, gc_proj_macs, gc_vnet_macs, gc_total_macs, gc_lat, gc_vram,
            pe_macs, pe_lat, pe_vram,
            regen_stats=stats_attach,
            include_progress_eval=with_pe,
        )
        out_path = out_dir / f"profile_{name}.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\n{'='*70}")
        print(f"Variant: {name}  (K={K:.2f}, progress_eval={with_pe})")
        print(f"{'='*70}")
        s = data["amortized_per_step"]
        print(f"  Amortized MACs/step : {s['macs_formatted']}")
        print(f"  Amortized latency   : {s['latency_ms']} ms")
        print(f"  Peak VRAM           : {s['vram_peak_mb']} MB")
        print(f"  Total params        : {s['total_params']:,}")
        print(f"  Breakdown           : {s['breakdown']}")
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
