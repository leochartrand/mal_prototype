"""
Calibrate progress evaluator delta_high threshold for DiT + GCBC.

Loads model and environment ONCE, then sweeps delta_high values.
Each config runs the same N sequences with the same random seeds.

Usage:
  python scripts/calibration/calibrate_delta_online.py \
      --flowdit_ckpt models/calvin/no_pe_5e5/model.pt \
      --gcbc_ckpt models/gcbc_abcd/gcbc_step400000.pt \
      --num_sequences 30 \
      --device 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import hydra
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
from calvin_env.envs.play_table_env import get_env

EP_LEN = 360


def count_success(results: list[int]) -> list[float]:
    count = Counter(results)
    return [sum(count[j] for j in range(i, 6)) / len(results) for i in range(1, 6)]


def rollout(env, model, task_oracle, subtask, val_annotations):
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    for step in range(EP_LEN):
        action = model.step(obs, lang_annotation, step=step)
        obs, _, _, current_info = env.step(action)
        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            return True
    return False


def evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    for subtask in eval_sequence:
        if rollout(env, model, task_oracle, subtask, val_annotations):
            success_counter += 1
        else:
            return success_counter
    return success_counter


def run_sweep(model, env, task_oracle, val_annotations, eval_sequences,
              configs: list[dict]) -> list[dict]:
    """Run eval for each config, reusing model and env."""
    all_results = []

    for cfg in configs:
        label = cfg["label"]
        delta_high = cfg["delta_high"]
        min_per_frame = cfg.get("min_per_frame", 1)
        max_per_frame = cfg.get("max_per_frame", 20)

        # Update progress evaluator in-place
        model.progress_eval.delta_high = delta_high
        model.progress_eval.delta_low = delta_high  # no hysteresis
        model.progress_eval.min_steps = min_per_frame
        model.progress_eval.max_steps = max_per_frame
        model.reset_regen_stats()

        results = []
        t0 = time.time()
        pbar = tqdm(eval_sequences, desc=label, position=0, leave=True)
        for initial_state, eval_sequence in pbar:
            result = evaluate_sequence(
                env, model, task_oracle, initial_state, eval_sequence, val_annotations)
            results.append(result)
            if len(results) >= 3:
                sr = count_success(results)
                pbar.set_description(
                    f"{label} | " +
                    " ".join([f"{i+1}/5:{v*100:.0f}%" for i, v in enumerate(sr)])
                )
        elapsed = time.time() - t0

        sr = count_success(results)
        regen = model.regen_stats
        avg_seq_len = float(np.mean(results))

        row = {
            "label": label,
            "delta_high": delta_high,
            "min_per_frame": min_per_frame,
            "max_per_frame": max_per_frame,
            "chain_sr": [round(s * 100, 1) for s in sr],
            "avg_seq_len": round(avg_seq_len, 3),
            "total_steps": regen["total_steps"],
            "subgoal_gens": regen["subgoal_gen_count"],
            "mean_regen_interval": round(regen["mean_regen_interval"], 1),
            "median_regen_interval": round(regen["median_regen_interval"], 1),
            "elapsed_min": round(elapsed / 60, 1),
        }
        all_results.append(row)

        # Print interim result
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  chain_sr: {row['chain_sr']}")
        print(f"  avg_seq_len: {row['avg_seq_len']}")
        print(f"  subgoal_gens: {row['subgoal_gens']} / {row['total_steps']} steps")
        print(f"  mean_regen_interval: {row['mean_regen_interval']}")
        print(f"  time: {row['elapsed_min']} min")
        print(f"{'='*60}\n")

    return all_results


def print_summary(all_results: list[dict]):
    """Print a comparison table."""
    print("\n" + "=" * 90)
    print(f"{'Config':<25} {'1/5':>6} {'2/5':>6} {'3/5':>6} {'4/5':>6} {'5/5':>6} "
          f"{'AvgLen':>7} {'Gens':>6} {'MeanInt':>8} {'Time':>6}")
    print("-" * 90)
    for r in all_results:
        sr = r["chain_sr"]
        print(f"{r['label']:<25} {sr[0]:>5.1f}% {sr[1]:>5.1f}% {sr[2]:>5.1f}% "
              f"{sr[3]:>5.1f}% {sr[4]:>5.1f}% {r['avg_seq_len']:>7.3f} "
              f"{r['subgoal_gens']:>6} {r['mean_regen_interval']:>7.1f}s "
              f"{r['elapsed_min']:>5.1f}m")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Calibrate delta_high threshold")
    parser.add_argument("--flowdit_ckpt", type=str, required=True)
    parser.add_argument("--gcbc_ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default=None,
                        help="DiT config YAML")
    parser.add_argument("--env_cfg", type=str, default="data/calvin/task_D_D/validation")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_sequences", type=int, default=30)
    parser.add_argument("--flowdit_num_steps", type=int, default=4)
    parser.add_argument("--prompt_cfg_scale", type=float, default=6.0)
    parser.add_argument("--context_cfg_scale", type=float, default=2.5)
    parser.add_argument("--output", type=str, default="results/calibration/delta_sweep.json")

    args = parser.parse_args()
    device = f"cuda:{args.device}"

    # ---- Load env once ----
    obs_space = {"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []}
    print("Loading CALVIN environment...")
    env = get_env(args.env_cfg, obs_space=obs_space, show_gui=False)

    # ---- Load model once ----
    from flowdit_calvin_wrapper import build_wrapper_from_config
    print("Loading DiT + GCBC wrapper...")
    model = build_wrapper_from_config(
        flowdit_ckpt=args.flowdit_ckpt,
        gcbc_ckpt=args.gcbc_ckpt,
        config_path=args.config,
        device=device,
        delta_high=0.90,  # placeholder, will override per config
        flowdit_num_steps=args.flowdit_num_steps,
        prompt_cfg_scale=args.prompt_cfg_scale,
        context_cfg_scale=args.context_cfg_scale,
    )
    print(f"Sampling: num_steps={args.flowdit_num_steps}, "
          f"prompt_cfg={args.prompt_cfg_scale}, context_cfg={args.context_cfg_scale}")
    print("Model ready.\n")

    # ---- Load task oracle + annotations ----
    import calvin_agent
    conf_dir = Path(calvin_agent.__file__).absolute().parents[1] / "conf"
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(
        conf_dir / "annotations/new_playtable_validation.yaml")

    # ---- Pre-fetch sequences (same seeds for all configs) ----
    eval_sequences = list(get_sequences(args.num_sequences))
    print(f"Loaded {len(eval_sequences)} evaluation sequences.\n")

    # ---- Define sweep configs ----
    configs = [
        # min=5 sweep across delta range
        {"label": "dh=0.80_min=5",  "delta_high": 0.80, "min_per_frame": 5},
        {"label": "dh=0.85_min=5",  "delta_high": 0.85, "min_per_frame": 5},
        {"label": "dh=0.90_min=5",  "delta_high": 0.90, "min_per_frame": 5},
        {"label": "dh=0.95_min=5",  "delta_high": 0.95, "min_per_frame": 5},
        {"label": "dh=0.97_min=5",  "delta_high": 0.97, "min_per_frame": 5},
        # min=10 sweep across delta range
        {"label": "dh=0.80_min=10", "delta_high": 0.80, "min_per_frame": 10},
        {"label": "dh=0.85_min=10", "delta_high": 0.85, "min_per_frame": 10},
        {"label": "dh=0.90_min=10", "delta_high": 0.90, "min_per_frame": 10},
        {"label": "dh=0.95_min=10", "delta_high": 0.95, "min_per_frame": 10},
        {"label": "dh=0.97_min=10", "delta_high": 0.97, "min_per_frame": 10},
    ]

    # ---- Run sweep ----
    all_results = run_sweep(model, env, task_oracle, val_annotations,
                            eval_sequences, configs)

    # ---- Summary ----
    print_summary(all_results)

    # ---- Save ----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
