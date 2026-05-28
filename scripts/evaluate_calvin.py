"""
CALVIN D→D evaluation for DiT-based policies (TaKSIE / GCBC-only / full DiT).

Runs the standard 1000-sequence × 5-chained-task benchmark with 360 steps per subtask.
Results are saved in the same JSON format as LCD/HULC baselines (append-mode).

Usage:
  # Full DiT + GCBC (TaKSIE-style)
  python scripts/evaluate_calvin.py \
      --flowdit_ckpt models/flowdit_droid_endgoal/model.pt \
      --gcbc_ckpt models/gcbc_abcd/gcbc_step400000.pt \
      --eval_log_dir results/baselines \
      --model_id taksie \
      --device 0

  # GCBC-only (no subgoal generation, uses ground-truth next-frame as goal)
  python scripts/evaluate_calvin.py \
      --gcbc_ckpt models/gcbc_abcd/gcbc_step400000.pt \
      --eval_log_dir results/baselines \
      --model_id gcbc \
      --mode gcbc_only \
      --device 0

  # Oracle-final: GT endpoint from training demos as fixed goal per subtask
  python scripts/evaluate_calvin.py \
      --gcbc_ckpt models/gcbc_abcd/gcbc_final.pt \
      --mode oracle_final \
      --model_id oracle_final \
      --device 0

  # Oracle-subgoal: GT intermediate frames every 20 steps (perfect subgoal gen)
  python scripts/evaluate_calvin.py \
      --gcbc_ckpt models/gcbc_abcd/gcbc_final.pt \
      --mode oracle_subgoal \
      --subgoal_interval 20 \
      --model_id oracle_subgoal \
      --device 0

  # Adjust progress evaluator threshold
  python scripts/evaluate_calvin.py \
      --flowdit_ckpt models/flowdit_droid_endgoal/model.pt \
      --gcbc_ckpt models/gcbc_abcd/gcbc_step400000.pt \
      --delta_high 0.85 \
      --max_per_frame 25 \
      --model_id taksie_dh085 \
      --device 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    get_env_state_for_initial_condition,
)
from calvin_env.envs.play_table_env import get_env

logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000


# ============================================================================
# Result helpers (matching LCD/HULC format)
# ============================================================================

def count_success(results: list[int]) -> list[float]:
    """Compute chain success rates for 1..5 consecutive tasks."""
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in range(i, 6))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(results, sequences, log_dir: Path, model_id: str):
    """Print results and append to results.json (LCD-compatible format)."""
    avg_seq_len = float(np.mean(results))
    chain_sr = {str(i + 1): sr for i, sr in enumerate(count_success(results))}

    print(f"\nResults for {model_id}:")
    print(f"Average successful sequence length: {avg_seq_len:.3f}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"  {i}: {sr * 100:.1f}%")

    # Per-task breakdown
    cnt_success: Counter = Counter()
    cnt_fail: Counter = Counter()
    for result, (_, sequence) in zip(results, sequences):
        for task in sequence[:result]:
            cnt_success[task] += 1
        if result < len(sequence):
            cnt_fail[sequence[result]] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in sorted(total):
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        sr = cnt_success[task] / total[task] * 100
        print(f"  {task}: {cnt_success[task]}/{total[task]} | SR: {sr:.1f}%")

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

    # Append to existing results.json
    results_path = log_dir / "results.json"
    previous = {}
    if results_path.exists():
        with open(results_path) as f:
            previous = json.load(f)

    previous[model_id] = data
    with open(results_path, "w") as f:
        json.dump(previous, f, indent=2)
    print(f"\nSaved to {results_path} (key: {model_id})")


# ============================================================================
# Evaluation loop
# ============================================================================

def evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence,
                      val_annotations, debug=False):
    """Evaluate one 5-task chain. Returns number of consecutive successes (0-5)."""
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    for subtask in eval_sequence:
        success = rollout(env, model, task_oracle, subtask, val_annotations, debug)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, debug=False):
    """Run one subtask rollout (up to EP_LEN steps). Returns True on success."""
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    if hasattr(model, 'set_task'):
        model.set_task(subtask)
    start_info = env.get_info()

    for step in range(EP_LEN):
        action = model.step(obs, lang_annotation, step=step)
        obs, _, _, current_info = env.step(action)

        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(f"  {subtask}: success at step {step}")
            return True

    if debug:
        print(f"  {subtask}: fail")
    return False


def evaluate_policy(model, env, model_id: str, eval_log_dir: str,
                    num_sequences: int = NUM_SEQUENCES, debug: bool = False):
    """Run full CALVIN D→D evaluation."""
    import calvin_agent
    conf_dir = Path(calvin_agent.__file__).absolute().parents[1] / "conf"
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(
        conf_dir / "annotations/new_playtable_validation.yaml")

    log_dir = Path(eval_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    eval_sequences = get_sequences(num_sequences)

    progress_log = log_dir / f"{model_id}_progress.log"
    results = []
    pbar = tqdm(eval_sequences, position=0, leave=True)
    with open(progress_log, "w") as pf:
        pf.write(f"# CALVIN eval: model_id={model_id}  n_seq={num_sequences}\n")
        pf.flush()
        for initial_state, eval_sequence in pbar:
            result = evaluate_sequence(
                env, model, task_oracle, initial_state, eval_sequence,
                val_annotations, debug)
            results.append(result)

            chain_sr = count_success(results)
            desc = " ".join([f"{i+1}/5:{v*100:.0f}%"
                             for i, v in enumerate(chain_sr)])
            avg = np.mean(results)

            # Stdout per-sequence — survives in log files (tqdm \r does not)
            print(f"[{len(results):3d}/{num_sequences}] result={result} | "
                  f"avg_len={avg:.2f} | {desc}", flush=True)

            if len(results) >= 3:
                pbar.set_description(desc)

            pf.write(f"[{len(results):4d}/{num_sequences}]  "
                     f"result={result}  avg_len={avg:.3f}  {desc}\n")
            pf.flush()

    print(f"Progress log: {progress_log}")
    print_and_save(results, list(get_sequences(num_sequences)), log_dir, model_id)

    return results


# ============================================================================
# GCBC-only wrapper (no DiT subgoals)
# ============================================================================

class GCBCOnlyWrapper:
    """GCBC-only policy that re-encodes each frame as both obs and goal.

    Without DiT to generate subgoals, this feeds the current observation's
    Theia encoding as both z_obs and z_goal. This tests the GCBC policy in
    isolation — essentially an identity-goal baseline.

    For a proper GCBC-only eval, you'd supply ground-truth future frames or
    use a different goal conditioning. This serves as a lower bound.
    """

    def __init__(self, gcbc_ckpt: str, theia_path: str = "models/theia_small_cdiv",
                 act_stats_path: str = "/mnt/sda1/Datasets/chal2525/gcbc_data_abcd/act_stats.npz",
                 gcbc_params: dict | None = None, act_pred_horizon: int = 5,
                 device: str = "cuda"):
        from transformers import AutoModel
        from models.diffusion_policy import FlowMatchingGCBCPolicy

        self.device = torch.device(device)
        self.act_pred_horizon = act_pred_horizon

        # Theia
        self.theia = AutoModel.from_pretrained(
            theia_path, trust_remote_code=True
        ).to(self.device).eval()
        for p in self.theia.parameters():
            p.requires_grad_(False)

        # GCBC
        _gp = gcbc_params or {
            "action_dim": 7,
            "act_pred_horizon": act_pred_horizon,
            "theia_dim": 384,
            "num_sampling_steps": 4,
            "time_dim": 32,
            "hidden_dim": 256,
            "num_blocks": 3,
            "dropout": 0.0,
        }
        self.gcbc = FlowMatchingGCBCPolicy(**_gp).to(self.device).eval()
        ckpt = torch.load(gcbc_ckpt, map_location=self.device, weights_only=False)
        state = ckpt.get("ema_model", ckpt.get("model", ckpt))
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.gcbc.load_state_dict(state, strict=True)

        # Action stats
        stats = np.load(act_stats_path)
        self.act_mean = torch.from_numpy(stats["mean"]).float().to(self.device)
        self.act_std = torch.from_numpy(stats["std"]).float().to(self.device)

        # Action ensembling
        from flowdit_calvin_wrapper import ActionEnsembleBuffer
        self._action_ensemble = ActionEnsembleBuffer(
            n_chunks=4, horizon=act_pred_horizon, action_dim=7)

    def reset(self):
        self._action_ensemble.reset()

    @torch.no_grad()
    def step(self, obs: dict, lang_annotation: str, step: int = 0) -> np.ndarray:
        rgb = obs["rgb_obs"]["rgb_static"]
        z_obs = self.theia.forward_feature(
            rgb, do_resize=True, do_rescale=True, do_normalize=True)
        # Use current obs as goal (identity baseline)
        z_goal = z_obs

        # GCBC was trained pure fp32 (no autocast) — must match at eval
        act_norm = self.gcbc.sample_actions(z_obs, z_goal)
        act_norm = act_norm.squeeze(0)
        actions = (act_norm * self.act_std + self.act_mean).cpu().numpy()

        ensembled = self._action_ensemble.insert_and_ensemble(actions)
        action = ensembled.copy()
        action[-1] = 1.0 if action[-1] >= 0 else -1.0
        return action


# ============================================================================
# Oracle GCBC baselines (GT goals from dataset)
# ============================================================================

class _OracleGCBCBase:
    """Shared init for oracle GCBC baselines.

    Loads Theia (for live obs encoding), GCBC policy, action stats, and
    pre-encoded training features with task-to-segment annotations.
    """

    _DEFAULT_FEATURES = "/mnt/sda1/Datasets/chal2525/gcbc_data_abcd/theia_features.npy"
    _DEFAULT_ANN = "/mnt/sda1/Datasets/chal2525/calvin/task_ABC_D/training/lang_annotations/auto_lang_ann.npy"
    _DEFAULT_STATS = "/mnt/sda1/Datasets/chal2525/gcbc_data_abcd/act_stats.npz"

    def __init__(self, gcbc_ckpt: str,
                 features_path: str | None = None,
                 ann_path: str | None = None,
                 act_stats_path: str | None = None,
                 theia_path: str = "models/theia_small_cdiv",
                 gcbc_params: dict | None = None,
                 act_pred_horizon: int = 5,
                 device: str = "cuda"):
        from transformers import AutoModel
        from models.diffusion_policy import FlowMatchingGCBCPolicy
        from flowdit_calvin_wrapper import ActionEnsembleBuffer

        self.device = torch.device(device)
        self.act_pred_horizon = act_pred_horizon

        # Theia (for encoding live observations)
        self.theia = AutoModel.from_pretrained(
            theia_path, trust_remote_code=True
        ).to(self.device).eval()
        for p in self.theia.parameters():
            p.requires_grad_(False)

        # GCBC policy
        _gp = gcbc_params or {
            "action_dim": 7,
            "act_pred_horizon": act_pred_horizon,
            "theia_dim": 384,
            "num_sampling_steps": 4,
            "time_dim": 32,
            "hidden_dim": 256,
            "num_blocks": 3,
            "dropout": 0.0,
        }
        self.gcbc = FlowMatchingGCBCPolicy(**_gp).to(self.device).eval()
        ckpt = torch.load(gcbc_ckpt, map_location=self.device, weights_only=False)
        state = ckpt.get("ema_model", ckpt.get("model", ckpt))
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.gcbc.load_state_dict(state, strict=True)

        # Action stats
        stats = np.load(act_stats_path or self._DEFAULT_STATS)
        self.act_mean = torch.from_numpy(stats["mean"]).float().to(self.device)
        self.act_std = torch.from_numpy(stats["std"]).float().to(self.device)

        # Pre-encoded Theia features (memory-mapped)
        feat_path = features_path or self._DEFAULT_FEATURES
        self.features = np.load(feat_path, mmap_mode="r")

        # Task → segment mapping from annotations
        ann = np.load(ann_path or self._DEFAULT_ANN, allow_pickle=True).item()
        self.task_to_segments: dict[str, list[tuple[int, int]]] = {}
        for task_name, (start, end) in zip(
                ann["language"]["task"], ann["info"]["indx"]):
            self.task_to_segments.setdefault(task_name, []).append((start, end))
        n_seg = sum(len(v) for v in self.task_to_segments.values())
        print(f"Oracle: {len(self.task_to_segments)} tasks, {n_seg} segments, "
              f"features {self.features.shape}")

        # Action ensembling
        self._action_ensemble = ActionEnsembleBuffer(
            n_chunks=4, horizon=act_pred_horizon, action_dim=7)
        self._z_goal = None

    @torch.no_grad()
    def _encode_obs(self, rgb):
        return self.theia.forward_feature(
            rgb, do_resize=True, do_rescale=True, do_normalize=True)

    def _load_features(self, idx):
        return torch.from_numpy(
            self.features[idx:idx + 1].astype(np.float32)).to(self.device)

    @torch.no_grad()
    def _act(self, z_obs, z_goal):
        # GCBC was trained pure fp32 (no autocast) — must match at eval
        act_norm = self.gcbc.sample_actions(z_obs, z_goal)
        act_norm = act_norm.squeeze(0)
        actions = (act_norm * self.act_std + self.act_mean).cpu().numpy()
        ensembled = self._action_ensemble.insert_and_ensemble(actions)
        action = ensembled.copy()
        action[-1] = 1.0 if action[-1] >= 0 else -1.0
        return action


class OracleFinalWrapper(_OracleGCBCBase):
    """GCBC with GT final frame as fixed goal per subtask.

    For each subtask, looks up a demonstration of that task in the training
    set and uses the LAST frame (completed state) as a fixed goal throughout.
    Isolates the policy: 'given a perfect endpoint goal, can it reach it?'
    """

    def set_task(self, task_name: str):
        if task_name in self.task_to_segments:
            segs = self.task_to_segments[task_name]
            _, end = segs[np.random.randint(len(segs))]
            self._z_goal = self._load_features(end)
        else:
            print(f"  WARNING: '{task_name}' not in annotations")
            self._z_goal = None

    def reset(self):
        self._action_ensemble.reset()

    @torch.no_grad()
    def step(self, obs, lang_annotation, step=0):
        z_obs = self._encode_obs(obs["rgb_obs"]["rgb_static"])
        z_goal = self._z_goal if self._z_goal is not None else z_obs
        return self._act(z_obs, z_goal)


class OracleSubgoalWrapper(_OracleGCBCBase):
    """GCBC with GT subgoals refreshed every N steps.

    Loads the full demonstration trajectory for each subtask and advances
    through it every ``subgoal_interval`` steps, simulating a perfect
    subgoal generator.  The last subgoal is always the completed state.
    """

    def __init__(self, *args, subgoal_interval: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.subgoal_interval = subgoal_interval
        self._demo_features: torch.Tensor | None = None

    def set_task(self, task_name: str):
        if task_name in self.task_to_segments:
            segs = self.task_to_segments[task_name]
            start, end = segs[np.random.randint(len(segs))]
            feats = self.features[start:end + 1].astype(np.float32)
            self._demo_features = torch.from_numpy(feats).to(self.device)
        else:
            print(f"  WARNING: '{task_name}' not in annotations")
            self._demo_features = None

    def reset(self):
        self._action_ensemble.reset()

    def _get_subgoal(self, step: int):
        """Map eval step to a demo frame used as subgoal."""
        if self._demo_features is None:
            return None
        L = len(self._demo_features)
        n_slots = max(1, EP_LEN // self.subgoal_interval)  # 18 for interval=20
        slot = step // self.subgoal_interval
        # Evenly spread across demo; last slot → final (completed) frame
        demo_idx = min(round((slot + 1) / n_slots * (L - 1)), L - 1)
        return self._demo_features[demo_idx:demo_idx + 1]

    @torch.no_grad()
    def step(self, obs, lang_annotation, step=0):
        z_obs = self._encode_obs(obs["rgb_obs"]["rgb_static"])
        z_goal = self._get_subgoal(step)
        if z_goal is None:
            z_goal = z_obs
        return self._act(z_obs, z_goal)


# ============================================================================
# LCBC wrapper (language-conditioned, no goal frame)
# ============================================================================

class LCBCWrapper:
    """Language-conditioned BC policy.

    Theia obs encoder + CLIP language embedding as goal. Mirrors GCBCOnlyWrapper
    in structure but uses a LangFlowMatchingGCBCPolicy.
    """

    def __init__(self, lcbc_ckpt: str,
                 theia_path: str = "models/theia_small_cdiv",
                 clip_path: str = "models/clip-vit-large-patch14",
                 act_stats_path: str = "/mnt/sda1/Datasets/chal2525/gcbc_data_abcd/act_stats.npz",
                 act_pred_horizon: int = 5,
                 device: str = "cuda"):
        from transformers import AutoModel, CLIPTokenizer, CLIPTextModel
        from models.lang_gcbc_policy import LangFlowMatchingGCBCPolicy
        from flowdit_calvin_wrapper import ActionEnsembleBuffer

        self.device = torch.device(device)
        self.act_pred_horizon = act_pred_horizon

        # Theia (live obs)
        self.theia = AutoModel.from_pretrained(
            theia_path, trust_remote_code=True
        ).to(self.device).eval()
        for p in self.theia.parameters():
            p.requires_grad_(False)

        # CLIP text encoder (language goal)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path)
        self.clip_text = CLIPTextModel.from_pretrained(clip_path).to(self.device).eval()
        for p in self.clip_text.parameters():
            p.requires_grad_(False)

        # LCBC policy
        self.lcbc = LangFlowMatchingGCBCPolicy(
            action_dim=7, act_pred_horizon=act_pred_horizon,
            theia_dim=384, lang_dim=768, proj_dim=256,
            hidden_dim=256, num_blocks=3, dropout=0.0,
            num_sampling_steps=4,
        ).to(self.device).eval()
        ckpt = torch.load(lcbc_ckpt, map_location=self.device, weights_only=False)
        state = ckpt.get("ema_model", ckpt.get("model", ckpt))
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.lcbc.load_state_dict(state, strict=True)

        # Action stats
        stats = np.load(act_stats_path)
        self.act_mean = torch.from_numpy(stats["mean"]).float().to(self.device)
        self.act_std = torch.from_numpy(stats["std"]).float().to(self.device)

        self._action_ensemble = ActionEnsembleBuffer(
            n_chunks=4, horizon=act_pred_horizon, action_dim=7)
        self._lang_cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _encode_lang(self, text: str) -> torch.Tensor:
        if text not in self._lang_cache:
            enc = self.clip_tokenizer([text], padding="max_length", truncation=True,
                                      max_length=77, return_tensors="pt").to(self.device)
            self._lang_cache[text] = self.clip_text(**enc).pooler_output.float()  # (1, 768)
        return self._lang_cache[text]

    def reset(self):
        self._action_ensemble.reset()

    @torch.no_grad()
    def step(self, obs: dict, lang_annotation: str, step: int = 0) -> np.ndarray:
        rgb = obs["rgb_obs"]["rgb_static"]
        z_obs = self.theia.forward_feature(
            rgb, do_resize=True, do_rescale=True, do_normalize=True)
        lang = self._encode_lang(lang_annotation)

        # LCBC trained pure fp32 — must match at eval
        act_norm = self.lcbc.sample_actions(z_obs, lang)
        act_norm = act_norm.squeeze(0)
        actions = (act_norm * self.act_std + self.act_mean).cpu().numpy()

        ensembled = self._action_ensemble.insert_and_ensemble(actions)
        action = ensembled.copy()
        action[-1] = 1.0 if action[-1] >= 0 else -1.0
        return action


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CALVIN D→D evaluation for DiT / GCBC")
    parser.add_argument("--flowdit_ckpt", type=str, default=None,
                        help="Path to DiT checkpoint (required for taksie mode)")
    parser.add_argument("--gcbc_ckpt", type=str, default=None,
                        help="Path to GCBC policy checkpoint (required for non-lcbc modes)")
    parser.add_argument("--lcbc_ckpt", type=str, default=None,
                        help="Path to LCBC policy checkpoint (required for lcbc mode)")
    parser.add_argument("--mode", type=str, default="taksie",
                        choices=["taksie", "fixed", "gcbc_only", "oracle_final", "oracle_subgoal", "lcbc"],
                        help="taksie = DiT+GCBC, gcbc_only = GCBC with identity goal, "
                             "oracle_final = GT endpoint goal, oracle_subgoal = GT subgoals every N steps, "
                             "lcbc = language-conditioned BC (no goal frame)")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Key name in results.json (default: mode name)")
    parser.add_argument("--eval_log_dir", type=str, default="results/baselines",
                        help="Directory to save results.json")
    parser.add_argument("--env_cfg", type=str, default="data/calvin/task_D_D/validation",
                        help="Path to CALVIN env config directory")
    parser.add_argument("--act_stats_path", type=str, default=None,
                        help="Path to act_stats.npz (overrides default)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--num_sequences", type=int, default=1000,
                        help="Number of evaluation sequences")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    # DiT / progress evaluator params
    parser.add_argument("--act_pred_horizon", type=int, default=5,
                        help="GCBC action chunk size — must match the checkpoint's training horizon")
    parser.add_argument("--distilled", action="store_true",
                        help="DiT checkpoint is a CFG-distilled student → bypass CFG (1 fwd per Euler step)")
    parser.add_argument("--ensemble_mode", type=str, default="uniform_mask",
                        choices=["uniform_mask", "exp", "open_loop"],
                        help="Action chunk handling at execution time")
    parser.add_argument("--exp_decay_k", type=float, default=0.5,
                        help="Decay for ACT-style exp weighting (only used when --ensemble_mode exp)")
    parser.add_argument("--open_loop_window", type=int, default=5,
                        help="Actions executed open-loop per regen (only used when --ensemble_mode open_loop)")
    parser.add_argument("--delta_high", type=float, default=0.90)
    parser.add_argument("--max_per_frame", type=int, default=20)
    parser.add_argument("--min_per_frame", type=int, default=1)
    parser.add_argument("--flowdit_num_steps", type=int, default=8)
    parser.add_argument("--context_cfg_scale", type=float, default=2.5)
    parser.add_argument("--prompt_cfg_scale", type=float, default=7.5)
    parser.add_argument("--config", type=str, default=None,
                        help="DiT config YAML (for model architecture params)")

    # Oracle baseline params
    parser.add_argument("--features_path", type=str, default=None,
                        help="Pre-encoded Theia features .npy (for oracle modes)")
    parser.add_argument("--ann_path", type=str, default=None,
                        help="CALVIN auto_lang_ann.npy path (for oracle modes)")
    parser.add_argument("--subgoal_interval", type=int, default=20,
                        help="Steps between subgoal advances (oracle_subgoal mode)")

    args = parser.parse_args()

    device = f"cuda:{args.device}"
    model_id = args.model_id or args.mode

    # ---- Build environment ----
    # Only request static + gripper RGB (exclude tactile sensor which needs
    # a full tacto install)
    obs_space = {"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []}
    print(f"Loading CALVIN environment from {args.env_cfg}...")
    env = get_env(args.env_cfg, obs_space=obs_space, show_gui=False)
    print("Environment ready.")

    # ---- Build policy ----
    if args.mode in ("taksie", "fixed"):
        if args.flowdit_ckpt is None:
            parser.error("--flowdit_ckpt is required for taksie/fixed mode")

        from flowdit_calvin_wrapper import build_wrapper_from_config
        overrides = dict(
            delta_high=args.delta_high,
            max_per_frame=args.max_per_frame,
            min_per_frame=args.min_per_frame,
            flowdit_num_steps=args.flowdit_num_steps,
            context_cfg_scale=args.context_cfg_scale,
            prompt_cfg_scale=args.prompt_cfg_scale,
            fixed_interval=(args.mode == "fixed"),
            act_pred_horizon=args.act_pred_horizon,
            distilled=args.distilled,
            ensemble_mode=args.ensemble_mode,
            exp_decay_k=args.exp_decay_k,
            open_loop_window=args.open_loop_window,
        )
        if args.act_stats_path:
            overrides["act_stats_path"] = args.act_stats_path
        model = build_wrapper_from_config(
            flowdit_ckpt=args.flowdit_ckpt,
            gcbc_ckpt=args.gcbc_ckpt,
            config_path=args.config,
            device=device,
            **overrides,
        )
        model.reset_regen_stats()
        if args.mode == "fixed":
            print(f"DiT+GCBC wrapper loaded (FIXED interval = {args.max_per_frame} steps, "
                  f"no progress tracking)")
        else:
            print(f"DiT+GCBC wrapper loaded (delta_high={args.delta_high}, "
                  f"max_per_frame={args.max_per_frame})")
    elif args.mode == "gcbc_only":
        gcbc_kwargs = {"gcbc_ckpt": args.gcbc_ckpt, "device": device}
        if args.act_stats_path:
            gcbc_kwargs["act_stats_path"] = args.act_stats_path
        model = GCBCOnlyWrapper(**gcbc_kwargs)
        print("GCBC-only wrapper loaded (identity goal baseline)")

    elif args.mode == "lcbc":
        if args.lcbc_ckpt is None:
            parser.error("--lcbc_ckpt is required for lcbc mode")
        lcbc_kwargs = {"lcbc_ckpt": args.lcbc_ckpt, "device": device}
        if args.act_stats_path:
            lcbc_kwargs["act_stats_path"] = args.act_stats_path
        model = LCBCWrapper(**lcbc_kwargs)
        print("LCBC wrapper loaded (Theia obs + CLIP language goal)")

    elif args.mode in ("oracle_final", "oracle_subgoal"):
        oracle_kwargs = {
            "gcbc_ckpt": args.gcbc_ckpt,
            "device": device,
        }
        if args.features_path:
            oracle_kwargs["features_path"] = args.features_path
        if args.ann_path:
            oracle_kwargs["ann_path"] = args.ann_path
        if args.act_stats_path:
            oracle_kwargs["act_stats_path"] = args.act_stats_path

        if args.mode == "oracle_final":
            model = OracleFinalWrapper(**oracle_kwargs)
            print("Oracle-final wrapper loaded (GT endpoint goal)")
        else:
            model = OracleSubgoalWrapper(
                **oracle_kwargs, subgoal_interval=args.subgoal_interval)
            print(f"Oracle-subgoal wrapper loaded (GT subgoals every "
                  f"{args.subgoal_interval} steps)")

    # ---- Run evaluation ----
    print(f"\nStarting CALVIN ABCD→D evaluation: {args.num_sequences} sequences, model_id={model_id}")
    t0 = time.time()
    results = evaluate_policy(
        model, env, model_id=model_id,
        eval_log_dir=args.eval_log_dir,
        num_sequences=args.num_sequences,
        debug=args.debug,
    )
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")

    # Print regen stats for taksie/fixed modes
    if args.mode in ("taksie", "fixed") and hasattr(model, "regen_stats"):
        stats = model.regen_stats
        print(f"\nSubgoal regen stats:")
        print(f"  Total steps: {stats['total_steps']}")
        print(f"  Subgoal generations: {stats['subgoal_gen_count']}")
        print(f"  Mean regen interval: {stats['mean_regen_interval']:.1f}")
        print(f"  Median regen interval: {stats['median_regen_interval']:.1f}")

        # Save regen stats alongside results
        regen_path = Path(args.eval_log_dir) / f"{model_id}_regen_stats.json"
        stats_save = {k: v for k, v in stats.items() if k != "regen_intervals"}
        stats_save["regen_interval_histogram"] = dict(Counter(stats["regen_intervals"]))
        with open(regen_path, "w") as f:
            json.dump(stats_save, f, indent=2)
        print(f"  Saved regen stats to {regen_path}")


if __name__ == "__main__":
    main()
