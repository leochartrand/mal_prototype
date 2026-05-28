"""
DiT + GCBC Policy wrapper for CALVIN LH-MTLC evaluation.

Implements the CalvinBaseModel interface:
    model.reset()              — called before each subtask
    model.step(obs, lang, step) — returns 7-dim action

Pipeline per step:
  1. Encode observation rgb_static with Theia → z_obs (1, 196, 384)
  2. On first step or progress evaluator trigger: generate subgoal with DiT
  3. Generate action chunk with FlowMatchingGCBCPolicy → (1, H, 7)
  4. Temporal action ensembling across overlapping chunks (TaKSIE-style)
  5. Denormalize action and return

Action ensembling matches TaKSIE's approach:
  - 4-chunk buffer, each chunk predicts H timesteps
  - Every step: shift buffer, insert new chunk, weighted-average column 0
  - Recency bias: newest chunk contributes H votes, oldest contributes 1
  - Smooths jerky transitions between chunks
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from models.diffusion_policy import FlowMatchingGCBCPolicy
from models.flowdit import DiTAir
from models.progress_evaluator import TheiaProgressEvaluator


# ============================================================================
# TaKSIE-style temporal action ensembling
# ============================================================================

class ActionEnsembleBuffer:
    """Overlapping action chunk buffer with recency-weighted ensembling.

    Matches TaKSIE's predict_action temporal smoothing:
    - Maintains N_chunks overlapping action chunks, each of length H
    - Every step: shift buffer forward, insert new chunk, compute weighted avg
    - Weight mask: newer chunks have more valid timesteps → higher weight
    - Result: smooth transitions instead of jerky chunk boundaries

    Buffer layout (N_chunks=4, H=4 example):
        Row 0 (newest): [a0, a1, a2, a3]  weight at col 0: 4
        Row 1:          [a0, a1, a2, -- ]  weight at col 0: 3
        Row 2:          [a0, a1, --, -- ]  weight at col 0: 2
        Row 3 (oldest): [a0, --, --, -- ]  weight at col 0: 1

    Action at current step = weighted mean of column 0 across all chunks.
    """

    def __init__(self, n_chunks: int = 4, horizon: int = 5, action_dim: int = 7,
                 weighting: str = "uniform_mask", exp_decay_k: float = 0.5):
        """weighting:
            "uniform_mask" — uniform mean over valid chunks (legacy).
            "exp"          — exp(-exp_decay_k * chunk_age) weighting, ACT-style.
        """
        self.n_chunks = n_chunks
        self.horizon = horizon
        self.action_dim = action_dim
        self.weighting = weighting
        self.exp_decay_k = exp_decay_k
        self.reset()

    def reset(self):
        self.buffer = np.zeros((self.n_chunks, self.horizon, self.action_dim))
        self.mask = np.zeros((self.n_chunks, self.horizon), dtype=bool)

    def _build_validity_mask(self):
        """Lower-triangular validity: chunk i has (H - i) valid timesteps."""
        validity = np.zeros((self.n_chunks, self.horizon), dtype=bool)
        for i in range(self.n_chunks):
            valid_len = self.horizon - i
            if valid_len > 0:
                validity[i, :valid_len] = True
        return validity

    def insert_and_ensemble(self, new_actions: np.ndarray) -> np.ndarray:
        """Insert new action chunk, shift buffer, return ensembled action.

        Args:
            new_actions: (H, action_dim) new action chunk

        Returns:
            action: (action_dim,) weighted-average action for current step
        """
        # Shift chunks: oldest is discarded
        self.buffer[1:, :, :] = self.buffer[:-1, :, :]
        self.mask[1:, :] = self.mask[:-1, :]

        # Shift timesteps within each chunk forward (advance cursor)
        self.buffer[:, :-1, :] = self.buffer[:, 1:, :]
        self.mask[:, :-1] = self.mask[:, 1:]

        # Apply validity mask (recency weighting)
        self.mask = self.mask & self._build_validity_mask()

        # Insert new chunk at row 0
        H = min(len(new_actions), self.horizon)
        self.buffer[0, :H, :] = new_actions[:H]
        self.buffer[0, H:, :] = 0.0
        self.mask[0, :] = False
        self.mask[0, :H] = True

        # Weighted average of column 0 (current timestep)
        col_mask = self.mask[:, 0]  # (n_chunks,) bool — True for chunks with valid data at this step
        if col_mask.sum() == 0:
            return new_actions[0]

        if self.weighting == "exp":
            # ACT-style exp(-k*i): newer chunks (i=0) weighted most, older less.
            age = np.arange(self.n_chunks, dtype=np.float32)  # 0 = newest, n-1 = oldest
            w = np.exp(-self.exp_decay_k * age) * col_mask.astype(np.float32)
        else:  # "uniform_mask"
            w = col_mask.astype(np.float32)

        w = w[:, None]  # (n_chunks, 1)
        action = (self.buffer[:, 0, :] * w).sum(axis=0) / w.sum(axis=0)
        return action


class FlowDiTCalvinWrapper:
    """CALVIN evaluation wrapper combining DiT subgoal generation,
    Theia progress evaluation, and diffusion GCBC action generation."""

    def __init__(
        self,
        flowdit_ckpt: str,
        gcbc_ckpt: str,
        theia_path: str = "models/theia_small_cdiv",
        text_model_path: str = "models/all-MiniLM-L6-v2",
        act_stats_path: str = "data/gcbc/training/act_stats.npz",
        # DiT params
        scale_factor: float = 1.702949,
        flowdit_num_steps: int = 50,
        context_cfg_scale: float = 2.5,
        prompt_cfg_scale: float = 7.5,
        # DiT model architecture
        flowdit_params: dict | None = None,
        # GCBC params
        gcbc_params: dict | None = None,
        # Progress evaluator params
        delta_high: float = 0.90,
        max_per_frame: int = 20,
        min_per_frame: int = 1,
        # Subgoal scheduling: if True, skip TheiaProgressEvaluator entirely and
        # regen every `max_per_frame` steps (SuSIE-style fixed interval).
        fixed_interval: bool = False,
        # Action params
        act_pred_horizon: int = 5,
        # DiT distilled student: bypass CFG (1 fwd per Euler step instead of 3)
        distilled: bool = False,
        # Action ensemble mode: how to combine overlapping chunks at execution time
        #   "uniform_mask" — uniform mean of valid chunks (current default, mild smoothing)
        #   "exp"          — ACT-style exp(-exp_decay_k * chunk_age) weighting; newer chunks dominate
        #   "open_loop"    — no ensemble; execute open_loop_window actions from a single chunk
        #                    before re-encoding obs and re-predicting (FLOWER-style)
        ensemble_mode: str = "uniform_mask",
        exp_decay_k: float = 0.5,
        open_loop_window: int = 5,
        # Device
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.scale_factor = scale_factor
        self.flowdit_num_steps = flowdit_num_steps
        self.context_cfg_scale = context_cfg_scale
        self.prompt_cfg_scale = prompt_cfg_scale
        self.act_pred_horizon = act_pred_horizon
        self.distilled = distilled
        assert ensemble_mode in ("uniform_mask", "exp", "open_loop"), \
            f"ensemble_mode must be one of {{uniform_mask, exp, open_loop}}, got {ensemble_mode}"
        self.ensemble_mode = ensemble_mode
        self.exp_decay_k = exp_decay_k
        self.open_loop_window = open_loop_window

        # ---- Theia encoder ----
        self.theia = AutoModel.from_pretrained(
            theia_path, trust_remote_code=True
        ).to(self.device).eval()
        for p in self.theia.parameters():
            p.requires_grad_(False)

        # ---- Text encoder (all-MiniLM-L6-v2) ----
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_path)
        self.text_model = AutoModel.from_pretrained(text_model_path).to(self.device).eval()
        for p in self.text_model.parameters():
            p.requires_grad_(False)
        self.max_text_len = 25
        self.text_dim = self.text_model.config.hidden_size  # 384

        # ---- DiT (subgoal generator) ----
        _fp = flowdit_params or {
            "latent_dim": 384,
            "num_patches": 196,
            "hidden_dim": 896,
            "depth": 18,
            "num_heads": 14,
            "text_dim": 384,
            "pooled_text_dim": 384,
            "max_text_len": 25,
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "use_pooled_text": False,
            "cfg_drop_prompt": 0.0,
            "cfg_drop_context": 0.0,
            "cfg_drop_both": 0.0,
        }
        self.flowdit = DiTAir(**_fp).to(self.device).eval()
        self._load_flowdit(flowdit_ckpt)

        # ---- GCBC policy ----
        _gp = gcbc_params or {
            "action_dim": 7,
            "act_pred_horizon": act_pred_horizon,
            "theia_dim": 384,
            "num_sampling_steps": 4,
            "time_dim": 32,
            "hidden_dim": 256,
            "num_blocks": 3,
            "dropout": 0.0,  # no dropout at inference
        }
        self.gcbc = FlowMatchingGCBCPolicy(**_gp).to(self.device).eval()
        self._load_gcbc(gcbc_ckpt)

        # ---- Action normalization stats ----
        stats = np.load(act_stats_path)
        self.act_mean = torch.from_numpy(stats["mean"]).float().to(self.device)   # (7,)
        self.act_std = torch.from_numpy(stats["std"]).float().to(self.device)     # (7,)

        # ---- Subgoal scheduling ----
        self.fixed_interval = fixed_interval
        self.fixed_interval_n = max_per_frame
        # Progress evaluator is only used when fixed_interval=False
        self.progress_eval = TheiaProgressEvaluator(
            delta_high=delta_high,
            max_steps=max_per_frame,
            min_steps=min_per_frame,
        ) if not fixed_interval else None

        # ---- Runtime state ----
        self._z_subgoal = None        # (1, 196, 384) current subgoal
        self._lang_cache = None       # cached (text_hidden, text_mask, text_pooled)
        self._prev_lang = None        # previous language string (for cache hit)

        # Temporal action ensembling buffer (used in {uniform_mask, exp} modes)
        n_ensemble_chunks = 4  # number of overlapping chunks
        self._action_ensemble = ActionEnsembleBuffer(
            n_chunks=n_ensemble_chunks,
            horizon=act_pred_horizon,
            action_dim=7,
            weighting=ensemble_mode if ensemble_mode in ("uniform_mask", "exp") else "uniform_mask",
            exp_decay_k=exp_decay_k,
        )
        self._steps_since_plan = 0    # steps since last action chunk generation

        # Open-loop cached chunk state (only used in ensemble_mode="open_loop")
        self._cached_chunk = None     # (H, 7) numpy — last predicted chunk
        self._chunk_position = 0      # how many actions consumed from cached chunk

        # ---- Subgoal regen tracking ----
        self._total_steps = 0
        self._subgoal_gen_count = 0
        self._regen_intervals = []    # steps between consecutive subgoal gens
        self._steps_since_last_gen = 0

    def _load_flowdit(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state = ckpt.get("model", ckpt)
        # Strip DDP 'module.' prefix
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.flowdit.load_state_dict(state, strict=False)
        if missing:
            print(f"[DiT] Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"[DiT] Unexpected keys: {unexpected[:5]}...")

    def _load_gcbc(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state = ckpt.get("ema_model", ckpt.get("model", ckpt))
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.gcbc.load_state_dict(state, strict=True)

    # ================================================================
    # Text encoding
    # ================================================================

    @torch.no_grad()
    def _encode_text(self, text: str):
        """Encode a single text string → (text_hidden, text_mask, text_pooled).

        Returns:
            text_hidden: (1, 25, 384)  per-token hidden states
            text_mask:   (1, 25)       attention mask
            text_pooled: (1, 384)      mask-weighted mean pool
        """
        if text == self._prev_lang and self._lang_cache is not None:
            return self._lang_cache

        inputs = self.tokenizer(
            [text], return_tensors="pt",
            padding="max_length", max_length=self.max_text_len, truncation=True,
        )
        attn_mask = inputs["attention_mask"].to(self.device)         # (1, 25)
        input_ids = inputs["input_ids"].to(self.device)
        token_type_ids = inputs.get("token_type_ids")
        model_inputs = {"input_ids": input_ids, "attention_mask": attn_mask}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids.to(self.device)

        hidden = self.text_model(**model_inputs).last_hidden_state   # (1, 25, 384)

        # Mask-weighted mean pool → pooled
        mask_exp = attn_mask.unsqueeze(-1).float()                   # (1, 25, 1)
        pooled = (hidden * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)  # (1, 384)

        self._lang_cache = (hidden, attn_mask, pooled)
        self._prev_lang = text
        return self._lang_cache

    # ================================================================
    # Theia encoding
    # ================================================================

    @torch.no_grad()
    def _encode_obs(self, rgb: np.ndarray) -> torch.Tensor:
        """Encode 200×200 uint8 RGB → Theia features (1, 196, 384).

        Theia's forward_feature handles resize/rescale/normalize internally.
        """
        # Theia expects HWC uint8 or a PIL Image; forward_feature handles preprocessing
        z = self.theia.forward_feature(
            rgb, do_resize=True, do_rescale=True, do_normalize=True,
        )
        return z  # (1, 196, 384)

    # ================================================================
    # Subgoal generation
    # ================================================================

    @torch.no_grad()
    def _generate_subgoal(self, z_obs: torch.Tensor, text_hidden, text_mask, text_pooled):
        """Use DiT to generate a subgoal in Theia latent space.

        Args:
            z_obs: (1, 196, 384) current observation encoding
            text_hidden: (1, 25, 384)
            text_mask: (1, 25)
            text_pooled: (1, 384)

        Returns:
            z_subgoal: (1, 196, 384) generated subgoal in Theia space
        """
        z_init = z_obs * self.scale_factor
        # Distilled student bakes CFG into its forward → 1 fwd/step (3× faster).
        ctx_w  = None if self.distilled else self.context_cfg_scale
        prom_w = None if self.distilled else self.prompt_cfg_scale
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_gen = self.flowdit.sample_euler(
                z_init,
                text_hidden,
                text_mask=text_mask,
                pooled_text_emb=text_pooled,
                num_steps=self.flowdit_num_steps,
                context_cfg_scale=ctx_w,
                prompt_cfg_scale=prom_w,
            )
        # Convert back from scaled space
        z_subgoal = z_gen / self.scale_factor
        return z_subgoal

    # ================================================================
    # Action generation
    # ================================================================

    @torch.no_grad()
    def _generate_action_chunk(self, z_obs: torch.Tensor, z_goal: torch.Tensor) -> np.ndarray:
        """Generate an action chunk using the GCBC diffusion policy.

        Args:
            z_obs: (1, 196, 384) current observation
            z_goal: (1, 196, 384) subgoal target

        Returns:
            actions: (H, 7) numpy array of denormalized actions
        """
        # GCBC was trained pure fp32 (no autocast) — must match at eval
        act_norm = self.gcbc.sample_actions(z_obs, z_goal)  # (1, H, 7)
        act_norm = act_norm.squeeze(0)  # (H, 7)
        # Denormalize
        actions = act_norm * self.act_std + self.act_mean
        return actions.cpu().numpy()

    def _binarize_gripper(self, action: np.ndarray) -> np.ndarray:
        """Binarize gripper action: last dim ∈ {-1, 1}."""
        action = action.copy()
        action[-1] = 1.0 if action[-1] >= 0 else -1.0
        return action

    # ================================================================
    # CalvinBaseModel interface
    # ================================================================

    def reset(self):
        """Reset state for a new subtask.

        Called by CALVIN eval between chained subtasks. Clears subgoal, action
        buffer, and text cache so the next subtask starts fresh.
        Note: regen stats are NOT reset here — they accumulate across the full eval.
        """
        self._z_subgoal = None
        self._action_ensemble.reset()
        self._steps_since_plan = 0
        self._lang_cache = None
        self._prev_lang = None
        if self.progress_eval is not None:
            self.progress_eval.reset()
        self._steps_since_last_gen = 0
        self._cached_chunk = None
        self._chunk_position = 0

    def step(self, obs: dict, lang_annotation: str, step: int = 0) -> np.ndarray:
        """Execute one control step.

        Args:
            obs: CALVIN observation dict with obs['rgb_obs']['rgb_static'] (200,200,3) uint8
            lang_annotation: Natural language instruction string
            step: Current step within the episode (0-indexed)

        Returns:
            action: (7,) numpy array — 6 EE deltas + 1 gripper {-1, 1}
        """
        self._total_steps += 1

        # ── Open-loop mode: replay cached chunk if available, skip Theia/GCBC ──
        if (self.ensemble_mode == "open_loop"
                and self._cached_chunk is not None
                and self._chunk_position < min(self.open_loop_window, self.act_pred_horizon)):
            # Keep counters in env-step units so subgoal regen scheduling
            # (min_per_frame / max_per_frame) tracks real frames, not the
            # number of full-pipeline calls.
            self._steps_since_last_gen += 1
            if self.progress_eval is not None:
                self.progress_eval.step_count += 1
            action = self._binarize_gripper(self._cached_chunk[self._chunk_position])
            self._chunk_position += 1
            return action

        # ── Full path: encode obs, maybe regenerate subgoal, predict action chunk ──
        rgb = obs["rgb_obs"]["rgb_static"]  # (200, 200, 3) uint8
        z_obs = self._encode_obs(rgb)  # (1, 196, 384)

        # Encode text (cached if same instruction; auto-invalidates on change)
        text_hidden, text_mask, text_pooled = self._encode_text(lang_annotation)

        self._steps_since_last_gen += 1

        # Generate subgoal on first step
        if self._z_subgoal is None:
            self._z_subgoal = self._generate_subgoal(z_obs, text_hidden, text_mask, text_pooled)
            self._subgoal_gen_count += 1
            self._steps_since_last_gen = 0
            if self.progress_eval is not None:
                self.progress_eval.reset()

        # Advance subgoal — either fixed interval or via progress evaluator
        if self.fixed_interval:
            advance = self._steps_since_last_gen >= self.fixed_interval_n
        else:
            advance, sim, reason = self.progress_eval.should_advance(z_obs, self._z_subgoal)
        if advance:
            self._regen_intervals.append(self._steps_since_last_gen)
            self._z_subgoal = self._generate_subgoal(z_obs, text_hidden, text_mask, text_pooled)
            self._subgoal_gen_count += 1
            self._steps_since_last_gen = 0
            if self.progress_eval is not None:
                self.progress_eval.advance_subgoal()
            self._action_ensemble.reset()  # flush stale actions on subgoal change
            self._steps_since_plan = 0

        # Predict action chunk
        actions = self._generate_action_chunk(z_obs, self._z_subgoal)  # (H, 7)

        if self.ensemble_mode == "open_loop":
            # Cache the chunk, execute first action, queue the rest for replay
            self._cached_chunk = actions
            self._chunk_position = 1
            return self._binarize_gripper(actions[0])
        else:
            # uniform_mask or exp: insert into ensemble buffer, average column 0
            ensembled = self._action_ensemble.insert_and_ensemble(actions)  # (7,)
            return self._binarize_gripper(ensembled)

    @property
    def regen_stats(self) -> dict:
        """Return subgoal regeneration statistics accumulated over the eval."""
        intervals = self._regen_intervals
        return {
            "total_steps": self._total_steps,
            "subgoal_gen_count": self._subgoal_gen_count,
            "mean_regen_interval": float(np.mean(intervals)) if intervals else 0.0,
            "median_regen_interval": float(np.median(intervals)) if intervals else 0.0,
            "std_regen_interval": float(np.std(intervals)) if intervals else 0.0,
            "min_regen_interval": int(np.min(intervals)) if intervals else 0,
            "max_regen_interval": int(np.max(intervals)) if intervals else 0,
            "regen_intervals": intervals,
        }

    def reset_regen_stats(self):
        """Reset accumulated regen stats (call before a new eval run)."""
        self._total_steps = 0
        self._subgoal_gen_count = 0
        self._regen_intervals = []
        self._steps_since_last_gen = 0


def build_wrapper_from_config(
    flowdit_ckpt: str,
    gcbc_ckpt: str,
    config_path: str | None = None,
    device: str = "cuda",
    **overrides,
) -> FlowDiTCalvinWrapper:
    """Convenience factory that loads a DiT config and builds the wrapper.

    Args:
        flowdit_ckpt: Path to DiT checkpoint (.pt)
        gcbc_ckpt: Path to GCBC policy checkpoint (.pt)
        config_path: Optional YAML config (uses defaults from flowdit.yaml if None)
        device: CUDA device string
        **overrides: Override any wrapper kwarg

    Returns:
        Configured FlowDiTCalvinWrapper ready for evaluation
    """
    import yaml

    defaults = {
        "theia_path": "models/theia_small_cdiv",
        "text_model_path": "models/all-MiniLM-L6-v2",
        "act_stats_path": "/mnt/sda1/Datasets/chal2525/gcbc_data_abcd/act_stats.npz",
        "scale_factor": 1.702949,
        "flowdit_num_steps": 8,
        "context_cfg_scale": 2.5,
        "prompt_cfg_scale": 7.5,
        "delta_high": 0.90,
        "max_per_frame": 20,
        "min_per_frame": 1,
        "act_pred_horizon": 5,
        "distilled": False,
        "ensemble_mode": "uniform_mask",
        "exp_decay_k": 0.5,
        "open_loop_window": 5,
    }

    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        # Extract DiT model params (strip model_type which is not a ctor arg)
        if "model_params" in cfg:
            fp = {k: v for k, v in cfg["model_params"].items() if k != "model_type"}
            # Ensure text dims match the text encoder (MiniLM = 384)
            fp.setdefault("text_dim", 384)
            fp.setdefault("pooled_text_dim", 384)
            fp.setdefault("max_text_len", 25)
            defaults["flowdit_params"] = fp
        if "scale_factor" in cfg:
            defaults["scale_factor"] = cfg["scale_factor"]

    defaults.update(overrides)

    return FlowDiTCalvinWrapper(
        flowdit_ckpt=flowdit_ckpt,
        gcbc_ckpt=gcbc_ckpt,
        device=device,
        **defaults,
    )
