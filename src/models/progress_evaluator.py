"""
Progress Evaluator — Theia patch-mean cosine similarity.

Replaces LIV-based progress evaluation used in TaKSIE.
Computes cosine similarity between current observation and subgoal,
both encoded in Theia latent space (196 patches × 384 dim).

The comparison is 196 dot products + normalization + mean. Microseconds.
"""

import torch
import torch.nn.functional as F


class TheiaProgressEvaluator:
    """Evaluates progress toward a subgoal using Theia patch-mean cosine similarity.

    At each control step:
        similarity = mean(cos_sim(z_current[i], z_subgoal[i]) for i in 196 patches)

    Subgoal advancement when:
        similarity > delta_high  OR  step_count > max_steps

    Hysteresis mode (optional):
        Once similarity crosses delta_high, it must drop below delta_low to re-arm.
        This prevents oscillation near the threshold.

    EMA smoothing (optional):
        Exponential moving average on raw similarity to reduce frame-to-frame wobble.
    """

    def __init__(
        self,
        delta_high: float = 0.90,
        delta_low: float = None,
        max_steps: int = 20,
        min_steps: int = 1,
        ema_alpha: float = 0.0,
    ):
        """
        Args:
            delta_high: Advance to next subgoal when similarity exceeds this.
            delta_low: Hysteresis lower bound. If None, no hysteresis (delta_low = delta_high).
            max_steps: Force advance after this many steps regardless of similarity.
            min_steps: Minimum steps before allowing threshold-based advance.
            ema_alpha: EMA smoothing factor (0 = no smoothing, higher = more smoothing).
                       ema_t = alpha * ema_{t-1} + (1 - alpha) * raw_t
        """
        self.delta_high = delta_high
        self.delta_low = delta_low if delta_low is not None else delta_high
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.ema_alpha = ema_alpha

        self.reset()

    def reset(self):
        """Reset state for a new rollout / new subgoal."""
        self.step_count = 0
        self.ema_similarity = None
        self._armed = True  # hysteresis arm state

    def advance_subgoal(self):
        """Call when transitioning to a new subgoal (resets step counter + EMA)."""
        self.step_count = 0
        self.ema_similarity = None
        self._armed = True

    @staticmethod
    def patch_mean_cosine(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
        """Compute patch-mean cosine similarity between two Theia embeddings.

        Args:
            z_a: (196, 384) or (1, 196, 384) — current observation embedding
            z_b: (196, 384) or (1, 196, 384) — subgoal embedding

        Returns:
            Scalar cosine similarity (mean over 196 patches).
        """
        if z_a.dim() == 3:
            z_a = z_a.squeeze(0)
        if z_b.dim() == 3:
            z_b = z_b.squeeze(0)
        # Per-patch cosine: (196,)
        cos = F.cosine_similarity(z_a, z_b, dim=-1)
        return cos.mean().item()

    def should_advance(self, z_current: torch.Tensor, z_subgoal: torch.Tensor) -> tuple:
        """Check if we should advance to the next subgoal.

        Args:
            z_current: Current observation in Theia space (196, 384) or (1, 196, 384)
            z_subgoal: Subgoal target in Theia space (196, 384) or (1, 196, 384)

        Returns:
            (advance: bool, similarity: float, reason: str)
            reason is one of: "threshold", "max_steps", "continue"
        """
        self.step_count += 1

        raw_sim = self.patch_mean_cosine(z_current, z_subgoal)

        # EMA smoothing
        if self.ema_alpha > 0 and self.ema_similarity is not None:
            sim = self.ema_alpha * self.ema_similarity + (1 - self.ema_alpha) * raw_sim
        else:
            sim = raw_sim
        self.ema_similarity = sim

        # Max steps override (respects min_steps)
        if self.step_count >= self.max_steps and self.step_count >= self.min_steps:
            return True, sim, "max_steps"

        # Min steps gate
        if self.step_count < self.min_steps:
            return False, sim, "continue"

        # Hysteresis logic
        if self._armed and sim >= self.delta_high:
            self._armed = False  # disarm until re-armed
            return True, sim, "threshold"

        if not self._armed and sim < self.delta_low:
            self._armed = True  # re-arm

        return False, sim, "continue"
