"""
Flow Matching Goal-Conditioned Behavioral Cloning (FM-GCBC) Policy.

Architecture:
  - TheiaConvProjector: (196, 384) patches → 256-d vector (Theia paper Table 8)
  - ConditionedActionMLP: 3 residual MLP blocks, hidden_dim=256
  - Sinusoidal timestep embedding (dim=32)
  - Layer norm, dropout=0.1
  - Predicts act_pred_horizon future actions (default: 5)
  - 4-step Euler ODE sampling

Input: Theia patch features (196×384), reshaped to a 14×14 grid and compressed
to a 256-d vector via 3 strided conv layers (Theia paper Appendix D.2 readout).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Theia Feature Projection (spatial-aware)
# ============================================================================

class TheiaConvProjector(nn.Module):
    """Theia paper Table 8 (Appendix D.2): 3-conv stack, no pooling.

    (B, 196, 384) patches → (B, 14, 14, 384) grid → 14→7→3→1 with ReLU → (B, 256).
    """

    out_dim = 256

    def __init__(self, in_dim: int = 384, grid_size: int = 14):
        super().__init__()
        self.grid_size = grid_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=4, stride=2, padding=1),  # 14→7
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),                # 7→3
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),                # 3→1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 196, in_dim) → (B, 256)"""
        B = x.shape[0]
        h = x.transpose(1, 2).reshape(B, -1, self.grid_size, self.grid_size)
        h = self.conv(h)  # (B, 256, 1, 1)
        return h.flatten(1)


# ============================================================================
# Flow matching components
# ============================================================================

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) int timesteps → (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualMLPBlock(nn.Module):
    """Residual MLP block with optional layer norm and dropout."""

    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.1, use_layer_norm: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(dim + cond_dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, cond], dim=-1)
        h = self.drop(F.mish(self.norm1(self.fc1(h))))
        h = self.drop(F.mish(self.norm2(self.fc2(h))))
        return x + h


class CrossAttentionBlock(nn.Module):
    """Cross-attention from action hidden state to spatial obs/goal features.

    Query: (B, 1, dim) from action hidden state
    Key/Value: (B, N, kv_dim) from concatenated obs+goal Theia patches
    """

    def __init__(self, dim: int, kv_dim: int = 384, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(kv_dim, dim)
        self.v_proj = nn.Linear(kv_dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """x: (B, dim), kv: (B, N, kv_dim) → (B, dim)"""
        B, N, _ = kv.shape
        q = self.q_proj(self.norm_q(x)).unsqueeze(1)  # (B, 1, dim)
        kv_normed = self.norm_kv(kv)
        k = self.k_proj(kv_normed)  # (B, N, dim)
        v = self.v_proj(kv_normed)  # (B, N, dim)

        # Multi-head reshape
        q = q.reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, -1)  # (B, dim)
        return x + self.out_proj(out)


class ConditionedActionMLP(nn.Module):
    """Conditioned MLP for action-space flow matching.

    Predicts velocity v = x_1 - x_0.

    Inputs:
        - Interpolated action: (B, act_pred_horizon * act_dim)
        - Observation embedding: (B, obs_dim)
        - Goal embedding: (B, goal_dim)
        - Timestep: (B,)
        - (optional) Spatial KV: (B, N, kv_dim) for cross-attention
    """

    def __init__(
        self,
        action_dim: int = 7,
        act_pred_horizon: int = 4,
        obs_dim: int = 256,
        goal_dim: int = 256,
        time_dim: int = 32,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_cross_attention: bool = False,
        cross_attn_kv_dim: int = 384,
        cross_attn_heads: int = 8,
        cross_attn_every_n: int = 2,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.act_pred_horizon = act_pred_horizon
        self.use_cross_attention = use_cross_attention
        flat_action_dim = action_dim * act_pred_horizon

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(time_dim)

        # Condition projection: obs + goal + timestep → cond_dim
        cond_input_dim = obs_dim + goal_dim + time_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection: interpolated action → hidden
        self.input_proj = nn.Linear(flat_action_dim, hidden_dim)

        # Residual MLP blocks
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, hidden_dim, dropout, use_layer_norm)
            for _ in range(num_blocks)
        ])

        # Optional cross-attention blocks (every N-th block)
        self.attn_blocks = nn.ModuleList()
        if use_cross_attention:
            for i in range(num_blocks):
                if (i + 1) % cross_attn_every_n == 0:
                    self.attn_blocks.append(
                        CrossAttentionBlock(hidden_dim, cross_attn_kv_dim,
                                            cross_attn_heads, dropout))
                else:
                    self.attn_blocks.append(None)

        # Output: predict velocity
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(hidden_dim, flat_action_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        obs_emb: torch.Tensor,
        goal_emb: torch.Tensor,
        timestep: torch.Tensor,
        spatial_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_t: (B, act_pred_horizon * action_dim) interpolated action
            obs_emb: (B, obs_dim) pooled current observation
            goal_emb: (B, goal_dim) pooled goal
            timestep: (B,) flow matching timestep
            spatial_kv: (B, N, kv_dim) concatenated obs+goal spatial patches

        Returns:
            Predicted velocity: (B, act_pred_horizon * action_dim)
        """
        t_emb = self.time_embed(timestep)
        cond = self.cond_proj(torch.cat([obs_emb, goal_emb, t_emb], dim=-1))

        h = self.input_proj(x_t)
        for i, block in enumerate(self.blocks):
            h = block(h, cond)
            if self.use_cross_attention and self.attn_blocks[i] is not None:
                h = self.attn_blocks[i](h, spatial_kv)
        return self.output_proj(h)


# ============================================================================
# Flow Matching Policy
# ============================================================================

class FlowMatchingGCBCPolicy(nn.Module):
    """Flow Matching Goal-Conditioned Behavioral Cloning policy.

    Conditional flow matching (Lipman et al., 2023):
    - Training: x_t = (1-t)*x_0 + t*x_1, predict velocity v = x_1 - x_0
    - Inference: Euler ODE integration from x_0 ~ N(0,I) to x_1 in N steps

    Architecture: TheiaConvProjector + ConditionedActionMLP.
    The network predicts velocity v = x_1 - x_0.

    When use_cross_attention=True, the velocity net also cross-attends to
    raw spatial patches (obs + goal concatenated, 392 tokens of theia_dim).
    """

    def __init__(
        self,
        action_dim: int = 7,
        act_pred_horizon: int = 5,
        theia_dim: int = 384,
        num_sampling_steps: int = 4,
        time_dim: int = 32,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.1,
        use_cross_attention: bool = False,
        cross_attn_heads: int = 8,
        cross_attn_every_n: int = 2,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.act_pred_horizon = act_pred_horizon
        self.num_sampling_steps = num_sampling_steps
        self.use_cross_attention = use_cross_attention
        self.theia_dim = theia_dim

        # Theia paper Table 8 readout — shared for obs and goal
        self.projector = TheiaConvProjector(in_dim=theia_dim)
        obs_dim = self.projector.out_dim
        goal_dim = self.projector.out_dim

        # Velocity network
        self.velocity_net = ConditionedActionMLP(
            action_dim=action_dim,
            act_pred_horizon=act_pred_horizon,
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            use_cross_attention=use_cross_attention,
            cross_attn_kv_dim=theia_dim,
            cross_attn_heads=cross_attn_heads,
            cross_attn_every_n=cross_attn_every_n,
        )

    def _build_spatial_kv(self, z_obs: torch.Tensor, z_goal: torch.Tensor):
        """Concat obs + goal patches → (B, 392, theia_dim) for cross-attention."""
        if not self.use_cross_attention:
            return None
        return torch.cat([z_obs, z_goal], dim=1)  # (B, 392, 384)

    def compute_loss(
        self,
        z_obs: torch.Tensor,
        z_goal: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Conditional flow matching loss.

        OT path: x_t = (1 - t) * x_0 + t * x_1
        Target velocity: u_t = x_1 - x_0

        Args:
            z_obs:   (B, 196, 384) current observation in Theia space
            z_goal:  (B, 196, 384) goal in Theia space
            actions: (B, act_pred_horizon, action_dim) GT action sequence
            mask:    optional (B, act_pred_horizon) with 1 at valid timesteps,
                     0 at padded ones. When None, all timesteps are supervised
                     (back-compat with fixed-chunk training).

        Returns:
            MSE loss scalar (averaged over valid positions and action dims).
        """
        B, H, D = actions.shape
        device = actions.device

        obs_emb = self.projector(z_obs)    # (B, proj_dim)
        goal_emb = self.projector(z_goal)  # (B, proj_dim)
        spatial_kv = self._build_spatial_kv(z_obs, z_goal)

        # x_1 = data (flattened actions), x_0 ~ N(0,I)
        x_1 = actions.reshape(B, -1)  # (B, H*D)
        x_0 = torch.randn_like(x_1)

        # Sample t ~ U(0, 1)
        t = torch.rand(B, device=device)

        # Interpolate: x_t = (1 - t) * x_0 + t * x_1
        t_expand = t.unsqueeze(-1)  # (B, 1)
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1

        # Target velocity: u = x_1 - x_0
        u_target = x_1 - x_0

        t_input = (t * 1000.0).long().clamp(0, 999)
        v_pred = self.velocity_net(x_t, obs_emb, goal_emb, t_input, spatial_kv)

        if mask is None:
            return F.mse_loss(v_pred, u_target)

        # Masked MSE: per-position mean over (valid_positions × D)
        sqerr = (v_pred - u_target).pow(2).reshape(B, H, D)
        flat_mask = mask.to(sqerr.dtype).unsqueeze(-1)  # (B, H, 1) → broadcasts over D
        loss = (sqerr * flat_mask).sum() / (flat_mask.sum() * D).clamp(min=1.0)
        return loss

    @torch.no_grad()
    def sample_actions(
        self,
        z_obs: torch.Tensor,
        z_goal: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate actions via Euler ODE integration.

        Integrates from x_0 ~ N(0,I) to x_1 in num_steps Euler steps.

        Args:
            z_obs: (1, 196, 384) or (B, 196, 384) current observation
            z_goal: (1, 196, 384) or (B, 196, 384) goal
            num_steps: Override number of Euler steps (default: self.num_sampling_steps)

        Returns:
            (B, act_pred_horizon, action_dim) predicted actions
        """
        N = num_steps or self.num_sampling_steps
        B = z_obs.shape[0]
        device = z_obs.device

        obs_emb = self.projector(z_obs)
        goal_emb = self.projector(z_goal)
        spatial_kv = self._build_spatial_kv(z_obs, z_goal)

        flat_dim = self.act_pred_horizon * self.action_dim
        x = torch.randn(B, flat_dim, device=device)

        dt = 1.0 / N
        for i in range(N):
            t_val = i / N
            t_input = torch.full((B,), int(t_val * 1000), device=device, dtype=torch.long).clamp(0, 999)
            v = self.velocity_net(x, obs_emb, goal_emb, t_input, spatial_kv)
            x = x + v * dt

        return x.reshape(B, self.act_pred_horizon, self.action_dim)
