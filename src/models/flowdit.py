"""
Flow Matching DiT for Affordance Prediction

A Diffusion Transformer adapted for flow matching that predicts target affordance 
states from initial observations and text commands. Uses sequence concatenation 
for source image conditioning and CROSS-ATTENTION for text conditioning.

Architecture:
- Input: z_t (noisy target), z_init (source observation), text_emb (command)
- Conditioning: AdaLN-Zero with timestep only
- Text conditioning: Cross-attention in every block
- Source conditioning: Sequence concatenation (init tokens attend with target tokens)
- Output: Velocity field prediction for flow matching
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class TextProjector(nn.Module):
    """Projects per-token text hidden states to model dimension.
    
    Applies same projection independently to each token position.
    Input: [B, seq_len, text_dim] -> Output: [B, seq_len, hidden_size]
    """
    def __init__(self, text_dim: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        return self.proj(text_emb)


class DiTBlock(nn.Module):
    """
    DiT block with:
    - Self-attention (for init<->target interaction)
    - Cross-attention to text (for text conditioning)
    - MLP
    All modulated by timestep via AdaLN-Zero.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Cross-attention to text
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # MLP
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        
        # AdaLN modulation: 9 vectors for timestep-based modulation
        # (shift, scale, gate) x 3 for (self-attn, cross-attn, mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size),
        )
        
        # Zero-initialize cross-attention output projection for stable training start
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(
        self, 
        x: torch.Tensor, 
        t_emb: torch.Tensor, 
        text_ctx: torch.Tensor,
        text_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]
            t_emb: Timestep embedding [B, D]
            text_ctx: Text context for cross-attention [B, N_text, D]
            text_key_padding_mask: Bool mask [B, N_text], True = padding (ignored)
        Returns:
            Output tensor [B, L, D]
        """
        # Get all modulation parameters from timestep
        mod = self.adaLN_modulation(t_emb).chunk(9, dim=1)
        shift_sa, scale_sa, gate_sa = mod[0], mod[1], mod[2]
        shift_ca, scale_ca, gate_ca = mod[3], mod[4], mod[5]
        shift_mlp, scale_mlp, gate_mlp = mod[6], mod[7], mod[8]
        
        # Self-attention (init <-> target interaction preserved)
        x_norm = modulate(self.norm1(x), shift_sa, scale_sa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + gate_sa.unsqueeze(1) * attn_out
        
        # Cross-attention to text (text guides generation)
        x_norm = modulate(self.norm_cross(x), shift_ca, scale_ca)
        cross_out, _ = self.cross_attn(
            x_norm, text_ctx, text_ctx,
            key_padding_mask=text_key_padding_mask,
            need_weights=False,
        )
        x = x + gate_ca.unsqueeze(1) * cross_out
        
        # MLP
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class FinalLayer(nn.Module):
    """Final layer with adaptive layer norm and linear projection."""
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class AffordanceFlowDiT(nn.Module):
    """
    Flow Matching DiT for Affordance Prediction with Cross-Attention Text Conditioning.
    
    Args:
        latent_dim: Dimension of Theia latent features (384 for Theia-small)
        num_patches: Number of spatial patches (196 for 14x14)
        hidden_dim: Transformer hidden dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        text_dim: Dimension of text embeddings (768 for CLIP)
        max_text_len: Maximum text sequence length (77 for CLIP tokenizer)
        mlp_ratio: MLP hidden dim multiplier
        dropout: Dropout rate
        cond_drop_prob: Probability of dropping text conditioning for CFG training
        context_drop_prob: Probability of dropping context (z_init) conditioning for two-scale CFG
    """
    def __init__(
        self,
        latent_dim: int = 384,
        num_patches: int = 196,
        hidden_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        text_dim: int = 768,
        max_text_len: int = 25,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cond_drop_prob: float = 0.1,
        context_drop_prob: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.cond_drop_prob = cond_drop_prob
        self.context_drop_prob = context_drop_prob
        self.max_text_len = max_text_len
        
        # Input projections
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.init_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Positional embeddings
        self.pos_embed_init = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.pos_embed_target = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        
        # Timestep embedding (only timestep now, text via cross-attention)
        self.time_embed = TimestepEmbedder(hidden_dim)
        
        # Text projection for cross-attention context
        self.text_proj = TextProjector(text_dim, hidden_dim)
        
        # Null text embedding for CFG (text dropped)
        self.null_text_emb = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Null context embedding for two-scale CFG (context/z_init dropped)
        self.null_context_emb = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        
        # Transformer blocks with cross-attention
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Output
        self.final_layer = FinalLayer(hidden_dim, latent_dim)
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False
        
        # Initialize weights
        self._init_weights()
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"FlowDiT initialized with {n_params/1e6:.2f}M parameters")

    def _init_weights(self):
        """Initialize weights with DiT-specific initialization."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize positional embeddings
        nn.init.normal_(self.pos_embed_init, std=0.02)
        nn.init.normal_(self.pos_embed_target, std=0.02)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out final layer
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_init: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        drop_text: Optional[torch.Tensor] = None,
        drop_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass predicting velocity field.
        
        Args:
            z_t: Noisy target latents [B, N, D]
            t: Timesteps [B]
            z_init: Initial observation latents [B, N, D]
            text_emb: Per-token text hidden states [B, seq_len, text_dim]
            text_mask: Attention mask [B, seq_len], 1=real 0=padding
            drop_text: Optional bool mask for dropping text conditioning [B]
            drop_context: Optional bool mask for dropping context conditioning [B]
        
        Returns:
            Predicted velocity [B, N, D]
        """
        B = z_t.shape[0]
        
        # Project inputs
        h_target = self.input_proj(z_t) + self.pos_embed_target
        h_init = self.init_proj(z_init) + self.pos_embed_init
        
        # Two-scale CFG: independently drop context (z_init) during training
        if self.training and self.context_drop_prob > 0:
            if drop_context is None:
                drop_context = torch.rand(B, device=z_t.device) < self.context_drop_prob
            null_ctx_init = self.null_context_emb.expand(B, -1, -1)  # [B, N, D]
            h_init = torch.where(
                drop_context.view(B, 1, 1),
                null_ctx_init,
                h_init,
            )
        
        # Concatenate init and target for self-attention
        h = torch.cat([h_init, h_target], dim=1)  # [B, 2*N, D]
        
        # Timestep embedding (for AdaLN)
        t_emb = self.time_embed(t)  # [B, D]
        
        # Text context for cross-attention [B, seq_len, D]
        text_ctx = self.text_proj(text_emb)
        
        # Convert attention mask to key_padding_mask (True = padding, to ignore)
        if text_mask is not None:
            text_key_padding_mask = (text_mask == 0)  # [B, seq_len]
        else:
            text_key_padding_mask = None
        
        # Two-scale CFG: independently drop text during training
        if self.training and self.cond_drop_prob > 0:
            if drop_text is None:
                drop_text = torch.rand(B, device=z_t.device) < self.cond_drop_prob
            null_ctx = self.null_text_emb.expand(B, text_ctx.shape[1], -1)
            text_ctx = torch.where(
                drop_text.view(B, 1, 1),
                null_ctx,
                text_ctx
            )
            # For dropped samples, unmask all positions (attend to null tokens)
            if text_key_padding_mask is not None:
                text_key_padding_mask = torch.where(
                    drop_text.view(B, 1),
                    torch.zeros_like(text_key_padding_mask),  # False = attend
                    text_key_padding_mask
                )
        
        # Transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    block, h, t_emb, text_ctx, text_key_padding_mask,
                    use_reentrant=False
                )
            else:
                h = block(h, t_emb, text_ctx, text_key_padding_mask)
        
        # Extract target tokens
        h_target_out = h[:, self.num_patches:, :]
        
        # Final projection
        v = self.final_layer(h_target_out, t_emb)
        
        return v

    @torch.no_grad()
    def sample_euler(
        self,
        z_init: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        cfg_scale: float = 1.0,
        context_cfg_scale: Optional[float] = None,
        prompt_cfg_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample using Euler method with optional two-scale CFG.
        
        Two-scale CFG (when context_cfg_scale and prompt_cfg_scale are provided):
            v = v_uncond + context_w * (v_context - v_uncond) + prompt_w * (v_full - v_context)
        where v_uncond has both signals dropped, v_context has only text dropped,
        and v_full has both signals active.
        
        Falls back to single-scale CFG when only cfg_scale is provided.
        
        Args:
            z_init: Initial observation latents [B, N, D]
            text_emb: Per-token text hidden states [B, seq_len, text_dim]
            text_mask: Attention mask [B, seq_len], 1=real 0=padding
            num_steps: Number of Euler steps
            cfg_scale: Single-scale CFG weight (used when two-scale params are None)
            context_cfg_scale: Two-scale CFG weight for context (spatial fidelity)
            prompt_cfg_scale: Two-scale CFG weight for text prompt (instruction following)
        
        Returns:
            Sampled target latents [B, N, D]
        """
        B = z_init.shape[0]
        device = z_init.device
        use_two_scale = context_cfg_scale is not None and prompt_cfg_scale is not None
        
        # Start from noise
        z = torch.randn_like(z_init)
        
        # Prepare text context
        text_ctx = self.text_proj(text_emb)  # [B, seq_len, D]
        if text_mask is not None:
            text_kpm = (text_mask == 0)  # key_padding_mask: True = padding
        else:
            text_kpm = None
        
        # Prepare null text context for CFG
        null_text_ctx = self.null_text_emb.expand(B, text_ctx.shape[1], -1)
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.ones(B, device=device) * (1.0 - i * dt)
            
            if use_two_scale:
                # Three-pass two-scale CFG
                # 1) Fully unconditional: both context and text dropped
                v_uncond = self._forward_with_ctx(z, t, z_init, null_text_ctx, None, use_null_context=True)
                # 2) Context only: real z_init, text dropped
                v_context = self._forward_with_ctx(z, t, z_init, null_text_ctx, None, use_null_context=False)
                # 3) Fully conditioned: real z_init, real text
                v_full = self._forward_with_ctx(z, t, z_init, text_ctx, text_kpm, use_null_context=False)
                # Two-scale guided velocity
                v = v_uncond + context_cfg_scale * (v_context - v_uncond) + prompt_cfg_scale * (v_full - v_context)
            elif cfg_scale != 1.0:
                # Single-scale CFG (text only, legacy behavior)
                v_cond = self._forward_with_ctx(z, t, z_init, text_ctx, text_kpm)
                v_uncond = self._forward_with_ctx(z, t, z_init, null_text_ctx, None)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self._forward_with_ctx(z, t, z_init, text_ctx, text_kpm)
            
            z = z - v * dt
        
        return z

    def _forward_with_ctx(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_init: torch.Tensor,
        text_ctx: torch.Tensor,
        text_key_padding_mask: Optional[torch.Tensor] = None,
        use_null_context: bool = False,
    ) -> torch.Tensor:
        """Forward pass with pre-computed text context.
        
        Args:
            z_t: Noisy target latents [B, N, D]
            t: Timesteps [B]
            z_init: Initial observation latents [B, N, D]
            text_ctx: Pre-projected text context [B, seq_len, hidden_dim]
            text_key_padding_mask: Bool mask [B, seq_len], True = padding
            use_null_context: If True, replace context tokens with learned null embedding
        """
        B = z_t.shape[0]
        
        h_target = self.input_proj(z_t) + self.pos_embed_target
        if use_null_context:
            h_init = self.null_context_emb.expand(B, -1, -1)
        else:
            h_init = self.init_proj(z_init) + self.pos_embed_init
        h = torch.cat([h_init, h_target], dim=1)
        
        t_emb = self.time_embed(t)
        
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    block, h, t_emb, text_ctx, text_key_padding_mask,
                    use_reentrant=False
                )
            else:
                h = block(h, t_emb, text_ctx, text_key_padding_mask)
        
        h_target_out = h[:, self.num_patches:, :]
        v = self.final_layer(h_target_out, t_emb)
        
        return v

    @torch.no_grad()
    def sample_adaptive(
        self,
        z_init: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        context_cfg_scale: Optional[float] = None,
        prompt_cfg_scale: Optional[float] = None,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> torch.Tensor:
        """Sample using adaptive ODE solver (dopri5) with optional two-scale CFG."""
        try:
            from torchdiffeq import odeint
        except ImportError:
            raise ImportError("torchdiffeq required. Install with: pip install torchdiffeq")
        
        B = z_init.shape[0]
        device = z_init.device
        use_two_scale = context_cfg_scale is not None and prompt_cfg_scale is not None
        
        text_ctx = self.text_proj(text_emb)  # [B, seq_len, D]
        if text_mask is not None:
            text_kpm = (text_mask == 0)
        else:
            text_kpm = None
        null_text_ctx = self.null_text_emb.expand(B, text_ctx.shape[1], -1)
        
        def ode_fn(t_scalar, z):
            t = torch.ones(B, device=device) * t_scalar
            
            if use_two_scale:
                v_uncond = self._forward_with_ctx(z, t, z_init, null_text_ctx, None, use_null_context=True)
                v_context = self._forward_with_ctx(z, t, z_init, null_text_ctx, None, use_null_context=False)
                v_full = self._forward_with_ctx(z, t, z_init, text_ctx, text_kpm, use_null_context=False)
                v = v_uncond + context_cfg_scale * (v_context - v_uncond) + prompt_cfg_scale * (v_full - v_context)
            elif cfg_scale != 1.0:
                v_cond = self._forward_with_ctx(z, t, z_init, text_ctx, text_kpm)
                v_uncond = self._forward_with_ctx(z, t, z_init, null_text_ctx, None)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self._forward_with_ctx(z, t, z_init, text_ctx, text_kpm)
            
            return v
        
        z0 = torch.randn_like(z_init)
        t_span = torch.tensor([1.0, 0.0], device=device)
        solution = odeint(ode_fn, z0, t_span, atol=atol, rtol=rtol, method='dopri5')
        
        return solution[1]

    def generate_fixed_steps(
        self,
        z_init: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        num_steps: int = 4,
        cfg_scale: float = 1.0,
        context_cfg_scale: Optional[float] = None,
        prompt_cfg_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Fixed-step Euler generation WITH gradient tracking.
        Used during adversarial training so discriminator gradients
        flow back through the generation process to update FlowDiT.

        Unlike sample_euler, this has NO @torch.no_grad() decorator.
        Supports two-scale CFG when context_cfg_scale and prompt_cfg_scale are provided.

        Args:
            z_init: Initial observation latents [B, N, D] (already scaled)
            text_emb: Per-token text hidden states [B, seq_len, text_dim]
            text_mask: Attention mask [B, seq_len], 1=real 0=padding
            num_steps: Number of fixed Euler steps (default 4)
            cfg_scale: Single-scale CFG weight (used when two-scale params are None)
            context_cfg_scale: Two-scale CFG weight for context (spatial fidelity)
            prompt_cfg_scale: Two-scale CFG weight for text prompt (instruction following)

        Returns:
            Generated goal latents [B, N, D] with full gradient graph
        """
        B = z_init.shape[0]
        device = z_init.device
        use_two_scale = context_cfg_scale is not None and prompt_cfg_scale is not None

        # Start from noise
        z = torch.randn_like(z_init)

        # Prepare text context (with gradients — text_proj is part of FlowDiT)
        text_ctx = self.text_proj(text_emb)
        if text_mask is not None:
            text_kpm = (text_mask == 0)
        else:
            text_kpm = None

        null_text_ctx = self.null_text_emb.expand(B, text_ctx.shape[1], -1)

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.ones(B, device=device) * (1.0 - i * dt)

            if use_two_scale:
                v_uncond = self._forward_with_ctx(z, t, z_init, null_text_ctx, None, use_null_context=True)
                v_context = self._forward_with_ctx(z, t, z_init, null_text_ctx, None, use_null_context=False)
                v_full = self._forward_with_ctx(z, t, z_init, text_ctx, text_kpm, use_null_context=False)
                v = v_uncond + context_cfg_scale * (v_context - v_uncond) + prompt_cfg_scale * (v_full - v_context)
            elif cfg_scale != 1.0:
                v_cond = self._forward_with_ctx(z, t, z_init, text_ctx, text_kpm)
                v_uncond = self._forward_with_ctx(z, t, z_init, null_text_ctx, None)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self._forward_with_ctx(z, t, z_init, text_ctx, text_kpm)

            z = z - v * dt

        return z


def flow_matching_loss(
    model: AffordanceFlowDiT,
    z_init: torch.Tensor,
    z_target: torch.Tensor,
    text_emb: torch.Tensor,
    text_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Compute flow matching loss.
    
    Args:
        model: AffordanceFlowDiT model
        z_init: Initial observation latents [B, N, D]
        z_target: Target observation latents [B, N, D]
        text_emb: Per-token text hidden states [B, seq_len, text_dim]
        text_mask: Attention mask [B, seq_len], 1=real 0=padding
        eps: Small constant for numerical stability
    
    Returns:
        MSE loss between predicted and target velocity
    """
    B = z_target.shape[0]
    device = z_target.device
    
    # Sample timesteps
    t = torch.rand(B, device=device)
    
    # Sample noise
    z_noise = torch.randn_like(z_target)
    
    # Interpolate
    t_exp = t.view(B, 1, 1)
    z_t = (1 - t_exp) * z_target + (eps + (1 - eps) * t_exp) * z_noise
    
    # Target velocity
    target_v = (1 - eps) * z_noise - z_target
    
    # Predict velocity
    v_pred = model(z_t, t, z_init, text_emb, text_mask=text_mask)
    
    # MSE loss
    loss = F.mse_loss(v_pred, target_v)
    
    return loss