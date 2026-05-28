"""
Flow Matching DiT

A Diffusion Transformer adapted for flow matching that predicts target
states from initial observations and text commands. The canonical model
used in this repo is `DiTAir`, which feeds both source and text tokens
into joint self-attention. Other variants are implemented for reference
or comparison: `DiT` (legacy cross-attention to text via `DiTBlock`),
`UViT` (U-Net-style skip connections), `MMDiT` (per-stream attention).

Architecture:
- Input: z_t (noisy target), z_init (source observation), text_emb (command)
- Conditioning: AdaLN-Zero with timestep (+ pooled text)
- AdaLN modulation: single-stream (shared AdaLN MLP reused for both streams; see DiTAir)
- Attention: joint self-attention over [init, target, text] tokens in every block
- Source conditioning: Sequence concatenation (init tokens attend with target tokens)
- Text conditioning: Sequence concatenation (text tokens joined in joint self-attention)
- Output: Velocity field prediction for flow matching
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


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
        

    def forward(
        self, 
        x: torch.Tensor, 
        t_emb: torch.Tensor, 
        text_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]
            t_emb: Timestep embedding [B, D]
            text_ctx: Text context for cross-attention [B, N_text, D]
                      (padding positions must be zero-masked before calling)
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
        
        # Cross-attention to text — no mask needed (padding zeroed at top level,
        # softmax naturally ignores zero-norm keys). Enables Flash Attention.
        x_norm = modulate(self.norm_cross(x), shift_ca, scale_ca)
        cross_out, _ = self.cross_attn(
            x_norm, text_ctx, text_ctx,
            need_weights=False,
        )
        x = x + gate_ca.unsqueeze(1) * cross_out
        
        # MLP
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class UViTBlock(nn.Module):
    """
    U-ViT block (Bao et al., NeurIPS 2023): shared QKV self-attention with
    text tokens concatenated into the sequence.  Text tokens are READ-ONLY —
    after attention, only image outputs are used; text is re-projected fresh
    from the frozen encoder output at each layer.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x, conditioning, text_ctx):
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(conditioning).chunk(6, dim=1)
        N_img = x.shape[1]
        x_norm = modulate(self.norm1(x), shift_sa, scale_sa)
        x_full = torch.cat([x_norm, text_ctx], dim=1)
        attn_out, _ = self.attn(x_full, x_full, x_full, need_weights=False)
        x = x + gate_sa.unsqueeze(1) * attn_out[:, :N_img, :]
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm2)
        return x


class DiTAirBlock(nn.Module):
    """
    DiT-Air block (Chen et al., 2025).

    Image and text tokens share QKV attention and MLP weights, but each stream
    is AdaLN-modulated by its own shift/scale/gate vector. Text tokens evolve
    through the network via residual connections.

    Forward: (x_img, x_txt, mod_img, mod_txt) → (x_img, x_txt)
    where mod_img and mod_txt are [B, 6*D] tensors encoding
    (shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp) per stream.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x_img, x_txt, mod_img, mod_txt):
        """
        x_img: [B, N_img, D], x_txt: [B, N_txt, D]
        mod_img, mod_txt: [B, 6*D] — shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp
        """
        shift_sa_i, scale_sa_i, gate_sa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i = mod_img.chunk(6, dim=1)
        shift_sa_t, scale_sa_t, gate_sa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = mod_txt.chunk(6, dim=1)
        N_img = x_img.shape[1]

        # Self-attention: normalize all tokens, modulate per stream, attend jointly
        x_all = torch.cat([x_img, x_txt], dim=1)
        x_norm = self.norm1(x_all)
        x_norm_all = torch.cat([
            modulate(x_norm[:, :N_img], shift_sa_i, scale_sa_i),
            modulate(x_norm[:, N_img:], shift_sa_t, scale_sa_t),
        ], dim=1)
        attn_out, _ = self.attn(x_norm_all, x_norm_all, x_norm_all, need_weights=False)
        x_img = x_img + gate_sa_i.unsqueeze(1) * attn_out[:, :N_img]
        x_txt = x_txt + gate_sa_t.unsqueeze(1) * attn_out[:, N_img:]

        # MLP: normalize all tokens, modulate per stream, shared MLP forward
        x_all = torch.cat([x_img, x_txt], dim=1)
        x_norm2 = self.norm2(x_all)
        x_norm2_all = torch.cat([
            modulate(x_norm2[:, :N_img], shift_mlp_i, scale_mlp_i),
            modulate(x_norm2[:, N_img:], shift_mlp_t, scale_mlp_t),
        ], dim=1)
        mlp_out = self.mlp(x_norm2_all)
        x_img = x_img + gate_mlp_i.unsqueeze(1) * mlp_out[:, :N_img]
        x_txt = x_txt + gate_mlp_t.unsqueeze(1) * mlp_out[:, N_img:]

        return x_img, x_txt


class MMDiTBlock(nn.Module):
    """
    Full MMDiT block (Esser et al., 2024 / FLUX): fully separate QKV, output
    projection, MLP, and AdaLN per modality.  Text tokens EVOLVE through the
    network via their own residual connections.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        mlp_hidden = int(hidden_size * mlp_ratio)
        # Image stream
        self.norm1_img = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv_img = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj_img = nn.Linear(hidden_size, hidden_size)
        self.norm2_img = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_img = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden), nn.GELU(approximate='tanh'),
            nn.Dropout(dropout), nn.Linear(mlp_hidden, hidden_size), nn.Dropout(dropout),
        )
        self.adaLN_img = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        # Text stream
        self.norm1_txt = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv_txt = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj_txt = nn.Linear(hidden_size, hidden_size)
        self.norm2_txt = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_txt = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden), nn.GELU(approximate='tanh'),
            nn.Dropout(dropout), nn.Linear(mlp_hidden, hidden_size), nn.Dropout(dropout),
        )
        self.adaLN_txt = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x_img, x_txt, conditioning):
        B, N_img, D = x_img.shape
        s_sa_i, sc_sa_i, g_sa_i, s_mlp_i, sc_mlp_i, g_mlp_i = \
            self.adaLN_img(conditioning).chunk(6, dim=-1)
        s_sa_t, sc_sa_t, g_sa_t, s_mlp_t, sc_mlp_t, g_mlp_t = \
            self.adaLN_txt(conditioning).chunk(6, dim=-1)
        q_img, k_img, v_img = self.qkv_img(
            modulate(self.norm1_img(x_img), s_sa_i, sc_sa_i)).chunk(3, dim=-1)
        q_txt, k_txt, v_txt = self.qkv_txt(
            modulate(self.norm1_txt(x_txt), s_sa_t, sc_sa_t)).chunk(3, dim=-1)
        Q = torch.cat([q_img, q_txt], dim=1)
        K = torch.cat([k_img, k_txt], dim=1)
        V = torch.cat([v_img, v_txt], dim=1)
        attn_out = F.scaled_dot_product_attention(
            Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2),
            K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2),
            V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2),
        ).transpose(1, 2).reshape(B, -1, D)
        x_img = x_img + g_sa_i.unsqueeze(1) * self.out_proj_img(attn_out[:, :N_img, :])
        x_txt = x_txt + g_sa_t.unsqueeze(1) * self.out_proj_txt(attn_out[:, N_img:, :])
        x_img = x_img + g_mlp_i.unsqueeze(1) * self.mlp_img(
            modulate(self.norm2_img(x_img), s_mlp_i, sc_mlp_i))
        x_txt = x_txt + g_mlp_t.unsqueeze(1) * self.mlp_txt(
            modulate(self.norm2_txt(x_txt), s_mlp_t, sc_mlp_t))
        return x_img, x_txt


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


class DiT(nn.Module):
    """
    Flow Matching DiT with Cross-Attention Text Conditioning.
    
    Args:
        latent_dim: Dimension of Theia latent features (384 for Theia-small)
        num_patches: Number of spatial patches (196 for 14x14)
        hidden_dim: Transformer hidden dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        text_dim: Dimension of per-token text embeddings (512 for TinyCLIP-29M)
        pooled_text_dim: Dimension of pooled text embeddings for AdaLN conditioning
        max_text_len: Maximum text sequence length (77 for CLIP tokenizer)
        mlp_ratio: MLP hidden dim multiplier
        dropout: Dropout rate
        cfg_drop_prompt: Probability of dropping only text/prompt conditioning
        cfg_drop_context: Probability of dropping only context (z_init) conditioning
        cfg_drop_both: Probability of dropping both prompt and context
        block_cls: DiT block class to use (default: DiTBlock with cross-attention)
    """
    _model_name = "DiT"

    def __init__(
        self,
        latent_dim: int = 384,
        num_patches: int = 196,
        hidden_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        text_dim: int = 768,
        pooled_text_dim: int = 512,
        max_text_len: int = 25,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cfg_drop_prompt: float = 0.05,
        cfg_drop_context: float = 0.05,
        cfg_drop_both: float = 0.05,
        use_pooled_text: bool = True,
        block_cls: type = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.cfg_drop_prompt = cfg_drop_prompt
        self.cfg_drop_context = cfg_drop_context
        self.cfg_drop_both = cfg_drop_both
        self.max_text_len = max_text_len
        self.pooled_text_dim = pooled_text_dim
        self.use_pooled_text = use_pooled_text
        
        # Input projections
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.init_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Positional embeddings
        self.pos_embed_init = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.pos_embed_target = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        
        # Timestep embedding
        self.time_embed = TimestepEmbedder(hidden_dim)
        
        # Pooled text projection for AdaLN conditioning (only when use_pooled_text=True)
        if self.use_pooled_text:
            self.pooled_text_proj = nn.Sequential(
                nn.Linear(pooled_text_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.null_pooled_text = nn.Parameter(torch.zeros(1, hidden_dim))
        
        # Text projection for cross-attention context
        self.text_proj = TextProjector(text_dim, hidden_dim)
        
        # Null text embedding for CFG (text dropped)
        self.null_text_emb = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Null context embedding for two-scale CFG (context/z_init dropped)
        self.null_context_emb = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        
        # Transformer blocks
        _block_cls = block_cls or DiTBlock
        self.blocks = nn.ModuleList([
            _block_cls(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Output
        self.final_layer = FinalLayer(hidden_dim, latent_dim)
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False
        
        # Initialize weights
        self._init_weights()
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"{self._model_name} initialized with {n_params/1e6:.2f}M parameters")

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
            for name in ('adaLN_modulation', 'adaLN_img', 'adaLN_txt'):
                mod = getattr(block, name, None)
                if mod is not None:
                    nn.init.constant_(mod[-1].weight, 0)
                    nn.init.constant_(mod[-1].bias, 0)
        
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
        pooled_text_emb: torch.Tensor = None,
        drop_text: Optional[torch.Tensor] = None,
        drop_context: Optional[torch.Tensor] = None,
        progress_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass predicting velocity field.
        
        Args:
            z_t: Noisy target latents [B, N, D]
            t: Timesteps [B]
            z_init: Initial observation latents [B, N, D]
            text_emb: Per-token text hidden states [B, seq_len, text_dim]
            text_mask: Attention mask [B, seq_len], 1=real 0=padding
            pooled_text_emb: Pooled text embedding [B, pooled_text_dim] for AdaLN
            drop_text: Optional bool mask for dropping text conditioning [B]
            drop_context: Optional bool mask for dropping context conditioning [B]
            progress_emb: Optional progress embedding [B, progress_dim] from GRU encoder
        
        Returns:
            Predicted velocity [B, N, D]
        """
        B = z_t.shape[0]
        
        # Categorical CFG dropout: sample which signals to drop per sample
        cfg_total = self.cfg_drop_prompt + self.cfg_drop_context + self.cfg_drop_both
        if self.training and cfg_total > 0:
            if drop_text is None or drop_context is None:
                # Categories: 0=keep both, 1=drop prompt only, 2=drop context only, 3=drop both
                probs = torch.tensor([
                    1.0 - cfg_total,        # keep both
                    self.cfg_drop_prompt,    # drop prompt only
                    self.cfg_drop_context,   # drop context only
                    self.cfg_drop_both,      # drop both
                ], device=z_t.device)
                categories = torch.multinomial(probs.expand(B, -1), 1).squeeze(1)  # [B]
                if drop_text is None:
                    drop_text = (categories == 1) | (categories == 3)
                if drop_context is None:
                    drop_context = (categories == 2) | (categories == 3)
        
        # Project inputs
        h_target = self.input_proj(z_t) + self.pos_embed_target
        h_init = self.init_proj(z_init) + self.pos_embed_init
        
        # Two-scale CFG: drop context (z_init) during training
        if self.training and drop_context is not None and drop_context.any():
            null_ctx_init = self.null_context_emb.expand(B, -1, -1)  # [B, N, D]
            h_init = torch.where(
                drop_context.view(B, 1, 1),
                null_ctx_init,
                h_init,
            )
        
        # Concatenate init and target for self-attention
        h = torch.cat([h_init, h_target], dim=1)  # [B, 2*N, D]
        
        # AdaLN conditioning = timestep + pooled text (+ optional progress)
        t_emb = self.time_embed(t)  # [B, D]
        if self.use_pooled_text:
            pooled_proj = self.pooled_text_proj(pooled_text_emb)  # [B, D]
            # CFG dropout: replace pooled text with null embedding when text is dropped
            if self.training and drop_text is not None and drop_text.any():
                pooled_proj = torch.where(
                    drop_text.view(B, 1),
                    self.null_pooled_text.expand(B, -1),
                    pooled_proj,
                )
            conditioning = t_emb + pooled_proj
        else:
            conditioning = t_emb
        
        # Add progress embedding if provided (zero-init proj ensures no-op at start)
        if progress_emb is not None:
            conditioning = self._inject_progress(conditioning, progress_emb)
        
        # Text context for cross-attention [B, seq_len, D]
        text_ctx = self.text_proj(text_emb)
        
        # Zero-mask padding positions so blocks need no mask arg.
        # Enables Flash Attention dispatch (no attn_mask / key_padding_mask).
        if text_mask is not None:
            text_ctx = text_ctx * text_mask.unsqueeze(-1)  # [B, seq_len, D]
        
        # Two-scale CFG: drop text during training (cross-attention tokens)
        if self.training and drop_text is not None and drop_text.any():
            null_ctx = self.null_text_emb.expand(B, text_ctx.shape[1], -1)
            text_ctx = torch.where(
                drop_text.view(B, 1, 1),
                null_ctx,
                text_ctx
            )
        
        # Transformer blocks (conditioning = t_emb + pooled_text goes to AdaLN)
        h = self._run_blocks(h, conditioning, text_ctx)
        
        # Extract target tokens
        h_target_out = h[:, self.num_patches:, :]
        
        # Final projection
        v = self.final_layer(h_target_out, conditioning)
        
        # DDP anchor: ensure null embeddings always participate in the gradient
        # graph even when CFG dropout doesn't activate for this batch.
        # Required for static_graph=True with stochastic dropout.
        if self.training:
            anchor = self.null_text_emb.sum() + self.null_context_emb.sum()
            if self.use_pooled_text:
                anchor = anchor + self.null_pooled_text.sum()
            v = v + 0.0 * anchor
        
        return v

    @torch.no_grad()
    def sample_euler(
        self,
        z_init: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        pooled_text_emb: torch.Tensor = None,
        num_steps: int = 50,
        cfg_scale: float = 1.0,
        context_cfg_scale: Optional[float] = None,
        prompt_cfg_scale: Optional[float] = None,
        progress_emb: Optional[torch.Tensor] = None,
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
            pooled_text_emb: Pooled text embedding [B, pooled_text_dim] for AdaLN
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
        
        # Prepare text context — zero-mask padding for Flash Attention
        text_ctx = self.text_proj(text_emb)  # [B, seq_len, D]
        if text_mask is not None:
            text_ctx = text_ctx * text_mask.unsqueeze(-1)
        
        # Prepare pooled text conditioning for AdaLN
        pooled_cond = self._get_pooled_conditioning(pooled_text_emb, use_null=False)
        null_pooled_cond = self._get_pooled_conditioning(pooled_text_emb, use_null=True)
        
        # Prepare null text context for CFG
        null_text_ctx = self.null_text_emb.expand(B, text_ctx.shape[1], -1)
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.ones(B, device=device) * (1.0 - i * dt)
            
            if use_two_scale:
                # Three-pass two-scale CFG
                v_uncond = self._forward_with_ctx(z, t, z_init, null_text_ctx, pooled_cond=null_pooled_cond, use_null_context=True, progress_emb=progress_emb)
                v_context = self._forward_with_ctx(z, t, z_init, null_text_ctx, pooled_cond=null_pooled_cond, use_null_context=False, progress_emb=progress_emb)
                v_full = self._forward_with_ctx(z, t, z_init, text_ctx, pooled_cond=pooled_cond, use_null_context=False, progress_emb=progress_emb)
                v = v_uncond + context_cfg_scale * (v_context - v_uncond) + prompt_cfg_scale * (v_full - v_context)
            elif cfg_scale != 1.0:
                v_cond = self._forward_with_ctx(z, t, z_init, text_ctx, pooled_cond=pooled_cond, progress_emb=progress_emb)
                v_uncond = self._forward_with_ctx(z, t, z_init, null_text_ctx, pooled_cond=null_pooled_cond, progress_emb=progress_emb)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self._forward_with_ctx(z, t, z_init, text_ctx, pooled_cond=pooled_cond, progress_emb=progress_emb)
            
            z = z - v * dt
        
        return z

    def _inject_progress(self, conditioning, progress_emb):
        """Inject progress embedding into conditioning. Override in subclasses."""
        if hasattr(self, 'progress_proj'):
            return conditioning + self.progress_proj(progress_emb)
        return conditioning

    def _run_blocks(self, h, conditioning, text_ctx):
        """Run transformer blocks. Override in subclasses for different block interfaces."""
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    block, h, conditioning, text_ctx,
                    use_reentrant=False
                )
            else:
                h = block(h, conditioning, text_ctx)
        return h

    def _get_pooled_conditioning(self, pooled_text_emb: torch.Tensor, use_null: bool = False) -> torch.Tensor:
        """Project pooled text embedding for AdaLN, or return null embedding."""
        if not self.use_pooled_text:
            B = pooled_text_emb.shape[0]
            return torch.zeros(B, self.hidden_dim, device=pooled_text_emb.device)
        if use_null:
            B = pooled_text_emb.shape[0]
            return self.null_pooled_text.expand(B, -1)
        return self.pooled_text_proj(pooled_text_emb)  # [B, hidden_dim]

    def _forward_with_ctx(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_init: torch.Tensor,
        text_ctx: torch.Tensor,
        pooled_cond: Optional[torch.Tensor] = None,
        use_null_context: bool = False,
        progress_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-computed text context.
        
        Args:
            z_t: Noisy target latents [B, N, D]
            t: Timesteps [B]
            z_init: Initial observation latents [B, N, D]
            text_ctx: Pre-projected text context [B, seq_len, hidden_dim]
                      (padding positions must be zero-masked before calling)
            pooled_cond: Pre-projected pooled text [B, hidden_dim] for AdaLN
            use_null_context: If True, replace context tokens with learned null embedding
            progress_emb: Optional progress embedding [B, progress_dim] from GRU encoder
        """
        B = z_t.shape[0]
        
        h_target = self.input_proj(z_t) + self.pos_embed_target
        if use_null_context:
            h_init = self.null_context_emb.expand(B, -1, -1)
        else:
            h_init = self.init_proj(z_init) + self.pos_embed_init
        h = torch.cat([h_init, h_target], dim=1)
        
        t_emb = self.time_embed(t)
        conditioning = t_emb + pooled_cond
        
        # Add progress embedding if provided
        if progress_emb is not None:
            conditioning = self._inject_progress(conditioning, progress_emb)
        
        h = self._run_blocks(h, conditioning, text_ctx)
        
        h_target_out = h[:, self.num_patches:, :]
        v = self.final_layer(h_target_out, conditioning)
        
        return v

    @torch.no_grad()
    def sample_adaptive(
        self,
        z_init: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        pooled_text_emb: torch.Tensor = None,
        cfg_scale: float = 1.0,
        context_cfg_scale: Optional[float] = None,
        prompt_cfg_scale: Optional[float] = None,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        progress_emb: Optional[torch.Tensor] = None,
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
            text_ctx = text_ctx * text_mask.unsqueeze(-1)
        null_text_ctx = self.null_text_emb.expand(B, text_ctx.shape[1], -1)
        
        # Prepare pooled text conditioning for AdaLN
        pooled_cond = self._get_pooled_conditioning(pooled_text_emb, use_null=False)
        null_pooled_cond = self._get_pooled_conditioning(pooled_text_emb, use_null=True)
        
        def ode_fn(t_scalar, z):
            t = torch.ones(B, device=device) * t_scalar
            
            if use_two_scale:
                v_uncond = self._forward_with_ctx(z, t, z_init, null_text_ctx, pooled_cond=null_pooled_cond, use_null_context=True, progress_emb=progress_emb)
                v_context = self._forward_with_ctx(z, t, z_init, null_text_ctx, pooled_cond=null_pooled_cond, use_null_context=False, progress_emb=progress_emb)
                v_full = self._forward_with_ctx(z, t, z_init, text_ctx, pooled_cond=pooled_cond, use_null_context=False, progress_emb=progress_emb)
                v = v_uncond + context_cfg_scale * (v_context - v_uncond) + prompt_cfg_scale * (v_full - v_context)
            elif cfg_scale != 1.0:
                v_cond = self._forward_with_ctx(z, t, z_init, text_ctx, pooled_cond=pooled_cond, progress_emb=progress_emb)
                v_uncond = self._forward_with_ctx(z, t, z_init, null_text_ctx, pooled_cond=null_pooled_cond, progress_emb=progress_emb)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self._forward_with_ctx(z, t, z_init, text_ctx, pooled_cond=pooled_cond, progress_emb=progress_emb)
            
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
        pooled_text_emb: torch.Tensor = None,
        num_steps: int = 4,
        cfg_scale: float = 1.0,
        context_cfg_scale: Optional[float] = None,
        prompt_cfg_scale: Optional[float] = None,
        progress_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fixed-step Euler generation WITH gradient tracking.
        Used during adversarial training so discriminator gradients
        flow back through the generation process to update DiT.

        Unlike sample_euler, this has NO @torch.no_grad() decorator.
        Supports two-scale CFG when context_cfg_scale and prompt_cfg_scale are provided.

        Args:
            z_init: Initial observation latents [B, N, D] (already scaled)
            text_emb: Per-token text hidden states [B, seq_len, text_dim]
            text_mask: Attention mask [B, seq_len], 1=real 0=padding
            pooled_text_emb: Pooled text embedding [B, pooled_text_dim] for AdaLN
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

        # Prepare text context (with gradients — text_proj is part of DiT)
        text_ctx = self.text_proj(text_emb)
        if text_mask is not None:
            text_ctx = text_ctx * text_mask.unsqueeze(-1)

        null_text_ctx = self.null_text_emb.expand(B, text_ctx.shape[1], -1)

        # Prepare pooled text conditioning for AdaLN
        pooled_cond = self._get_pooled_conditioning(pooled_text_emb, use_null=False)
        null_pooled_cond = self._get_pooled_conditioning(pooled_text_emb, use_null=True)

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.ones(B, device=device) * (1.0 - i * dt)

            if use_two_scale:
                v_uncond = self._forward_with_ctx(z, t, z_init, null_text_ctx, pooled_cond=null_pooled_cond, use_null_context=True, progress_emb=progress_emb)
                v_context = self._forward_with_ctx(z, t, z_init, null_text_ctx, pooled_cond=null_pooled_cond, use_null_context=False, progress_emb=progress_emb)
                v_full = self._forward_with_ctx(z, t, z_init, text_ctx, pooled_cond=pooled_cond, use_null_context=False, progress_emb=progress_emb)
                v = v_uncond + context_cfg_scale * (v_context - v_uncond) + prompt_cfg_scale * (v_full - v_context)
            elif cfg_scale != 1.0:
                v_cond = self._forward_with_ctx(z, t, z_init, text_ctx, pooled_cond=pooled_cond, progress_emb=progress_emb)
                v_uncond = self._forward_with_ctx(z, t, z_init, null_text_ctx, pooled_cond=null_pooled_cond, progress_emb=progress_emb)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self._forward_with_ctx(z, t, z_init, text_ctx, pooled_cond=pooled_cond, progress_emb=progress_emb)

            z = z - v * dt

        return z


# ============================================================================
# Attention Variant DiT Subclasses
# ============================================================================

class UViT(DiT):
    """DiT with U-ViT attention blocks."""
    _model_name = "UViT"

    def __init__(self, **kwargs):
        super().__init__(block_cls=UViTBlock, **kwargs)


class MMDiT(DiT):
    """DiT with Full MMDiT attention blocks (evolving text stream)."""
    _model_name = "MMDiT"

    def __init__(self, **kwargs):
        super().__init__(block_cls=MMDiTBlock, **kwargs)

    def _run_blocks(self, h, conditioning, text_ctx):
        """Override: text tokens evolve through Full MMDiT blocks."""
        x_txt = text_ctx
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h, x_txt = torch.utils.checkpoint.checkpoint(
                    block, h, x_txt, conditioning,
                    use_reentrant=False,
                )
            else:
                h, x_txt = block(h, x_txt, conditioning)
        if self.training:
            h = h + 0.0 * x_txt.sum()
        return h


class DiTAir(DiT):
    """
    DiT-Air with single-stream shared AdaLN.

    Architectural choice: the original DiT-Air (Chen et al., 2025) modulates
    image and text token streams with two independent AdaLN MLPs (dual-stream).
    We share a single AdaLN MLP whose output is reused for both streams —
    halving AdaLN parameters and a small chunk of the per-step FLOPs for no
    measurable quality regression on our DROID/CALVIN setups. All DiT
    checkpoints in this repo are trained under this single-stream variant.
    """
    _model_name = "DiTAir"

    def __init__(self, **kwargs):
        super().__init__(block_cls=DiTAirBlock, **kwargs)

        # Single AdaLN MLP, output reused for both image and text streams below.
        out_dim = 6 * self.hidden_dim
        self.shared_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, out_dim),
        )
        nn.init.zeros_(self.shared_adaLN[-1].weight)
        nn.init.zeros_(self.shared_adaLN[-1].bias)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"{self._model_name} (single-stream shared AdaLN): {n_params/1e6:.2f}M parameters")

    def _run_blocks(self, h, conditioning, text_ctx):
        x_txt = text_ctx
        mod = self.shared_adaLN(conditioning)
        for block in self.blocks:
            # Same modulation vector passed for both streams ⇒ single-stream variant.
            if self.gradient_checkpointing and self.training:
                h, x_txt = torch.utils.checkpoint.checkpoint(
                    block, h, x_txt, mod, mod, use_reentrant=False,
                )
            else:
                h, x_txt = block(h, x_txt, mod, mod)
        return h


def _sample_timesteps(B: int, device: torch.device, schedule: str = "uniform",
                      latent_numel: int = 75264) -> torch.Tensor:
    """Sample training timesteps according to the chosen timestep distribution.

    Args:
        B: Batch size.
        device: Target device.
        schedule: One of "uniform", "rae_shift", "logit_normal".
        latent_numel: Number of elements per latent (N*D), used by RAE shift.

    Returns:
        t: [B] tensor of timesteps in (0, 1).
    """
    if schedule == "uniform":
        return torch.rand(B, device=device)
    elif schedule == "rae_shift":
        # Zheng et al. 2025 — shift toward t≈1 for high-dim latents.
        # alpha = sqrt(latent_numel / 4096); default 196*384 = 75264 → α ≈ 4.29
        alpha = math.sqrt(latent_numel / 4096)
        u = torch.rand(B, device=device)
        return alpha * u / (1.0 + (alpha - 1.0) * u)
    elif schedule == "logit_normal":
        # SD3 / FLUX Kontext — logit-normal(0, 1).
        return torch.sigmoid(torch.randn(B, device=device))
    else:
        raise ValueError(f"Unknown timestep distribution: {schedule}")


def flow_matching_loss(
    model: DiT,
    z_init: torch.Tensor,
    z_target: torch.Tensor,
    text_emb: torch.Tensor,
    text_mask: Optional[torch.Tensor] = None,
    pooled_text_emb: torch.Tensor = None,
    eps: float = 1e-5,
    timestep_distribution: str = "uniform",
    progress_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B = z_target.shape[0]
    device = z_target.device

    latent_numel = z_target.shape[1] * z_target.shape[2]
    t = _sample_timesteps(B, device, schedule=timestep_distribution, latent_numel=latent_numel)

    z_noise = torch.randn_like(z_target)
    t_exp   = t.view(B, 1, 1)
    z_t     = (1 - t_exp) * z_target + (eps + (1 - eps) * t_exp) * z_noise
    target_v = (1 - eps) * z_noise - z_target

    v_pred = model(z_t, t, z_init, text_emb, text_mask=text_mask,
                   pooled_text_emb=pooled_text_emb, progress_emb=progress_emb)

    # Mean cubic loss: mean(|v_pred - target_v|^3). Penalizes large residuals more than MSE.
    return (v_pred - target_v).abs().pow(3).mean()

