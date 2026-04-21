"""
DFPIR: Degradation-Aware Feature Perturbation for All-in-One Image Restoration
Based on CVPR 2025 paper by Tian et al.
Paper: https://github.com/TxpHome/DFPIR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


# ─────────────────────────────────────────────
# 1.  Utility layers
# ─────────────────────────────────────────────

class LayerNorm(nn.Module):
    """Layer-norm that works on (B, C, H, W) tensors."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)          # B H W C
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)        # B C H W


class FeedForward(nn.Module):
    """Point-wise FFN with depth-wise 3×3 conv expansion."""
    def __init__(self, dim, ffn_factor=2.66):
        super().__init__()
        hidden = int(dim * ffn_factor)
        self.proj_in  = nn.Conv2d(dim, hidden * 2, 1)
        self.dw       = nn.Conv2d(hidden * 2, hidden * 2, 3,
                                  padding=1, groups=hidden * 2)
        self.proj_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.dw(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * F.gelu(x2)
        return self.proj_out(x)


# ─────────────────────────────────────────────
# 2.  Restormer Transformer Block (backbone)
# ─────────────────────────────────────────────

class MultiDConvHeadTransposedAttention(nn.Module):
    """Channel-wise transposed self-attention from Restormer."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv    = nn.Conv2d(dim, dim * 3, 1)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3,
                                 padding=1, groups=dim * 3)
        self.proj   = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (h c) x y -> b h c (x y)', h=self.num_heads)
        k = rearrange(k, 'b (h c) x y -> b h c (x y)', h=self.num_heads)
        v = rearrange(v, 'b (h c) x y -> b h c (x y)', h=self.num_heads)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out  = (attn @ v)
        out  = rearrange(out, 'b h c (x y) -> b (h c) x y', x=H, y=W)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_factor=2.66):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn  = MultiDConvHeadTransposedAttention(dim, num_heads)
        self.norm2 = LayerNorm(dim)
        self.ffn   = FeedForward(dim, ffn_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# 3.  Degradation Guidance Module (DGM)
# ─────────────────────────────────────────────

class DegradationGuidanceModule(nn.Module):
    """
    Maps CLIP text-prompt embedding → channel guidance vector P_ec ∈ R^{2C}.
    During inference without CLIP we fall back to a learned per-task embedding.
    """
    def __init__(self, prompt_dim: int, out_dim: int, num_tasks: int = 5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(prompt_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        # fallback: simple learned task tokens
        self.task_embeds = nn.Embedding(num_tasks, prompt_dim)

    def forward(self, prompt, task_id=None):
        """
        Args:
            prompt  : (B, prompt_dim) CLIP embedding  OR  None
            task_id : (B,) int tensor  – used when prompt is None
        Returns:
            (B, out_dim)
        """
        if prompt is None:
            assert task_id is not None
            prompt = self.task_embeds(task_id)   # (B, prompt_dim)
        return self.mlp(prompt)                   # (B, out_dim)


# ─────────────────────────────────────────────
# 4.  DGCPM – channel-wise perturbation
# ─────────────────────────────────────────────

class DegradationGuidedChannelPerturbationModule(nn.Module):
    """
    Expand channels 2×, shuffle according to P_ec top-K order,
    then halve back to original channel count.
    """
    def __init__(self, dim: int, prompt_dim: int, num_tasks: int = 5):
        super().__init__()
        self.expand  = nn.Conv2d(dim, dim * 2, 1)
        self.shrink  = nn.Conv2d(dim * 2, dim, 1)
        self.ln      = LayerNorm(dim * 2)
        self.dgm     = DegradationGuidanceModule(prompt_dim, dim * 2, num_tasks)

    def forward(self, Fn, prompt=None, task_id=None):
        """
        Args:
            Fn       : (B, C, H, W)
            prompt   : (B, prompt_dim) or None
            task_id  : (B,) or None
        Returns:
            Q : (B, C, H, W)  channel-perturbed features
        """
        B, C, H, W = Fn.shape
        F2n = self.ln(self.expand(Fn))              # (B, 2C, H, W)

        # guidance → channel-order indices
        pec = self.dgm(prompt, task_id)             # (B, 2C)
        _, idx = torch.sort(pec, dim=1, descending=True)  # top-K order
        # Permute channels according to the per-sample order
        idx_4d = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        F2n_shuffled = torch.gather(F2n, 1, idx_4d)

        Q = self.shrink(F2n_shuffled)               # (B, C, H, W)
        return Q


# ─────────────────────────────────────────────
# 5.  CAAPM – attention-wise perturbation
# ─────────────────────────────────────────────

class ChannelAdaptedAttentionPerturbationModule(nn.Module):
    """
    Cross-attention between shuffled Q and original Fn (K, V),
    with top-K masking on the attention map (perturbation factor γ).
    """
    def __init__(self, dim: int, gamma: float = 0.9):
        super().__init__()
        self.gamma = gamma
        self.temperature = nn.Parameter(torch.ones(1))

        self.q_proj = nn.Sequential(nn.Conv2d(dim, dim, 1),
                                    nn.Conv2d(dim, dim, 3, padding=1, groups=dim))
        self.k_proj = nn.Sequential(nn.Conv2d(dim, dim, 1),
                                    nn.Conv2d(dim, dim, 3, padding=1, groups=dim))
        self.v_proj = nn.Sequential(nn.Conv2d(dim, dim, 1),
                                    nn.Conv2d(dim, dim, 3, padding=1, groups=dim))
        self.out_proj = nn.Conv2d(dim, dim, 1)
        self.ffn      = FeedForward(dim)
        self.norm     = LayerNorm(dim)

    def forward(self, Q, Fn):
        """
        Args:
            Q  : (B, C, H, W) – output of DGCPM (shuffled)
            Fn : (B, C, H, W) – original encoder features
        Returns:
            Fb : (B, C, H, W)
        """
        B, C, H, W = Fn.shape
        q = self.q_proj(Q)     # (B, C, H, W)
        k = self.k_proj(Fn)
        v = self.v_proj(Fn)

        # Reshape → (B, C, HW)
        q = q.view(B, C, -1)
        k = k.view(B, C, -1)
        v = v.view(B, C, -1)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        # Channel-wise transposed attention  (B, C, C)
        attn = torch.bmm(q, k.transpose(1, 2)) * self.temperature
        attn = attn.softmax(dim=-1)

        # Top-K masking  (keep γ fraction of entries per row)
        k_keep = max(1, int(C * self.gamma))
        topk_vals, _ = torch.topk(attn, k_keep, dim=-1)
        threshold     = topk_vals[:, :, -1:].expand_as(attn)
        mask          = (attn >= threshold).float()
        attn          = attn * mask

        # Weighted sum over value tokens  (B, C, HW)
        out = torch.bmm(attn, v)
        out = out.view(B, C, H, W)

        Fa = self.out_proj(out)
        Fb = Fa + self.ffn(self.norm(Fa))
        return Fb


# ─────────────────────────────────────────────
# 6.  DGPB = DGCPM + CAAPM
# ─────────────────────────────────────────────

class DegradationGuidedPerturbationBlock(nn.Module):
    def __init__(self, dim: int, prompt_dim: int,
                 num_tasks: int = 5, gamma: float = 0.9):
        super().__init__()
        self.dgcpm = DegradationGuidedChannelPerturbationModule(
            dim, prompt_dim, num_tasks)
        self.caapm = ChannelAdaptedAttentionPerturbationModule(dim, gamma)

    def forward(self, Fn, prompt=None, task_id=None):
        Q  = self.dgcpm(Fn, prompt, task_id)   # Eq. (2)
        Fb = self.caapm(Q, Fn)                  # Eq. (3-4)
        return Fb


# ─────────────────────────────────────────────
# 7.  Encoder / Decoder helpers
# ─────────────────────────────────────────────

class DownSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1)

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# ─────────────────────────────────────────────
# 8.  Full DFPIR network
# ─────────────────────────────────────────────

class DFPIR(nn.Module):
    """
    4-level encoder-decoder (Restormer backbone) with DGPB blocks
    inserted at each skip-connection stage.

    Default architecture mirrors the paper:
        blocks_per_level = [4, 6, 6, 8]
        base channels    = 48
    """
    def __init__(
        self,
        in_channels:  int   = 3,
        out_channels: int   = 3,
        dim:          int   = 48,
        num_blocks:   list  = [4, 6, 6, 8],
        num_heads:    list  = [1, 2, 4, 8],
        ffn_factor:   float = 2.66,
        num_tasks:    int   = 5,
        prompt_dim:   int   = 512,     # CLIP ViT-B/32 embedding size
        gamma:        float = 0.9,
    ):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, dim, 3, padding=1)

        # ── encoder ──────────────────────────────────────────────────────────
        self.enc1 = nn.Sequential(*[
            TransformerBlock(dim, num_heads[0], ffn_factor)
            for _ in range(num_blocks[0])])
        self.down1 = DownSample(dim)          # dim → 2*dim

        self.enc2 = nn.Sequential(*[
            TransformerBlock(dim * 2, num_heads[1], ffn_factor)
            for _ in range(num_blocks[1])])
        self.down2 = DownSample(dim * 2)      # 2*dim → 4*dim

        self.enc3 = nn.Sequential(*[
            TransformerBlock(dim * 4, num_heads[2], ffn_factor)
            for _ in range(num_blocks[2])])
        self.down3 = DownSample(dim * 4)      # 4*dim → 8*dim

        # ── bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(*[
            TransformerBlock(dim * 8, num_heads[3], ffn_factor)
            for _ in range(num_blocks[3])])

        # ── DGPBs (one per skip connection) ──────────────────────────────────
        self.dgpb1 = DegradationGuidedPerturbationBlock(dim,     prompt_dim, num_tasks, gamma)
        self.dgpb2 = DegradationGuidedPerturbationBlock(dim * 2, prompt_dim, num_tasks, gamma)
        self.dgpb3 = DegradationGuidedPerturbationBlock(dim * 4, prompt_dim, num_tasks, gamma)
        self.dgpb4 = DegradationGuidedPerturbationBlock(dim * 8, prompt_dim, num_tasks, gamma)

        # ── decoder ──────────────────────────────────────────────────────────
        self.up3   = UpSample(dim * 8)        # 8*dim → 4*dim (space×2)
        self.fuse3 = nn.Conv2d(dim * 8, dim * 4, 1)
        self.dec3  = nn.Sequential(*[
            TransformerBlock(dim * 4, num_heads[2], ffn_factor)
            for _ in range(num_blocks[2])])

        self.up2   = UpSample(dim * 4)
        self.fuse2 = nn.Conv2d(dim * 4, dim * 2, 1)
        self.dec2  = nn.Sequential(*[
            TransformerBlock(dim * 2, num_heads[1], ffn_factor)
            for _ in range(num_blocks[1])])

        self.up1   = UpSample(dim * 2)
        self.fuse1 = nn.Conv2d(dim * 2, dim, 1)
        self.dec1  = nn.Sequential(*[
            TransformerBlock(dim, num_heads[0], ffn_factor)
            for _ in range(num_blocks[0])])

        self.head  = nn.Conv2d(dim, out_channels, 3, padding=1)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x, prompt=None, task_id=None):
        """
        Args:
            x        : (B, 3, H, W)  degraded input image  [0, 1]
            prompt   : (B, 512)      CLIP text embedding  – or None
            task_id  : (B,)          int tensor  0..num_tasks-1  – or None
                       (used only when prompt is None)
        Returns:
            (B, 3, H, W)  restored image residual added to x
        """
        # shallow features
        F0 = self.stem(x)                            # (B, C,   H,   W)

        # encoder
        F1 = self.enc1(F0)                           # (B, C,   H,   W)
        F2 = self.enc2(self.down1(F1))               # (B, 2C,  H/2, W/2)
        F3 = self.enc3(self.down2(F2))               # (B, 4C,  H/4, W/4)
        Fe = self.bottleneck(self.down3(F3))         # (B, 8C,  H/8, W/8)

        # perturbation at each skip connection
        F1p = self.dgpb1(F1, prompt, task_id)
        F2p = self.dgpb2(F2, prompt, task_id)
        F3p = self.dgpb3(F3, prompt, task_id)
        Fep = self.dgpb4(Fe, prompt, task_id)

        # decoder with skip connections
        d3 = self.up3(Fep)                           # (B, 4C,  H/4, W/4)
        d3 = self.fuse3(torch.cat([d3, F3p], dim=1))
        d3 = self.dec3(d3)

        d2 = self.up2(d3)                            # (B, 2C,  H/2, W/2)
        d2 = self.fuse2(torch.cat([d2, F2p], dim=1))
        d2 = self.dec2(d2)

        d1 = self.up1(d2)                            # (B, C,   H,   W)
        d1 = self.fuse1(torch.cat([d1, F1p], dim=1))
        d1 = self.dec1(d1)

        out = self.head(d1) + x                      # global residual
        return out
