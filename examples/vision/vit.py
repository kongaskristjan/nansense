"""Simple pre-norm Vision Transformer for small-image classification.

Follows Dosovitskiy et al. 2020 ("An Image is Worth 16x16 Words") at
ViT-Tiny-ish proportions, with two simplifications that suit playgrad's
fx-traced visualization: attention is spelled out with explicit linear
layers and `F.scaled_dot_product_attention` (a tuple-returning
`nn.MultiheadAttention` would capture nothing — playgrad only stores
tensor-valued node outputs), and the classification head global-average-
pools the tokens instead of using a class token.

Note that token-shaped activations (`[tokens, dim]` per sample) are
neither `[C, H, W]` nor `[F]`, so playgrad's layer cards show these
layers without activation/gradient strips.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    """Multi-head self-attention with a fused qkv projection."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        # [B, N, D] -> [B, N, 3, H, d] -> [3, B, H, N, d]; `-1` keeps the
        # batch dim dynamic, and explicit indexing (no tuple unpacking)
        # keeps the module fx-traceable.
        qkv = self.qkv(x).reshape(-1, x.shape[1], 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)  # [B, H, N, d]
        out = out.transpose(1, 2).reshape(-1, x.shape[1], self.num_heads * self.head_dim)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm encoder block: LN -> attention and LN -> MLP, both residual."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mlp_ratio * dim)
        self.fc2 = nn.Linear(mlp_ratio * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class SimpleViT(nn.Module):
    """Patch embedding -> learned position embedding -> `depth` pre-norm
    transformer blocks -> LayerNorm -> token mean pool -> linear head."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 10,
        dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} not divisible by patch_size={patch_size}")
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.blocks = nn.Sequential(
            *[TransformerBlock(dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)  # [B, D, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pool over tokens
        return self.head(x)


def vit_tiny(image_size: int, patch_size: int, num_classes: int = 10) -> SimpleViT:
    return SimpleViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes)
