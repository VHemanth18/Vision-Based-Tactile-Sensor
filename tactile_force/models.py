"""Force regression networks: tactile frame (3 or 6 channels) -> contact force (scaled).

Two architectures:
    cnn  - convolutional network (local deformation features, translation-equivariant)
    vit  - Vision Transformer (patch tokens + self-attention over the whole contact patch)
"""
from __future__ import annotations

import torch
from torch import nn


# ----------------------------------------------------------------------------- CNN
def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3 conv -> BN -> ReLU layers, then 2x2 max-pooling (halves H and W)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNNRegressor(nn.Module):
    """Stacked conv blocks -> global average pooling -> MLP head -> 1 output (force).

    Global average pooling makes the head independent of the input resolution and of where the
    contact patch lies on the sensor.
    """

    def __init__(self, in_ch: int = 6, channels: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3,
                 pool: str = "avg"):
        super().__init__()
        blocks, c_prev = [], in_ch
        for c in channels:
            blocks.append(_conv_block(c_prev, c))
            c_prev = c
        self.features = nn.Sequential(*blocks)
        if pool not in ("avg", "max", "avgmax"):
            raise ValueError(f"model.cnn.pool must be avg | max | avgmax, got {pool}")
        self.pool_mode = pool
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        n_feat = 2 * c_prev if pool == "avgmax" else c_prev
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(n_feat, 128),
                                  nn.ReLU(inplace=True), nn.Linear(128, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        if self.pool_mode == "avg":
            z = self.pool(f)
        elif self.pool_mode == "max":
            z = self.maxpool(f)
        else:
            z = torch.cat([self.pool(f), self.maxpool(f)], dim=1)
        return self.head(z).squeeze(1)


# ----------------------------------------------------------------------------- Vision Transformer
class PatchEmbedding(nn.Module):
    """Split the image into non-overlapping p x p patches and project each to `dim`.

    A conv with kernel = stride = p is exactly "flatten each patch, then apply one shared Linear".
    """

    def __init__(self, in_ch: int, dim: int, patch: int, input_hw: tuple[int, int]):
        super().__init__()
        h, w = input_hw
        if h % patch or w % patch:
            raise ValueError(f"preprocess.size {input_hw} must be divisible by model.vit.patch_size={patch}")
        self.grid = (h // patch, w // patch)
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, dim)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        if dim % heads:
            raise ValueError(f"vit.dim={dim} must be divisible by vit.heads={heads}")
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) / self.head_dim ** 0.5   # (B, heads, N, N)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)         # concat heads
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Pre-norm encoder block: x + MHSA(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTRegressor(nn.Module):
    """Vision Transformer with a 1-output regression head on the [CLS] token."""

    def __init__(self, in_ch: int, input_hw: tuple[int, int], patch_size: int = 8, dim: int = 192,
                 depth: int = 6, heads: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.patch = PatchEmbedding(in_ch, dim, patch_size, input_hw)
        n_tokens = self.patch.grid[0] * self.patch.grid[1]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, dim))  # learned positions
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed
        x = self.norm(self.blocks(self.drop(x)))
        return self.head(x[:, 0]).squeeze(1)



ARCHS = ("cnn", "vit")


def build_model(cfg, in_ch: int) -> nn.Module:
    m = cfg.model
    if m.arch == "cnn":
        c = m.get("cnn", {})
        return CNNRegressor(in_ch, tuple(c.get("channels", (32, 64, 128, 256))), c.get("dropout", 0.3),
                            c.get("pool", "avg"))
    if m.arch == "vit":
        v = m.get("vit", {})
        return ViTRegressor(in_ch, tuple(cfg.preprocess.size), v.get("patch_size", 8), v.get("dim", 192),
                            v.get("depth", 6), v.get("heads", 6), v.get("mlp_ratio", 4.0), v.get("dropout", 0.1))
    raise ValueError(f"Unknown model.arch: {m.arch}; choose from {ARCHS}")
