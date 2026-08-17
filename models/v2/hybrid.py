"""v2.0 hybrid family: standard-conv stem + Transformer encoder.

``v2_hybrid_s`` (~16M) and ``v2_hybrid_l`` (~41M) follow a compact
CNN + Transformer design:

  - a standard 3x3 convolution stem downsamples 256x256 → 16x16 and
    projects to ``embed_dim`` channels (token grid of 16x16 = 256 tokens)
  - a learned positional embedding is added to the tokens
  - a pre-LN Transformer encoder (multi-head self-attention + MLP) models
    global relationships between spatial tokens
  - global average pooling over tokens feeds the dual classification heads

Attention is implemented with plain matmul/softmax so ONNX export
(opset 13, dynamic batch) stays clean. Models are trained from scratch.
"""

import torch
import torch.nn as nn

from models.v2.blocks import ConvBnAct, DualHead


class Attention(nn.Module):
    """Multi-head self-attention (manual implementation, ONNX-friendly)."""

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: attention + MLP with residual connections."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HybridNet(nn.Module):
    """CNN stem + Transformer encoder + dual heads.

    Args:
        stem_widths: Channel widths of the 4 stride-2 stem convs.
        embed_dim: Transformer embedding dimension (= stem output channels).
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim ratio inside transformer blocks.
        num_handedness / num_presence: Output class counts.
        dropout: Dropout before the classification heads.
    """

    def __init__(self, stem_widths, embed_dim, depth, num_heads,
                 mlp_ratio=4.0, num_handedness=2, num_presence=2, dropout=0.1):
        super().__init__()
        convs = []
        in_ch = 1
        for w in stem_widths:
            convs.append(ConvBnAct(in_ch, w, 3, stride=2))
            in_ch = w
        assert in_ch == embed_dim, (
            f"stem_widths must end at embed_dim ({in_ch} != {embed_dim})"
        )
        self.stem = nn.Sequential(*convs)  # 256 -> 16x16 grid

        self.pos_embed = nn.Parameter(
            torch.zeros(1, 16 * 16, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.heads = DualHead(embed_dim, num_handedness, num_presence, dropout)

    def forward(self, x):
        x = self.stem(x)  # (B, D, 16, 16)
        B, D, _, _ = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, 256, D)
        tokens = tokens + self.pos_embed
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)  # (B, 256, D)
        features = tokens.mean(dim=1)  # global average pooling over tokens
        return self.heads(features)


def v2_hybrid_s(pretrained=False, num_handedness=2, num_presence=2):
    """CNN stem + Transformer encoder, ~16M params (trained from scratch)."""
    return HybridNet(
        stem_widths=[64, 128, 256, 384], embed_dim=384,
        depth=8, num_heads=6, mlp_ratio=4.0,
        num_handedness=num_handedness, num_presence=num_presence,
    )


def v2_hybrid_l(pretrained=False, num_handedness=2, num_presence=2):
    """CNN stem + Transformer encoder, ~40M params (trained from scratch)."""
    return HybridNet(
        stem_widths=[96, 192, 384, 512], embed_dim=512,
        depth=12, num_heads=8, mlp_ratio=4.0,
        num_handedness=num_handedness, num_presence=num_presence,
    )
