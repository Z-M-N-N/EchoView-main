"""上下文聚合层: Perceiver-style 交叉注意力池化 —— 规格书 §4.4 / §8 aggregation/perceiver_pooler.py"""
from typing import Optional

import torch
import torch.nn as nn


class PerceiverPooler(nn.Module):
    """以可学习潜变量 L 为 Query, 以 F 为 Key/Value 执行交叉注意力.

    - 可学习潜变量: L (M, d_model), M=8
    - 输出: M 个输出向量均值池化 -> X_context (B, d_context)
    - 备选: 展平后经 MLP 降维
    """

    def __init__(
        self,
        d_model: int = 512,
        d_context: int = 512,
        num_latents: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_context = d_context
        self.num_latents = num_latents

        # 可学习潜变量 (M, d_model)
        self.latents = nn.Parameter(torch.randn(num_latents, d_model) * 0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # 投影到 d_context
        self.projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_context),
        )

    def forward(
        self,
        f: torch.Tensor,                       # (B, K, d_model) 联合表征
        video_mask: Optional[torch.Tensor] = None,  # (B, K)
    ) -> torch.Tensor:
        B, K, _ = f.shape

        # Query = L 扩展 (B, M, d_model)
        q = self.latents.unsqueeze(0).expand(B, self.num_latents, self.d_model)
        k = f
        v = f

        if video_mask is not None:
            key_padding_mask = (video_mask == 0).bool()   # (B, K)
        else:
            key_padding_mask = None

        attn_out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.norm1(q + attn_out)
        out = self.norm2(h + self.ffn(h))      # (B, M, d_model)

        # 均值池化 + 投影
        pooled = out.mean(dim=1)               # (B, d_model)
        x_context = self.projection(pooled)    # (B, d_context)
        return x_context
