"""跨模态对齐层: 单路 Q-K-V 交叉注意力 (支持掩码) —— 规格书 §4.3 / §8 fusion/cross_attention.py"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """Text 为 Query, Video 为 Key/Value 的单路交叉注意力 (参数共享于所有 K 路).

    内部子层:
      - MHA: d_model, num_heads, dropout
      - 残差连接 + LayerNorm
      - FFN: (d_model -> 4*d_model -> d_model) + ReLU + 残差 + LayerNorm

    输出张量: F (B, K, d_model)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_ratio: int = 4,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        z_text: torch.Tensor,       # (B, d_model)
        z_video: torch.Tensor,      # (B, K, d_model)
        video_mask: Optional[torch.Tensor] = None,   # (B, K) 1 表示存在
    ) -> torch.Tensor:
        B, K, _ = z_video.shape

        # 以 Text 为 Query: 复制到每路
        q = z_text.unsqueeze(1).expand(B, K, self.d_model)   # (B, K, d_model)
        # Key/Value = 各路视频特征 (参数共享, 直接每路独立 attention)
        k = z_video
        v = z_video

        # key_padding_mask: (B, K), True 表示被 mask (padding)
        if video_mask is not None:
            key_padding_mask = (video_mask == 0).bool()      # (B, K)
        else:
            key_padding_mask = None

        # 对每路独立执行 cross-attention:
        # 将 (B,K,d) -> (B*K, 1, d) 与 (B*K, 1, d) 一次完成
        q_flat = q.reshape(B * K, 1, self.d_model)           # (B*K,1,512)
        k_flat = k.reshape(B * K, 1, self.d_model)
        v_flat = v.reshape(B * K, 1, self.d_model)

        if key_padding_mask is not None:
            kpm = key_padding_mask.reshape(B * K, 1)         # (B*K, 1) batch 模式需要 2D
        else:
            kpm = None

        attn_out, _ = self.attn(
            q_flat, k_flat, v_flat,
            key_padding_mask=kpm,
            need_weights=False,
        )                                                    # (B*K,1,512)
        attn_out = attn_out.view(B, K, self.d_model)

        # 残差 + LayerNorm (残差接文本侧 Query)
        h = self.norm1(q + attn_out)
        # FFN + 残差 + LayerNorm
        out = self.norm2(h + self.ffn(h))
        # 掩码: 缺失路置零
        if video_mask is not None:
            out = out * video_mask.unsqueeze(-1).float()
        return out                                            # (B,K,d_model)


class CrossModalFusion(nn.Module):
    """跨模态对齐层封装: 对每路视频独立执行, 参数共享于所有 K 路."""

    def __init__(self, d_model=512, num_heads=8, dropout=0.1, ffn_ratio=4):
        super().__init__()
        self.block = CrossAttentionBlock(
            d_model=d_model, num_heads=num_heads,
            dropout=dropout, ffn_ratio=ffn_ratio,
        )

    def forward(self, z_text, z_video, video_mask=None):
        """z_text: (B,d), z_video: (B,K,d) -> F (B,K,d)"""
        return self.block(z_text, z_video, video_mask)
