"""路由网络: 门控 + 噪声注入 + Top-K 稀疏化 —— 规格书 §4.5.1 / §8 moe/router.py"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Router Network
    - 拓扑: MLP (d_context -> 1024 -> N_experts)
    - 激活: GELU (中间层), 输出层无激活
    - 噪声注入(训练): logits += Normal(0, std=0.01)
    - 门控: G = Softmax(logits) ∈ R^(B×N)
    - Top-K 稀疏化: 保留权重最高 K_top 个专家, 其余置零并重新归一化
    - 输出: 稀疏门控权重 G_sparse ∈ R^(B×N)
    """

    def __init__(
        self,
        d_context: int = 512,
        num_experts: int = 6,
        hidden: int = 1024,
        top_k: int = 2,
        noise_std: float = 0.01,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        self.net = nn.Sequential(
            nn.Linear(d_context, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

    def _add_noise(self, logits: torch.Tensor, training: bool) -> torch.Tensor:
        """仅训练时注入高斯噪声."""
        if training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        return logits

    @staticmethod
    def _top_k_sparse(g: torch.Tensor, k: int) -> torch.Tensor:
        """Top-K 稀疏化并重新归一化."""
        if k >= g.shape[-1]:
            return g
        topk_vals, topk_idx = torch.topk(g, k=k, dim=-1)
        mask = torch.zeros_like(g)
        mask.scatter_(-1, topk_idx, 1.0)
        g_sparse = g * mask
        # 重新归一化
        denom = g_sparse.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        g_sparse = g_sparse / denom
        return g_sparse

    def forward(
        self,
        x: torch.Tensor,          # (B, d_context)
        use_noise: bool = True,
        force_top_k: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (G_sparse, G_dense)."""
        logits = self.net(x)                         # (B, N)
        logits = self._add_noise(logits, training=use_noise)
        g_dense = F.softmax(logits, dim=-1)          # (B, N)
        if force_top_k:
            g_sparse = self._top_k_sparse(g_dense, self.top_k)
        else:
            g_sparse = g_dense
        return g_sparse, g_dense
