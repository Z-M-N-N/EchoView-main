"""MoE 层: 整合路由与专家, 计算加权求和及辅助统计 —— 规格书 §4.5 / §8 moe/moe_layer.py"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from moe.router import Router
from moe.experts import Experts


class MoELayer(nn.Module):
    """MoE 决策层:
      Logits = Σ_i [ G_sparse[:, i] * Expert_i(X_context) ]
    同时返回用于辅助损失的统计量 (fraction / importance).
    """

    def __init__(
        self,
        d_context: int = 512,
        num_experts: int = 6,
        top_k: int = 2,
        router_hidden: int = 1024,
        expert_hidden: int = 256,
        out_dim: int = 3,
        noise_std: float = 0.01,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.out_dim = out_dim

        self.router = Router(
            d_context=d_context,
            num_experts=num_experts,
            hidden=router_hidden,
            top_k=top_k,
            noise_std=noise_std,
            dropout=dropout,
        )
        self.experts = Experts(
            d_context=d_context,
            hidden=expert_hidden,
            out_dim=out_dim,
            num_experts=num_experts,
        )

    def forward(
        self,
        x_context: torch.Tensor,          # (B, d_context)
        use_noise: bool = True,
        force_top_k: bool = True,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """返回 (logits (B, out_dim), aux_stats)."""
        g_sparse, g_dense = self.router(x_context, use_noise=use_noise, force_top_k=force_top_k)
        expert_out = self.experts(x_context)   # (B, N, out_dim)

        # Logits = Σ_i G_sparse[:,i] * Expert_i(X)  -> (B, 3)
        logits = torch.einsum("bn,bno->bo", g_sparse, expert_out)

        aux = {"g_sparse": g_sparse, "g_dense": g_dense}
        if return_aux:
            with torch.no_grad():
                B = x_context.shape[0]
                # fraction_i = (1/B) Σ 1_{G_sparse[:,i] > 0}  (被选中样本比例)
                fraction = (g_sparse > 0).float().mean(dim=0)      # (N,)
                # importance_i = (1/B) Σ G_sparse[:,i]            (平均路由权重)
                importance = g_sparse.mean(dim=0)                  # (N,)
                aux["fraction"] = fraction
                aux["importance"] = importance
                # 路由熵 (评估用)
                entropy = -(g_dense * torch.log(g_dense.clamp_min(1e-6))).sum(-1).mean()
                aux["router_entropy"] = entropy
                aux["top1_expert"] = g_dense.argmax(dim=-1)
        return logits, aux
