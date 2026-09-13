"""MoE 损失: 负载均衡 + 重要性 (原始任务) + 路由成本损失 (路由任务) —— 规格书 §5"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ 原始 MoE 任务损失 (规格书 §5) ============

def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """分类主损失: 交叉熵."""
    return F.cross_entropy(logits, labels)


def load_balancing_loss(fraction: torch.Tensor, importance: torch.Tensor, num_experts: int) -> torch.Tensor:
    """负载均衡辅助损失: L_load = N * Σ_i ( importance_i * fraction_i )."""
    return num_experts * torch.sum(importance * fraction)


def importance_loss(importance: torch.Tensor) -> torch.Tensor:
    """重要性辅助损失: L_importance = Σ_i ( importance_i )^2."""
    return torch.sum(importance ** 2)


class MoELoss(nn.Module):
    """总损失: L_total = L_cls + α·L_load + β·L_importance (原始诊断任务)."""

    def __init__(self, alpha: float = 0.01, beta: float = 0.01, num_experts: int = 6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_experts = num_experts

    def forward(self, logits, labels, aux=None, return_parts=False):
        l_cls = classification_loss(logits, labels)
        l_load = torch.tensor(0.0, device=logits.device)
        l_imp = torch.tensor(0.0, device=logits.device)
        if aux is not None and "fraction" in aux and "importance" in aux:
            l_load = load_balancing_loss(aux["fraction"], aux["importance"], self.num_experts)
            l_imp = importance_loss(aux["importance"])
        total = l_cls + self.alpha * l_load + self.beta * l_imp
        if return_parts:
            return total, {"cls": l_cls, "load": l_load, "importance": l_imp}
        return total


# ============ 路由任务损失 (3 类路由: PVD/MCD/CMD) ============

class RouterCostLoss(nn.Module):
    """路由损失 (论文公式):
        L_router = CE(p_k, r_k^*)                      # 模型选择交叉熵
        L_cost   = p_k^T · [c_pvd, c_mcd, c_cmd]      # 期望视图输入成本
        L_total  = L_router + λ_cost · L_cost
    成本向量: PVD=1/3, MCD=2/3, CMD=1 (按其视图输入配置归一化).
    λ_cost 控制"选对模型"与"视图输入效率"的权衡.
    """

    def __init__(self, cost_vector: Tuple[float, ...] = (1 / 3, 2 / 3, 1.0),
                 lambda_cost: float = 0.5):
        super().__init__()
        # 成本向量按类别顺序: 0=PVD, 1=MCD, 2=CMD
        self.register_buffer("cost", torch.tensor(cost_vector, dtype=torch.float32))
        self.lambda_cost = lambda_cost

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                aux=None, return_parts: bool = False):
        """logits: (B,3), labels: (B,) 0/1/2."""
        l_router = classification_loss(logits, labels)                      # CE
        p = F.softmax(logits, dim=-1)                                        # (B,3)
        cost = self.cost.to(logits.device)                                   # 与输入同设备
        l_cost = torch.mean(p @ cost)                                        # 期望成本
        total = l_router + self.lambda_cost * l_cost
        if return_parts:
            return total, {"cls": l_router, "cost": l_cost, "l_cost": l_cost}
        return total


def build_loss(cfg: Dict) -> nn.Module:
    """根据配置选择损失: 路由任务用 RouterCostLoss, 原始任务用 MoELoss."""
    lcfg = cfg.get("loss", {})
    if lcfg.get("router", False):
        return RouterCostLoss(
            cost_vector=lcfg.get("cost_vector", (1 / 3, 2 / 3, 1.0)),
            lambda_cost=lcfg.get("lambda_cost", 0.5),
        )
    return MoELoss(
        alpha=lcfg.get("alpha", 0.01),
        beta=lcfg.get("beta", 0.01),
        num_experts=cfg["model"]["moe"]["num_experts"],
    )
