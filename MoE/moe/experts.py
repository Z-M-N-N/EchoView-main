"""专家网络群: N 个并行 MLP —— 规格书 §4.5.2 / §8 moe/experts.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Experts(nn.Module):
    """专家网络群 (N_experts=6, 各专家独立参数).

    拓扑: MLP (d_context=512 -> 256 -> 3)
    激活: 中间层 ReLU, 输出层无激活
    并行计算: 使用矩阵分块 (einsum) 实现高效 batch 推理.

    前向: 对每个专家计算 Expert_i(X) -> (B, N, out_dim)
    """

    def __init__(self, d_context: int = 512, hidden: int = 256, out_dim: int = 3, num_experts: int = 6):
        super().__init__()
        self.num_experts = num_experts
        self.d_context = d_context
        self.hidden = hidden
        self.out_dim = out_dim

        # 参数以 (N, in, out) 分块存储
        self.w1 = nn.Parameter(torch.empty(num_experts, d_context, hidden))
        self.b1 = nn.Parameter(torch.zeros(num_experts, hidden))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden, out_dim))
        self.b2 = nn.Parameter(torch.zeros(num_experts, out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.w1, a=0, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.w2, a=0, mode="fan_out", nonlinearity="linear")
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_context) -> (B, N, out_dim) 各专家输出."""
        B = x.shape[0]
        # (B, d) x (N, d, h) -> (B, N, h)
        h = torch.einsum("bd,ndh->bnh", x, self.w1) + self.b1.unsqueeze(0)
        h = F.relu(h)
        out = torch.einsum("bnh,nho->bno", h, self.w2) + self.b2.unsqueeze(0)
        return out  # (B, N, out_dim)
