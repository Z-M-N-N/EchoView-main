import torch
import torch.nn as nn
from torch.nn import functional as Fn
from vision_multitask_model import VisionMultiTask

H = 16

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, 4 * H)
        self.up_proj = nn.Linear(H, 4 * H)
        self.down_proj = nn.Linear(4 * H, H)
    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))

class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(H, H)
        self.proj = nn.Linear(H, H)
    def forward(self, x):
        return self.proj(self.qkv(x))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attn()
        self.mlp = MLP()
    def forward(self, x):
        return self.mlp(self.attn(x))

class Merger(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(H, H), nn.Linear(H, H))
    def forward(self, x):
        return self.mlp(x)

class FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(4)])
        self.merger = Merger()
    def forward(self, x, grid_thw=None):
        for b in self.blocks:
            x = b(x)
        return self.merger(x)

torch.manual_seed(0)
vis = FakeVisual()
m = VisionMultiTask(vis, num_binary_tasks=28, reg_tasks=7, use_lora=True,
                    lora_r=4, lora_alpha=8, lora_dropout=0.0, lora_target="attn+mlp")

lora_names = [n for n, p in m.named_parameters() if "lora_" in n]
n_lora_layers = len(lora_names) // 2
print("LoRA 层数量 =", n_lora_layers, "（期望 20）")
assert n_lora_layers == 20, "LoRA 注入数量不对"
print("示例参数名:", lora_names[0])

trainable = [n for n, p in m.named_parameters() if p.requires_grad]
n_lora_tr = sum(1 for n in trainable if "lora_" in n)
n_head_tr = sum(1 for n in trainable if not ("lora_" in n))
print("可训练参数: lora=%d, head=%d" % (n_lora_tr, n_head_tr))
assert all(("lora_" in n) or n.startswith("classifier") or n.startswith("reg_head") for n in trainable)

# 扰动 lora_B，让 LoRA 增量非零，真正检验合并逻辑
with torch.no_grad():
    for n, p in m.named_parameters():
        if n.endswith(".lora_B"):
            p.normal_(0, 0.1)
x = torch.randn(2, 5, H)
cls_o, reg_o = m(x)
print("forward 输出形状:", tuple(cls_o.shape), tuple(reg_o.shape))
cls_o.sum().backward()
assert m.visual.blocks[0].attn.qkv.lora_A.grad is not None, "LoRA 无梯度"

sd = m.get_merged_state_dict()
assert not any("lora_" in k for k in sd), "合并后不应残留 lora 键"
vis2 = FakeVisual()
m2 = VisionMultiTask(vis2, num_binary_tasks=28, reg_tasks=7, use_lora=False)
miss, unexp = m2.load_state_dict(sd, strict=True)
print("合并权重加载: missing=%d unexpected=%d" % (len(miss), len(unexp)))
assert not miss and not unexp
with torch.no_grad():
    c1, r1 = m(x)
    c2, r2 = m2(x)
ok = torch.allclose(c1, c2, atol=1e-5) and torch.allclose(r1, r2, atol=1e-5)
print("LoRA 模型 vs 合并后普通模型 输出一致:", ok)
assert ok

mw = m.visual.blocks[0].attn.qkv.merged_weight()
with torch.no_grad():
    out_lora = m.visual.blocks[0].attn.qkv(x)
    out_merged = Fn.linear(x, mw.to(x.dtype), m.visual.blocks[0].attn.qkv.bias)
eq = torch.allclose(out_lora, out_merged, atol=1e-5)
print("merged_weight 数学等价:", eq)
assert eq

ma = VisionMultiTask(FakeVisual(), num_binary_tasks=28, reg_tasks=7, temporal_pool="attention")
ca, ra = ma(x)
print("attention pool 输出:", tuple(ca.shape), tuple(ra.shape))
assert ca.shape == (2, 28) and ra.shape == (2, 7)
print("ALL MOCK TESTS PASSED")
