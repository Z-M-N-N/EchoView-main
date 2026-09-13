"""冒烟测试: 验证全链路前向/反向、张量维度、损失计算 — 无需真实权重 (随机初始化)."""
import os
import sys
import random

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    from utils.config import load_config
    cfg = load_config(cfg_path)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device
    print(f"[smoke] device={device}")

    from model.multimodal_moe import MultimodalMoE
    model = MultimodalMoE(cfg).to(device)
    model.train()

    B, K, T, C, H, W = 2, 4, 16, 3, 224, 224
    S = cfg["input"]["max_seq_len"]
    input_ids = torch.randint(0, 30000, (B, S), device=device)
    text_mask = torch.ones((B, S), device=device)
    video = torch.randn(B, K, T, C, H, W, device=device)
    video_mask = torch.ones((B, K), device=device)
    # 模拟动态 K: 路3缺失
    video_mask[:, 2] = 0
    label = torch.randint(0, 3, (B,), device=device)

    from losses.moe_losses import MoELoss
    criterion = MoELoss(alpha=cfg["loss"]["alpha"], beta=cfg["loss"]["beta"],
                        num_experts=cfg["model"]["moe"]["num_experts"])

    logits, aux = model(input_ids, text_mask, video, video_mask)
    print(f"[smoke] logits shape: {tuple(logits.shape)} (期望 (B,3)=({B},{cfg['input']['num_classes']}))")
    loss, parts = criterion(logits, label, aux, return_parts=True)
    print(f"[smoke] loss={loss.item():.4f} cls={parts['cls'].item():.4f} "
          f"load={parts['load'].item():.4f} imp={parts['importance'].item():.4f}")

    loss.backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    print(f"[smoke] 可训练参数中有梯度的比例: {sum(grads)}/{len(grads)}")
    assert logits.shape == (B, 3), f"输出维度错误: {tuple(logits.shape)}"
    assert torch.isfinite(loss), "loss 出现 NaN"
    print("[smoke] 前向+反向+损失 全部通过 ✅")

    # 推理模式 (无噪声, Top-K 强制)
    model.eval()
    with torch.no_grad():
        logits2, aux2 = model(input_ids, text_mask, video, video_mask,
                              use_noise=False, force_top_k=True)
        probs = torch.softmax(logits2, dim=-1)
        print(f"[smoke] 推理概率: {probs.cpu().numpy().round(3).tolist()}")
        assert torch.allclose(probs.sum(-1), torch.ones(B, device=device)), "概率和≠1"
    print("[smoke] 推理模式通过 ✅")


if __name__ == "__main__":
    main()
