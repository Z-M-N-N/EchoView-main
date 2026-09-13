"""推理入口: 返回概率向量, 禁用噪声与 Top-K 强制 —— 规格书 §8 / inference/predictor.py"""
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from model.multimodal_moe import MultimodalMoE


class Predictor:
    """推理器:
    - 加载 checkpoint
    - 推理时固定 torch 随机种子, 关闭 Router 噪声
    - 返回 3 维概率分布 (左转/直行/右转)
    """

    LABELS = ["左转", "直行", "右转"]

    def __init__(self, cfg: Dict, checkpoint_path: str, device: str = "cuda"):
        torch.manual_seed(cfg["inference"].get("seed", 42))
        np.random.seed(cfg["inference"].get("seed", 42))
        self.cfg = cfg
        self.device = device
        self.model = MultimodalMoE(cfg)
        self.model.to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print(f"[Predictor] 加载 checkpoint: {checkpoint_path}")

    @torch.no_grad()
    def predict(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        video: torch.Tensor,
        video_mask: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> Dict:
        """返回概率分布与辅助路由信息. 推理时 use_noise=False, force_top_k=True."""
        text_ids = text_ids.to(self.device)
        text_mask = text_mask.to(self.device)
        video = video.to(self.device)
        video_mask = video_mask.to(self.device)

        logits, aux = self.model(
            text_ids, text_mask, video, video_mask,
            use_noise=False,          # 禁用噪声
            force_top_k=True,         # Top-K 强制 (推理固定)
            return_aux=True,
        )
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        topk_idx = aux["top1_expert"].cpu().numpy() if "top1_expert" in aux else None

        result = {
            "probs": probs,                                    # (B, 3)
            "label_names": self.LABELS,
            "predictions": probs.argmax(-1).tolist(),
        }
        if topk_idx is not None:
            result["top1_expert"] = topk_idx.tolist()
        return result
