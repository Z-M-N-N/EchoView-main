"""batch 组装: 动态填充 K 维度，生成注意力掩码 —— 规格书 §8 data/collator.py"""
from typing import Dict, List

import torch


class MoECollator:
    """将 dataset 返回的单样本 dict 组装为 batch dict.

    由于 dataset 已把 K pad 到 num_videos_max，这里主要做 stack,
    同时保留 video_mask 作为各路是否存在/有效的掩码。
    """

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        text_ids = torch.stack([b["text_ids"] for b in batch])        # (B,S)
        text_mask = torch.stack([b["text_mask"] for b in batch])      # (B,S)
        video = torch.stack([b["video"] for b in batch])              # (B,K,T,C,H,W)
        video_mask = torch.stack([b["video_mask"] for b in batch])    # (B,K)
        label = torch.stack([b["label"] for b in batch])              # (B,)

        return {
            "text_ids": text_ids,
            "text_mask": text_mask,
            "video": video,
            "video_mask": video_mask,
            "label": label,
        }
