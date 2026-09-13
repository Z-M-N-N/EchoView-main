"""训练数据集: 实现 __getitem__，返回 (text_tokens, video_tensor, label, mask) —— 规格书 §8 data/dataset.py
支持两种视频加载模式:
  - npy:   读取预抽取帧缓存 (T,H,W,C) uint8
  - video: 用 cv2/decord 从 mp4 实时均匀抽帧 (推荐, 大数据集省存储)
"""
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.transforms import get_video_transforms


class MultimodalMoEDataset(Dataset):
    """多模态 MoE 数据集 (超声诊断场景: 文本+多路超声视频 -> 是/否).

    索引 JSON 结构 (规格书 §3.2 适配):
    {
      "id": "str",                       # 原患者 id
      "text": "str",                     # 诊断问题文本
      "video_paths": {"A4C": "/abs/a.mp4", "Parasternal_Long": "/abs/b.mp4", ...},  # 视图名->路径 (动态 K)
      "label": 0 | 1,                    # mark -1 -> 0, mark 1 -> 1
      "metadata": {"diag_item": "主动脉瓣增厚", ...}
    }
    缺失/空 video_paths 视为该样本无视频 (K 相应减少).
    """

    def __init__(
        self,
        index_path: str,
        tokenizer=None,
        max_seq_len: int = 64,
        num_frames: int = 16,
        num_videos_max: int = 6,
        frame_size: int = 224,
        video_root: str = "",
        transform=None,
        preprocess_mode: str = "video",   # npy | video
    ):
        with open(index_path, "r", encoding="utf-8") as f:
            self.samples: List[Dict] = json.load(f)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_frames = num_frames
        self.num_videos_max = num_videos_max
        self.frame_size = frame_size
        self.video_root = video_root
        self.transform = transform
        self.preprocess_mode = preprocess_mode

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        if self.video_root:
            return os.path.join(self.video_root, path)
        return path

    def _load_video_frames(self, path: str) -> torch.Tensor:
        """加载单路视频 -> (num_frames, H, W, C) uint8 tensor."""
        p = self._resolve_path(path)
        if self.preprocess_mode == "npy":
            arr = np.load(p, allow_pickle=False)          # (T,H,W,C) uint8
            return torch.from_numpy(arr)
        # video 模式: cv2 实时均匀抽帧
        import cv2
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {p}")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError(f"视频无帧: {p}")
        # 均匀采样 num_frames 帧
        total = len(frames)
        if total <= self.num_frames:
            idx = list(range(total))
            while len(idx) < self.num_frames:
                idx.append(idx[-1])
        else:
            step = total / self.num_frames
            idx = [min(int((i + 0.5) * step), total - 1) for i in range(self.num_frames)]
        out = []
        for i in idx:
            f = cv2.resize(frames[i], (self.frame_size, self.frame_size),
                           interpolation=cv2.INTER_LINEAR)
            out.append(f)
        # BGR -> RGB
        arr = np.stack(out).astype(np.uint8)              # (n,H,W,3)
        arr = arr[..., ::-1].copy()
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # ---------- 文本: tokenize -> (S,) ----------
        text = sample.get("text", "")
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                text,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_ids = tokens["input_ids"].squeeze(0)          # (S,)
            text_mask = tokens["attention_mask"].squeeze(0)    # (S,)
        else:
            text_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
            text_mask = torch.ones(self.max_seq_len, dtype=torch.long)

        # ---------- 视频: 逐路加载 -> (K, T, C, H, W), 动态 K ----------
        video_paths = sample.get("video_paths", {})
        # 支持 dict(视图名->路径) 或 list(路径)
        if isinstance(video_paths, dict):
            paths = list(video_paths.values())
        elif isinstance(video_paths, list):
            paths = video_paths
        else:
            paths = []

        videos = []
        video_mask = torch.zeros(self.num_videos_max, dtype=torch.long)
        for i, p in enumerate(paths[:self.num_videos_max]):
            if not p:
                continue
            try:
                frames = self._load_video_frames(p)                  # (T,H,W,C) uint8
                vid = self.transform(frames, self.num_frames)       # (T,C,H,W) float
                videos.append(vid)
                video_mask[i] = 1
            except Exception as e:
                print(f"[dataset] 跳过 video_paths[{i}]={p}: {e}")
        K = len(videos)
        if K == 0:
            video_tensor = torch.zeros(
                self.num_videos_max, self.num_frames, 3,
                self.frame_size, self.frame_size)
        else:
            video_tensor = torch.stack(videos)                      # (K,T,C,H,W)
            if K < self.num_videos_max:                             # pad 到 K_max
                pad = torch.zeros(
                    self.num_videos_max - K, self.num_frames, 3,
                    self.frame_size, self.frame_size)
                video_tensor = torch.cat([video_tensor, pad], dim=0)

        return {
            "text_ids": text_ids,        # (S,)
            "text_mask": text_mask,      # (S,)
            "video": video_tensor,       # (K_max,T,C,H,W)
            "video_mask": video_mask,    # (K_max,)
            "label": torch.tensor(int(sample["label"]), dtype=torch.long),
        }
