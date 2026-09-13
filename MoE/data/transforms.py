"""视频时序/空间增强注册函数 (架构规格书 §4.2 / §8 data/transforms.py)"""
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def uniform_sample_indices(num_frames: int, total_frames: int) -> List[int]:
    """均匀采样 num_frames 帧索引 (规格书: 均匀采样16帧)."""
    if total_frames <= num_frames:
        # 不足时循环补帧
        idx = list(range(total_frames))
        while len(idx) < num_frames:
            idx.append(idx[-1])
        return idx[:num_frames]
    step = total_frames / num_frames
    return [min(int((i + 0.5) * step), total_frames - 1) for i in range(num_frames)]


def random_sample_indices(num_frames: int, total_frames: int) -> List[int]:
    """随机采样 num_frames 帧索引 (训练时增强)."""
    if total_frames <= num_frames:
        idx = list(range(total_frames))
        while len(idx) < num_frames:
            idx.append(random.randrange(total_frames))
        return idx[:num_frames]
    return sorted(random.sample(range(total_frames), num_frames))


@dataclass
class VideoTransforms:
    """视频变换集合: 时序采样 + 空间缩放/翻转."""
    spatial_size: int = 224
    temporal_mode: str = "uniform"   # uniform | random
    random_hflip: bool = False
    mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])

    def __post_init__(self):
        self._spatial = T.Compose([
            T.Resize((self.spatial_size, self.spatial_size)),
        ])
        self._normalize = T.Normalize(self.mean, self.std)

    def _sample_indices(self, total_frames: int, num_frames: int) -> List[int]:
        if self.temporal_mode == "random":
            return random_sample_indices(num_frames, total_frames)
        return uniform_sample_indices(num_frames, total_frames)

    def __call__(self, frames: torch.Tensor, num_frames: int = 16) -> torch.Tensor:
        """frames: (T, H, W, C) uint8 [0,255] -> (num_frames, C, H, W) float32 归一化."""
        T_ = frames.shape[0]
        idx = self._sample_indices(T_, num_frames)
        frames = frames[idx]                       # (n, H, W, C) uint8
        # HWC -> CHW (Resize 期望 CHW)
        frames = frames.permute(0, 3, 1, 2)        # (n, C, H, W)
        # 空间处理 (Resize 保持通道维)
        frames = torch.stack([self._spatial(f) for f in frames])  # (n,3,224,224)
        frames = frames.float() / 255.0
        if self.random_hflip and random.random() < 0.5:
            frames = torch.stack([TF.hflip(f) for f in frames])
        frames = self._normalize(frames)
        return frames.contiguous()


# 标准化图像/视频变换 (与 VideoMAE 官方一致: mean=std=0.5)
def get_video_transforms(cfg: Dict[str, Any]) -> VideoTransforms:
    tcfg = cfg.get("transforms", {})
    return VideoTransforms(
        spatial_size=tcfg.get("spatial_size", 224),
        temporal_mode=tcfg.get("temporal_sample", "uniform"),
        random_hflip=tcfg.get("random_hflip", False),
    )
