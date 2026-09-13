"""视频编码器: 冻结 VideoMAE-v2-Base + 可训练投影层 —— 规格书 §4.2 / §8 encoders/video_encoder.py"""
import importlib
import logging
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# 本地 custom VideoMAEv2 的 modeling 模块目录 (若用本地权重则注册导入路径)
_LOCAL_VMAE_DIR = "/home/mzhao/暂时不用/对比模型/video-echo-clip-main/VideoMAEv2-Base"


class VideoEncoder(nn.Module):
    """Video Encoder (按规格书):
    - Backbone: VideoMAE-v2-Base (冻结)
    - 采样: 16 帧均匀采样, 224x224
    - 输出: 全局平均池化视频级特征
    - 投影: Linear(768 -> d_model) 可训练
    - 输出张量: Z_video (B, K, d_model)
    说明: 每路视频独立编码 (在此模块对 [B*K] 批量处理), 参数共享.
    """

    def __init__(
        self,
        d_model: int = 512,
        backbone_path: Optional[str] = None,
        freeze_backbone: bool = True,
        projection_dim: int = 512,
        num_frames: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.projection_dim = projection_dim
        self.num_frames = num_frames

        # ---------- 解析 VideoMAE 权重路径 ----------
        # 优先使用显式指定路径；未指定时尝试本地默认路径
        if not backbone_path or not os.path.isdir(backbone_path):
            if os.path.isdir(_LOCAL_VMAE_DIR):
                backbone_path = _LOCAL_VMAE_DIR
                logger.info(f"[VideoEncoder] 使用本地 VideoMAEv2: {backbone_path}")

        if backbone_path and os.path.isdir(backbone_path):
            self._register_local_module(backbone_path)
            logger.info(f"[VideoEncoder] 从本地加载 VideoMAE-v2: {backbone_path}")
            self.backbone = self._load_local_vmae(backbone_path)
        else:
            logger.warning(
                f"[VideoEncoder] 未找到 VideoMAE 权重 {backbone_path!r}，使用随机初始化 "
                f"(请自行下载权重后指定 backbone_path)"
            )
            self.backbone = self._random_init_vmae()

        # 冻结 backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.freeze_backbone = freeze_backbone

        # 确定 backbone 输出维度
        vid_hidden = getattr(self.backbone.config, "hidden_size", 768)
        embed_dim = getattr(
            getattr(self.backbone.config, "model_config", None), "embed_dim", vid_hidden
        ) if hasattr(self.backbone.config, "model_config") else vid_hidden
        self.backbone_hidden = int(embed_dim)
        logger.info(f"[VideoEncoder] VideoMAE 输出维度: {self.backbone_hidden}")

        # 可训练投影层: 768 -> d_model
        self.projection = nn.Sequential(
            nn.LayerNorm(self.backbone_hidden),
            nn.Linear(self.backbone_hidden, self.projection_dim),
            nn.Dropout(0.1),
        )

    # ---------- 加载工具 ----------
    @staticmethod
    def _register_local_module(path: str):
        if path not in sys.path:
            sys.path.insert(0, path)

    def _load_local_vmae(self, path: str):
        """加载本地 custom VideoMAEv2 (auto_map 指向 modeling_videomaev2.VideoMAEv2)."""
        from transformers import AutoModel
        model = AutoModel.from_pretrained(path, trust_remote_code=True)
        return model

    def _random_init_vmae(self):
        """无权重时随机初始化一个 VideoMAE (仅框架冒烟测试用)."""
        try:
            from transformers import AutoConfig, AutoModel
            cfg = AutoConfig.from_pretrained(_LOCAL_VMAE_DIR, trust_remote_code=True)
            return AutoModel.from_config(cfg)
        except Exception as e:
            logger.warning(f"[VideoEncoder] 随机初始化失败({e})，退回占位模块")
            return _DummyVideoBackbone()

    # ---------- 前向 ----------
    def forward(
        self,
        video: torch.Tensor,      # (B, K, T, C, H, W) float32, K<=K_max
        video_mask: torch.Tensor, # (B, K) bool/long
    ) -> torch.Tensor:
        """返回 Z_video (B, K, d_model)."""
        B, K = video.shape[0], video.shape[1]
        # 合并 B*K 批量
        v = video.view(B * K, *video.shape[2:])   # (B*K, T, C, H, W)

        # 实际存在的路 (mask=1) 才需要编码; 缺失路保持全零特征
        valid = video_mask.view(-1).bool()        # (B*K,)
        z = torch.zeros(B * K, self.backbone_hidden, device=video.device)

        if valid.any():
            vv = v[valid]                        # (n, T, C, H, W)
            # VideoMAE 期望 (n, C, T, H, W)
            vv = vv.permute(0, 2, 1, 3, 4).contiguous()
            if self.freeze_backbone:
                with torch.no_grad():
                    feats = self.backbone(vv)     # (n, 768)
            else:
                feats = self.backbone(vv)
            # 可能是 (n,768) 或 (n,d,768) 形态, 统一池化到 (n,768)
            if feats.dim() == 3:
                feats = feats.mean(dim=1)
            z[valid] = feats

        z = self.projection(z)                    # (B*K, d_model)
        z = z.view(B, K, self.projection_dim)     # (B, K, d_model)
        # 缺失路特征置零 (避免投影层给随机输入产生非零特征)
        z = z * video_mask.unsqueeze(-1).float()
        return z


class _DummyVideoBackbone(nn.Module):
    """无权重时的兜底占位 backbone (仅保证维度正确)."""
    def __init__(self):
        super().__init__()
        self.config = type("C", (), {"hidden_size": 768})()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.head = nn.Linear(64, 768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, T, C, H, W) -> (n, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        f = self.encoder(x).flatten(1)
        return self.head(f)
