"""完整模型装配: 单模态编码 -> 跨模态对齐 -> 上下文聚合 -> MoE 决策 —— 规格书 §4 / §8 model/multimodal_moe.py"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from encoders.text_encoder import TextEncoder
from encoders.video_encoder import VideoEncoder
from fusion.cross_attention import CrossModalFusion
from aggregation.perceiver_pooler import PerceiverPooler
from moe.moe_layer import MoELayer


class MultimodalMoE(nn.Module):
    """端到端多模态 MoE 路由决策模型:
        文本 + 多路视频 -> 3 类动作概率 (左转/直行/右转)
    层级:
      Layer 1: 单模态编码层 (Text/Video Encoder, 冻结 backbone + 可训练投影)
      Layer 2: 跨模态对齐层 (Cross-Attention Fusion, Text 为 Query)
      Layer 3: 上下文聚合层 (Perceiver-style 池化)
      Layer 4: MoE 决策层 (Router + Experts + Top-K 加权)
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        mcfg = cfg["model"]

        self.text_encoder = TextEncoder(
            d_model=mcfg["d_model"],
            backbone_path=mcfg["text_encoder"].get("backbone_path"),
            backbone_name=mcfg["text_encoder"].get("backbone", "bert-base-uncased"),
            freeze_backbone=mcfg["text_encoder"].get("freeze_backbone", True),
            projection_dim=mcfg["text_encoder"].get("projection_dim", mcfg["d_model"]),
        )
        self.video_encoder = VideoEncoder(
            d_model=mcfg["d_model"],
            backbone_path=mcfg["video_encoder"].get("backbone_path"),
            freeze_backbone=mcfg["video_encoder"].get("freeze_backbone", True),
            projection_dim=mcfg["video_encoder"].get("projection_dim", mcfg["d_model"]),
            num_frames=mcfg["video_encoder"].get("num_frames", 16),
        )
        self.fusion = CrossModalFusion(
            d_model=mcfg["d_model"],
            num_heads=mcfg["num_heads"],
            dropout=mcfg["dropout"],
            ffn_ratio=mcfg.get("ffn_ratio", 4),
        )
        self.aggregator = PerceiverPooler(
            d_model=mcfg["d_model"],
            d_context=mcfg.get("d_context", mcfg["d_model"]),
            num_latents=mcfg.get("num_latents", 8),
            num_heads=mcfg["num_heads"],
            dropout=mcfg["dropout"],
        )
        moe = mcfg["moe"]
        self.moe = MoELayer(
            d_context=mcfg.get("d_context", mcfg["d_model"]),
            num_experts=moe.get("num_experts", 6),
            top_k=moe.get("top_k", 2),
            router_hidden=moe.get("router_hidden", 1024),
            expert_hidden=moe.get("expert_hidden", 256),
            out_dim=cfg["input"].get("num_classes", 3),
            noise_std=moe.get("router_noise_std", 0.01),
            dropout=mcfg["dropout"],
        )

        # 梯度检查点 (VideoMAE 显存不足时)
        self.gradient_checkpointing = cfg["train"].get("gradient_checkpointing", False)
        if self.gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def _enable_gradient_checkpointing(self):
        try:
            if hasattr(self.video_encoder.backbone, "gradient_checkpointing_enable"):
                self.video_encoder.backbone.gradient_checkpointing_enable()
                print("[Model] VideoMAE gradient checkpointing enabled")
        except Exception as e:
            print(f"[Model] gradient checkpointing 启用失败: {e}")

    def encode_text(self, input_ids, attention_mask):
        return self.text_encoder(input_ids, attention_mask)

    def encode_video(self, video, video_mask):
        return self.video_encoder(video, video_mask)

    def forward(
        self,
        text_ids: torch.Tensor,       # (B, S)
        text_mask: torch.Tensor,      # (B, S)
        video: torch.Tensor,          # (B, K, T, C, H, W)
        video_mask: torch.Tensor,     # (B, K)
        use_noise: bool = True,
        force_top_k: bool = True,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Layer 1: 单模态编码
        z_text = self.text_encoder(text_ids, text_mask)                  # (B, d)
        z_video = self.video_encoder(video, video_mask)                  # (B, K, d)

        # Layer 2: 跨模态对齐 (Text 为 Query, Video 为 KV)
        f = self.fusion(z_text, z_video, video_mask)                     # (B, K, d)

        # Layer 3: 上下文聚合
        x_context = self.aggregator(f, video_mask)                       # (B, d_context)

        # Layer 4: MoE 决策
        logits, aux = self.moe(x_context, use_noise=use_noise,
                               force_top_k=force_top_k, return_aux=return_aux)   # (B, 3)

        aux["x_context"] = x_context
        return logits, aux

    def get_trainable_parameters(self):
        """返回可训练参数 (冻结 backbone 不参与)."""
        return [p for p in self.parameters() if p.requires_grad]
