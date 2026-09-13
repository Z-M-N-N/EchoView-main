"""文本编码器: 冻结 BERT-backbone + 可训练投影层 —— 规格书 §4.1 / §8 encoders/text_encoder.py"""
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig

logger = logging.getLogger(__name__)


def _new_bert_config() -> BertConfig:
    """本地构建 BERT-base-uncased 配置 (不联网, 用于随机初始化/冒烟测试)."""
    return BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=0,
    )


class TextEncoder(nn.Module):
    """Text Encoder
    - Backbone: BERT-base-uncased (冻结)
    - 输出: [CLS] 隐状态
    - 投影: Linear(768 -> d_model) 可训练
    - 输出张量: Z_text (B, d_model)
    """

    def __init__(
        self,
        d_model: int = 512,
        backbone_path: Optional[str] = None,
        backbone_name: str = "bert-base-uncased",
        freeze_backbone: bool = True,
        projection_dim: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.projection_dim = projection_dim

        # ---------- 加载 BERT ----------
        if backbone_path and os.path.isdir(backbone_path):
            logger.info(f"[TextEncoder] 从本地加载 BERT: {backbone_path}")
            self.backbone = AutoModel.from_pretrained(backbone_path)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(backbone_path)
            except Exception:
                self.tokenizer = None
        else:
            logger.warning(
                f"[TextEncoder] 未找到权重 {backbone_path!r}，使用随机初始化 "
                f"(请将 bert-base-uncased 下载到该路径后替换)"
            )
            # 纯本地构建 config, 不联网 (g5 无法访问 huggingface.co)
            cfg = _new_bert_config()
            self.backbone = AutoModel.from_config(cfg)
            self.tokenizer = None
            logger.warning("[TextEncoder] tokenizer 未加载 (无权重), 请在放置权重后重新实例化")

        # 冻结 backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.freeze_backbone = freeze_backbone

        bert_hidden = self.backbone.config.hidden_size
        logger.info(f"[TextEncoder] BERT hidden_size={bert_hidden}")

        # 可训练投影层: 768 -> d_model
        self.projection = nn.Sequential(
            nn.LayerNorm(bert_hidden),
            nn.Linear(bert_hidden, self.projection_dim),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,          # (B, S)
        attention_mask: torch.Tensor,     # (B, S)
    ) -> torch.Tensor:
        """返回 Z_text (B, d_model)."""
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(
                    input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state      # (B, S, 768)
        cls_token = last_hidden[:, 0, :]             # (B, 768)
        z_text = self.projection(cls_token)          # (B, d_model)
        return z_text
