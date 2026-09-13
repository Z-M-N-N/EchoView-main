"""训练入口 —— 规格书 §8 (顶层入口)
用法:
  python train.py --config configs/default.yaml
"""
import argparse
import logging
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg, tokenizer):
    from data.dataset import MultimodalMoEDataset
    from data.collator import MoECollator
    from data.transforms import get_video_transforms
    from torch.utils.data import DataLoader

    transform = get_video_transforms(cfg["data"])
    dcfg = cfg["data"]
    ics = cfg["input"]

    kwargs = dict(
        tokenizer=tokenizer,
        max_seq_len=ics["max_seq_len"],
        num_frames=ics["num_frames"],
        num_videos_max=ics["num_videos_max"],
        frame_size=ics["frame_size"],
        video_root=dcfg.get("video_root", ""),
        preprocess_mode=dcfg.get("preprocess_mode", "video"),
        transform=transform,
    )
    train_set = MultimodalMoEDataset(dcfg["train_index"], **kwargs)
    val_set = MultimodalMoEDataset(dcfg["val_index"], **kwargs) if os.path.exists(dcfg["val_index"]) else None

    collator = MoECollator()
    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
        collate_fn=collator, num_workers=cfg["train"].get("num_workers", 0),
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set, batch_size=cfg["train"]["batch_size"], shuffle=False,
            collate_fn=collator, num_workers=cfg["train"].get("num_workers", 0),
        )
    return train_loader, val_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    from utils.config import load_config
    cfg = load_config(args.config)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 保证相对路径以项目根为基准
    set_seed(cfg["seed"])
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    from model.multimodal_moe import MultimodalMoE
    model = MultimodalMoE(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: 总 {n_params/1e6:.2f}M, 可训练 {n_train/1e6:.2f}M")

    # tokenizer (仅本地存在时加载, 绝不联网; 缺失时 dataset 使用占位 token 序列)
    mcfg = cfg["model"]["text_encoder"]
    bp = mcfg.get("backbone_path")
    tokenizer = None
    if bp and os.path.isdir(bp):
        try:
            tokenizer = AutoTokenizer.from_pretrained(bp)
        except Exception:
            tokenizer = None
    if tokenizer is None:
        logger.warning("BERT tokenizer 未加载 (权重目录不存在)。"
                       "训练将使用占位 token 序列; 请在 weights/bert-base-uncased 放置权重后重训。")

    train_loader, val_loader = build_loaders(cfg, tokenizer)

    from trainer.trainer import Trainer
    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
