"""训练循环: 梯度累积、AMP、Checkpoint、评估 —— 规格书 §6 / §8 trainer/trainer.py"""
import datetime
import logging
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_optimizer(model: nn.Module, cfg: Dict):
    """AdamW + 按参数组设置学习率 (主干低学习率, 其余 1e-4)."""
    ocfg = cfg["optim"]
    backbone_lr = ocfg.get("backbone_lr", ocfg["lr"])
    trainable = model.get_trainable_parameters()

    # 以 id() 判断参数归属 (避免张量 == 产生布尔歧义)
    backbone_ids = {id(p) for p in model.text_encoder.backbone.parameters()}
    backbone_ids |= {id(p) for p in model.video_encoder.backbone.parameters()}

    backbone_params, other_params = [], []
    for p in trainable:
        if id(p) in backbone_ids:
            backbone_params.append(p)
        else:
            other_params.append(p)

    param_groups = [
        {"params": other_params, "lr": ocfg["lr"]},
    ]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    return torch.optim.AdamW(param_groups, lr=ocfg["lr"], weight_decay=ocfg.get("weight_decay", 1e-5))


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, tokenizer=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = cfg["device"]
        self.model.to(self.device)

        self.optimizer = build_optimizer(model, cfg)
        self.scaler = GradScaler("cuda", enabled=cfg["train"].get("amp", True))
        self.accum_steps = cfg["train"].get("accumulation_steps", 1)
        self.max_grad_norm = cfg["train"].get("max_grad_norm", 1.0)
        self.epochs = cfg["train"].get("epochs", 50)
        self.log_interval = cfg["train"].get("log_interval", 20)
        self.output_dir = cfg.get("output_dir", "./outputs")
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)

        from losses.moe_losses import build_loss
        self.criterion = build_loss(cfg)

        # Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2) + warmup
        ocfg = cfg["optim"]
        steps_per_epoch = len(train_loader)
        t0_steps = ocfg.get("t_0", 10) * steps_per_epoch
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=max(t0_steps, 1), T_mult=ocfg.get("t_mult", 2),
            eta_min=ocfg.get("eta_min", 1e-6),
        )
        self.warmup_steps = ocfg.get("warmup_steps", 1000)
        self.global_step = 0

    def _to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss, total_cls, total_load, total_imp = 0.0, 0.0, 0.0, 0.0
        total_cost = 0.0
        n_batches = 0
        t0 = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)
            use_noise = self.cfg["model"]["moe"].get("use_noise", True)

            with autocast("cuda", enabled=self.cfg["train"].get("amp", True)):
                logits, aux = self.model(
                    batch["text_ids"], batch["text_mask"],
                    batch["video"], batch["video_mask"],
                    use_noise=use_noise, force_top_k=True,
                )
                loss, parts = self.criterion(
                    logits, batch["label"], aux, return_parts=True)

            self.scaler.scale(loss).backward()

            if (i + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            self.global_step += 1
            # warmup 手动线性上升 (前 warmup_steps 步从 0.1*lr -> lr)
            if self.global_step <= self.warmup_steps:
                frac = 0.1 + 0.9 * (self.global_step / max(self.warmup_steps, 1))
                for g in self.optimizer.param_groups:
                    g["lr"] = g.get("initial_lr", self.cfg["optim"]["lr"]) * frac
            else:
                self.scheduler.step()

            total_loss += loss.item()
            total_cls += parts["cls"].item()
            total_load += parts.get("load", torch.tensor(0.0)).item() if isinstance(parts.get("load", 0.0), torch.Tensor) else parts.get("load", 0.0)
            total_imp += parts.get("importance", 0.0) if isinstance(parts.get("importance", 0.0), float) else (parts.get("importance", torch.tensor(0.0)).item() if torch.is_tensor(parts.get("importance")) else 0.0)
            total_cost += parts.get("cost", 0.0) if isinstance(parts.get("cost", 0.0), float) else (parts.get("cost", torch.tensor(0.0)).item() if torch.is_tensor(parts.get("cost")) else 0.0)
            n_batches += 1

            if (i + 1) % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                extra = ""
                if "cost" in parts:
                    extra = f" cost={parts['cost'].item():.4f}"
                else:
                    extra = f" load={parts.get('load', torch.tensor(0.0)).item():.4f} imp={parts.get('importance', torch.tensor(0.0)).item():.4f}"
                logger.info(
                    f"[E{epoch}] step {i+1}/{len(self.train_loader)} "
                    f"loss={loss.item():.4f} cls={parts['cls'].item():.3f}{extra} "
                    f"lr={lr:.2e} ({time.time()-t0:.0f}s)"
                )

        return {
            "loss": total_loss / max(n_batches, 1),
            "cls": total_cls / max(n_batches, 1),
            "load": total_load / max(n_batches, 1),
            "importance": total_imp / max(n_batches, 1),
            "cost": total_cost / max(n_batches, 1),
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        correct = 0
        total = 0
        losses = []
        from collections import Counter
        expert_counter = Counter()

        for batch in self.val_loader:
            batch = self._to_device(batch)
            logits, aux = self.model(
                batch["text_ids"], batch["text_mask"],
                batch["video"], batch["video_mask"],
                use_noise=False, force_top_k=True,
            )
            loss = self.criterion(logits, batch["label"], aux)
            losses.append(loss.item())
            preds = logits.argmax(dim=-1)
            correct += (preds == batch["label"]).sum().item()
            total += batch["label"].size(0)
            if "top1_expert" in aux:
                expert_counter.update(aux["top1_expert"].tolist())

        acc = correct / max(total, 1)
        return {
            "val_loss": float(np.mean(losses)) if losses else 0.0,
            "val_acc": acc,
            "expert_usage": dict(expert_counter),
        }

    def save_checkpoint(self, epoch: int, tag: str = "last"):
        path = os.path.join(self.output_dir, "checkpoints", f"{tag}.pt")
        torch.save({
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "cfg": self.cfg,
        }, path)
        logger.info(f"[Trainer] checkpoint saved: {path}")
        return path

    def fit(self):
        best_acc = 0.0
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            extra = f"cost={train_metrics['cost']:.4f}" if train_metrics.get("cost") is not None else (
                f"load={train_metrics['load']:.4f} imp={train_metrics['importance']:.4f}")
            logger.info(f"[E{epoch}] TRAIN loss={train_metrics['loss']:.4f} "
                        f"cls={train_metrics['cls']:.3f} {extra}")

            if epoch % self.cfg["train"].get("eval_interval", 1) == 0 and self.val_loader is not None:
                val_metrics = self.evaluate()
                logger.info(f"[E{epoch}] VAL loss={val_metrics['val_loss']:.4f} "
                            f"acc={val_metrics['val_acc']:.4f} "
                            f"expert_usage={val_metrics['expert_usage']}")
                if val_metrics["val_acc"] >= best_acc:
                    best_acc = val_metrics["val_acc"]
                    self.save_checkpoint(epoch, tag="best")

            if epoch % self.cfg["train"].get("save_interval", 1) == 0:
                self.save_checkpoint(epoch, tag="last")

        self.save_checkpoint(self.epochs, tag="final")
        logger.info(f"[Trainer] 训练完成, best val_acc={best_acc:.4f}")
        return best_acc
