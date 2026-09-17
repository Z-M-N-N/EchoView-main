import argparse
import json
import os
import torch
import warnings
import logging
warnings.filterwarnings("ignore")
# 抑制第三方库 info 级日志（decord/transformers 等无关噪音）
for _lg in ["transformers", "myqwen_vl_utils", "qwen_vl_utils", "datasets", "accelerate", "tokenizers"]:
    logging.getLogger(_lg).setLevel(logging.ERROR)
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import gc
import math
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from myqwen_vl_utils import process_vision_info
from dataset_utils import VideoMultiTaskDataset, get_videos_multilabels
from vision_multitask_model import VisionMultiTask
import torch.nn.functional as F

# ============ GPU信息显示 ============
def print_gpu_info():
    print("="*50)
    print("GPU 信息:")
    print("="*50)
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"可见GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\n显卡 {i}:")
            print(f"  名称: {torch.cuda.get_device_name(i)}")
            print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print("="*50)

# ============ 分布式初始化 ============
def setup_distributed(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

# ============ 多任务损失函数（修复：num_binary_tasks=28） ============
class MultiTaskLoss(nn.Module):
    def __init__(self, num_binary_tasks=28, num_reg_tasks=7, pos_weights=None):  # 27 → 28
        super().__init__()
        self.num_binary_tasks = num_binary_tasks
        self.num_reg_tasks = num_reg_tasks
        self.log_vars = nn.Parameter(torch.zeros(2))
        
        if pos_weights is not None:
            self.register_buffer('pos_weights', torch.tensor(pos_weights, dtype=torch.float32))
        else:
            self.register_buffer('pos_weights', torch.ones(num_binary_tasks))
    
    def forward(self, binary_logits, regression_values, binary_labels, regression_labels):
        device = binary_logits.device
        if self.pos_weights is not None:
            pos_weight = self.pos_weights.to(device).unsqueeze(0).expand_as(binary_labels)
            loss_per_sample = -(
                pos_weight * binary_labels * F.logsigmoid(binary_logits) + 
                (1 - binary_labels) * F.logsigmoid(-binary_logits)
            )
            binary_loss = loss_per_sample.mean()
        else:
            binary_loss = nn.functional.binary_cross_entropy_with_logits(
                binary_logits, binary_labels
            )
        
        regression_loss = nn.functional.mse_loss(regression_values, regression_labels)
        
        precision_binary = torch.exp(-self.log_vars[0])
        precision_reg = torch.exp(-self.log_vars[1])
        
        total_loss = (precision_binary * binary_loss + self.log_vars[0] + 
                     precision_reg * regression_loss + self.log_vars[1])
        # total_loss=(precision_binary * binary_loss + self.log_vars[0])
        return {
            'total_loss': total_loss,
            'binary_loss': binary_loss,
            'regression_loss': regression_loss,
            'binary_weight': precision_binary.detach().item(),
            'reg_weight': precision_reg.detach().item()
        }
    
# ============ 训练器（支持DDP，修复警告） ============
class MultiTaskTrainerDDP:
    def __init__(self, model, device, rank, is_distributed=False, gradient_accumulation_steps=4):
        self.model = model.to(device)
        self.device = device
        self.rank = rank
        self.is_distributed = is_distributed
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._global_step = 0
        
        if is_distributed:
            # 修复：设置 find_unused_parameters=False 以提高性能
            self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=False)
    
    def train_epoch(self, dataloader, optimizer, criterion, scheduler=None):
        self.model.train()
        total_loss = 0
        all_binary_preds = []
        all_binary_labels = []
        all_reg_preds = []
        all_reg_labels = []
        
        optimizer.zero_grad()
        show_progress = (not self.is_distributed) or (self.rank == 0)
        progress_bar = tqdm(dataloader, desc="Training", disable=not show_progress)
        
        for batch_idx, batch in enumerate(progress_bar):
            pixel_values = batch["pixel_values_videos"].to(self.device, non_blocking=True)
            grid_thw = batch["video_grid_thw"].to(self.device, non_blocking=True)
            binary_labels = batch["binary_labels"].to(self.device, non_blocking=True)
            regression_labels = batch["regression_labels"].to(self.device, non_blocking=True)
            
            binary_logits, regression_values = self.model(pixel_values, grid_thw)
            
            losses = criterion(binary_logits, regression_values, binary_labels, regression_labels)
            loss = losses['total_loss']
            
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                self._global_step += 1
                if scheduler is not None:
                    scheduler.step()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            binary_preds = torch.sigmoid(binary_logits) > 0.5
            all_binary_preds.extend(binary_preds.cpu().numpy())
            all_binary_labels.extend(binary_labels.cpu().numpy())
            all_reg_preds.extend(regression_values.detach().cpu().numpy())
            all_reg_labels.extend(regression_labels.cpu().numpy())
            
            if show_progress and len(all_binary_labels) > 0:
                binary_acc = (binary_preds == binary_labels).float().mean().item()
                reg_mae = torch.abs(regression_values - regression_labels).mean().item()
                progress_bar.set_postfix({
                    'loss': f'{total_loss / (batch_idx + 1):.4f}',
                    'b_acc': f'{binary_acc:.4f}',
                    'r_mae': f'{reg_mae:.4f}'
                })
        
        if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            self._global_step += 1
            if scheduler is not None:
                scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        
        binary_labels_array = np.array(all_binary_labels)
        binary_preds_array = np.array(all_binary_preds)
        binary_acc = (binary_preds_array == binary_labels_array).mean()
        
        reg_preds_array = np.array(all_reg_preds)
        reg_labels_array = np.array(all_reg_labels)
        reg_mae = np.abs(reg_preds_array - reg_labels_array).mean()
        reg_rmse = np.sqrt(((reg_preds_array - reg_labels_array) ** 2).mean())
        
        return avg_loss, binary_acc, reg_mae, reg_rmse
    
    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        all_binary_preds = []
        all_binary_labels = []
        all_reg_preds = []
        all_reg_labels = []
        
        show_progress = (not self.is_distributed) or (self.rank == 0)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", disable=not show_progress):
                pixel_values = batch["pixel_values_videos"].to(self.device)
                grid_thw = batch["video_grid_thw"].to(self.device)
                binary_labels = batch["binary_labels"].to(self.device)
                regression_labels = batch["regression_labels"].to(self.device)
                
                binary_logits, regression_values = self.model(pixel_values, grid_thw)
                losses = criterion(binary_logits, regression_values, binary_labels, regression_labels)
                
                total_loss += losses['total_loss'].item()
                
                binary_preds = torch.sigmoid(binary_logits) > 0.5
                all_binary_preds.extend(binary_preds.cpu().numpy())
                all_binary_labels.extend(binary_labels.cpu().numpy())
                all_reg_preds.extend(regression_values.cpu().numpy())
                all_reg_labels.extend(regression_labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        binary_labels_array = np.array(all_binary_labels)
        binary_preds_array = np.array(all_binary_preds)
        binary_acc = (binary_preds_array == binary_labels_array).mean()
        
        reg_preds_array = np.array(all_reg_preds)
        reg_labels_array = np.array(all_reg_labels)
        reg_mae = np.abs(reg_preds_array - reg_labels_array).mean()
        reg_rmse = np.sqrt(((reg_preds_array - reg_labels_array) ** 2).mean())
        
        return avg_loss, binary_acc, reg_mae, reg_rmse

# ============ 辅助函数 ============
def compute_pos_weights(binary_labels, num_tasks):
    """计算每个任务的 pos_weight = 负样本数 / 正样本数"""
    labels_array = np.array(binary_labels)
    pos_counts = labels_array.sum(axis=0)
    neg_counts = len(labels_array) - pos_counts
    pos_weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    return pos_weights

# ============ 分布式训练函数 ============
def train_multitask_model_distributed(rank, world_size, train_json, model_id,
                                      num_epochs=10, unfreeze_layers=0, weight_output="pth",batch_size=64,
                                      resume_from=None,
                                      use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.0,
                                      lora_target='attn', temporal_pool='mean',
                                      head_lr=1e-4, visual_lr=None, warmup_ratio=0.03):
    """分布式训练主函数"""
    
    setup_distributed(rank, world_size)
    
    if rank == 0:
        print("=" * 60)
        print(f"Training Multi-Task Model (Distributed)")
        print(f"World Size: {world_size}, Epochs: {num_epochs}")
        print("=" * 60)
        print_gpu_info()
    
    # 加载模型
    if rank == 0:
        print("\nLoading Qwen2.5-VL model...")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True
    )
    vision_encoder = model.visual
    
    classifier_model = VisionMultiTask(
        vision_encoder,
        num_binary_tasks=28,  # 注意这里是28
        reg_tasks=7,
        unfreeze_layers=unfreeze_layers,
        unfreeze_merger=False,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target=lora_target,
        temporal_pool=temporal_pool
    )
    
    if rank == 0:
        print(f"Model created with hidden size: {classifier_model.hidden_size}")
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    # 加载数据
    if rank == 0:
        print("\nLoading data...")
    video_paths, binary_labels, regression_labels = get_videos_multilabels(train_json)
    
    if len(video_paths) == 0:
        if rank == 0:
            print("No valid videos found!")
        cleanup_distributed()
        return
    
    # 划分数据集
    train_size = int(0.8 * len(video_paths))
    train_paths = video_paths[:train_size]
    train_binary = binary_labels[:train_size]
    train_regression = regression_labels[:train_size]
    val_paths = video_paths[train_size:]
    val_binary = binary_labels[train_size:]
    val_regression = regression_labels[train_size:]
    
    if rank == 0:
        print(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")
    
    # 创建数据集
    train_dataset = VideoMultiTaskDataset(
        train_paths, train_binary, train_regression, 
        processor, num_frames=8*2
    )
    val_dataset = VideoMultiTaskDataset(
        val_paths, val_binary, val_regression,
        processor, num_frames=8*2
    )
    
    # 创建分布式DataLoader
    BATCH_SIZE = batch_size
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # ============ 优化器：分层学习率（视觉编码器小 LR 防破坏预训练特征，头部大 LR） ============
    if visual_lr is None:
        visual_lr = 1e-4 if use_lora else 1e-5   # LoRA 适配器可用稍大 LR；全量解冻用小 LR
    named_params = list(classifier_model.named_parameters())
    head_grad = [p for n, p in named_params if p.requires_grad and not n.startswith('visual.')]
    visual_grad = [p for n, p in named_params if p.requires_grad and n.startswith('visual.')]
    if not visual_grad:
        raise RuntimeError("没有可训练的视觉参数：请设置 --unfreeze_layers>0 或 --use_lora 1")
    param_groups = [
        {'params': head_grad, 'lr': head_lr},
        {'params': visual_grad, 'lr': visual_lr},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-5)
    if rank == 0:
        print(f"⚙️ 优化器：head_lr={head_lr}, visual_lr={visual_lr}，"
              f"可训练 visual 参数={sum(p.numel() for p in visual_grad):,}，head 参数={sum(p.numel() for p in head_grad):,}")

    # ============ 调度器：warmup + 余弦退火（按 step 更新） ============
    total_steps = num_epochs * max(1, len(train_loader))
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-4, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    # ============ 续训：从已保存的 best 权重继续（2026-08-25 添加） ============
    start_epoch = 0
    if resume_from:
        ckpt = torch.load(resume_from, map_location=f'cuda:{rank}', weights_only=False)
        # LoRA 模式下 checkpoint 是"已合并"权重（不含 lora_A/lora_B），用 strict=False 恢复：
        # 基础权重从合并点继续，LoRA 适配器重新从零初始化，等价于"把 LoRA 结果固化后继续训"。
        classifier_model.load_state_dict(ckpt['model_state_dict'], strict=(not use_lora))
        start_epoch = int(ckpt['epoch']) + 1
        if rank == 0:
            print(f"🔄 从 {resume_from} 续训：epoch {start_epoch+1}/{num_epochs}，"
                  f"上次 val_binary_acc={ckpt.get('val_binary_acc', '?')}" +
                  ("（LoRA 适配器将重新初始化）" if use_lora else ""))
    # =======================================================================

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambda,
        last_epoch=(start_epoch * len(train_loader) - 1) if resume_from else -1
    )
    if rank == 0:
        print(f"📉 调度器：warmup_ratio={warmup_ratio}，前 {warmup_steps}/{total_steps} 步线性升温，之后余弦退火")

    # ============ 修复：使用正确的任务数量 28 ============
    if rank == 0:
        pos_weights = compute_pos_weights(train_binary, num_tasks=28)  # 27 → 28
        print(f"Positive weights: {pos_weights}")
        pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(f"cuda:{rank}")
    else:
        pos_weights_tensor = torch.zeros(28, dtype=torch.float32).to(f"cuda:{rank}")  # 27 → 28
    
    dist.broadcast(pos_weights_tensor, src=0)
    pos_weights = pos_weights_tensor.cpu().numpy()
    
    # ============ 修复：使用正确的任务数量 28 ============
    criterion = MultiTaskLoss(num_binary_tasks=28, num_reg_tasks=7, pos_weights=pos_weights)  # 27 → 28
    
    # 训练器
    trainer = MultiTaskTrainerDDP(
        classifier_model, 
        device=f"cuda:{rank}",
        rank=rank,
        is_distributed=True,
        gradient_accumulation_steps=4
    )
    
    if rank == 0:
        os.makedirs(weight_output, exist_ok=True)
    
    best_val_acc = 0
    best_val_mae = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
        
        train_loss, train_binary_acc, train_reg_mae, train_reg_rmse = trainer.train_epoch(
            train_loader, optimizer, criterion, scheduler
        )
        
        if rank == 0:
            print(f"Train - Loss: {train_loss:.4f}, Binary Acc: {train_binary_acc:.4f}, "
                  f"Reg MAE: {train_reg_mae:.4f}, Reg RMSE: {train_reg_rmse:.4f}")
        
        val_loss, val_binary_acc, val_reg_mae, val_reg_rmse = trainer.evaluate(
            val_loader, criterion
        )
        
        if rank == 0:
            print(f"Val   - Loss: {val_loss:.4f}, Binary Acc: {val_binary_acc:.4f}, "
                  f"Reg MAE: {val_reg_mae:.4f}, Reg RMSE: {val_reg_rmse:.4f}")
        
        if rank == 0:
            model_to_save = trainer.model.module if trainer.is_distributed else trainer.model
            
            if val_binary_acc > best_val_acc:
                best_val_acc = val_binary_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.get_merged_state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_binary_acc': val_binary_acc,
                    'val_reg_mae': val_reg_mae,
                    'val_reg_rmse': val_reg_rmse,
                }, f'{weight_output}/best_multitask_model.pth')
                print(f"✓ Saved best model (binary_acc: {val_binary_acc:.4f})")
            
            if val_reg_mae < best_val_mae:
                best_val_mae = val_reg_mae
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'val_binary_acc': val_binary_acc,
                    'val_reg_mae': val_reg_mae,
                }, f'{weight_output}/best_regression_model.pth')
                print(f"✓ Saved best regression model (MAE: {val_reg_mae:.4f})")
        
        torch.cuda.empty_cache()
        gc.collect()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best Binary Accuracy: {best_val_acc:.4f}")
        print(f"Best Regression MAE: {best_val_mae:.4f}")
        print(f"{'='*60}")
    
    cleanup_distributed()

# ============ 主函数 ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='多任务视频理解训练（分布式）')
    
    parser.add_argument('--train_json', type=str, required=True, help='训练数据json文件路径')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=10, help='训练轮数')
    
    parser.add_argument('--unfreeze_layers', type=int, default=0, help='解冻视觉编码器层数')
    parser.add_argument('--base_model', type=str, required=True, help='Qwen2.5-VL模型路径')
    parser.add_argument('--weight_output', type=str, default='./checkpoints', help='模型保存路径')
    parser.add_argument('--resume_from', type=str, default=None, help='从 .pth checkpoint 续训')
    parser.add_argument('--lr', type=float, default=1e-4, help='分类/回归头部学习率')
    parser.add_argument('--visual_lr', type=float, default=None, help='视觉编码器(含LoRA)学习率；默认: LoRA=1e-4, 解冻=1e-5')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='warmup 步数占总步数比例')
    parser.add_argument('--use_lora', type=int, default=0, help='是否对视觉编码器注入LoRA (1=是, 0=否)')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA 秩')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout')
    parser.add_argument('--lora_target', type=str, default='attn', help="LoRA 目标: attn / mlp / attn+mlp")
    parser.add_argument('--temporal_pool', type=str, default='mean', help="时序池化: mean / attention")

    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    
    print("=" * 60)
    print("Multi-Task Training Configuration (Distributed):")
    print(f"  Train JSON: {args.train_json}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  BATCH_SIZE: {args.batch_size}")
    print(f"  Unfreeze Layers: {args.unfreeze_layers}")
    print(f"  Use LoRA: {args.use_lora} (r={args.lora_r}, alpha={args.lora_alpha}, target={args.lora_target})")
    print(f"  Temporal Pool: {args.temporal_pool}")
    print(f"  head lr={args.lr}, visual lr={args.visual_lr}, warmup_ratio={args.warmup_ratio}")
    print(f"  Model: {args.base_model}")
    print(f"  GPUs: {world_size} (controlled by CUDA_VISIBLE_DEVICES)")
    print("=" * 60)
    
    if world_size == 0:
        print("❌ No GPUs available!")
        exit(1)
    
    mp.spawn(
        train_multitask_model_distributed,
        args=(world_size, args.train_json, args.base_model,
              args.num_epochs, args.unfreeze_layers, args.weight_output, args.batch_size,
              args.resume_from, args.use_lora, args.lora_r, args.lora_alpha, args.lora_dropout,
              args.lora_target, args.temporal_pool, args.lr, args.visual_lr, args.warmup_ratio),
        nprocs=world_size,
        join=True
    )

