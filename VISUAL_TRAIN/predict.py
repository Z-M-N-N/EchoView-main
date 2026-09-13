import argparse
import json
import os
# def setup_cuda(cuda_num=0):
#     """设置CUDA环境"""
#     print(f"Setting CUDA_VISIBLE_DEVICES={cuda_num}")
#     os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_num}"
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#     os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# setup_cuda(cuda_num=2)

import torch
import warnings
import logging
warnings.filterwarnings("ignore")
# 抑制第三方库 info 级日志（decord/transformers 等无关噪音）
for _lg in ["transformers", "myqwen_vl_utils", "qwen_vl_utils", "datasets", "accelerate", "tokenizers"]:
    logging.getLogger(_lg).setLevel(logging.ERROR)
print("="*50)
print("GPU 信息:")
print("="*50)

# 检查CUDA是否可用
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 当前设备信息
    current_device = torch.cuda.current_device()
    print(f"当前设备索引: {current_device}")
    print(f"当前设备名称: {torch.cuda.get_device_name(current_device)}")
    print(f"可见GPU数量: {torch.cuda.device_count()}")
    
    # 显示所有可见显卡
    for i in range(torch.cuda.device_count()):
        print(f"\n显卡 {i}:")
        print(f"  名称: {torch.cuda.get_device_name(i)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # 当前显存使用情况
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  已分配显存: {allocated:.2f} GB")
        print(f"  保留显存: {reserved:.2f} GB")
    
    print("="*50)
else:
    print("警告: CUDA不可用，将使用CPU训练")

import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    precision_score,
    recall_score, 
    f1_score, 
    roc_auc_score, 
    mean_squared_error, 
    mean_absolute_error
)
from scipy.stats import pearsonr

import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from myqwen_vl_utils import process_vision_info

from dataset_utils import VideoMultiTaskDataset, get_videos_multilabels
from vision_multitask_model import VisionMultiTask


def load_model(model_id, checkpoint_path, unfreeze_layers=0, device="cuda"):
    """加载训练好的多任务模型"""
    print(f"\nLoading Qwen2.5-VL model from {model_id}...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True
    )
    vision_encoder = base_model.visual
    
    # 创建多任务模型
    model = VisionMultiTask(
        vision_encoder,
        num_binary_tasks=28,
        reg_tasks=7,
        unfreeze_layers=unfreeze_layers,
        unfreeze_merger=False
    )
    
    # 加载训练好的权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    model = model.to(device)
    model.eval()
    
    return model


class MultiTaskEvaluator:
    """多任务评估器"""
    
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        
    def evaluate(self, dataloader, num_binary_tasks=28, num_reg_tasks=7, save_predictions=True, max_batches=None):
        """
        全面评估多任务模型

        Args:
            dataloader: 数据加载器
            num_binary_tasks: 二分类任务数量
            num_reg_tasks: 回归任务数量
            save_predictions: 是否保存详细的预测结果
            max_batches: 最大评估批次数（None=评估全部）

        Returns:
            dict: 包含所有评估指标的字典
        """
        self.model.eval()
        
        # 初始化存储
        all_binary_logits = []
        all_binary_labels = []
        all_reg_preds = []
        all_reg_labels = []
        
        # 新增：记录输入输出详情
        all_inputs = []  # 记录视频路径或ID
        all_outputs = []  # 记录模型输出详情
        
        # 收集所有预测和标签
        with torch.no_grad():
            c=0
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                c+=1
                if max_batches is not None and c > max_batches: break
                pixel_values = batch["pixel_values_videos"].to(self.device)
                grid_thw = batch["video_grid_thw"].to(self.device)
                binary_labels = batch["binary_labels"].to(self.device)
                regression_labels = batch["regression_labels"].to(self.device)
                
                binary_logits, regression_values = self.model(pixel_values, grid_thw)
                
                all_binary_logits.append(binary_logits.cpu().numpy())
                all_binary_labels.append(binary_labels.cpu().numpy())
                all_reg_preds.append(regression_values.cpu().numpy())
                all_reg_labels.append(regression_labels.cpu().numpy())
                
                # ========== 记录输入输出详情 ==========
                if save_predictions:
                    # 获取视频ID或路径（如果有）
                    video_ids = batch.get("video_id", [f"sample_{batch_idx}_{i}" for i in range(len(pixel_values))])
                    if isinstance(video_ids, torch.Tensor):
                        video_ids = video_ids.cpu().numpy().tolist()
                    elif isinstance(video_ids, str):
                        video_ids = [video_ids]
                    
                    # 收集每个样本的预测详情
                    for i in range(len(pixel_values)):
                        # 二分类结果
                        binary_probs_i = 1 / (1 + np.exp(-binary_logits[i].cpu().numpy()))
                        binary_preds_i = (binary_probs_i > 0.5).astype(np.float32)
                        
                        # 回归结果
                        reg_preds_i = regression_values[i].cpu().numpy()
                        
                        # 构建输出记录
                        output_record = {
                            "batch_idx": batch_idx,
                            "sample_idx": i,
                            "video_id": video_ids[i] if i < len(video_ids) else f"unknown_{batch_idx}_{i}",
                            "input": {
                                "grid_thw": grid_thw[i].cpu().numpy().tolist() if torch.is_tensor(grid_thw) else grid_thw[i],
                            },
                            "output": {
                                "binary": {
                                    "logits": binary_logits[i].cpu().numpy().tolist(),
                                    "probs": binary_probs_i.tolist(),
                                    "predictions": binary_preds_i.tolist(),
                                    "labels": binary_labels[i].cpu().numpy().tolist(),
                                },
                                "regression": {
                                    "predictions": reg_preds_i.tolist(),
                                    "labels": regression_labels[i].cpu().numpy().tolist(),
                                }
                            }
                        }
                        
                        # 如果有视频路径，也记录下来
                        if "video_path" in batch:
                            video_paths = batch["video_path"]
                            if isinstance(video_paths, list) and i < len(video_paths):
                                output_record["input"]["video_path"] = video_paths[i]
                            elif isinstance(video_paths, str):
                                output_record["input"]["video_path"] = video_paths
                        
                        all_outputs.append(output_record)
                        all_inputs.append({
                            "video_id": video_ids[i] if i < len(video_ids) else f"unknown_{batch_idx}_{i}",
                            "grid_thw": grid_thw[i].cpu().numpy().tolist() if torch.is_tensor(grid_thw) else grid_thw[i],
                            "video_path": batch.get("video_path", [""])[i] if "video_path" in batch else None
                        })
        
        # 转换为numpy数组
        binary_logits = np.concatenate(all_binary_logits, axis=0)  # [N, 28]
        binary_labels = np.concatenate(all_binary_labels, axis=0)  # [N, 28]
        reg_preds = np.concatenate(all_reg_preds, axis=0)          # [N, 7]
        reg_labels = np.concatenate(all_reg_labels, axis=0)        # [N, 7]
        
        # 计算二分类概率
        binary_probs = 1 / (1 + np.exp(-binary_logits))
        binary_preds = (binary_probs > 0.5).astype(np.float32)
        
        results = {}
        
        # ========== 1. 每个二分类头的详细评估 ==========
        results['binary'] = self._evaluate_binary_heads(
            binary_labels, binary_preds, binary_probs, num_binary_tasks
        )
        
        # ========== 2. 每个回归头的详细评估 ==========
        results['regression'] = self._evaluate_regression_heads(
            reg_labels, reg_preds, num_reg_tasks
        )
        
        # ========== 3. 总体指标 ==========
        results['overall'] = self._compute_overall_metrics(
            binary_labels, binary_preds, binary_probs,
            reg_labels, reg_preds
        )
        
        # ========== 4. 原始数组（始终保存，供分片合并重算指标） ==========
        results['predictions'] = {
            'binary_logits': binary_logits,
            'binary_probs': binary_probs,
            'binary_preds': binary_preds,
            'binary_labels': binary_labels,
            'reg_preds': reg_preds,
            'reg_labels': reg_labels,
        }
        # ========== 5. 逐样本预测详情（体积大，仅 save_predictions=True 时保存） ==========
        if save_predictions:
            results['predictions']['detailed'] = all_outputs
            results['predictions']['inputs'] = all_inputs

        return results
    def _evaluate_binary_heads(self, labels, preds, probs, num_tasks):
        """评估每个二分类头"""
        results = []
        
        for task_id in range(num_tasks):
            # 获取当前任务的标签和预测
            task_labels = labels[:, task_id]
            task_preds = preds[:, task_id]
            task_probs = probs[:, task_id]
            
            # 检查是否有正样本
            n_positive = task_labels.sum()
            n_negative = len(task_labels) - n_positive
            
            task_result = {
                'task_id': task_id,
                'n_samples': len(task_labels),
                'n_positive': int(n_positive),
                'n_negative': int(n_negative),
            }
            
            # 如果所有样本都是同一类，AUC无法计算
            if n_positive == 0 or n_negative == 0:
                task_result['accuracy'] = accuracy_score(task_labels, task_preds)
                task_result['precision'] = 0.0
                task_result['recall'] = 0.0
                task_result['f1'] = 0.0
                task_result['auc'] = 0.0
                task_result['warning'] = 'No positive or no negative samples'
            else:
                # Accuracy
                task_result['accuracy'] = accuracy_score(task_labels, task_preds)
                
                # Precision (精确率)
                task_result['precision'] = precision_score(task_labels, task_preds, pos_label=1, zero_division=0)
                
                # Recall (正类召回率)
                task_result['recall'] = recall_score(task_labels, task_preds, pos_label=1, zero_division=0)
                
                # F1 Score
                task_result['f1'] = f1_score(task_labels, task_preds, pos_label=1, zero_division=0)
                
                # AUC
                try:
                    task_result['auc'] = roc_auc_score(task_labels, task_probs)
                except:
                    task_result['auc'] = 0.0
            
            results.append(task_result)
        
        return results
    
    def _evaluate_regression_heads(self, labels, preds, num_tasks):
        """评估每个回归头"""
        results = []
        
        for task_id in range(num_tasks):
            task_labels = labels[:, task_id]
            task_preds = preds[:, task_id]
            
            # 计算各种回归指标
            mse = mean_squared_error(task_labels, task_preds)
            mae = mean_absolute_error(task_labels, task_preds)
            rmse = np.sqrt(mse)
            
            # 计算R²
            ss_res = np.sum((task_labels - task_preds) ** 2)
            ss_tot = np.sum((task_labels - np.mean(task_labels)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # 计算Pearson相关系数
            try:
                pearson_corr, _ = pearsonr(task_labels, task_preds)
            except:
                pearson_corr = 0.0
            
            results.append({
                'task_id': task_id,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'pearson_corr': pearson_corr
            })
        
        return results
    
    def _compute_overall_metrics(self, binary_labels, binary_preds, binary_probs, 
                                  reg_labels, reg_preds):
        """计算总体指标"""
        # 二分类总体
        overall_acc = accuracy_score(binary_labels.flatten(), binary_preds.flatten())
        overall_precision = precision_score(binary_labels.flatten(), binary_preds.flatten(), zero_division=0)
        
        # 每个任务的平均值
        per_task_acc = np.mean([accuracy_score(binary_labels[:, i], binary_preds[:, i]) 
                               for i in range(binary_labels.shape[1])])
        per_task_precision = np.mean([precision_score(binary_labels[:, i], binary_preds[:, i], 
                                                     pos_label=1, zero_division=0) 
                                     for i in range(binary_labels.shape[1])])
        per_task_recall = np.mean([recall_score(binary_labels[:, i], binary_preds[:, i], 
                                               pos_label=1, zero_division=0) 
                                  for i in range(binary_labels.shape[1])])
        per_task_f1 = np.mean([f1_score(binary_labels[:, i], binary_preds[:, i], 
                                       pos_label=1, zero_division=0) 
                              for i in range(binary_labels.shape[1])])
        
        # 回归总体
        overall_mse = mean_squared_error(reg_labels.flatten(), reg_preds.flatten())
        overall_mae = mean_absolute_error(reg_labels.flatten(), reg_preds.flatten())
        overall_rmse = np.sqrt(overall_mse)
        
        return {
            'binary': {
                'overall_accuracy': overall_acc,
                'overall_precision': overall_precision,
                'avg_per_task_accuracy': per_task_acc,
                'avg_per_task_precision': per_task_precision,
                'avg_per_task_recall': per_task_recall,
                'avg_per_task_f1': per_task_f1,
            },
            'regression': {
                'overall_mse': overall_mse,
                'overall_mae': overall_mae,
                'overall_rmse': overall_rmse,
            }
        }


def print_results(results, num_binary_tasks=28, num_reg_tasks=7):
    """美观打印评估结果"""
    print("\n" + "="*80)
    print("MULTI-TASK EVALUATION RESULTS")
    print("="*80)
    
    # ===== 二分类头结果 =====
    print("\n" + "="*80)
    print("BINARY CLASSIFICATION HEADS (28 tasks)")
    print("="*80)
    print(f"{'Task':<6} {'Acc':<8} {'Prec':<8} {'Recall':<10} {'F1':<10} {'AUC':<10} {'Pos':<8} {'Neg':<8}")
    print("-"*80)
    
    binary_results = results['binary']
    binary_accs = []
    binary_precisions = []
    binary_recalls = []
    binary_f1s = []
    binary_aucs = []
    
    for task_result in binary_results[:num_binary_tasks]:
        task_id = task_result['task_id']
        acc = task_result['accuracy']
        precision = task_result.get('precision', 0.0)
        recall = task_result['recall']
        f1 = task_result['f1']
        auc = task_result['auc']
        pos = task_result['n_positive']
        neg = task_result['n_negative']
        
        binary_accs.append(acc)
        binary_precisions.append(precision)
        binary_recalls.append(recall)
        binary_f1s.append(f1)
        binary_aucs.append(auc)
        
        # 标记警告
        warning = task_result.get('warning', '')
        warn_mark = " ⚠️" if warning else ""
        
        print(f"{task_id:<6} {acc:.4f}   {precision:.4f}   {recall:.4f}     {f1:.4f}     {auc:.4f}   {pos:<8} {neg:<8}{warn_mark}")
    
    # 二分类平均值
    print("-"*80)
    print(f"{'AVG':<6} {np.mean(binary_accs):.4f}   {np.mean(binary_precisions):.4f}   {np.mean(binary_recalls):.4f}     {np.mean(binary_f1s):.4f}     {np.mean(binary_aucs):.4f}")
    print("="*80)
    
    # ===== 回归头结果 =====
    print("\n" + "="*80)
    print("REGRESSION HEADS (7 tasks)")
    print("="*80)
    print(f"{'Task':<6} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R²':<10} {'Pearson':<10}")
    print("-"*80)
    
    reg_results = results['regression']
    reg_maes = []
    reg_r2s = []
    
    for task_result in reg_results[:num_reg_tasks]:
        task_id = task_result['task_id']
        mse = task_result['mse']
        mae = task_result['mae']
        rmse = task_result['rmse']
        r2 = task_result['r2']
        pearson = task_result['pearson_corr']
        
        reg_maes.append(mae)
        reg_r2s.append(r2)
        
        print(f"{task_id:<6} {mse:.6f}   {mae:.6f}   {rmse:.6f}   {r2:.4f}     {pearson:.4f}")
    
    # 回归平均值
    print("-"*80)
    print(f"{'AVG':<6} {'-':<12} {np.mean(reg_maes):.6f}   {'-':<12} {np.mean(reg_r2s):.4f}")
    print("="*80)
    
    # ===== 总体指标 =====
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    
    overall = results['overall']
    print(f"\n[Binary Classification]")
    print(f"  Overall Accuracy:          {overall['binary']['overall_accuracy']:.4f}")
    print(f"  Overall Precision:         {overall['binary']['overall_precision']:.4f}")
    print(f"  Avg Per-Task Accuracy:     {overall['binary']['avg_per_task_accuracy']:.4f}")
    print(f"  Avg Per-Task Precision:    {overall['binary']['avg_per_task_precision']:.4f}")
    print(f"  Avg Per-Task Recall:       {overall['binary']['avg_per_task_recall']:.4f}")
    print(f"  Avg Per-Task F1:           {overall['binary']['avg_per_task_f1']:.4f}")
    
    print(f"\n[Regression]")
    print(f"  Overall MSE:               {overall['regression']['overall_mse']:.6f}")
    print(f"  Overall MAE:               {overall['regression']['overall_mae']:.6f}")
    print(f"  Overall RMSE:              {overall['regression']['overall_rmse']:.6f}")
    
    print("\n" + "="*80)
    
    # 保存详细结果到文件
    save_detailed_results(results, 'pred_output/evaluation_results.json')


def save_detailed_results(results, filename):
    """保存详细结果到JSON文件"""
    # 转换numpy类型为Python原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        else:
            return obj
    
    # 递归转换
    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(item) for item in obj]
        else:
            return convert_to_serializable(obj)
    
    serializable_results = recursive_convert(results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nDetailed results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='测试多任务视频理解模型')
    
    parser.add_argument('--test_json', type=str, required=True, help='测试数据json文件路径')
    parser.add_argument('--model_id', type=str, required=True,help='Qwen2.5-VL模型路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')

    # parser.add_argument('--cuda', type=int, default=0, help='CUDA设备编号')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--num_frames', type=int, default=8, help='视频采样帧数')
    parser.add_argument('--num_binary_tasks', type=int, default=28, help='二分类任务数量')
    parser.add_argument('--num_reg_tasks', type=int, default=7, help='回归任务数量')
    parser.add_argument('--unfreeze_layers', type=int, default=0, help='回归任务数量')
    parser.add_argument('--max_batches', type=int, default=None, help='最大评估批次数（None=评估全部）')
    parser.add_argument('--shard_idx', type=int, default=0, help='分片索引（0 起）')
    parser.add_argument('--num_shards', type=int, default=1, help='总分片数')
    parser.add_argument('--save_predictions', type=str, default='true', choices=['true', 'false'],
                        help='是否保存逐样本预测详情（大数据集建议 false）')
    parser.add_argument('--output_dir', type=str, default='.', help='原始数组 npz 输出目录')

    args = parser.parse_args()

    save_predictions = (args.save_predictions == 'true')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("MULTI-TASK EVALUATION CONFIGURATION")
    print("="*80)
    print(f"  Test JSON:         {args.test_json}")
    print(f"  Checkpoint:        {args.checkpoint}")
    print(f"  Model:             {args.model_id}")
    print(f"  Batch Size:        {args.batch_size}")
    print(f"  Num Frames:        {args.num_frames}")
    print(f"  Binary Tasks:      {args.num_binary_tasks}")
    print(f"  Regression Tasks:  {args.num_reg_tasks}")
    print("="*80)
    
    # 设置CUDA
    # setup_cuda(args.cuda)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # 加载处理器
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    
    # 加载测试数据
    print(f"\nLoading test data from {args.test_json}...")
    video_paths, binary_labels, regression_labels = get_videos_multilabels(args.test_json)
    
    if len(video_paths) == 0:
        print("No valid videos found!")
        return
    
    print(f"Loaded {len(video_paths)} test samples")

    # ===== 分片（多卡并行评估） =====
    if args.num_shards > 1:
        total = len(video_paths)
        per = (total + args.num_shards - 1) // args.num_shards
        start = args.shard_idx * per
        end = min(start + per, total)
        video_paths = video_paths[start:end]
        binary_labels = binary_labels[start:end]
        regression_labels = regression_labels[start:end]
        print(f"[shard {args.shard_idx}/{args.num_shards}] samples {start}:{end} -> {len(video_paths)}")

    # 创建测试数据集
    test_dataset = VideoMultiTaskDataset(
        video_paths, binary_labels, regression_labels,
        processor, num_frames=args.num_frames
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # 加载模型
    model = load_model(
        args.model_id, 
        args.checkpoint, 
        args.unfreeze_layers, 
        device
    )
    
    # 评估
    print("\nStarting evaluation...")
    evaluator = MultiTaskEvaluator(model, device)
    results = evaluator.evaluate(
        test_loader,
        args.num_binary_tasks,
        args.num_reg_tasks,
        save_predictions=save_predictions,
        max_batches=args.max_batches
    )

    # ===== 保存原始数组（供分片合并重算全量指标） =====
    npz_path = os.path.join(args.output_dir, f"raw_shard{args.shard_idx}.npz")
    np.savez(npz_path,
             binary_logits=results['predictions']['binary_logits'],
             binary_probs=results['predictions']['binary_probs'],
             binary_preds=results['predictions']['binary_preds'],
             binary_labels=results['predictions']['binary_labels'],
             reg_preds=results['predictions']['reg_preds'],
             reg_labels=results['predictions']['reg_labels'])
    print(f"\n✓ 原始数组已保存: {npz_path} ({len(results['predictions']['binary_labels'])} samples)")

    # 打印结果
    print_results(results, args.num_binary_tasks, args.num_reg_tasks)
    
    print("\n✓ Evaluation completed!")


if __name__ == "__main__":
    main()