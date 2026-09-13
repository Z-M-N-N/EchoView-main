import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import pandas as pd

# ===== 加载数据 =====
def load_results(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ===== 计算不同阈值下的指标 =====
def find_optimal_threshold(results):
    """
    通过遍历不同阈值，找到最优的 yes_prob 阈值
    
    Args:
        results: 包含 yes_prob 和 label 的列表
    
    Returns:
        最优阈值和对应的指标
    """
    # 提取 yes_prob 和真实标签
    yes_probs = np.array([r["yes_prob"] for r in results])
    labels = np.array([1 if r["label"] == "Yes" else 0 for r in results])
    
    # 定义要测试的阈值范围
    thresholds = np.arange(0.01, 0.99, 0.01)
    
    best_metrics = {
        "threshold": 0.5,
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "auc": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0
    }
    
    all_metrics = []
    
    for threshold in thresholds:
        # 根据阈值预测
        preds = (yes_probs >= threshold).astype(int)
        
        # 计算指标
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        
        # 混淆矩阵
        tp = np.sum((labels == 1) & (preds == 1))
        fp = np.sum((labels == 0) & (preds == 1))
        tn = np.sum((labels == 0) & (preds == 0))
        fn = np.sum((labels == 1) & (preds == 0))
        
        metrics = {
            "threshold": float(threshold),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }
        all_metrics.append(metrics)
        
        # 更新最优指标（根据 F1 分数）
        if f1 > best_metrics["f1"]:
            best_metrics = metrics.copy()
        # 如果 F1 相同，选择更接近 0.5 的阈值
        elif f1 == best_metrics["f1"] and abs(threshold - 0.5) < abs(best_metrics["threshold"] - 0.5):
            best_metrics = metrics.copy()
    
    # 计算 AUC
    fpr, tpr, _ = roc_curve(labels, yes_probs)
    roc_auc = auc(fpr, tpr)
    
    # 将 AUC 添加到最优指标中
    best_metrics["auc"] = float(roc_auc)
    
    return best_metrics, all_metrics

# ===== 主程序 =====
def main():
    all_results = []  # 存储所有诊断项的结果
    
    for file_path in Path("result").glob("*.json"):
        diag_name = str(file_path).split("/")[-1].replace(".json", "")
        json_path = file_path
        results = load_results(json_path)
        
        print("=" * 70)
        print(f"📊 阈值优化分析 - {diag_name}")
        print("=" * 70)
        print(f"总样本数: {len(results)}")
        
        # 统计标签分布
        yes_count = sum(1 for r in results if r["label"] == "Yes")
        no_count = len(results) - yes_count
        print(f"标签分布: Yes={yes_count}, No={no_count}")
        
        # 计算整体 AUC（作为额外参考）
        yes_probs = np.array([r["yes_prob"] for r in results])
        labels = np.array([1 if r["label"] == "Yes" else 0 for r in results])
        fpr, tpr, _ = roc_curve(labels, yes_probs)
        overall_auc = auc(fpr, tpr)
        print(f"整体 AUC: {overall_auc:.4f}")
        print("=" * 70)
        
        # 找到最优阈值
        best_metrics, all_metrics = find_optimal_threshold(results)
        
        print("\n🎯 最优阈值 (基于 F1 分数):")
        print("=" * 70)
        print(f"  诊断项:      {diag_name}")
        print(f"  Threshold:   {best_metrics['threshold']:.2f}")
        print(f"  Accuracy:    {best_metrics['accuracy']:.4f}")
        print(f"  Precision:   {best_metrics['precision']:.4f}")
        print(f"  Recall:      {best_metrics['recall']:.4f}")
        print(f"  F1 Score:    {best_metrics['f1']:.4f}")
        print(f"  AUC:         {best_metrics['auc']:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TP (Yes->Yes): {best_metrics['tp']}")
        print(f"    FP (No->Yes):  {best_metrics['fp']}")
        print(f"    TN (No->No):   {best_metrics['tn']}")
        print(f"    FN (Yes->No):  {best_metrics['fn']}")
        print("=" * 70)
        
        # 输出使用方式
        print("\n💡 使用方式:")
        print(f"   if yes_prob >= {best_metrics['threshold']:.2f}:")
        print(f"       pred = 'Yes'")
        print(f"   else:")
        print(f"       pred = 'No'")
        print("=" * 70)
        
        # 保存当前诊断项的结果
        result_entry = {
            '诊断项': diag_name,
            '总样本数': len(results),
            '阳性样本数(Yes)': yes_count,
            '阴性样本数(No)': no_count,
            '最优阈值': best_metrics['threshold'],
            '准确率(Accuracy)': best_metrics['accuracy'],
            '精确率(Precision)': best_metrics['precision'],
            '召回率(Recall)': best_metrics['recall'],
            'F1分数(F1)': best_metrics['f1'],
            'AUC': best_metrics['auc'],
            'TP(真阳性)': best_metrics['tp'],
            'FP(假阳性)': best_metrics['fp'],
            'TN(真阴性)': best_metrics['tn'],
            'FN(假阴性)': best_metrics['fn'],
            '整体AUC': overall_auc
        }
        all_results.append(result_entry)
    
    # ===== 保存所有结果到Excel =====
    if all_results:
        # 创建DataFrame
        df_results = pd.DataFrame(all_results)
        
        # 按诊断项排序
        df_results = df_results.sort_values('诊断项')
        
        # 保存为Excel文件
        output_file = 'threshold_optimization_results.xlsx'
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 主结果表
            df_results.to_excel(writer, sheet_name='完整结果', index=False)
            
            # 汇总统计
            summary = pd.DataFrame({
                '指标': ['诊断项总数', '总样本数', '阳性样本总数', '阴性样本总数'],
                '数值': [
                    len(df_results),
                    df_results['总样本数'].sum(),
                    df_results['阳性样本数(Yes)'].sum(),
                    df_results['阴性样本数(No)'].sum()
                ]
            })
            summary.to_excel(writer, sheet_name='汇总统计', index=False)
            
            # 最佳性能Top10 (按F1分数排序)
            best_f1 = df_results.nlargest(10, 'F1分数(F1)')[
                ['诊断项', 'F1分数(F1)', '准确率(Accuracy)', '精确率(Precision)', 
                 '召回率(Recall)', 'AUC', '最优阈值', '总样本数']
            ]
            best_f1.to_excel(writer, sheet_name='F1最佳Top10', index=False)
            
            # 最佳AUC Top10
            best_auc = df_results.nlargest(10, 'AUC')[
                ['诊断项', 'AUC', 'F1分数(F1)', '准确率(Accuracy)', 
                 '精确率(Precision)', '召回率(Recall)', '最优阈值', '总样本数']
            ]
            best_auc.to_excel(writer, sheet_name='AUC最佳Top10', index=False)
            
            # 调整列宽
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"\n{'='*70}")
        print(f"✅ 所有结果已保存至: {output_file}")
        print(f"{'='*70}")
        print(f"共处理 {len(df_results)} 个诊断项")
        print(f"总样本数: {df_results['总样本数'].sum()}")
        print(f"阳性样本总数: {df_results['阳性样本数(Yes)'].sum()}")
        print(f"阴性样本总数: {df_results['阴性样本数(No)'].sum()}")
        
        # 打印平均性能
        print(f"\n📊 平均性能指标:")
        print(f"  平均F1分数: {df_results['F1分数(F1)'].mean():.4f}")
        print(f"  平均准确率: {df_results['准确率(Accuracy)'].mean():.4f}")
        print(f"  平均精确率: {df_results['精确率(Precision)'].mean():.4f}")
        print(f"  平均召回率: {df_results['召回率(Recall)'].mean():.4f}")
        print(f"  平均AUC: {df_results['AUC'].mean():.4f}")
        print("=" * 70)
        
        # 打印部分结果预览
        print("\n📋 结果预览 (前10个诊断项):")
        print(df_results[['诊断项', '总样本数', '最优阈值', 'F1分数(F1)', 'AUC']].head(10).to_string(index=False))
        
    return df_results

if __name__ == "__main__":
    df_results = main()