#!/usr/bin/env python3
"""
报告JSON合并与数据集分割工具
功能:
1. 递归查找所有 report.json 文件
2. 随机划分路径列表为训练集和测试集
3. 分别合并生成 reports.json, train_data.json, test_data.json
"""

import os
import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ==================== 工具函数 ====================

def is_number(s):
    """判断字符串是否为数字"""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def find_report_files_recursive(root_path, filename="report.json"):
    """递归查找所有 report.json 文件"""
    root_path = Path(root_path)
    found_files = []
    
    for file_path in root_path.rglob(filename):
        if file_path.is_file():
            found_files.append(str(file_path))
    
    return found_files


def load_and_validate_report(file_path):
    """加载并验证 report.json 文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查数据有效性
        if "measurements" not in data:
            return None
        
        # 检查是否所有值都是数字
        measurements = data["measurements"]
        if any(not is_number(v) for v in measurements.values()):
            return None
        
        return data
        
    except (json.JSONDecodeError, Exception):
        return None


def merge_report_files(report_files, output_file, desc="合并报告", data_cache=None):
    """
    合并多个 report.json 文件
    
    Args:
        report_files: report.json 文件路径列表
        output_file: 输出文件路径
        desc: 进度条描述
        data_cache: 数据缓存字典，避免重复读取
        
    Returns:
        dict: 包含合并统计信息
    """
    merged_data = []
    skipped_count = 0
    data_cache = data_cache or {}
    
    for file_path in tqdm(report_files, desc=desc):
        group = str(file_path).split('/')[-3]
        
        # 从缓存获取或加载
        if file_path in data_cache:
            data = data_cache[file_path]
        else:
            data = load_and_validate_report(file_path)
            data_cache[file_path] = data
        
        if data is None:
            skipped_count += 1
            continue
        
        merged_data.append({
            "source": str(file_path),
            "group": group,
            "data": data
        })
    
    # 保存合并结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    return {
        "total_files": len(report_files),
        "merged_count": len(merged_data),
        "skipped_count": skipped_count,
        "output_file": str(output_file)
    }


# ==================== 主函数 ====================

def process_reports(input_folder, output_folder=".", test_size=0.1, random_seed=42):
    """
    处理报告文件：生成 reports.json, train_data.json, test_data.json
    
    Args:
        input_folder: 输入文件夹路径（包含 report.json 的根目录）
        output_folder: 输出文件夹路径（默认当前目录）
        test_size: 测试集比例（默认 0.1）
        random_seed: 随机种子（默认 42）
    """
    # 路径处理
    input_path = Path(input_folder).resolve()
    output_path = Path(output_folder).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 输出文件路径
    all_output = output_path / "reports.json"
    train_output = output_path / "train_data.json"
    test_output = output_path / "test_data.json"
    
    print("=" * 70)
    print("📂 报告文件处理工具")
    print("=" * 70)
    print(f"输入目录: {input_path}")
    print(f"输出目录: {output_path}")
    print(f"测试集比例: {test_size:.0%}")
    print(f"随机种子: {random_seed}")
    
    # 步骤1: 查找所有 report.json
    print(f"\n步骤1: 查找所有 report.json 文件...")
    report_files = find_report_files_recursive(input_path)
    
    if not report_files:
        print("❌ 未找到任何 report.json 文件")
        return
    
    print(f"✅ 找到 {len(report_files)} 个 report.json 文件")
    
    # 步骤2: 划分训练集和测试集
    print(f"\n步骤2: 随机划分数据集...")
    train_paths, test_paths = train_test_split(
        report_files, 
        test_size=test_size, 
        random_state=random_seed
    )
    
    print(f"训练集: {len(train_paths)} 个文件 ({len(train_paths)/len(report_files)*100:.1f}%)")
    print(f"测试集: {len(test_paths)} 个文件 ({len(test_paths)/len(report_files)*100:.1f}%)")
    
    # 步骤3: 合并数据（使用缓存避免重复读取）
    print(f"\n步骤3: 合并生成三个数据集...")
    data_cache = {}  # 共享缓存，每个文件只读一次
    
    # 合并完整数据集
    all_stats = merge_report_files(
        report_files, all_output, 
        desc="生成 reports.json", 
        data_cache=data_cache
    )
    
    # 合并训练集
    train_stats = merge_report_files(
        train_paths, train_output, 
        desc="生成 train_data.json", 
        data_cache=data_cache
    )
    
    # 合并测试集
    test_stats = merge_report_files(
        test_paths, test_output, 
        desc="生成 test_data.json", 
        data_cache=data_cache
    )
    
    # 保存划分信息
    split_info = {
        "total_files": len(report_files),
        "train_files": len(train_paths),
        "test_files": len(test_paths),
        "test_size": test_size,
        "random_seed": random_seed
    }
    info_file = output_path / "split_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    # 总结
    print("\n" + "=" * 70)
    print("✅ 处理完成！")
    print("=" * 70)
    print(f"总文件数: {len(report_files)}")
    print(f"有效报告: {all_stats['merged_count']}")
    print(f"跳过文件: {all_stats['skipped_count']}")
    print(f"\n📁 输出文件（保存在 {output_path}）:")
    print(f"  - reports.json      : {all_stats['merged_count']} 条（全部数据）")
    print(f"  - train_data.json   : {train_stats['merged_count']} 条（训练集）")
    print(f"  - test_data.json    : {test_stats['merged_count']} 条（测试集）")
    print(f"  - split_info.json   : 划分信息")


# ==================== 命令行入口 ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="递归查找并合并 report.json 文件，生成完整集、训练集和测试集"
    )
    parser.add_argument(
        "input_folder",
        help="输入文件夹路径（包含 report.json 的根目录）"
    )
    parser.add_argument(
        "-o", "--output",
        default=".",
        help="输出文件夹路径（默认: 当前目录）"
    )
    parser.add_argument(
        "-s", "--test-size",
        type=float,
        default=0.1,
        help="测试集比例，范围 0-1（默认: 0.1）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）"
    )
    
    args = parser.parse_args()
    
    # 参数验证
    if not 0 < args.test_size < 1:
        print("❌ 错误: test_size 必须在 0 到 1 之间")
        return
    
    process_reports(
        input_folder=args.input_folder,
        output_folder=args.output,
        test_size=args.test_size,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()