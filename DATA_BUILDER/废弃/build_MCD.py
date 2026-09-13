"""构建「全部视角视频 + 默认提示词」的 SFT 对话数据。

优化版本：一次遍历报告，同时构建所有诊断项的训练/测试数据。

对每条报告，取其全部视频，随机采样至多 4 个（`random.sample`），
生成一条包含多个视频的对话样本。
"""
import argparse
import os
import random
from pathlib import Path
from tqdm import tqdm

from common import balance_data, match_report_label, load_json, merge_json_files, save_json
from common import diag_info, train_reports, test_reports, video_path


def build_all_diag_data_once(save_path, reports, sample_num, diag_items, is_train=True):
    """一次性遍历报告，生成所有诊断项的数据。

    Args:
        save_path: 输出目录
        reports: 报告列表
        sample_num: 每个诊断项的采样数量
        diag_items: 需要处理的诊断项列表
        is_train: 是否为训练集
    """
    os.makedirs(save_path, exist_ok=True)

    # 初始化数据结构：每个诊断项一个列表
    all_data = {diag_item: [] for diag_item in diag_items}

    # 预加载所有诊断项的配置
    diag_configs = {}
    for diag_item in diag_items:
        prompt = diag_info[diag_item]["prompt"]
        P = diag_info[diag_item]["P"]
        N = diag_info[diag_item]["N"]

        # 检查是否有分类参考文件
        split_key_file = f"diag_words/{diag_item}.json"
        if os.path.exists(split_key_file):
            print(f"分类文件存在: {split_key_file}")
            P = load_json(split_key_file)
            print("读取分类参考！")

        diag_configs[diag_item] = {
            "prompt": prompt,
            "P": P,
            "N": N,
        }

    print(f"开始处理 {len(reports)} 条报告，生成 {len(diag_items)} 个诊断项的数据...")

    # 缓存视频文件路径，避免重复扫描
    video_cache = {}

    for report in tqdm(reports, desc="处理报告"):
        try:
            report_data = report["data"]
            report_txt = report_data["diagnosticopinion"] + report_data["findings_text"]
        except (KeyError, TypeError) as e:
            continue

        path = report["group"] + "/" + report_data["path"]

        # 获取该报告的所有视频文件（缓存）
        if path not in video_cache:
            video_path_full = Path(video_path) / path
            if video_path_full.exists():
                video_cache[path] = list(video_path_full.glob("*.mp4"))
            else:
                video_cache[path] = []

        mp4_files = video_cache[path]
        if not mp4_files:
            continue

        # 对每个诊断项：全部视角中随机采样至多 4 个视频
        for diag_item in diag_items:
            config = diag_configs[diag_item]

            multi_vs = [str(mp4) for mp4 in random.sample(mp4_files, min(4, len(mp4_files)))]

            # 匹配标签
            value, mark = match_report_label(config["P"], config["N"], report_txt)

            # 一条报告生成一条包含多个视频的样本
            conv = {
                "id": path,
                "video": multi_vs,
                "mark": mark,
                "diag_item": diag_item,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<video>" * len(multi_vs) + config["prompt"]
                    },
                    {
                        "from": "gpt",
                        "value": value
                    }
                ]
            }
            all_data[diag_item].append(conv)

    # 对每个诊断项进行平衡处理和保存
    print("\n开始平衡和保存数据...")
    for diag_item in tqdm(diag_items, desc="保存诊断项"):
        data = all_data[diag_item]
        if not data:
            print(f"警告: {diag_item} 没有数据")
            continue

        # 平衡数据
        balanced_data = balance_data(data, sample_num, is_train)

        # 保存文件
        save_path_file = os.path.join(save_path, f"{diag_item}.json")
        save_json(balanced_data, save_path_file)
        print(f"{diag_item}: 原始 {len(data)} 条，平衡后 {len(balanced_data)} 条")

    print(f"视频缓存命中: {len(video_cache)} 个不同路径")
    return all_data


def run_optimized_builder(diag_items, root_path, builder_mode):
    """优化版本：一次性构建所有诊断项的数据"""

    print(f"\n开始优化构建，共 {len(diag_items)} 个诊断项")
    print("=" * 60)

    # 处理训练集
    print("\n>>> 处理训练集")
    train_save_path = os.path.join(f"{root_path}/{builder_mode}", "train")
    build_all_diag_data_once(
        save_path=train_save_path,
        reports=train_reports,
        sample_num=15000,
        diag_items=diag_items,
        is_train=True
    )

    # 处理测试集
    print("\n>>> 处理测试集")
    test_save_path = os.path.join(f"{root_path}/{builder_mode}", "test")
    build_all_diag_data_once(
        save_path=test_save_path,
        reports=test_reports,
        sample_num=1000,
        diag_items=diag_items,
        is_train=False
    )

    # 合并所有训练集和测试集
    print("\n" + "=" * 60)
    print("开始合并所有诊断项的数据...")

    merge_json_files(
        directory=f"{root_path}/{builder_mode}/train",
        output_file="merged_train_list.json"
    )
    merge_json_files(
        directory=f"{root_path}/{builder_mode}/test",
        output_file="merged_test_list.json"
    )

    print("所有任务完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='优化版：一次遍历构建所有诊断项数据')
    parser.add_argument('--diag_item', type=str, help='诊断项，或 "all" 表示处理所有诊断项')
    parser.add_argument('--output', type=str, help='输出根目录')
    args = parser.parse_args()

    builder_mode = "MCD"
    diag_info = load_json("diag_info.json")

    diag_items = list(diag_info.keys())
    root_path = args.output

    if args.diag_item == "all":
        run_optimized_builder(diag_items, root_path, builder_mode)
    else:
        print(f"处理单个诊断项: {args.diag_item}")
        diag_items = [args.diag_item]
        run_optimized_builder(diag_items, root_path, builder_mode)
