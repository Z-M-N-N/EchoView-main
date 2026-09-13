"""SFT 数据构建的公共工具与全局数据配置。

本模块提供：
- JSON 读写工具：``load_json`` / ``save_json``
- 报告文本标签匹配：``match_report_label``（判定某诊断项在报告中的是/否标记）
- 训练数据采样平衡：``balance_data``
- 目录下 JSON 合并：``merge_json_files``

同时在本模块导入时加载全局数据（报告全集/训练集/测试集、诊断配置、视频根目录），
供各 build_*.py 构建脚本以 ``from common import ...`` 方式复用。
"""
import glob
import json
import os
import random
from typing import Any, Dict, List

from sklearn.model_selection import train_test_split


# ==================== JSON 读写工具 ====================

def load_json(file_path):
    """读取 JSON 文件并返回解析后的对象。"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, file_path):
    """将数据以 UTF-8 缩进格式写入 JSON 文件。"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ==================== 全局数据配置 ====================
# 以下数据在导入本模块时一次性加载，供所有构建脚本使用。
# reports 为全量报告合并文件，train_reports/test_reports 为划分后的训练/测试集。
# reports = load_json("../../Data/dicom_videos_group/Train_Test_JSON/reports.json")
# train_reports = load_json("../../Data/dicom_videos_group/Train_Test_JSON/train_data.json")
# test_reports = load_json("../../Data/dicom_videos_group/Train_Test_JSON/test_data.json")
# # 视频文件的根目录，报告中的相对路径与其拼接得到完整视频目录。
# video_path = "../../Data/dicom_videos_group"
# 各诊断项配置：prompt（提问）、P/N（阳性/阴性关键词）、primary_views 等。
diag_info = load_json("/home/mzhao/ECHO_VIEW/EchoView-main/DATA_BUILDER/diag_info.json")

    
# ==================== 报告标签匹配 ====================

def match_report_label(P, N, report_txt):
    """根据关键词匹配报告文本，判定诊断项的标签。

    逻辑（优先级从高到低）：
    1. 命中 P（阳性关键词）→ 返回（"是", 1）
    2. 否则命中 N（阴性关键词）→ 返回（"否", -1）
    3. 都未命中 → 返回（"否", 0），mark=0 表示"无法明确判定"

    Args:
        P: list[str]，阳性关键词列表
        N: list[str]，阴性关键词列表
        report_txt: str，拼接后的报告文本

    Returns:
        tuple[str, int]: (诊断结果 "是"/"否", 标记 1/-1/0)
    """
    mark = 0
    value = None
    for p in P:
        if p in report_txt:
            value = "是"
            mark = 1
            break
    if value is None:
        for n in N:
            if n in report_txt:
                value = "否"
                mark = -1
                break
    if value is None:
        value = "否"
        mark = 0
    return value, mark

def balance_data(
    conv_data: List[Dict[str, Any]],
    random_seed=42
):
    # return balance_data_down_resample(conv_data,sample_num,is_train)
    return balance_data_up_resample(conv_data,random_seed=random_seed)





# ==================== 数据采样平衡 ====================
def balance_data_down_resample(
    conv_data: List[Dict[str, Any]],
    random_seed: int = None
) -> List[Dict[str, Any]]:
    """
    从 conv 数据中按类别比例采样以平衡数据集：
    - mark=1 与 mark∈{0, -1} 各采样 min(len(mark=1), len(mark∈{0,-1})) 条（不足时有放回补齐）
    - 最终打乱顺序返回

    Args:
        conv_data: list，包含所有 conv 的列表（每项含 "mark" 字段）
        random_seed: int，随机种子，用于结果复现

    Returns:
        list: 采样平衡后的数据
    """
    if random_seed is not None:
        random.seed(random_seed)

    # 按 mark 分类：正类(1) 和 负类(0和-1融合)
    positive = []  # mark=1
    negative = []  # mark=0 或 -1
    invalid_count = 0
    
    for item in conv_data:
        mark = item.get("mark")
        if mark == 1:
            positive.append(item)
        # elif mark == 0 or mark == -1:
        elif mark == -1:
            negative.append(item)
        else:
            invalid_count += 1

    if invalid_count > 0:
        print(f"警告: 发现 {invalid_count} 条无效 mark 数据，已忽略")

    # 打印各类原始数量
    print(f"原始数据: mark=1: {len(positive)}, mark∈{{0,-1}}: {len(negative)}")

    # 计算采样数量（取两者中的较小值）
    # if is_train:
    sample_count = min(len(positive), len(negative))
    print(f"采样策略: mark=1 和 mark∈{{0,-1}} 各采样 {sample_count} 条")

    sampled = []

    # 采样 mark=1
   
    if len(positive) < len(negative):
        sampled=positive
        sampled.extend(random.choices(negative, k=sample_count))
    else:
        sampled=negative
        sampled.extend(random.choices(positive, k=sample_count))

    # 打乱顺序
    random.shuffle(sampled)

    # 统计实际采样结果
    final_counts = {1: 0, "0&-1": 0}
    for item in sampled:
        mark = item.get("mark")
        if mark == 1:
            final_counts[1] += 1
        elif mark == 0 or mark == -1:
            final_counts["0&-1"] += 1

    print(f"采样结果: mark=1: {final_counts[1]}, mark∈{{0,-1}}: {final_counts['0&-1']}")
    print(f"实际总数: {len(sampled)}")
    print(f"采样比例: mark=1 : mark∈{{0,-1}} = {final_counts[1]}:{final_counts['0&-1']}")

    return sampled

def balance_data_up_resample(
    conv_data: List[Dict[str, Any]],
    random_seed: int = None
) -> List[Dict[str, Any]]:
    """
    从 conv 数据中按类别比例采样以平衡数据集：
    - mark=1 与 mark∈{0, -1} 各采样 min(len(mark=1), len(mark∈{0,-1})) 条（不足时有放回补齐）
    - 最终打乱顺序返回

    Args:
        conv_data: list，包含所有 conv 的列表（每项含 "mark" 字段）
        random_seed: int，随机种子，用于结果复现

    Returns:
        list: 采样平衡后的数据
    """
    if random_seed is not None:
        random.seed(random_seed)

    # 按 mark 分类：正类(1) 和 负类(0和-1融合)
    positive = []  # mark=1
    negative = []  # mark=0 或 -1
    invalid_count = 0
    
    for item in conv_data:
        mark = item.get("mark")
        if mark == 1:
            positive.append(item)
        elif mark == 0 or mark == -1:
            negative.append(item)
        else:
            invalid_count += 1

    if invalid_count > 0:
        print(f"警告: 发现 {invalid_count} 条无效 mark 数据，已忽略")

    # 打印各类原始数量
    print(f"原始数据: mark=1: {len(positive)}, mark∈{{0,-1}}: {len(negative)}")

    # 计算采样数量（取两者中的较小值）

    sample_count = max(len(positive), len(negative))

    if len(positive) > len(negative):
        sampled = positive

        if len(negative) < sample_count:
            print(f"警告: mark=1 只有 {len(negative)} 条数据，将进行有放回采样到 {sample_count} 条")
        sampled.extend(random.choices(negative, k=sample_count))
    else:
        sampled = negative
        if len(positive) < sample_count:
            print(f"警告: mark=1 只有 {len(positive)} 条数据，将进行有放回采样到 {sample_count} 条")
        sampled.extend(random.choices(positive, k=sample_count))




    # 打乱顺序
    random.shuffle(sampled)

    # 统计实际采样结果
    final_counts = {1: 0, "0&-1": 0}
    for item in sampled:
        mark = item.get("mark")
        if mark == 1:
            final_counts[1] += 1
        elif mark == 0 or mark == -1:
            final_counts["0&-1"] += 1

    print(f"采样结果: mark=1: {final_counts[1]}, mark∈{{0,-1}}: {final_counts['0&-1']}")
    print(f"实际总数: {len(sampled)}")
    print(f"采样比例: mark=1 : mark∈{{0,-1}} = {final_counts[1]}:{final_counts['0&-1']}")

    return sampled
# ==================== JSON 文件合并 ====================


def split_train_test(conv_list, test_size=0.2, random_state=42):
    """
    将数据划分为训练集和测试集
    
    Args:
        conv_list: 对话列表
        test_size: 测试集比例，默认0.2
        random_state: 随机种子
    
    Returns:
        train_list, test_list
    """
    if len(conv_list) == 0:
        return [], []
    
    # 使用 sklearn 的 train_test_split
    train_list, test_list = train_test_split(
        conv_list,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    return train_list, test_list



def merge_json_files(directory, output_file="merged.json"):
    """合并指定目录下的所有 JSON 文件为列表。

    Args:
        directory: 包含 JSON 文件的目录路径
        output_file: 输出文件名（保存在 directory 下）

    Returns:
        list | None: 合并后的数据列表；目录下无 JSON 文件时返回 None
    """
    json_files = glob.glob(os.path.join(directory, "*.json"))

    if not json_files:
        print(f"在 {directory} 中没有找到JSON文件")
        return None

    print(f"找到 {len(json_files)} 个JSON文件")

    merged_data = []

    for file_path in json_files:
        if "list" in file_path:
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 如果是列表，扩展合并
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    # 如果是字典，添加文件名作为标识
                    file_name = os.path.splitext(os.path.basename(file_path))[0]
                    merged_data.append({
                        "source": file_name,
                        "data": data
                    })

                print(f"✓ 已合并: {os.path.basename(file_path)}")

        except Exception as e:
            print(f"✗ 读取失败 {os.path.basename(file_path)}: {e}")

    # 保存合并后的文件
    output_path = os.path.join(directory, output_file)
    random.shuffle(merged_data)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 合并完成！共 {len(merged_data)} 条记录保存到: {output_path}")
    return merged_data
