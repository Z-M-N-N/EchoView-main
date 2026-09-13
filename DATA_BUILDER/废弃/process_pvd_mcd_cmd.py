import os
import json
import argparse  # 添加命令行参数解析
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from common import balance_data, load_json, merge_json_files, save_json, split_train_test
from common import diag_info as global_diag_info

# 所有超声切面列表
ALL_VIEWS = [
    "A4C",
    "Parasternal Long",
    "Apical Doppler",
    "Parasternal Short",
    "A3C",
    "A2C",
    "A5C",
    "Subcostal",
    "Doppler Parasternal Long",
    "Doppler Parasternal Short",
    "SSN"
]

# def get_all_subfolders(root_dir, exclude_pattern="Train_Test"):
#     """递归获取所有子文件夹，并过滤掉包含排除模式的文件夹"""
#     subfolders = []
#     try:
#         for item in os.listdir(root_dir):
#             item_path = os.path.join(root_dir, item)
#             if os.path.isdir(item_path):
#                 if exclude_pattern not in item:
#                     subfolders.append(item_path)
#                     subfolders.extend(get_all_subfolders(item_path, exclude_pattern))
#                 else:
#                     print(f"跳过: {item_path}")
#     except PermissionError:
#         print(f"权限不足，无法访问: {root_dir}")
#     return subfolders

def get_all_subfolders(root_dir, exclude_pattern="Train_Test"):
    """获取根目录下的直接子文件夹"""
    folders = []
    try:
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            for item2 in os.listdir(item_path):
                item2_path = os.path.join(item_path, item2)
                if os.path.isdir(item2_path) and exclude_pattern not in item2:
                    folders.append(item2_path)
    except Exception as e:
        print(f"错误: {e}")
    return folders



def deal_pvd(folder, diag_info):
    """处理单个文件夹，返回诊断对话列表（仅使用主要切面）"""
    if not isinstance(folder, str):
        print(f"错误: folder 类型为 {type(folder)}，期望 str")
        return None
    
    report_json_path = os.path.join(folder, "report.json")
    if not os.path.exists(report_json_path):
        return None
    
    try:
        data = load_json(report_json_path)
        labels = data.get("labels")
        if labels is None:
            return None
    except Exception as e:
        return None

    diag_list = list(diag_info.keys())
    diag_convs = {}
    for diag_item in diag_list:
        if diag_item in labels:
            primary_views = diag_info[diag_item]["primary_views"][0]
            prompt = diag_info[diag_item]["prompt"]
            value = labels[diag_item]["value"]
            mark = labels[diag_item]["mark"]
            if mark == 0:
                continue

            video_path = os.path.join(folder, f"{primary_views.replace(' ', '_')}.mp4")
            if not os.path.exists(video_path):
                continue
                
            conv = {
                "id": folder,
                "video": [str(video_path)],
                "diag_item": diag_item,
                "mark": mark,
                "conversations": [
                    {"from": "human", "value": "<video>\n" + prompt},
                    {"from": "gpt", "value": "是" if value == 1 else "否"}
                ]
            }
            diag_convs[diag_item] = conv
    return diag_convs if diag_convs else None

def deal_mcd(folder, diag_info):
    """处理单个文件夹，返回诊断对话列表（使用主要+辅助切面）"""
    if not isinstance(folder, str):
        print(f"错误: folder 类型为 {type(folder)}，期望 str")
        return None
    
    report_json_path = os.path.join(folder, "report.json")
    if not os.path.exists(report_json_path):
        return None
    
    try:
        data = load_json(report_json_path)
        labels = data.get("labels")
        if labels is None:
            return None
    except Exception as e:
        return None

    diag_list = list(diag_info.keys())
    diag_convs = {}
    for diag_item in diag_list:
        if diag_item in labels:
            primary_views = diag_info[diag_item]["primary_views"][0]
            prompt = diag_info[diag_item]["prompt"]
            value = labels[diag_item]["value"]
            mark = labels[diag_item]["mark"]
            if mark == 0:
                continue

            supplementary_views_primarys = diag_info[diag_item]["primary_views"] + diag_info[diag_item]["supplementary_views"]
            video_paths = []
            for view in supplementary_views_primarys:
                video_path = os.path.join(folder, f"{view.replace(' ', '_')}.mp4")
                if os.path.exists(video_path):
                    video_paths.append(str(video_path))
            
            if not video_paths:
                continue
                
            conv = {
                "id": folder,
                "video": video_paths,
                "diag_item": diag_item,
                "mark": mark,
                "conversations": [
                    {"from": "human", "value": "<video>\n" + prompt},
                    {"from": "gpt", "value": "是" if value == 1 else "否"}
                ]
            }
            diag_convs[diag_item] = conv
    return diag_convs if diag_convs else None

def deal_cmd(folder, diag_info):
    """处理单个文件夹，返回诊断对话列表（使用所有切面）"""
    if not isinstance(folder, str):
        print(f"错误: folder 类型为 {type(folder)}，期望 str")
        return None
    
    report_json_path = os.path.join(folder, "report.json")
    if not os.path.exists(report_json_path):
        return None
    
    try:
        data = load_json(report_json_path)
        labels = data.get("labels")
        if labels is None:
            return None
    except Exception as e:
        return None

    diag_list = list(diag_info.keys())
    diag_convs = {}
    for diag_item in diag_list:
        if diag_item in labels:
            primary_views = diag_info[diag_item]["primary_views"][0]
            prompt = diag_info[diag_item]["prompt"]
            value = labels[diag_item]["value"]
            mark = labels[diag_item]["mark"]
            if mark == 0:
                continue

            # 使用所有预定义的切面列表
            video_paths = []
            for view in ALL_VIEWS:
                video_path = os.path.join(folder, f"{view.replace(' ', '_')}.mp4")
                if os.path.exists(video_path):
                    video_paths.append(str(video_path))

            if not video_paths:
                continue

            conv = {
                "id": folder,
                "video": video_paths,
                "diag_item": diag_item,
                "mark": mark,
                "conversations": [
                    {"from": "human", "value": "<video>\n" + prompt},
                    {"from": "gpt", "value": "是" if value == 1 else "否"}
                ]
            }
            diag_convs[diag_item] = conv
    return diag_convs if diag_convs else None

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="处理医学诊断视频数据集")
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='数据根目录路径')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['pvd', 'mcd', 'cmd'],
                       help='处理模式: pvd(主要切面), mcd(主要+辅助切面), cmd(所有切面)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='测试集比例，默认0.2')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子，默认42')
    parser.add_argument('--mode_dir_name', type=str, default='Train_Test_JSON2',
                           help='输出目录模式，默认')
    # 解析命令行参数
    args = parser.parse_args()
    
    root = args.data_dir
    mode = args.mode
    test_size = args.test_size
    random_seed = args.random_seed
    mode_dir_name= args.mode_dir_name
    
    print("=" * 60)
    print("开始处理诊断数据...")
    print("=" * 60)
    print(f"数据目录: {root}")
    print(f"处理模式: {mode}")
    print(f"测试集比例: {test_size}")
    print(f"随机种子: {random_seed}")
    
    # 选择处理模式
    if mode == 'pvd':
        print("模式: PVD (Primary View Diagnosis) - 仅使用主要切面")
        deal_mode = deal_pvd
    elif mode == 'mcd':
        print("模式: MCD (Multi-Channel Diagnosis) - 使用主要+辅助切面")
        deal_mode = deal_mcd
    elif mode == 'cmd':
        print("模式: CMD (Comprehensive Multi-Channel Diagnosis) - 使用所有切面")
        deal_mode = deal_cmd
    else:
        print(f"错误: 未知模式 '{mode}'")
        return

    # 第一步：扫描文件夹
    print("\n[1/4] 扫描所有子文件夹...")
    all_subfolders = get_all_subfolders(root)
    print(f"找到 {len(all_subfolders)} 个子文件夹")
    
    if not global_diag_info:
        print("错误: 无法加载 diag_info.json")
        return
    
    print(f"加载了 {len(global_diag_info)} 个诊断项")
    
    # 第二步：筛选包含 report.json 的文件夹
    print("\n[2/4] 筛选包含 report.json 的文件夹...")
    test_folders = []
    for folder in tqdm(all_subfolders, desc="扫描文件夹", unit="个"):
        report_json_path = os.path.join(folder, "report.json")
        if os.path.exists(report_json_path):
            test_folders.append(folder)
    
    print(f"找到 {len(test_folders)} 个包含 report.json 的文件夹")
    
    # 第三步：处理数据
    print("\n[3/4] 处理诊断数据...")
    json_results = {}
    success_count = 0
    fail_count = 0
    
    for folder in tqdm(test_folders, desc="处理文件夹", unit="个"):
        result = deal_mode(folder, global_diag_info)
        
        if result is None:
            fail_count += 1
            continue
        success_count += 1
        for diag_item, conv in result.items():
            json_results.setdefault(diag_item, []).append(conv)
    
    # 第四步：保存结果
    print("\n[4/4] 保存结果...")
    output_dir_train = os.path.join(os.path.dirname(root), f"{mode_dir_name}/{mode}/train")
    output_dir_test = os.path.join(os.path.dirname(root), f"{mode_dir_name}/{mode}/test")
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_test, exist_ok=True)

    for diag_item, conv_list in tqdm(json_results.items(), desc="保存文件", unit="个"):
        train_list, test_list = split_train_test(conv_list, test_size=test_size, random_state=random_seed)
        
        train_list = balance_data(train_list, random_seed=random_seed)
        output_path = os.path.join(output_dir_train, f"{diag_item}.json")
        save_json(train_list, output_path)
        
        output_path = os.path.join(output_dir_test, f"{diag_item}.json")
        save_json(test_list, output_path)
    
    # 合并JSON文件
    merge_json_files(output_dir_train, "merged_train_list.json")
    merge_json_files(output_dir_test, "merged_test_list.json")
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("处理完成！统计信息：")
    print("=" * 60)
    print(f"✓ 成功处理: {success_count} 个文件夹")
    print(f"✗ 处理失败: {fail_count} 个文件夹")
    print(f"📁 诊断类型: {len(json_results)} 种")
    print(f"📄 总对话数: {sum(len(v) for v in json_results.values())} 条")
    
    print("\n各诊断类型统计：")
    for diag_item, conv_list in json_results.items():
        print(f"  - {diag_item}: {len(conv_list)} 条")
    
    print(f"\n结果已保存到: {output_dir_train} 和 {output_dir_test}")
    print("=" * 60)

if __name__ == "__main__":
    main()