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


MV_NAME = ['主动脉根部内径', '左房内径', '左室舒张末内径', '左室收缩末内径', '室间隔厚度', '左室后壁厚度', '肺动脉收缩压']
def deal_vcr(folder, diag_info):
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
    binary_labels = []
    for diag_item in diag_list:
        binary_labels.append(labels.get(diag_item, {"value": 0, "mark": 0})["value"])

    measurements= data.get("measurements", {})
    regression_labels = [measurements.get(name, 0) for name in MV_NAME]

    video_paths = []

    cvr_convs=[]
    for view in ALL_VIEWS:
        video_path = os.path.join(folder, f"{view.replace(' ', '_')}.mp4")
        if os.path.exists(video_path):
            # video_paths.append(str(video_path))
    
            conv = {
                "video": str(video_path),
                "binary_labels": binary_labels,
                "regression_labels": regression_labels
            }
            cvr_convs.append(conv)

    return cvr_convs if cvr_convs else None

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="处理医学诊断视频数据集")
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='数据根目录路径')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['vcr'],
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
  
    deal_mode = deal_vcr  # 默认使用 deal_vcr 处理所有切面
    
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

    ALL_CONVS=[]
    for folder in tqdm(test_folders, desc="处理文件夹", unit="个"):
        vcr_convs = deal_mode(folder, global_diag_info)
        
        if vcr_convs is None:
            fail_count += 1
            continue
        success_count += 1
        ALL_CONVS.extend(vcr_convs)
    
    # 第四步：保存结果
    print("\n[4/4] 保存结果...")
    output_dir_train = os.path.join(os.path.dirname(root), f"{mode_dir_name}/{mode}/train")
    output_dir_test = os.path.join(os.path.dirname(root), f"{mode_dir_name}/{mode}/test")
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_test, exist_ok=True)

    train_list, test_list = split_train_test(ALL_CONVS, test_size=test_size, random_state=random_seed)
    
    output_path = os.path.join(output_dir_train, f"visual_multi_task.json")
    save_json(train_list, output_path)
    
    output_path = os.path.join(output_dir_test, f"visual_multi_task.json")
    save_json(test_list, output_path)
    
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
