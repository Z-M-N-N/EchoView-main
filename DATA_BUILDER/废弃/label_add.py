import os
import json
from tqdm import tqdm  # 导入进度条
from common import balance_data, load_json, merge_json_files, save_json, match_report_label
from common import diag_info

def get_all_subfolders(root_dir, exclude_pattern="Train_Test"):
    """递归获取所有子文件夹，并过滤掉包含排除模式的文件夹"""
    subfolders = []
    
    try:
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            
            if os.path.isdir(item_path):
                if exclude_pattern not in item:
                    subfolders.append(item_path)
                    subfolders.extend(get_all_subfolders(item_path, exclude_pattern))
                else:
                    print(f"跳过: {item_path}")
    except PermissionError:
        print(f"权限不足，无法访问: {root_dir}")
    
    return subfolders

def safe_save_json(file_path, data):
    """
    安全的保存JSON文件，确保文件路径是字符串
    
    Args:
        file_path: 文件路径（字符串）
        data: 要保存的数据
    """
    # 确保 file_path 是字符串
    if not isinstance(file_path, str):
        print(f"错误: file_path 类型为 {type(file_path)}，期望 str")
        print(f"file_path 内容: {file_path}")
        return False
    
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存文件失败 {file_path}: {e}")
        return False

def check_label(folder, diag_info, save=True, verbose=False):
    """
    检查并添加标签到 report.json
    
    Args:
        folder: 文件夹路径
        diag_info: 诊断信息配置
        save: 是否保存修改到文件，默认为True
        verbose: 是否显示详细信息，默认为False
    
    Returns:
        dict: 包含标签的字典，失败返回None
    """
    # 确保 folder 是字符串
    if not isinstance(folder, str):
        if verbose:
            print(f"错误: folder 类型为 {type(folder)}，期望 str")
        return None
    
    report_json_path = os.path.join(folder, "report.json")
    
    if not os.path.exists(report_json_path):
        return None
    
    try:
        # 加载数据
        report_data = load_json(report_json_path)
        
        # 确保 report_data 是字典
        if not isinstance(report_data, dict):
            if verbose:
                print(f"警告: {folder} 的 report.json 不是字典格式")
            return None
        
        # 检查必需的字段
        if "diagnosticopinion" not in report_data or "findings_text" not in report_data:
            if verbose:
                print(f"警告: {folder} 的 report.json 缺少必需字段")
            return None
        
        report_txt = report_data["diagnosticopinion"] + report_data["findings_text"]
        
        # 生成标签
        diag_items = list(diag_info.keys())
        labels = {}
        for diag_item in diag_items:
            config = diag_info[diag_item]
            value, mark = match_report_label(config["P"], config["N"], report_txt)
            labels[diag_item] ={"value": 1 if value == "是" else 0,
                                "mark": mark}
        
        # 添加标签到 report_data
        report_data["labels"] = labels
        report_data["mark"] = mark
        
        # 保存到文件
        if save:
            if isinstance(report_json_path, str):
                success = safe_save_json(report_json_path, report_data)
                if not success and verbose:
                    print(f"✗ 保存失败: {folder}")
            else:
                if verbose:
                    print(f"✗ 错误: report_json_path 不是字符串类型: {type(report_json_path)}")
                return None
        
        return labels
        
    except Exception as e:
        if verbose:
            print(f"✗ 处理 {folder} 时出错: {e}")
        return None

# 主程序
if __name__ == "__main__":
    print("=" * 70)
    print("开始处理诊断标签生成...")
    print("=" * 70)
    
    root = "/home/mzhao/ECHO_VIEW/Data/dicom_videos_group"
    
    # 第一步：扫描所有子文件夹
    print("\n[1/4] 扫描所有子文件夹...")
    all_subfolders = get_all_subfolders(root)
    print(f"✓ 找到 {len(all_subfolders)} 个子文件夹")
    
    if not diag_info:
        print("错误: 无法加载 diag_info.json")
        exit(1)
    
    print(f"✓ 加载了 {len(diag_info)} 个诊断项")
    
    # 第二步：筛选包含 report.json 的文件夹
    print("\n[2/4] 筛选包含 report.json 的文件夹...")
    test_folders = []
    # 使用 tqdm 显示扫描进度
    for folder in tqdm(all_subfolders, desc="扫描文件夹", unit="个", ncols=80):
        report_json_path = os.path.join(folder, "report.json")
        if os.path.exists(report_json_path):
            test_folders.append(folder)
    
    print(f"✓ 找到 {len(test_folders)} 个包含 report.json 的文件夹")
    
    # 第三步：处理标签生成
    print("\n[3/4] 生成并保存标签...")
    success_count = 0
    fail_count = 0
    
    # 使用 tqdm 显示处理进度
    pbar = tqdm(test_folders, desc="处理文件夹", unit="个", ncols=80)
    
    for folder in pbar:
        # 更新进度条描述，显示当前处理的文件夹名
        folder_name = os.path.basename(folder)
        pbar.set_postfix_str(folder_name[:30])  # 限制长度避免过长
        
        result = check_label(folder, diag_info, save=True, verbose=False)
        
        if result is not None:
            success_count += 1
        else:
            fail_count += 1
        
        # 更新进度条额外信息
        pbar.set_postfix({
            '成功': success_count,
            '失败': fail_count
        })
    
    # 第四步：统计结果
    print("\n[4/4] 生成统计报告...")
    
    print("\n" + "=" * 70)
    print("处理完成！统计信息：")
    print("=" * 70)
    print(f"✓ 成功处理: {success_count} 个文件夹")
    print(f"✗ 处理失败: {fail_count} 个文件夹")
    print(f"📊 成功率: {success_count/(success_count+fail_count)*100:.1f}%")
    print("=" * 70)
    
    # 可选：显示部分成功处理的示例
    if success_count > 0:
        print("\n示例：成功处理的文件夹（前5个）")
        sample_folders = [f for f in test_folders if check_label(f, diag_info, save=False) is not None][:5]
        for i, folder in enumerate(sample_folders, 1):
            print(f"  {i}. {os.path.basename(folder)}")