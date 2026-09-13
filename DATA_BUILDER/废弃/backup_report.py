
import os
import shutil
from datetime import datetime

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
    except PermissionError:
        print(f"权限不足，无法访问: {root_dir}")
    
    return subfolders


def backup_report_json(root_dir, backup_suffix="_backup", verbose=True):
    """
    备份所有子文件夹中的 report.json
    
    Args:
        root_dir: 根目录路径
        backup_suffix: 备份文件后缀，默认 "_backup"
        verbose: 是否打印详细信息，默认True
    """
    # 获取所有子文件夹
    print(f"正在扫描文件夹: {root_dir}")
    all_subfolders = get_all_subfolders(root_dir)
    print(f"找到 {len(all_subfolders)} 个子文件夹\n")
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, folder in enumerate(all_subfolders, 1):
        report_path = os.path.join(folder, "report.json")
        
        # 检查是否存在 report.json
        if not os.path.exists(report_path):
            skip_count += 1
            if verbose:
                print(f"[{i}/{len(all_subfolders)}] 跳过（无report.json）: {folder}")
            continue
        
        try:
            # 生成备份文件名
            backup_path = os.path.join(folder, f"report{backup_suffix}.json")
            
            # 如果备份文件已存在，添加时间戳
            if os.path.exists(backup_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(folder, f"report_{timestamp}{backup_suffix}.json")
            
            # 复制文件
            shutil.copy2(report_path, backup_path)
            success_count += 1
            
            if verbose:
                print(f"[{i}/{len(all_subfolders)}] ✓ 已备份: {backup_path}")
                
        except Exception as e:
            fail_count += 1
            print(f"[{i}/{len(all_subfolders)}] ✗ 备份失败: {folder}")
            print(f"   错误信息: {e}")
    
    # 打印统计结果
    print("\n" + "=" * 60)
    print("备份完成！")
    print(f"总文件夹数: {len(all_subfolders)}")
    print(f"成功备份: {success_count}")
    print(f"备份失败: {fail_count}")
    print(f"跳过（无report.json）: {skip_count}")
    print("=" * 60)


if __name__ == "__main__":
    # 设置要备份的根目录
    root = "/home/mzhao/ECHO_VIEW/Data/Echo-View/dicom_videos_group"
    
    # 执行备份
    # 方式1: 默认后缀 "_backup"，生成 report_backup.json
    backup_report_json(root)
    
    # 方式2: 自定义后缀
    # backup_report_json(root, backup_suffix="_20260908")
    
    # 方式3: 不打印详细信息
    # backup_report_json(root, verbose=False)