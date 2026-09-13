import json
import os
from collections import defaultdict

def load_json_file(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"加载文件失败 {file_path}: {e}")
        return None

def build_id_dict(data_list):
    """构建ID到output的映射字典"""
    id_dict = {}
    for item in data_list:
        if 'input' in item and 'id' in item['input']:
            item_id = item['input']['id']
            output = item.get('output', '')
            # 保存完整的item以便后续使用
            id_dict[item_id] = {
                'output': output,
                'item': item
            }
    return id_dict

def compare_json_files(file1_path, file2_path, output_file=None):
    """
    比较两个JSON文件中相同ID的output
    
    Args:
        file1_path: 第一个JSON文件路径
        file2_path: 第二个JSON文件路径
        output_file: 输出结果文件路径（可选）
    """
    # 加载两个文件
    print(f"加载文件1: {file1_path}")
    data1 = load_json_file(file1_path)
    if data1 is None:
        return
    
    print(f"加载文件2: {file2_path}")
    data2 = load_json_file(file2_path)
    if data2 is None:
        return
    
    # 如果数据是字典且包含'data'字段，则提取数据
    if isinstance(data1, dict) and 'data' in data1:
        data1 = data1['data']
    if isinstance(data2, dict) and 'data' in data2:
        data2 = data2['data']
    
    # 确保是列表
    if not isinstance(data1, list):
        data1 = [data1]
    if not isinstance(data2, list):
        data2 = [data2]
    
    print(f"文件1包含 {len(data1)} 条记录")
    print(f"文件2包含 {len(data2)} 条记录")
    
    # 构建ID映射
    dict1 = build_id_dict(data1)
    dict2 = build_id_dict(data2)
    
    print(f"文件1有 {len(dict1)} 个唯一ID")
    print(f"文件2有 {len(dict2)} 个唯一ID")
    
    # 找出共同的ID
    common_ids = set(dict1.keys()) & set(dict2.keys())
    print(f"共同的ID数量: {len(common_ids)}")
    
    # 统计结果
    results = {
        'total_common': len(common_ids),
        'same': 0,
        'different': 0,
        'only_in_file1': len(set(dict1.keys()) - set(dict2.keys())),
        'only_in_file2': len(set(dict2.keys()) - set(dict1.keys())),
        'details': []
    }
    
    # 比较相同ID的output
    for item_id in common_ids:
        output1 = dict1[item_id]['output']
        output2 = dict2[item_id]['output']
        
        is_same = output1 == output2
        if is_same:
            results['same'] += 1
        else:
            results['different'] += 1
        
        results['details'].append({
            'id': item_id,
            'output_file1': output1,
            'output_file2': output2,
            'is_same': is_same,
            'diag_item': dict1[item_id]['item']['input'].get('diag_item', ''),
            'video': dict1[item_id]['item']['input'].get('video', [])
        })
    
    # 找出只在文件1中的ID
    only_in_1 = set(dict1.keys()) - set(dict2.keys())
    for item_id in only_in_1:
        results['details'].append({
            'id': item_id,
            'status': 'only_in_file1',
            'output_file1': dict1[item_id]['output'],
            'output_file2': None,
            'is_same': False,
            'diag_item': dict1[item_id]['item']['input'].get('diag_item', ''),
            'video': dict1[item_id]['item']['input'].get('video', [])
        })
    
    # 找出只在文件2中的ID
    only_in_2 = set(dict2.keys()) - set(dict1.keys())
    for item_id in only_in_2:
        results['details'].append({
            'id': item_id,
            'status': 'only_in_file2',
            'output_file1': None,
            'output_file2': dict2[item_id]['output'],
            'is_same': False,
            'diag_item': dict2[item_id]['item']['input'].get('diag_item', ''),
            'video': dict2[item_id]['item']['input'].get('video', [])
        })
    
    # 打印统计结果
    print("\n" + "="*60)
    print("比较结果统计")
    print("="*60)
    print(f"共同ID总数: {results['total_common']}")
    print(f"  - Output相同: {results['same']} ({results['same']/results['total_common']*100:.2f}%)")
    print(f"  - Output不同: {results['different']} ({results['different']/results['total_common']*100:.2f}%)")
    print(f"只在文件1中: {results['only_in_file1']}")
    print(f"只在文件2中: {results['only_in_file2']}")
    
    # 按诊断项统计
    diag_stats = defaultdict(lambda: {'same': 0, 'different': 0, 'total': 0})
    for detail in results['details']:
        if detail.get('is_same') is not None and 'status' not in detail:
            diag = detail.get('diag_item', 'unknown')
            diag_stats[diag]['total'] += 1
            if detail['is_same']:
                diag_stats[diag]['same'] += 1
            else:
                diag_stats[diag]['different'] += 1
    
    if diag_stats:
        print("\n按诊断项统计:")
        print("-"*40)
        for diag, stats in sorted(diag_stats.items()):
            same_rate = stats['same']/stats['total']*100 if stats['total'] > 0 else 0
            print(f"{diag:15} 总数:{stats['total']:3d} 相同:{stats['same']:3d} ({same_rate:.1f}%) 不同:{stats['different']:3d}")
    
    # 显示不同的详细列表
    different_items = [d for d in results['details'] if d.get('is_same') is False and 'status' not in d]
    if different_items:
        print(f"\nOutput不同的ID列表 (共{len(different_items)}个):")
        print("-"*60)
        for detail in different_items[:20]:  # 只显示前20个
            print(f"ID: {detail['id']}")
            print(f"  诊断项: {detail['diag_item']}")
            print(f"  文件1: {detail['output_file1']}")
            print(f"  文件2: {detail['output_file2']}")
            print()
        if len(different_items) > 20:
            print(f"... 还有 {len(different_items)-20} 个不同的项")
    
    # 保存结果到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存到: {output_file}")
    
    return results

def compare_json_files_advanced(file1_path, file2_path, output_file=None, detailed_output=None):
    """
    增强版比较函数，支持更多分析和输出格式
    """
    results = compare_json_files(file1_path, file2_path, output_file)
    
    if results is None:
        return
    
    # 生成详细的CSV报告
    if detailed_output:
        import csv
        csv_file = detailed_output if detailed_output.endswith('.csv') else detailed_output + '.csv'
        
        with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', '诊断项', '文件1 Output', '文件2 Output', '是否相同', '状态'])
            
            for detail in results['details']:
                status = detail.get('status', '')
                if status == 'only_in_file1':
                    writer.writerow([detail['id'], detail['diag_item'], detail['output_file1'], '', 'N/A', '仅文件1'])
                elif status == 'only_in_file2':
                    writer.writerow([detail['id'], detail['diag_item'], '', detail['output_file2'], 'N/A', '仅文件2'])
                else:
                    writer.writerow([
                        detail['id'],
                        detail['diag_item'],
                        detail['output_file1'],
                        detail['output_file2'],
                        '是' if detail['is_same'] else '否',
                        '相同' if detail['is_same'] else '不同'
                    ])
        
        print(f"CSV报告已保存到: {csv_file}")
    
    return results

def main():
    diag_name="左室增大"
    # 设置文件路径
    file2 = f"/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/zhaomeng/ECHO_VIEW/result_out_EchoNet-Dynamic/inference_{diag_name}.json"  # 替换为实际路径
    file1 = f"/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/zhaomeng/OTHER_ECHO/PanEcho/result_out_EchoNet-Dynamic/{diag_name}.json"  # 替换为实际路径
    
    # 方式1: 基本比较
    compare_json_files(file1, file2, f"相关性比较/{diag_name}comparison_results.json")
    
    # 方式2: 增强版比较（包含CSV输出）
    # compare_json_files_advanced(
    #     file1, 
    #     file2, 
    #     output_file="comparison_results.json",
    #     detailed_output="comparison_details.csv"
    # )

if __name__ == "__main__":
    main()