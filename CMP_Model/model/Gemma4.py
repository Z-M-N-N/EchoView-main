# ============================================================
# 1. CUDA 设置
# ============================================================
import json
import os
import argparse
import random
import numpy as np
from pathlib import Path
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def setup_cuda(cuda_num: int = 5, expandable_segments: bool = True) -> int:
    """设置CUDA环境变量"""
    print(f"Setting CUDA_VISIBLE_DEVICES={cuda_num}")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_num}"
    
    if expandable_segments:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    return cuda_num

# 导入深度学习库
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForMultimodalLM
from tqdm import tqdm
import gc

warnings.filterwarnings("ignore")

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def cmp_ref_response(result):
    """比较预测和参考答案是否一致"""
    ref = result["msg"]["conversations"][-1]["value"]
    res = result["response"]
    return 1 if ref == res else 0

def get_token_probabilities(logits, tokenizer, target_tokens=["是", "否"]):
    """
    从logits中提取目标token的概率
    
    Args:
        logits: 模型的输出logits [batch_size, vocab_size]
        tokenizer: 处理器中的tokenizer
        target_tokens: 目标token列表
    
    Returns:
        dict: 目标token及其概率
    """
    # 应用softmax获取概率
    probs = F.softmax(logits, dim=-1)
    
    # 获取目标token的ID
    token_ids = []
    for token in target_tokens:
        # 使用tokenizer编码
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        token_ids.append(token_id)
    
    # 获取概率
    result = {}
    for i, token in enumerate(target_tokens):
        prob = probs[0, token_ids[i]].item()
        result[token] = prob
    
    # 归一化
    total = sum(result.values())
    if total > 0:
        for token in result:
            result[token] /= total
    
    return result

# ============================================================
# 2. 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Gemma 视频理解推理脚本')
    parser.add_argument('--model_path', type=str, default="/home/mzhao/ECHO_VIEW/对比模型/gemma-4-12B-it", 
                       help='模型路径')
    parser.add_argument('--data_path', type=str, 
                       default="/home/mzhao/ECHO_VIEW/EchoView-main/SFT_BUILDER/SFT_JSON/only_video/test/", 
                       help='测试数据JSON文件路径')
    parser.add_argument('--diag_item', type=str, default="all", help='诊断项，all表示全部')
    parser.add_argument('--sample_size', type=int, default=200, help='每个疾病处理的样本数量')
    parser.add_argument('--num_frames', type=int, default=8, help='提取的视频帧数')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='最大生成token数')
    parser.add_argument('--cuda_num', type=int, default=5, help='指定使用的GPU编号')
    parser.add_argument('--device_map', type=str, default="auto", help='设备映射策略')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_results', type=str, default="/home/mzhao/ECHO_VIEW/实验结果/实验二多模型对比/Gemma4-12B", 
                       help='保存结果目录')
    parser.add_argument('--print_result', action='store_true', default=False, help='打印结果')
    args = parser.parse_args()
    
    # 设置CUDA
    setup_cuda(args.cuda_num)
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # ============ 加载模型 ============
    print("="*70)
    print("正在加载模型...")
    print("="*70)
    
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map=args.device_map
    )
    model.eval()
    
    print(f"模型设备: {model.device}")
    print("模型加载完成!")
    
    # ============ 准备诊断列表 ============
    diag_names = [
        "二尖瓣反流", "二尖瓣狭窄", "二尖瓣脱垂", "三尖瓣反流",
        "主动脉瓣反流", "主动脉瓣狭窄", "主动脉瓣钙化", "主动脉瓣增厚",
        "二叶式主动脉瓣", "主动脉窦部增宽", "升主动脉增宽",
        "左房增大", "左室增大", "右房增大", "右室增大", "双房增大",
        "左室壁增厚", "室间隔基底段增厚", "心尖部增厚", "肥厚型心肌病",
        "室壁瘤", "心包积液", "间隔缺损", "瓣环钙化",
        "收缩活动减弱", "人工主动脉瓣", "人工支架", "起搏器",
    ]
    
    if args.diag_item != "all":
        diag_names = [args.diag_item]
    
    # 创建输出目录
    os.makedirs(args.save_results, exist_ok=True)
    
    # ============ 汇总所有疾病的结果 ============
    all_summary = []
    
    # ============ 主循环：遍历所有疾病 ============
    for diag_name in diag_names:
        print(f"\n{'='*70}")
        print(f"🔬 正在处理: {diag_name}")
        print(f"{'='*70}")
        
        results = []
        
        # 加载当前疾病的数据
        # data_path = os.path.join(args.data_path, f"{diag_name}.json")
        # if not os.path.exists(data_path):
        #     print(f"⚠️ 数据文件不存在: {data_path}")
        #     continue

        if f"{diag_name}.json" in os.listdir(args.save_results):
            print(f"{diag_name}.json 已存在，跳过处理。")
            continue


        try:
            print("正在处理：",f"{args.data_path}/{diag_name}.json")
            with open(f"{args.data_path}/{diag_name}.json",'r') as f:
                data =  json.load(f)
        except:
            print("无此项：",diag_name)
            continue
        
        # 采样数据
        if args.sample_size < len(data):
            sampled_data = data[:args.sample_size]
        else:
            sampled_data = data
        print(f"📊 实际处理: {len(sampled_data)} 条数据")
        
        # ============ 推理循环 ============
        pbar = tqdm(sampled_data, desc=f"处理 {diag_name}", unit="sample")
        
        for idx, item in enumerate(pbar):
            try:
                video_path = item["video"][0]
                prompt = item["conversations"][0]['value'].replace("<video>\n", "")
                ground_truth = item["conversations"][1]['value'] if len(item["conversations"]) > 1 else ""
                
                # 构建消息
                messages = [
                    {
                        'role': 'user',
                        'content': [
                            {"type": "video", "video": video_path},
                            {'type': 'text', 'text': prompt}
                        ]
                    }
                ]
                
                # Process input - 修正：将num_frames放在processor_kwargs中
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    processor_kwargs={"num_frames": args.num_frames}
                ).to(model.device)
                
                input_len = inputs["input_ids"].shape[-1]
                
                # 使用return_dict_in_generate=True获取logits
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=args.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False  # 使用贪婪解码以获得稳定的logits
                )
                
                # 解码生成的文本
                response = processor.decode(outputs.sequences[0][input_len:], skip_special_tokens=False)
                parsed_response = processor.parse_response(response, prefix=inputs["input_ids"])
                
                # 获取第一个生成token的logits（用于概率计算）
                # outputs.scores[0] 是第一个token的logits
                first_token_logits = outputs.scores[0]  # [batch_size, vocab_size]
                
                # 计算"是"和"否"的概率
                token_probs = get_token_probabilities(first_token_logits, processor.tokenizer, ["是", "否"])
                
                # 构造结果 - 完全按照Qwen2.5-VL的格式
                result = {
                    "msg": item,  # 保存完整的原始数据
                    "response": parsed_response['content'],  # 预测结果
                    "prob_y": token_probs.get("是", 0.5),  # "是"的概率
                    "prob_n": token_probs.get("否", 0.5),  # "否"的概率
                }
                
                # 计算是否正确
                is_correct = cmp_ref_response(result)
                result["cmp"] = "✅" if is_correct == 1 else "❌"
                
                # 保存结果
                results.append(result)
                
                # 动态更新进度条 - 计算当前准确率
                cmp_list = [1 if r.get("cmp") == "✅" else 0 for r in results]
                current_acc = sum(cmp_list) / len(cmp_list) if cmp_list else 0
                pbar.set_postfix({
                    'Acc': f'{current_acc:.4f}',
                    'Correct': f'{sum(cmp_list)}/{len(cmp_list)}'
                })
                
                # 每处理10个样本清理一次显存
                if len(results) % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # 可选：打印结果
                if args.print_result:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                    
            except Exception as e:
                print(f"\n⚠️ 处理第{idx}条数据时出错: {e}")
                import traceback
                traceback.print_exc()
                # 出错时也保存结果，按照Qwen2.5-VL格式
                result = {
                    "msg": item,
                    "response": "ERROR",
                    "prob_y": 0.5,
                    "prob_n": 0.5,
                    "error": str(e)
                }
                results.append(result)
                continue
        
        # ============ 计算评估指标（完全按照Qwen2.5-VL格式） ============
        if results:
            # 提取真实标签和预测 - 完全按照Qwen2.5-VL的方式
            y_true = []
            y_probs = []
            y_pred = []
            
            for item in results:
                if "error" not in item:
                    # 真实标签：根据conversations[-1]的值
                    y_true.append(1 if item["msg"]["conversations"][1]["value"] == '是' else 0)
                    # 预测概率：使用prob_y
                    y_probs.append(item['prob_y'])
                    # 预测标签：根据response是否为"是"
                    y_pred.append(1 if item["response"] == "是" else 0)
                else:
                    y_true.append(0)
                    y_probs.append(0.5)
                    y_pred.append(0)
            
            # 计算指标
            try:
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                auc = roc_auc_score(y_true, y_probs) if len(set(y_true)) > 1 else 0.5
                
                print(f"\n📊 {diag_name} 评估结果:")
                print(f"  准确率: {accuracy:.4f}")
                print(f"  精确率: {precision:.4f}")
                print(f"  召回率: {recall:.4f}")
                print(f"  F1分数: {f1:.4f}")
                print(f"  AUC: {auc:.4f}")
                print(f"  总样本: {len(results)}")
                
                # 完全按照Qwen2.5-VL的格式保存
                metrics_results = {
                    "metrics": {
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                        "auc": float(auc)
                    },
                    "results": results
                }
                
            except Exception as e:
                print(f"⚠️ 计算指标时出错: {e}")
                metrics_results = {
                    "metrics": {
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "auc": 0.0
                    },
                    "results": results
                }
            
            # 保存该疾病的结果（命名方式与Qwen2.5-VL一致）
            save_path = os.path.join(args.save_results, f"{diag_name}.json")
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_results, f, ensure_ascii=False, indent=2)
            print(f"💾 结果已保存到: {save_path}")
            
            # 计算正确数
            correct_count = sum(1 for r in results if r.get("cmp") == "✅")
            
            # 添加到汇总
            all_summary.append({
                "disease": diag_name,
                "accuracy": metrics_results["metrics"]["accuracy"],
                "precision": metrics_results["metrics"]["precision"],
                "recall": metrics_results["metrics"]["recall"],
                "f1": metrics_results["metrics"]["f1"],
                "auc": metrics_results["metrics"]["auc"],
                "total_samples": len(results),
                "correct_count": correct_count
            })
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # ============ 保存汇总结果 ============
    # if all_summary:
    #     summary_path = os.path.join(args.save_results, "summary_all_diseases.json")
    #     with open(summary_path, 'w', encoding='utf-8') as f:
    #         json.dump(all_summary, f, ensure_ascii=False, indent=2)
    #     print(f"\n{'='*70}")
    #     print("📊 所有疾病汇总结果:")
    #     print(f"{'='*70}")
        
    #     # 计算平均指标
    #     avg_acc = sum(s["accuracy"] for s in all_summary) / len(all_summary)
    #     avg_precision = sum(s["precision"] for s in all_summary) / len(all_summary)
    #     avg_recall = sum(s["recall"] for s in all_summary) / len(all_summary)
    #     avg_f1 = sum(s["f1"] for s in all_summary) / len(all_summary)
    #     avg_auc = sum(s["auc"] for s in all_summary) / len(all_summary)
        
    #     print(f"平均准确率: {avg_acc:.4f}")
    #     print(f"平均精确率: {avg_precision:.4f}")
    #     print(f"平均召回率: {avg_recall:.4f}")
    #     print(f"平均F1分数: {avg_f1:.4f}")
    #     print(f"平均AUC: {avg_auc:.4f}")
    #     print(f"💾 汇总结果已保存到: {summary_path}")
        
    #     # 打印每个疾病的结果
    #     print(f"\n详细结果:")
    #     print(f"{'疾病':<15s} | {'准确率':<8s} | {'精确率':<8s} | {'召回率':<8s} | {'F1':<8s} | {'AUC':<8s} | {'正确/总数'}")
    #     print("-" * 80)
    #     for s in all_summary:
    #         print(f"{s['disease']:<15s} | {s['accuracy']:.4f}  | {s['precision']:.4f}  | {s['recall']:.4f}  | {s['f1']:.4f}  | {s['auc']:.4f}  | {s['correct_count']}/{s['total_samples']}")

if __name__ == "__main__":
    main()