import os
import argparse
import json
import math
import hashlib
from pathlib import Path
import sys
import requests
import ast
import random
import warnings
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,roc_auc_score,confusion_matrix
# ============ CUDA设置（尽早执行） ============
# def setup_cuda(cuda_num=2, expandable_segments=True):
#     """设置CUDA环境变量，应在导入torch等库之前调用"""
#     print(f"Setting CUDA_VISIBLE_DEVICES={cuda_num}")
#     os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_num}"
    
#     if expandable_segments:
#         os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
#     os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#     return cuda_num

# # 默认设置（可以被命令行参数覆盖）
# DEFAULT_CUDA_NUM = 5
# setup_cuda(DEFAULT_CUDA_NUM)

# 导入深度学习库
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
print("="*50)
print("GPU 信息:")
print("="*50)

# 检查CUDA是否可用
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 当前设备信息
    current_device = torch.cuda.current_device()
    print(f"当前设备索引: {current_device}")
    print(f"当前设备名称: {torch.cuda.get_device_name(current_device)}")
    print(f"可见GPU数量: {torch.cuda.device_count()}")
    
    # 显示所有可见显卡
    for i in range(torch.cuda.device_count()):
        print(f"\n显卡 {i}:")
        print(f"  名称: {torch.cuda.get_device_name(i)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # 当前显存使用情况
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  已分配显存: {allocated:.2f} GB")
        print(f"  保留显存: {reserved:.2f} GB")
    
    print("="*50)
else:
    print("警告: CUDA不可用，将使用CPU训练")
# ============ 自定义函数定义 ============
def convert_to_native(obj):
        """递归地将 NumPy 类型转换为 Python 原生类型"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
def download_video(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")

def get_video_frames(video_path, num_frames=128, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)
    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    if video_path.startswith('http://') or video_path.startswith('https://'):
        video_file_path = os.path.join(cache_dir, f'{video_hash}.mp4')
        if not os.path.exists(video_file_path):
            download_video(video_path, video_file_path)
    else:
        video_file_path = video_path

    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)
    
    return video_file_path, frames, timestamps

def create_image_grid(images, num_columns=8):
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = math.ceil(len(images) / num_columns)
    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))
    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)
    return grid_image

def apply_lora(model, lora_path):
    """加载LoRA权重到模型"""
    model = PeftModel.from_pretrained(model, lora_path)
    return model

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

import cv2
import os

def sample_video_frames(video_path, output_dir=None, num_frames=8):
    """
    从视频中均匀采样指定数量的帧并保存为图片
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出图片的目录，如果为None则保存在视频同目录下的frames文件夹
        num_frames: 采样帧数，默认为8
    
    Returns:
        list: 保存的图片路径数组
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 创建输出目录
    if output_dir is None:
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(video_dir, f"{video_name}_frames")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        raise ValueError("无法获取视频帧数")
    
    # 如果视频帧数少于需要采样的帧数，调整采样数量
    actual_frames = min(num_frames, total_frames)
    
    # 计算均匀采样的帧索引
    frame_indices = []
    if actual_frames == 1:
        frame_indices = [total_frames // 2]  # 取中间帧
    else:
        # 均匀分布，从第0帧到最后一帧
        step = (total_frames - 1) / (actual_frames - 1)
        frame_indices = [int(round(i * step)) for i in range(actual_frames)]
    
    # 采样并保存帧
    saved_paths = []
    frame_count = 0
    
    for idx in frame_indices:
        # 定位到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"警告: 无法读取第 {idx} 帧，跳过")
            continue
        
        # 生成保存路径
        frame_filename = f"frame_{frame_count:03d}.jpg"
        save_path = os.path.join(output_dir, frame_filename)
        
        # 保存图片
        cv2.imwrite(save_path, frame)
        saved_paths.append(save_path)
        frame_count += 1
    
    cap.release()
    
    # 如果实际保存的帧数少于预期，补充提示
    if len(saved_paths) < num_frames:
        print(f"注意: 实际保存了 {len(saved_paths)} 帧 (请求 {num_frames} 帧)")
    
    return saved_paths
def inference(model, processor, video, prompt,num_frames,diag_item):
    """执行多模态推理"""

    image_paths = sample_video_frames(video,output_dir="images")

    messages=[
            {
            "role": "user", 
            "content": [{"type":"image","image": ip} for ip in image_paths]
            }
    ]

    prompt = f"""
        你必须严格遵守输出格式。

        请观察这些图片，判断是否存在{diag_item}现象？

        你的回答必须且只能从以下两个词中选择一个：
        - 是
        - 否

        不允许输出其他任何内容，包括标点符号、空格、解释说明。
""" 


    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": prompt},  # 文字放最前面
                *[{"type": "image", "image": ip} for ip in image_paths]  # 图片放后面
            ]
        }
    ]

    

    try:
    #     if video is None:
    #         messages = [
    #             {"role": "user", "content": [{"type": "text", "text": prompt}]}
    #         ]
    #     else:
    #         messages = [
    #             {"role": "user", "content": [
    #                 {"type": "video", "video": video,
    #                     "nframes": num_frames,                    # 每秒1帧
    #                     "min_pixels": 224 * 224,
    #                     "max_pixels": 224 * 224
    #                   },
    #                 {"type": "text", "text": prompt}
    #             ]}
    #         ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs)
    
        inputs.to(model.device)


        # with torch.no_grad():
        #     output_ids = model.generate(**inputs)
        
        # generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        # output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # # 清理显存
        # del inputs
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        
        # return output_text[0]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,  # 只生成1个token
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=True, temperature=0.5
            )

            # 获取生成的token ID（用于解码文本）
            generated_ids = outputs.sequences  # 完整序列
            # 或者直接使用 outputs.sequences 减去输入长度
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs.sequences)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # print(output_text)
            
            # 提取第一个token的logits
            first_token_logits = outputs.scores[0]  # shape: [batch_size, vocab_size]
            probs = torch.softmax(first_token_logits, dim=-1)
            
            # 获取"是"/"否"的概率
            yes_id = processor.tokenizer.encode("是", add_special_tokens=False)[0]
            no_id = processor.tokenizer.encode("否", add_special_tokens=False)[0]
            yes_prob = probs[0, yes_id].item()
            no_prob = probs[0, no_id].item()
            total = yes_prob + no_prob
            yes_prob = yes_prob / total
            no_prob = no_prob / total
            

            
            # 清理显存
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 返回文本和概率
            return output_text[0], {"是": yes_prob, "否": no_prob}
        
    except Exception as e:
        print(f"Inference error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
    



# def cmp_ref_response(response,ref):
#     if "：正常" in ref and "：正常" in response:
#         return 1
#     elif "：异常" in ref and "：异常" in response:
#         return 1
#     else:
#         return 0
def cmp_ref_response(result):
    ref=result["msg"]["conversations"][-1]["value"]
    res=result["response"]
    # print(ref,res)
    if ref==res:
        return 1
    return 0
# ============ 主函数 ============
def main():
    parser = argparse.ArgumentParser(description='Qwen2.5-VL 视频理解推理脚本')
    parser.add_argument('--model_path', type=str, default="/home/mzhao/ECHO_VIEW/对比模型/Lingshu-7B", help='基础模型路径')
    parser.add_argument('--processor_path', type=str, default="/home/mzhao/ECHO_VIEW/对比模型/Lingshu-7B", help='处理器路径')
    parser.add_argument('--lora_path', type=str, default="", help='LoRA权重路径')
    parser.add_argument('--diag_item_config', type=str, default="/home/mzhao/ECHO_VIEW/EchoView-main/DATA_BUILDER/diag_info.json", help='')
    parser.add_argument('--device_map', type=str, default="auto", help='设备映射策略 (auto/cuda/cpu)')
    parser.add_argument('--torch_dtype', type=str, default="auto", choices=['auto', 'float16', 'float32'], help='torch数据类型 (auto/float16/float32)')
    parser.add_argument('--cuda_num', type=int, default=2, help='指定使用的GPU编号 (0-7)')
    parser.add_argument('--expandable_segments', type=bool, default=True, help='是否启用PyTorch动态显存扩展')
    parser.add_argument('--data_path', type=str, default="", help='测试数据JSON文件路径')
    parser.add_argument('--diag_item', type=str, default="all", help='诊断项')
    parser.add_argument('--data_root', type=str, default="", help='视频数据根目录')
    parser.add_argument('--sample_size', type=int, default=200, help='随机采样的样本数量')
    parser.add_argument('--num_frames', type=int, default=64, help='提取的视频帧数')
    parser.add_argument('--cache_dir', type=str, default='.cache', help='缓存目录')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='最大生成token数')
    parser.add_argument('--total_pixels', type=int, default=20480 * 32 * 32, help='视频帧最大像素数')
    parser.add_argument('--min_pixels', type=int, default=64 * 32 * 32, help='视频帧最小像素数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_results', type=str, default="/home/mzhao/ECHO_VIEW/实验结果/实验二多模型对比/Lingshu", help='保存结果到指定JSON文件')
    parser.add_argument('--print_result', type=bool, default=False, help='打印结果')
    args = parser.parse_args()
    
    
    # 验证CUDA设置
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"当前空闲显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    # ============ 设置随机种子 ============
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # ============ 加载模型 ============
    print("正在加载模型...")
    processor = AutoProcessor.from_pretrained(args.processor_path)

        # 处理torch_dtype
    if args.torch_dtype == "float16":
        torch_dtype = torch.float16
    elif args.torch_dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = "auto"
    
    model, output_loading_info = AutoModelForVision2Seq.from_pretrained(
    # model, output_loading_info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, 
        torch_dtype=torch_dtype, 
        device_map=args.device_map, 
        output_loading_info=True,
        low_cpu_mem_usage=True
    )
    
    if args.lora_path =='':
        print("lora设置为不加载！")
    else:
        print("加载lora:",args.lora_path)
        model = apply_lora(model, args.lora_path)
    model.eval()  # 设置为评估模式
    
    print("模型加载完成!")
    print(f"Loading info: {output_loading_info}")
    
    # 显示模型所在设备
    if hasattr(model, 'device'):
        print(f"模型设备: {model.device}")
    

    diag_items = list(load_json(args.diag_item_config).keys())
    if args.diag_item!="all":
        diag_items=[args.diag_item]

    for diag_item in diag_items[0:]:
        args.diag_item=diag_item
        # ============ 加载数据 ============
        test_json=args.data_path+"/"+args.diag_item+".json"
        print(f"正在加载数据: {test_json}")
        try:
            print("正在处理：",test_json)
            with open(test_json,'r') as f:
                sft_json = load_json(test_json)
                print(f"总样本数: {len(sft_json)}")
        except:
            print("无此项：",diag_item)
            continue


        if f"{diag_item}.json" in os.listdir(args.save_results):
            print(f"{diag_item}.json 已存在，跳过处理。")
            continue
        
    
        
        # 随机采样
        if args.sample_size < len(sft_json):
            # sampled_data = random.sample(sft_json, k=args.sample_size)
            sampled_data = sft_json[0:args.sample_size]
        else:
            sampled_data = sft_json
        print(f"采样样本数: {len(sampled_data)}")
        
        # ============ 推理循环 ============
        cmp_list = []
        results = []
        
        pbar = tqdm(sampled_data, desc="Processing videos", unit="video")
  
        for msg in pbar:
            # if "：异常" not in msg["conversations"][-1]["value"]:
            #     continue

            video_url = args.data_root + msg["video"][0]
            prompt = msg["conversations"][0]["value"].replace("<video>", "").strip()
            
            try:
                # video_path, frames, timestamps = get_video_frames(
                #     video_url, num_frames=args.num_frames, cache_dir=args.cache_dir
                # )
                
                # 可选：显示图像网格
                # image_grid = create_image_grid(frames, num_columns=8)
                # display(image_grid.resize((640, 640)))
                
                response,prob_y_n = inference(
                    model, processor, video_url, prompt,args.num_frames,diag_item
                    
                )



                result = {
                    # "video": video_url,
                    # "prompt": prompt,
                    "msg":msg,
                    "response": response,
                    # "ref": msg["conversations"][1]["value"],
                    "prob_y":prob_y_n["是"],
                    "prob_n":prob_y_n["否"]
                }
                
                is_correct = cmp_ref_response(result)
                result["cmp"] = "✅" if is_correct==1 else "❌"
                cmp_list.append(1 if is_correct else 0)
                # 保存结果
                results.append(result)
                # 动态更新进度条
                current_acc = sum(cmp_list) / len(cmp_list)
                pbar.set_postfix({
                    'Acc': f'{current_acc:.4f}',
                    'Correct': f'{sum(cmp_list)}/{len(cmp_list)}'
                })
                
                
                
                # 每处理10个视频清理一次显存
                if len(results) % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 可选：打印结果
                if args.print_result=="Ture":
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                
            except Exception as e:
                print(f"\n错误处理视频 {video_url}: {e}")
                continue

        data = results
        # 提取真实标签和预测概率
        y_true = []
        y_probs = []
        y_pred=[]

        for item in data:
            # 真实标签：根据response字段
            y_true.append(1 if item["msg"]["conversations"][1]["value"] == '是' else 0)
            # 预测概率：使用prob_y
            y_probs.append(item['prob_y'])

        # 根据概率得到预测类别（阈值0.5）
            y_pred.append(1 if item["response"]=="是" else 0)

        # print(y_true)
        # print(y_pred)
        # 计算指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_probs)

        metrics_results={
            "metrics":{
                    "accuracy":accuracy,
                    "precision":precision,
                    "recall":recall,
                    "f1":f1,
                    "auc":auc
                },
            "results":results
        }
        # 打印结果
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"AUC: {auc:.4f}")

        if not os.path.exists(args.save_results):
                os.makedirs(args.save_results)
        with open(f"{args.save_results}/{args.diag_item}.json", 'w', encoding='utf-8') as f:
            json.dump(metrics_results, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    main()