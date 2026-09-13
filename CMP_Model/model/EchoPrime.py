import os

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
# 项目内模块

sys.path.insert(0, "/home/mzhao/ECHO_VIEW/对比模型/EchoPrime")
PROJECT_ROOT = Path("/home/mzhao/ECHO_VIEW/对比模型/EchoPrime")

# 2. 将项目根目录添加到 Python 路径
sys.path.insert(0, str(PROJECT_ROOT))

# 3. 切换到项目根目录（这样相对路径就能正确工作）
os.chdir(PROJECT_ROOT)
from echo_prime import EchoPrime
# def setup_cuda(cuda_num=2, expandable_segments=True):
#     """设置CUDA环境变量，应在导入torch等库之前调用"""
#     print(f"Setting CUDA_VISIBLE_DEVICES={cuda_num}")
#     os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_num}"
    
#     if expandable_segments:
#         os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
#     os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#     return cuda_num

# # 默认设置（可以被命令行参数覆盖）
# DEFAULT_CUDA_NUM = 2
# setup_cuda(DEFAULT_CUDA_NUM)

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)






import cv2
import numpy as np
import torch

def read_video_frames(video_path, resize=None):
    """
    读取视频的所有帧
    
    参数:
        video_path: 视频文件路径
        resize: 调整大小，如 (224, 224) 或 None
    
    返回:
        frames: numpy数组，形状为 (帧数, 高度, 宽度, 通道数)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if resize is not None:
            frame = cv2.resize(frame, resize)
        
        # OpenCV默认是BGR格式，转为RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frames.append(frame)
    
    cap.release()
    
    return np.array(frames)


def sample_frames(frames, num_frames=16):
    """
    从视频帧中采样固定数量的帧，不足则重复
    
    参数:
        frames: numpy数组，形状为 (总帧数, 高度, 宽度, 通道数)
        num_frames: 目标帧数，默认16
    
    返回:
        sampled: numpy数组，形状为 (num_frames, 高度, 宽度, 通道数)
    """
    total_frames = frames.shape[0]
    
    if total_frames == 0:
        raise ValueError("视频没有读取到任何帧")
    
    if total_frames >= num_frames:
        # 均匀采样 num_frames 帧
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        sampled = frames[indices]
    else:
        # 帧数不足，循环重复填充
        repeat_times = num_frames // total_frames + 1
        repeated = np.tile(frames, (repeat_times, 1, 1, 1))
        sampled = repeated[:num_frames]
    
    return sampled


def process_videos_to_tensor(video_paths, num_frames=16, resize=None, normalize=True):
    """
    处理多个视频，采样固定帧数，转换为PyTorch Tensor
    
    参数:
        video_paths: 视频文件路径列表
        num_frames: 目标帧数，默认16
        resize: 调整大小，如 (224, 224) 或 None
        normalize: 是否归一化到 [0, 1]，默认True
    
    返回:
        tensor: torch.Tensor，形状为 (视频数, 通道数, 帧数, 高度, 宽度)
                数据类型为 float32
    """
    all_videos = []
    
    for i, video_path in enumerate(video_paths):
        # print(f"正在处理视频 {i+1}/{len(video_paths)}: {video_path}")
        
        # 1. 读取视频
        frames = read_video_frames(video_path, resize)
        # print(f"  - 原始帧数: {frames.shape[0]}")
        
        # 2. 采样固定帧数
        sampled = sample_frames(frames, num_frames)
        # print(f"  - 采样后帧数: {sampled.shape[0]}")
        # print(f"  - 帧尺寸: {sampled.shape[1]} x {sampled.shape[2]}")
        
        all_videos.append(sampled)
    
    # 3. 堆叠所有视频: (视频数, 帧数, 高度, 宽度, 通道数)
    stacked = np.stack(all_videos, axis=0)
    # print(f"\n堆叠后形状: {stacked.shape}")  # (2, 16, H, W, 3)
    # print(f"堆叠后数据类型: {stacked.dtype}")
    # print(f"像素值范围: [{stacked.min()}, {stacked.max()}]")
    
    # 4. 转换为PyTorch Tensor
    tensor = torch.from_numpy(stacked)
    
    # 5. 转换为float32并归一化
    if normalize:
        # 归一化到 [0, 1]
        tensor = tensor.float() / 255.0
        # print(f"\n归一化后像素范围: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    else:
        # 只转换类型，不归一化
        tensor = tensor.float()
    
    # 6. 调整维度顺序: (视频数, 帧数, 高度, 宽度, 通道数) -> (视频数, 通道数, 帧数, 高度, 宽度)
    tensor = tensor.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
    
    return tensor


# ========== 使用示例 ==========
if __name__ == "__main__":

    
    
       
    ep = EchoPrime()
    

    diag_map={
        "二尖瓣反流":"mitral_regurgitation",
        "二尖瓣狭窄":"mitral_stenosis",
        "瓣环钙化":"mitral_annular_calcification",
        "主动脉瓣反流":"aortic_regurgitation",
        "主动脉瓣狭窄":"aortic_stenosis",
        "二叶式主动脉瓣":"bicuspid_aov_morphology",
        "三尖瓣反流":"tricuspid_valve_regurgitation",
        "心包积液":"pericardial_effusion",
        "收缩活动减弱":"rv_systolic_function_depressed",
        "右室增大":"right_ventricle_dilation",
        "左房增大":"left_atrium_dilation",
        "右房增大":"right_atrium_dilation",
        "起搏器":"pacemaker"
    }


    yuzhi={
    "二叶式主动脉瓣": 0,
    "三尖瓣反流": 0.00,
    "右室增大": 0.00,
    "主动脉瓣狭窄": 0.00,
    "二尖瓣狭窄": 0,
    "左房增大": 0,
    "二尖瓣反流": 0.02,
    "心包积液": 0.00,
    "右房增大": 0.00,
    "收缩活动减弱": 0.00,
    "主动脉瓣反流": 0,
    "瓣环钙化": 0,
    "起搏器":0
}


    # output_dir="/home/mzhao/ECHO_VIEW/实验结果/实验二多模型对比/EchoPrime"
    # SFT_JSON_DIR="/home/mzhao/ECHO_VIEW/EchoView-main/SFT_BUILDER/SFT_JSON/only_video/test"


    if len(sys.argv) < 4:
            print("用法: python script.py <output_dir> <sft_json_dir>")
            sys.exit(1)
        
    output_dir = sys.argv[1]
    SFT_JSON_DIR = sys.argv[2]
    test_sample_num=int(sys.argv[3])
    # diag="起搏器"
    for diag in diag_map:

        test_json=f"{SFT_JSON_DIR}/{diag}.json"

        
        if f"{diag}.json" in os.listdir(output_dir):
            print(f"{diag}.json 已存在，跳过处理。")
            continue


        try:
            print("正在处理：",test_json)
            with open(test_json,'r') as f:
                data=load_json(test_json)
        except:
            print("无此项：",diag)
            continue
        

        ALL_Result=[]
        for item in tqdm(data[:test_sample_num], desc="处理进度"):
            videos=item["video"]
            # 处理视频，采样16帧，调整到224x224，归一化到 [0, 1]
            stack_of_videos = process_videos_to_tensor(
                videos,
                num_frames=16,
                resize=(224, 224),  # (宽度, 高度)
                normalize=True  # 重要：归一化到 [0, 1]
            )
            # print("\n" + "=" * 50)
            # print("最终结果:")
            # print(f"  形状: {stack_of_videos.shape}")  # (2, 3, 16, 224, 224)
            # print(f"  数据类型: {stack_of_videos.dtype}")
            # print(f"  像素范围: [{stack_of_videos.min().item():.4f}, {stack_of_videos.max().item():.4f}]")
            
            # 使用 EchoPrime
            
            
            # print(f"\n输入形状: {stack_of_videos.shape}")
            
            # 编码视频
            encoded_study = ep.encode_study(stack_of_videos, visualize=False)
            
            # print("Feature logits")
            result_dict=ep.predict_metrics(encoded_study)

            # print(result_dict[diag_map[diag]])
            save_r={
                diag:result_dict[diag_map[diag]]
            }


            prob_y=result_dict[diag_map[diag]]
            prob_n=1.0-prob_y

            label=item["conversations"][-1]["value"]



            pred_label="是" if prob_y >yuzhi[diag] else "否"

            
            fina_save={
                "msg":item,
                "response":pred_label,
                "prob_y":prob_y,
                "prob_n":prob_n,
                "cmp":"✅" if label==pred_label else "❌"
            }
            ALL_Result.append(fina_save)


        data = ALL_Result
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
            "results":ALL_Result
        }


        with open(f"{output_dir}/{diag}.json", 'w', encoding='utf-8') as f:
            json.dump(metrics_results, f, ensure_ascii=False, indent=2, default=str)






    