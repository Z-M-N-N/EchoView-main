"""
PanEcho 推理脚本（支持多切面视频 → study-level 聚合）

从超声视频（.mp4/.avi 等）中提取 16 帧，运行 PanEcho 模型，
输出 39 项超声心动图临床指标的预测结果。

用法: 修改下方 VIDEO_DIR / VIDEO_PATHS 等配置后直接运行。
    python inference.py
"""

import json
import os
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
sys.path.insert(0, "/home/mzhao/ECHO_VIEW/对比模型/PanEcho/src")
from models import FrameTransformer, MultiTaskModel


# ═══════════════════════════════════════════════════════════════════════════
#  配置（在此修改）
# ═══════════════════════════════════════════════════════════════════════════

# 方式 A：指定目录，自动扫描所有视频
                           # 留空则不扫描
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mpeg', '.mpg', '.m4v')

# 方式 B：手动列出各切面视频路径
VIDEO_PATHS = [
    # 例如一次检查的多个切面：
    # '/path/to/study001_a4c.mp4',
    # '/path/to/study001_a2c.mp4',
    # '/path/to/study001_plax.mp4',
]

CHECKPOINT = '/home/mzhao/ECHO_VIEW/对比模型/PanEcho/panecho.pt'                 # 预训练权重路径
TASKS = 'all'                               # 'all' 或 ['EF', 'LVSystolicFunction', ...]
CLIP_LEN = 16                               # 每个视频抽取帧数
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STUDY_NAME = 'study001'                     # 本次检查标识（仅用于显示）


# ═══════════════════════════════════════════════════════════════════════════
#  任务定义
# ═══════════════════════════════════════════════════════════════════════════

def load_task_list():
    """从 content/tasks.pkl 加载任务列表。"""
    pkl_path = Path("/home/mzhao/ECHO_VIEW/对比模型/PanEcho/content/tasks.pkl")
    task_dict = pd.read_pickle(str(pkl_path))
    task_list = []
    for name, info in task_dict.items():
        task = lambda: None
        task.task_name = name
        task.task_type = info['task_type']
        task.class_names = info['class_names']
        task.class_indices = np.arange(info['class_names'].size)
        task.mean = info['mean']
        task_list.append(task)
    return task_list


def build_model(task_list, clip_len=16, tasks='all', checkpoint=None, device='cpu',
                activations=True):
    """构建 PanEcho 模型并加载权重。"""
    arch = 'convnext_tiny'
    n_layers = 4
    n_heads = 8
    pooling = 'mean'
    transformer_dropout = 0.
    fc_dropout = 0.25

    encoder = FrameTransformer(arch, n_heads, n_layers, transformer_dropout, pooling, clip_len)
    encoder_dim = encoder.encoder.n_features
    model = MultiTaskModel(encoder, encoder_dim, task_list, fc_dropout, activations)

    if checkpoint is not None:
        print(f'加载权重: {checkpoint}')
        state = torch.load(checkpoint, map_location='cpu')
        weights = state['weights'] if 'weights' in state else state
        if 'encoder.time_encoder.pe' in weights:
            del weights['encoder.time_encoder.pe']
        model.load_state_dict(weights, strict=False)
    else:
        print('未指定权重 → 随机初始化')

    all_task_names = [t.task_name for t in task_list]
    if tasks != 'all':
        for t in all_task_names:
            if t not in tasks:
                delattr(model, t + '_head')
        model.tasks = [t for t in task_list if t.task_name in tasks]

    return model.to(device)


# ═══════════════════════════════════════════════════════════════════════════
#  预处理
# ═══════════════════════════════════════════════════════════════════════════

def build_transform():
    return v2.Compose([
        v2.Resize((256, 256)),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])


def load_clip(path: str, clip_len: int = 16):
    """从视频文件中取中间 clip_len 帧，返回 (clip_len, H, W, 3) ndarray。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频文件: {path}')

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total >= clip_len:
        start = (total - clip_len) // 2
    else:
        start = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = []
    for _ in range(clip_len):
        ret, frame = cap.read()
        if not ret:
            frames.append(frames[-1].copy())
        else:
            frames.append(frame)
    cap.release()

    return np.stack(frames, axis=0)  # (F, H, W, 3)


def preprocess(video_np: np.ndarray, transform) -> torch.Tensor:
    """ndarray → 模型输入 (1, 3, clip_len, 224, 224)。"""
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2)
    video = transform(video)
    video = video.permute(1, 0, 2, 3)
    return video.unsqueeze(0)

def format_output(output: dict, prefix: str = '') -> str:
    """格式化输出，将 tensor 转为 numpy 以便显示和序列化"""
    lines = []
    lines.append(f'{prefix}{"=" * 60}')
    lines.append(f'{prefix}{"Task":<36} {"Value":<20}')
    lines.append(f'{prefix}{"-" * 60}')

    for task_name, pred in output.items():
        # 将 tensor 移到 CPU 并转为 numpy
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        
        # 展平处理
        pred_flat = pred.flatten()
        
        if len(pred_flat) == 1:
            lines.append(f'{prefix}{task_name:<36} {pred_flat[0]:<20.4f}')
        else:
            # 对于多分类任务，使用 softmax 归一化
            exp_pred = np.exp(pred_flat - np.max(pred_flat))
            probs = exp_pred / exp_pred.sum()
            lines.append(f'{prefix}{task_name:<36} {", ".join(f"{p:.3f}" for p in probs)}')

    lines.append(f'{prefix}{"=" * 60}')
    return '\n'.join(lines)


def convert_tensors_to_numpy(output: dict) -> dict:
    """递归地将字典中的所有 tensor 转换为 numpy"""
    result = {}
    for key, value in output.items():
        if torch.is_tensor(value):
            result[key] = value.detach().cpu().numpy().tolist()
        elif isinstance(value, dict):
            result[key] = convert_tensors_to_numpy(value)
        elif isinstance(value, list):
            result[key] = [
                v.detach().cpu().numpy().tolist() if torch.is_tensor(v) else v 
                for v in value
            ]
        else:
            result[key] = value
    return result




TASK_DEFINITIONS = {
        'pericardial-effusion': {'type': 'binary_classification', 'classes': ['mild_mod_severe']},
        'LVWallThickness-increased-any': {'type': 'binary_classification', 'classes': ['Increased']},
        'LVWallThickness-increased-modsev': {'type': 'binary_classification', 'classes': ['Moderately|severely increased']},
        'LVWallMotionAbnormalities': {'type': 'binary_classification', 'classes': ['None', 'Present']},
        'RVSystolicFunction': {'type': 'binary_classification', 'classes': ['Decreased']},
        'RASize': {'type': 'binary_classification', 'classes': ['Dilated']},
        'AVStructure': {'type': 'binary_classification', 'classes': ['Bicuspid']},
        'LVOT20mmHg': {'type': 'binary_classification', 'classes': ['Present']},
        'MVStenosis': {'type': 'binary_classification', 'classes': ['Mild|Moderate|Severe']},
        'RAP-8-or-higher': {'type': 'binary_classification', 'classes': ['Present']},
        'LVSize': {'type': 'multi-class_classification', 'classes': ['Mildly Increased', 'Moderately|Severely Increased', 'Normal']},
        'LVSystolicFunction': {'type': 'multi-class_classification', 'classes': ['Mildly Decreased', 'Moderately|Severely Decreased', 'Normal|Hyperdynamic']},
        'LVDiastolicFunction': {'type': 'multi-class_classification', 'classes': ['Mild|Indeterminate', 'Moderate|Severe', 'Normal']},
        'RVSize': {'type': 'multi-class_classification', 'classes': ['Mildly Increased', 'Moderately|Severely Increased', 'Normal']},
        'LASize': {'type': 'multi-class_classification', 'classes': ['Mildly Dilated', 'Moderately|Severely Dilated', 'Normal']},
        'AVStenosis': {'type': 'multi-class_classification', 'classes': ['Mild|Moderate', 'None', 'Severe']},
        'AVRegurg': {'type': 'multi-class_classification', 'classes': ['Mild', 'Moderate|Severe', 'None|Trace']},
        'MVRegurgitation': {'type': 'multi-class_classification', 'classes': ['Mild', 'Moderate|Severe', 'None|Trace']},
        'TVRegurgitation': {'type': 'multi-class_classification', 'classes': ['Mild', 'Moderate|Severe', 'None|Trace']},
        'EF': {'type': 'regression', 'unit': '%'},
        'GLS': {'type': 'regression', 'unit': '%'},
        'LVEDV': {'type': 'regression', 'unit': 'cm^3'},
        'LVESV': {'type': 'regression', 'unit': 'cm^3'},
        'LVSV': {'type': 'regression', 'unit': 'cm^3'},
        'IVSd': {'type': 'regression', 'unit': 'cm'},
        'LVPWd': {'type': 'regression', 'unit': 'cm'},
        'LVIDs': {'type': 'regression', 'unit': 'cm'},
        'LVIDd': {'type': 'regression', 'unit': 'cm'},
        'LVOTDiam': {'type': 'regression', 'unit': 'cm'},
        'E|EAvg': {'type': 'regression', 'unit': 'N/A'},
        'RVSP': {'type': 'regression', 'unit': 'mmHg'},
        'RVIDd': {'type': 'regression', 'unit': 'cm'},
        'TAPSE': {'type': 'regression', 'unit': 'cm'},
        'RVSVel': {'type': 'regression', 'unit': 'cm/s'},
        'LAIDs2D': {'type': 'regression', 'unit': 'cm'},
        'LAVol': {'type': 'regression', 'unit': 'cm^3'},
        'RADimensionM-L(cm)': {'type': 'regression', 'unit': 'cm'},
        'AVPkVel(m|s)': {'type': 'regression', 'unit': 'm/s'},
        'TVPkGrad': {'type': 'regression', 'unit': 'mmHg'},
        'AORoot': {'type': 'regression', 'unit': 'cm'},
    }

diag_map = {


    "心包积液": "pericardial-effusion",
    "左室壁增厚": "LVWallThickness-increased-any",
    "收缩活动减弱": "LVSystolicFunction",
    "右房增大":"RASize",
    "二叶式主动脉瓣": "AVStructure",
    "二尖瓣狭窄": "MVStenosis",
    "左室增大":"LVSize",
    "右室增大": "RVSize",
    "左房增大": "LASize",
    "主动脉瓣狭窄": "AVStenosis",
    "主动脉瓣反流": "AVRegurg",
    "二尖瓣反流": "MVRegurgitation",
    "三尖瓣反流": "TVRegurgitation",
   
    # "左室大小": "LVSize",




    # "左室壁增厚2": "LVWallThickness-increased-modsev",
    # "左心室壁是否存在运动异常": "LVWallMotionAbnormalities",
    # "左心室舒张功能": "LVDiastolicFunction",
    
    # "收缩活动减弱": "RVSystolicFunction",
    
    # "右房增大": "RASize",
    # "左心室流出道压力差": "LVOT20mmHg",
    

    # "右心房压力": "RAP-8-or-higher",
}

DIAGNOSIS_THRESHOLDS = {
    "三尖瓣反流": 0.54,
    "主动脉瓣反流": 0.24,
    "主动脉瓣狭窄": 0.12,
    "二叶式主动脉瓣": 0.0,  # 空值用None表示
    "二尖瓣反流": 0.46,
    "二尖瓣狭窄": 0.02,
    "右室增大": 0.12,
    "右房增大": 0.24,
    "左室增大": 0.02,
    "左室壁增厚": 0.41,
    "左房增大": 0.26,
    "心包积液": 0.03,
    "收缩活动减弱": 0.09
}




# ═══════════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    assert Path(CHECKPOINT).exists(), f'权重文件不存在: {CHECKPOINT}'
    # ── 构建模型 ──────────────────────────────────────────────────────
    print(f'构建 PanEcho 模型… (device={DEVICE})')
    task_list = load_task_list()
    model = build_model(task_list, clip_len=CLIP_LEN,
                        tasks=TASKS, checkpoint=str(CHECKPOINT),
                        device=DEVICE)
    model.eval()
    transform = build_transform()
    if len(sys.argv) < 4:
        print("用法: python script.py <output_dir> <sft_json_dir>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    SFT_JSON_DIR = sys.argv[2]
    test_sample_num=int(sys.argv[3])

    for k,v in diag_map.items():
       
        diag_item=k

        if f"{diag_item}.json" in os.listdir(output_dir):
            print(f"{diag_item}.json 已存在，跳过处理。")
            continue


        try:
            print("正在处理：",f"{SFT_JSON_DIR}/{diag_item}.json")
            with open(f"{SFT_JSON_DIR}/{diag_item}.json",'r') as f:
                diag_file = json.load(f)
        except:
            print("无此项：",diag_item)
            continue
        print(diag_item)
        result=[]
        
        for item in tqdm(diag_file[0:test_sample_num], desc="处理进度"):
            # break
            if item["diag_item"]!=diag_item:
                continue
            # VIDEO_DIR="../VIDEO_DATA/Batch02/"+item['id']
            # # ── 收集视频文件列表 ──────────────────────────────────────────────
            # if VIDEO_PATHS:
            #     video_files = [Path(p) for p in VIDEO_PATHS]
            # elif VIDEO_DIR:
            #     video_dir = Path(VIDEO_DIR)
            #     assert video_dir.is_dir(), f'目录不存在: {VIDEO_DIR}'
            #     video_files = sorted([f for f in video_dir.iterdir()
            #                         if f.suffix.lower() in VIDEO_EXTS])
            # else:
            #     print('错误：请设置 VIDEO_DIR 或 VIDEO_PATHS')
            #     sys.exit(1)

            # assert video_files, '未找到任何视频文件'

            # ── 对每个视频独立推理 ────────────────────────────────────────────


            video_files=item["video"]
            all_preds = []
 
            for vp in video_files:
                # print(f'  推理: {vp.name}')
                video_np = load_clip(str(vp), CLIP_LEN)
                x = preprocess(video_np, transform).to(DEVICE)
                with torch.no_grad():
                    pred = model(x)
                all_preds.append(pred)


                
            # ── Study-level 聚合（按论文：取 mean） ────────────────────────────
            # print(f'\n=== Study-level 聚合: {STUDY_NAME} ({len(all_preds)} 个视频) ===')
            study_pred = {k: torch.stack([p[k] for p in all_preds]).mean(dim=0)
                        for k in all_preds[0]}
            # print(format_output(study_pred))
            study_pred_numpy = convert_tensors_to_numpy(study_pred)

            probs=study_pred_numpy[v]

            mode=TASK_DEFINITIONS[v]["type"]


            en_diag=v
            thresh=DIAGNOSIS_THRESHOLDS[k]
            if mode == "multi-class_classification":
                for prob in probs:
                    max_index = np.argmax(prob)
                    label = TASK_DEFINITIONS[en_diag]["classes"][max_index]
                    
                    if en_diag == "AVStenosis":
                        if prob[0] + prob[2] > thresh:
                            pred = 1
                        else:
                            pred = 0
                        y_prob=prob[0] + prob[2]
                        n_prob=prob[1]
                    else:
                        if prob[0] + prob[1] > thresh:
                            pred = 1
                        else:
                            pred = 0
            
                        y_prob=prob[0] + prob[1]
                        n_prob=prob[2]
            elif mode == "binary_classification":
                for prob in probs:
                    if prob[0] > thresh:
                        pred = 1
                    else:
                        pred = 0
                    y_prob=prob[0]
                    n_prob=1.0-prob[0]

            label=item["conversations"][-1]["value"]
            pred_label="是" if pred==1 else "否"
            all={
                "msg":item,
                "response":pred_label,
                "prob_y":y_prob,
                "prob_n":n_prob,
                "cmp":"✅" if label==pred_label else "❌"
            }
            result.append(all)

        try:
            data = result
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
        except:
            accuracy=0.0
            precision=0.0
            recall=0.0
            f1=0.0
            auc=0.0
        metrics_results={
            "metrics":{
                    "accuracy":accuracy,
                    "precision":precision,
                    "recall":recall,
                    "f1":f1,
                    "auc":auc
                },
            "results":result
        }


        # with open(f"result_echUNAL/{diag_item}.json", 'w', encoding='utf-8') as f:
        #     json.dump(metrics_results, f, ensure_ascii=False, indent=2, default=str)

        # import os

        # 确保目录存在
        
        os.makedirs(output_dir, exist_ok=True)

        # 然后写入文件
        with open(f"{output_dir}/{diag_item}.json", 'w', encoding='utf-8') as f:
            json.dump(metrics_results, f, ensure_ascii=False, indent=2)







