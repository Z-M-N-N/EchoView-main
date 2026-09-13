import json
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
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
# DEFAULT_CUDA_NUM = 2
# setup_cuda(DEFAULT_CUDA_NUM)
import csv  # 添加这个导入
from cli import HuatuoChatbot

# ===== 加载模型 =====
bot = HuatuoChatbot("/home/mzhao/ECHO_VIEW/对比模型/HuatuoGPT-Vision-7B")

# 调整生成参数，使输出更稳定
bot.gen_kwargs['temperature'] = 0.1
bot.gen_kwargs['max_new_tokens'] = 5


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_yes_no_from_text(text):
    """从文本中提取 Yes 或 No"""
    if text is None:
        return None
    text_lower = text.lower().strip()
    if "yes" in text_lower and "no" not in text_lower:
        return "Yes"
    elif "no" in text_lower and "yes" not in text_lower:
        return "No"
    elif "yes" in text_lower and "no" in text_lower:
        return "Yes" if text_lower.find("yes") < text_lower.find("no") else "No"
    else:
        return None

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
# ===== 疾病任务描述 =====
TASK_DESCRIPTIONS = {
    "二尖瓣反流": "Determine if mitral regurgitation is present.",
    "二尖瓣狭窄": "Determine if mitral stenosis is present.",
    "二尖瓣脱垂": "Determine if mitral valve prolapse is present.",
    "二叶式主动脉瓣": "Determine if bicuspid aortic valve is present.",
    "三尖瓣反流": "Determine if tricuspid regurgitation is present.",
    "主动脉瓣反流": "Determine if aortic regurgitation is present.",
    "主动脉瓣狭窄": "Determine if aortic stenosis is present.",
    "主动脉瓣钙化": "Determine if aortic valve calcification is present.",
    "主动脉瓣增厚": "Determine if aortic valve thickening is present.",
    "主动脉窦部增宽": "Determine if aortic sinus dilation is present.",
    "升主动脉增宽": "Determine if ascending aortic dilation is present.",
    "瓣环钙化": "Determine if annular calcification is present.",
    "人工主动脉瓣": "Determine if prosthetic aortic valve is present.",
    "人工支架": "Determine if prosthetic stent is present.",
    "起搏器": "Determine if pacemaker is present.",
    "室壁瘤": "Determine if ventricular aneurysm is present.",
    "收缩活动减弱": "Determine if reduced systolic function is present.",
    "双房增大": "Determine if biatrial enlargement is present.",
    "右房增大": "Determine if right atrial enlargement is present.",
    "右室增大": "Determine if right ventricular enlargement is present.",
    "左房增大": "Determine if left atrial enlargement is present.",
    "左室增大": "Determine if left ventricular enlargement is present.",
    "左室壁增厚": "Determine if left ventricular wall thickening is present.",
    "室间隔基底段增厚": "Determine if basal septal thickening is present.",
    "心尖部增厚": "Determine if apical thickening is present.",
    "肥厚型心肌病": "Determine if hypertrophic cardiomyopathy is present.",
    "心包积液": "Determine if pericardial effusion is present.",
    "间隔缺损": "Determine if septal defect is present.",
}


# ===== 通用规则模板 =====
RULE_TEMPLATE = """CRITICAL RULES:
- If the condition is present, output EXACTLY: Yes
- If the condition is NOT present, output EXACTLY: No
- Do NOT include any explanation, description, or disclaimer
- Do NOT ask for more information or additional views
- Output ONLY the single word "Yes" or "No"

Answer:"""


def build_prompt(diag_name):
    """根据疾病名称构建完整的提示词"""
    task = TASK_DESCRIPTIONS.get(diag_name)
    if task is None:
        task = f"Determine if {diag_name} is present."
    
    return f"You are a binary classifier for echocardiogram images.\nYour task: {task}\n\n{RULE_TEMPLATE}"


# ===== 提取图像路径的辅助函数 =====
def extract_image_paths(content):
    """从对话内容中提取所有图像路径"""
    image_paths = []
    for item in content:
        if isinstance(item, dict) and "image" in item:
            image_paths.append(item["image"])
    return image_paths


# ===== 疾病列表 =====
diag_names = [
    "二尖瓣反流", "二尖瓣狭窄", "二尖瓣脱垂", "三尖瓣反流",
    "主动脉瓣反流", "主动脉瓣狭窄", "主动脉瓣钙化", "主动脉瓣增厚",
    "二叶式主动脉瓣", "主动脉窦部增宽", "升主动脉增宽",
    "左房增大", "左室增大", "右房增大", "右室增大", "双房增大",
    "左室壁增厚", "室间隔基底段增厚", "心尖部增厚", "肥厚型心肌病",
    "室壁瘤", "心包积液", "间隔缺损", "瓣环钙化",
    "收缩活动减弱", "人工主动脉瓣", "人工支架", "起搏器",
]


def main():
    # ===== 创建输出目录 =====
    # output_dir = "/home/mzhao/ECHO_VIEW/实验结果/实验二多模型对比/HuaTuo"

    if len(sys.argv) < 4:
            print("用法: python script.py <output_dir> <sft_json_dir>")
            sys.exit(1)
        
    output_dir = sys.argv[1]
    SFT_JSON_DIR = sys.argv[2]
    test_sample_num=int(sys.argv[3])


    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(SFT_JSON_DIR, exist_ok=True)

    # ===== 汇总所有疾病的结果 =====
    all_summary = []

    # ===== 主循环：遍历所有疾病 =====
    for diag_name in diag_names:
        print(f"\n{'='*70}")
        print(f"🔬 正在处理: {diag_name}")
        print(f"{'='*70}")
        
        results = []
        correct = 0
        total = 0
        
        # 加载当前疾病的数据
        if f"{diag_name}.json" in os.listdir(output_dir):
            print(f"{diag_name}.json 已存在，跳过处理。")
            continue

        try:
            print("正在处理：",f"{SFT_JSON_DIR}/{diag_name}.json")
            with open(f"{SFT_JSON_DIR}/{diag_name}.json",'r') as f:
                data =  json.load(f)
        except:
            print("无此项：",diag_name)
            continue

        # 构建当前疾病的提示词
        query = build_prompt(diag_name)
        
        for idx, item in enumerate(tqdm(data[0:test_sample_num], desc="处理中")):
            # try:
            video_path = item["video"][0]
            
            # 提取图像路径
            image_paths = sample_video_frames(video_path,output_dir="/home/mzhao/ECHO_VIEW/OutMODEL/HuatuoGPT-Vision-main/images")


            # print(image_paths)
            
            if not image_paths:
                print(f"⚠️ 样本 {idx+1}: 没有找到图像路径，跳过")
                continue
            
            true_label = item["conversations"][-1]["value"]
            
            # 调用模型推理
            answers, yes_prob, no_prob = bot.inference(query, image_paths, return_probs=True)
            
            # 提取预测标签

            pred_text = answers[0] if answers else ""
            pred_label = extract_yes_no_from_text(pred_text)
            
            # 打印结果
            # print(f"  图像路径: {image_paths[0] if image_paths else 'N/A'}")
            # print(f"  模型输出: {pred_text}")
            # print(f"  Yes 概率: {yes_prob:.4f}" if yes_prob is not None else "  Yes 概率: N/A")
            # print(f"  No 概率: {no_prob:.4f}" if no_prob is not None else "  No 概率: N/A")
            
            # ===== 计算正确率 =====
            total += 1
            is_correct = (pred_label == true_label)
            if is_correct:
                correct += 1
            
            # ===== 记录结果 =====
            label=item["conversations"][-1]["value"]
            pred_label="是" if answers=="Yes" else "否"

            result = {
                "msg":item,
                "response":pred_label,
                "prob_y": yes_prob,
                "prob_n": no_prob,
                "cmp": "✅" if label==pred_label else "❌"
            }
            results.append(result)
            
            # ===== 打印进度 =====
            status = "✅" if is_correct else "❌"
            prob_str = f", Yes={yes_prob:.4f}" if yes_prob is not None else ""
            # print(f"  {status} 样本 {idx+1}: Label={true_label}, Pred={pred_label}{prob_str}\n")
                
            # except Exception as e:
            #     print(f"❌ 样本 {idx+1} 处理失败: {e}")
            #     continue
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


        with open(f"{output_dir}/{diag_name}.json", 'w', encoding='utf-8') as f:
            json.dump(metrics_results, f, ensure_ascii=False, indent=2, default=str)

# ============================================================
# 8. 入口
# ============================================================
if __name__ == "__main__":
    main()