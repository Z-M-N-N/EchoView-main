import json
import os
import sys
import cv2
import torch
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# ============================================================
# 1. CUDA 设置
# ============================================================
# def setup_cuda(cuda_num: int = 2, expandable_segments: bool = True) -> int:
#     """设置CUDA环境变量"""
#     print(f"Setting CUDA_VISIBLE_DEVICES={cuda_num}")
#     os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_num}"
    
#     if expandable_segments:
#         os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
#     os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#     return cuda_num

# DEFAULT_CUDA_NUM = 2
# setup_cuda(DEFAULT_CUDA_NUM)

# ============================================================
# 2. 模型加载
# ============================================================
MODEL_PATH = "/home/mzhao/ECHO_VIEW/对比模型/Medgemma-1.5-4b-it/medgemma-1.5-4b-it"
IMAGE_OUTPUT_DIR = "images"


def load_model(model_path: str):
    """加载模型和处理器"""
    print(f"Loading model from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path, 
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    print("✅ Model loaded successfully")
    return processor, model

processor, model = load_model(MODEL_PATH)

# ============================================================
# 3. 工具函数
# ============================================================
def load_json(file_path: str) -> dict:
    """加载JSON文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def sample_video_frames(video_path: str, output_dir: str = None, num_frames: int = 8) -> list:
    """
    从视频中均匀采样帧并保存为图片
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录（可选）
        num_frames: 采样帧数
        
    Returns:
        保存的图片路径列表
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 创建输出目录
    if output_dir is None:
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(video_dir, f"{video_name}_frames")

    # output_dir=output_dir+"/"+video_path.split("/")[-2].replace(".mp4","")
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError("无法获取视频帧数")
        
        # 计算采样帧索引（均匀分布）
        actual_frames = min(num_frames, total_frames)
        if actual_frames == 1:
            frame_indices = [total_frames // 2]
        else:
            step = (total_frames - 1) / (actual_frames - 1)
            frame_indices = [int(round(i * step)) for i in range(actual_frames)]
        
        # 采样并保存
        saved_paths = []
        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"⚠️ 跳过无法读取的帧: {idx}")
                continue
            
            save_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
            # print(save_path)
            cv2.imwrite(save_path, frame)
            saved_paths.append(save_path)
        
        if not saved_paths:
            raise ValueError("未能提取任何有效帧")
        
        return saved_paths
        
    finally:
        cap.release()

def extract_yes_no_from_text(text: str) -> str:
    """
    从文本中提取 Yes/No 标签
    
    Args:
        text: 模型输出文本
        
    Returns:
        "Yes" 或 "No"，无法提取时返回 None
    """
    if not text:
        return None
    
    text_lower = text.lower().strip()
    
    # 直接匹配中文
    if "是" in text and "否" not in text:
        return "Yes"
    if "否" in text and "是" not in text:
        return "No"
    
    # 匹配英文
    has_yes = "yes" in text_lower
    has_no = "no" in text_lower
    
    if has_yes and not has_no:
        return "Yes"
    if has_no and not has_yes:
        return "No"
    if has_yes and has_no:
        return "Yes" if text_lower.find("yes") < text_lower.find("no") else "No"
    
    return None

def get_token_probabilities(logits, tokenizer):
    """
    获取 Yes/No 的 token 概率并归一化
    
    Args:
        logits: 模型输出的 logits
        tokenizer: tokenizer 对象
        
    Returns:
        (yes_prob, no_prob): 归一化后的概率
    """
    if logits is None or len(logits) == 0:
        return 0.5, 0.5
    
    probs = torch.softmax(logits[0], dim=-1)
    
    # 支持多个 token 匹配
    yes_tokens = ["Yes", "yes", "是"]
    no_tokens = ["No", "no", "否"]
    
    yes_prob = 0.0
    no_prob = 0.0
    
    for token in yes_tokens:
        try:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            if token_ids:
                token_id = token_ids[0]
                if token_id < len(probs):
                    yes_prob += probs[token_id].item()
        except:
            pass
    
    for token in no_tokens:
        try:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            if token_ids:
                token_id = token_ids[0]
                if token_id < len(probs):
                    no_prob += probs[token_id].item()
        except:
            pass
    
    # 归一化
    total = yes_prob + no_prob
    if total > 1e-8:
        yes_prob = yes_prob / total
        no_prob = no_prob / total
    else:
        yes_prob = 0.5
        no_prob = 0.5
    
    return yes_prob, no_prob

# ============================================================
# 4. 任务配置
# ============================================================
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

RULE_TEMPLATE = """CRITICAL RULES:
- If the condition is present, output EXACTLY: Yes
- If the condition is NOT present, output EXACTLY: No
- Do NOT include any explanation, description, or disclaimer
- Do NOT ask for more information or additional views
- Output ONLY the single word "Yes" or "No"

Answer:"""

def build_prompt(diag_name: str) -> str:
    """构建完整的提示词"""
    task = TASK_DESCRIPTIONS.get(diag_name)
    if task is None:
        task = f"Determine if {diag_name} is present."
    return f"You are a binary classifier for echocardiogram images.\nYour task: {task}\n\n{RULE_TEMPLATE}"

# ============================================================
# 5. 疾病列表
# ============================================================
DIAG_NAMES = [
    # 瓣膜性疾病
    "二尖瓣反流", "二尖瓣狭窄", "二尖瓣脱垂", "三尖瓣反流",
    "主动脉瓣反流", "主动脉瓣狭窄", "主动脉瓣钙化", "主动脉瓣增厚", "二叶式主动脉瓣",
    # 主动脉疾病
    "主动脉窦部增宽", "升主动脉增宽",
    # 心腔增大
    "左房增大", "左室增大", "右房增大", "右室增大", "双房增大",
    # 心肌/室壁病变
    "左室壁增厚", "室间隔基底段增厚", "心尖部增厚", "肥厚型心肌病", "室壁瘤",
    # 心包疾病
    "心包积液",
    # 先天性/结构异常
    "间隔缺损",
    # 钙化/退行性变
    "瓣环钙化",
    # 功能异常
    "收缩活动减弱",
    # 人工植入物
    "人工主动脉瓣", "人工支架", "起搏器",
]

# ============================================================
# 6. 单样本推理
# ============================================================
def process_single_sample(item: dict, prompt_text: str, image_output_dir: str):
    """
    处理单个样本
    
    Returns:
        tuple: (pred_label, true_label, yes_prob, no_prob, pred_text)
    """
    # 提取视频帧
    video_path = item["video"][0]
    image_paths = sample_video_frames(
        video_path,
        output_dir=image_output_dir,
        num_frames=8
    )
    
    # 构建消息
    messages = {
        "role": "user",
        "content": []
    }
    
    # 添加图片
    for img_path in image_paths:
        messages["content"].append({
            "type": "image",
            "image": img_path
        })
    
    # 添加文本提示（放在最后）
    messages["content"].append({
        "type": "text",
        "text": prompt_text
    })
    
    # 应用 chat template
    inputs = processor.apply_chat_template(
        [messages],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            min_new_tokens=1,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            num_beams=2,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=False
        )
    
    # 解码生成的文本
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[:, input_len:]
    pred_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    pred_text = pred_text.replace("<end_of_turn>", "").replace("\n", "").strip()
    
    # 获取概率
    if outputs.scores is not None and len(outputs.scores) > 0:
        yes_prob, no_prob = get_token_probabilities(
            outputs.scores[0], 
            processor.tokenizer
        )
    else:
        yes_prob, no_prob = 0.5, 0.5
    
    # 提取预测标签
    pred_label = extract_yes_no_from_text(pred_text)
    if pred_label is None:
        pred_label = "Yes" if yes_prob >= 0.5 else "No"
    
    # 真实标签
    true_label = "Yes" if item["conversations"][-1]["value"] == "是" else "No"
    
    return pred_label, true_label, yes_prob, no_prob, pred_text

# ============================================================
# 7. 主循环
# ============================================================
def main():
    # 创建输出目录
    # output_dir = "/home/mzhao/ECHO_VIEW/实验结果/实验二多模型对比/Medgemma4B"
    # DATA_BASE_PATH = "/home/mzhao/ECHO_VIEW/EchoView-main/SFT_BUILDER/SFT_JSON/only_video/test"

    if len(sys.argv) < 4:
        print("用法: python script.py <output_dir> <sft_json_dir>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    DATA_BASE_PATH = sys.argv[2]
    test_sample_num=int(sys.argv[3])

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    
    max_samples = test_sample_num
    
    for diag_name in DIAG_NAMES:
        print(f"\n{'='*70}")
        print(f"🔬 正在处理: {diag_name}")
        print(f"{'='*70}")

        if f"{diag_name}.json" in os.listdir(output_dir):
            print(f"{diag_name}.json 已存在，跳过处理。")
            continue


        try:
            print("正在处理：",f"{DATA_BASE_PATH}/{diag_name}.json")
            with open(f"{DATA_BASE_PATH}/{diag_name}.json",'r') as f:
                data =  json.load(f)
        except:
            print("无此项：",diag_name)
            continue

        
        # # 加载数据
        # data_path = os.path.join(DATA_BASE_PATH, f"{diag_name}.json")
        # if not os.path.exists(data_path):
        #     print(f"⚠️ 数据文件不存在: {data_path}")
            continue
        
        
        print(f"📊 加载了 {len(data)} 条数据")
        
        # 构建提示词
        prompt_text = build_prompt(diag_name)
        
        # 处理样本
        results = []
        failed_count = 0
        
        for idx, item in enumerate(tqdm(data[:max_samples], desc="处理中")):
            try:
                pred_label, true_label, yes_prob, no_prob, pred_text = process_single_sample(
                    item, prompt_text, IMAGE_OUTPUT_DIR
                )
                
                # 记录结果
                label = item["conversations"][-1]["value"]
                pred_label_chinese = "是" if pred_label == "Yes" else "否"
                is_correct = (pred_label == true_label)
                
                result = {
                    "msg": item,
                    "response": pred_label_chinese,
                    "prob_y": yes_prob,
                    "prob_n": no_prob,
                    "cmp": "✅" if label == pred_label_chinese else "❌"
                }
                results.append(result)
                
                # 打印进度
                if (idx + 1) % 20 == 0:
                    status = "✅" if is_correct else "❌"
                    print(f"  {status} 样本 {idx+1}/{len(data[:max_samples])}: "
                          f"True={true_label}, Pred={pred_label}, "
                          f"Yes={yes_prob:.3f}, No={no_prob:.3f}")
                
            except Exception as e:
                failed_count += 1
                print(f"❌ 样本 {idx+1} 失败: {str(e)}")
                # 可选：保存失败样本信息
                continue
        
        print(f"✅ 完成: 成功 {len(results)} 个, 失败 {failed_count} 个")
        
        # ===== 计算指标 =====
        if not results:
            print(f"⚠️ {diag_name} 没有有效结果")
            continue
        
        y_true = []
        y_probs = []
        y_pred = []
        
        for item in results:
            # 注意：使用 conversations[-1] 获取最后一个对话（标签）
            y_true.append(1 if item["msg"]["conversations"][-1]["value"] == '是' else 0)
            y_probs.append(item['prob_y'])
            y_pred.append(1 if item["response"] == "是" else 0)
        
        # 计算指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC 需要至少两个类别
        if len(set(y_true)) > 1 and len(set(y_probs)) > 1:
            auc = roc_auc_score(y_true, y_probs)
        else:
            auc = 0.5
        
        metrics_results = {
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
                "total_samples": len(y_true)
            },
            "results": results
        }
        
        # 保存结果
        output_path = os.path.join(output_dir, f"{diag_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n📈 {diag_name} 指标:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        print(f"  Samples:   {len(y_true)}")
    
    print(f"\n{'='*70}")
    print("✅ 所有疾病处理完成！")
    print(f"{'='*70}")

# ============================================================
# 8. 入口
# ============================================================
if __name__ == "__main__":
    main()