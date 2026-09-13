# -*- coding: utf-8 -*-
"""
VLM 注意力可视化工具（输入方式不变 + 支持保存图片）
"""

import os

# ========== 0. GPU 配置（必须在 torch 之前）==========
cuda_num = 5
print(f"Use CUDA: {cuda_num}")
os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_num}"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hashlib
import math
import warnings

import requests
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image

from decord import VideoReader, cpu
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForVision2Seq

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


# ==================== 全局配置 ====================
class CFG:
    # 模型
    model_path = "/home/mzhao/ECHO_VIEW/weights_lora/Model/vlm_best_merged_model_PVD"
    # model_path = "/home/mzhao/公共大模型/Qwen2.5-VL-7B-Instruct"

    attn_implementation = "eager"
    dtype = "auto"
    device = "cuda:0"

    # 视频
    fps = 4
    min_pixels = 224 * 224
    max_pixels = 224 * 224
    min_frames = 16
    max_frames = 64
    num_frames_cache = 64

    # 生成
    max_new_tokens = 200
    do_sample = False
    num_beams = 1

    # 可视化
    heatmap_alpha = 0.5
    grid_columns = 8
    close_after_save = False   # True=脚本模式省内存；False=notebook 里继续显示

    # 字体
    font_path = "/home/mzhao/ECHO_VIEW/EchoView-main/VLM_TRAIN/simhei中易黑体.ttf"


# ==================== 字体 ====================
def _load_font(path):
    try:
        return fm.FontProperties(fname=path)
    except Exception:
        print(f"警告: 无法加载字体 {path}，使用默认字体")
        return None


font_prop = _load_font(CFG.font_path)


# ==================== 视频下载 / 抽帧 ====================
def download_video(url, dest_path, chunk_size=8192):
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")


def _resolve_video_path(video_path, cache_dir):
    """URL -> 本地缓存；本地路径原样返回。"""
    os.makedirs(cache_dir, exist_ok=True)
    if video_path.startswith(("http://", "https://")):
        h = hashlib.md5(video_path.encode("utf-8")).hexdigest()
        local = os.path.join(cache_dir, f"{h}.mp4")
        if not os.path.exists(local):
            download_video(video_path, local)
        return local
    return video_path


def get_video_frames(video_path, num_frames=128, cache_dir=".cache"):
    """decord 等间隔抽帧 + npy 缓存。"""
    video_file_path = _resolve_video_path(video_path, cache_dir)
    video_hash = hashlib.md5(video_path.encode("utf-8")).hexdigest()

    frames_cache = os.path.join(cache_dir, f"{video_hash}_{num_frames}_frames.npy")
    ts_cache = os.path.join(cache_dir, f"{video_hash}_{num_frames}_timestamps.npy")

    if os.path.exists(frames_cache) and os.path.exists(ts_cache):
        try:
            return video_file_path, np.load(frames_cache), np.load(ts_cache)
        except Exception as e:
            print(f"[warn] 缓存读取失败，重新抽帧: {e}")

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError(f"视频为空: {video_file_path}")

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(int(i)) for i in indices])

    np.save(frames_cache, frames)
    np.save(ts_cache, timestamps)
    return video_file_path, frames, timestamps


def extract_frames(video_path, num_frames=None):
    """cv2 抽帧（可视化渲染用）。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames is None or num_frames >= total_frames:
        num_frames = total_frames

    if num_frames > 1:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        frame_indices = [total_frames // 2]

    sampled = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            sampled.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(sampled), frame_indices


# ==================== 拼图 ====================
def create_image_grid(images, num_columns=8):
    pil_images = [Image.fromarray(img.astype(np.uint8)) for img in images]
    num_rows = math.ceil(len(pil_images) / num_columns)
    w, h = pil_images[0].size
    grid = Image.new("RGB", (num_columns * w, num_rows * h))
    for idx, im in enumerate(pil_images):
        grid.paste(im, ((idx % num_columns) * w, (idx // num_columns) * h))
    return grid


# ==================== LoRA ====================
def apply_lora(model, lora_path):
    from peft import PeftModel
    return PeftModel.from_pretrained(model, lora_path)


# ==================== 注意力集中度指标 ====================
def calc_attention_concentration(att_map):
    """
    计算注意力集中度的5个核心指标
    输入: att_map - 形状为 [H, W] 或 [T, H, W] 的注意力图
    """
    if att_map.ndim == 3:
        att_map = np.mean(att_map, axis=0)

    flat = att_map.flatten().astype(np.float64)
    flat = flat / (flat.sum() + 1e-10)

    # 1. 空间熵
    nz = flat[flat > 0]
    entropy = float(-np.sum(nz * np.log(nz)))

    # 2. 基尼系数
    sorted_vals = np.sort(flat)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    gini = float((n + 1 - 2 * np.sum(cumsum) / (cumsum[-1] + 1e-10)) / n)

    # 3. 有效面积比
    cumsum_desc = np.cumsum(sorted_vals[::-1])
    idx_90 = int(np.argmax(cumsum_desc >= 0.9))
    area_ratio = (idx_90 + 1) / n

    # 4. 峰值均值比
    peak_mean = float(att_map.max() / (att_map.mean() + 1e-10))

    # 5. 方差
    variance = float(np.var(att_map))

    return {
        "entropy": entropy,
        "gini": gini,
        "area_ratio": area_ratio,
        "peak_mean": peak_mean,
        "variance": variance,
    }


# ==================== 保存图片辅助 ====================
def _sanitize_filename(name: str, max_len: int = 60) -> str:
    """去掉文件名里的非法字符。"""
    bad = '<>:"/\\|?*\n\r\t'
    cleaned = "".join("_" if c in bad else c for c in name).strip(" .")
    if not cleaned:
        cleaned = "attn"
    return cleaned[:max_len]


def _save_figure(fig, check_words, save_path=None):
    """
    保存图片。
    save_path:
        - None                  -> ./attn_outputs/attn_<check_words>.png
        - 目录（存在或以 / 结尾） -> <dir>/attn_<check_words>.png
        - 其他                  -> 视为完整文件路径
    """
    default_dir = "attn_outputs"

    if save_path is None:
        out_dir = default_dir
        fname = f"attn_{_sanitize_filename(check_words)}.png"
        full = os.path.join(out_dir, fname)
    elif os.path.isdir(save_path) or save_path.endswith(os.sep):
        out_dir = save_path
        fname = f"attn_{_sanitize_filename(check_words)}.png"
        full = os.path.join(out_dir, fname)
    else:
        out_dir = os.path.dirname(save_path) or "."
        full = save_path

    os.makedirs(out_dir, exist_ok=True)
    try:
        fig.savefig(full, dpi=150, bbox_inches="tight")
        abspath = os.path.abspath(full)
        print(f"图片已保存到: {abspath}")
        return abspath
    except Exception as e:
        print(f"[warn] 保存图片失败: {e}")
        return None


# ==================== 可视化（已修复横向间隙） ====================
def overlay_heatmaps_triple(images, heatmaps, check_words, alpha=0.6, save_path=None):
    """
    可视化注意力热力图（两行：原图 / 叠加图）
    - 使用 aspect="auto" 消除 imshow 保持比例导致的横向留白
    - gridspec_kw wspace=0 hspace=0 确保子图之间无间隙
    返回: (plt, saved_path)
    """
    N = len(images)
    fig, axes = plt.subplots(
        2, N, figsize=(2 * N, 4),
        gridspec_kw={"wspace": 0, "hspace": 0},
    )
    if N == 1:
        axes = axes.reshape(2, 1)

    for i in range(N):
        img = images[i].astype(np.uint8)
        heatmap = heatmaps[i]

        # 缩放到图像尺寸
        hm_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # 归一化到 0-255
        mn, mx = hm_resized.min(), hm_resized.max()
        if mx - mn > 1e-8:
            hm_norm = (hm_resized - mn) / (mx - mn)
        else:
            hm_norm = hm_resized - mn
        hm_u8 = (hm_norm * 255).astype(np.uint8)

        # JET 色图
        hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

        # 叠加
        overlay = cv2.addWeighted(img, 1 - alpha, hm_color, alpha, 0)

        # 第一行：原图（aspect="auto" 让图填满 axes，消除横向留白）
        axes[0, i].imshow(img, aspect="auto")
        axes[0, i].set_title(f"Frame {i + 1}", fontsize=10)
        axes[0, i].axis("off")

        # 第二行：叠加图
        axes[1, i].imshow(overlay, aspect="auto")
        axes[1, i].axis("off")

    # 行标签
    axes[0, 0].set_ylabel("Original", fontsize=12, fontweight="bold",
                          rotation=90, labelpad=10)
    axes[1, 0].set_ylabel("Overlay", fontsize=12, fontweight="bold",
                          rotation=90, labelpad=10)

    # plt.suptitle(f'Attention Visualization for "{check_words}"',
    #              fontsize=14, fontproperties=font_prop, y=1.02)
    fig.subplots_adjust(wspace=0, hspace=0, left=0.02, right=0.98,
                        top=0.92, bottom=0.02)

    # ---------- 保存 ----------
    saved_path = _save_figure(fig, check_words, save_path)

    # 脚本模式：关掉 figure 省内存；notebook 模式：保留显示
    if CFG.close_after_save:
        plt.close(fig)

    return plt, saved_path


# ==================== 推理 ====================
def inference(model, processor, video, prompt):
    """执行模型推理（输入签名保持原样）。"""
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "video": video,
                    "fps": CFG.fps,
                    "min_pixels": CFG.min_pixels,
                    "max_pixels": CFG.max_pixels,
                    "min_frames": CFG.min_frames,
                    "max_frames": CFG.max_frames,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to(CFG.device)

    generation_kwargs = {
        "do_sample": CFG.do_sample,
        "num_beams": CFG.num_beams,
        "max_new_tokens": CFG.max_new_tokens,
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=False,
            output_attentions=True,
        )

    generated_ids = outputs.sequences
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True
    )[0]

    return {
        "success": True,
        "inputs_dict": inputs,
        "outputs_dict": outputs,
        "input_msg": messages,
        "response_msg": response,
    }


# ==================== 视觉 token 区间 ====================
def _find_vision_range(tokens, processor, vis=True):
    vision_start = vision_end = -1
    for idx, token in enumerate(tokens):
        word = processor.tokenizer.decode([token])
        if word in ("<|image_pad|>", "<|video_pad|>"):
            continue
        if vis:
            print(idx, word, end=" ")
        if word == "<|vision_start|>":
            vision_start = idx
        if word == "<|vision_end|>":
            vision_end = idx

    if vision_start < 0 or vision_end < 0:
        raise RuntimeError("未在输入序列中找到 <|vision_start|> / <|vision_end|>")
    return vision_start + 1, vision_end


# ==================== 注意力提取与可视化 ====================
def extract_attentions(gpt_msg, result, processor, save_path=None, vis=True):
    """
    提取指定输出 token 对视觉 token 的注意力并可视化。
    save_path: None / 目录 / 完整文件路径
    返回: dict(plot, saved_path, check_words, attention_map, concentration)
    """
    output_atts = result["outputs_dict"]["attentions"]
    response = result["response_msg"]
    print(response)

    generated_ids = result["outputs_dict"].sequences
    input_ids = result["inputs_dict"].input_ids

    # ---------- 视觉特征的时空维度 ----------
    if hasattr(result["inputs_dict"], "image_grid_thw"):
        t, h, w = result["inputs_dict"]["image_grid_thw"][0]
        modality = "image"
    elif hasattr(result["inputs_dict"], "video_grid_thw"):
        t, h, w = result["inputs_dict"]["video_grid_thw"][0]
        modality = "video"
    else:
        raise RuntimeError("输入中既无 image_grid_thw 也无 video_grid_thw")

    if vis:
        print("\nbatch, h, w:", (t, h, w))

    # ---------- 输入 token 序列 & 视觉区间 ----------
    generated_ids_all = [
        out_ids[:len(in_ids)]
        for in_ids, out_ids in zip(input_ids, generated_ids)
    ]

    att_range_start, att_range_end = _find_vision_range(
        generated_ids_all[0], processor, vis=vis
    )

    generated_ids_output = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(input_ids, generated_ids)
    ]

    print("\n" + "=" * 50)
    print("输出token序列:")
    for idx, token in enumerate(generated_ids_output[0]):
        word = processor.tokenizer.decode([token])
        print(idx, word, end=" ")
    print("\n")

    print("注意力提取区间:", att_range_start, att_range_end)

    # ---------- 交互式输入：保持原样 ----------
    word_num_start, word_num_end = map(
        int, input("请输入起始和结束索引（空格分割）: ").split()
    )
    print("输入的词索引:", word_num_start, word_num_end)

    out_len = len(generated_ids_output[0])
    if not (0 <= word_num_start < word_num_end <= out_len):
        raise ValueError(
            f"索引越界: [{word_num_start}, {word_num_end}) 超出输出长度 {out_len}"
        )

    check_words = processor.decode(
        generated_ids_output[0][word_num_start:word_num_end],
        skip_special_tokens=False,
    )
    print("\n\n检查关键词:", check_words)

    atts = output_atts[word_num_start:word_num_end]

    # ---------- 动态获取 head 数量 ----------
    num_heads = atts[0][0].shape[1]
    if vis:
        print("注意力头数量:", num_heads)

    # ---------- 组织为 (words, layers, heads, visual_tokens) ----------
    all_words_attention = []
    for word_att in atts:
        all_lays_attention = []
        for lay_att in word_att:
            heads_attention = []
            for head_idx in range(num_heads):
                att_list = lay_att[0, head_idx, 0, att_range_start:att_range_end]
                heads_attention.append(att_list.cpu().float())
            all_lays_attention.append(heads_attention)
        all_words_attention.append(all_lays_attention)

    all_words_attention = np.array(all_words_attention)
    if vis:
        print("all_words_attention shape:", all_words_attention.shape)

    # ---------- 聚合 ----------
    word_att = np.mean(all_words_attention, axis=0)   # (layers, heads, V)
    layer_att = np.mean(word_att, axis=0)             # (heads, V)
    head_att = np.mean(layer_att, axis=0)             # (V,)

    if vis:
        print("word_att shape:", word_att.shape)
        print("layer_att shape:", layer_att.shape)
        print("head_att shape:", head_att.shape)
        print("视觉token数量:", len(generated_ids_all[0][att_range_start:att_range_end]))

    # ---------- reshape ----------
    # Qwen2.5-VL 的 patch_size = 2
    reshape_att = head_att.reshape([t, h // 2, w // 2])
    print("重塑后注意力图 shape:", reshape_att.shape)

    conc = calc_attention_concentration(reshape_att)
    print(f"空间熵: {conc['entropy']:.4f}")
    print(f"基尼系数: {conc['gini']:.4f}")
    print(f"有效面积比: {conc['area_ratio'] * 100:.2f}%")
    print(f"峰值均值比: {conc['peak_mean']:.2f}")
    print(f"方差: {conc['variance']:.6f}")

    # ---------- 可视化 ----------
    if modality == "image":
        file_path = gpt_msg["image"][0]
        img = Image.open(file_path)
        frames = np.array([np.array(img)])
        if vis:
            print("frames.shape:", frames.shape)
            print("head_att.shape:", reshape_att.shape)
        plt_obj, saved_path = overlay_heatmaps_triple(
            frames, reshape_att, check_words,
            alpha=CFG.heatmap_alpha, save_path=save_path,
        )
        return {
            "plot": plt_obj,
            "saved_path": saved_path,
            "check_words": check_words,
            "attention_map": reshape_att,
            "concentration": conc,
        }

    # 视频
    video_path = gpt_msg["video"][0]
    print("正在处理视频数据...", video_path)
    frames, frame_indices = extract_frames(video_path, num_frames=t)
    if vis:
        print("frames.shape:", frames.shape)
        print("reshape_att.shape:", reshape_att.shape)

    plt_obj, saved_path = overlay_heatmaps_triple(
        frames, reshape_att, check_words,
        alpha=CFG.heatmap_alpha, save_path=save_path,
    )
    return {
        "plot": plt_obj,
        "saved_path": saved_path,
        "check_words": check_words,
        "attention_map": reshape_att,
        "concentration": conc,
    }


# ==================== 主流程（保持原结构） ====================
model_path = CFG.model_path

processor = AutoProcessor.from_pretrained(model_path)
model, output_loading_info = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=CFG.dtype,
    device_map="auto",
    output_loading_info=True,
    attn_implementation=CFG.attn_implementation,
)
print("output_loading_info", output_loading_info)


### 2. 有视频
msg = {
    "id": "Dataset/group_1/2024031509524300050569525e9",
    "video": [
        "../../Data/dicom_videos_group/Dataset/group_1/2024031509524300050569525e9/Doppler_Parasternal_Long.mp4"
    ],
    "diag_item": "二尖瓣反流",
    "mark": 1,
    "conversations": [
        {
            "from": "human",
            "value": "<video>\n请分析这段心脏超声视频，请根据超声影像特征判断是否存在二尖瓣反流现象。只输出'是'或'否'",
        },
        {"from": "gpt", "value": "是"},
    ],
}





msg =   {
    "id": "Dataset/group_1/2024042221485800050569525e9",
    "video": [
      "/home/mzhao/ECHO_VIEW/Data/dicom_videos_group/Dataset/group_1/2024042221485800050569525e9/Doppler_Parasternal_Long.mp4"
    ],
    "diag_item": "二尖瓣反流",
    "mark": -1,
    "conversations": [
      {
        "from": "human",
        "value": "<video>\n请分析这段心脏超声视频，请根据超声影像特征判断是否存在二尖瓣反流现象。只输出'是'或'否'"
      },
      {
        "from": "gpt",
        "value": "否"
      }
    ]
  }







msg["conversations"][0]["value"] = (
    "<video>\n请查看这段超声视频，并仔细观察二尖瓣的位置是否有二尖瓣反流的特征，"
    "如果有，描述其严重程度。"
)






root = ""
msg["video"][0] = os.path.join(root, msg["video"][0])
video_url = msg["video"][0]
prompt = msg["conversations"][0]["value"].replace("<video>", "").strip()

video_path, frames, timestamps = get_video_frames(
    video_url, num_frames=CFG.num_frames_cache
)
print(len(frames))

image_grid = create_image_grid(frames, num_columns=CFG.grid_columns)
# display(image_grid.resize((640, 640)))  # noqa: F821  (notebook 环境)

print("视频路径:", video_path)
model.to("cuda:0")

result = inference(model, processor, video_path, prompt)
print(result["response_msg"])

# ============ 保存路径的三种用法（任选其一） ============
# 1) 默认保存到 ./attn_outputs/attn_<关键词>.png
out = extract_attentions(msg, result, processor, save_path=None, vis=True)

# 2) 指定目录
# out = extract_attentions(msg, result, processor, save_path="outputs/", vis=True)

# 3) 指定完整文件路径
# out = extract_attentions(msg, result, processor,
#                          save_path="outputs/mitral_attn.png", vis=True)

print("保存结果:", out["saved_path"])