#!/usr/bin/env python
"""chat_moe.py — MoE 路由 + Qwen2.5-VL 超声诊断问答 (3 类路由: PVD/MCD/CMD)

流程:
  1. 输入样本: JSON 格式 (id / video[] / diag_item / mark / conversations)
     - 支持单样本, 也支持样本列表 (批量)
  2. MoE 路由器推理 -> 3 类概率 [PVD, MCD, CMD], argmax 选中类别
  3. 按类别加载对应 Qwen2.5-VL 模型 (批量时模型缓存复用, 同类不重复加载):
       0 -> PVD  (vlm_best_merged_model_PVD, 主要切面 1 路)
       1 -> MCD  (vlm_best_merged_model_MCD, 主要+辅助切面)
       2 -> CMD  (vlm_best_merged_model_CMD, 所有切面)
  4. 按类别从 diag_info.json 筛选视频子集 (PVD 最少 -> CMD 最多)
  5. 用选中模型跑 chat.py 同款问答, 输出诊断结果

用法:
  # 单样本 JSON 文件
  python chat_moe.py --input sample.json [--json]

  # 批量: 文件内为 JSON 数组 -> 逐条推理, 输出结果数组
  python chat_moe.py --input batch.json --json > result.json

  # 从标准输入读取 (管道)
  echo '{...}' | python chat_moe.py --json

  # 快速测试: 单视频 + 问题
  python chat_moe.py --video xx/A4C.mp4 --question "..." --diag_item 二尖瓣反流

  # 自定义 MoE checkpoint / 配置 (默认: 全量 3 类路由 best.pt, 自动回退小规模)
"""
import argparse
import json
import os
import re
import sys

# ---------- 可选 GPU (可在命令行 --gpu 覆盖) ----------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")   # 默认空闲卡; 训练卡勿用

_EV_ROOT = os.path.dirname(os.path.abspath(__file__))           # EchoView-main
_MOE_ROOT = os.path.join(_EV_ROOT, "MoE")
_MODEL_ROOT = os.path.join(_EV_ROOT, "..", "weights_lora", "Model")
_DIAG_INFO = os.path.join(_EV_ROOT, "DATA_BUILDER", "diag_info.json")

# MoE 3 类路由 -> 诊断模型子目录名
CLASS_NAMES = {0: "PVD", 1: "MCD", 2: "CMD"}
CLASS_MODELS = {
    0: "vlm_best_merged_model_PVD",
    1: "vlm_best_merged_model_MCD",
    2: "vlm_best_merged_model_CMD",
}
CLASS_VIEW_DESC = {
    0: "主要切面(1路)",
    1: "主要+辅助切面",
    2: "所有切面",
}


def log(*args):
    """进度/提示信息 -> stderr, 保证 stdout 只输出结果 JSON."""
    print(*args, file=sys.stderr, flush=True)


def load_diag_info():
    """加载 28 个疾病的切面配置 (primary_views / supplementary_views / prompt)."""
    try:
        with open(_DIAG_INFO, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"[chat_moe] 警告: 无法加载 diag_info.json ({e}), 视频筛选走 fallback")
        return {}


def normalize_view(view: str) -> str:
    """视图名 'Parasternal Long' -> 文件名匹配键 'Parasternal_Long'."""
    return view.replace(" ", "_").strip()


def select_videos_for_class(video_paths, cls_idx, diag_info, diag_item):
    """按类别 + 疾病切面配置筛选视频子集.

    PVD(0): primary_views[0] 1 路;  MCD(1): primary+supplementary;  CMD(2): 全部.
    返回 (videos_used, desc).
    """
    paths = list(video_paths or [])
    if not paths:
        return [], "无视频"

    if cls_idx == 2:                                   # CMD: 所有切面
        return paths, f"全部 {len(paths)} 路视频"

    by_name = {}
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        by_name.setdefault(base, p)

    views = []
    if diag_item and diag_item in diag_info:
        info = diag_info[diag_item]
        if cls_idx == 0:
            views = [info["primary_views"][0]] if info.get("primary_views") else []
        else:  # cls_idx == 1 (MCD)
            views = list(info.get("primary_views", [])) + list(info.get("supplementary_views", []))

        matched = [by_name[normalize_view(v)] for v in views if normalize_view(v) in by_name]
        if matched:
            kind = "主要切面" if cls_idx == 0 else "主要+辅助切面"
            return matched, f"{kind} {len(matched)}/{len(views)} 路视频匹配"

    # fallback (无 diag_info 或匹配失败): PVD 用第 1 路, MCD 用前 2 路
    n = 1 if cls_idx == 0 else 2
    return paths[:n], f"fallback 前 {n} 路视频"


def extract_question(sample, diag_info):
    """从样本提取诊断问题文本 (conversations 优先, 其次 diag_item / text)."""
    conv = sample.get("conversations") or []
    for c in conv:
        if c.get("from") == "human":
            t = re.sub(r"<video>", "", c.get("value", "")).strip()
            if t:
                return t
    if sample.get("text"):
        return sample["text"]
    d = sample.get("diag_item", "")
    if d:
        info = diag_info.get(d, {})
        return info.get("prompt", f"请查看视频,根据超声影像特征判断是否存在{d},输出是或否.")
    return "请分析这段心脏超声视频,判断是否存在异常,只输出是或否"


def extract_yes_no(text: str) -> str:
    """从模型输出文本提取 是/否."""
    t = text.strip()
    m = re.search(r"[是|否]", t)
    if m:
        return m.group(0)
    return t if t else ""


def build_messages(videos_used, question):
    """构造 Qwen2.5-VL messages (多视频: 每路一个 video item)."""
    content = []
    n = len(videos_used)
    fps = 16.0 if n <= 1 else max(2.0, 16.0 / n)
    for vp in videos_used:
        content.append({
            "type": "video",
            "video": vp,
            "max_pixels": 224 * 224 * 3,
            "fps": fps,
        })
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


# ---------- Qwen 模型加载 + 推理 (支持缓存复用) ----------
_qwen_cache = {}   # model_name -> (model, processor)


def load_qwen(model_path):
    """加载 Qwen2.5-VL 模型, 带缓存 (同路径只加载一次)."""
    if model_path in _qwen_cache:
        return _qwen_cache[model_path]
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    log(f"[chat_moe] 加载 Qwen 模型: {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)
    _qwen_cache[model_path] = (model, processor)
    return model, processor


def run_qwen(model, processor, videos_used, question, max_new_tokens=16):
    """用已加载模型推理, 返回 (output_text, answer)."""
    import torch
    from qwen_vl_utils import process_vision_info

    messages = build_messages(videos_used, question)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt", **video_kwargs)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    out = output_text[0].strip() if output_text else ""
    return out, extract_yes_no(out)


# ---------- 单样本处理 ----------
def process_sample(sample, api, diag_info, verbose=False):
    """处理单个样本, 返回结果 dict."""
    video_paths = sample.get("video", [])
    if not video_paths and isinstance(sample.get("video_paths"), dict):
        video_paths = list(sample["video_paths"].values())
    diag_item = sample.get("diag_item", "")
    question = extract_question(sample, diag_info)

    # 1. MoE 路由
    moe_result = api.predict(sample)
    probs = moe_result.get("probabilities") or moe_result.get("router_probs") or [0.0, 0.0, 0.0]
    if len(probs) < 3:
        log(f"[chat_moe] 警告: MoE 输出 {len(probs)} 维, 非 3 类路由器")
        probs = list(probs) + [0.0] * (3 - len(probs))
    cls_idx = int(max(range(3), key=lambda i: probs[i]))
    cls_name = CLASS_NAMES[cls_idx]
    model_name = CLASS_MODELS[cls_idx]

    # 2. 视图筛选
    videos_used, view_desc = select_videos_for_class(video_paths, cls_idx, diag_info, diag_item)

    if verbose:
        log(f"[chat_moe] MoE 3类概率: PVD={probs[0]:.4f} MCD={probs[1]:.4f} CMD={probs[2]:.4f}")
        log(f"[chat_moe] logits: {[round(x,4) for x in moe_result.get('logits',[])]}")
        for k in ("video_count", "top1_expert"):
            if k in moe_result:
                log(f"[chat_moe] {k}: {moe_result[k]}")

    log(f"[chat_moe] 路由: {cls_name} ({CLASS_VIEW_DESC[cls_idx]}) -> {model_name} | 视频: {view_desc}")

    if not videos_used:
        return {
            "id": sample.get("id", ""), "diag_item": diag_item,
            "selected_class": cls_idx, "selected_model": cls_name,
            "router_probs": [round(float(p), 4) for p in probs],
            "view_desc": view_desc, "videos_used": [],
            "question": question, "qwen_output": "", "answer": "", "error": "无可用视频",
        }

    # 3. Qwen 问答 (缓存复用)
    model_path = os.path.join(_MODEL_ROOT, model_name)
    model, processor = load_qwen(model_path)
    out_text, answer = run_qwen(model, processor, videos_used, question)

    result = {
        "id": sample.get("id", ""),
        "diag_item": diag_item,
        "selected_class": cls_idx,
        "selected_model": cls_name,
        "model_path": model_path,
        "view_desc": view_desc,
        "router_probs": [round(float(p), 4) for p in probs],
        "videos_used": videos_used,
        "question": question,
        "qwen_output": out_text,
        "answer": answer,
    }
    if sample.get("mark") is not None:
        result["label_expected"] = "是" if sample["mark"] == 1 else "否"
        result["correct"] = (answer == result["label_expected"])
    return result


def main():
    ap = argparse.ArgumentParser(description="MoE 路由 + Qwen 超声诊断问答 (3 类路由)")
    ap.add_argument("--input", default=None, help="样本 JSON 文件 (单样本或数组=批量); 缺省从 stdin 读取")
    ap.add_argument("--video", default=None, help="快速模式: 单视频路径")
    ap.add_argument("--question", default=None, help="快速模式: 诊断问题")
    ap.add_argument("--diag_item", default=None, help="诊断项 (用于视频切面筛选)")
    ap.add_argument("--moe_checkpoint",
                    default=os.path.join(_MOE_ROOT, "outputs_echo_router", "checkpoints", "best.pt"),
                    help="MoE checkpoint (默认: 全量 3 类路由 best.pt)")
    ap.add_argument("--moe_config",
                    default=os.path.join(_MOE_ROOT, "configs", "echo_router.yaml"),
                    help="MoE 配置 (默认: echo_router.yaml)")
    ap.add_argument("--gpu", default="5", help="CUDA_VISIBLE_DEVICES, 默认 5 (空闲 A6000)")
    ap.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    ap.add_argument("--verbose", action="store_true", help="打印 MoE 路由细节")
    ap.add_argument("--max_batch", type=int, default=0, help="批量上限, 0=不限 (调试用)")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ---------- 读取样本 (文件 / stdin / --video 快速模式) ----------
    samples = []
    is_batch = False
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        is_batch = isinstance(data, list)
        samples = data if is_batch else [data]
    elif args.video:
        is_batch = False
        samples = [{
            "id": os.path.basename(os.path.dirname(args.video)) or "sample",
            "video": [args.video],
            "diag_item": args.diag_item or "",
            "mark": None,
            "conversations": [{"from": "human", "value": "<video>\n" + (args.question or "")}],
        }]
    else:
        data = json.load(sys.stdin)
        is_batch = isinstance(data, list)
        samples = data if is_batch else [data]

    if args.max_batch > 0:
        samples = samples[:args.max_batch]

    # ---------- MoE API (一次性加载) ----------
    if not os.path.exists(args.moe_checkpoint):
        alt = os.path.join(_MOE_ROOT, "outputs_echo_router_small", "checkpoints", "best.pt")
        if os.path.exists(alt):
            log(f"[chat_moe] 提示: 未找到 {args.moe_checkpoint}")
            log(f"[chat_moe] 自动回退到小规模 3 类路由模型: {alt}")
            args.moe_checkpoint = alt
        else:
            log(f"[chat_moe] 错误: MoE checkpoint 不存在: {args.moe_checkpoint}")
            sys.exit(1)

    sys.path.insert(0, _MOE_ROOT)
    from inference.predict_api import InferenceAPI
    log(f"[chat_moe] MoE checkpoint: {args.moe_checkpoint}")
    api = InferenceAPI(checkpoint_path=args.moe_checkpoint, config_path=args.moe_config)

    diag_info = load_diag_info()

    # ---------- 逐条处理 ----------
    results = []
    n = len(samples)
    for i, sample in enumerate(samples, 1):
        log(f"\n[chat_moe] === 样本 {i}/{n} id={sample.get('id','')} diag_item={sample.get('diag_item','')} ===")
        try:
            r = process_sample(sample, api, diag_info, verbose=args.verbose)
        except Exception as e:
            import traceback
            log(f"[chat_moe] 样本 {i} 处理失败: {e}")
            if args.verbose:
                traceback.print_exc()
            r = {"id": sample.get("id", ""), "diag_item": sample.get("diag_item", ""),
                 "error": str(e)}
        results.append(r)
        if not args.json:
            log(f"[chat_moe] 样本 {i} 结果: {r.get('answer','?')} (模型={r.get('selected_model','?')})")

    # ---------- 输出 ----------
    # 原始输入为列表 -> 输出数组; 单样本 -> 输出单对象
    out = results if is_batch else (results[0] if results else None)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        log(f"\n[chat_moe] 共处理 {len(results)} 条, 完成.")


if __name__ == "__main__":
    main()
