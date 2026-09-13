"""从 Echo-View_Result 生成 MoE 项目数据索引 —— 超声诊断 (是/否) 二分类任务.

输入: Echo-View_Result/<mode>/*.json  (28 个疾病结果集, 每条含 id/video/diag_item/mark/conversations)
输出: MoE/data/index/train.json, val.json
每条样本格式 (规格书 §3.2 适配):
{
  "id": 原患者 id,
  "text": 诊断问题 (来自 conversations human, 去掉 <video> 标记),
  "video_paths": {"视图名": "绝对路径", ...},   # 动态 K (1~6 路)
  "label": 0|1,                                # mark -1 -> 0, 1 -> 1
  "metadata": {"diag_item": "疾病名"}
}
切分策略: 按患者 id 分层随机切分 (避免同一患者泄漏到训练/验证)。
用法:
  python scripts/build_echo_index.py --src /home/mzhao/ECHO_VIEW/Data/dicom_videos_group/Echo-View_Result \
      --mode mcd --out data/index --val_ratio 0.15
"""
import argparse
import glob
import json
import os
import re
import random
from collections import defaultdict

import numpy as np


def clean_text(value: str) -> str:
    """去掉 <video> 标记与空白, 保留诊断问题."""
    t = value.replace("<video>", "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def view_name(path: str) -> str:
    """从绝对路径提取视图名: /.../A4C.mp4 -> A4C."""
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def load_disease_file(path: str, keep_all: bool = True):
    """读取单个疾病结果 JSON -> 样本列表.
    keep_all=True 保留所有 results; 否则只保留 response 标记为正确的样本 (可选).
    """
    d = json.load(open(path, encoding="utf-8"))
    out = []
    for r in d.get("results", []):
        msg = r.get("msg", {})
        if not msg.get("video"):
            continue
        vid = msg.get("video")
        m = msg.get("mark", -1)
        conv = msg.get("conversations", [])
        text = ""
        if conv and conv[0].get("value"):
            text = clean_text(conv[0]["value"])
        vp = {}
        for v in vid:
            vp[view_name(v)] = v
        out.append({
            "id": msg.get("id", ""),
            "text": text,
            "video_paths": vp,
            "label": 1 if m == 1 else 0,
            "metadata": {"diag_item": msg.get("diag_item", "")},
        })
    return out


def split_by_patient(samples, val_ratio: float, seed: int):
    """按患者 id 分层切分 (同患者所有样本进同一集合)."""
    rng = random.Random(seed)
    by_patient = defaultdict(list)
    for s in samples:
        by_patient[s["id"]].append(s)
    patients = list(by_patient.keys())
    # 先按患者 label 分布分层
    pos_patients = [p for p in patients if any(s["label"] == 1 for s in by_patient[p])]
    neg_patients = [p for p in patients if p not in set(pos_patients)]
    rng.shuffle(pos_patients)
    rng.shuffle(neg_patients)
    n_val_pos = max(1, int(len(pos_patients) * val_ratio))
    n_val_neg = max(1, int(len(neg_patients) * val_ratio))
    val_patients = set(pos_patients[:n_val_pos] + neg_patients[:n_val_neg])
    train, val = [], []
    for s in samples:
        (val if s["id"] in val_patients else train).append(s)
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/mzhao/ECHO_VIEW/Data/dicom_videos_group/Echo-View_Result",
                    help="Echo-View_Result 根目录")
    ap.add_argument("--mode", default="mcd", choices=["mcd", "cmd", "pvd"],
                    help="使用哪个评估集 (mcd=主要+辅助切面, 推荐)")
    ap.add_argument("--out", default="data/index", help="索引输出目录 (相对 MoE 项目根)")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0,
                    help="每疾病最多取多少条 (调试用, 0=全部)")
    args = ap.parse_args()

    src_dir = os.path.join(args.src, args.mode)
    files = sorted(glob.glob(os.path.join(src_dir, "*.json")))
    files = [f for f in files if not f.endswith("result.xlsx")]
    print(f"[build_echo_index] 读取 {src_dir} 下 {len(files)} 个疾病 json")

    all_samples = []
    per_diag = {}
    for f in files:
        diag = os.path.splitext(os.path.basename(f))[0]
        if diag == "result":
            continue
        samples = load_disease_file(f)
        if args.limit > 0:
            samples = samples[:args.limit]
        per_diag[diag] = samples
        all_samples.extend(samples)
        pos = sum(1 for s in samples if s["label"] == 1)
        print(f"  {diag}: {len(samples)} 条 (阳性 {pos})")

    print(f"\n合计: {len(all_samples)} 条")
    if not all_samples:
        print("无样本, 退出")
        return

    train, val = split_by_patient(all_samples, args.val_ratio, args.seed)
    pos_tr = sum(1 for s in train if s["label"] == 1)
    pos_va = sum(1 for s in val if s["label"] == 1)
    print(f"train: {len(train)} 条 (阳性 {pos_tr}, {pos_tr/len(train)*100:.1f}%)")
    print(f"val  : {len(val)} 条 (阳性 {pos_va}, {pos_va/len(val)*100:.1f}%)")

    os.makedirs(args.out, exist_ok=True)
    for name, data in [("train", train), ("val", val)]:
        p = os.path.join(args.out, f"{name}.json")
        with open(p, "w", encoding="utf-8") as fo:
            json.dump(data, fo, ensure_ascii=False)
        print(f"  写入 {p} ({len(data)} 条)")

    # 额外: 按疾病拆分索引目录 (供需要独立训练的疾病使用)
    diag_dir = os.path.join(args.out, "by_diag")
    os.makedirs(diag_dir, exist_ok=True)
    for diag, samples in per_diag.items():
        tr = [s for s in train if s["metadata"]["diag_item"] == diag]
        va = [s for s in val if s["metadata"]["diag_item"] == diag]
        with open(os.path.join(diag_dir, f"{diag}_train.json"), "w", encoding="utf-8") as fo:
            json.dump(tr, fo, ensure_ascii=False)
        with open(os.path.join(diag_dir, f"{diag}_val.json"), "w", encoding="utf-8") as fo:
            json.dump(va, fo, ensure_ascii=False)
    print(f"  按疾病拆分索引已写入 {diag_dir}")

    # 示例
    print("\n示例样本:")
    print(json.dumps(train[0], ensure_ascii=False, indent=2)[:600])


if __name__ == "__main__":
    main()
