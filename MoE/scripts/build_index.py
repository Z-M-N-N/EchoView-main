"""生成训练集/验证集 JSON 索引文件 —— 规格书 §3.2 / §8 scripts/build_index.py
从目录结构生成:
  index/
    train.json / val.json
目录约定: (每样本一个目录)
  <split>/
    <sample_id>/
      text.txt              # 文本指令
      label.txt             # 0/1/2
      front.mp4 或 front.npy
      left.mp4 / right.mp4 / rear.mp4  (可选)
      metadata.json         # {"weather":"sunny","scene":"urban"} (可选)
"""
import argparse
import json
import os


VIEW_KEYS = ["front", "left", "right", "rear"]


def collect_samples(root: str, split: str):
    split_dir = os.path.join(root, split)
    samples = []
    if not os.path.isdir(split_dir):
        return samples
    video_exts = (".mp4", ".avi", ".npy", ".npz")
    for sid in sorted(os.listdir(split_dir)):
        sdir = os.path.join(split_dir, sid)
        if not os.path.isdir(sdir):
            continue
        # 文本
        text = ""
        tpath = os.path.join(sdir, "text.txt")
        if os.path.exists(tpath):
            text = open(tpath, encoding="utf-8").read().strip()
        # 标签
        lpath = os.path.join(sdir, "label.txt")
        if not os.path.exists(lpath):
            continue
        label = int(open(lpath).read().strip())
        # 视频路
        video_paths = {}
        for key in VIEW_KEYS:
            for ext in video_exts:
                cand = os.path.join(sdir, f"{key}{ext}")
                if os.path.exists(cand):
                    video_paths[key] = os.path.relpath(cand, split_dir)
                    break
        # 元数据
        metadata = {}
        mpath = os.path.join(sdir, "metadata.json")
        if os.path.exists(mpath):
            metadata = json.load(open(mpath, encoding="utf-8"))
        samples.append({
            "id": sid,
            "text": text,
            "video_paths": video_paths,
            "label": label,
            "metadata": metadata,
        })
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="数据根目录(含 train/ val/ 子目录)")
    ap.add_argument("--out", default="data/index", help="索引输出目录")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for split in ["train", "val"]:
        samples = collect_samples(args.root, split)
        out_path = os.path.join(args.out, f"{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"[build_index] {split}: {len(samples)} 条 -> {out_path}")


if __name__ == "__main__":
    main()
