"""生成假数据用于冒烟测试/框架验证.
生成:
  data/index/train.json, val.json
  data/preprocessed/<split>/<id>/<view>.npy  (T,H,W,C) uint8
用法: python scripts/generate_fake_data.py --n_train 32 --n_val 8
"""
import argparse
import json
import os
import random

import numpy as np

VIEWS = ["front", "left", "right", "rear"]
TEXTS = [
    "前方路口请直行", "前方路口请左转", "前方路口请右转",
    "保持当前车道直行", "请向左变道", "请向右变道",
    "test drive straight", "turn left at junction", "turn right at junction",
]


def make_fake_video(size=224, num_frames=16, h=64, w=64, seed=0):
    """生成有简单运动模式的假帧 (移动色块) -> (num_frames, size, size, 3) uint8"""
    rng = np.random.RandomState(seed)
    frames = []
    for t in range(num_frames):
        frame = rng.randint(0, 40, (size, size, 3), dtype=np.uint8)
        dx = int((t / num_frames) * (w // 2))
        color = (200, 60, 60)
        frame[20:20 + h, 20 + dx:20 + dx + w] = color
        frames.append(frame)
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=32)
    ap.add_argument("--n_val", type=int, default=8)
    ap.add_argument("--out", default="data", help="数据输出根目录")
    args = ap.parse_args()

    root = args.out
    index_dir = os.path.join(root, "index")
    os.makedirs(index_dir, exist_ok=True)

    for split, n in [("train", args.n_train), ("val", args.n_val)]:
        samples = []
        for i in range(n):
            sid = f"{split}_{i:05d}"
            srcdir = os.path.join(root, "preprocessed", split, sid)
            os.makedirs(srcdir, exist_ok=True)
            video_paths = {}
            k = random.randint(1, 4)                      # 动态 K (1~4 路)
            chosen = random.sample(VIEWS, k)
            for view in chosen:
                vp = os.path.join(srcdir, f"{view}.npy")
                frames = make_fake_video(seed=random.randint(0, 10000))
                np.save(vp, frames)
                # 索引路径相对项目根 (train.py 会 chdir 到项目根), 带 data/ 前缀
                video_paths[view] = os.path.join("data", "preprocessed", split, sid, f"{view}.npy")
            samples.append({
                "id": sid,
                "text": random.choice(TEXTS),
                "video_paths": video_paths,
                "label": random.randint(0, 2),
                "metadata": {"weather": random.choice(["sunny", "rainy"]), "scene": "urban"},
            })
        out_path = os.path.join(index_dir, f"{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"[generate_fake_data] {split}: {n} 条 -> {out_path}")


if __name__ == "__main__":
    main()
