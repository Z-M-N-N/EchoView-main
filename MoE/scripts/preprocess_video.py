"""视频预处理: 解码、抽帧、缩放到 224 —— 规格书 §8 scripts/preprocess_video.py
将原始视频抽帧为 .npy (T,H,W,C) uint8 缓存, 供 dataset.py 的 npy 模式加载.
用法:
  python scripts/preprocess_video.py --video path/video.mp4 --out data/preprocessed/x.mp4.npy --num_frames 16
"""
import argparse
import os

import cv2
import numpy as np


def uniform_indices(total: int, n: int) -> np.ndarray:
    if total <= n:
        idx = np.arange(total)
        pad = np.repeat(total - 1, n - total)
        return np.concatenate([idx, pad]).astype(np.int64)
    step = total / n
    return np.array([min(int((i + 0.5) * step), total - 1) for i in range(n)], dtype=np.int64)


def extract_frames(video_path: str, num_frames: int = 16, size: int = 224) -> np.ndarray:
    """读取视频 -> (num_frames, size, size, 3) uint8."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"视频无帧: {video_path}")

    total = len(frames)
    idx = uniform_indices(total, num_frames)
    out = []
    for i in idx:
        f = cv2.resize(frames[i], (size, size), interpolation=cv2.INTER_LINEAR)
        out.append(f)
    return np.stack(out).astype(np.uint8)   # (T, H, W, C) BGR注意顺序


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    frames = extract_frames(args.video, args.num_frames, args.size)
    # 统一转 RGB (视频处理常规; 若背噪要求 BGR 可去掉)
    frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    np.save(args.out, frames)
    print(f"已保存 {args.out} 形状={frames.shape}")


if __name__ == "__main__":
    main()
