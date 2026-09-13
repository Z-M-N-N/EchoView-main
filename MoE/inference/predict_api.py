"""推理调用接口: 输入单个超声样本 (id/video/diag_item/mark/conversations), 输出模型预测.

用法:
  1. Python 函数调用:
     from inference.predict_api import InferenceAPI
     api = InferenceAPI(checkpoint_path="outputs_echo_small/checkpoints/best.pt",
                        config_path="configs/echo_small.yaml", device="cuda:0")
     result = api.predict({...})
     print(result)

  2. 命令行 (JSON 文件):
     python -m inference.predict_api --checkpoint outputs_echo_small/checkpoints/best.pt \
         --config configs/echo_small.yaml --input sample.json [--device cuda:0]

  3. 标准输入 (管道):
     echo '{"id": "xxx", "video": [...], "diag_item": "xxx", "mark": -1, "conversations": [...]}' | \
         python -m inference.predict_api --checkpoint xxx.pt --config configs/echo_small.yaml

  4. 全量模型快捷方式:
     python -m inference.predict_api --checkpoint_all --input sample.json

输出字段:
  {
    "id": 原样本 id,
    "diag_item": 疾病名,
    "prediction": "是" | "否",
    "prob_yes": 0~1,     # P(阳性)
    "prob_no":  0~1,     # P(阴性)
    "probabilities": [P(否), P(是)],   # 与模型输出顺序一致
    "logits": [...],
    "label_expected": "是"|"否" (若输入含 mark),
    "video_count": 实际使用视频路数,
    "top1_expert": 路由到的专家编号,
    "router_probs": N 个专家的门控权重
  }
"""
import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.config import load_config
from data.transforms import VideoTransforms
from model.multimodal_moe import MultimodalMoE


class InferenceAPI:
    """单样本推理封装: 从样本 dict -> 模型预测结果."""

    def __init__(self, checkpoint_path: str, config_path: str = "configs/echo_small.yaml",
                 device: Optional[str] = None):
        self.cfg = load_config(config_path)
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.cfg["device"] = device

        torch.manual_seed(self.cfg.get("seed", 42))
        np.random.seed(self.cfg.get("seed", 42))

        self.model = MultimodalMoE(self.cfg).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # 文本 tokenizer (与训练一致: 多语言 BERT)
        self.tokenizer = None
        tcp = self.cfg["model"]["text_encoder"]
        bp = tcp.get("backbone_path")
        if bp and os.path.isdir(bp):
            from transformers import AutoTokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(bp)
            except Exception:
                self.tokenizer = None
        if self.tokenizer is None:
            print(f"[InferenceAPI] 警告: tokenizer 加载失败 ({bp}), 文本将使用占位序列")

        # 视频变换 (与训练一致: uniform 16帧, 224, mean/std 0.5)
        tcfg = self.cfg["data"].get("transforms", {})
        self.transform = VideoTransforms(
            spatial_size=tcfg.get("spatial_size", 224),
            temporal_mode=tcfg.get("temporal_sample", "uniform"),
            random_hflip=False,
        )
        self.num_frames = self.cfg["input"]["num_frames"]
        self.num_videos_max = self.cfg["input"]["num_videos_max"]
        self.frame_size = self.cfg["input"]["frame_size"]
        self.max_seq_len = self.cfg["input"]["max_seq_len"]

    # ---------- 文本处理 ----------
    @staticmethod
    def clean_text(value: str) -> str:
        return re.sub(r"\s+", " ", value.replace("<video>", "").strip())

    def _tokenize(self, text: str):
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                text, max_length=self.max_seq_len,
                padding="max_length", truncation=True, return_tensors="pt")
            return tokens["input_ids"], tokens["attention_mask"]
        return (torch.zeros(1, self.max_seq_len, dtype=torch.long),
                torch.ones(1, self.max_seq_len, dtype=torch.long))

    # ---------- 视频处理 ----------
    def _load_video(self, path: str) -> torch.Tensor:
        """从 mp4 实时抽帧 -> (num_frames, C, H, W) float, 与训练预处理一致."""
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {path}")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError(f"视频无帧: {path}")
        total = len(frames)
        n = self.num_frames
        if total <= n:
            idx = list(range(total))
            while len(idx) < n:
                idx.append(idx[-1])
        else:
            step = total / n
            idx = [min(int((i + 0.5) * step), total - 1) for i in range(n)]
        out = []
        for i in idx:
            f = cv2.resize(frames[i], (self.frame_size, self.frame_size),
                           interpolation=cv2.INTER_LINEAR)
            out.append(f)
        arr = np.stack(out).astype(np.uint8)          # (n,H,W,3) BGR
        arr = arr[..., ::-1].copy()                    # BGR -> RGB
        return self.transform(torch.from_numpy(arr), n)

    def _build_video_tensor(self, video_paths: List[str]):
        """构建 (1,K,T,C,H,W) 视频张量与 (1,K) 掩码; 加载失败的路以零张量占位."""
        videos, mask = [], []
        for p in video_paths[: self.num_videos_max]:
            try:
                videos.append(self._load_video(p))
                mask.append(1)
            except Exception as e:
                print(f"[InferenceAPI] 跳过视频 {p}: {e}")
                videos.append(torch.zeros(
                    self.num_frames, 3, self.frame_size, self.frame_size))
                mask.append(0)
        K = len(videos)
        pad_to = self.num_videos_max
        if K == 0:
            v = torch.zeros(pad_to, self.num_frames, 3, self.frame_size, self.frame_size)
        else:
            v = torch.stack(videos)                    # (K,T,C,H,W)
            if K < pad_to:
                pad = torch.zeros(pad_to - K, self.num_frames, 3,
                                  self.frame_size, self.frame_size)
                v = torch.cat([v, pad], dim=0)
        m = torch.tensor(mask + [0] * (pad_to - len(mask)), dtype=torch.long)
        return v.unsqueeze(0).to(self.device), m.unsqueeze(0).to(self.device)

    # ---------- 主推理 ----------
    @torch.no_grad()
    def predict(self, sample: Dict) -> Dict:
        # 文本
        conv = sample.get("conversations", [])
        text = sample.get("text", "")
        if not text and conv:
            text = self.clean_text(conv[0].get("value", ""))
        elif not text:
            text = self.clean_text(sample.get("diag_item", "") or "")
        input_ids, attn_mask = self._tokenize(text)
        input_ids = input_ids.to(self.device)
        attn_mask = attn_mask.to(self.device)

        # 视频 (兼容 video 列表 或 video_paths 字典)
        video_paths = sample.get("video", [])
        if not video_paths and isinstance(sample.get("video_paths"), dict):
            video_paths = list(sample["video_paths"].values())
        elif not video_paths and isinstance(sample.get("video_paths"), list):
            video_paths = sample["video_paths"]
        video, vmask = self._build_video_tensor(video_paths)

        logits, aux = self.model(
            input_ids, attn_mask, video, vmask,
            use_noise=False, force_top_k=True, return_aux=True)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()   # [P(否), P(是)] (2类)
        logits_np = logits[0].cpu().numpy()
        pred_idx = int(probs.argmax())

        label_names = ["否", "是"]
        result = {
            "id": sample.get("id", ""),
            "diag_item": sample.get("diag_item", ""),
            "text_used": text,
            "prediction": label_names[pred_idx] if pred_idx < len(label_names) else str(pred_idx),
            "prob_yes": float(probs[1]) if len(probs) > 1 else float(probs[0]),
            "prob_no": float(probs[0]),
            "probabilities": [float(x) for x in probs],
            "logits": [float(x) for x in logits_np],
            "video_count": int(vmask[0].sum().item()),
        }
        if "mark" in sample:
            result["label_expected"] = "是" if sample["mark"] == 1 else "否"
        if "top1_expert" in aux:
            result["top1_expert"] = int(aux["top1_expert"][0].item())
        if "g_dense" in aux:
            result["router_probs"] = [float(x) for x in aux["g_dense"][0].cpu().numpy()]
        return result

    # ---------- 批量预测 ----------
    @torch.no_grad()
    def predict_batch(self, samples: List[Dict]) -> List[Dict]:
        return [self.predict(s) for s in samples]


def _read_input(path: Optional[str]) -> Dict:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.load(sys.stdin)


def main():
    ap = argparse.ArgumentParser(description="MoE 超声诊断单样本推理接口")
    ap.add_argument("--checkpoint", default=None, help="checkpoint .pt 路径")
    ap.add_argument("--config", default="configs/echo_small.yaml")
    ap.add_argument("--input", default=None, help="样本 JSON 文件; 缺省从 stdin 读取")
    ap.add_argument("--device", default=None, help="设备, 如 cuda:0 / cuda:4 / cpu (默认 cuda:0)")
    ap.add_argument("--checkpoint_all", action="store_true",
                    help="使用全量训练 checkpoint (outputs_echo/checkpoints/best.pt)")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt = args.checkpoint
    cfg_path = args.config
    if args.checkpoint_all or (ckpt is None and os.path.exists(
            os.path.join(base, "outputs_echo", "checkpoints", "best.pt"))):
        ckpt = os.path.join(base, "outputs_echo", "checkpoints", "best.pt")
        cfg_path = os.path.join(base, "configs", "echo_binary.yaml")
    if ckpt is None:
        ckpt = os.path.join(base, "outputs_echo_small", "checkpoints", "best.pt")
        if not os.path.exists(ckpt):
            print("未找到 checkpoint, 请用 --checkpoint 指定路径")
            sys.exit(1)

    sample = _read_input(args.input)
    if isinstance(sample, list):
        api = InferenceAPI(checkpoint_path=ckpt, config_path=cfg_path, device=args.device)
        outs = api.predict_batch(sample)
        print(json.dumps(outs, ensure_ascii=False, indent=2))
    else:
        api = InferenceAPI(checkpoint_path=ckpt, config_path=cfg_path, device=args.device)
        out = api.predict(sample)
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
