"""模型下载脚本 (使用 hf-mirror 镜像, 因为 g5 无法直连 huggingface.co).

用法:
  # 下载中文 BERT (处理中文诊断问题, 推荐)
  python scripts/download_models.py --model bert-base-chinese

  # 下载英文 BERT (规格书默认, 但无法处理中文文本)
  python scripts/download_models.py --model bert-base-uncased

  # 下载到指定目录 (默认 weights/<模型名>)
  python scripts/download_models.py --model bert-base-chinese --out weights/bert-base-chinese
"""
import argparse
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 关键: 走镜像


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-chinese",
                    choices=["bert-base-chinese", "bert-base-uncased"],
                    help="要下载的文本编码器")
    ap.add_argument("--out", default=None, help="输出目录 (默认 weights/<model>)")
    args = ap.parse_args()

    out = args.out or os.path.join("weights", args.model)
    os.makedirs(out, exist_ok=True)

    print(f"通过镜像 {os.environ['HF_ENDPOINT']} 下载 {args.model} -> {out}")
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id=f"google-bert/{args.model}",
        local_dir=out,
    )
    print(f"✅ 下载完成: {path}")


if __name__ == "__main__":
    main()
