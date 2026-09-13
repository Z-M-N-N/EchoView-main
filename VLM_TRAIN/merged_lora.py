

from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
print("="*50)
print("GPU 信息:")
print("="*50)

# 检查CUDA是否可用
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 当前设备信息
    current_device = torch.cuda.current_device()
    print(f"当前设备索引: {current_device}")
    print(f"当前设备名称: {torch.cuda.get_device_name(current_device)}")
    print(f"可见GPU数量: {torch.cuda.device_count()}")
    
    # 显示所有可见显卡
    for i in range(torch.cuda.device_count()):
        print(f"\n显卡 {i}:")
        print(f"  名称: {torch.cuda.get_device_name(i)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # 当前显存使用情况
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  已分配显存: {allocated:.2f} GB")
        print(f"  保留显存: {reserved:.2f} GB")
    
    print("="*50)
else:
    print("警告: CUDA不可用，将使用CPU训练")

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True, help="基座模型路径")
    parser.add_argument("--lora_checkpoint", type=str, required=True, help="LoRA检查点路径")
    parser.add_argument("--save_dir", type=str, required=True, help="保存路径")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 50)
    print("加载 LoRA 模型")
    print("=" * 50)
    print(f"基座模型: {args.base_model_path}")
    print(f"LoRA路径: {args.lora_checkpoint}")
    print(f"保存路径: {args.save_dir}")
    print("=" * 50)
    
    # 1. 加载基座模型
    print("\n📦 加载基座模型...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device == "cuda" else args.device
    )
    
    # 2. 加载 LoRA
    print("🔧 加载 LoRA 适配器...")
    peft_model = PeftModel.from_pretrained(base_model, args.lora_checkpoint)
    
    # 3. 融合并保存
    print("🔄 融合 LoRA 权重...")
    merged_model = peft_model.merge_and_unload()
    
    # 4. 保存
    print(f"💾 保存到: {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)
    merged_model.save_pretrained(args.save_dir)
    
    # 5. 保存 processor
    processor = AutoProcessor.from_pretrained(args.base_model_path)
    processor.save_pretrained(args.save_dir)
    
    print("✅ 完成！")
    print(f"\n使用方式:")
    print(f"model = Qwen2_5_VLForConditionalGeneration.from_pretrained('{args.save_dir}')")
    print(f"processor = AutoProcessor.from_pretrained('{args.save_dir}')")

if __name__ == "__main__":
    main()