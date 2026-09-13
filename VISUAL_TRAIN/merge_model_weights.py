
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
import os
import argparse
from pathlib import Path
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
def parse_args():
    parser = argparse.ArgumentParser(description="Merge fine-tuned visual weights into Qwen2.5-VL model")
    
    # 必需参数
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the fine-tuned checkpoint file (.pth)"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the base Qwen2.5-VL model"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the merged model"
    )
    
    # 可选参数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to load model on (default: cuda if available else cpu)"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model loading (default: auto)"
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        default=True,
        help="Use safetensors for saving (default: True)"
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_false",
        dest="safe_serialization",
        help="Disable safetensors, use .bin instead"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code for tokenizer/processor (default: True)"
    )
    parser.add_argument(
        "--no_trust_remote_code",
        action="store_false",
        dest="trust_remote_code",
        help="Disable trust_remote_code"
    )
    
    return parser.parse_args()

def load_checkpoint(checkpoint_path, device):
    """加载checkpoint文件"""
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    
    # 将设备转换为torch.device对象
    if device == "cuda" and torch.cuda.is_available():
        map_location = "cuda"
    elif device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU instead")
        map_location = "cpu"
    else:
        map_location = "cpu"
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    
    print(f"Checkpoint keys: {checkpoint.keys()}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Val Acc: {checkpoint.get('val_acc', 'N/A')}")
    
    return checkpoint

def extract_vision_weights(state_dict):
    """从state_dict中提取视觉权重"""
    vision_state_dict = {}
    
    for k, v in state_dict.items():
        if k.startswith('visual.'):
            new_key = k.replace('visual.', '')
            vision_state_dict[new_key] = v
    
    print(f"\n📊 Extracted {len(vision_state_dict)} visual weight keys")
    
    # 显示转换示例
    if len(vision_state_dict) > 0:
        print("键名转换示例:")
        old_keys = [k for k in state_dict.keys() if k.startswith('visual.')][:3]
        new_keys = list(vision_state_dict.keys())[:3]
        for old, new in zip(old_keys, new_keys):
            print(f"  {old} → {new}")
    
    return vision_state_dict

def load_base_model(base_model_path, device, torch_dtype):
    """加载基础模型"""
    print(f"\n📦 Loading base model from: {base_model_path}")
    
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    
    # device_map支持"auto"，但用于transformer的from_pretrained
    device_map = "auto" if device == "cuda" else device
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=dtype_map.get(torch_dtype, "auto"),
        device_map=device_map
    )
    
    return model

def load_tokenizer_processor(base_model_path, trust_remote_code):
    """加载tokenizer和processor"""
    print("\n🔧 Loading tokenizer and processor...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=trust_remote_code
    )
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=trust_remote_code
    )
    
    return tokenizer, processor

def merge_vision_weights(model, vision_state_dict):
    """合并视觉权重到模型"""
    print("\n🔄 Merging visual weights into model...")
    
    missing, unexpected = model.visual.load_state_dict(vision_state_dict, strict=False)
    
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    
    if len(missing) == 0 and len(unexpected) == 0:
        print("✅ 视觉权重加载完全成功！")
    else:
        if missing:
            print(f"⚠️ 缺失的键（前10个）:")
            for k in missing[:10]:
                print(f"  - {k}")
        if unexpected:
            print(f"⚠️ 意外的键（前10个）:")
            for k in unexpected[:10]:
                print(f"  - {k}")
    
    return missing, unexpected

def save_model(model, tokenizer, processor, save_dir, safe_serialization):
    """保存完整模型"""
    print(f"\n💾 Saving model to: {save_dir}")
    
    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    model.save_pretrained(save_dir, safe_serialization=safe_serialization)
    print(f"✅ Model saved to {save_dir}")
    
    # 保存tokenizer
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Tokenizer saved to {save_dir}")
    
    # 保存processor
    processor.save_pretrained(save_dir)
    print(f"✅ Processor saved to {save_dir}")

def verify_loaded_model(save_dir, trust_remote_code):
    """验证保存的模型可以正确加载"""
    print("\n🔍 Verifying saved model...")
    
    try:
        loaded_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            save_dir,
            torch_dtype="auto",
            device_map="auto"
        )
        loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=trust_remote_code)
        loaded_processor = AutoProcessor.from_pretrained(save_dir, trust_remote_code=trust_remote_code)
        
        print("✅ Model, tokenizer and processor loaded successfully!")
        print(f"   - Model device: {loaded_model.device}")
        print(f"   - Model dtype: {loaded_model.dtype}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading verification failed: {e}")
        return False

def print_file_list(save_dir):
    """打印保存的文件列表"""
    print(f"\n📁 Saved files:")
    total_size = 0
    
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            total_size += size
            
            # 转换为合适的单位
            if size > 1024**3:
                size_str = f"{size / (1024**3):.2f} GB"
            elif size > 1024**2:
                size_str = f"{size / (1024**2):.2f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.2f} KB"
            else:
                size_str = f"{size} B"
            
            print(f"  - {file} ({size_str})")
    
    # 总大小
    if total_size > 1024**3:
        total_str = f"{total_size / (1024**3):.2f} GB"
    elif total_size > 1024**2:
        total_str = f"{total_size / (1024**2):.2f} MB"
    else:
        total_str = f"{total_size / 1024:.2f} KB"
    
    print(f"\n📊 Total size: {total_str}")

def main():
    # 解析命令行参数
    args = parse_args()
    
    print("=" * 60)
    print("🚀 Qwen2.5-VL Model Merger")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Base model: {args.base_model_path}")
    print(f"Save dir: {args.save_dir}")
    print(f"Device: {args.device}")
    print(f"Torch dtype: {args.torch_dtype}")
    print(f"Safe serialization: {args.safe_serialization}")
    print("=" * 60)
    
    try:
        # 1. 加载checkpoint
        checkpoint = load_checkpoint(args.checkpoint_path, args.device)
        
        # 2. 提取视觉权重
        vision_state_dict = extract_vision_weights(checkpoint['model_state_dict'])
        
        # 3. 加载基础模型
        model = load_base_model(args.base_model_path, args.device, args.torch_dtype)
        
        # 4. 合并视觉权重
        merge_vision_weights(model, vision_state_dict)
        
        # 5. 加载tokenizer和processor
        tokenizer, processor = load_tokenizer_processor(args.base_model_path, args.trust_remote_code)
        
        # 6. 保存模型
        save_model(model, tokenizer, processor, args.save_dir, args.safe_serialization)
        
        # 7. 验证
        verify_loaded_model(args.save_dir, args.trust_remote_code)
        
        # 8. 显示文件列表
        print_file_list(args.save_dir)
        
        # 9. 显示使用说明
        print("\n" + "=" * 60)
        print("🎉 All steps completed successfully!")
        print("\n📖 Usage:")
        print(f"from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor")
        print(f"model = Qwen2_5_VLForConditionalGeneration.from_pretrained('{args.save_dir}')")
        print(f"processor = AutoProcessor.from_pretrained('{args.save_dir}')")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())