import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from myqwen_vl_utils import process_vision_info
from vision_multitask_model import VisionMultiTask

MODEL_ID = "/home/mzhao/公共大模型/Qwen2.5-VL-7B-Instruct"
print("加载 Qwen2.5-VL-7B ...")
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True).to("cuda")
processor = AutoProcessor.from_pretrained(MODEL_ID)
vis = base.visual

# 用与训练完全一致的管线生成视频输入（合成16帧视频）
from PIL import Image
frames = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(16)]
messages = [{"role": "user", "content": [
    {"type": "video", "video": frames, "nframes": 16,
     "min_pixels": 224 * 224, "max_pixels": 224 * 224}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(text=text, images=images, videos=videos, padding=True,
                   return_tensors="pt", **video_kwargs)
pv = inputs["pixel_values_videos"].unsqueeze(0).to("cuda")   # [1, M, D] 加 batch 维
grid = inputs["video_grid_thw"].to("cuda")                     # 已是 [1, 3]
print("输入形状 pixel_values:", tuple(pv.shape), "grid_thw:", grid.tolist())

import copy
m = VisionMultiTask(copy.deepcopy(vis), num_binary_tasks=28, reg_tasks=7,
                    use_lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.0,
                    lora_target="attn", temporal_pool="mean").to("cuda")
print("LoRA 层数:", sum(1 for n, p in m.named_parameters() if "lora_B" in n))

with torch.no_grad():
    for n, p in m.named_parameters():
        if n.endswith(".lora_B"):
            p.normal_(0, 0.01)

with torch.no_grad():
    cls_o, reg_o = m(pv, grid)
print("LoRA 模型输出:", tuple(cls_o.shape), tuple(reg_o.shape))
assert cls_o.shape == (1, 28) and reg_o.shape == (1, 7)

sd = m.get_merged_state_dict()
assert not any("lora_" in k for k in sd), "合并后残留 lora 键"

m2 = VisionMultiTask(vis, num_binary_tasks=28, reg_tasks=7, use_lora=False,
                     temporal_pool="mean").to("cuda")
miss, unexp = m2.load_state_dict(sd, strict=True)
print("普通模型加载合并权重: missing=%d unexpected=%d" % (len(miss), len(unexp)))
assert not miss and not unexp

with torch.no_grad():
    cls_2, reg_2 = m2(pv, grid)
diff_c = (cls_o - cls_2.to(cls_o.dtype)).abs().max().item()
diff_r = (reg_o - reg_2.to(reg_o.dtype)).abs().max().item()
mag_c = cls_o.abs().max().item()
print("cls 最大绝对差=%.5f (输出量级=%.3f)  reg 最大绝对差=%.6f" % (diff_c, mag_c, diff_r))
# 层级校验（fp32 下应完全一致；bf16 下差异仅为舍入噪声）
from torch.nn import functional as Fn
with torch.no_grad():
    lora_lin = m.visual.blocks[0].attn.qkv
    merged_w = lora_lin.merged_weight().to(lora_lin.weight.dtype)
    x32 = torch.randn(64, lora_lin.in_features, dtype=torch.float32, device="cuda")
    # fp32 精确等价
    w32 = lora_lin.weight.float(); a32 = lora_lin.lora_A.float(); b32 = lora_lin.lora_B.float()
    out_lora32 = Fn.linear(x32, w32, lora_lin.bias.float()) + (x32 @ a32.t() @ b32.t()) * lora_lin.scaling
    out_mrg32 = Fn.linear(x32, (w32 + b32 @ a32 * lora_lin.scaling), lora_lin.bias.float())
    exact = torch.allclose(out_lora32, out_mrg32, atol=1e-4)
    print("fp32 单层数学等价(应为 True):", exact, " maxdiff=%.2e" % (out_lora32 - out_mrg32).abs().max().item())
    assert exact
# 模型级：bf16 全链路传播会有舍入放大，用相对误差断言（10%）
ok_c = diff_c <= 0.10 * max(1.0, mag_c)
ok_r = diff_r <= 0.10
print("输出一致性(cls=%.1f%% , reg=%.1f%%) -> %s" % (
    diff_c / max(1.0, mag_c) * 100, diff_r * 100, "PASS" if (ok_c and ok_r) else "FAIL"))
assert ok_c and ok_r

# LoRA 权重相对原预训练权重的改动幅度（应有微小但非零的扰动）
# 直接校验：导出的合并权重 == 模块内按公式合并的权重，且相对原预训练权重有非零变化
lora_lin = m.visual.blocks[0].attn.qkv
orig_w = lora_lin.weight.float()
merged_ref = lora_lin.merged_weight()
exported = sd["visual.blocks.0.attn.qkv.weight"].float()
eq_export = torch.allclose(exported, merged_ref, atol=1e-3)
loRA_change = (merged_ref - orig_w).abs().mean().item()
print("导出权重 == 公式合并权重:", eq_export)
print("LoRA 相对原权重平均变化(应>0): %.6f" % loRA_change)
assert eq_export and loRA_change > 0
print("REAL MODEL TEST PASSED")
