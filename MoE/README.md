# 多模态 MoE 路由决策系统 (Multimodal MoE Routing)

依据架构规格书 v1.0 实现：**文本 + 多路视频 → 3 类动作概率 (左转/直行/右转)** 的端到端推理框架。

## 环境
- Python: `conda run -n Qwen_back python ...` (g5 上的 Qwen_back 环境)
- 依赖: torch>=2.7, transformers>=4.57, timm, decord, opencv-python, pyyaml

## 目录结构 (对应规格书 §8)
```
MoE/
├── configs/default.yaml          # 全部超参数
├── data/                         # 数据集 + collator + transforms
├── encoders/                     # Text / Video 编码器
├── fusion/cross_attention.py     # 跨模态对齐层
├── aggregation/perceiver_pooler.py # 上下文聚合层
├── moe/                          # Router + Experts + MoE Layer
├── model/multimodal_moe.py       # 完整模型装配
├── losses/moe_losses.py          # 负载均衡 + 重要性损失
├── trainer/trainer.py            # 训练循环 (AMP/梯度累积/Checkpoint)
├── inference/predictor.py        # 推理入口
├── scripts/                      # 数据处理脚本
├── train.py                      # 训练入口
└── weights/                      # 权重放置目录
```

## 数据说明 (超声诊断任务)

**原始数据**: `/home/mzhao/ECHO_VIEW/Data/dicom_videos_group/Echo-View_Result`
- 28 个疾病独立二分类结果 (mark: 1=阳性, -1=阴性), 每样本 1~6 路超声视频 + 诊断问题文本
- `mcd/`(主要+辅助切面) / `cmd/`(所有切面) / `pvd/`(主要切面) 三个评估集

**生成的索引** (`data/index/`):
```json
{
  "id": "Dataset/group_4/xxx",
  "text": "请查看视频,请根据超声影像特征判断主动脉瓣是否增厚,输出是或否.",
  "video_paths": {"A4C": "/abs/path/A4C.mp4", "Parasternal_Long": "/abs/path/...", ...},
  "label": 0,                    // mark: 1 -> 1, -1 -> 0
  "metadata": {"diag_item": "主动脉瓣增厚"}
}
```
- train.json ~33.2万条, val.json ~5.8万条 (按患者 id 分层切分, 防泄漏)
- 视频从 mp4 实时抽帧 (16帧 224x224), 无需预存帧
- 全量生成: `python scripts/build_echo_index.py --mode mcd --out data/index --val_ratio 0.15`

**注意**: 99% 样本 <=6 路视频, 个别样本有 7~9 路 (罕见视图), 当前 `num_videos_max=6` 会截断。
文本编码器建议使用中文/多语言 BERT (g5 已有 paraphrase-multilingual-MiniLM-L12-v2), 英文 bert-base-uncased 无法处理中文诊断问题。

## 快速开始


### 1. 数据准备
```bash
# 生成假数据 (冒烟测试用)
python scripts/generate_fake_data.py --n_train 32 --n_val 8

# 或放入真实视频数据后生成索引:
#   数据目录结构: <root>/train/<sample_id>/{text.txt,label.txt,front.mp4,...}
python scripts/build_index.py --root /path/to/data --out data/index
# 视频预处理: 抽帧为 npy
python scripts/preprocess_video.py --video xx.mp4 --out data/preprocessed/xx.mp4.npy
```

### 2. 冒烟测试 (验证维度/前向/损失, 无需真实权重)
```bash
python scripts/smoke_test.py
```

### 2.1 快速训练验证 (假数据, 1 epoch)
```bash
# 生成 32+8 条假视频数据 (代码已预置在 data/, 可用此命令重新生成)
python scripts/generate_fake_data.py --n_train 32 --n_val 8
# 跑 1 epoch 冒烟训练 (配置: batch=4, epochs=1, 输出到 outputs_smoke/)
python train.py --config configs/smoke_train.yaml
```

### 3. 训练
```bash
python train.py --config configs/default.yaml
```

### 4. 推理
```python
from inference.predictor import Predictor
import torch, yaml
cfg = yaml.safe_load(open("configs/default.yaml"))
p = Predictor(cfg, "outputs/checkpoints/best.pt")
# 构造 batch 输入: text_ids, text_mask, video(B,K,T,C,H,W), video_mask
result = p.predict(text_ids, text_mask, video, video_mask)
print(result["probs"], result["predictions"])
```

## 模型权重说明
- **Text Backbone**: `bert-base-uncased`。默认读取 `weights/bert-base-uncased/`
  (如不存在则随机初始化)。请自行下载放到该目录。
- **Video Backbone**: VideoMAE-v2-Base。默认指向
  `/home/mzhao/暂时不用/对比模型/video-echo-clip-main/VideoMAEv2-Base`
  (可修改 `configs/default.yaml` 中 `video_encoder.backbone_path` 替换)。

## 规格书关键参数速查
| 项 | 值 |
|---|---|
| d_model / d_context | 512 / 512 |
| N_experts / Top-K | 6 / 2 |
| Latents M | 8 |
| Router noise std | 0.01 (仅训练) |
| α / β (辅助损失系数) | 0.01 / 0.01 |
| Optimizer | AdamW, lr=1e-4, wd=1e-5 |
| Scheduler | CosineAnnealingWarmRestarts (T0=10, Tmult=2) + warmup 1000 |
| Batch / Accum | 32 / 2 (等效 64) |
| AMP | FP16 |

## 动态 K 处理
`video_mask` (B,K) 标记各路是否存在；缺失路在编码后置零、
跨模态注意力 key_padding_mask 掩盖、聚合层同样掩盖。
