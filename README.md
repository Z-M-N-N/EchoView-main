# EchoView — 超声心动图多任务智能诊断系统

基于多模态大语言模型（Qwen2.5-VL）与多专家路由（MoE）的超声影像智能诊断框架，完整覆盖 **数据构建 → 视觉多任务训练 → VLM 微调 → MoE 路由 → 对比评估** 全流程。

## ✨ 功能概览

- **多任务诊断**：支持 28 个疾病二分类 + 7 个回归任务
- **多评估集**：`PVD`（主要切面）/ `MCD`（主要+辅助切面）/ `CMD`（所有切面）
- **MoE 路由**：文本 + 多路（1~6 路）超声视频作为输入，动态路由到最合适的基座模型
- **对比基线**：内置 PanEcho、EchoPrime、Lingshu、Medgemma、HuatuoGPT-Vision、Gemma4、Qwen2.5-VL、Qwen3-VL 等模型统一评估
- **置信区间**：bootstrap（500 次）统计结果

## 🧩 系统架构

![EchoView 系统架构](Model.png)

## 📁 目录结构

| 目录/文件 | 说明 |
|---|---|
| `DATA_BUILDER/` | 数据构建：从原始 DICOM 视频生成 VCR / PVD / MCD / CMD 的训练与测试 JSON |
| `VISUAL_TRAIN/` | 视觉多任务模型训练与预测（Qwen2.5-VL，28 二分类 + 7 回归） |
| `VLM_TRAIN/` | VLM LoRA 微调与评估（PVD / MCD / CMD，DeepSpeed） |
| `MoE/` | 多模态 MoE 路由决策系统（文本 + 多路视频 → 路由到 Qwen 模型） |
| `CMP_Model/` | 对比基线模型统一推理脚本 |
| `chat.py` / `chat_moe.py` | 交互问答诊断（直接调用 / MoE 路由） |
| `run_chat_moe.sh` | MoE 路由 + Qwen 问答一键诊断 |
| `run_train&eval.sh` | 批量训练 + 评估 |

## ⚙️ 环境依赖

- Python 3.10，conda 环境：`Qwen_back`
- 主要依赖：`torch`、`transformers`、`deepspeed`、`qwen-vl-utils`、`timm`、`decord`、`opencv-python`、`scikit-learn`、`pandas`、`pyyaml`
- 部分配置文件中的基座模型、数据路径为本机绝对路径，换机运行时需按实际环境修改对应 `*.yaml` / `*.sh` 中的路径。

## 🚀 快速开始

### 1. 数据准备（DATA_BUILDER）

```bash
cd DATA_BUILDER
# 从 Dataset 原始数据生成 VCR / PVD / MCD / CMD 训练测试 JSON
bash run_build_all.sh
```

### 2. 视觉多任务训练（VISUAL_TRAIN）

```bash
cd VISUAL_TRAIN
# 训练：28 二分类 + 7 回归多任务，基于 Qwen2.5-VL-7B-Instruct
bash run_train.sh <num_epochs> <unfreeze_layers>

# 预测：输出到 pred_output/
bash run_predict.sh
```

### 3. VLM LoRA 微调与评估（VLM_TRAIN）

```bash
cd VLM_TRAIN
# 在 tmux 中启动 LoRA 微调（train_mode / GPU / batch_size 等通过环境变量配置）
bash run_train.sh

# 评估（PVD / MCD / CMD）
bash run_eval.sh
```

### 4. MoE 路由系统（MoE）

```bash
cd MoE
# 冒烟测试（验证维度 / 前向 / 损失）
python scripts/smoke_test.py

# 训练
python train.py --config configs/default.yaml

# 推理
python -c "from inference.predictor import Predictor; ..."
```

### 5. 一键问答诊断

```bash
# MoE 路由 + Qwen 问答（读取输入 JSON，批量出结果）
bash run_chat_moe.sh
```

### 6. 对比模型评估（CMP_Model）

```bash
cd CMP_Model
# 通过开关选择要运行的模型，如 PanEcho / EchoPrime / Qwen2.5-VL / Qwen3-VL ...
bash run_all.sh
```

## 📊 数据说明

- 原始数据：超声 DICOM 视频，每样本 1~6 路视频 + 诊断问题文本
- 二分类标签：`mark` 1 = 阳性，-1 = 阴性
- 评估集划分：`PVD`（主切面）、`MCD`（主+辅助切面）、`CMD`（所有切面）
- 视频在训练/推理时从 `mp4` 实时抽帧（16 帧 × 224×224），按患者分层切分防数据泄漏

## 🧠 模型方法

- **视觉多任务模型**：Qwen2.5-VL 微调，同时输出多任务二分类与回归结果
- **VLM LoRA**：DeepSpeed + LoRA 低秩微调基座模型
- **MoE 路由**：文本 + 多路视频编码 → 语义路由 → Top-K 专家（Qwen 各尺寸模型）加权决策

## 📝 License / 说明

> 本仓库为研究用途。嵌入式第三方模型实现见各子目录对应声明；数据、权重等大文件不随仓库上传（详见 `.gitignore`）。
