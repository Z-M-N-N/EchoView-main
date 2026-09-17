#!/bin/bash

# 默认值
NUM_EPOCHS=${1:-1}
UNFREEZE_LAYERS=${2:-""}        # 留空时自动：LoRA 模式=0，非 LoRA 模式=3

TRAIN_JSON=${3:-"../../Data/dicom_videos_group/Train_Test_JSON/vcr/train/visual_multi_task.json"}
# BASE_MODEL=${4:-"../../COMPARE_MODEL/qwen2.5_Model"}
BASE_MODEL=${4:-"/home/mzhao/公共大模型/Qwen2.5-VL-7B-Instruct"}
WEIGHT_OUTPUT=${5:-"./pth_fix"}
RESUME_FROM=${6:-""}

# ===== 新增参数（强化视觉编码器） =====
# 默认开启 LoRA：低秩适配强化编码器，导回时合并回原权重，通用能力不易退化
USE_LORA=${7:-1}                # 1=LoRA 强化编码器（推荐），0=仅解冻最后N层
LORA_R=${8:-8}
LORA_ALPHA=${9:-16}
LORA_DROPOUT=${10:-0.0}
LORA_TARGET=${11:-"attn"}        # attn / mlp / attn+mlp
TEMPORAL_POOL=${12:-"mean"}      # mean / attention（attention 需训练/推理都开启）
VISUAL_LR=${13:-""}              # 留空自动：LoRA=1e-4，解冻=1e-5
WARMUP_RATIO=${14:-0.03}

BATCH_SIZE=1

# 若未显式传解冻层数：LoRA 模式默认 0，非 LoRA 模式保持原默认 3
if [ -z "$UNFREEZE_LAYERS" ]; then
    if [ "$USE_LORA" = "1" ]; then
        UNFREEZE_LAYERS=0
    else
        UNFREEZE_LAYERS=3
    fi
fi

if [ -f "/opt/anaconda/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="/opt/anaconda"
    EchoView_dir="/home/mzhao/ECHO_VIEW/EchoView-main"
    cd "${EchoView_dir}/VISUAL_TRAIN" || exit 1
else
    CONDA_BASE="/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/miniconda3"
    EchoView_dir="/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/zhaomeng/EchoView-main"
    cd "${EchoView_dir}/VISUAL_TRAIN" || exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate Qwen_back

export CUDA_VISIBLE_DEVICES=5
export NPROC_PER_NODE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 构建 resume_from 参数
RESUME_ARG=""
if [ -n "$RESUME_FROM" ]; then
    RESUME_ARG="--resume_from $RESUME_FROM"
fi

VISUAL_LR_ARG=""
if [ -n "$VISUAL_LR" ]; then
    VISUAL_LR_ARG="--visual_lr $VISUAL_LR"
fi

python train.py \
    --train_json "$TRAIN_JSON" \
    --num_epochs $NUM_EPOCHS \
    --unfreeze_layers $UNFREEZE_LAYERS \
    --base_model "$BASE_MODEL" \
    --weight_output "$WEIGHT_OUTPUT" \
    --batch_size $BATCH_SIZE \
    --use_lora $USE_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target "$LORA_TARGET" \
    --temporal_pool "$TEMPORAL_POOL" \
    --warmup_ratio $WARMUP_RATIO \
    $VISUAL_LR_ARG \
    $RESUME_ARG

echo "Training completed at $(date)"
