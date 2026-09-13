#!/bin/bash

# 默认值
NUM_EPOCHS=${1:-1}
UNFREEZE_LAYERS=${2:-3}

TRAIN_JSON=${3:-"../../Data/dicom_videos_group/Train_Test_JSON2/vcr/train/visual_multi_task.json"}
# BASE_MODEL=${4:-"../../COMPARE_MODEL/qwen2.5_Model"}
BASE_MODEL=${4:-"/home/mzhao/公共大模型/Qwen2.5-VL-7B-Instruct"}
WEIGHT_OUTPUT=${5:-"../../weights_lora/Visual/pth_fix"}
RESUME_FROM=${6:-""}

BATCH_SIZE=1

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

python train.py \
    --train_json "$TRAIN_JSON" \
    --num_epochs $NUM_EPOCHS \
    --unfreeze_layers $UNFREEZE_LAYERS \
    --base_model "$BASE_MODEL" \
    --weight_output "$WEIGHT_OUTPUT" \
    --batch_size $BATCH_SIZE \
    $RESUME_ARG

echo "Training completed at $(date)"