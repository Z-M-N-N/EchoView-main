#!/bin/bash

# 默认值




PTH_PATH=${5:-"./pth/best_multitask_model.pth"}
SAVE_DIR=${6:-"./pth/qwen2.5-vl-7b-visual_finetuned"}


if [ -f "/opt/anaconda/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="/opt/anaconda"
    EchoView_dir="/home/mzhao/ECHO_VIEW/EchoView-main"
    BASE_MODEL=${4:-"/home/mzhao/公共大模型/Qwen2.5-VL-7B-Instruct"}
    cd "${EchoView_dir}/VISUAL_TRAIN" || exit 1
else
    CONDA_BASE="/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/miniconda3"
    BASE_MODEL=${4:-"../../COMPARE_MODEL/qwen2.5_Model"}
    EchoView_dir="/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/zhaomeng/EchoView-main"
    cd "${EchoView_dir}/VISUAL_TRAIN" || exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate Qwen_back

export CUDA_VISIBLE_DEVICES=5
export NPROC_PER_NODE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


python merge_model_weights.py \
    --checkpoint_path $PTH_PATH \
    --base_model_path $BASE_MODEL \
    --save_dir $SAVE_DIR