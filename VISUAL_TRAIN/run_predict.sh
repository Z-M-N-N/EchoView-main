#!/bin/bash

# 默认值

TEST_JSON=${3:-"../../Data/dicom_videos_group/Train_Test_JSON2/vcr/test/visual_multi_task.json"}
# BASE_MODEL=${4:-"../../COMPARE_MODEL/qwen2.5_Model"}
BASE_MODEL=${4:-"/home/mzhao/公共大模型/Qwen2.5-VL-7B-Instruct"}
PTH_PATH=${5:-"./pth_fix/best_multitask_model.pth"}




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



python predict.py \
    --test_json $TEST_JSON \
    --model_id $BASE_MODEL \
    --checkpoint $PTH_PATH \
    --batch_size 4 --num_frames 16 --num_binary_tasks 28 --num_reg_tasks 7 \
    --shard_idx 0 --num_shards 0 --save_predictions false --temporal_pool mean \
    --output_dir pred_output
    # --output_dir pred_output > pred_output/shard${idx}.log 2>&1 &

echo "Test completed at $(date)"



# #!/bin/bash
# cd /home/mzhao/ECHO_VIEW/EchoView-main/VISUAL_TRAIN
# mkdir -p pred_output
# gpus=(0 1 2 3 4 6)
# for idx in 0 1 2 3 4 5; do
#   gpu=${gpus[$idx]}
#   CUDA_VISIBLE_DEVICES=$gpu setsid nohup /home/mzhao/.conda/envs/Qwen_back/bin/python predict.py \
#     --test_json ../../SFT_JSON/video_to_multi_task/test/visual_multi_task.json \
#     --model_id ../../weights/qwen2.5-vl-7b-visual_finetuned \
#     --checkpoint ../../weights/pth/best_multitask_model.pth \
#     --batch_size 4 --num_frames 16 --num_binary_tasks 28 --num_reg_tasks 7 \
#     --shard_idx $idx --num_shards 6 --save_predictions false \
#     --output_dir pred_output > pred_output/shard${idx}.log 2>&1 &
#   echo "launched shard $idx on GPU $gpu"
# done
# echo "all launched"
