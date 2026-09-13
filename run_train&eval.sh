#!/bin/bash
# ============================================================
# run_all.sh —— 一次性运行所有诊断项（训练 + 评估）
# 命令列表形式（同 run_trian.sh 风格）：每行一个诊断项，默认全部启用；
# 想跳过某项，在该行行首加 # 注释掉即可。
# 每行调用 VLM_TRAIN/train_eval.sh：训练完自动等待并评估。
# 本文件放在 EchoView-main 下。
#
# 用法:
#   cd EchoView-main
#   setsid nohup bash run_all.sh > run_all.log 2>&1 &   # 解耦后台跑（推荐）
#   或直接: bash run_all.sh                              # 前台跑
# ============================================================

# ---- conda 检测（与 run_trian.sh 一致） ----
if [ -f "/opt/anaconda/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="/opt/anaconda"
    EchoView_dir="../ECHO_VIEW/EchoView-main"
else
    CONDA_BASE="/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/miniconda3"
    EchoView_dir="/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/xxxxx/EchoView-main"
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate Qwen_back
export CONDA_BASE          # 导出，供下层脚本及 tmux 子 shell 使用

cd "${EchoView_dir}/VLM_TRAIN" || exit 1



# ---- 统一在此处配置，export 下发给下层脚本（train_eval.sh / vlm_train_only_video_cluster_1.sh / eval.sh） ----
export train_mode="PVD"                                   # 数据/输出子目录名：only_video


export GPU_ID="5"                            # GPU ID 列表
export GPU_NUM="1"                                   # GPU 数量
export batch_size=4                                        # 训练批次大小
export save_step=1000                                        # 每多少步保存一次 checkpoint
export epoch=1                                            # 训练轮数



export llm="../../weights_lora/Model/vlm_best_merged_model_PVD"    # 基座模型权重
Data_Path="../../Data/dicom_videos_group"

export train_json="${Data_Path}/Train_Test_JSON/${train_mode}/train/merged_train_list.json"  # 训练数据 json
export lora_output_dir="../../weights_lora/Lora/${train_mode}_fix"              # checkpoint目录


export grad_accum_steps=1                                        # 梯度累积步数
export lr=1e-4                                                   # 学习率
export CUDA_VISIBLE_DEVICES=$GPU_ID
export NPROC_PER_NODE=$GPU_NUM
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"




clean=1   # 1=每个诊断训练前删除旧输出目录（防续训）；0=保留


bash run_train.sh








Data_Path="../../Data/dicom_videos_group"
export eval_diag_name="all"                                   # 评估诊断项名（对应 SFT_JSON 下的子目录）
export lora_checkpoint="" #设置空则不加载权重
export eval_sample_num=100000


#============================PVD============================================
export train_mode="pvd"                                   # 数据/输出子目录名：only_video

export GPU_ID="4"                            # GPU ID 列表
export GPU_NUM="1"                                   # GPU 数量
export llm="../../weights_lora/Model/vlm_best_merged_model_PVD"    # 基座模型权重

export test_json_dir="${Data_Path}/Train_Test_JSON/${train_mode}/test"         
export result_save_dir="${Data_Path}/Echo-View_Result/${train_mode}"
bash run_eval.sh


#============================MCD============================================
export train_mode="mcd"                                   # 数据/输出子目录名：only_video

export GPU_ID="5"                            # GPU ID 列表
export GPU_NUM="1"                                   # GPU 数量
export llm="../../weights_lora/Model/vlm_best_merged_model_MCD"    # 基座模型权重

export test_json_dir="${Data_Path}/Train_Test_JSON/${train_mode}/test"         
export result_save_dir="${Data_Path}/Echo-View_Result/${train_mode}"
bash run_eval.sh

#============================CMD============================================
export train_mode="cmd"                                   # 数据/输出子目录名：only_video

export GPU_ID="6"                            # GPU ID 列表
export GPU_NUM="1"                                   # GPU 数量
export llm="../../weights_lora/Model/vlm_best_merged_model_CMD"    # 基座模型权重

export test_json_dir="${Data_Path}/Train_Test_JSON/${train_mode}/test"         
export result_save_dir="${Data_Path}/Echo-View_Result/${train_mode}"
bash run_eval.sh



echo ""
echo "=============================================="
echo "🏁 全部命令执行完毕: $(date)"
echo "=============================================="
