#!/bin/bash

# ============ 配置区 ============

# 超参数
lr=${lr:-1e-4}
batch_size=${batch_size:-4}
grad_accum_steps=${grad_accum_steps:-1}

# DeepSpeed
deepspeed="./scripts/zero3.json"

# 是否使用 tmux：true 用 tmux，false 直接前台跑
USE_TMUX=${USE_TMUX:-false}

# ============ 日志配置 ============
log_dir="${lora_output_dir}/logs"
mkdir -p "$log_dir"

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${log_dir}/train_${train_mode}.log"

# ============ Tmux 配置 ============
SESSION_NAME="vlm_train_${train_mode}"
TMUX_LOG="${log_dir}/tmux_${train_mode}.log"

# ============ 训练函数 ============
run_train() {
    # 激活 conda 环境
    source ${CONDA_BASE}/etc/profile.d/conda.sh
    conda activate Qwen_back

    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    export NPROC_PER_NODE=$NPROC_PER_NODE
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    export MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

    # 打印配置
    echo "=========================================="
    echo "训练配置"
    echo "=========================================="
    echo " train_mode      : $train_mode"
    echo " llm             : $llm"
    echo " train_json      : $train_json"
    echo " lora_output_dir : $lora_output_dir"
    echo " lr              : $lr"
    echo " batch_size      : $batch_size"
    echo " grad_accum      : $grad_accum_steps"
    echo " effective_batch : $((batch_size * grad_accum_steps))"
    echo "=========================================="
    echo "开始时间: $(date)"
    echo ""

    # 启动训练
    torchrun \
        --nproc_per_node=$NPROC_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train/train_qwen.py \
        --deepspeed $deepspeed \
        --model_name_or_path "$llm" \
        --dataset_use "ECHO_HEART_train%100" \
        --dataset_use_eval "ECHO_HEART_test%100" \
        --data_json "$train_json" \
        --data_flatten True \
        --tune_mm_vision True \
        --tune_mm_mlp True \
        --tune_mm_llm True \
        --bf16 \
        --lora_enable True \
        --output_dir "$lora_output_dir" \
        --num_train_epochs $epoch \
        --per_device_train_batch_size $batch_size \
        --per_device_eval_batch_size $batch_size \
        --gradient_accumulation_steps $grad_accum_steps \
        --max_pixels 50176 \
        --min_pixels 784 \
        --eval_strategy "steps" \
        --eval_steps $save_step \
        --save_strategy "steps" \
        --save_steps $save_step \
        --save_total_limit 26 \
        --learning_rate $lr \
        --weight_decay 0.05 \
        --warmup_ratio 0.03 \
        --max_grad_norm 1 \
        --lr_scheduler_type "cosine" \
        --logging_steps 8 \
        --model_max_length 40960 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --run_name "qwen2vl-baseline" \
        --report_to none 2>&1 | tee "$log_file"

    echo ""
    echo "=========================================="
    echo "训练结束时间: $(date)"
    echo "=========================================="
    echo "按 Ctrl+D 退出或使用 tmux attach -t $SESSION_NAME 重新连接"
}

# ============ 分支执行 ============
if [ "$USE_TMUX" = "true" ]; then
    # ---------- tmux 模式 ----------
    echo "=========================================="
    echo "🚀 将在 tmux 会话中启动训练"
    echo "=========================================="
    echo "会话名称: $SESSION_NAME"
    echo "日志文件: $log_file"
    echo "Tmux 日志: $TMUX_LOG"
    echo "=========================================="

    # 检查 tmux 是否可用
    if ! command -v tmux &> /dev/null; then
        echo "❌ 错误: tmux 未安装！"
        echo "请安装 tmux: sudo apt-get install tmux  (Ubuntu/Debian)"
        exit 1
    fi

    # 创建 tmux 会话
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "⚠️ 会话 $SESSION_NAME 已存在，尝试使用不同的名称..."
        SESSION_NAME="vlm_train_${train_mode}_$(date +%Y%m%d_%H%M%S)"
    fi

    tmux new-session -d -s "$SESSION_NAME"

    if [ $? -ne 0 ]; then
        echo "❌ Tmux 会话创建失败"
        exit 1
    fi

    echo "✅ Tmux 会话创建成功: $SESSION_NAME"

    # 在 tmux 会话中重新调起本脚本，强制前台模式
    tmux send-keys -t "$SESSION_NAME" \
        "USE_TMUX=false bash $0" C-m

    echo ""
    echo "=========================================="
    echo "✅ 训练已在 tmux 会话中启动！"
    echo "=========================================="
    echo ""
    echo "📌 常用命令:"
    echo "  查看训练输出: tmux attach -t $SESSION_NAME"
    echo "  退出 tmux (不中断训练): Ctrl+B, 然后按 D"
    echo "  查看会话列表: tmux ls"
    echo "  终止会话: tmux kill-session -t $SESSION_NAME"
    echo "=========================================="
    echo "📝 日志文件: $log_file"
    echo "=========================================="

    # 自动附加到 tmux 会话（如果脚本在终端运行）
    if [ -t 0 ]; then
        echo ""
        echo "⏳ 3秒后自动进入 tmux 会话..."
        sleep 3
        tmux attach -t "$SESSION_NAME"
    fi
else
    # ---------- 普通前台模式 ----------
    echo "=========================================="
    echo "🚀 直接在前台启动训练"
    echo "=========================================="
    echo "日志文件: $log_file"
    echo "=========================================="

    run_train
fi
