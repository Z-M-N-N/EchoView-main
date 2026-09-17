#!/bin/bash


# ============ 配置路径 ============
# 创建输出目录
mkdir -p "${result_save_dir}"

# ============ 日志配置 ============
log_dir="${result_save_dir}"
mkdir -p "$log_dir"

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${log_dir}/inference_${timestamp}.log"

# ============ Tmux 配置 ============
SESSION_NAME="${train_mode}_${timestamp}"
TMUX_LOG="${log_dir}/tmux_${timestamp}.log"

echo "=========================================="
echo "🔬 将在 tmux 会话中启动推理"
echo "=========================================="
echo "  eval_diag_name      : $eval_diag_name"
echo "  GPU_ID         : $GPU_ID"
echo "  GPU_NUM        : $GPU_NUM"
echo "  checkpoint     : $checkpoint"
echo "  sample_num       : $sample_num"
echo "  llm     : $llm"
echo "  lora_checkpoint      : $lora_checkpoint"
echo "  test_json_dir      : $test_json_dir"
echo "  result_save_dir      : $result_save_dir"
echo "=========================================="
echo "会话名称: $SESSION_NAME"
echo "日志文件: $log_file"
echo "=========================================="

# 检查 tmux 是否可用
if ! command -v tmux &> /dev/null; then
    echo "❌ 错误: tmux 未安装！"
    echo "请安装 tmux: sudo apt-get install tmux  (Ubuntu/Debian)"
    exit 1
fi

# 创建 tmux 会话
tmux new-session -d -s "$SESSION_NAME" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️ 会话 $SESSION_NAME 已存在，尝试使用不同的名称..."
    SESSION_NAME="vlm_infer_${eval_diag_name}_$(date +%Y%m%d_%H%M%S)"
    tmux new-session -d -s "$SESSION_NAME"
fi

echo "✅ Tmux 会话创建成功: $SESSION_NAME"

# 在 tmux 会话中执行推理命令
tmux send-keys -t "$SESSION_NAME" "
    # 激活 conda 环境
    source ${CONDA_BASE}/etc/profile.d/conda.sh
    conda activate Qwen_back
    
    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export NPROC_PER_NODE=$GPU_NUM
    export PYTORCH_CUDA_ALLOC_CONF=\"expandable_segments:True\"
    
    # 打印配置
    echo \"==========================================\"
    echo \"🔬 推理配置\"
    echo \"==========================================\"
    echo \"  eval_diag_name      : $eval_diag_name\"
    echo \"  GPU_ID         : $GPU_ID\"
    echo \"  GPU_NUM        : $GPU_NUM\"
    echo \"  eval_sample_num       : $eval_sample_num\"
    echo \"  llm     : $llm\"
    echo \"  lora_checkpoint      : $lora_checkpoint\"
    echo \"  test_json_dir      : $test_json_dir\"
    echo \"  result_save_dir      : $result_save_dir\"
    echo \"==========================================\"
    echo \"开始时间: $(date)\"
    echo \"\"
    
    # 启动推理
    python inference.py \
        --model_path \"${llm}\" \
        --diagnostic_mode \"${diagnostic_mode}\" \
        --lora_path_PVD \"${lora_path_PVD}\" \
        --lora_path_MCD \"${lora_path_MCD}\" \
        --lora_path_CMD \"${lora_path_CMD}\" \
        --data_path \"${test_json_dir}\" \
        --diag_item \"${eval_diag_name}\" \
        --sample_size ${eval_sample_num} \
        --save_results \"${result_save_dir}\"
    
    # 推理完成后计算置信区间
    echo \"\"
    echo \"📊 开始计算置信区间...\"
    python 置信区间计算.py \
        --json_floder \"${result_save_dir}\" \
        --output \"${result_save_dir}/result.xlsx\" \
        --bootstrap 500
    
    echo \"\"
    echo \"==========================================\"
    echo \"✅ 推理完成！\"
    echo \"📁 结果保存至: ${result_save_dir}\"
    echo \"结束时间: $(date)\"
    echo \"==========================================\"
    echo \"按 Ctrl+D 退出或使用 tmux attach -t $SESSION_NAME 重新连接\"
    exec bash
"

echo ""
echo "=========================================="
echo "✅ 推理已在 tmux 会话中启动！"
echo "=========================================="
echo ""
echo "📌 常用命令:"
echo "  查看推理输出: tmux attach -t $SESSION_NAME"
echo "  退出 tmux (不中断推理): Ctrl+B, 然后按 D"
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
    # tmux attach -t "$SESSION_NAME"
fi
