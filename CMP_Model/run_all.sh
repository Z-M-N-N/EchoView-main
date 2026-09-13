#!/bin/bash

# ============================================
# 配置和函数定义
# ============================================

# 定义数据集名称
dataset_name="dicom_videos_group"
test_dir="/home/mzhao/ECHO_VIEW/Data/${dataset_name}/Train_Test_JSON/pvd/test"

# 根据环境选择 conda 路径
if [ -f "/opt/anaconda/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="/opt/anaconda"
    EchoView_dir="/home/mzhao/ECHO_VIEW/EchoView-main/CMP_Model/model"
else
    CONDA_BASE="/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/miniconda3"
    EchoView_dir="/cpfs01/projects-HDD/cfff-3782eb030d9c_HDD/zs13367/zhaomeng/Data/dicom_videos_group"
fi

# 加载 conda
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# 切换到工作目录
cd "${EchoView_dir}" || exit 1

# ============================================
# 选择要运行的模型（true=运行, false=跳过）
# ============================================
RUN_PANECHO=true
RUN_ECHOPRIME=false
RUN_QWEN25_VL=false
RUN_QWEN3_VL=false
RUN_LINGSHU=false
RUN_MEDGEMMA=false
RUN_HUATUOGPT=false
RUN_GEMMA4=false

# ============================================
# 设置样本数量
# ============================================
sample_num=200

# ============================================
# 函数：运行模型推理
# ============================================
run_model() {
    local model_name=$1
    local conda_env=$2
    local sample_size=$3
    local save_dir=$4
    local cmd=$5
    
    echo ""
    echo "=========================================="
    echo "开始运行: $model_name"
    echo "环境: $conda_env"
    echo "保存目录: $save_dir"
    echo "样本数量: $sample_size"
    echo "=========================================="
    
    # 激活指定的 conda 环境
    conda activate "$conda_env"
    if [ $? -ne 0 ]; then
        echo "❌ 激活环境 $conda_env 失败"
        return 1
    fi
    
    # 创建保存目录
    if [ ! -d "$save_dir" ]; then
        echo "📁 创建目录: $save_dir"
        mkdir -p "$save_dir"
        if [ $? -ne 0 ]; then
            echo "❌ 目录创建失败"
            return 1
        fi
    else
        echo "📁 目录已存在: $save_dir"
    fi
    
    # 执行命令
    echo "🚀 执行: $cmd"
    eval "$cmd"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ $model_name 执行成功"
        
        # 检查置信区间计算脚本是否存在
        if [ -f "置信区间计算.py" ]; then
            echo "📊 计算置信区间..."
            python 置信区间计算.py \
                --json_floder "$save_dir" \
                --output "${save_dir}/result.xlsx" \
                --bootstrap 500
            if [ $? -eq 0 ]; then
                echo "✅ 置信区间计算完成"
            else
                echo "⚠️  置信区间计算失败"
            fi
        else
            echo "⚠️  置信区间计算.py 不存在，跳过"
        fi
    else
        echo "❌ $model_name 执行失败 (退出码: $exit_code)"
        return 1
    fi
    
    echo "=========================================="
    echo ""
}

# ============================================
# 运行各个模型（根据开关执行）
# ============================================

# 1. PanEcho (环境: Qwen_back)
if [ "$RUN_PANECHO" = true ]; then
    save_dir="/home/mzhao/ECHO_VIEW/Data/${dataset_name}/PanEcho_Result"
    run_model "PanEcho" "Qwen_back" "$sample_num" "$save_dir" \
        "python PanEcho.py \"$save_dir\" \"$test_dir\" $sample_num"
fi

# 2. EchoPrime (环境: Qwen_back)
if [ "$RUN_ECHOPRIME" = true ]; then
    save_dir="/home/mzhao/ECHO_VIEW/Data/${dataset_name}/EchoPrime_Result"
    run_model "EchoPrime" "Qwen_back" "$sample_num" "$save_dir" \
        "python EchoPrime.py \"$save_dir\" \"$test_dir\" $sample_num"
fi

# 3. Qwen2.5-VL (环境: Qwen_back)
if [ "$RUN_QWEN25_VL" = true ]; then
    save_dir="/home/mzhao/ECHO_VIEW/Data/${dataset_name}/Qwen25-VL_Result"
    run_model "Qwen25-VL" "Qwen_back" "$sample_num" "$save_dir" \
        "python Qwen25VL.py \
            --model_path \"/home/mzhao/公共大模型/Qwen2.5-VL-7B-Instruct\" \
            --data_path \"$test_dir\" \
            --sample_size $sample_num \
            --save_results \"$save_dir\""
fi

# 4. Qwen3-VL (环境: Qwen_back)
if [ "$RUN_QWEN3_VL" = true ]; then
    save_dir="/home/mzhao/ECHO_VIEW/Data/${dataset_name}/Qwen3-VL_Result"
    run_model "Qwen3-VL" "Qwen_back" "$sample_num" "$save_dir" \
        "python Qwen3VL.py \
            --model_path \"/home/mzhao/ECHO_VIEW/对比模型/Qwen3-VL-8B-Instruct\" \
            --data_path \"$test_dir\" \
            --sample_size $sample_num \
            --save_results \"$save_dir\""
fi

# 5. Lingshu (环境: Qwen_back)
if [ "$RUN_LINGSHU" = true ]; then
    save_dir="/home/mzhao/ECHO_VIEW/Data/${dataset_name}/Lingshu_Result"
    run_model "Lingshu" "Qwen_back" "$sample_num" "$save_dir" \
        "python Lingshu.py \
            --model_path \"/home/mzhao/ECHO_VIEW/对比模型/Lingshu-7B\" \
            --data_path \"$test_dir\" \
            --sample_size $sample_num \
            --save_results \"$save_dir\""
fi

# 6. Medgemma (环境: Qwen_back)
if [ "$RUN_MEDGEMMA" = true ]; then
    save_dir="/home/mzhao/ECHO_VIEW/Data/${dataset_name}/Medgemma_Result"
    run_model "Medgemma" "Qwen_back" "$sample_num" "$save_dir" \
        "python Medgemma.py \"$save_dir\" \"$test_dir\" $sample_num"
fi

# 7. HuatuoGPT (环境: qwen_back_clone)
if [ "$RUN_HUATUOGPT" = true ]; then
    save_dir="/home/mzhao/ECHO_VIEW/Data/${dataset_name}/HuatuoGPT_Result"
    run_model "HuatuoGPT" "qwen_back_clone" "$sample_num" "$save_dir" \
        "python HuatuoGPT-Vision-main/inference.py \"$save_dir\" \"$test_dir\" $sample_num"
fi

# 8. Gemma4 (环境: qwen_back_clone2)
if [ "$RUN_GEMMA4" = true ]; then
    save_dir="/home/mzhao/ECHO_VIEW/Data/${dataset_name}/Gemma4_Result"
    run_model "Gemma4" "qwen_back_clone2" "$sample_num" "$save_dir" \
        "python Gemma4.py \
            --save_results \"$save_dir\" \
            --data_path \"$test_dir\" \
            --sample_size $sample_num"
fi

echo ""
echo "=========================================="
echo "🎉 选中的模型运行完成！"
echo "样本数量: $sample_num"
echo "=========================================="