#!/usr/bin/env bash
# ============================================================
# run_chat_moe.sh — 超声诊断一键调用 (MoE 路由 + Qwen 问答)
#
#   bash run_chat_moe.sh      # 直接跑, 输入从静态文件读取, 可批量
#
# 修改下面 "静态配置区" 即可调整: 输入文件 / GPU / 模型 / 输出文件
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ===================== 静态配置区 =====================
GPU=5                                          # 推理 GPU 号 (默认空闲 A6000)
PYTHON=/home/mzhao/.conda/envs/Qwen_back/bin/python
INPUT_JSON="/home/mzhao/ECHO_VIEW/EchoView-main/MoE/data/test_batch_200.json"
                                               # 输入样本文件: 单个 JSON 或数组(批量)
MOE_CHECKPOINT=""                              # 留空=自动 (全量优先, 回退小规模)
MOE_CONFIG="/home/mzhao/ECHO_VIEW/EchoView-main/MoE/configs/echo_router.yaml"
OUTPUT_JSON="/home/mzhao/ECHO_VIEW/EchoView-main/MoE/data/result_test_batch_200.json"
                                               # 结果输出文件 (纯净 JSON)
# =======================================================

export CUDA_VISIBLE_DEVICES="$GPU"

if [ ! -f "$INPUT_JSON" ]; then
    echo "[run_chat_moe] 错误: 输入文件不存在: $INPUT_JSON" >&2
    exit 1
fi

CMD=("$PYTHON" chat_moe.py --input "$INPUT_JSON" --json --moe_config "$MOE_CONFIG")
if [ -n "$MOE_CHECKPOINT" ]; then
    CMD+=("--moe_checkpoint" "$MOE_CHECKPOINT")
fi

echo "[run_chat_moe] 输入: $INPUT_JSON"
echo "[run_chat_moe] GPU: $GPU"
echo "[run_chat_moe] 处理中(进度见终端), 结果写入: $OUTPUT_JSON"

# stdout -> 纯净 JSON 文件; 进度/提示 -> 终端(stderr)
"${CMD[@]}" > "$OUTPUT_JSON"

# ---------- 结果摘要 ----------
python3 - "$OUTPUT_JSON" <<'EOF'
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception as e:
    print(f"[run_chat_moe] 无法读取结果文件: {e}", file=sys.stderr)
    sys.exit(1)
items = data if isinstance(data, list) else [data]
print(f"\n[run_chat_moe] 完成, 共 {len(items)} 条结果:")
for r in items:
    ans = r.get("answer") or r.get("error") or "?"
    flag = " ✓" if r.get("correct") is True else (" ✗" if r.get("correct") is False else "")
    print(f"  - {str(r.get('id','?')):<45} {str(r.get('diag_item','?')):<12} 模型={str(r.get('selected_model','?')):<4} 诊断={ans}{flag}")
print(f"  完整结果: {sys.argv[1]}")
EOF
