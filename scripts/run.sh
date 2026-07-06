#!/usr/bin/env bash
# Run iterative RandOpt on ONE 8-GPU node.
#
#   pip install -e .                      # from the repo root
#   bash scripts/run_8gpu.sh
#   MODEL=Qwen/Qwen2.5-3B-Instruct ROUNDS=4 bash scripts/run_8gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."               # repo root
export VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# The flat layout is importable as `iterative_randopt` only once installed
# (package-dir maps the name onto "."); editable-install it if needed.
python3 -c "import iterative_randopt" 2>/dev/null || pip install -e .

CONFIG="${CONFIG:-configs/iterative_randopt_gsm8k.yaml}"
ARGS=(--config "$CONFIG")
[ -n "${MODEL:-}" ]  && ARGS+=(--model "$MODEL")
[ -n "${DATASET:-}" ] && ARGS+=(--dataset "$DATASET")
[ -n "${ROUNDS:-}" ] && ARGS+=(--rounds "$ROUNDS")
[ -n "${NUM_GPUS:-}" ] && ARGS+=(--num_gpus "$NUM_GPUS")
[ -n "${OUTPUT_DIR:-}" ] && ARGS+=(--output_dir "$OUTPUT_DIR")

echo "[run_8gpu] config=$CONFIG extra=${ARGS[*]}"
exec python3 -m iterative_randopt "${ARGS[@]}"
