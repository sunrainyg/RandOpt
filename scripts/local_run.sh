#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA_DEVICES="${CUDA_DEVICES:-0}"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DATASET="countdown"
TRAIN_DATA_PATH="data/countdown/countdown.json"
TEST_DATA_PATH="data/countdown/countdown.json"

TP=1
NUM_GPUS="$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")"
NUM_ENGINES=$((NUM_GPUS / TP))
(( NUM_GPUS % TP == 0 )) || { echo "NUM_GPUS must be divisible by TP"; exit 1; }

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_TOKEN="${HF_TOKEN:-}"
export VLLM_NO_USAGE_STATS=1

python3 randopt.py \
  --dataset "$DATASET" \
  --train_data_path "$TRAIN_DATA_PATH" \
  --test_data_path "$TEST_DATA_PATH" \
  --model_name "$MODEL" \
  --num_engines "$NUM_ENGINES" \
  --tp "$TP" \
  --train_samples 50 \ 
  --precision bfloat16 \
  --population_size 32 \
  --top_k_ratios "0.1,0.2" \
  --sigma_values "0.0005,0.001,0.002" \
  --max_tokens 512 \
  --global_seed 42 \
  --experiment_dir "randopt-experiment-local" \
  --cuda_devices "$CUDA_DEVICES"
