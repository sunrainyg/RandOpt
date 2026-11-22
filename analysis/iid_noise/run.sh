#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}; DATASET=${DATASET:-countdown}; OUT=${OUT:-iid_noise_results}
GPUS=${GPUS:-0,1,2,3,4,5}; IFS=, read -ra G <<< "$GPUS"; N=$(( ${#G[@]} / 2 ))
A=$(IFS=,; echo "${G[*]:0:N}"); B=$(IFS=,; echo "${G[*]:N:N}")
mkdir -p $OUT
common=(--dataset $DATASET --model_name $MODEL --num_engines $N --tp 1 --train_samples 200 --test_samples 1000
        --precision bfloat16 --population_size ${POP:-600} --sigma_values 0.0005,0.001,0.002
        --top_k_ratios 0.01,0.05,0.1 --max_tokens 1024 --global_seed 42 "$@")
python randopt.py "${common[@]}" --cuda_devices $A --experiment_dir $OUT/shared > $OUT/shared.log 2>&1 &
sleep 30
python analysis/iid_noise/randopt_iid.py "${common[@]}" --cuda_devices $B --experiment_dir $OUT/iid > $OUT/iid.log 2>&1 &
wait
grep -A4 "Majority voting" $OUT/shared.log $OUT/iid.log
