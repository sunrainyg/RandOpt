#!/bin/bash
set -e
NGPU=${NGPU:-8}; OUT=${OUT:-landscape_results}
S=analysis/landscape/sample_perturbations.py
for M in "$@"; do
  TP=1; [[ $M == *32B* ]] && TP=2
  N=$((NGPU / TP))
  for i in $(seq 0 $((N - 1))); do
    GPUS=$(seq -s, $((i * TP)) $((i * TP + TP - 1)))
    CUDA_VISIBLE_DEVICES=$GPUS python $S --model Qwen/$M --tp $TP \
      --seeds ${SEEDS:-1-1000} --base-runs ${BASE_RUNS:-64} --repetition-penalty ${RP:-1.05} --shard $i --num-shards $N --out-dir $OUT &
    sleep 10
  done
  wait
done
