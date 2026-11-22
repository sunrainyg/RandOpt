#!/bin/bash
set -e
export OUT=${OUT:-landscape_results}
bash analysis/landscape/sample_all.sh \
  Qwen2.5-0.5B-Instruct Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct Qwen2.5-32B-Instruct
python analysis/landscape/figure2.py --out figure2.png \
  $OUT/Qwen2.5-{0.5B,3B,7B,32B}-Instruct/sigma0.001_rp1.05_len1536
