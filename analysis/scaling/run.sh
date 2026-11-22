#!/bin/bash
set -e
export OUT=${OUT:-scaling_results} RP=${RP:-1.02}
bash analysis/landscape/sample_all.sh \
  Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct Qwen2.5-3B-Instruct Qwen2.5-32B-Instruct
python analysis/scaling/scaling.py --out scaling.png \
  $OUT/Qwen2.5-{0.5B,1.5B,3B,32B}-Instruct/sigma0.001_rp${RP}_len1536
