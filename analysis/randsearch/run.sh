#!/bin/bash
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); WS=$(dirname "$REPO")
VENV=${VENV:-$WS/venv312}; PY=$VENV/bin/python
export PYTHONUNBUFFERED=1 HF_HOME=${HF_HOME:-$WS/hf_home} HF_TOKEN="${HF_TOKEN:-}" VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=$REPO
MODEL="${MODEL:-allenai/Olmo-3-7B-Instruct}"; NAME="${NAME:-run}"; NUM_ENGINES="${NUM_ENGINES:4}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M)}"; RUNS_ROOT="${RUNS_ROOT:-$HERE/runs}"
OUT=$RUNS_ROOT/${STAMP}_${NAME}; mkdir -p "$OUT"
log() { echo "[run] $(date -u +%Y-%m-%d\ %H:%M:%S) $*" | tee -a "$OUT/run.log"; }
log "out=$OUT model=$MODEL engines=$NUM_ENGINES args=$*"
log "randopt commit $(git -C $REPO rev-parse HEAD)"; cp $HERE/randomsearch.py $HERE/table.py "$OUT/" 2>/dev/null
LOCAL="$($PY $HERE/stage_model.py --model_name "$MODEL" 2>&1 | tee -a "$OUT/run.log" | tail -1)"
log "engines load the model from $LOCAL"
$PY $HERE/randomsearch.py --model_name "$MODEL" --model_local_path "$LOCAL" --num_engines "$NUM_ENGINES" --out_dir "$OUT" "$@" 2>&1 | tee -a "$OUT/run.log"
log "exit=${PIPESTATUS[0]}"
for d in $(find "$OUT" -name meta.json -printf '%h\n'); do $PY $HERE/table.py "$d" > "$d/table.md" 2> "$d/table.err"; log "table: $d/table.md"; done
