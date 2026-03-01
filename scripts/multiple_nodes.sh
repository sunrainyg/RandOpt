#!/bin/bash
#SBATCH --job-name=
#SBATCH --nodes=5
#SBATCH --account=
#SBATCH --ntasks-per-node=
#SBATCH --cpus-per-task=
#SBATCH --gres=gpu:4
#SBATCH --time=
#SBATCH --output=
#SBATCH --error=
#SBATCH --environment=

TP=1
GPUS_PER_NODE=4
MODEL="allenai/Olmo-3-7B-Instruct"
DATASET="countdown"
TRAIN_DATA_PATH="data/countdown/countdown.json"
TEST_DATA_PATH="data/countdown/countdown.json"
HEAD_IP_FILE=/path/to/RandOpt/logs/ray_head_ip_${SLURM_JOB_ID}

TRAIN_SAMPLES=200
TEST_SAMPLES=""
PRECISION="bfloat16"
MAX_TOKENS="1024"
POPULATION_SIZE=5000
TOP_K_RATIOS="0.04,0.01,0.05,0.1"
SIGMA_VALUES="0.0005,0.001,0.002"
EXPERIMENT_DIR="randopt-experiment-paper"
SEED=42
HF_TOKEN="${HF_TOKEN:-}"

NUM_ENGINES=$((SLURM_NNODES * GPUS_PER_NODE / TP))
(( GPUS_PER_NODE % TP == 0 )) || { echo "GPUS_PER_NODE must be divisible by TP"; exit 1; }
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
HEAD_PORT=6379
rm -f "$HEAD_IP_FILE"

echo "=== Multi node: nodes=${SLURM_NNODES}, gpus/node=${GPUS_PER_NODE}, TP=${TP}, engines=${NUM_ENGINES} ==="

CMD="python3 randopt.py"
CMD="$CMD --dataset $DATASET --model_name $MODEL --num_engines $NUM_ENGINES --tp $TP"
CMD="$CMD --train_data_path $TRAIN_DATA_PATH --test_data_path $TEST_DATA_PATH --train_samples $TRAIN_SAMPLES"
CMD="$CMD --population_size $POPULATION_SIZE --sigma_values $SIGMA_VALUES"
CMD="$CMD --precision $PRECISION --max_tokens $MAX_TOKENS --global_seed $SEED --experiment_dir $EXPERIMENT_DIR"
[[ -n "$TOP_K_RATIOS" ]] && CMD="$CMD --top_k_ratios $TOP_K_RATIOS"
[[ -n "$TEST_SAMPLES" ]] && CMD="$CMD --test_samples $TEST_SAMPLES"

srun --cpu-bind=none -ul --container-writable bash -c "
    export PIP_CACHE_DIR=/tmp/pip_cache_\$SLURM_NODEID
    mkdir -p \$PIP_CACHE_DIR
    pip install --cache-dir=\$PIP_CACHE_DIR --target=/tmp/python_packages_\$SLURM_NODEID pyarrow -q 2>/dev/null || echo 'pyarrow install skipped'
    export PYTHONPATH=/tmp/python_packages_\$SLURM_NODEID:\$PYTHONPATH

    cd /path/to/RandOpt
    export TRITON_CACHE_DIR=/tmp/triton_cache_\$SLURM_NODEID
    export TORCH_COMPILE_CACHE_DIR=/tmp/torch_compile_cache_\$SLURM_NODEID
    export VLLM_TORCH_COMPILE_CACHE_DIR=/tmp/vllm_torch_cache_\$SLURM_NODEID
    mkdir -p \$TRITON_CACHE_DIR \$TORCH_COMPILE_CACHE_DIR \$VLLM_TORCH_COMPILE_CACHE_DIR

    export HF_HOME=/path/to/huggingface
    mkdir -p \$HF_HOME
    export HF_TOKEN=$HF_TOKEN

    export VLLM_NO_USAGE_STATS=1
    export VLLM_DISABLE_COMPILE_SAMPLER=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export RAY_DEDUP_LOGS=1
    export RAY_LOG_TO_STDERR=0
    export VLLM_LOGGING_LEVEL=WARNING
    export VLLM_CONFIGURE_LOGGING=0
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    if [[ \$(hostname) == $HEAD_NODE ]]; then
        HEAD_IP=\$(hostname -i | awk '{print \$1}')
        echo \$HEAD_IP > $HEAD_IP_FILE
        ray start --head --port=$HEAD_PORT --node-ip-address=\$HEAD_IP
        sleep 15
        export RAY_ADDRESS=\$HEAD_IP:$HEAD_PORT
        $CMD
        ray stop
    else
        sleep 10
        HEAD_IP=\$(cat $HEAD_IP_FILE)
        ray start --address=\$HEAD_IP:$HEAD_PORT --block
    fi
"
