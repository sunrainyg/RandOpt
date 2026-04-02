#!/bin/bash
#SBATCH --job-name=
#SBATCH --output=
#SBATCH --error=
#SBATCH --partition=
#SBATCH --nodes=
#SBATCH --cpus-per-task=
#SBATCH --mem=
#SBATCH --time=

# Configuration
export SEED=42
export HF_TOKEN=


MODEL="allenai/Olmo-3-7B-Instruct"
# MODEL="allenai/Olmo-3-1025-7B"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
# MODEL="Qwen/Qwen2.5-7B-Instruct"
# MODEL="Qwen/Qwen2.5-3B-Instruct"
# MODEL="Qwen/Qwen2.5-1.5B-Instruct"
# MODEL="Qwen/Qwen2.5-0.5B-Instruct"

# Extract short model name (e.g., "Olmo-3-7B-Instruct" from "allenai/Olmo-3-7B-Instruct")
MODEL_NAME=$(basename $MODEL)

export WANDB_API_KEY=
export WANDB_ENTITY=
export WANDB_PROJECT=
export WANDB_NAME=
export SAVE_DIR=
export N_GPUS=

export BASE_MODEL=$MODEL
export DATA_DIR=/path/to/data

export LOG_FILE=verl_countdown_es_${MODEL_NAME}.log

export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa

export HF_TOKEN=$HF_TOKEN
if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli login --token "$HF_TOKEN"
else
    python3 -m huggingface_hub.commands.huggingface_cli login --token "$HF_TOKEN"
fi

mkdir -p $SAVE_DIR
mkdir -p logs_countdown_es

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_es \
    es.sigma=0.001 \
    es.alpha=0.0005 \
    es.population_size=30 \
    es.num_engines=$N_GPUS \
    es.precision=bfloat16 \
    es.max_tokens=1024 \
    es.temperature=0.0 \
    es.eval_batch_size=256 \
    es.gpu_memory_utilization=0.7 \
    es.global_seed=$SEED \
    es.verbose=false \
    es.worker_extension_cls='verl.workers.rollout.vllm_rollout.es_worker_extension.WorkerExtension' \
    model.path=$BASE_MODEL \
    data.task_type=countdown \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_max_samples=200 \
    data.val_max_samples=-1 \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$WANDB_NAME \
    trainer.logger='["wandb"]' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.total_epochs=300 \
    trainer.test_freq=5 2>&1 | tee "$LOG_FILE"
