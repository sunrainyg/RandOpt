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


if [[ "$MODEL_NAME" == *"0.5B"* ]]; then
    export ROLLOUT_TP_SIZE=2
else
    export ROLLOUT_TP_SIZE=4
fi
export BASE_MODEL=$MODEL
export DATA_DIR=/path/to/data

export LOG_FILE=verl_countdown_grpo_qwen${MODEL_SIZE}.log

export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa

export HF_TOKEN=$HF_TOKEN
if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli login --token "$HF_TOKEN"
else
    python3 -m huggingface_hub.commands.huggingface_cli login --token "$HF_TOKEN"
fi

mkdir -p $SAVE_DIR
mkdir -p logs_countdown_grpo

    
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    trainer.default_local_dir=$SAVE_DIR \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.seed=${SEED:-42} \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    '+actor_rollout_ref.model.override_config={attn_implementation: sdpa}' \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.seed=${SEED:-42} \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.data_loader_seed=${SEED:-42} \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger='[\"wandb\"]' \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$WANDB_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10000 \
    trainer.test_freq=5 \
    trainer.total_epochs=375 2>&1 | tee "$LOG_FILE"
