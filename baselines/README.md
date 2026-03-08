# Baselines

Baselines for the Neural Thickets paper

We integrate baselines (PPO, GRPO, ES) into a single codebase built on [VERL](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning for LLMs).

## Supported Models

| Family | Models |
|--------|--------|
| **OLMo** | `allenai/Olmo-3-7B-Instruct`, `allenai/Olmo-3-1025-7B` |
| **Qwen** | `Qwen/Qwen2.5-{0.5B,1.5B,3B,7B}-Instruct` |
| **Llama** | `meta-llama/Llama-3.1-8B-Instruct` |
| **DeepSeek** | `deepseek-ai/deepseek-llm-7b-chat` |

And more models on Hugging Face are supported. Just change the model ID.

## Install

### 1. Create environment

```bash
conda create -n baseline python==3.12
conda activate baseline
```

### 2. Install the package

```bash
cd baselines
pip install --no-deps -e .
```

### 3. Install dependencies

```bash
# Local
bash scripts/install_vllm_sglang_mcore.sh

# On SLURM cluster
sbatch install/install_vllm_sglang_mcore_slurm.sh
```

Set `USE_MEGATRON=0` or `USE_SGLANG=0` to skip optional components:

```bash
USE_MEGATRON=0 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh
```

## Prepare Data

Preprocessing scripts live in `examples/data_preprocess/`. Each script downloads and converts a dataset into parquet format.

**Countdown** (used by the default GRPO script):

```bash
python3 examples/data_preprocess/countdown.py --local_save_dir data/countdown/
```

**GSM8K:**

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k/
```

**On a SLURM cluster:**

```bash
# Edit SBATCH headers in run_jobs/data_prepare.sh first
sbatch run_jobs/data_prepare.sh
```

Other available datasets: `math500`, `mbpp`, `olympiadbench`, `uspto50k`, `math_dataset`, `multiturn`, etc. See `examples/data_preprocess/` for the full list.

## Run GRPO

### Single-node

Edit `run_jobs/countdown_grpo.sh` and fill in the required fields:

```bash
export HF_TOKEN=<your_hf_token>
export WANDB_API_KEY=<your_key>
export WANDB_ENTITY=<your_entity>
export WANDB_PROJECT=<your_project>
export WANDB_NAME=<run_name>
export SAVE_DIR=/path/to/checkpoints
export DATA_DIR=/path/to/data/countdown
export N_GPUS=8
```

Then submit:

```bash
sbatch run_jobs/countdown_grpo.sh
```

### Local run (no SLURM)

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/countdown/train.parquet \
    data.val_files=data/countdown/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=allenai/Olmo-3-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=375
```

## Run PPO

### Single-node

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=data/gsm8k/train.parquet \
    data.val_files=data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=deepseek-ai/deepseek-llm-7b-chat \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    critic.model.path=deepseek-ai/deepseek-llm-7b-chat \
    critic.optim.lr=1e-5 \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=15
```

### Multi-node

Add `trainer.nnodes=<N>` and launch with SLURM across multiple nodes. See `examples/ppo_trainer/` for model-specific scripts (Qwen, DeepSeek, Gemma, etc.).

## Run ES (Evolution Strategy)

ES training uses a separate config. Key hyperparameters are in `verl/trainer/config/es_trainer.yaml`:

```bash
python3 -m verl.trainer.main_ppo \
    --config-name es_trainer \
    es.sigma=0.001 \
    es.alpha=0.0005 \
    es.population_size=30 \
    es.num_engines=4 \
    es.num_iterations=800 \
    es.model_name=allenai/Olmo-3-7B-Instruct \
    es.train_data_path=data/countdown/train.parquet \
    es.eval_data_path=data/countdown/test.parquet
```

## Acknowledgements

This codebase is modified from [VERL](https://github.com/volcengine/verl).
