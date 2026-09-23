# RandOpt

<p align="center">
  <img src="assets/neural_thickets.gif" alt="Neural Thickets" width="100%">
</p>

**Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights**

[Yulu Gan](https://yulugan.com), [Phillip Isola](https://web.mit.edu/phillipi/)

[Paper](https://arxiv.org/pdf/2603.12228)          |         [Project Page](https://thickets.mit.edu)    |   [Openreview](https://openreview.net/forum?id=92oF5bU4cU) |    Starting with a 1D Experiment: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SsBrfQ-iFKuGElWjTNiFoX4dtMaCzCGy?usp=sharing)


## News
- **[2026-07]** **Iterative RandOpt** is now on the [**`iterative-randopt`**](https://github.com/sunrainyg/RandOpt/tree/iterative-randopt) branch! Also support the `pip install`-able implementation that drops into existing **verl** or **huggingface/trl** setups.


## Requirements

### Option1: Python / Conda
```bash
(optional) conda activate your_env
pip install -r requirements.txt
```

### Option2: Docker

From the directory containing `RandOpt/`:

| Step | Command |
|------|---------|
| **Build** | `docker build -f RandOpt/docker/Dockerfile_vllm -t randopt-vllm:latest .` |
| **Run** | `docker run -it --gpus all randopt-vllm:latest bash` |
| **Run** (with data) | `docker run -it --gpus all -v /path/to/RandOpt/data:/workspace/data randopt-vllm:latest bash` |


## Run RandOpt

### Post-train on your own dataset
Please follow the instructions in [CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md)

### Post-train on a standard dataset
First download the data here: [data/README.md](data/README.md)

Then, from the `RandOpt` directory:

| Mode | Command |
|------|---------|
| **Single node** | `sbatch scripts/single_node.sh` |
| **Multiple nodes** | `sbatch scripts/multiple_nodes.sh` |
| **Local** (no Slurm) | `bash scripts/local_run.sh` |

## Distill top-k models into a single model
Please follow the instructions in [distillation/README.md](distillation/README.md).

## Serve many seeds at once as LoRA adapters

`scripts/make_seed_lora.py` materialises each seed as a PEFT LoRA adapter
instead of perturbing the full weights in place. A stock `vllm serve` can then
hold all seeds simultaneously and a client picks one per request through the
OpenAI `model` field, so no weight swapping happens between requests.

```bash
python3 scripts/make_seed_lora.py \
  --model Qwen/Qwen2.5-32B-Instruct --num_seeds 20 \
  --output_dir ./seed_loras --rank 16 --noise_scale 0.001

vllm serve Qwen/Qwen2.5-32B-Instruct --enable-lora --max-loras 20 --max-lora-rank 16 \
  --lora-modules $(for i in $(seq 0 19); do printf 'seed%d=./seed_loras/seed%d ' $i $i; done)

curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "seed7", "messages": [{"role": "user", "content": "..."}]}'
```

Only `config.json` is read, so generation runs on CPU in seconds. For each
targeted linear layer the two LoRA factors are drawn from `torch.randn` seeded
with the seed and scaled so the applied delta has per-element std
`noise_scale`, the counterpart of the full-matrix `sigma`. The same
`(model, seed, rank, alpha, noise_scale)` reproduces the same adapter.

**This is not equivalent to full-matrix RandOpt.** A rank-`r` delta with the
same per-element std concentrates its energy in `r` directions, so its
operator norm is about `sqrt(d / r)` larger than dense noise of the same std.
Start from a sigma at or below your dense setting and tune from there.

### Results

Qwen2.5-32B-Instruct, greedy decoding, rank 16 / alpha 16 adapters on all
attention and MLP projections. pass@k is the unbiased estimator of
[Chen et al. 2021](https://arxiv.org/abs/2107.03374) over the n generations per
example (one per seed, or one per sample for the base-model rows); pass@n is
the coverage.

**Facts-search** (multi-hop retrieval agent, 200 test examples, seeds ranked on
a disjoint 200-example train split):

| Setting | n | pass@1 | pass@2 | pass@5 | pass@10 | pass@n |
|---|---|---|---|---|---|---|
| Base model, T=1.0 samples | 20 | 0.436 | 0.457 | 0.478 | 0.490 | 0.500 |
| Full-matrix perturbation, sigma 0.001 | 18 | 0.429 | 0.500 | 0.572 | 0.609 | 0.630 |
| Seed LoRAs, `noise_scale=0.001` | 10 | 0.450 | 0.530 | 0.612 | 0.660 | 0.660 |

Seed LoRAs kept pass@1 at the base model's level and gave the best pass@k at
every k, with fewer seeds than the full-matrix run.

**Countdown** (arithmetic puzzle, 20 test examples, seeds ranked on 80 train
examples):

| Setting | n | pass@1 | pass@2 | pass@5 | pass@10 | Majority vote |
|---|---|---|---|---|---|---|
| Base model, T=0.6 samples | 10 | 0.490 | 0.619 | 0.776 | 0.900 | 0.70 |
| Full-matrix perturbation, sigma 0.001, top-10 of 100 | 10 | 0.525 | 0.647 | 0.769 | 0.800 | 0.75 |
| Seed LoRAs, `noise_scale=0.01` | 6 | degenerate | - | - | - | - |
| Seed LoRAs, `noise_scale=0.001` | 6 | coherent, low diversity | - | - | - | - |

At `noise_scale=0.01` five or six of the six seed LoRAs produced repetitive,
non-terminating output on Countdown prompts, so no accuracy is reported. At
`noise_scale=0.001` every seed was coherent, but seeds differed only modestly
from each other and we did not run the full evaluation. On a reasoning task
like Countdown the low-rank concentration bites harder than on the agentic
facts-search task, so treat the LoRA path as a serving convenience whose sigma
must be re-tuned per task, not as a drop-in replacement for full-matrix seeds.

## Run Baselines
Please follow the instructions in [baselines/README.md](baselines/README.md)

## Having questions?
Open an issue @ [github.com/sunrainyg/RandOpt/issues](https://github.com/sunrainyg/RandOpt/issues/new).


## Citation
```bib
@misc{gan2026neuralthickets,
      title={Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights}, 
      author={Yulu Gan and Phillip Isola},
      year={2026},
      eprint={2603.12228},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.12228}, 
}
```
