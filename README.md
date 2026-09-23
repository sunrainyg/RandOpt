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
  --output_dir ./seed_loras --rank 1 --noise_scale 0.001

vllm serve Qwen/Qwen2.5-32B-Instruct --enable-lora --max-loras 20 --max-lora-rank 1 \
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
In our runs, Qwen2.5-32B-Instruct at `noise_scale=0.01`, rank 16 degenerated
on Countdown while dense perturbation at the same sigma did not; at
`noise_scale=0.001` on a multi-hop retrieval task the seed LoRAs matched dense
seeds on per-seed accuracy and coverage. Start from a sigma an order of
magnitude below your dense setting and tune from there.

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
