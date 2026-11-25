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

## Serve seeds behind an OpenAI-compatible API

`randopt_server.py` starts the standard vLLM OpenAI server and adds four routes
that drive `utils/worker_extn.py` on every TP rank. This lets an external
evaluator (an agent loop, a tool-calling harness, any OpenAI client) switch the
served weights between seeds without restarting vLLM or touching Ray.

```bash
python3 randopt_server.py \
  --model Qwen/Qwen2.5-32B-Instruct \
  --worker-extension-cls utils.worker_extn.WorkerExtension \
  --tensor-parallel-size 4 --max-model-len 16384
```

All vLLM server flags are forwarded unchanged. The extra routes:

| Route | Body | Effect |
|-------|------|--------|
| `POST /store_base` | `{}` | Snapshot current weights as the reset target |
| `POST /perturb` | `{"seed": 7, "sigma": 0.001, "negate": false}` | `W += sigma * N(0,1)` seeded per parameter |
| `POST /restore` | same as `/perturb` | Subtract the same noise again |
| `POST /reset` | `{}` | Copy the `/store_base` snapshot back |

```bash
curl -X POST localhost:8000/store_base
curl -X POST localhost:8000/perturb -H 'content-type: application/json' \
     -d '{"seed": 7, "sigma": 0.001}'
# ... run the evaluation for seed 7 through /v1/chat/completions ...
curl -X POST localhost:8000/reset
```

Prefer `/store_base` + `/reset` over `/restore` when weights are bf16 or FP8:
`/restore` re-adds the negated noise and is exact only up to rounding, while
`/reset` copies the snapshot back bit-for-bit. Perturbation skips quantization
scale tensors (`weight_scale`, `scale_inv`, ...) by default; set
`PERTURB_SCALES=1` to include them and `PERTURB_VISUAL=1` to perturb the vision
tower of VL models.

Smoke test against a running server:

```bash
python3 scripts/test_perturb_restore.py --model Qwen/Qwen2.5-32B-Instruct
```

## Distill top-k models into a single model
Please follow the instructions in [distillation/README.md](distillation/README.md).

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
