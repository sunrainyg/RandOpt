from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional


@dataclass
class IterativeRandOptConfig:
    # ---- what to train -------------------------------------------------------
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset: str = "gsm8k"
    output_dir: str = "iterative_randopt_runs"

    # ---- loop ----------------------------------------------------------------
    rounds: int = 4                 # number of rejection-sampling / SFT rounds
    num_generations: int = 16       # rollouts sampled per question
    gen_temperature: float = 1.0    # rollout sampling temperature
    n_canonical: int = 8            # max distinct correct traces kept / question
    keep_incorrect_frac: float = 0.0  # 0.0 = pure rejection sampling (correct-only)
    sft_from_base: bool = True      # True = canonical ReST^EM (SFT the ORIGINAL base every round); False = continual (warm-start from prev round's model)
    cumulative: bool = True         # accumulate deduped correct traces across rounds
    max_train_questions: int = 7473  # full gsm8k train split
    max_tokens: int = 1024          # rollout / eval max new tokens

    # ---- weight-space search --------------------------------------------------
    population: int = 30            # Gaussian perturbations
    sigma: float = 1e-3            # perturbation scale  theta -> theta + sigma*noise(seed)
    top_k: int = 8                 # keep the top-k highest-scoring perturbations to distill
    search_score_samples: int = 256  # questions used to score each perturbed model

    # ---- SFT hyper-parameters -------------------------------------------------
    epochs: float = 2.0
    lr: float = 1e-4
    warmup_ratio: float = 0.1
    lora_rank: int = 192
    lora_alpha: Optional[int] = None   # default = 2 * lora_rank
    lora_dropout: float = 0.05
    # Effective batch = per_device_batch * grad_accum * num_gpus. The reference
    # runs used ~1024 (4 nodes x 8 GPU, pdb=2, ga=16). On ONE 8-GPU node we set
    # grad_accum=64 (2 * 64 * 8 = 1024) so the effective batch is preserved.
    per_device_batch_size: int = 2
    grad_accum: int = 64
    precision: str = "bfloat16"
    seed: int = 42

    # ---- eval ----------------------------------------------------------------
    # Distilled accuracy can drift down in late rounds (esp. the continual variant);
    # keep + expose the highest-scoring (greedy) round's checkpoint rather than the last.
    keep_best: bool = True          # preserve best round's model; symlink it to <out>/best

    # ---- single-node infra ---------------------------------------------------
    num_gpus: int = 8
    gpu_memory_utilization: float = 0.75

    def resolved_lora_alpha(self) -> int:
        return self.lora_alpha if self.lora_alpha is not None else 2 * self.lora_rank

    @classmethod
    def from_yaml(cls, path: str) -> "IterativeRandOptConfig":
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
