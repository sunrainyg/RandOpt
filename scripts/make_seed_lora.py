#!/usr/bin/env python3
"""
Generate random *seed LoRA adapters*: one PEFT adapter per seed, so that a
stock vLLM server can switch between perturbed models per request.

This is a LoRA-shaped variant of RandOpt's seeded weight perturbation
(``utils/worker_extn.py::perturb_self_weights``). Instead of adding
full-shape noise to every parameter on-GPU, each seed is materialised ahead
of time as a low-rank adapter that vLLM serves with ``--enable-lora`` and that
a client selects through the OpenAI ``model`` field (``model=seed7``).

For every targeted linear layer (weight ``[out_features, in_features]``) one
vector of length ``rank * (in_features + out_features)`` is drawn from
``torch.randn`` seeded with ``int(seed)`` and split into the two factors:

    A = first  rank * in_features  values -> (rank, in_features)
    B = last   rank * out_features values -> (out_features, rank)

Each factor is scaled by ``sqrt(noise_scale * sqrt(rank) / alpha)`` so the
PEFT delta ``(alpha / rank) * (B @ A)`` has per-element std ``noise_scale``,
the same magnitude as the full-matrix perturbation ``noise_scale * N(0, 1)``.
The same ``(model, seed, rank, alpha, noise_scale)`` always reproduces the
same adapter.

Caveat: a rank-r delta with the same per-element std as full-matrix noise
concentrates its energy in r directions, so its operator norm is roughly
``sqrt(d / r)`` larger. Seed LoRAs are a cheap way to serve many seeds
side by side, not a drop-in equivalent of the full-matrix search; expect to
retune ``noise_scale`` downward relative to the full-matrix sigma.

Example:
    python3 scripts/make_seed_lora.py \\
        --model Qwen/Qwen2.5-32B-Instruct --num_seeds 20 \\
        --output_dir ./seed_loras --noise_scale 0.001

Then serve:
    vllm serve Qwen/Qwen2.5-32B-Instruct --enable-lora --max-loras 20 \\
        --lora-modules $(for i in $(seq 0 19); do printf 'seed%d=./seed_loras/seed%d ' $i $i; done)
"""
import json
import math
import os
from typing import Optional

import fire
import torch

# Default LoRA targets: the attention and MLP projections most decoder-only
# transformers expose as separate nn.Linear layers. Override with --target_modules.
_DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

_ADAPTER_PREFIX = "seed"


def _build_meta_model(model, trust_remote_code):
    """Instantiate the model skeleton on the meta device (no weights, no GPU)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    with torch.device("meta"):
        skeleton = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    return config, skeleton


def _collect_target_linears(model, target_modules):
    """Map each targeted linear module's path to its (out_features, in_features)."""
    targets = set(target_modules)
    linears = {}
    for name, module in model.named_modules():
        if name.split(".")[-1] not in targets:
            continue
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if in_features is None or out_features is None:
            continue
        linears[name] = (out_features, in_features)
    return linears


def _generate_lora_pair(out_features, in_features, rank, seed, factor, dtype):
    """Draw one combined vector for this layer and split it into LoRA A and B.

    Seeded with ``int(seed)`` like perturb_self_weights, the same seed for
    every module, so layers with identical (in + out) receive identical factors.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    combined = torch.randn(rank * (in_features + out_features), generator=generator)
    combined.mul_(factor)

    split_at = rank * in_features
    lora_a = combined[:split_at].reshape(rank, in_features).to(dtype)
    lora_b = combined[split_at:].reshape(out_features, rank).to(dtype)
    return lora_a, lora_b


def _build_adapter_tensors(linears, seed, rank, factor, dtype):
    """Build the PEFT-format state dict for all targeted layers for one seed."""
    tensors = {}
    for module_path, (out_features, in_features) in linears.items():
        lora_a, lora_b = _generate_lora_pair(out_features, in_features, rank, seed, factor, dtype)
        key_prefix = f"base_model.model.{module_path}"
        tensors[f"{key_prefix}.lora_A.weight"] = lora_a
        tensors[f"{key_prefix}.lora_B.weight"] = lora_b
    return tensors


def _build_adapter_config(model, target_modules, rank, alpha):
    """PEFT LoraConfig serialisation accepted by both PEFT and vLLM."""
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "auto_mapping": None,
        "base_model_name_or_path": model,
        "revision": None,
        "inference_mode": True,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "bias": "none",
        "target_modules": sorted(target_modules),
        "modules_to_save": None,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "use_rslora": False,
        "use_dora": False,
    }


def _resolve_dtype(dtype_arg, config):
    if dtype_arg is not None:
        return _DTYPES[dtype_arg]
    # transformers >= 5 renamed config.torch_dtype to config.dtype.
    for attr in ("dtype", "torch_dtype"):
        config_dtype = getattr(config, attr, None)
        if isinstance(config_dtype, torch.dtype):
            return config_dtype
    return torch.bfloat16


def _factor_std(noise_scale, rank, alpha):
    """Per-factor std so the delta (alpha/rank)*(B@A) has per-element std noise_scale.

    Var[(B@A)_ij] = rank * f**4, so std[delta] = (alpha / sqrt(rank)) * f**2.
    """
    return math.sqrt(noise_scale * math.sqrt(rank) / alpha)


def _write_adapter(adapter_dir, tensors, adapter_config):
    from safetensors.torch import save_file

    os.makedirs(adapter_dir, exist_ok=True)
    save_file(tensors, os.path.join(adapter_dir, "adapter_model.safetensors"))
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as config_file:
        json.dump(adapter_config, config_file, indent=2)


def _validate_args(rank, num_seeds, noise_scale, dtype):
    if rank < 1:
        raise SystemExit("--rank must be >= 1")
    if num_seeds < 1:
        raise SystemExit("--num_seeds must be >= 1")
    if noise_scale <= 0:
        raise SystemExit("--noise_scale must be > 0")
    if dtype is not None and dtype not in _DTYPES:
        raise SystemExit(f"--dtype must be one of {sorted(_DTYPES)}")


def make_seed_loras(
    model: str,
    num_seeds: int,
    output_dir: str = "./seed_loras",
    rank: int = 1,
    alpha: Optional[float] = None,
    noise_scale: float = 0.001,
    target_modules=_DEFAULT_TARGET_MODULES,
    dtype: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """Write one PEFT LoRA adapter per seed under output_dir/seed<N>.

    Args:
        model: HF hub id or local directory; only config.json is read.
        num_seeds: Number of adapters to generate (seeds 0 .. N-1).
        output_dir: Directory that receives seed0/, seed1/, ...
        rank: LoRA rank.
        alpha: LoRA alpha (PEFT scaling alpha/rank). The factor magnitude
            compensates for it, so it cancels out of the delta std. Defaults to rank.
        noise_scale: Target per-element std of the weight delta; the analogue of
            the full-matrix perturbation's sigma.
        target_modules: Linear module leaf names to perturb.
        dtype: Adapter tensor dtype (bfloat16/float16/float32). Defaults to the model's.
        trust_remote_code: Allow custom modeling code for non-standard architectures.
    """
    _validate_args(rank, num_seeds, noise_scale, dtype)
    alpha = float(alpha) if alpha is not None else float(rank)
    target_modules = list(target_modules)

    config, skeleton = _build_meta_model(model, trust_remote_code)
    resolved_dtype = _resolve_dtype(dtype, config)
    linears = _collect_target_linears(skeleton, target_modules)
    if not linears:
        raise SystemExit(
            f"No linear layers matched target modules {target_modules}. "
            "Inspect the model and pass --target_modules explicitly."
        )

    factor = _factor_std(noise_scale, rank, alpha)
    adapter_config = _build_adapter_config(model, target_modules, rank, alpha)
    for seed in range(num_seeds):
        tensors = _build_adapter_tensors(linears, seed, rank, factor, resolved_dtype)
        _write_adapter(os.path.join(output_dir, f"{_ADAPTER_PREFIX}{seed}"), tensors, adapter_config)

    print(
        f"Generated {num_seeds} adapters in {output_dir}\n"
        f"  rank={rank}  alpha={alpha}  noise_scale={noise_scale:.4g}  "
        f"factor(std of A,B)={factor:.4g}\n"
        f"  dtype={str(resolved_dtype).replace('torch.', '')}  layers={len(linears)}\n"
        "Serve with: vllm serve <model> --enable-lora --lora-modules "
        f"{_ADAPTER_PREFIX}0={output_dir}/{_ADAPTER_PREFIX}0 ..."
    )


if __name__ == "__main__":
    fire.Fire(make_seed_loras)
