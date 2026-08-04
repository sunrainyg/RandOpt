"""Normalization utilities for Modular-Norm RandOpt perturbations.

The implementation supports five perturbation methods:

* isotropic:       delta = radius * noise
* frobenius:       delta = radius * noise / ||noise||_F
* natural_norm:    delta = radius * noise / ||noise||_natural
* modular_shell:   delta_l = radius * noise_l / (s_l ||noise_l||_natural)
* modular_radial:  delta = radius * noise / max_l s_l ||noise_l||_natural

For the first implementation on standard Hugging Face/vLLM transformers, the
modular scale assumes unit module sensitivity. Group-level mass is distributed
equally across parameters belonging to the group, giving

    s_l = total_mass / parameter_mass_l.

This is a practical approximation of the recursive modular norm for standard
pretrained transformers. Exact Modula coefficients require reconstructing the
full module tree and its sensitivities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Tuple

import torch

SUPPORTED_PERTURBATION_METHODS: Tuple[str, ...] = (
    "isotropic",
    "frobenius",
    "natural_norm",
    "modular_shell",
    "modular_radial",
)

# Input : hidden : output roughly follows the 1 : m : 1 allocation discussed
# in the modular-norm paper. Hidden mass is split between attention and MLP.
# Small nonzero masses keep norm/bias/other tensors perturbable by default.
DEFAULT_GROUP_MASSES: Dict[str, float] = {
    "embedding": 1.0,
    "attention": 0.5,
    "mlp": 0.5,
    "head": 1.0,
    "norm": 0.1,
    "other": 0.1,
}

_EPS = 1e-12


def load_mass_config(value: object | None) -> Dict[str, float]:
    """Load and validate a group-mass configuration.

    Args:
        value: One of:
            * None: use ``DEFAULT_GROUP_MASSES``;
            * mapping: merged over defaults;
            * JSON string;
            * path to a JSON file.

    Returns:
        A complete mapping from group name to non-negative mass.

    A zero mass excludes that group from modular perturbations. It does not
    affect isotropic, Frobenius, or natural-norm perturbations.
    """

    if value is None:
        return dict(DEFAULT_GROUP_MASSES)

    parsed: object
    if isinstance(value, Mapping):
        parsed = dict(value)
    elif isinstance(value, str):
        candidate = Path(value)
        if candidate.is_file():
            parsed = json.loads(candidate.read_text(encoding="utf-8"))
        else:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "--mass_config must be a JSON object or a path to a JSON file"
                ) from exc
    else:
        raise TypeError("mass_config must be None, a mapping, a JSON string, or a path")

    if not isinstance(parsed, Mapping):
        raise ValueError("mass_config must decode to a JSON object")

    config = dict(DEFAULT_GROUP_MASSES)
    unknown = set(parsed) - set(DEFAULT_GROUP_MASSES)
    if unknown:
        raise ValueError(
            f"Unknown mass groups: {sorted(unknown)}. "
            f"Allowed groups: {sorted(DEFAULT_GROUP_MASSES)}"
        )

    for key, raw_value in parsed.items():
        mass = float(raw_value)
        if not torch.isfinite(torch.tensor(mass)) or mass < 0.0:
            raise ValueError(f"Mass for group '{key}' must be finite and non-negative")
        config[str(key)] = mass

    if sum(config.values()) <= 0.0:
        raise ValueError("At least one mass group must be positive")

    return config


def parameter_group(name: str) -> str:
    """Classify a parameter name into a coarse transformer module group."""

    lowered = name.lower()

    # Check output heads before generic embedding patterns because some models
    # tie or name their output projection using embedding-related terminology.
    if any(token in lowered for token in ("lm_head", "output_projection", "unembed")):
        return "head"

    if any(
        token in lowered
        for token in (
            "embed_tokens",
            "tok_embeddings",
            "word_embeddings",
            ".wte.",
            "position_embeddings",
            ".wpe.",
            "embedding",
        )
    ):
        return "embedding"

    if any(
        token in lowered
        for token in (
            "self_attn",
            "attention",
            ".attn.",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "query_key_value",
        )
    ):
        return "attention"

    if any(
        token in lowered
        for token in (
            ".mlp.",
            "feed_forward",
            "feedforward",
            "gate_proj",
            "up_proj",
            "down_proj",
            ".fc1.",
            ".fc2.",
        )
    ):
        return "mlp"

    if any(token in lowered for token in ("layernorm", "layer_norm", "rmsnorm", "rms_norm", "norm.")):
        return "norm"

    return "other"


def frobenius_norm(tensor: torch.Tensor, eps: float = _EPS) -> float:
    """Return the Frobenius/L2 norm as a positive Python float."""

    value = torch.linalg.vector_norm(tensor.detach().float()).item()
    return max(float(value), eps)


def _spectral_norm_power_iteration(
    matrix: torch.Tensor,
    power_iterations: int = 8,
    eps: float = _EPS,
) -> float:
    """Estimate a matrix spectral norm using deterministic power iteration.

    The all-ones initial vector avoids introducing a second random-number
    stream, which is important because perturbation and restoration must use
    exactly the same denominator.
    """

    if matrix.ndim != 2:
        raise ValueError("Spectral norm requires a rank-2 tensor")
    if power_iterations < 1:
        raise ValueError("power_iterations must be at least 1")

    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        return eps

    work = matrix.detach().float()
    v = torch.ones(cols, device=work.device, dtype=work.dtype)
    v = v / torch.linalg.vector_norm(v).clamp_min(eps)

    for _ in range(power_iterations):
        u = torch.mv(work, v)
        u = u / torch.linalg.vector_norm(u).clamp_min(eps)
        v = torch.mv(work.transpose(0, 1), u)
        v = v / torch.linalg.vector_norm(v).clamp_min(eps)

    sigma = torch.linalg.vector_norm(torch.mv(work, v)).item()
    del work, u, v
    return max(float(sigma), eps)


def natural_norm(
    name: str,
    tensor: torch.Tensor,
    power_iterations: int = 8,
    eps: float = _EPS,
) -> float:
    """Compute a practical natural norm for a parameter perturbation.

    * Embedding matrices: maximum L2 norm over token/position rows.
    * Other matrices: spectral/operator norm.
    * 1-D elementwise scale/bias tensors: L-infinity norm.
    * Scalars: absolute value.
    * Higher-rank fallback: Frobenius norm.

    Hugging Face embedding weights normally have shape
    ``(num_embeddings, embedding_dim)``, hence the row-wise convention.
    """

    detached = tensor.detach()
    group = parameter_group(name)

    if detached.ndim == 2 and group == "embedding":
        row_norms = torch.linalg.vector_norm(detached.float(), dim=1)
        value = row_norms.max().item() if row_norms.numel() else 0.0
        del row_norms
        return max(float(value), eps)

    if detached.ndim == 2:
        return _spectral_norm_power_iteration(
            detached,
            power_iterations=power_iterations,
            eps=eps,
        )

    if detached.ndim == 1:
        value = detached.float().abs().max().item() if detached.numel() else 0.0
        return max(float(value), eps)

    if detached.ndim == 0:
        return max(float(detached.float().abs().item()), eps)

    return frobenius_norm(detached, eps=eps)


def build_modular_scales(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    mass_config: Mapping[str, float] | None = None,
    should_perturb: Callable[[str], bool] | None = None,
) -> Dict[str, float]:
    """Construct approximate architecture-aware coefficients ``s_l``.

    Group mass is divided equally among perturbable parameters in that group.
    Assuming unit sensitivity, recursive composition gives

        s_l = total_active_mass / parameter_mass_l.

    A group with zero mass receives an infinite scale, which makes its modular
    perturbation exactly zero.
    """

    config = load_mass_config(mass_config)
    predicate = should_perturb or (lambda _name: True)
    parameters = [(name, param) for name, param in named_parameters if predicate(name)]

    group_counts: MutableMapping[str, int] = {group: 0 for group in config}
    for name, _ in parameters:
        group_counts[parameter_group(name)] += 1

    present_positive_groups = {
        group
        for group, count in group_counts.items()
        if count > 0 and config[group] > 0.0
    }
    total_mass = sum(config[group] for group in present_positive_groups)
    if total_mass <= 0.0:
        raise ValueError("No perturbable parameter belongs to a positive-mass group")

    scales: Dict[str, float] = {}
    for name, _ in parameters:
        group = parameter_group(name)
        group_mass = config[group]
        count = group_counts[group]
        if group_mass <= 0.0 or count <= 0:
            scales[name] = float("inf")
            continue
        per_parameter_mass = group_mass / count
        scales[name] = total_mass / per_parameter_mass

    return scales


def normalization_denominator(
    method: str,
    name: str,
    noise: torch.Tensor,
    modular_scale: float = 1.0,
    global_radial_denominator: float | None = None,
    power_iterations: int = 8,
) -> float:
    """Return the scalar denominator for one noise tensor."""

    if method not in SUPPORTED_PERTURBATION_METHODS:
        raise ValueError(
            f"Unknown perturbation method '{method}'. "
            f"Choose from {SUPPORTED_PERTURBATION_METHODS}."
        )

    if method == "isotropic":
        return 1.0
    if method == "frobenius":
        return frobenius_norm(noise)

    if method == "modular_radial":
        if global_radial_denominator is None:
            raise ValueError("modular_radial requires a global radial denominator")
        return max(float(global_radial_denominator), _EPS)

    local_natural_norm = natural_norm(
        name,
        noise,
        power_iterations=power_iterations,
    )
    if method == "natural_norm":
        return local_natural_norm
    if method == "modular_shell":
        if modular_scale == float("inf"):
            return float("inf")
        return max(float(modular_scale) * local_natural_norm, _EPS)

    raise AssertionError(f"Unhandled method: {method}")
