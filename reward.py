"""Reward functions with a TRL-compatible signature.

TRL's online trainers (GRPOTrainer / OnlineDPOTrainer) expect reward functions of
the form ``reward_func(prompts, completions, **kwargs) -> list[float]`` where
``kwargs`` carries the remaining dataset columns (e.g. ``ground_truth``). We expose
exactly that signature so a TRL user's existing reward plumbing works unchanged,
while delegating to the SAME proven grader used by the reference pipeline.
"""
from __future__ import annotations

from typing import Any, List, Sequence


def _completion_text(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, (list, tuple)) and c and isinstance(c[-1], dict):
        return c[-1].get("content", "")
    return str(c)


def _ground_truths(n: int, **kwargs) -> List[str]:
    gt = kwargs.get("ground_truth")
    if gt is None:
        rm = kwargs.get("reward_model")
        if isinstance(rm, (list, tuple)):
            gt = [x.get("ground_truth") if isinstance(x, dict) else x for x in rm]
    if gt is None:
        raise ValueError("gsm8k reward needs a 'ground_truth' column in the dataset")
    if isinstance(gt, (str, int, float)):
        gt = [gt] * n
    return [str(x) for x in gt]


def gsm8k_reward(prompts: Sequence[Any], completions: Sequence[Any], **kwargs) -> List[float]:
    """1.0 for a correct GSM8K final answer else 0.0 (strict then flexible match)."""
    from .rewards import gsm8k as _g  # vendored proven grader

    texts = [_completion_text(c) for c in completions]
    gts = _ground_truths(len(texts), **kwargs)
    scores: List[float] = []
    for text, gt in zip(texts, gts):
        s = _g.compute_score(text, gt, method="strict")
        if s == 0:
            s = _g.compute_score(text, gt, method="flexible")
        scores.append(float(s))
    return scores


REWARDS = {"gsm8k": gsm8k_reward}


def get_reward(dataset: str):
    if dataset not in REWARDS:
        raise ValueError(f"No built-in reward for dataset '{dataset}'. "
                         f"Available: {sorted(REWARDS)}. Pass your own reward_func.")
    return REWARDS[dataset]
