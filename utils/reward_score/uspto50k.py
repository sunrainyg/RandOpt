"""Reward scoring utilities for USPTO-50K classification."""

import re


def extract_answer(response: str) -> str:
    """Extract class number from `<answer>...</answer>` or fallback number scan."""
    matches = re.findall(r"<answer>\s*(\d+)\s*</answer>", response, re.IGNORECASE)
    if matches:
        return matches[-1]

    matches = re.findall(r"\b([1-9]|10)\b", response)
    if matches:
        return matches[-1]

    return ""


def compute_score(answer: str, ground_truth: str, score: float = 1.0) -> float:
    """Return `score` when predicted class matches ground truth, else 0.0."""
    if not answer:
        return 0.0

    try:
        pred_class = int(str(answer).strip())
        true_class = int(str(ground_truth).strip())
        return score if pred_class == true_class else 0.0
    except (ValueError, TypeError):
        return 0.0
