"""Reward scoring utilities for ROCStories sentence ordering."""

import re
from typing import List, Sequence, Set


def extract_answer(response: str) -> str:
    """Extract ordering answer from free-form model response."""
    response = response.strip()

    match = re.search(r"([A-E])\s*,\s*([A-E])\s*,\s*([A-E])\s*,\s*([A-E])\s*,\s*([A-E])", response.upper())
    if match:
        return ",".join(match.groups())

    match = re.search(r"([A-E])\s+([A-E])\s+([A-E])\s+([A-E])\s+([A-E])", response.upper())
    if match:
        return ",".join(match.groups())

    match = re.search(r"([A-E])([A-E])([A-E])([A-E])([A-E])", response.upper())
    if match:
        return ",".join(match.groups())

    letters = re.findall(r"[A-E]", response.upper())
    if len(letters) >= 5:
        return ",".join(letters[:5])

    return response.upper().strip()


def parse_order(answer: str) -> List[str]:
    """Parse normalized answer string to label list."""
    answer = answer.upper().replace(" ", "")
    if "," in answer:
        return [x.strip() for x in answer.split(",")]
    return list(answer.replace(",", ""))


def compute_score(
    pred_labels: Sequence[str],
    gold_labels: Sequence[str],
    valid_labels: Set[str],
    num_sentences: int = 5,
    position_weight: float = 0.6,
    adjacent_weight: float = 0.4,
) -> float:
    """Compute 60/40 lenient score used by ROCStories handler."""
    if len(pred_labels) != num_sentences or len(gold_labels) != num_sentences:
        return 0.0
    if set(pred_labels) != valid_labels:
        return 0.0

    # 1) Position accuracy
    position_correct = sum(1 for p, g in zip(pred_labels, gold_labels) if p == g)
    position_score = position_correct / float(num_sentences)

    # 2) Adjacent relative-order bonus
    pred_pos = {label: i for i, label in enumerate(pred_labels)}
    adjacent_correct = 0
    for i in range(num_sentences - 1):
        gold_first = gold_labels[i]
        gold_second = gold_labels[i + 1]
        if pred_pos[gold_first] < pred_pos[gold_second]:
            adjacent_correct += 1
    adjacent_score = adjacent_correct / float(num_sentences - 1)

    return position_weight * position_score + adjacent_weight * adjacent_score
