# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

import re
from typing import Any, Dict, List, Optional


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """Check whether response follows <think>...</think><answer>...</answer> format.

    Returns:
        1.0 - full ``<think>...</think>\n<answer>...</answer>`` format
        0.5 - only ``<answer>`` tag present
        0.1 - only ``<think>`` tag present
        0.0 - neither tag present
    """
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)

    if full_format_match:
        return 1.0

    reward = 0.0
    if think_match:
        reward += 0.1
    if answer_match:
        reward += 0.5
    return reward


def answer_reward_function(response: str, numbers: List[int] = None, target: int = None) -> float:
    """Return 1.0 only when last <answer> uses all numbers exactly once and reaches target."""
    answer_regex = r"<answer>(.*?)<\/answer>"
    all_matches = re.findall(answer_regex, response, re.DOTALL)
    if not all_matches:
        return 0.0

    answer_content = all_matches[-1]
    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not answer_content:
        return 0.0
    if not re.match(allowed_chars, answer_content):
        return 0.0

    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0

    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except Exception:
        return 0.0
    return 0.0


def compute_score(solution_str, ground_truth, format_score=0.1, score=1.0):
    """Compute score for countdown task.

    The total reward combines a format component and an answer component,
    matching the RandOpt reward formulation::

        reward = 0.1 * format_reward + answer_reward

    Args:
        solution_str: model output
        ground_truth: dict with 'target' and 'numbers'
        format_score: (unused, kept for call-site compatibility)
        score: (unused, kept for call-site compatibility)

    Returns:
        dict with 'score' (float reward), 'acc' (1.0/0.0 answer correctness),
        and 'format_reward' / 'answer_reward' breakdown.
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']

    format_reward = format_reward_function("<think>" + solution_str)
    answer_reward = answer_reward_function(solution_str, numbers, target)
    reward = format_reward * 0.1 + answer_reward

    return {
        "score": reward,
        "acc": 1.0 if answer_reward >= 1.0 else 0.0,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
    }
