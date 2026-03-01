"""Reward scoring utilities for Countdown."""

import re
from typing import Any, Dict, List, Optional


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """Check whether response follows <think>...</think><answer>...</answer> format."""
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
    """Return 1.0 only when last <answer> uses all numbers once and reaches target."""
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


def reward_function(
    response: str,
    numbers: List[int] = None,
    target: int = None,
    end_token: str = None,
) -> Dict[str, Any]:
    """Total reward = 0.1 * format_reward + answer_reward."""
    format_reward = format_reward_function("<think>" + response, end_token)
    answer_reward = answer_reward_function(response, numbers, target)
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }


def compute_score(response: str, ground_truth: Dict) -> float:
    """Return full countdown reward (format + answer components)."""
    result = reward_function(response, ground_truth["numbers"], ground_truth["target"])
    return result["reward"]


def is_answer_correct(response: str, ground_truth: Dict) -> bool:
    """Return True only when the answer component is fully correct."""
    result = reward_function(response, ground_truth["numbers"], ground_truth["target"])
    return result["reward_info"]["answer_reward"] >= 1.0
