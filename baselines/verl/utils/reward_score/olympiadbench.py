# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""
OlympiadBench reward function aligned with RandOpt utils.reward_score.math.
"""

import re
from typing import Optional

_SOLUTION_CLIP_CHARS = 500


def normalize_answer(answer: str) -> str:
    if answer is None:
        return ""
    answer = answer.strip()
    answer = re.sub(r"\s+", " ", answer)
    return answer


def simplify_latex(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "")
    s = s.replace("\\!", "")
    s = s.replace("\\;", "")
    s = s.replace("\\:", "")
    s = s.replace("\\ ", " ")
    s = s.replace("\\quad", " ")
    s = s.replace("\\qquad", " ")
    s = s.replace("\\text{", "{")
    s = s.replace("\\mathrm{", "{")
    s = s.replace("\\mathbf{", "{")
    s = s.replace("\\textbf{", "{")
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")
    s = s.replace("^{\\circ}", "^\\circ")
    s = s.replace("°", "^\\circ")
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s


def try_numeric_comparison(answer: str, truth: str) -> bool:
    try:
        def parse_fraction(v: str) -> float:
            v = v.strip()
            frac_match = re.match(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", v)
            if frac_match:
                num = float(frac_match.group(1))
                denom = float(frac_match.group(2))
                return num / denom
            return float(v)

        ans_val = parse_fraction(answer)
        truth_val = parse_fraction(truth)
        return abs(ans_val - truth_val) < 1e-9
    except (ValueError, ZeroDivisionError):
        return False


def normalize_for_comparison(s: str) -> str:
    if not s:
        return ""
    s = simplify_latex(s)
    s = s.replace(" ", "")
    if not re.search(r"[\\{}^_]", s):
        s = s.lower()
    return s


def try_coordinate_comparison(answer: str, truth: str) -> bool:
    def extract_coords(s: str):
        s = s.replace(" ", "")
        numbers = re.findall(r"-?\d+\.?\d*", s)
        return [float(n) for n in numbers] if numbers else None

    ans_coords = extract_coords(answer)
    truth_coords = extract_coords(truth)

    if ans_coords and truth_coords and len(ans_coords) == len(truth_coords):
        return all(abs(a - t) < 1e-9 for a, t in zip(ans_coords, truth_coords))
    return False


def extract_answer(solution_str: str, method: str = "strict") -> Optional[str]:
    assert method in ["strict", "flexible"]

    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        def extract_boxed_content(s: str) -> list[str]:
            results = []
            i = 0
            while i < len(s):
                if s[i:].startswith("\\boxed{"):
                    start = i + 7
                    depth = 1
                    j = start
                    while j < len(s) and depth > 0:
                        if s[j] == "{":
                            depth += 1
                        elif s[j] == "}":
                            depth -= 1
                        j += 1
                    if depth == 0:
                        results.append(s[start : j - 1])
                    i = j
                else:
                    i += 1
            return results

        matches = extract_boxed_content(solution_str)
        if matches:
            return matches[-1]

        boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        matches = re.findall(boxed_pattern, solution_str)
        if matches:
            return matches[-1]

        simple_pattern = r"\\boxed\{([^}]+)\}"
        matches = re.findall(simple_pattern, solution_str)
        if matches:
            return matches[-1]

        return None

    answer = extract_answer(solution_str, method="strict")
    if answer is not None:
        return answer

    inline_math_patterns = [
        r"is\s+\\\(([^)]+)\\\)\.?\s*$",
        r"is\s+\$([^$]+)\$\.?\s*$",
        r"=\s*\\\(([^)]+)\\\)\.?\s*$",
        r"=\s*\$([^$]+)\$\.?\s*$",
    ]

    for pattern in inline_math_patterns:
        matches = re.findall(pattern, solution_str, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].strip()

    patterns = [
        r"answer\s+is\s*[:\s]*\**\\\(([^)]+)\\\)\**",
        r"answer\s+is\s*[:\s]*\**\$([^$]+)\$\**",
        r"answer\s+is\s*[:\s]*\**([^\n.*]+)\**",
        r"(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?(?:answer\s+is\s*)?[:\s]*\**([^\n.*]+)\**",
        r"(?:final\s+)?answer[:\s]+\**([^\n.*]+)\**",
        r"=\s*([^\n=]+)$",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, solution_str, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    return None


def _is_correct_answer(answer: str, ground_truth: str) -> bool:
    norm_answer = normalize_answer(answer)
    norm_truth = normalize_answer(ground_truth)

    if norm_answer == norm_truth:
        return True
    if norm_answer.replace(" ", "") == norm_truth.replace(" ", ""):
        return True

    simp_answer = simplify_latex(norm_answer)
    simp_truth = simplify_latex(norm_truth)

    if simp_answer == simp_truth:
        return True
    if simp_answer.replace(" ", "") == simp_truth.replace(" ", ""):
        return True

    final_answer = normalize_for_comparison(norm_answer)
    final_truth = normalize_for_comparison(norm_truth)

    if final_answer == final_truth:
        return True
    if try_numeric_comparison(simp_answer, simp_truth):
        return True
    if try_coordinate_comparison(simp_answer, simp_truth):
        return True

    return False


def compute_score(solution_str: str, ground_truth: str, extra_info: dict = None) -> dict:
    del extra_info  # Kept for compatibility with existing reward_fn signature.

    pred = extract_answer(solution_str, method="strict")
    if pred is None:
        pred = extract_answer(solution_str, method="flexible")

    correct = _is_correct_answer(pred, ground_truth) if pred is not None else False
    reward = 1.0 if correct else 0.0
    format_found = pred is not None

    return {
        "score": reward,
        "acc": 1.0 if correct else 0.0,
        "format_found": 1.0 if format_found else 0.0,
        "pred": str(pred) if pred is not None else "",
    }
