"""Reward scoring utilities for GQA."""

import re
from typing import Callable, Optional

_FILLER_WORDS = {
    "it", "they", "there", "this", "that", "is", "are", "was", "were",
    "i", "a", "an", "the", "in", "on", "of", "to", "its", "he", "she",
    "do", "does", "did", "has", "have", "had", "not", "be", "been",
    "very", "also", "just", "quite", "really", "probably", "likely",
    "no", "yes", "so", "if", "at", "by", "for", "as", "up",
}


def clean_extracted(text: str) -> Optional[str]:
    """Clean extracted phrase and keep only short answer spans."""
    text = re.sub(r'[.!,;:\'"]+$', "", text).strip()
    text = re.split(r"\s+(?:and|but|because|which|that|since|so|as)\s+", text)[0].strip()
    words = text.split()
    if 1 <= len(words) <= 3:
        return text
    return None


def extract_boxed(response: str) -> Optional[str]:
    """Extract and shorten answer from \\boxed{...} when present."""
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", response)
    if not matches:
        return None

    ans = matches[-1].strip()
    ans = re.sub(r'[.!,;:\'"*]+$', "", ans).strip()
    ans = re.sub(r"^(a|an|the)\s+", "", ans, flags=re.IGNORECASE).strip()
    if not ans:
        return None

    words = ans.split()
    if len(words) <= 4:
        return ans

    yn = re.match(r"^(yes|no)\b", ans, re.IGNORECASE)
    if yn:
        return yn.group(1)

    inner_patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*[:：]?\s*(.+?)\.?\s*$",
        r"(?:it|they|there)\s+(?:is|are)\s+(?:a|an|the)?\s*(.+?)\.?\s*$",
    ]
    for pat in inner_patterns:
        m = re.search(pat, ans, re.IGNORECASE)
        if m:
            inner = m.group(1).strip()
            inner = re.sub(r'[.!,;:\'"*]+$', "", inner).strip()
            inner = re.sub(r"^(a|an|the)\s+", "", inner, flags=re.IGNORECASE).strip()
            if inner and len(inner.split()) <= 4:
                return inner

    parts = re.split(r"\s*[,]\s*|\s+(?:and|or|but|with|in|on|of|to|from|near|above|below|next to)\s+", ans)
    first_part = parts[0].strip() if parts else ""
    first_part = re.sub(r"^(a|an|the)\s+", "", first_part, flags=re.IGNORECASE).strip()
    if first_part and len(first_part.split()) <= 3:
        return first_part

    content_words = [w.rstrip(".,;:!?") for w in words if w.lower().rstrip(".,;:!?") not in _FILLER_WORDS]
    if content_words and len(content_words) <= 3:
        return " ".join(content_words)
    if content_words:
        return content_words[0]

    return ans


def extract_answer(response: str) -> str:
    """Extract short free-form GQA answer with boxed-answer priority."""
    response = response.strip()
    if not response:
        return ""

    boxed = extract_boxed(response)
    if boxed is not None:
        return boxed

    answer_patterns = [
        r"(?:^|\n)\s*\**\s*(?:final\s+)?answer\s*\**\s*[:：]\s*(.+?)\.?\s*$",
        r"(?:^|\n)\s*the\s+(?:final\s+)?answer\s+is\s*[:：]?\s*(.+?)\.?\s*$",
        r"(?:^|\n)\s*(?:so|therefore|thus|hence)[,.]?\s+(?:the\s+)?(?:answer\s+is\s+)?(.+?)\.?\s*$",
    ]
    for pat in answer_patterns:
        m = re.search(pat, response, re.IGNORECASE | re.MULTILINE)
        if m:
            ans = m.group(1).strip()
            ans = re.sub(r"^[:\-,;\.]+\s*", "", ans).strip()
            ans = re.sub(r'[.!,;:\'"*]+$', "", ans).strip()
            ans = re.sub(r"^(a|an|the)\s+", "", ans, flags=re.IGNORECASE).strip()
            words = ans.split()
            if 1 <= len(words) <= 3:
                return ans

    first_line = response.split("\n")[0].strip()
    for prefix in ["answer:", "the answer is:", "the answer is", "response:", "a:"]:
        if first_line.lower().startswith(prefix):
            first_line = first_line[len(prefix):].strip()
            break

    first_line = re.sub(r"^[:\-,;\.]+\s*", "", first_line).strip()
    yes_no_match = re.match(r"^(yes|no)\b", first_line, re.IGNORECASE)
    if yes_no_match:
        return yes_no_match.group(1).lower()

    cleaned = re.sub(r"[.!,;:\'\"]+$", "", first_line).strip()
    words = cleaned.split()
    if len(words) <= 3:
        return cleaned

    lower = cleaned.lower()
    m = re.match(
        r"^(?:it|they|there|that|this|he|she)\s+"
        r"(?:is|are|was|were|looks?|appears?|seems?)\s+"
        r"(?:to be\s+)?"
        r"(?:a|an|the|like\s+(?:a|an|the)?\s*)?\s*"
        r"(.+)$",
        lower,
    )
    if m:
        extracted = clean_extracted(m.group(1))
        if extracted:
            return extracted

    m = re.match(r"^the\s+\w+\s+(?:is|are|was|were)\s+(.+)$", lower)
    if m:
        extracted = clean_extracted(m.group(1))
        if extracted:
            return extracted

    m = re.match(
        r"^i\s+(?:think|believe|guess|would say)\s+"
        r"(?:(?:it|they|that|this)\s+)?"
        r"(?:(?:is|are|was|were|'s)\s+)?"
        r"(?:a|an|the)?\s*"
        r"(.+)$",
        lower,
    )
    if m:
        extracted = clean_extracted(m.group(1))
        if extracted:
            return extracted

    fallback = re.split(r"\s+(?:and|but|because|which|that|since|so|as)\s+", cleaned)[0].strip()
    fallback_words = fallback.split()
    content_words = [w for w in fallback_words if w.lower() not in _FILLER_WORDS]
    if content_words:
        result = " ".join(content_words[-2:]) if len(content_words) >= 2 else content_words[-1]
        return re.sub(r"[.!,;:\'\"]+$", "", result).strip()

    return " ".join(fallback_words[-2:])


def compute_score(
    response: str,
    ground_truth,
    extract_answer: Callable[[str], str],
    normalize_answer: Callable[[str], str],
    match_answer: Callable[[str, str], bool],
    whole_word_search: Callable[[str, str], bool],
    score: float = 1.0,
) -> float:
    """Compute GQA reward with extraction + flexible fallback checks."""
    if ground_truth is None:
        return 0.0

    if isinstance(ground_truth, dict):
        gt_answer = ground_truth.get("answer", "")
    else:
        gt_answer = str(ground_truth)

    gt = normalize_answer(gt_answer)
    if not gt:
        return 0.0

    pred = normalize_answer(extract_answer(response))
    if match_answer(pred, gt):
        return score

    first_line = response.strip().split("\n")[0]
    if whole_word_search(first_line, gt):
        return score

    return 0.0
