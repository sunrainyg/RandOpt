"""rejection sampling + cumulative merge.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class Rollout:
    """A single sampled trace for one question.

    ``token_ids`` is optional (used by trainers that consume token ids directly);
    ``text`` is the decoded completion; ``answer`` is the extracted final answer;
    ``correct`` is the reward>0 verdict.
    """

    __slots__ = ("text", "answer", "correct", "token_ids")

    def __init__(self, text: str, answer: str, correct: bool,
                 token_ids: Optional[Sequence[int]] = None):
        self.text = text
        self.answer = str(answer).strip() if answer else ""
        self.correct = bool(correct)
        self.token_ids = list(token_ids) if token_ids is not None else None


def select_correct_traces(
    rollouts: Iterable[Rollout],
    n_canonical: int = 8,
    keep_incorrect_frac: float = 0.0,
) -> List[Rollout]:
    """Rejection-sampling selection for ONE question.

    Mirrors ``distill_kd_gen.py`` (on-policy path):
      * dedup rollouts by decoded text,
      * rank CORRECT traces by (matches self-consistency majority answer, brevity),
      * keep up to ``n_canonical`` correct traces,
      * optionally backfill with up to ``keep_incorrect_frac * n_canonical``
        incorrect traces (default 0.0 -> pure rejection sampling, correct-only).

    Returns the kept rollouts (may be empty if the question yielded nothing).
    """
    n_keep = max(1, n_canonical)
    n_wrong_max = int(round(max(0.0, keep_incorrect_frac) * n_keep))

    seen, uniq = set(), []
    votes: Counter = Counter()
    for r in rollouts:
        if r.answer:
            votes[r.answer] += 1
        if r.text in seen:
            continue
        seen.add(r.text)
        uniq.append(r)
    if not uniq:
        return []

    maj_ans = votes.most_common(1)[0][0] if votes else None
    correct_list = [u for u in uniq if u.correct]
    # majority-aligned first, then shorter traces first.
    correct_list.sort(key=lambda u: (maj_ans is None or u.answer != maj_ans,
                                     len(u.token_ids) if u.token_ids is not None else len(u.text)))
    wrong_list = [u for u in uniq if not u.correct]

    kept = correct_list[:n_keep]
    room = n_keep - len(kept)
    if room > 0 and n_wrong_max > 0:
        kept += wrong_list[:min(room, n_wrong_max)]
    if not kept:  # all wrong -> keep a minimal set only if we allow incorrect
        kept = wrong_list[:max(1, n_wrong_max)] if n_wrong_max > 0 else []
    return kept


def _response_key(row: Dict[str, Any]) -> Tuple:
    """Hashable dedup key for a training row.
    """
    prompt = row.get("prompt")
    if "response_ids" in row and row["response_ids"] is not None:
        seq = row["response_ids"]
        seq = seq.tolist() if hasattr(seq, "tolist") else seq
        return (prompt, tuple(int(t) for t in seq))
    if "response" in row and row["response"] is not None:
        return (prompt, row["response"])
    return tuple(sorted((k, str(v)) for k, v in row.items()))


def cumulative_dedup(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dedup the union of per-round training rows (rejection-sampling growth).

    Identical semantics to the reference ``merge_cumulative`` helper.
    """
    out, seen = [], set()
    for row in rows:
        k = _response_key(row)
        if k in seen:
            continue
        seen.add(k)
        out.append(row)
    return out
