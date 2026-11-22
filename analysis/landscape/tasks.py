from __future__ import annotations

import re

MATH_PROMPT = "{q} Let's think step by step and output the final answer after ####."
# TinyZero-style Countdown prompt
CD_PROMPT = (
    "Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic "
    "operations (+, -, *, /) and each number can only be used once. Show your work in <work> ... </work> tags. "
    "And return the final answer in <answer> ... </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)


def load_tasks(n: int = 200) -> dict:
    """Returns {task: [{"messages": [...], ...gold fields}]}, plus "nll": the raw wikitext-2 test string."""
    from datasets import load_dataset

    user = lambda content: [{"role": "user", "content": content}]
    out = {}
    ds = load_dataset("openai/gsm8k", "main", split="train").select(range(n))
    out["gsm8k"] = [{"messages": user(MATH_PROMPT.format(q=r["question"].strip())),
                     "gold": r["answer"].split("####")[-1].strip()} for r in ds]
    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train").select(range(n))
    out["countdown"] = [{"messages": user(CD_PROMPT.format(numbers=list(r["nums"]), target=r["target"])),
                         "nums": list(r["nums"]), "target": r["target"]} for r in ds]
    ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train").select(range(n))
    out["olympiad"] = [{"messages": user(MATH_PROMPT.format(q=r["question"].strip())),
                        "gold": list(r["final_answer"])} for r in ds]
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    out["nll"] = "\n\n".join(t for t in ds["text"] if t.strip())
    return out


#graders
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _last_number(s: str) -> float | None:
    m = _NUM.findall(s.replace("$", ""))
    if not m:
        return None
    try:
        return float(m[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def grade_gsm8k(pred: str, gold: str) -> bool:
    """last number after '####' (or anywhere if there is no '####'), compared numerically"""
    g = _last_number(gold)
    tail = pred.split("####")[-1] if "####" in pred else pred
    p = _last_number(tail)
    if p is None and tail is not pred:
        p = _last_number(pred)
    return p is not None and g is not None and abs(p - g) < 1e-4


_CD_OK = re.compile(r"^[\d\s\+\-\*/\(\)\.]+$")


def grade_countdown(pred: str, nums: list[int], target: int) -> bool:
    """last <answer>: uses every number exactly once and evaluates to the target"""
    answers = re.findall(r"<answer>(.*?)</answer>", pred, flags=re.S)
    if not answers:
        return False
    eq = answers[-1].strip()
    if "=" in eq:
        eq = eq.split("=")[0].strip()
    if not eq or not _CD_OK.match(eq) or "**" in eq or "//" in eq:  # only + - * / are allowed
        return False
    if sorted(int(x) for x in re.findall(r"\d+", eq)) != sorted(int(x) for x in nums):
        return False
    try:
        val = eval(eq, {"__builtins__": {}}, {})  # digits / operators / parentheses only (regex-checked)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(val, (int, float)) and abs(val - target) < 1e-5


def _norm_math(s: str) -> str:
    s = s.strip().strip(".").strip()
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\%", "").replace("%", "").replace(" ", "").replace("\\text{", "{")
    s = re.sub(r"\{([^{}]*)\}", lambda m: m.group(1) if re.fullmatch(r"[\w\.\-]+", m.group(1)) else m.group(0), s)
    return s.rstrip(".")


def _to_float(s: str) -> float | None:
    s = _norm_math(s)
    m = (re.fullmatch(r"\\frac(-?\d+)(-?\d+)", s) or re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
         or re.fullmatch(r"(-?\d+)/(-?\d+)", s))
    try:
        return float(m.group(1)) / float(m.group(2)) if m else float(s.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def grade_olympiad(pred: str, golds: list[str]) -> bool:
    """answer after '####' (else last \\boxed{}, else last line); normalised exact or numeric match"""
    if "####" in pred:
        ans = pred.split("####")[-1].strip().split("\n")[0]
    else:
        boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", pred)
        ans = boxed[-1] if boxed else pred.strip().split("\n")[-1]
    a, af = _norm_math(ans), _to_float(ans)
    for g in golds:
        gn = _norm_math(g)
        if a == gn or (gn and a.endswith(gn)):
            return True
        gf = _to_float(g)
        if af is not None and gf is not None and abs(af - gf) <= 1e-6 * max(1.0, abs(gf)):
            return True
    return False


def grade(task: str, pred: str, row: dict) -> bool:
    if task == "gsm8k":
        return grade_gsm8k(pred, row["gold"])
    if task == "countdown":
        return grade_countdown(pred, row["nums"], row["target"])
    return grade_olympiad(pred, row["gold"])
