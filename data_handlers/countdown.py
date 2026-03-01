"""Countdown dataset handler."""
import json
import re
from typing import Dict, List, Optional, Tuple

from utils.reward_score import countdown as countdown_reward

from .base import DatasetHandler

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)

ANSWER_TAG_REGEX = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
FORMULA_ALLOWED_CHARS = re.compile(r"^[0-9+\-*/() ]+$")
MAX_FORMULA_NUMBER = 10**3
NUMERIC_TOLERANCE = 1e-9
ANSWER_MATCH_TOLERANCE = 1e-5


class CountdownHandler(DatasetHandler):
    """Handler for the Countdown arithmetic puzzle dataset."""

    name = "countdown"
    default_train_path = "data/countdown/countdown.json"
    default_test_path = "data/countdown/countdown.json"
    default_max_tokens = 1024

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def load_data(
        self,
        path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> List[Dict]:
        with open(path, "r") as f:
            all_data = json.load(f)
        
        task_datas = []
        for item in all_data:
            numbers = item["numbers"]
            target = item["target"]
            user_content = USER_TEMPLATE.format(numbers=numbers, target=target)
            
            task_datas.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_content}
                ],
                "ground_truth": {"numbers": numbers, "target": target},
                "numbers": numbers,
                "target": target,
            })
            if max_samples and len(task_datas) >= max_samples:
                break
        return task_datas

    # -------------------------------------------------------------------------
    # Reward and correctness
    # -------------------------------------------------------------------------

    def compute_reward(self, response: str, ground_truth: dict) -> float:
        return countdown_reward.compute_score(response, ground_truth)
    
    def is_answer_correct(self, response: str, ground_truth: dict) -> bool:
        """Check if answer is actually correct (not just formatted correctly).
        
        For countdown task, we only care about answer_reward (0 or 1),
        not format_reward (0-0.11). This is used for ensemble evaluation.
        """
        return countdown_reward.is_answer_correct(response, ground_truth)

    # -------------------------------------------------------------------------
    # Answer extraction
    # -------------------------------------------------------------------------

    def extract_answer(self, response: str) -> str:
        """Extract the last answer from <answer>...</answer> tags."""
        matches = ANSWER_TAG_REGEX.findall(response)
        return matches[-1].strip() if matches else ""
    
    def extract_answer_for_voting(
        self,
        response: str,
        numbers: Optional[list] = None,
        debug: bool = False,
    ) -> Tuple[str, bool, Optional[Dict]]:
        """Extract formula, validate numbers, and evaluate to get numeric result.
        
        Returns (numeric_result_str, is_valid, reject_info) tuple.
        is_valid = True only if formula uses exactly the given numbers.
        reject_info = dict with rejection details (None if valid)
        
        For countdown task, different formulas can give the same answer.
        E.g., '1+2+3' and '3+2+1' both equal 6.
        So we should vote on the numeric result, not the formula string.
        """
        formula = self.extract_answer(response)
        if not formula:
            return "", False, {"reason": "no_answer", "formula": ""}
        
        # Validate: only allow numbers and basic operators
        if not FORMULA_ALLOWED_CHARS.match(formula):
            return "", False, {"reason": "invalid_chars", "formula": formula[:100]}
        
        # Check if formula uses exactly the given numbers
        is_valid = False
        used_numbers = None
        if numbers is not None:
            used_numbers = [int(n) for n in re.findall(r"\d+", formula)]
            # Reject formulas with absurdly large numbers (likely model hallucination)
            if any(n > MAX_FORMULA_NUMBER for n in used_numbers):
                return "", False, {"reason": "number_too_large", "formula": formula[:100]}
            is_valid = (sorted(used_numbers) == sorted(numbers))
        
        try:
            result = eval(formula, {"__builtins__": None}, {})
            if abs(result - round(result)) < NUMERIC_TOLERANCE:
                result_str = str(int(round(result)))
            else:
                result_str = str(result)
            
            if not is_valid:
                return result_str, False, {
                    "reason": "wrong_numbers",
                    "formula": formula[:100],
                    "expected": sorted(numbers) if numbers else None,
                    "got": sorted(used_numbers) if used_numbers else None,
                    "result": result_str
                }
            return result_str, True, None
        except (SyntaxError, ZeroDivisionError, TypeError, ValueError, OverflowError) as e:
            return "", False, {"reason": "eval_error", "formula": formula[:100], "error": str(e)}

    # -------------------------------------------------------------------------
    # Formatting and comparison
    # -------------------------------------------------------------------------

    def format_answer_for_check(self, answer: str) -> str:
        return f"<answer>{answer}</answer>"

    def is_voted_answer_correct(self, voted_answer: str, ground_truth: dict) -> bool:
        """Check if the voted numeric answer equals the target.
        
        For countdown task with formula evaluation during voting,
        the voted_answer is already a numeric result (as string).
        We just need to check if it equals the target.
        
        NOTE: This only checks the numeric value. The validation of whether
        the formula uses correct numbers is done during voting collection
        (only valid formulas contribute to voting).
        """
        if not voted_answer:
            return False
        try:
            result = float(voted_answer)
            target = float(ground_truth["target"])
            return abs(result - target) < ANSWER_MATCH_TOLERANCE
        except (ValueError, KeyError):
            return False

    def get_target_for_comparison(self, ground_truth: dict) -> str:
        """Get target number as string for comparison."""
        return str(ground_truth["target"])
