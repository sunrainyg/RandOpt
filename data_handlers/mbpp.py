"""MBPP (Mostly Basic Python Problems) dataset handler."""
from typing import Dict, List, Optional

from utils.reward_score import mbpp as mbpp_reward

from .base import DatasetHandler

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are a Python programming assistant. Write clean, correct Python code to solve the given problem."
)

USER_TEMPLATE = (
    "{text}\n\n"
    "Your code should pass these tests:\n{tests}\n\n"
    "Think through your solution in <think> </think> tags.\n"
    "Return your final Python code in <answer> </answer> tags, e.g.:\n"
    "<answer>\ndef solution(x):\n    return x + 1\n</answer>"
)

# -----------------------------------------------------------------------------
# Handler
# -----------------------------------------------------------------------------


class MBPPHandler(DatasetHandler):
    """Handler for the MBPP Python programming dataset."""

    name = "mbpp"
    default_train_path = "google-research-datasets/mbpp"
    default_test_path = "google-research-datasets/mbpp"
    default_max_tokens = 2048

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def load_data(
        self,
        path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        start_index: int = 0,
    ) -> List[Dict]:
        """Load MBPP data from HuggingFace or local disk.
        
        Loads from 'full' subset. When train_path == test_path,
        main code loads all and slices: train=[:train_samples], test=[train_samples:].
        
        Args:
            path: Either HuggingFace dataset name (e.g., 'google-research-datasets/mbpp')
                  or local disk path (e.g., 'data/mbpp_full')
        """
        from datasets import load_dataset, load_from_disk
        import os
        
        # Check if path is local directory
        if os.path.isdir(path):
            # Load from local disk (saved via save_to_disk)
            ds = load_from_disk(path)
        else:
            # Load from HuggingFace
            ds = load_dataset(path, "full")
        
        # Combine all splits: train(374) + validation(90) + test(500) = 964
        all_items = []
        for split_name in ["train", "validation", "test"]:
            if split_name in ds:
                all_items.extend(ds[split_name])
        
        # Limit total samples
        total_needed = min(len(all_items), max_samples) if max_samples else len(all_items)
        
        task_datas = []
        for idx in range(total_needed):
            item = all_items[idx]
            text = item["text"]
            code = item["code"]
            test_list = item["test_list"]
            test_setup_code = item.get("test_setup_code", "")
            
            # Format tests for display
            tests_str = "\n".join(test_list[:3])  # Show first 3 tests
            
            task_datas.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": USER_TEMPLATE.format(text=text, tests=tests_str)}
                ],
                "ground_truth": {
                    "code": code,
                    "test_list": test_list,
                    "test_setup_code": test_setup_code,
                },
                "task_id": item["task_id"],
            })
        
        return task_datas

    # -------------------------------------------------------------------------
    # Reward and extraction
    # -------------------------------------------------------------------------

    def compute_reward(self, response: str, ground_truth: dict) -> float:
        """Compute reward: 1.0 if all tests pass, 0.0 otherwise."""
        code = self.extract_answer(response)
        return mbpp_reward.compute_score(code, ground_truth)

    def extract_answer(self, response: str) -> str:
        """Extract code from <answer>...</answer> tags."""
        return mbpp_reward.extract_answer(response)

    def extract_answer_for_voting(self, response: str) -> str:
        """For voting, use the extracted code as-is (no normalization to preserve syntax)."""
        return self.extract_answer(response)

    def is_answer_correct(self, response: str, ground_truth: dict) -> bool:
        """Check if answer passes all tests."""
        return self.compute_reward(response, ground_truth) > 0

    def is_voted_answer_correct(self, voted_answer: str, ground_truth: dict) -> bool:
        """Check if voted code passes all tests."""
        if not voted_answer:
            return False
        return mbpp_reward.compute_score(voted_answer, ground_truth) > 0

    def format_answer_for_check(self, answer: str) -> str:
        """Format answer for checking."""
        return f"<answer>\n{answer}\n</answer>"
