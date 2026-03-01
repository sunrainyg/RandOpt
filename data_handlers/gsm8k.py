"""GSM8K dataset handler."""
from typing import Dict, List, Optional

import pandas as pd

from .base import DatasetHandler


class GSM8KHandler(DatasetHandler):
    """Handler for the GSM8K math word problem dataset."""

    name = "gsm8k"
    default_train_path = "data/gsm8k/train.parquet"
    default_test_path = "data/gsm8k/test.parquet"
    default_max_tokens = 1024

    def __init__(self):
        from utils.reward_score import gsm8k as gsm8k_reward
        self.reward_module = gsm8k_reward

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def load_data(
        self,
        path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> List[Dict]:
        df = pd.read_parquet(path)
        task_datas = []
        for row in df.to_dict("records"):
            messages = row["prompt"].tolist()
            ground_truth = row["reward_model"]["ground_truth"]
            task_datas.append({
                "messages": messages,
                "ground_truth": str(ground_truth),
            })
            if max_samples and len(task_datas) >= max_samples:
                break
        return task_datas

    # -------------------------------------------------------------------------
    # Reward and extraction
    # -------------------------------------------------------------------------

    def compute_reward(self, response: str, ground_truth: str) -> float:
        reward = self.reward_module.compute_score(response, ground_truth, method="strict")
        if reward == 0:
            reward = self.reward_module.compute_score(response, ground_truth, method="flexible")
        return reward

    def extract_answer(self, response: str) -> str:
        answer = self.reward_module.extract_solution(response, method="strict")
        if answer is None:
            answer = self.reward_module.extract_solution(response, method="flexible")
        return answer if answer is not None else ""

    def format_answer_for_check(self, answer: str) -> str:
        return f"#### {answer}"
