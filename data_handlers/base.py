"""Abstract base class for dataset handlers."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class DatasetHandler(ABC):
    """Abstract base for dataset handlers.

    To add a new dataset:
    1. Create a new file in datasets/ folder
    2. Implement a class inheriting from DatasetHandler
    3. Register it in datasets/__init__.py
    """

    name: str
    default_train_path: str
    default_test_path: str
    default_max_tokens: int = 1024

    # -------------------------------------------------------------------------
    # Abstract interface
    # -------------------------------------------------------------------------

    @abstractmethod
    def load_data(
        self,
        path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        start_index: int = 0,
    ) -> List[Dict]:
        """Load dataset and return list of task data dicts."""
        pass

    @abstractmethod
    def compute_reward(self, response: str, ground_truth) -> float:
        """Compute reward for a single response."""
        pass

    @abstractmethod
    def extract_answer(self, response: str) -> str:
        """Extract answer from response for majority voting."""
        pass

    # -------------------------------------------------------------------------
    # Optional overrides
    # -------------------------------------------------------------------------

    def format_answer_for_check(self, answer: str) -> str:
        """Format extracted answer for reward computation.
        
        Override this if the dataset needs special formatting
        (e.g., wrapping in #### or \\boxed{}).
        """
        return answer

    def extract_answer_for_voting(self, response: str) -> str:
        """Extract answer for voting/comparison purposes.
        
        Default: same as extract_answer.
        Override this if the dataset needs to evaluate formulas to get numeric results
        (e.g., countdown evaluates "(1+2)*3" to get "9").
        """
        return self.extract_answer(response)

    def get_target_for_comparison(self, ground_truth) -> str:
        """Get the target answer for comparison.
        
        Default: convert ground_truth to string.
        Override if ground_truth is a dict (e.g., countdown has {"numbers": [...], "target": 24}).
        """
        return str(ground_truth)

    def is_answer_correct(self, response: str, ground_truth) -> bool:
        """Check if answer is correct for ensemble evaluation.
        
        Default implementation: reward > 0
        Override this if the dataset has special reward structure
        (e.g., countdown has format_reward + answer_reward).
        """
        return self.compute_reward(response, ground_truth) > 0

    def postprocess_outputs(self, outputs, task_datas) -> float:
        """Compute average reward from vLLM outputs."""
        rewards = []
        for output, data in zip(outputs, task_datas):
            response = output.outputs[0].text
            ground_truth = data.get("ground_truth")
            if ground_truth is None:
                rewards.append(0.0)
            else:
                rewards.append(float(self.compute_reward(response, ground_truth)))
        return float(np.mean(rewards))
