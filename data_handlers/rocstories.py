"""ROCStories dataset handler - sentence ordering task.

Source: https://huggingface.co/datasets/shawon/rocstories-combined
Input: 5 shuffled sentences
Output: correct order (e.g., "B,C,E,D,A")
Reward: 60% position accuracy + 40% adjacent pair bonus
Accuracy: exact match only (all 5 positions correct)
"""
import os
from typing import Dict, List, Optional

from utils.reward_score import rocstories as rocstories_reward

from .base import DatasetHandler

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SENTENCE_LABELS = ["A", "B", "C", "D", "E"]
VALID_LABELS = {"A", "B", "C", "D", "E"}
NUM_SENTENCES = 5
POSITION_WEIGHT = 0.6
ADJACENT_WEIGHT = 0.4


class ROCStoriesHandler(DatasetHandler):
    """Handler for ROCStories sentence ordering task."""

    name = "rocstories"
    default_train_path = "data/rocstories/train.parquet"
    default_test_path = "data/rocstories/test.parquet"
    default_max_tokens = 64

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
        """Load ROCStories sentence ordering dataset."""
        import pandas as pd
        
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Data not found: {path}. Download with:\n"
                f"python -c \"from datasets import load_dataset; "
                f"d=load_dataset('shawon/rocstories-combined'); "
                f"d['train'].to_parquet('data/rocstories/train.parquet')\""
            )
        
        df = pd.read_parquet(path)
        return self._process_dataframe(df, max_samples, start_index)
    
    def _process_dataframe(self, df, max_samples: int, start_index: int) -> List[Dict]:
        """Process dataframe into task data format."""
        task_datas = []
        
        for idx, row in df.iterrows():
            if idx < start_index:
                continue
            
            title = row.get('title', '')
            shuffled_sentences = row['shuffled_sentences']
            gold_order = row['gold_order']
            
            # Create prompt with labeled sentences (A, B, C, D, E)
            labels = SENTENCE_LABELS
            sentences_text = "\n".join([
                f"Sentence {labels[i]}: {sent}" 
                for i, sent in enumerate(shuffled_sentences)
            ])
            
            # Convert gold_order to letter format for ground truth
            # gold_order[i] tells us: shuffled_sentences[gold_order[i]] should be at position i
            # So if gold_order = [1, 2, 4, 3, 0], it means:
            # - Position 0 (first) should have shuffled_sentences[1] (which is labeled B)
            # - Position 1 (second) should have shuffled_sentences[2] (which is labeled C)
            # etc.
            gold_labels = [labels[g] for g in gold_order]
            gold_answer = ",".join(gold_labels)
            
            prompt = f"""Below are 5 sentences from a story, but they are in the wrong order.
Please arrange them in the correct chronological order.

Title: {title}

{sentences_text}

Output the correct order as comma-separated letters (e.g., B,A,D,E,C).
Only output the letters, nothing else."""
            
            system_prompt = "You are a helpful assistant that excels at story comprehension and logical reasoning. Given shuffled sentences from a story, you carefully analyze the narrative flow and temporal cues to determine the correct chronological order."
            
            task_datas.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "ground_truth": {
                    "gold_order": list(gold_order),  # [1, 2, 4, 3, 0]
                    "gold_labels": gold_labels,      # ['B', 'C', 'E', 'D', 'A']
                    "gold_answer": gold_answer,      # "B,C,E,D,A"
                },
                "story_id": idx,
            })
            
            if max_samples and len(task_datas) >= max_samples:
                break

        return task_datas

    # -------------------------------------------------------------------------
    # Answer extraction
    # -------------------------------------------------------------------------

    def extract_answer(self, response: str) -> str:
        """Extract the order from model response."""
        return rocstories_reward.extract_answer(response)
    
    def extract_answer_for_voting(self, response: str) -> str:
        """Extract normalized answer for voting (ensures consistent format across models).
        
        Returns a standardized string like "BCEDA" (no commas, no spaces) for voting,
        or empty string if invalid.
        """
        answer = self.extract_answer(response)
        labels = self._parse_order(answer)
        
        # Validate: must have exactly 5 unique labels A-E
        if len(labels) != NUM_SENTENCES:
            return ""
        
        if set(labels) != VALID_LABELS:
            return ""
        
        # Return normalized format: "BCEDA" (no separators)
        return "".join(labels)

    def _parse_order(self, answer: str) -> List[str]:
        """Parse answer string to list of labels."""
        return rocstories_reward.parse_order(answer)

    # -------------------------------------------------------------------------
    # Reward and correctness
    # -------------------------------------------------------------------------

    def compute_reward(self, response: str, ground_truth) -> float:
        """Reward = position accuracy (60%) + adjacent pair bonus (40%).
        
        - Position accuracy: fraction of sentences in correct position
        - Adjacent pair bonus: fraction of consecutive pairs in correct relative order
        """
        if ground_truth is None:
            return 0.0
        
        if isinstance(ground_truth, dict):
            gold_labels = ground_truth.get("gold_labels", [])
        else:
            return 0.0
        
        answer = self.extract_answer(response)
        pred_labels = self._parse_order(answer)
        
        # Check if we have valid prediction
        if len(pred_labels) != NUM_SENTENCES:
            return 0.0
        
        # Check if all labels are valid and unique
        if set(pred_labels) != VALID_LABELS:
            return 0.0

        return rocstories_reward.compute_score(
            pred_labels=pred_labels,
            gold_labels=gold_labels,
            valid_labels=VALID_LABELS,
            num_sentences=NUM_SENTENCES,
            position_weight=POSITION_WEIGHT,
            adjacent_weight=ADJACENT_WEIGHT,
        )
    
    def _compute_lenient_accuracy(self, pred_labels: List[str], gold_labels: List[str]) -> float:
        """Compute lenient accuracy: 60% position + 40% adjacent pair bonus.
        
        Same as compute_reward - gives partial credit for:
        1. Sentences in correct position (60%)
        2. Adjacent pairs in correct relative order (40%)
        """
        return rocstories_reward.compute_score(
            pred_labels=pred_labels,
            gold_labels=gold_labels,
            valid_labels=VALID_LABELS,
            num_sentences=NUM_SENTENCES,
            position_weight=POSITION_WEIGHT,
            adjacent_weight=ADJACENT_WEIGHT,
        )

    def is_answer_correct(self, response: str, ground_truth) -> float:
        """Return lenient accuracy (0.0 to 1.0) - 60% position + 40% adjacent."""
        if ground_truth is None:
            return 0.0
        
        if isinstance(ground_truth, dict):
            gold_labels = ground_truth.get("gold_labels", [])
        else:
            return 0.0
        
        answer = self.extract_answer(response)
        pred_labels = self._parse_order(answer)
        
        return self._compute_lenient_accuracy(pred_labels, gold_labels)

    def is_voted_answer_correct(self, voted_answer: str, ground_truth) -> float:
        """Return lenient accuracy for voted answer (0.0 to 1.0).
        
        voted_answer is already in normalized format from extract_answer_for_voting: "BCEDA"
        """
        if isinstance(ground_truth, dict):
            gold_labels = ground_truth.get("gold_labels", [])
        else:
            return 0.0
        
        # voted_answer is normalized format like "BCEDA" (from extract_answer_for_voting)
        if len(voted_answer) == 5 and all(c in 'ABCDE' for c in voted_answer):
            pred_labels = list(voted_answer)
        else:
            answer = self.extract_answer(voted_answer)
            pred_labels = self._parse_order(answer)
        
        return self._compute_lenient_accuracy(pred_labels, gold_labels)

    def postprocess_outputs(self, outputs, task_datas) -> float:
        """Compute average lenient accuracy (60% position + 40% adjacent).
        
        Returns: average score (0.0 to 1.0)
        """
        import numpy as np
        
        scores = []
        for output, data in zip(outputs, task_datas):
            response = output.outputs[0].text
            ground_truth = data.get("ground_truth")
            if ground_truth is not None:
                score = self.is_answer_correct(response, ground_truth)
                scores.append(score)
            else:
                scores.append(0.0)
        
        return float(np.mean(scores)) if scores else 0.0
