"""OlympiadBench (Open-ended Math) dataset handler."""
from typing import Dict, List, Optional

import pandas as pd

from .base import DatasetHandler

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def clean_ground_truth(answer: str) -> str:
    """Clean ground truth answer: remove $ signs and normalize.
    
    OlympiadBench ground truth comes with $ signs like '$\\frac{1}{2}$'
    but model outputs in \\boxed{} don't have $ signs.
    """
    if not answer:
        return ""
    answer = str(answer).strip()
    # Remove leading/trailing $ signs
    if answer.startswith('$') and answer.endswith('$'):
        answer = answer[1:-1].strip()
    elif answer.startswith('$'):
        answer = answer[1:].strip()
    elif answer.endswith('$'):
        answer = answer[:-1].strip()
    return answer


# -----------------------------------------------------------------------------
# Handler
# -----------------------------------------------------------------------------


class OlympiadBenchHandler(DatasetHandler):
    """Handler for the OlympiadBench open-ended math dataset."""

    name = "olympiadbench"
    default_train_path = "data/olympiadbench/OE_TO_maths_en_COMP.parquet"
    default_test_path = "data/olympiadbench/OE_TO_maths_en_COMP.parquet"
    default_max_tokens = 2048
    
    INSTRUCTION = "Let’s think step by step and output the final answer after ####."
    
    def __init__(self):
        from utils.reward_score import math as math_reward
        self.reward_module = math_reward

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
        for _, row in df.iterrows():
            raw_answer = row["final_answer"][0] if len(row["final_answer"]) > 0 else ""
            answer = clean_ground_truth(raw_answer)
            
            # Add instruction to ensure model outputs \boxed{} format
            question_with_instruction = row["question"] + "\n\n" + self.INSTRUCTION
            
            task_datas.append({
                "messages": [{"role": "user", "content": question_with_instruction}],
                "ground_truth": answer,
                "ground_truth_raw": raw_answer,  # Keep original for debug
                "subject": row.get("subject", "Math"),
                "subfield": row.get("subfield", ""),
                "answer_type": row.get("answer_type", "Numerical"),
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
        return f"\\boxed{{{answer}}}"

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------

    def postprocess_outputs_with_debug(self, outputs, task_datas, sample_size: int = 20) -> dict:
        """Compute accuracy with detailed debug info for OlympiadBench.
        
        Returns a dict with:
            - overall stats (accuracy, extraction success rate, etc.)
            - per-sample debug info for sample_size random samples
        """
        import random
        import numpy as np
        
        total = len(outputs)
        correct = 0
        extracted = 0
        by_answer_type = {}  # Track accuracy by answer type
        
        all_debug_info = []
        
        for i, (output, data) in enumerate(zip(outputs, task_datas)):
            response = output.outputs[0].text
            ground_truth = data.get("ground_truth", "")
            ground_truth_raw = data.get("ground_truth_raw", ground_truth)
            answer_type = data.get("answer_type", "Numerical")
            
            # Initialize answer_type stats
            if answer_type not in by_answer_type:
                by_answer_type[answer_type] = {"total": 0, "correct": 0, "extracted": 0}
            by_answer_type[answer_type]["total"] += 1
            
            # Extract answer
            pred = self.extract_answer(response)
            
            if pred:
                extracted += 1
                by_answer_type[answer_type]["extracted"] += 1
            
            # Compute reward
            reward = self.compute_reward(response, ground_truth)
            is_correct = reward > 0
            
            if is_correct:
                correct += 1
                by_answer_type[answer_type]["correct"] += 1
            
            # Store debug info
            all_debug_info.append({
                "idx": i,
                "answer_type": answer_type,
                "ground_truth_raw": ground_truth_raw,
                "ground_truth_clean": ground_truth,
                "extracted_answer": pred,
                "is_correct": is_correct,
                "reward": reward,
                "response_last_500": response[-500:] if len(response) > 500 else response,
            })
        
        # Sample random examples for detailed report
        sample_indices = random.sample(range(total), min(sample_size, total))
        sampled_debug = [all_debug_info[i] for i in sample_indices]
        
        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "extracted": extracted,
            "extraction_rate": extracted / total if total > 0 else 0,
            "by_answer_type": by_answer_type,
            "sampled_debug": sampled_debug,
        }
    
    def print_debug_report(self, debug_info: dict):
        """Print a formatted debug report for OlympiadBench."""
        print(f"\n{'='*70}")
        print(f"OLYMPIADBENCH DEBUG REPORT")
        print(f"{'='*70}")
        
        print(f"\n[Overall Stats]")
        print(f"  Total samples: {debug_info['total']}")
        print(f"  Correct: {debug_info['correct']} ({debug_info['accuracy']*100:.2f}%)")
        print(f"  Extracted: {debug_info['extracted']} ({debug_info['extraction_rate']*100:.2f}%)")
        
        print(f"\n[Stats by Answer Type]")
        for atype, stats in debug_info['by_answer_type'].items():
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            ext = stats['extracted'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {atype}: {stats['correct']}/{stats['total']} ({acc:.1f}% acc, {ext:.1f}% extracted)")
        
        print(f"\n[Sample Debug Info (showing incorrect/failed extractions first)]")
        # Sort to show failures first
        samples = sorted(debug_info['sampled_debug'], 
                        key=lambda x: (x['is_correct'], bool(x['extracted_answer'])))
        
        for sample in samples[:15]:  # Show up to 15
            status = "✓" if sample['is_correct'] else "✗"
            print(f"\n  --- Sample {sample['idx']} [{sample['answer_type']}] {status} ---")
            print(f"  GT (raw): {sample['ground_truth_raw']}")
            print(f"  GT (clean): {sample['ground_truth_clean']}")
            print(f"  Extracted: {sample['extracted_answer'] or '(NONE)'}")
            print(f"  Reward: {sample['reward']}")
            # Show last part of response to see if \boxed{} is there
            resp_snippet = sample['response_last_500'].replace('\n', ' ')[:200]
            print(f"  Response end: ...{resp_snippet}...")
        
        print(f"\n{'='*70}\n")
