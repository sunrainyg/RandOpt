"""MATH-500 dataset handler."""
import json
from typing import Dict, List, Optional

import numpy as np

from .base import DatasetHandler

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

INSTRUCTION = "Let's think step by step and output the final answer after ####"


class MATH500Handler(DatasetHandler):
    """Handler for the MATH-500 benchmark."""

    name = "math500"
    default_train_path = "data/math-500/test.jsonl"
    default_test_path = "data/math-500/test.jsonl"
    default_max_tokens = 2048

    def __init__(self, debug: bool = False):
        from utils.reward_score import math as math_reward
        self.reward_module = math_reward
        self.debug = debug

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
        task_datas = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if line.strip():
                    # Skip samples before start_index
                    if idx < start_index:
                        continue
                    row = json.loads(line)
                    # Add instruction to ensure model outputs \boxed{} format
                    problem_with_instruction = row["problem"] + "\n\n" + INSTRUCTION
                    task_datas.append({
                        "messages": [{"role": "user", "content": problem_with_instruction}],
                        "ground_truth": row["answer"],
                        "problem": row["problem"],
                        "subject": row.get("subject", ""),
                        "level": row.get("level", 0),
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

    def postprocess_outputs_with_debug(self, outputs, task_datas, sample_size: int = 10) -> dict:
        """Compute rewards with detailed debug info for diagnosis.
        
        Returns dict with:
            - 'accuracy': float (mean reward)
            - 'correct': int (number of correct answers)
            - 'total': int (total samples)
            - 'extraction_failures': int (failed to extract answer)
            - 'sample_details': list of sample comparisons
            - 'by_subject': dict of accuracy by subject
            - 'by_level': dict of accuracy by level
        """
        rewards = []
        details = []
        extraction_failures = 0
        by_subject = {}
        by_level = {}
        
        for idx, (output, data) in enumerate(zip(outputs, task_datas)):
            response = output.outputs[0].text
            ground_truth = data.get("ground_truth", "")
            subject = data.get("subject", "unknown")
            level = data.get("level", 0)
            
            # Extract answer
            extracted_strict = self.reward_module.extract_solution(response, method="strict")
            extracted_flexible = self.reward_module.extract_solution(response, method="flexible")
            extracted = extracted_strict if extracted_strict is not None else extracted_flexible
            
            if extracted is None:
                extraction_failures += 1
            
            # Compute reward
            reward_strict = self.reward_module.compute_score(response, ground_truth, method="strict")
            reward_flexible = 0.0
            if reward_strict == 0:
                reward_flexible = self.reward_module.compute_score(response, ground_truth, method="flexible")
            final_reward = reward_strict if reward_strict > 0 else reward_flexible
            
            rewards.append(final_reward)
            
            # Track by subject
            if subject not in by_subject:
                by_subject[subject] = {"correct": 0, "total": 0}
            by_subject[subject]["total"] += 1
            if final_reward > 0:
                by_subject[subject]["correct"] += 1
            
            # Track by level
            if level not in by_level:
                by_level[level] = {"correct": 0, "total": 0}
            by_level[level]["total"] += 1
            if final_reward > 0:
                by_level[level]["correct"] += 1
            
            # Store sample details (limited to sample_size)
            if len(details) < sample_size or (final_reward == 0 and len([d for d in details if d['reward'] == 0]) < sample_size // 2):
                # Normalize for comparison
                norm_extracted = self.reward_module.normalize_answer(extracted) if extracted else ""
                norm_gt = self.reward_module.normalize_answer(ground_truth)
                
                detail = {
                    "idx": idx,
                    "ground_truth": ground_truth,
                    "extracted_strict": extracted_strict,
                    "extracted_flexible": extracted_flexible,
                    "extracted_final": extracted,
                    "normalized_extracted": norm_extracted,
                    "normalized_gt": norm_gt,
                    "reward_strict": reward_strict,
                    "reward_flexible": reward_flexible,
                    "reward": final_reward,
                    "subject": subject,
                    "level": level,
                    "response_tail": response[-500:] if len(response) > 500 else response,  # Last 500 chars
                }
                
                # Only keep if we need more samples or this is a failure case we want to track
                if len(details) < sample_size:
                    details.append(detail)
                elif final_reward == 0:
                    # Replace a correct sample with this failure sample
                    for i, d in enumerate(details):
                        if d['reward'] > 0:
                            details[i] = detail
                            break
        
        correct = sum(1 for r in rewards if r > 0)
        accuracy = float(np.mean(rewards))
        
        # Compute subject/level accuracies
        for subject in by_subject:
            by_subject[subject]["accuracy"] = by_subject[subject]["correct"] / by_subject[subject]["total"]
        for level in by_level:
            by_level[level]["accuracy"] = by_level[level]["correct"] / by_level[level]["total"]
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(rewards),
            "extraction_failures": extraction_failures,
            "sample_details": details,
            "by_subject": by_subject,
            "by_level": dict(sorted(by_level.items())),
        }

    def print_debug_report(self, debug_info: dict):
        """Print a formatted debug report."""
        print(f"\n{'='*70}")
        print(f"MATH-500 DEBUG REPORT")
        print(f"{'='*70}")
        
        print(f"\n📊 Overall: {debug_info['correct']}/{debug_info['total']} = {debug_info['accuracy']*100:.2f}%")
        print(f"❌ Extraction failures: {debug_info['extraction_failures']} ({debug_info['extraction_failures']/debug_info['total']*100:.1f}%)")
        
        print(f"\n📚 By Subject:")
        for subject, stats in sorted(debug_info['by_subject'].items(), key=lambda x: -x[1]['accuracy']):
            print(f"  {subject:25s}: {stats['correct']:3d}/{stats['total']:3d} = {stats['accuracy']*100:5.1f}%")
        
        print(f"\n📈 By Level:")
        for level, stats in debug_info['by_level'].items():
            print(f"  Level {level}: {stats['correct']:3d}/{stats['total']:3d} = {stats['accuracy']*100:5.1f}%")
        
        print(f"\n🔍 Sample Details (showing failures):")
        for i, detail in enumerate(debug_info['sample_details']):
            if detail['reward'] == 0:  # Show failures
                print(f"\n  --- Sample {detail['idx']} ({detail['subject']}, L{detail['level']}) ---")
                print(f"  Ground Truth: '{detail['ground_truth']}'")
                print(f"  Extracted (strict): {detail['extracted_strict']}")
                print(f"  Extracted (flexible): {detail['extracted_flexible']}")
                print(f"  Normalized GT: '{detail['normalized_gt']}'")
                print(f"  Normalized Ext: '{detail['normalized_extracted']}'")
                print(f"  Response tail: ...{detail['response_tail'][-200:]}")
        
        print(f"\n{'='*70}")
