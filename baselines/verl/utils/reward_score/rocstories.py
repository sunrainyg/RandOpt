"""ROCStories reward score computation for sentence ordering task.

Input: 5 shuffled sentences
Output: correct order (e.g., "B,C,E,D,A")
Reward: 60% position accuracy + 40% adjacent pair bonus
"""
import re
from typing import Any, Dict, List, Union


def _parse_order(answer: str) -> List[str]:
    """Parse answer string to list of labels."""
    answer = answer.upper().replace(" ", "")
    if "," in answer:
        return [x.strip() for x in answer.split(",")]
    else:
        return list(answer.replace(",", ""))


def extract_answer(response: str) -> str:
    """Extract the order from model response."""
    response = response.strip()
    
    # Try to find pattern like "A,B,C,D,E" or "A, B, C, D, E"
    match = re.search(r'([A-E])\s*,\s*([A-E])\s*,\s*([A-E])\s*,\s*([A-E])\s*,\s*([A-E])', response.upper())
    if match:
        return ",".join(match.groups())
    
    # Try space-separated
    match = re.search(r'([A-E])\s+([A-E])\s+([A-E])\s+([A-E])\s+([A-E])', response.upper())
    if match:
        return ",".join(match.groups())
    
    # Try continuous letters
    match = re.search(r'([A-E])([A-E])([A-E])([A-E])([A-E])', response.upper())
    if match:
        return ",".join(match.groups())
    
    # Try to find any 5 letters A-E in order
    letters = re.findall(r'[A-E]', response.upper())
    if len(letters) >= 5:
        return ",".join(letters[:5])
    
    # Fallback: return cleaned response
    return response.upper().strip()


def compute_score(
    solution_str: str,
    ground_truth: Union[Dict, List, str, Any],
    **kwargs,
) -> Dict[str, Any]:
    """Compute score for ROCStories sentence ordering task.
    
    Reward = 60% position accuracy + 40% adjacent pair bonus
    
    Args:
        solution_str: Model's generated order (e.g., "B,C,E,D,A")
        ground_truth: Dict with 'gold_labels' or list of correct labels
    
    Returns:
        Dict with 'score' (0.0 to 1.0), 'acc', 'position_score', 'adjacent_score'
    """
    if ground_truth is None:
        return {
            "score": 0.0,
            "acc": 0.0,
            "position_score": 0.0,
            "adjacent_score": 0.0,
        }
    
    # Parse ground truth
    gold_labels = []
    if isinstance(ground_truth, dict):
        gold_labels = ground_truth.get("gold_labels", [])
        # Convert numpy array to list if needed
        if hasattr(gold_labels, 'tolist'):
            gold_labels = gold_labels.tolist()
        # If gold_labels is empty, try gold_answer
        if len(gold_labels) == 0 and "gold_answer" in ground_truth:
            gold_answer = ground_truth["gold_answer"]
            if hasattr(gold_answer, 'item'):
                gold_answer = gold_answer.item()
            gold_labels = _parse_order(str(gold_answer))
    elif isinstance(ground_truth, (list, tuple)):
        gold_labels = list(ground_truth)
    elif isinstance(ground_truth, str):
        gold_labels = _parse_order(ground_truth)
    else:
        # Try to convert to string and parse
        try:
            if hasattr(ground_truth, 'tolist'):
                gold_labels = ground_truth.tolist()
            else:
                gold_labels = _parse_order(str(ground_truth))
        except Exception:
            return {
                "score": 0.0,
                "acc": 0.0,
                "position_score": 0.0,
                "adjacent_score": 0.0,
            }
    
    # Ensure gold_labels is a list
    if hasattr(gold_labels, 'tolist'):
        gold_labels = gold_labels.tolist()
    
    if len(gold_labels) != 5:
        return {
            "score": 0.0,
            "acc": 0.0,
            "position_score": 0.0,
            "adjacent_score": 0.0,
        }
    
    # Extract and parse prediction
    answer = extract_answer(solution_str)
    pred_labels = _parse_order(answer)
    
    # Check if we have valid prediction
    if len(pred_labels) != 5:
        return {
            "score": 0.0,
            "acc": 0.0,
            "position_score": 0.0,
            "adjacent_score": 0.0,
        }
    
    # Check if all labels are valid and unique
    valid_labels = set(['A', 'B', 'C', 'D', 'E'])
    if set(pred_labels) != valid_labels:
        return {
            "score": 0.0,
            "acc": 0.0,
            "position_score": 0.0,
            "adjacent_score": 0.0,
        }
    
    # 1. Position accuracy (60%): fraction of sentences in correct position
    position_correct = sum(1 for p, g in zip(pred_labels, gold_labels) if p == g)
    position_score = position_correct / 5.0
    
    # 2. Adjacent pair bonus (40%): are consecutive sentences in correct relative order?
    pred_pos = {label: i for i, label in enumerate(pred_labels)}
    
    adjacent_correct = 0
    for i in range(4):  # 4 adjacent pairs
        gold_first = gold_labels[i]
        gold_second = gold_labels[i + 1]
        # In prediction, is gold_first before gold_second?
        if pred_pos[gold_first] < pred_pos[gold_second]:
            adjacent_correct += 1
    adjacent_score = adjacent_correct / 4.0
    
    # Combined reward: 60% position + 40% adjacent
    score = 0.6 * position_score + 0.4 * adjacent_score
    
    # Exact match accuracy
    acc = 1.0 if pred_labels == gold_labels else 0.0
    
    return {
        "score": score,
        "acc": acc,
        "position_score": position_score,
        "adjacent_score": adjacent_score,
    }
