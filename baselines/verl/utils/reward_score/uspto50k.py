"""USPTO-50K reward score computation for chemical reaction classification."""
import re
from typing import Any, Dict, Union


def extract_answer(response: str) -> str:
    """Extract class number from <answer>...</answer> tags or fallback to numbers."""
    # Try <answer> tags first
    matches = re.findall(r"<answer>\s*(\d+)\s*</answer>", response, re.IGNORECASE)
    if matches:
        return matches[-1]
    
    # Fallback: look for standalone numbers 1-10
    matches = re.findall(r'\b([1-9]|10)\b', response)
    if matches:
        return matches[-1]
    
    return ""


def compute_score(
    solution_str: str,
    ground_truth: Union[str, int],
    **kwargs,
) -> Dict[str, Any]:
    """Compute score for USPTO-50K reaction classification.
    
    Args:
        solution_str: Model's response containing the predicted class
        ground_truth: The correct reaction class (1-10)
    
    Returns:
        Dict with 'score' (0.0 or 1.0), 'acc', and 'pred'
    """
    pred = extract_answer(solution_str)
    
    if not pred:
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": "",
            "format_found": 0.0,
        }
    
    try:
        pred_class = int(pred.strip())
        true_class = int(ground_truth)
        correct = pred_class == true_class
        
        return {
            "score": 1.0 if correct else 0.0,
            "acc": 1.0 if correct else 0.0,
            "pred": str(pred_class),
            "format_found": 1.0,
        }
    except (ValueError, TypeError):
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": pred,
            "format_found": 1.0,  # Format was found but couldn't parse
        }
