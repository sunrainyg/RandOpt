# Copyright 2024
# Math reward scoring for MATH-500 dataset
# Extracts answers from \boxed{} format and compares with ground truth

import re
from typing import Optional, Tuple

_SOLUTION_CLIP_CHARS = 500


def normalize_answer(answer: str) -> str:
    """Normalize a math answer for comparison."""
    if answer is None:
        return ""
    # Remove spaces
    answer = answer.strip()
    # Remove leading/trailing whitespace inside
    answer = re.sub(r'\s+', ' ', answer)
    return answer


def simplify_latex(s: str) -> str:
    """Apply common LaTeX simplifications for comparison."""
    if not s:
        return ""
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "")  # Remove thin space (was " ", now "")
    s = s.replace("\\!", "")  # Remove negative thin space
    s = s.replace("\\;", "")  # Remove medium space
    s = s.replace("\\:", "")  # Remove medium space
    s = s.replace("\\ ", " ")  # Explicit space
    s = s.replace("\\quad", " ")
    s = s.replace("\\qquad", " ")
    s = s.replace("\\text{", "{")  # Remove \text wrapper
    s = s.replace("\\mathrm{", "{")
    s = s.replace("\\mathbf{", "{")
    s = s.replace("\\textbf{", "{")
    s = s.replace("\\dfrac", "\\frac")  # Normalize fraction
    s = s.replace("\\tfrac", "\\frac")
    s = s.replace("^{\\circ}", "^\\circ")  # Normalize degrees
    s = s.replace("°", "^\\circ")
    # Remove commas used as thousand separators in numbers (e.g., "11,111" -> "11111")
    # But preserve commas in coordinate pairs like "(-1, 6)"
    s = re.sub(r'(\d),(\d{3})', r'\1\2', s)  # 11,111 -> 11111
    s = re.sub(r'(\d),(\d{3})', r'\1\2', s)  # Apply twice for longer numbers
    s = re.sub(r'(\d),(\d{3})', r'\1\2', s)  # Apply thrice for even longer
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    return s


def try_numeric_comparison(answer: str, truth: str) -> bool:
    """Try to compare as numbers if both can be parsed."""
    try:
        # Handle fractions like \frac{a}{b}
        def parse_fraction(s):
            s = s.strip()
            # Match \frac{num}{denom}
            frac_match = re.match(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', s)
            if frac_match:
                num = float(frac_match.group(1))
                denom = float(frac_match.group(2))
                return num / denom
            # Try direct float parse
            return float(s)
        
        ans_val = parse_fraction(answer)
        truth_val = parse_fraction(truth)
        return abs(ans_val - truth_val) < 1e-9
    except (ValueError, ZeroDivisionError):
        return False


def normalize_for_comparison(s: str) -> str:
    """Normalize string for final comparison - remove all spaces and lowercase."""
    if not s:
        return ""
    # First apply latex simplification
    s = simplify_latex(s)
    # Remove ALL spaces (including inside coordinates)
    s = s.replace(" ", "")
    # Lowercase for text answers like "Evelyn"
    # But don't lowercase math - check if it's pure text
    if not re.search(r'[\\{}^_]', s):  # No LaTeX commands
        s = s.lower()
    return s


def try_coordinate_comparison(answer: str, truth: str) -> bool:
    """Try to compare coordinates/tuples like (-1, 6) vs (-1,6)."""
    # Extract numbers from coordinate-like strings
    def extract_coords(s):
        # Match patterns like (a, b) or (a,b) or [a, b]
        s = s.replace(" ", "")
        # Find all numbers (including negative and decimal)
        numbers = re.findall(r'-?\d+\.?\d*', s)
        return [float(n) for n in numbers] if numbers else None
    
    ans_coords = extract_coords(answer)
    truth_coords = extract_coords(truth)
    
    if ans_coords and truth_coords and len(ans_coords) == len(truth_coords):
        return all(abs(a - t) < 1e-9 for a, t in zip(ans_coords, truth_coords))
    return False


def extract_solution(solution_str: str, method: str = "strict") -> Optional[str]:
    """Extract the answer from a model response.
    
    Args:
        solution_str: The model's response string
        method: 'strict' looks for \\boxed{}, 'flexible' looks for last boxed or answer pattern
    
    Returns:
        Extracted answer string or None
    """
    assert method in ["strict", "flexible"]
    
    # For long strings, only look at the end (optimization)
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]
    
    if method == "strict":
        # Look for \boxed{...} pattern - handles nested braces
        # Find all \boxed{ occurrences and extract content with proper brace matching
        def extract_boxed_content(s):
            """Extract content from \boxed{...} with proper brace matching."""
            results = []
            i = 0
            while i < len(s):
                if s[i:].startswith('\\boxed{'):
                    start = i + 7  # After \boxed{
                    depth = 1
                    j = start
                    while j < len(s) and depth > 0:
                        if s[j] == '{':
                            depth += 1
                        elif s[j] == '}':
                            depth -= 1
                        j += 1
                    if depth == 0:
                        results.append(s[start:j-1])
                    i = j
                else:
                    i += 1
            return results
        
        matches = extract_boxed_content(solution_str)
        if matches:
            return matches[-1]  # Return last match
        
        # Fallback: Try regex patterns
        boxed_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = re.findall(boxed_pattern, solution_str)
        if matches:
            return matches[-1]
        
        # Try simpler pattern for single-level braces
        simple_pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(simple_pattern, solution_str)
        if matches:
            return matches[-1]
        
        return None
        
    elif method == "flexible":
        # First try strict method
        answer = extract_solution(solution_str, method="strict")
        if answer is not None:
            return answer
        
        # Try to find answer in \(...\) or $...$ at the end of sentences
        # Pattern: "is \(answer\)" or "is $answer$"
        inline_math_patterns = [
            r'is\s+\\\(([^)]+)\\\)\.?\s*$',  # is \(answer\).
            r'is\s+\$([^$]+)\$\.?\s*$',      # is $answer$.
            r'=\s*\\\(([^)]+)\\\)\.?\s*$',   # = \(answer\).
            r'=\s*\$([^$]+)\$\.?\s*$',       # = $answer$.
        ]
        
        for pattern in inline_math_patterns:
            matches = re.findall(pattern, solution_str, re.IGNORECASE | re.MULTILINE)
            if matches:
                return matches[-1].strip()
        
        # Try other patterns
        # Look for "answer is X" or "= X" at the end
        patterns = [
            r'answer\s+is\s*[:\s]*\**\\\(([^)]+)\\\)\**',  # answer is \(X\)
            r'answer\s+is\s*[:\s]*\**\$([^$]+)\$\**',      # answer is $X$
            r'answer\s+is\s*[:\s]*\**([^\n.*]+)\**',
            r'(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?(?:answer\s+is\s*)?[:\s]*\**([^\n.*]+)\**',
            r'(?:final\s+)?answer[:\s]+\**([^\n.*]+)\**',
            r'=\s*([^\n=]+)$',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, solution_str, re.IGNORECASE)
            if matches:
                return matches[-1].strip()
        
        return None


def compute_score(solution_str: str, ground_truth: str, method: str = "strict", 
                  format_score: float = 0.0, score: float = 1.0) -> float:
    """Compute the score for a MATH problem.
    
    Args:
        solution_str: The model's response
        ground_truth: The ground truth answer
        method: 'strict' or 'flexible' extraction
        format_score: Score when format is correct but answer is wrong
        score: Score for correct answer
    
    Returns:
        Score value
    """
    answer = extract_solution(solution_str, method=method)
    if answer is None:
        return 0.0
    
    # Normalize both answers for comparison
    norm_answer = normalize_answer(answer)
    norm_truth = normalize_answer(ground_truth)
    
    # Direct string match
    if norm_answer == norm_truth:
        return score
    
    # Try without spaces
    if norm_answer.replace(" ", "") == norm_truth.replace(" ", ""):
        return score
    
    # Try with LaTeX simplifications
    simp_answer = simplify_latex(norm_answer)
    simp_truth = simplify_latex(norm_truth)
    
    if simp_answer == simp_truth:
        return score
    
    if simp_answer.replace(" ", "") == simp_truth.replace(" ", ""):
        return score
    
    # Try fully normalized comparison (remove all spaces, handle case)
    final_answer = normalize_for_comparison(norm_answer)
    final_truth = normalize_for_comparison(norm_truth)
    
    if final_answer == final_truth:
        return score
    
    # Try numeric comparison for simple numeric answers
    if try_numeric_comparison(simp_answer, simp_truth):
        return score
    
    # Try coordinate/tuple comparison like (-1, 6) vs (-1,6)
    if try_coordinate_comparison(simp_answer, simp_truth):
        return score
    
    return format_score


def compute_score_with_debug(solution_str: str, ground_truth: str, method: str = "strict") -> Tuple[float, dict]:
    """Compute score with debug info showing what was compared.
    
    Returns:
        (score, debug_info) where debug_info contains comparison details
    """
    answer = extract_solution(solution_str, method=method)
    
    debug_info = {
        "method": method,
        "extracted": answer,
        "ground_truth": ground_truth,
        "match_type": None,
    }
    
    if answer is None:
        debug_info["match_type"] = "extraction_failed"
        return 0.0, debug_info
    
    norm_answer = normalize_answer(answer)
    norm_truth = normalize_answer(ground_truth)
    debug_info["normalized_answer"] = norm_answer
    debug_info["normalized_truth"] = norm_truth
    
    if norm_answer == norm_truth:
        debug_info["match_type"] = "exact_match"
        return 1.0, debug_info
    
    if norm_answer.replace(" ", "") == norm_truth.replace(" ", ""):
        debug_info["match_type"] = "no_space_match"
        return 1.0, debug_info
    
    simp_answer = simplify_latex(norm_answer)
    simp_truth = simplify_latex(norm_truth)
    debug_info["simplified_answer"] = simp_answer
    debug_info["simplified_truth"] = simp_truth
    
    if simp_answer == simp_truth:
        debug_info["match_type"] = "latex_simplified_match"
        return 1.0, debug_info
    
    if simp_answer.replace(" ", "") == simp_truth.replace(" ", ""):
        debug_info["match_type"] = "latex_no_space_match"
        return 1.0, debug_info
    
    # Fully normalized comparison
    final_answer = normalize_for_comparison(norm_answer)
    final_truth = normalize_for_comparison(norm_truth)
    debug_info["final_answer"] = final_answer
    debug_info["final_truth"] = final_truth
    
    if final_answer == final_truth:
        debug_info["match_type"] = "fully_normalized_match"
        return 1.0, debug_info
    
    if try_numeric_comparison(simp_answer, simp_truth):
        debug_info["match_type"] = "numeric_match"
        return 1.0, debug_info
    
    if try_coordinate_comparison(simp_answer, simp_truth):
        debug_info["match_type"] = "coordinate_match"
        return 1.0, debug_info
    
    debug_info["match_type"] = "no_match"
    return 0.0, debug_info

