"""MBPP reward score computation for Python code generation."""
import re
import signal
import threading
from typing import Any, Dict, List, Union
from contextlib import contextmanager


class TimeoutError(Exception):
    pass


def _is_main_thread():
    """Check if we're running in the main thread."""
    return threading.current_thread() is threading.main_thread()


@contextmanager
def timeout(seconds=5):
    """Context manager for execution timeout.
    
    Uses signal.SIGALRM if in main thread, otherwise skips timeout
    (since signal only works in main thread).
    """
    if not _is_main_thread():
        # Can't use signal in non-main thread, just yield without timeout
        yield
        return
    
    def handler(signum, frame):
        raise TimeoutError("Code execution timed out")
    
    try:
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except ValueError:
        # signal.signal can raise ValueError if not in main thread
        yield


def extract_code(response: str) -> str:
    """Extract code from <answer>...</answer> tags or code blocks."""
    # Try <answer> tags first
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    # Fallback: try to extract python code block
    code_matches = re.findall(r"```python\n?(.*?)```", response, re.DOTALL)
    if code_matches:
        return code_matches[-1].strip()
    
    # Try generic code blocks
    code_matches = re.findall(r"```\n?(.*?)```", response, re.DOTALL)
    if code_matches:
        return code_matches[-1].strip()
    
    return ""


def execute_code_with_tests(
    code: str,
    test_list: List[str],
    test_setup_code: str = "",
    timeout_sec: int = 5
) -> Dict[str, Any]:
    """Execute code and run test assertions.
    
    Returns dict with 'passed' (bool), 'passed_count', 'total_tests', 'error'
    """
    if not code.strip():
        return {
            "passed": False,
            "passed_count": 0,
            "total_tests": len(test_list),
            "error": "Empty code",
        }
    
    # Build execution environment
    full_code = ""
    # Handle test_setup_code - could be string, list, or numpy array
    if test_setup_code is not None:
        # Check if it's a non-empty string
        if isinstance(test_setup_code, str) and test_setup_code.strip():
            full_code += test_setup_code + "\n"
        # Check if it's a non-empty list/array
        elif hasattr(test_setup_code, '__len__') and len(test_setup_code) > 0:
            if hasattr(test_setup_code, 'tolist'):
                test_setup_code = test_setup_code.tolist()
            setup_str = "\n".join([s for s in test_setup_code if s])
            if setup_str:
                full_code += setup_str + "\n"
    full_code += code + "\n"
    
    # Execute code to define functions
    namespace = {}
    try:
        with timeout(timeout_sec):
            exec(full_code, namespace)
    except TimeoutError:
        return {
            "passed": False,
            "passed_count": 0,
            "total_tests": len(test_list),
            "error": "Code execution timed out",
        }
    except Exception as e:
        return {
            "passed": False,
            "passed_count": 0,
            "total_tests": len(test_list),
            "error": f"Code execution error: {str(e)[:100]}",
        }
    
    # Run each test
    passed_count = 0
    for test in test_list:
        try:
            with timeout(timeout_sec):
                exec(test, namespace)
            passed_count += 1
        except AssertionError:
            continue
        except TimeoutError:
            continue
        except Exception:
            continue
    
    return {
        "passed": passed_count == len(test_list),
        "passed_count": passed_count,
        "total_tests": len(test_list),
        "error": None if passed_count == len(test_list) else "Some tests failed",
    }


def compute_score(
    solution_str: str,
    ground_truth: Union[Dict, Any],
    **kwargs,
) -> Dict[str, Any]:
    """Compute score for MBPP code generation.
    
    Args:
        solution_str: Model's response containing the code
        ground_truth: Dict with 'test_list' and optionally 'test_setup_code'
    
    Returns:
        Dict with 'score' (0.0 or 1.0), 'acc', 'passed_count', 'total_tests'
    """
    code = extract_code(solution_str)
    
    if not code:
        return {
            "score": 0.0,
            "acc": 0.0,
            "passed_count": 0,
            "total_tests": 0,
            "format_found": 0.0,
            "error": "No code found in response",
        }
    
    # Handle ground_truth format
    if isinstance(ground_truth, dict):
        test_list = ground_truth.get("test_list", [])
        test_setup_code = ground_truth.get("test_setup_code", "")
    else:
        # Try to convert from other formats
        try:
            if hasattr(ground_truth, 'item'):
                ground_truth = dict(ground_truth)
            test_list = ground_truth.get("test_list", [])
            test_setup_code = ground_truth.get("test_setup_code", "")
        except Exception:
            return {
                "score": 0.0,
                "acc": 0.0,
                "passed_count": 0,
                "total_tests": 0,
                "format_found": 1.0,
                "error": "Invalid ground_truth format",
            }
    
    # Convert test_list if needed (e.g., from numpy array)
    if hasattr(test_list, 'tolist'):
        test_list = test_list.tolist()
    test_list = list(test_list)
    
    # Convert test_setup_code if it's a list or numpy array
    if test_setup_code is None:
        test_setup_code = ""
    elif hasattr(test_setup_code, 'tolist'):
        # numpy array - convert to list then join
        items = test_setup_code.tolist()
        test_setup_code = "\n".join([s for s in items if s]) if items else ""
    elif isinstance(test_setup_code, list):
        test_setup_code = "\n".join([s for s in test_setup_code if s])
    elif not isinstance(test_setup_code, str):
        test_setup_code = ""
    
    result = execute_code_with_tests(code, test_list, test_setup_code)
    
    return {
        "score": 1.0 if result["passed"] else 0.0,
        "acc": 1.0 if result["passed"] else 0.0,
        "passed_count": result["passed_count"],
        "total_tests": result["total_tests"],
        "format_found": 1.0,
        "error": result.get("error"),
    }
