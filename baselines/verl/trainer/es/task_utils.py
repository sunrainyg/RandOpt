"""
Task-specific utilities for ES training.
Includes reward functions and prompt processors for various tasks.
"""

from typing import Any, Callable, Dict
from vllm import TokensPrompt


def _has_chat_template(tokenizer) -> bool:
    """Check if tokenizer has a chat template (instruct models have one, base models don't)."""
    return hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None


def _format_messages_for_base_model(messages: list, tokenizer) -> str:
    """Format messages for base model (no chat template).
    
    Converts messages to a simple text format:
    - system: content followed by newlines
    - user: "### Input:\n{content}\n\n### Output:\n"
    - assistant: content directly
    """
    text_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # Extract text from content (handle both string and multi-modal formats)
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_pieces = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_pieces.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_pieces.append(item)
            text = "".join(text_pieces)
        else:
            text = str(content) if content else ""
        
        if role == "system":
            text_parts.append(f"{text}\n\n")
        elif role == "user":
            text_parts.append(f"### Input:\n{text}\n\n### Output:\n")
        elif role == "assistant":
            text_parts.append(f"{text}")
        else:
            text_parts.append(f"{text}\n")
    
    return "".join(text_parts)


def _apply_chat_template_or_format(messages: list, tokenizer, add_generation_prompt: bool = True) -> str:
    """Apply chat template if available, otherwise format for base model.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: The tokenizer to use
        add_generation_prompt: Whether to add generation prompt (only used for instruct models)
    
    Returns:
        Formatted string ready for tokenization
    """
    if _has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False
        )
    else:
        return _format_messages_for_base_model(messages, tokenizer)


# ======================= GSM8K Task =======================

GSM8K_SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)

GSM8K_USER_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Show your work and put your final answer after ####.\n\n"
    "Problem: {question}"
)

GSM8K_RESPONSE_PROMPT = "Let me solve this step by step.\n"


def create_gsm8k_prompt_processor(
    system_message: str = None,
    user_template: str = None,
    response_prompt: str = None
) -> Callable:
    """Create a prompt processor for GSM8K task."""
    sys_msg = system_message or GSM8K_SYSTEM_MESSAGE
    user_tmpl = user_template or GSM8K_USER_TEMPLATE
    resp_prompt = response_prompt or GSM8K_RESPONSE_PROMPT
    
    def process_context(task_data: Dict[str, Any], tokenizer) -> TokensPrompt:
        """Process GSM8K task data into a prompt.
        
        Supports two formats:
        1. verl parquet format: {"prompt": [{"role": "user", "content": "..."}], "reward_model": {"ground_truth": "..."}}
        2. Simple format: {"question": "...", "answer": "..."}
        """
        # Check for verl parquet format
        if "prompt" in task_data and isinstance(task_data["prompt"], (list, tuple)):
            # verl format - use existing prompt messages
            messages = list(task_data["prompt"])
            if not isinstance(messages[0], dict):
                # Numpy array of dicts
                messages = [dict(m) for m in messages]
            formatted = _apply_chat_template_or_format(messages, tokenizer)
        else:
            # Simple format: {"question": "...", "answer": "..."}
            question = task_data.get("question", task_data.get("extra_info", {}).get("question", ""))
            user_content = user_tmpl.format(question=question)
            
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_content}
            ]
            
            formatted = _apply_chat_template_or_format(messages, tokenizer)
        
        prompts = tokenizer(formatted)
        return TokensPrompt(prompt_token_ids=prompts['input_ids'])
    
    return process_context


def create_gsm8k_reward_fn(
    method: str = "strict",
    format_score: float = 0.1,
    correct_score: float = 1.0,
    response_prompt: str = None
) -> Callable:
    """Create a reward function for GSM8K task."""
    from verl.utils.reward_score.gsm8k import compute_score, extract_solution
    resp_prompt = response_prompt or GSM8K_RESPONSE_PROMPT
    
    def reward_fn(response: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute reward for GSM8K response.
        
        Supports two formats:
        1. verl parquet format: {"reward_model": {"ground_truth": "72"}, "extra_info": {"answer": "..."}}
        2. Simple format: {"question": "...", "answer": "... #### 72"}
        """
        import re
        
        # Get ground truth - check verl format first
        if "reward_model" in task_data and "ground_truth" in task_data["reward_model"]:
            ground_truth = str(task_data["reward_model"]["ground_truth"]).replace(",", "")
        else:
            # Simple format: extract from answer field
            answer_str = task_data.get("answer", task_data.get("extra_info", {}).get("answer", ""))
            
            # GSM8K format: "... #### 123"
            gt_match = re.search(r"####\s*([\d,\.\-]+)", answer_str)
            if gt_match:
                ground_truth = gt_match.group(1).replace(",", "")
            else:
                # Try to find last number
                numbers = re.findall(r"([\d,\.\-]+)", answer_str)
                ground_truth = numbers[-1].replace(",", "") if numbers else ""
        
        # Compute score
        full_response = resp_prompt + response if resp_prompt else response
        result = compute_score(
            solution_str=full_response,
            ground_truth=ground_truth,
            method=method,
            format_score=format_score,
            score=correct_score
        )
        
        # Extract predicted answer for logging
        pred_answer = extract_solution(full_response, method=method)
        
        reward_val = result["score"]
        answer_reward = result["acc"]
        format_reward = 1.0 if pred_answer is not None else 0.0
        
        return {
            "reward": reward_val,
            "reward_info": {
                "format_reward": format_reward,
                "answer_reward": answer_reward,
                "pred_answer": pred_answer,
                "ground_truth": ground_truth,
            }
        }
    
    return reward_fn


# ======================= MATH Task =======================

MATH_SYSTEM_MESSAGE = (
    "You are a helpful math assistant. Solve the problem step by step and "
    "put your final answer in \\boxed{}."
)

MATH_USER_TEMPLATE = "Problem: {problem}"

MATH_RESPONSE_PROMPT = ""


def create_math_prompt_processor(
    system_message: str = None,
    user_template: str = None,
) -> Callable:
    """Create a prompt processor for MATH task."""
    sys_msg = system_message or MATH_SYSTEM_MESSAGE
    user_tmpl = user_template or MATH_USER_TEMPLATE
    
    def process_context(task_data: Dict[str, Any], tokenizer) -> TokensPrompt:
        """Process MATH task data into a prompt.
        
        Supports two formats:
        1. verl parquet format: {"prompt": [{"role": "user", "content": "..."}], "reward_model": {"ground_truth": "..."}}
        2. Simple format: {"problem": "...", "answer": "..."}
        """
        # Check for verl parquet format
        if "prompt" in task_data and isinstance(task_data["prompt"], (list, tuple)):
            # verl format - use existing prompt messages
            messages = list(task_data["prompt"])
            if not isinstance(messages[0], dict):
                # Numpy array of dicts
                messages = [dict(m) for m in messages]
            formatted = _apply_chat_template_or_format(messages, tokenizer)
        else:
            # Simple format: {"problem": "...", "answer": "..."}
            problem = task_data.get("problem", task_data.get("question", ""))
            user_content = user_tmpl.format(problem=problem)
            
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_content}
            ]
            
            formatted = _apply_chat_template_or_format(messages, tokenizer)
        
        prompts = tokenizer(formatted)
        return TokensPrompt(prompt_token_ids=prompts['input_ids'])
    
    return process_context


def create_math_reward_fn() -> Callable:
    """Create a reward function for MATH task."""
    from verl.utils.reward_score.math_reward import compute_score
    
    def reward_fn(response: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute reward for MATH response.
        
        Supports two formats:
        1. verl parquet format: {"reward_model": {"ground_truth": "..."}}
        2. Simple format: {"answer": "...", "solution": "..."}
        """
        # Get ground truth - check verl format first
        if "reward_model" in task_data and "ground_truth" in task_data["reward_model"]:
            ground_truth = str(task_data["reward_model"]["ground_truth"])
        else:
            ground_truth = task_data.get("answer", task_data.get("solution", ""))
        
        score = compute_score(
            solution_str=response,
            ground_truth=ground_truth,
        )
        
        return {
            "reward": float(score),
            "reward_info": {
                "format_reward": 1.0 if score > 0 else 0.0,
                "answer_reward": float(score),
                "ground_truth": ground_truth,
            }
        }
    
    return reward_fn


# ======================= MATH-500 Task =======================

def create_math500_prompt_processor() -> Callable:
    """Create a prompt processor for MATH-500 task (uses verl parquet format)."""
    
    def process_context(task_data: Dict[str, Any], tokenizer) -> TokensPrompt:
        """Process MATH-500 task data into a prompt.
        
        Uses verl parquet format: {"prompt": [{"role": "user", "content": "..."}]}
        """
        # verl parquet format - use existing prompt messages
        # Check for prompt field (can be list, tuple, or numpy array)
        if "prompt" in task_data and hasattr(task_data["prompt"], '__len__') and not isinstance(task_data["prompt"], str):
            messages = list(task_data["prompt"])
            if not isinstance(messages[0], dict):
                messages = [dict(m) for m in messages]
            formatted = _apply_chat_template_or_format(messages, tokenizer)
        else:
            raise ValueError("MATH-500 task requires verl parquet format with 'prompt' field")
        
        prompts = tokenizer(formatted)
        return TokensPrompt(prompt_token_ids=prompts['input_ids'])
    
    return process_context


def create_math500_reward_fn() -> Callable:
    """Create a reward function for MATH-500 task using the specific implementation."""
    from verl.utils.reward_score.math500 import compute_score
    
    def reward_fn(response: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute reward for MATH-500 response."""
        # Get ground truth from verl parquet format
        if "reward_model" in task_data and "ground_truth" in task_data["reward_model"]:
            ground_truth = str(task_data["reward_model"]["ground_truth"])
        else:
            ground_truth = task_data.get("answer", task_data.get("solution", ""))
        
        result = compute_score(
            solution_str=response,
            ground_truth=ground_truth,
        )
        
        return {
            "reward": result["score"],
            "reward_info": {
                "format_reward": result.get("format_found", 0.0),  # 1.0 if answer was extracted, 0.0 otherwise
                "answer_reward": result["acc"],  # 1.0 if answer is correct, 0.0 otherwise
                "pred": result.get("pred", ""),  # Extracted prediction for debugging
                "ground_truth": ground_truth,
            }
        }
    
    return reward_fn


# ======================= OlympiadBench Task =======================

def create_olympiadbench_prompt_processor() -> Callable:
    """Create a prompt processor for OlympiadBench task (uses verl parquet format)."""
    
    def process_context(task_data: Dict[str, Any], tokenizer) -> TokensPrompt:
        """Process OlympiadBench task data into a prompt.
        
        Uses verl parquet format: {"prompt": [{"role": "user", "content": "..."}]}
        """
        # verl parquet format - use existing prompt messages
        # Check for prompt field (can be list, tuple, or numpy array)
        if "prompt" in task_data and hasattr(task_data["prompt"], '__len__') and not isinstance(task_data["prompt"], str):
            messages = list(task_data["prompt"])
            if not isinstance(messages[0], dict):
                messages = [dict(m) for m in messages]
            formatted = _apply_chat_template_or_format(messages, tokenizer)
        else:
            raise ValueError("OlympiadBench task requires verl parquet format with 'prompt' field")
        
        prompts = tokenizer(formatted)
        return TokensPrompt(prompt_token_ids=prompts['input_ids'])
    
    return process_context


def create_olympiadbench_reward_fn() -> Callable:
    """Create a reward function for OlympiadBench task using the specific implementation."""
    from verl.utils.reward_score.olympiadbench import compute_score
    
    def reward_fn(response: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute reward for OlympiadBench response."""
        # Get ground truth from verl parquet format
        if "reward_model" in task_data and "ground_truth" in task_data["reward_model"]:
            ground_truth = str(task_data["reward_model"]["ground_truth"])
        else:
            ground_truth = task_data.get("answer", "")
        
        # Get extra_info for answer_type
        extra_info = task_data.get("extra_info", {})
        if hasattr(extra_info, 'item'):
            extra_info = dict(extra_info)
        
        result = compute_score(
            solution_str=response,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        
        return {
            "reward": result["score"],
            "reward_info": {
                "format_reward": result.get("format_found", 0.0),  # 1.0 if answer was extracted, 0.0 otherwise
                "answer_reward": result["acc"],  # 1.0 if answer is correct, 0.0 otherwise
                "pred": result.get("pred", ""),  # Extracted prediction for debugging
                "ground_truth": ground_truth,
            }
        }
    
    return reward_fn


# ======================= Countdown Task =======================

COUNTDOWN_RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


def _get_countdown_data(task_data: Dict[str, Any]):
    """Extract numbers and target from task data (supports both JSON and parquet formats)."""
    # Parquet format: {"reward_model": {"ground_truth": {"target": ..., "numbers": ...}}}
    if "reward_model" in task_data and "ground_truth" in task_data["reward_model"]:
        gt = task_data["reward_model"]["ground_truth"]
        if isinstance(gt, dict):
            return gt.get("numbers", gt.get("nums")), gt.get("target")
    
    # Simple JSON format: {"numbers": [...], "target": ...}
    numbers = task_data.get("numbers", task_data.get("nums"))
    target = task_data.get("target")
    return numbers, target


def create_countdown_prompt_processor(
    system_message: str = None,
    user_template: str = None,
) -> Callable:
    """Create a prompt processor for Countdown task (supports parquet format)."""
    
    def process_context(task_data: Dict[str, Any], tokenizer) -> TokensPrompt:
        """Process Countdown task data into a prompt.
        
        Supports two formats:
        1. verl parquet format: {"prompt": [{"role": "user", "content": "..."}], "reward_model": {"ground_truth": {...}}}
           NOTE: In verl parquet format, content may already contain the full chat template
        2. Simple format: {"numbers": [...], "target": ...}
        """
        # Check for verl parquet format
        if "prompt" in task_data and isinstance(task_data["prompt"], (list, tuple)):
            messages = list(task_data["prompt"])
            if isinstance(messages[0], dict):
                msg = messages[0] if not hasattr(messages[0], 'item') else dict(messages[0])
            else:
                msg = dict(messages[0])
            
            content = msg.get("content", "")
            
            # Check if content already contains chat template markers (pre-formatted)
            # verl parquet format often has the full formatted string in content
            if "<|im_start|>" in content or "<|begin_of_text|>" in content:
                # Content is already fully formatted, use it directly
                formatted = content
            else:
                # Content is not pre-formatted, apply chat template or format for base model
                if not isinstance(messages[0], dict):
                    messages = [dict(m) for m in messages]
                formatted = _apply_chat_template_or_format(messages, tokenizer)
        else:
            # Simple format - build prompt from numbers/target
            numbers, target = _get_countdown_data(task_data)
            user_content = (
                f"Using the numbers {numbers}, create an equation that equals {target}. "
                "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
                "Show your work in <think> </think> tags. "
                "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer."},
                {"role": "user", "content": user_content}
            ]
            formatted = _apply_chat_template_or_format(messages, tokenizer)
        
        prompts = tokenizer(formatted)
        return TokensPrompt(prompt_token_ids=prompts['input_ids'])
    
    return process_context


def _answer_reward(answer_content, target, numbers):
    """Return 1.0 only when answer uses all numbers exactly once and reaches target."""
    import re
    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not answer_content:
        return 0.0
    if not re.match(allowed_chars, answer_content):
        return 0.0

    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted([int(n) for n in numbers]):
        return 0.0

    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except Exception:
        return 0.0
    return 0.0


def _format_reward(response):
    """Check whether response follows <think>...</think><answer>...</answer> format."""
    import re
    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)

    if full_format_match:
        return 1.0

    reward = 0.0
    if think_match:
        reward += 0.1
    if answer_match:
        reward += 0.5
    return reward


def create_countdown_reward_fn(response_prompt: str = None) -> Callable:
    """Create a reward function for Countdown task."""
    resp_prompt = response_prompt or COUNTDOWN_RESPONSE_PROMPT
    
    def reward_fn(response: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute reward for Countdown response.
        
        reward = 0.1 * format_reward + answer_reward
        """
        import re
        
        numbers, target = _get_countdown_data(task_data)
        
        if isinstance(target, str):
            try:
                target = float(target)
            except (ValueError, TypeError):
                return {"reward": 0.0, "reward_info": {"format_reward": 0.0, "answer_reward": 0.0, "error": "invalid_target"}}
        
        full_response = resp_prompt + response
        
        format_rwd = _format_reward(full_response)
        
        answer_matches = re.findall(r"<answer>(.*?)</answer>", full_response, re.DOTALL)
        if answer_matches:
            answer_rwd = _answer_reward(answer_matches[-1].strip(), target, numbers)
        else:
            answer_rwd = 0.0
        
        total_reward = 0.1 * format_rwd + answer_rwd
        
        return {
            "reward": total_reward,
            "reward_info": {
                "format_reward": format_rwd,
                "answer_reward": answer_rwd,
            },
        }
    
    return reward_fn


# ======================= USPTO-50K Task =======================

USPTO50K_SYSTEM_MESSAGE = (
    "You are an expert organic chemist. Your task is to classify chemical reactions "
    "into one of 10 standard reaction categories based on the transformation type.\n\n"
    "## Reaction Classes:\n"
    "1: Heteroatom alkylation/arylation - N, O, S attacking C (e.g., SN2, ether formation)\n"
    "2: Acylation - Forming C=O bonds with N, O, S (e.g., amide, ester formation)\n"
    "3: C-C bond formation - New C-C bonds (e.g., Suzuki, Heck, Grignard)\n"
    "4: Heterocycle formation - Creating rings with N, O, S\n"
    "5: Protections - Adding protecting groups (Boc, Bn, TBS, etc.)\n"
    "6: Deprotections - Removing protecting groups\n"
    "7: Reductions - Adding H, removing O (e.g., ketone→alcohol, nitro→amine)\n"
    "8: Oxidations - Adding O, removing H (e.g., alcohol→ketone)\n"
    "9: Functional group interconversion - Changing one FG to another\n"
    "10: Functional group addition - Adding new FG to molecule (e.g., halogenation)"
)

USPTO50K_USER_TEMPLATE = (
    "Classify this reaction:\n\n"
    "Reactants >> Product:\n{rxn_smiles}\n\n"
    "Analyze the key transformation and output the class number (1-10) in <answer>X</answer> tags."
)


def create_uspto50k_prompt_processor() -> Callable:
    """Create a prompt processor for USPTO-50K task."""
    
    def process_context(task_data: Dict[str, Any], tokenizer) -> TokensPrompt:
        """Process USPTO-50K task data into a prompt."""
        # Check for verl parquet format
        if "prompt" in task_data and hasattr(task_data["prompt"], '__len__') and not isinstance(task_data["prompt"], str):
            messages = list(task_data["prompt"])
            if not isinstance(messages[0], dict):
                messages = [dict(m) for m in messages]
            formatted = _apply_chat_template_or_format(messages, tokenizer)
        else:
            # Simple format: build from rxn_smiles
            rxn_smiles = task_data.get("rxn_smiles", "")
            user_content = USPTO50K_USER_TEMPLATE.format(rxn_smiles=rxn_smiles)
            
            messages = [
                {"role": "system", "content": USPTO50K_SYSTEM_MESSAGE},
                {"role": "user", "content": user_content}
            ]
            
            formatted = _apply_chat_template_or_format(messages, tokenizer)
        
        prompts = tokenizer(formatted)
        return TokensPrompt(prompt_token_ids=prompts['input_ids'])
    
    return process_context


def create_uspto50k_reward_fn() -> Callable:
    """Create a reward function for USPTO-50K task."""
    from verl.utils.reward_score.uspto50k import compute_score
    
    def reward_fn(response: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute reward for USPTO-50K response."""
        # Get ground truth
        if "reward_model" in task_data and "ground_truth" in task_data["reward_model"]:
            ground_truth = task_data["reward_model"]["ground_truth"]
        else:
            ground_truth = task_data.get("ground_truth", task_data.get("class", ""))
        
        result = compute_score(response, ground_truth)
        
        return {
            "reward": result["score"],
            "reward_info": {
                "format_reward": result.get("format_found", 0.0),
                "answer_reward": result["acc"],
                "pred": result.get("pred", ""),
                "ground_truth": str(ground_truth),
            }
        }
    
    return reward_fn


# ======================= CommonGen Task =======================

COMMON_GEN_USER_TEMPLATE = (
    "Generate a single, coherent sentence that uses ALL of the following concepts: {concepts}\n\n"
    "Requirements:\n"
    "- Use all concepts (any form/tense is acceptable)\n"
    "- Write exactly one natural sentence\n"
    "- Do not explain, just output the sentence directly."
)


def create_common_gen_prompt_processor() -> Callable:
    """Create a prompt processor for CommonGen task."""
    
    def process_context(task_data: Dict[str, Any], tokenizer) -> TokensPrompt:
        """Process CommonGen task data into a prompt."""
        # Check for verl parquet format
        if "prompt" in task_data and hasattr(task_data["prompt"], '__len__') and not isinstance(task_data["prompt"], str):
            messages = list(task_data["prompt"])
            if not isinstance(messages[0], dict):
                messages = [dict(m) for m in messages]
            formatted = _apply_chat_template_or_format(messages, tokenizer)
        else:
            # Simple format: build from concepts
            concepts = task_data.get("concepts", task_data.get("ground_truth", []))
            if hasattr(concepts, 'tolist'):
                concepts = concepts.tolist()
            concepts_str = ', '.join(concepts) if isinstance(concepts, list) else str(concepts)
            user_content = COMMON_GEN_USER_TEMPLATE.format(concepts=concepts_str)
            
            messages = [{"role": "user", "content": user_content}]
            
            formatted = _apply_chat_template_or_format(messages, tokenizer)
        
        prompts = tokenizer(formatted)
        return TokensPrompt(prompt_token_ids=prompts['input_ids'])
    
    return process_context


def create_common_gen_reward_fn() -> Callable:
    """Create a reward function for CommonGen task."""
    from verl.utils.reward_score.common_gen import compute_score
    
    def reward_fn(response: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute reward for CommonGen response."""
        # Get ground truth (list of concepts)
        if "reward_model" in task_data and "ground_truth" in task_data["reward_model"]:
            ground_truth = task_data["reward_model"]["ground_truth"]
        else:
            ground_truth = task_data.get("ground_truth", task_data.get("concepts", []))
        
        result = compute_score(response, ground_truth)
        
        return {
            "reward": result["score"],
            "reward_info": {
                "format_reward": 1.0,  # No specific format required
                "answer_reward": result["acc"],
                "found_count": result.get("found_count", 0),
                "total_concepts": result.get("total_concepts", 0),
            }
        }
    
    return reward_fn


# ======================= MBPP Task =======================

MBPP_SYSTEM_MESSAGE = (
    "You are a Python programming assistant. Write clean, correct Python code to solve the given problem."
)

MBPP_USER_TEMPLATE = (
    "{text}\n\n"
    "Your code should pass these tests:\n{tests}\n\n"
    "Think through your solution in <think> </think> tags.\n"
    "Return your final Python code in <answer> </answer> tags, e.g.:\n"
    "<answer>\ndef solution(x):\n    return x + 1\n</answer>"
)


def create_mbpp_prompt_processor() -> Callable:
    """Create a prompt processor for MBPP task."""
    
    def process_context(task_data: Dict[str, Any], tokenizer) -> TokensPrompt:
        """Process MBPP task data into a prompt."""
        # Check for verl parquet format
        if "prompt" in task_data and hasattr(task_data["prompt"], '__len__') and not isinstance(task_data["prompt"], str):
            messages = list(task_data["prompt"])
            if not isinstance(messages[0], dict):
                messages = [dict(m) for m in messages]
            formatted = _apply_chat_template_or_format(messages, tokenizer)
        else:
            # Simple format: build from text and test_list
            text = task_data.get("text", "")
            test_list = task_data.get("test_list", [])
            if hasattr(test_list, 'tolist'):
                test_list = test_list.tolist()
            tests_str = "\n".join(test_list[:3]) if test_list else ""
            
            user_content = MBPP_USER_TEMPLATE.format(text=text, tests=tests_str)
            
            messages = [
                {"role": "system", "content": MBPP_SYSTEM_MESSAGE},
                {"role": "user", "content": user_content}
            ]
            
            formatted = _apply_chat_template_or_format(messages, tokenizer)
        
        prompts = tokenizer(formatted)
        return TokensPrompt(prompt_token_ids=prompts['input_ids'])
    
    return process_context


def create_mbpp_reward_fn() -> Callable:
    """Create a reward function for MBPP task."""
    from verl.utils.reward_score.mbpp import compute_score
    
    def reward_fn(response: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute reward for MBPP response."""
        # Get ground truth (dict with test_list)
        if "reward_model" in task_data and "ground_truth" in task_data["reward_model"]:
            ground_truth = task_data["reward_model"]["ground_truth"]
        else:
            ground_truth = {
                "test_list": task_data.get("test_list", []),
                "test_setup_code": task_data.get("test_setup_code", ""),
            }
        
        result = compute_score(response, ground_truth)
        
        return {
            "reward": result["score"],
            "reward_info": {
                "format_reward": result.get("format_found", 0.0),
                "answer_reward": result["acc"],
                "passed_count": result.get("passed_count", 0),
                "total_tests": result.get("total_tests", 0),
                "error": result.get("error"),
            }
        }
    
    return reward_fn


# ======================= ROCStories Task =======================

ROCSTORIES_SYSTEM_MESSAGE = (
    "You are a helpful assistant that excels at story comprehension and logical reasoning. "
    "Given shuffled sentences from a story, you carefully analyze the narrative flow and "
    "temporal cues to determine the correct chronological order."
)


def create_rocstories_prompt_processor() -> Callable:
    """Create a prompt processor for ROCStories task."""
    
    def process_context(task_data: Dict[str, Any], tokenizer) -> TokensPrompt:
        """Process ROCStories task data into a prompt."""
        # Check for verl parquet format
        if "prompt" in task_data and hasattr(task_data["prompt"], '__len__') and not isinstance(task_data["prompt"], str):
            messages = list(task_data["prompt"])
            if not isinstance(messages[0], dict):
                messages = [dict(m) for m in messages]
            formatted = _apply_chat_template_or_format(messages, tokenizer)
        else:
            raise ValueError("ROCStories task requires verl parquet format with 'prompt' field")
        
        prompts = tokenizer(formatted)
        return TokensPrompt(prompt_token_ids=prompts['input_ids'])
    
    return process_context


def create_rocstories_reward_fn() -> Callable:
    """Create a reward function for ROCStories task."""
    from verl.utils.reward_score.rocstories import compute_score
    
    def reward_fn(response: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute reward for ROCStories response."""
        # Get ground truth
        if "reward_model" in task_data and "ground_truth" in task_data["reward_model"]:
            ground_truth = task_data["reward_model"]["ground_truth"]
        else:
            ground_truth = task_data.get("ground_truth", task_data.get("gold_labels", []))
        
        result = compute_score(response, ground_truth)
        
        return {
            "reward": result["score"],
            "reward_info": {
                "format_reward": 1.0 if result["score"] > 0 else 0.0,
                "answer_reward": result["acc"],
                "position_score": result.get("position_score", 0.0),
                "adjacent_score": result.get("adjacent_score", 0.0),
            }
        }
    
    return reward_fn


# ======================= Task Factory =======================

def get_task_components(task_type: str, config: Dict[str, Any] = None) -> tuple:
    """
    Get prompt processor and reward function for a given task type.
    
    Args:
        task_type: One of "countdown", "gsm8k", "math", "math500", "olympiadbench", or "custom"
        config: Optional configuration dict with task-specific settings
    
    Returns:
        Tuple of (prompt_processor, reward_fn)
    """
    config = config or {}
    
    # Get config values
    user_template = config.get("user_template")
    system_message = config.get("system_message")
    response_prompt = config.get("response_prompt")
    
    # Check if templates are task-specific (avoid using wrong templates)
    # Countdown templates have {numbers} and {target}
    is_countdown_template = user_template and "{numbers}" in user_template
    # GSM8K templates have {question}
    is_gsm8k_template = user_template and "{question}" in user_template
    # MATH templates have {problem}
    is_math_template = user_template and "{problem}" in user_template
    
    if task_type == "countdown":
        prompt_processor = create_countdown_prompt_processor(
            system_message=system_message,
            user_template=user_template if is_countdown_template else None,
        )
        reward_fn = create_countdown_reward_fn(
            response_prompt=response_prompt,
        )
    
    elif task_type == "gsm8k":
        # Only use custom templates if they're appropriate for GSM8K
        prompt_processor = create_gsm8k_prompt_processor(
            system_message=system_message if not is_countdown_template else None,
            user_template=user_template if is_gsm8k_template else None,
            response_prompt=response_prompt if not is_countdown_template else None,
        )
        reward_fn = create_gsm8k_reward_fn(
            method=config.get("extraction_method", "strict"),
            format_score=config.get("format_score", 0.1),
            correct_score=config.get("correct_score", 1.0),
            response_prompt=response_prompt if not is_countdown_template else None,
        )
    
    elif task_type == "math":
        # Only use custom templates if they're appropriate for MATH
        prompt_processor = create_math_prompt_processor(
            system_message=system_message if not is_countdown_template else None,
            user_template=user_template if is_math_template else None,
        )
        reward_fn = create_math_reward_fn()
    
    elif task_type == "math500":
        # MATH-500 specific task with dedicated reward function
        prompt_processor = create_math500_prompt_processor()
        reward_fn = create_math500_reward_fn()
    
    elif task_type == "olympiadbench":
        # OlympiadBench specific task with dedicated reward function
        prompt_processor = create_olympiadbench_prompt_processor()
        reward_fn = create_olympiadbench_reward_fn()
    
    elif task_type == "uspto50k":
        # USPTO-50K chemical reaction classification
        prompt_processor = create_uspto50k_prompt_processor()
        reward_fn = create_uspto50k_reward_fn()
    
    elif task_type == "common_gen":
        # CommonGen constrained text generation
        prompt_processor = create_common_gen_prompt_processor()
        reward_fn = create_common_gen_reward_fn()
    
    elif task_type == "mbpp":
        # MBPP Python code generation
        prompt_processor = create_mbpp_prompt_processor()
        reward_fn = create_mbpp_reward_fn()
    
    elif task_type == "rocstories":
        # ROCStories sentence ordering
        prompt_processor = create_rocstories_prompt_processor()
        reward_fn = create_rocstories_reward_fn()
    
    else:
        raise ValueError(f"Unknown task type: {task_type}. Use 'countdown', 'gsm8k', 'math', 'math500', 'olympiadbench', 'uspto50k', 'common_gen', 'mbpp', 'rocstories', or implement custom.")
    
    return prompt_processor, reward_fn
