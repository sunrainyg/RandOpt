# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    # Check if tokenizer has a chat template (instruct models have one, base models don't)
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        logger.info("Tokenizer does not have a chat_template (likely a base model), skipping system prompt initialization")
        return []
    
    try:
        token1 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
        )
        token2 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
        )
        # get system prompt tokens
        system_prompt = token1[: -(len(token2) - len(token1))]
        return system_prompt
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to initialize system prompt: {e}. Returning empty list.")
        return []


def extract_system_prompt_and_generation(tokenizer):
    """
    Extract system prompt and generation prompt tokens from a tokenizer's chat template.
    
    Returns:
        Tuple of (system_prompt, generate_prompt) token lists, or ([], []) if no chat template
    """
    # Check if tokenizer has a chat template (instruct models have one, base models don't)
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        logger.info("Tokenizer does not have a chat_template (likely a base model), returning empty prompts")
        return [], []
    
    try:
        token1 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
        )
        token2 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
        )
        # get system prompt tokens
        system_prompt = token1[: -(len(token2) - len(token1))]
        # get generate prompt tokens
        token3 = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
        generate_prompt = token3[len(token1) :]

        return system_prompt, generate_prompt
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to extract system/generation prompts: {e}. Returning empty lists.")
        return [], []
