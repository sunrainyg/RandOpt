
"""
Main entry point for Evolution Strategy (ES) training.

This module provides a Hydra-based entry point for ES fine-tuning,
similar to main_ppo.py but for zeroth-order optimization.
"""

import json
import os
import socket
import tempfile
import time

import hydra
import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from vllm import TokensPrompt

from verl.trainer.es.ray_trainer import RayESTrainer
from verl.utils.device import auto_set_device


# Default prompt templates (can be overridden via config)
DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)

DEFAULT_USER_TEMPLATE_COUNTDOWN = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)

DEFAULT_RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


def create_countdown_prompt_processor(system_message=None, user_template=None, response_prompt=None):
    """Create a prompt processor for the countdown task."""
    system_msg = system_message or DEFAULT_SYSTEM_MESSAGE
    user_tmpl = user_template or DEFAULT_USER_TEMPLATE_COUNTDOWN
    resp_prompt = response_prompt or DEFAULT_RESPONSE_PROMPT
    
    def process_context(task_data, tokenizer):
        numbers = task_data["numbers"]
        target = task_data["target"]
        user_content = user_tmpl.format(numbers=numbers, target=target)
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content}
        ]
        
        formatted = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        prompts = tokenizer(formatted)
        return TokensPrompt(prompt_token_ids=prompts['input_ids'])
    
    return process_context


def create_countdown_reward_fn(response_prompt=None):
    """Create a reward function for the countdown task."""
    resp_prompt = response_prompt or DEFAULT_RESPONSE_PROMPT
    
    # Import the reward function from the task module
    # This can be customized based on task
    def reward_fn(response, task_data):
        """Compute reward for countdown task."""
        try:
            from countdown.countdown_task import reward_function
            full_response = resp_prompt + response
            return reward_function(full_response, task_data["numbers"], task_data["target"])
        except ImportError:
            # Fallback: simple reward based on answer extraction
            import re
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if answer_match:
                try:
                    expr = answer_match.group(1).strip()
                    result = eval(expr)
                    target = task_data["target"]
                    if abs(result - target) < 1e-6:
                        return {"reward": 1.0, "reward_info": {"format_reward": 1.0, "answer_reward": 1.0}}
                    return {"reward": 0.1, "reward_info": {"format_reward": 1.0, "answer_reward": 0.0}}
                except Exception:
                    return {"reward": 0.0, "reward_info": {"format_reward": 0.0, "answer_reward": 0.0}}
            return {"reward": 0.0, "reward_info": {"format_reward": 0.0, "answer_reward": 0.0}}
    
    return reward_fn


@hydra.main(config_path="config", config_name="es_trainer", version_base=None)
def main(config):
    """Main entry point for ES training with Hydra configuration management."""
    auto_set_device(config)
    run_es(config)


def run_es(config) -> None:
    """Initialize Ray cluster and run ES training process."""
    from pprint import pprint
    
    # Print configuration
    print(f"ES Training - hostname: {socket.gethostname()}, PID: {os.getpid()}")
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    
    # Initialize Ray
    if not ray.is_initialized():
        # Clean Ray environment
        os.environ.pop("RAY_ADDRESS", None)
        os.environ.pop("RAY_HEAD_IP", None)
        os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
        
        unique_dir = tempfile.mkdtemp(prefix=f"ray_es_session_{int(time.time())}_")
        
        ray.init(
            address="local",
            include_dashboard=False,
            ignore_reinit_error=True,
            _temp_dir=unique_dir,
            dashboard_port=None
        )
    
    # Load tokenizer
    model_path = config.model.path
    trust_remote_code = config.model.get("trust_remote_code", False)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    
    # For vLLM, we can use the model name directly without downloading
    # vLLM will handle model loading internally
    base_model_path = model_path
    
    # Load data
    train_data = []
    eval_data = []
    
    if hasattr(config.data, 'train_files') and config.data.train_files:
        train_data = load_data(config.data.train_files)
        if hasattr(config.data, 'train_max_samples') and config.data.train_max_samples > 0:
            train_data = train_data[:config.data.train_max_samples]
    
    if hasattr(config.data, 'val_files') and config.data.val_files:
        eval_data = load_data(config.data.val_files)
        if hasattr(config.data, 'val_max_samples') and config.data.val_max_samples > 0:
            eval_data = eval_data[:config.data.val_max_samples]
    
    print(f"Loaded {len(train_data)} training samples, {len(eval_data)} evaluation samples")
    
    # Create task-specific components
    task_type = config.data.get("task_type", "countdown")
    
    # Built-in task types: countdown, gsm8k, math, math500, olympiadbench, uspto50k, common_gen, mbpp, rocstories
    if task_type in ["countdown", "gsm8k", "math", "math500", "olympiadbench", "uspto50k", "common_gen", "mbpp", "rocstories"]:
        from verl.trainer.es.task_utils import get_task_components
        task_config = OmegaConf.to_container(config.data, resolve=True)
        prompt_processor, reward_fn = get_task_components(task_type, task_config)
    elif task_type == "custom":
        # Custom task: load from config
        prompt_processor = None
        reward_fn = None
        reward_fn_path = config.data.get('reward_fn_path')
        reward_fn_name = config.data.get('reward_fn_name')
        if reward_fn_path and reward_fn_name:
            from verl.utils.import_utils import load_extern_object
            reward_fn = load_extern_object(reward_fn_path, reward_fn_name)
        
        prompt_processor_path = config.data.get('prompt_processor_path')
        prompt_processor_name = config.data.get('prompt_processor_name')
        if prompt_processor_path and prompt_processor_name:
            from verl.utils.import_utils import load_extern_object
            prompt_processor = load_extern_object(prompt_processor_path, prompt_processor_name)
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Use 'countdown', 'gsm8k', 'math', 'math500', 'olympiadbench', 'uspto50k', 'common_gen', 'mbpp', 'rocstories', or 'custom'")
    
    # Pre-process prompts for evaluation data
    if prompt_processor and eval_data:
        for d in eval_data:
            d["context"] = prompt_processor(d, tokenizer)
    
    # Initialize trainer
    trainer = RayESTrainer(
        config=config,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        val_reward_fn=reward_fn,
        train_data=train_data,
        eval_data=eval_data,
        prompt_processor=prompt_processor,
    )
    
    # Initialize workers
    trainer.init_workers(base_model_path)
    
    # Start training
    trainer.fit()
    
    ray.shutdown()


def load_data(file_path):
    """Load data from JSON or Parquet file."""
    if isinstance(file_path, list):
        file_path = file_path[0]
    
    if file_path.endswith('.parquet'):
        import pandas as pd
        df = pd.read_parquet(file_path)
        return df.to_dict('records')
    elif file_path.endswith('.json'):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    else:
        # Try JSON first, then parquet
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception:
            import pandas as pd
            return pd.read_parquet(file_path).to_dict('records')


if __name__ == "__main__":
    main()
