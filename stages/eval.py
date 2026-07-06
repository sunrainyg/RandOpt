#!/usr/bin/env python3
"""
Evaluate a single model checkpoint on the test set.

Used to verify that the distilled (single) model retains the accuracy of the
top-k output ensemble. Correctness logic mirrors `evaluate_base_model` in
`randopt.py` exactly so the numbers are directly comparable to:
  - base_test_accuracy
  - ensemble_results[K] (majority voting)
from a RandOpt run's results.json.
"""

import argparse
import json
import os
import sys

# Allow running from the distillation/ dir or the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

from ..data import get_dataset_handler, list_datasets


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a single model on the test set")
    p.add_argument("--model", type=str, required=True,
                   help="Model path or HF id to evaluate (e.g. distilled checkpoint dir)")
    p.add_argument("--dataset", type=str, default="gsm8k", choices=list_datasets())
    p.add_argument("--test_data_path", type=str, default=None,
                   help="Override handler default test path")
    p.add_argument("--train_data_path", type=str, default=None,
                   help="Only used when train/test share a file (e.g. MATH500)")
    p.add_argument("--train_samples", type=int, default=200,
                   help="Used only for same-file split datasets")
    p.add_argument("--test_samples", type=int, default=None,
                   help="Max test samples (None = all)")
    p.add_argument("--precision", type=str, default="bfloat16",
                   choices=["float16", "bfloat16"])
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--label", type=str, default="model",
                   help="Label used in the printed/saved summary")
    p.add_argument("--output_json", type=str, default=None,
                   help="Optional path to write the accuracy json")
    return p.parse_args()


def load_test_data(handler, args):
    train_path = args.train_data_path or handler.default_train_path
    test_path = args.test_data_path or handler.default_test_path
    if train_path == test_path:
        all_data = handler.load_data(train_path, split="train", max_samples=None)
        if args.test_samples is None:
            test_datas = all_data[args.train_samples:]
        else:
            test_datas = all_data[args.train_samples:args.train_samples + args.test_samples]
        if len(test_datas) < 50:
            test_datas = all_data
    else:
        test_datas = handler.load_data(test_path, split="test", max_samples=args.test_samples)
    return test_datas


def extract_vote_answer(handler, response_text, data):
    """Extract the (voting) answer string from a response, '' if none/invalid."""
    if handler.name == "countdown":
        numbers = data.get("numbers")
        answer, is_valid, _ = handler.extract_answer_for_voting(response_text, numbers=numbers)
        return answer if is_valid else ""
    elif hasattr(handler, "extract_answer_for_voting"):
        return handler.extract_answer_for_voting(response_text) or ""
    return handler.extract_answer(response_text) or ""


def check_answer(handler, answer, data):
    """Check an already-extracted answer string against the ground truth."""
    if not answer:
        return False
    if hasattr(handler, "is_voted_answer_correct"):
        return bool(handler.is_voted_answer_correct(answer, data["ground_truth"]))
    formatted = handler.format_answer_for_check(answer)
    return bool(handler.is_answer_correct(formatted, data["ground_truth"]))


def is_correct(handler, response_text, data):
    """Same correctness logic as randopt.evaluate_base_model (test path)."""
    return check_answer(handler, extract_vote_answer(handler, response_text, data), data)


def main():
    args = parse_args()
    handler = get_dataset_handler(args.dataset)
    max_tokens = args.max_tokens or handler.default_max_tokens

    print("=" * 60)
    print(f"SINGLE-MODEL EVAL [{args.label}]")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset} | max_tokens: {max_tokens} | tp: {args.tp}")

    test_datas = load_test_data(handler, args)
    print(f"Test samples: {len(test_datas)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    is_instruct = any(x in args.model.lower() for x in ["instruct", "chat", "it"]) \
        or tokenizer.chat_template is not None

    def format_prompt(messages):
        if is_instruct and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return "\n".join(m["content"] for m in messages) + "\n"

    test_prompts = [format_prompt(d["messages"]) for d in test_datas]

    llm = LLM(
        model=args.model,
        dtype=args.precision,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(temperature=0.0, seed=args.global_seed, max_tokens=max_tokens)
    outputs = llm.generate(test_prompts, sampling_params, use_tqdm=True)

    correct = 0
    for output, data in zip(outputs, test_datas):
        if is_correct(handler, output.outputs[0].text, data):
            correct += 1
    acc = correct / len(test_datas) if test_datas else 0.0
    print(f"\n[{args.label}] Test accuracy: {acc*100:.2f}% ({correct}/{len(test_datas)})")

    result = {
        "label": args.label,
        "model": args.model,
        "dataset": args.dataset,
        "test_samples": len(test_datas),
        "correct": correct,
        "accuracy": acc,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
