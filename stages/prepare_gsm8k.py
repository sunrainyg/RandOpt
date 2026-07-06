#!/usr/bin/env python3
"""
Download GSM8K from the HF hub and write it to the parquet format expected by
the RandOpt GSM8KHandler (data/gsm8k/train.parquet, data/gsm8k/test.parquet).

Mirrors verl's examples/data_preprocess/gsm8k.py but without importing verl,
so it runs inside the stock vLLM image.
"""

import argparse
import os
import re

import datasets


def extract_solution(solution_str: str) -> str:
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None, f"no #### answer in: {solution_str[-100:]}"
    return solution.group(0).split("#### ")[1].replace(",", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="data/gsm8k")
    args = parser.parse_args()

    data_source = "openai/gsm8k"
    dataset = datasets.load_dataset(data_source, "main")

    instruction = 'Let\'s think step by step and output the final answer after "####".'

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question") + " " + instruction
            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            return {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
        return process_fn

    os.makedirs(args.local_save_dir, exist_ok=True)
    train = dataset["train"].map(make_map_fn("train"), with_indices=True)
    test = dataset["test"].map(make_map_fn("test"), with_indices=True)
    train.to_parquet(os.path.join(args.local_save_dir, "train.parquet"))
    test.to_parquet(os.path.join(args.local_save_dir, "test.parquet"))
    print(f"[prepare_gsm8k] wrote {len(train)} train / {len(test)} test rows to {args.local_save_dir}")


if __name__ == "__main__":
    main()
