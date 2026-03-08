# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-500 dataset to parquet format
"""

import argparse
import json
import os

import pandas as pd

DATA_SOURCE = "HuggingFaceH4/MATH-500"
INSTRUCTION = "Let's think step by step and output the final answer after ####"


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def process_data(data, split):
    processed = []
    for idx, item in enumerate(data):
        processed.append({
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": item["problem"] + " " + INSTRUCTION}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": item["answer"]},
            "extra_info": {
                "split": split,
                "index": idx,
                "subject": item.get("subject", ""),
                "level": item.get("level", ""),
                "unique_id": item.get("unique_id", ""),
            },
        })
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Path to test.jsonl")
    parser.add_argument("--local_save_dir", default="~/data/math500", help="Save directory")
    parser.add_argument("--train_size", type=int, default=200, help="Number of samples for training")
    args = parser.parse_args()

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    data = load_jsonl(args.input_path)
    
    # Split: first 200 for train, remaining 300 for test
    train_data = data[:args.train_size]
    test_data = data[args.train_size:]
    
    train_processed = process_data(train_data, "train")
    test_processed = process_data(test_data, "test")

    # Save as parquet
    pd.DataFrame(train_processed).to_parquet(os.path.join(local_save_dir, "train.parquet"))
    pd.DataFrame(test_processed).to_parquet(os.path.join(local_save_dir, "test.parquet"))
    print(f"Saved {len(train_processed)} train, {len(test_processed)} test to {local_save_dir}")
