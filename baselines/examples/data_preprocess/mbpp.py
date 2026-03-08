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
Preprocess the MBPP dataset to parquet format for verl training.
"""

import argparse
import os

import pandas as pd
from datasets import load_from_disk, load_dataset


SYSTEM_MESSAGE = (
    "You are a Python programming assistant. Write clean, correct Python code to solve the given problem."
)

USER_TEMPLATE = (
    "{text}\n\n"
    "Your code should pass these tests:\n{tests}\n\n"
    "Think through your solution in <think> </think> tags.\n"
    "Return your final Python code in <answer> </answer> tags, e.g.:\n"
    "<answer>\ndef solution(x):\n    return x + 1\n</answer>"
)


def process_mbpp(input_path: str, output_dir: str, use_sanitized: bool = True):
    """Process MBPP dataset to verl parquet format."""
    data_source = "mbpp"
    
    # Load the dataset
    if os.path.isdir(input_path):
        # Load from local disk
        ds = load_from_disk(input_path)
    else:
        # Load from HuggingFace
        subset = "sanitized" if use_sanitized else "full"
        ds = load_dataset("google-research-datasets/mbpp", subset)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process train split (combine train + validation)
    train_data = []
    for split_name in ["train", "validation"]:
        if split_name not in ds:
            continue
        for idx, item in enumerate(ds[split_name]):
            text = item.get("prompt", item.get("text", ""))
            test_list = list(item["test_list"])
            test_imports = item.get("test_imports", "")
            code = item.get("code", "")
            task_id = item.get("task_id", idx)
            
            # Format tests for display (show first 3)
            tests_str = "\n".join(test_list[:3])
            
            user_content = USER_TEMPLATE.format(text=text, tests=tests_str)
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_content}
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "test_list": test_list,
                        "test_setup_code": test_imports,
                        "code": code,
                    }
                },
                "extra_info": {
                    "split": "train",
                    "index": len(train_data),
                    "task_id": task_id,
                    "text": text,
                },
            }
            train_data.append(data)
    
    # Process test split
    test_data = []
    if "test" in ds:
        for idx, item in enumerate(ds["test"]):
            text = item.get("prompt", item.get("text", ""))
            test_list = list(item["test_list"])
            test_imports = item.get("test_imports", "")
            code = item.get("code", "")
            task_id = item.get("task_id", idx)
            
            # Format tests for display (show first 3)
            tests_str = "\n".join(test_list[:3])
            
            user_content = USER_TEMPLATE.format(text=text, tests=tests_str)
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_content}
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "test_list": test_list,
                        "test_setup_code": test_imports,
                        "code": code,
                    }
                },
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "task_id": task_id,
                    "text": text,
                },
            }
            test_data.append(data)
    
    # Save to parquet
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(output_dir, "train.parquet"))
    print(f"Saved {len(train_data)} training samples")
    
    test_df = pd.DataFrame(test_data)
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"))
    print(f"Saved {len(test_data)} test samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="/capstor/store/cscs/swissai/a143/yl/RandOptimization/data/mbpp_sanitized",
                        help="Path to MBPP dataset (local or HuggingFace)")
    parser.add_argument("--output_dir", default="/capstor/store/cscs/swissai/a143/yl/RandOptimization/data/mbpp_verl",
                        help="Directory to save processed parquet files")
    parser.add_argument("--use_sanitized", action="store_true", default=True,
                        help="Use sanitized subset (default: True)")
    
    args = parser.parse_args()
    
    process_mbpp(args.input_path, args.output_dir, args.use_sanitized)
