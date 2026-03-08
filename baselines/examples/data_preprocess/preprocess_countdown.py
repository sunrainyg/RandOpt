"""Preprocess countdown.json to parquet format for verl training."""

import json
import os
import argparse
import pandas as pd

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)

USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)


def make_prompt(item, template_type='qwen-instruct'):
    """Create prompt messages from data item."""
    numbers = item['numbers']
    target = item['target']
    del template_type  # Kept for backward compatibility with existing CLI flags.
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_TEMPLATE.format(numbers=numbers, target=target)},
    ]


def process_data(input_path, output_dir, train_size=None, test_size=1024, template_type='qwen-instruct'):
    """Process countdown.json into train/test parquet files."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    total = len(data)
    if train_size is None:
        train_size = total - test_size
    
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    
    def convert_to_verl_format(items, split):
        records = []
        for idx, item in enumerate(items):
            record = {
                'data_source': 'countdown',
                'prompt': make_prompt(item, template_type),
                'ability': 'math',
                'reward_model': {
                    'style': 'rule',
                    'ground_truth': {
                        'target': str(item['target']),
                        'numbers': item['numbers']
                    }
                },
                'extra_info': {
                    'split': split,
                    'index': idx,
                    'solution': item.get('solution', '')
                }
            }
            records.append(record)
        return records
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_records = convert_to_verl_format(train_data, 'train')
    test_records = convert_to_verl_format(test_data, 'test')
    
    pd.DataFrame(train_records).to_parquet(os.path.join(output_dir, 'train.parquet'))
    pd.DataFrame(test_records).to_parquet(os.path.join(output_dir, 'test.parquet'))
    
    print(f"Saved {len(train_records)} train samples and {len(test_records)} test samples to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/capstor/store/cscs/swissai/a143/yl/RandOptimization/data/countdown/countdown.json')
    parser.add_argument('--output_dir', default='/capstor/store/cscs/swissai/a143/yl/RandOptimization/data/countdown')
    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--template_type', default='qwen-instruct')
    args = parser.parse_args()
    
    process_data(args.input, args.output_dir, args.train_size, args.test_size, args.template_type)
