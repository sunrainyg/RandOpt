"""Preprocess olympiadbench dataset to parquet format for verl training"""

import os
import pandas as pd

DATA_SOURCE = "olympiadbench"
INSTRUCTION = "Let's think step by step and output the final answer after ####"

def process_data(df, indices, split):
    processed = []
    for idx in indices:
        row = df.iloc[idx]
        # Handle final_answer as list/array
        answer = row['final_answer']
        if hasattr(answer, 'tolist'):  # numpy array
            answer = answer.tolist()
        if isinstance(answer, list):
            answer = answer[0] if len(answer) == 1 else '|'.join(str(a) for a in answer)
        # Clean latex formatting from answer
        answer = str(answer).strip().replace('$', '')
        
        processed.append({
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": row['question'] + " " + INSTRUCTION}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "id": row.get('id', ''),
                "answer_type": row.get('answer_type', ''),
                "difficulty": row.get('difficulty', ''),
            },
        })
    return processed


if __name__ == "__main__":
    input_path = '/capstor/store/cscs/swissai/a143/yl/RandOptimization/data/olympiadbench/OE_TO_maths_en_COMP.parquet'
    save_dir = '/capstor/store/cscs/swissai/a143/yl/RandOptimization/data/olympiadbench'
    train_size = 200

    df = pd.read_parquet(input_path)
    total = len(df)
    
    train_data = process_data(df, range(train_size), "train")
    test_data = process_data(df, range(train_size, total), "test")

    pd.DataFrame(train_data).to_parquet(os.path.join(save_dir, "train.parquet"))
    pd.DataFrame(test_data).to_parquet(os.path.join(save_dir, "test.parquet"))
    print(f"Saved {len(train_data)} train, {len(test_data)} test to {save_dir}")
