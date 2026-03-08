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
Preprocess the USPTO-50K dataset to parquet format for verl training.
"""

import argparse
import os
import re

import pandas as pd


SYSTEM_MESSAGE = (
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

USER_TEMPLATE = (
    "Classify this reaction:\n\n"
    "Reactants >> Product:\n{rxn_smiles}\n\n"
    "Analyze the key transformation and output the class number (1-10) in <answer>X</answer> tags."
)


def simplify_smiles(smiles: str) -> str:
    """Remove atom mapping numbers from SMILES for cleaner prompts."""
    # Remove atom mapping like [c:1], [CH3:10] -> [c], [CH3]
    simplified = re.sub(r':(\d+)\]', ']', smiles)
    return simplified


def process_uspto50k(input_path: str, output_path: str, split: str):
    """Process USPTO-50K dataset to verl parquet format."""
    data_source = "uspto50k"
    
    # Load the raw data
    df = pd.read_parquet(input_path)
    
    processed_data = []
    for idx, row in df.iterrows():
        rxn_smiles = simplify_smiles(row["rxn_smiles"])
        reaction_class = row["class"]
        
        user_content = USER_TEMPLATE.format(rxn_smiles=rxn_smiles)
        
        data = {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_content}
            ],
            "ability": "chemistry",
            "reward_model": {"style": "rule", "ground_truth": str(reaction_class)},
            "extra_info": {
                "split": split,
                "index": idx,
                "rxn_smiles": row["rxn_smiles"],
                "prod_smiles": row.get("prod_smiles", ""),
                "reaction_class": reaction_class,
            },
        }
        processed_data.append(data)
    
    # Create DataFrame and save
    output_df = pd.DataFrame(processed_data)
    output_df.to_parquet(output_path)
    print(f"Saved {len(processed_data)} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/capstor/store/cscs/swissai/a143/yl/RandOptimization/data/uspto_50k",
                        help="Directory containing raw USPTO-50K parquet files")
    parser.add_argument("--output_dir", default="/capstor/store/cscs/swissai/a143/yl/RandOptimization/data/uspto_50k_verl",
                        help="Directory to save processed parquet files")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process train and test splits
    for split in ["train", "test"]:
        input_path = os.path.join(args.input_dir, f"{split}.parquet")
        output_path = os.path.join(args.output_dir, f"{split}.parquet")
        
        if os.path.exists(input_path):
            process_uspto50k(input_path, output_path, split)
        else:
            print(f"Warning: {input_path} not found, skipping {split} split")
