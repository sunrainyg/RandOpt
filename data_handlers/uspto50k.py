"""USPTO-50K dataset handler for chemical reaction classification."""
import re
from typing import Dict, List, Optional

from utils.reward_score import uspto50k as uspto50k_reward

from .base import DatasetHandler

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

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


class USPTO50KHandler(DatasetHandler):
    """Handler for USPTO-50K chemical reaction classification."""

    name = "uspto50k"
    default_train_path = "data/uspto_50k/train.parquet"
    default_test_path = "data/uspto_50k/test.parquet"
    default_max_tokens = 64  # Allow brief reasoning before answer

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def load_data(
        self,
        path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        start_index: int = 0,
    ) -> List[Dict]:
        """Load USPTO-50K data from local parquet or HuggingFace.
        
        Dataset has ~49k train and ~1k validation/test samples.
        Task: Classify reaction into 1 of 10 classes.
        """
        from datasets import load_dataset
        
        # Load from local parquet or HuggingFace
        if path.endswith('.parquet'):
            ds = load_dataset('parquet', data_files=path, split='train')
        else:
            # HuggingFace path
            hf_split = "validation" if split == "test" else "train"
            ds = load_dataset(path, split=hf_split)
        
        total_available = len(ds)
        total_needed = min(total_available, max_samples) if max_samples else total_available
        
        task_datas = []
        for idx in range(total_needed):
            row = ds[idx]
            
            # Simplify reaction SMILES for prompt (remove atom mapping numbers)
            rxn_smiles = self._simplify_smiles(row["rxn_smiles"])
            prod_smiles = row["prod_smiles"]
            reaction_class = row["class"]
            
            task_datas.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": USER_TEMPLATE.format(rxn_smiles=rxn_smiles)}
                ],
                "ground_truth": str(reaction_class),
                "reaction_class": reaction_class,
            })
        
        return task_datas

    # -------------------------------------------------------------------------
    # Reward and extraction
    # -------------------------------------------------------------------------

    def _simplify_smiles(self, smiles: str) -> str:
        """Remove atom mapping numbers from SMILES for cleaner prompts."""
        # Remove atom mapping like [c:1], [CH3:10] -> [c], [CH3]
        simplified = re.sub(r':(\d+)\]', ']', smiles)
        return simplified

    def compute_reward(self, response: str, ground_truth: str) -> float:
        """Compute reward: 1.0 if class matches, 0.0 otherwise."""
        answer = self.extract_answer(response)
        return uspto50k_reward.compute_score(answer, ground_truth)

    def extract_answer(self, response: str) -> str:
        """Extract class number from <answer>...</answer> tags."""
        return uspto50k_reward.extract_answer(response)

    def extract_answer_for_voting(self, response: str) -> str:
        """Extract class for voting."""
        return self.extract_answer(response)

    def is_answer_correct(self, response: str, ground_truth: str) -> bool:
        """Check if answer is correct."""
        return self.compute_reward(response, ground_truth) == 1.0

    def is_voted_answer_correct(self, voted_answer: str, ground_truth: str) -> bool:
        """Check if voted answer matches ground truth."""
        if not voted_answer:
            return False
        try:
            return int(voted_answer) == int(ground_truth)
        except (ValueError, TypeError):
            return False

    def format_answer_for_check(self, answer: str) -> str:
        """Format answer for checking."""
        return f"<answer>{answer}</answer>"
