# How to Use RandOpt with Your Own Dataset

Only **3 steps**: prepare data, write handler + reward, run.

## Step 1: Prepare Data

Put a JSON file in `data/your_dataset/data.json`:

```json
[
  {"question": "What is 2 + 2?", "answer": "4"},
  {"question": "Capital of France?", "answer": "Paris"}
]
```

---

## Step 2: Write Handler + Reward

### 2a. Reward function — `utils/reward_score/your_dataset.py`

```python
"""Reward scoring for your dataset."""
import re


def extract_answer(response: str) -> str:
    """Pull the answer out of the model's response."""
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return m.group(1).strip() if m else response.strip().split("\n")[-1].strip()


def compute_score(response: str, ground_truth: str) -> float:
    """Return 1.0 if correct, 0.0 otherwise."""
    ans = extract_answer(response)
    return 1.0 if ans.strip().lower() == str(ground_truth).strip().lower() else 0.0
```

### 2b. Data handler — `data_handlers/your_dataset.py`

```python
"""Your dataset handler."""
import json
from typing import Dict, List, Optional
from utils.reward_score import your_dataset as your_reward
from .base import DatasetHandler


class YourDatasetHandler(DatasetHandler):
    name = "your_dataset"
    default_train_path = "data/your_dataset/data.json"
    default_test_path = "data/your_dataset/data.json"
    default_max_tokens = 512

    def load_data(self, path, split="train", max_samples=None) -> List[Dict]:
        with open(path) as f:
            raw = json.load(f)
        out = []
        for item in raw:
            out.append({
                "messages": [{"role": "user", "content": item["question"]}],
                "ground_truth": item["answer"],
            })
            if max_samples and len(out) >= max_samples:
                break
        return out
```

### 2c. Register — in `data_handlers/__init__.py` add:

```python
from .your_dataset import YourDatasetHandler          # add this

DATASET_HANDLERS = {
    # ... existing entries ...
    "your_dataset": YourDatasetHandler,                # add this
}
```

---

## Step 3: Run

```bash
python3 randopt.py \
  --dataset your_dataset \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --population_size 500 \
  --sigma_values "0.0005,0.001,0.002" \
  --num_engines 4 \
```

Done!