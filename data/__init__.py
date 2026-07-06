"""Dataset handlers (slim registry).

Only the datasets needed by the released iterative-RandOpt recipe are wired in
here. Add a handler by implementing ``DatasetHandler`` and registering it below.
"""
from .base import DatasetHandler
from .gsm8k import GSM8KHandler

DATASET_HANDLERS = {
    "gsm8k": GSM8KHandler,
}


def get_dataset_handler(name: str) -> DatasetHandler:
    if name not in DATASET_HANDLERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_HANDLERS)}")
    return DATASET_HANDLERS[name]()


def list_datasets() -> list:
    return list(DATASET_HANDLERS)
