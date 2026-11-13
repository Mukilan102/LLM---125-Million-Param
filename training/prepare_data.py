"""Prepare and split instruction fine-tuning data with length diversity."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backend.config import DATA_DIR
from training.dataset import load_instruction_entries


def build_splits(
    primary_path: Path = DATA_DIR / "instruction-data.json",
    length_aware_path: Path = DATA_DIR / "length_aware_instructions.json",
    seed: int = 42,
    train_ratio: float = 0.85,
    val_ratio: float = 0.05,
    length_aware_repeat: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load, shuffle, and split instruction entries into train, val, and test sets."""
    base_entries = load_instruction_entries(primary_path)
    length_entries = load_instruction_entries(length_aware_path) if length_aware_path.exists() else []

    rng = random.Random(seed)
    rng.shuffle(base_entries)
    rng.shuffle(length_entries)

    def _split(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :]

    b_train, b_val, b_test = _split(base_entries)
    l_train, l_val, l_test = _split(length_entries)

    # Slightly upweight length-aware examples in training so GPT-2 124M internalizes
    # both brief and detailed structured stopping patterns cleanly.
    train_data = b_train + (l_train * length_aware_repeat)
    val_data = b_val + l_val
    test_data = b_test + l_test

    rng.shuffle(train_data)
    return train_data, val_data, test_data


if __name__ == "__main__":
    train_data, val_data, test_data = build_splits()
    summary = {
        "train_count": len(train_data),
        "val_count": len(val_data),
        "test_count": len(test_data),
    }
    print(json.dumps(summary, indent=2))

