"""Instruction dataset preparation with padding and response-only loss masking."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import tiktoken
import torch
from torch.utils.data import Dataset

from model.generation import GPT2_EOS_TOKEN_ID


def format_input(entry: Dict[str, Any]) -> str:
    """Format an instruction entry into the prompt format from GPT.ipynb."""
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


def load_instruction_entries(
    primary_path: str | Path,
    extra_paths: Optional[Sequence[str | Path]] = None,
) -> List[Dict[str, Any]]:
    """Load instruction dataset JSON files and optionally merge curated length-aware examples."""
    with open(primary_path, "r", encoding="utf-8") as f:
        entries: List[Dict[str, Any]] = json.load(f)

    if extra_paths:
        for path in extra_paths:
            p = Path(path)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    extra = json.load(f)
                entries.extend(extra)

    return entries


class InstructionDataset(Dataset):
    """Pre-tokenized instruction dataset storing token IDs and prompt boundary lengths."""

    def __init__(self, data: List[Dict[str, Any]], tokenizer: tiktoken.Encoding) -> None:
        self.data = data
        self.encoded_texts: List[List[int]] = []
        self.prompt_lengths: List[int] = []

        for entry in data:
            prompt_text = format_input(entry) + "\n\n### Response:\n"
            prompt_ids = tokenizer.encode(prompt_text)
            response_ids = tokenizer.encode(str(entry["output"]))
            full_ids = prompt_ids + response_ids

            self.encoded_texts.append(full_ids)
            self.prompt_lengths.append(len(prompt_ids))

    def __getitem__(self, index: int) -> Tuple[List[int], int]:
        return self.encoded_texts[index], self.prompt_lengths[index]

    def __len__(self) -> int:
        return len(self.data)


BatchItem = Union[Tuple[List[int], int], List[int]]


def custom_collate_fn(
    batch: Sequence[BatchItem],
    pad_token_id: int = GPT2_EOS_TOKEN_ID,
    ignore_index: int = -100,
    allowed_max_length: Optional[int] = None,
    device: str | torch.device = "cpu",
    mask_prompt: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad batch sequences and mask padding + prompt tokens for response-only loss.

    Supports both `(token_ids, prompt_length)` tuples (from InstructionDataset)
    and plain `token_ids` lists.
    """
    normalized_items: List[Tuple[List[int], int]] = []
    for item in batch:
        if isinstance(item, tuple) and len(item) == 2:
            tokens, prompt_len = item
            normalized_items.append((list(tokens), int(prompt_len)))
        else:
            normalized_items.append((list(item), 0))  # type: ignore[arg-type]

    batch_max_length = max(len(tokens) + 1 for tokens, _ in normalized_items)

    inputs_lst: List[torch.Tensor] = []
    targets_lst: List[torch.Tensor] = []

    for tokens, prompt_len in normalized_items:
        # Append one <|endoftext|> token so the model learns to stop cleanly
        new_item = tokens + [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1], dtype=torch.long)
        targets = torch.tensor(padded[1:], dtype=torch.long)

        # Mask extra padding tokens after the first EOS token
        mask = targets == pad_token_id
        indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Response-only loss masking: ignore prompt tokens (up to "### Response:\n")
        if mask_prompt and prompt_len > 1:
            mask_end = min(prompt_len - 1, targets.numel())
            targets[:mask_end] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

