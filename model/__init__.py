"""Custom GPT-2 model architecture, text generation, and checkpoint utilities."""

import os
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _scripts_dir = Path(sys.prefix) / "Scripts"
    if _scripts_dir.is_dir():
        os.add_dll_directory(str(_scripts_dir))

from model.gpt import (
    GELU,
    GPT_CONFIG_124M,
    GPT_PRETRAINED_CONFIG_124M,
    MODEL_CONFIGS,
    FeedForward,
    GPTModel,
    MultiHeadAttention,
    Normalize,
    TransformerBlock,
    get_device,
)
from model.generation import (
    GPT2_EOS_TOKEN_ID,
    clean_generated_response,
    format_chat_prompt,
    generate,
    generate_with_metadata,
    get_tokenizer,
    text_to_token_ids,
    token_ids_to_text,
)
from model.checkpoint import (
    assign,
    load_checkpoint,
    load_or_convert_pretrained_124m,
    load_weights_into_gpt,
    save_checkpoint,
)

__all__ = [
    "GELU",
    "GPT_CONFIG_124M",
    "GPT_PRETRAINED_CONFIG_124M",
    "MODEL_CONFIGS",
    "FeedForward",
    "GPTModel",
    "MultiHeadAttention",
    "Normalize",
    "TransformerBlock",
    "get_device",
    "GPT2_EOS_TOKEN_ID",
    "clean_generated_response",
    "format_chat_prompt",
    "generate",
    "generate_with_metadata",
    "get_tokenizer",
    "text_to_token_ids",
    "token_ids_to_text",
    "assign",
    "load_checkpoint",
    "load_or_convert_pretrained_124m",
    "load_weights_into_gpt",
    "save_checkpoint",
]

