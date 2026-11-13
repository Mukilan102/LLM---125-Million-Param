"""Application configuration for model checkpoints, paths, and defaults."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

PRETRAINED_CHECKPOINT_PATH = Path(
    os.getenv(
        "GPT2_PRETRAINED_CHECKPOINT",
        str(CHECKPOINTS_DIR / "gpt2_pretrained_124m.pth"),
    )
)
INSTRUCT_CHECKPOINT_PATH = Path(
    os.getenv(
        "GPT2_INSTRUCT_CHECKPOINT",
        str(CHECKPOINTS_DIR / "gpt2_instruct_124m.pth"),
    )
)

# Default TensorFlow GPT-2 directory (prefers local workspace gpt2 folder)
_LOCAL_TF_DIR = PROJECT_ROOT / "gpt2"
_LEGACY_TF_DIR = Path(r"D:\LLM\LLM Material\LLM\gpt2")
TF_MODELS_DIR = Path(
    os.getenv(
        "GPT2_TF_MODELS_DIR",
        str(_LOCAL_TF_DIR if _LOCAL_TF_DIR.exists() else _LEGACY_TF_DIR),
    )
)

# Allowed CORS origins for frontend
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 40
DEFAULT_MAX_TOKENS_CEILING = 256

