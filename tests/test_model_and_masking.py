"""Tests for GPTModel architecture, generation, checkpointing, and response-only loss masking."""

from pathlib import Path

import torch

from model.checkpoint import load_checkpoint, save_checkpoint
from model.generation import (
    GPT2_EOS_TOKEN_ID,
    generate,
    generate_with_metadata,
    get_tokenizer,
)
from model.gpt import GPTModel
from training.dataset import InstructionDataset, custom_collate_fn


TINY_CFG = {
    "vocab_size": 50257,
    "context_length": 64,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


def test_gpt_model_forward_shape() -> None:
    model = GPTModel(TINY_CFG)
    model.eval()
    x = torch.randint(0, 100, (2, 10))
    logits = model(x)
    assert logits.shape == (2, 10, TINY_CFG["vocab_size"])


def test_response_only_loss_masking_and_padding() -> None:
    tokenizer = get_tokenizer()
    entries = [
        {
            "instruction": "What is 2+2?",
            "input": "",
            "output": "4.",
        },
        {
            "instruction": "Explain briefly what Python is.",
            "input": "",
            "output": "Python is a high-level programming language.",
        },
    ]
    dataset = InstructionDataset(entries, tokenizer)
    batch = [dataset[0], dataset[1]]
    inputs, targets = custom_collate_fn(batch, pad_token_id=GPT2_EOS_TOKEN_ID, mask_prompt=True)

    assert inputs.shape == targets.shape
    for i, (tokens, prompt_len) in enumerate(batch):
        # Prompt tokens before response must be masked with -100
        assert torch.all(targets[i, : prompt_len - 1] == -100)
        # First response token (at targets[i, prompt_len - 1]) must NOT be masked
        assert targets[i, prompt_len - 1].item() == tokens[prompt_len]
        # First EOS token at end of response must NOT be masked
        eos_pos = len(tokens) - 1
        assert targets[i, eos_pos].item() == GPT2_EOS_TOKEN_ID
        # Any padding tokens after first EOS must be masked with -100
        if eos_pos + 1 < targets.size(1):
            assert torch.all(targets[i, eos_pos + 1 :] == -100)


def test_generate_stops_on_eos_early() -> None:
    model = GPTModel(TINY_CFG)
    model.eval()
    # Force out_head bias/weights so token 50256 is always argmax
    with torch.no_grad():
        model.out_head.weight.zero_()
        # Set norm/emb to predictable non-zero values and bias via weight row
        model.out_head.weight[GPT2_EOS_TOKEN_ID].fill_(10.0)
        model.final_norm.scale.fill_(1.0)
        model.final_norm.shift.fill_(1.0)

    idx = torch.tensor([[10, 20, 30]], dtype=torch.long)
    result = generate_with_metadata(
        model=model,
        idx=idx,
        max_new_tokens=20,
        context_size=TINY_CFG["context_length"],
        temperature=0.0,
        top_k=None,
        eos_id=GPT2_EOS_TOKEN_ID,
    )
    assert result.stopped_reason == "eos"
    assert result.tokens_generated == 0


def test_generate_topk_and_temperature() -> None:
    torch.manual_seed(42)
    model = GPTModel(TINY_CFG)
    model.eval()
    idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = generate(
        model=model,
        idx=idx,
        max_new_tokens=5,
        context_size=TINY_CFG["context_length"],
        temperature=0.8,
        top_k=10,
        eos_id=None,
    )
    assert out.shape == (1, 8)


def test_checkpoint_save_and_load_roundtrip(tmp_path: Path) -> None:
    model = GPTModel(TINY_CFG)
    ckpt_path = tmp_path / "test_ckpt.pth"
    save_checkpoint(
        model,
        ckpt_path,
        config=TINY_CFG,
        metadata={"model_variant": "instruction-finetuned"},
    )

    loaded_model, loaded_cfg, metadata = load_checkpoint(ckpt_path, torch.device("cpu"))
    assert loaded_cfg["emb_dim"] == TINY_CFG["emb_dim"]
    assert metadata["model_variant"] == "instruction-finetuned"

    x = torch.tensor([[5, 6, 7]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), loaded_model(x))

