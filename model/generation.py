"""Autoregressive text generation, tokenization, and prompt formatting utilities."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence

import tiktoken
import torch

GPT2_EOS_TOKEN_ID = 50256
_LOCAL_TIKTOKEN_CACHE = Path(__file__).resolve().parent.parent / ".tiktoken_cache"
if _LOCAL_TIKTOKEN_CACHE.is_dir():
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_LOCAL_TIKTOKEN_CACHE))


@lru_cache(maxsize=1)
def get_tokenizer() -> tiktoken.Encoding:
    """Return cached GPT-2 tiktoken tokenizer."""
    return tiktoken.get_encoding("gpt2")


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    """Encode text into a (1, seq_len) token tensor, allowing <|endoftext|>."""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    """Decode a (1, seq_len) or (seq_len,) token tensor back into text."""
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


@dataclass
class GenerationResult:
    """Structured result from autoregressive token generation."""

    token_ids: torch.Tensor
    tokens_generated: int
    stopped_reason: str  # "eos" | "max_tokens" | "stop_sequence"


def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = GPT2_EOS_TOKEN_ID,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """Generate tokens autoregressively with Top-K filtering, temperature, and EOS stopping."""
    result = generate_with_metadata(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
        repetition_penalty=repetition_penalty,
    )
    return result.token_ids


def generate_with_metadata(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = GPT2_EOS_TOKEN_ID,
    stop_sequences: Optional[Sequence[str]] = None,
    tokenizer: Optional[tiktoken.Encoding] = None,
    repetition_penalty: float = 1.0,
) -> GenerationResult:
    """Autoregressively generate tokens and track stopping reason and count."""
    if max_new_tokens <= 0:
        return GenerationResult(
            token_ids=idx,
            tokens_generated=0,
            stopped_reason="max_tokens",
        )

    prompt_len = idx.size(1)
    stopped_reason = "max_tokens"

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if repetition_penalty is not None and repetition_penalty > 1.0 and idx.size(1) > prompt_len:
            generated_window = idx[:, max(prompt_len, idx.size(1) - 64) :]
            for b in range(logits.size(0)):
                seen_tokens = torch.unique(generated_window[b])
                scores = logits[b, seen_tokens]
                logits[b, seen_tokens] = torch.where(
                    scores < 0,
                    scores * repetition_penalty,
                    scores / repetition_penalty,
                )

        if top_k is not None and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            top_logits, _ = torch.topk(logits, k, dim=-1)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype),
                logits,
            )

        if temperature > 0.0:
            scaled_logits = logits / float(temperature)
            probs = torch.softmax(scaled_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            stopped_reason = "eos"
            break

        idx = torch.cat((idx, idx_next), dim=1)

        if stop_sequences and tokenizer is not None:
            generated_only = idx[:, prompt_len:]
            decoded_so_far = token_ids_to_text(generated_only, tokenizer)
            if any(seq in decoded_so_far for seq in stop_sequences):
                stopped_reason = "stop_sequence"
                break

    tokens_generated = idx.size(1) - prompt_len
    return GenerationResult(
        token_ids=idx,
        tokens_generated=tokens_generated,
        stopped_reason=stopped_reason,
    )


def format_chat_prompt(
    user_message: str,
    history: Optional[List[dict]] = None,
    use_instruction_format: bool = True,
) -> str:
    """Format user input (and optional prior turns) into an Alpaca-style instruction prompt."""
    clean_message = user_message.strip()
    if not use_instruction_format:
        if not history:
            return clean_message
        lines = []
        for turn in history[-4:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {turn.get('content', '').strip()}")
        lines.append(f"User: {clean_message}")
        lines.append("Assistant:")
        return "\n".join(lines)

    history_context = ""
    if history:
        recent_turns = history[-4:]
        formatted_turns = []
        for turn in recent_turns:
            role_label = "User" if turn.get("role") == "user" else "Assistant"
            content = str(turn.get("content", "")).strip()
            if content:
                formatted_turns.append(f"{role_label}: {content}")
        if formatted_turns:
            history_context = "\n\n### Input:\nPrevious conversation:\n" + "\n".join(
                formatted_turns
            )

    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{clean_message}"
        f"{history_context}"
        "\n\n### Response:\n"
    )


def clean_generated_response(text: str) -> str:
    """Strip end-of-text markers and any accidental section spillover."""
    cleaned = text.replace("<|endoftext|>", "")
    for stop_marker in ("### Instruction:", "### Input:", "### Response:"):
        if stop_marker in cleaned:
            cleaned = cleaned.split(stop_marker, 1)[0]
    return cleaned.strip()

