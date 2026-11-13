"""Thread-safe GPT-2 inference engine separated from API routing."""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from backend.token_budget import select_token_budget
from model.checkpoint import load_checkpoint, load_or_convert_pretrained_124m
from model.generation import (
    GPT2_EOS_TOKEN_ID,
    clean_generated_response,
    format_chat_prompt,
    generate_with_metadata,
    get_tokenizer,
    text_to_token_ids,
    token_ids_to_text,
)
from model.gpt import GPTModel, get_device


@dataclass
class InferenceOutput:
    """Result returned by GPT2InferenceEngine.generate_reply."""

    response: str
    model_variant: str
    device: str
    tokens_generated: int
    token_budget: int
    length_category: str
    stopped_reason: str
    generation_time_ms: float


class GPT2InferenceEngine:
    """Manages lifecycle and thread-safe autoregressive generation for GPTModel."""

    def __init__(
        self,
        instruct_checkpoint_path: Path,
        pretrained_checkpoint_path: Path,
        tf_models_dir: Path,
        device: Optional[torch.device] = None,
    ) -> None:
        self.instruct_checkpoint_path = Path(instruct_checkpoint_path)
        self.pretrained_checkpoint_path = Path(pretrained_checkpoint_path)
        self.tf_models_dir = Path(tf_models_dir)
        self.device = device or get_device(require_gpu=True)
        self.tokenizer = get_tokenizer()
        self._lock = threading.Lock()

        self.model: Optional[GPTModel] = None
        self.config: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.model_variant: str = "unloaded"
        self.active_checkpoint_path: str = ""
        self.load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def context_length(self) -> int:
        return int(self.config.get("context_length", 1024))

    def load(self) -> None:
        """Load the instruction-tuned checkpoint if available, else pretrained 124M."""
        with self._lock:
            self.load_error = None
            try:
                if self.instruct_checkpoint_path.exists():
                    model, config, metadata = load_checkpoint(
                        self.instruct_checkpoint_path, self.device
                    )
                    self.model = model
                    self.config = config
                    self.metadata = metadata
                    self.model_variant = metadata.get(
                        "model_variant", "instruction-finetuned"
                    )
                    self.active_checkpoint_path = str(self.instruct_checkpoint_path)
                else:
                    model, config, metadata = load_or_convert_pretrained_124m(
                        pretrained_pth_path=self.pretrained_checkpoint_path,
                        tf_models_dir=self.tf_models_dir,
                        device=self.device,
                    )
                    self.model = model
                    self.config = config
                    self.metadata = metadata
                    self.model_variant = metadata.get("model_variant", "pretrained")
                    self.active_checkpoint_path = str(self.pretrained_checkpoint_path)

                self.model.eval()
            except Exception as exc:
                self.model = None
                self.model_variant = "error"
                self.load_error = str(exc)
                raise

    def generate_reply(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        max_tokens: int = 256,
        auto_length: bool = True,
    ) -> InferenceOutput:
        """Generate an autoregressive reply safely under a lock."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError(
                self.load_error or "Model is not loaded yet."
            )

        use_instruction_format = self.model_variant != "pretrained"
        formatted_prompt = format_chat_prompt(
            user_message=message,
            history=history,
            use_instruction_format=use_instruction_format,
        )

        encoded_prompt = text_to_token_ids(formatted_prompt, self.tokenizer)
        # Keep within model context window
        max_prompt_tokens = max(16, self.context_length - 64)
        if encoded_prompt.size(1) > max_prompt_tokens:
            encoded_prompt = encoded_prompt[:, -max_prompt_tokens:]

        prompt_token_count = encoded_prompt.size(1)
        budget_decision = select_token_budget(
            prompt=message,
            user_max_tokens=max_tokens,
            auto_length=auto_length,
            context_length=self.context_length,
            prompt_token_count=prompt_token_count,
        )

        stop_sequences = (
            ["### Instruction:", "### Input:", "### Response:"]
            if use_instruction_format
            else ["\nUser:", "\nAssistant:"]
        )

        start_time = time.perf_counter()
        with self._lock:
            self.model.eval()
            idx = encoded_prompt.to(self.device)
            result = generate_with_metadata(
                model=self.model,
                idx=idx,
                max_new_tokens=budget_decision.token_budget,
                context_size=self.context_length,
                temperature=temperature,
                top_k=top_k,
                eos_id=GPT2_EOS_TOKEN_ID,
                stop_sequences=stop_sequences,
                tokenizer=self.tokenizer,
                repetition_penalty=1.18,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        generated_tokens = result.token_ids[:, prompt_token_count:]
        raw_text = token_ids_to_text(generated_tokens, self.tokenizer)
        cleaned_text = clean_generated_response(raw_text)

        if not cleaned_text:
            cleaned_text = "I am sorry, I could not generate a complete response for that prompt."

        return InferenceOutput(
            response=cleaned_text,
            model_variant=self.model_variant,
            device=str(self.device),
            tokens_generated=result.tokens_generated,
            token_budget=budget_decision.token_budget,
            length_category=budget_decision.length_category,
            stopped_reason=result.stopped_reason,
            generation_time_ms=round(elapsed_ms, 2),
        )

