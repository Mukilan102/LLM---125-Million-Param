"""Instruction fine-tuning script for custom GPT-2 124M."""

import argparse
import time
from functools import partial
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from backend.config import (
    DATA_DIR,
    INSTRUCT_CHECKPOINT_PATH,
    PRETRAINED_CHECKPOINT_PATH,
    TF_MODELS_DIR,
)
from model.checkpoint import load_or_convert_pretrained_124m, save_checkpoint
from model.generation import (
    GPT2_EOS_TOKEN_ID,
    clean_generated_response,
    generate_with_metadata,
    get_tokenizer,
    text_to_token_ids,
    token_ids_to_text,
)
from model.gpt import get_device
from training.dataset import InstructionDataset, custom_collate_fn, format_input
from training.prepare_data import build_splits


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Compute masked cross-entropy loss on response tokens only."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(),
        ignore_index=-100,
    )
    return loss


def calc_loss_loader(
    data_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """Compute mean loss over a DataLoader (or first `num_batches`)."""
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    batches_to_eval = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= batches_to_eval:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += float(loss.item())
    return total_loss / batches_to_eval


def evaluate_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int,
) -> Tuple[float, float]:
    """Evaluate training and validation loss without tracking gradients."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    if was_training:
        model.train()
    return train_loss, val_loss


def generate_sample_response(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    entry: dict,
    max_new_tokens: int = 96,
) -> str:
    """Generate a deterministic sample response during training inspection."""
    was_training = model.training
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    prompt = format_input(entry) + "\n\n### Response:\n"
    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    with torch.no_grad():
        result = generate_with_metadata(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temperature=0.0,
            top_k=None,
            eos_id=GPT2_EOS_TOKEN_ID,
            repetition_penalty=1.18,
        )
    generated = result.token_ids[:, encoded.size(1) :]
    text = clean_generated_response(token_ids_to_text(generated, tokenizer))
    if was_training:
        model.train()
    return text


def train_instruction_model(
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum_steps: int = 8,
    lr: float = 5e-5,
    weight_decay: float = 0.1,
    allowed_max_length: int = 512,
    eval_freq: int = 30,
    eval_iter: int = 15,
    output_checkpoint: Path = INSTRUCT_CHECKPOINT_PATH,
) -> dict:
    """Fine-tune pretrained GPT-2 124M on instruction data with response-only loss masking."""
    torch.manual_seed(123)
    device = get_device(require_gpu=True)
    torch.cuda.manual_seed_all(123)
    print(f"Using GPU device: {device} ({torch.cuda.get_device_name(0)})")

    model, config, _ = load_or_convert_pretrained_124m(
        pretrained_pth_path=PRETRAINED_CHECKPOINT_PATH,
        tf_models_dir=TF_MODELS_DIR,
        device=device,
    )
    tokenizer = get_tokenizer()

    train_data, val_data, test_data = build_splits(
        primary_path=DATA_DIR / "instruction-data.json",
        length_aware_path=DATA_DIR / "length_aware_instructions.json",
    )
    print(
        f"Dataset splits -> train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}"
    )

    collate_fn = partial(
        custom_collate_fn,
        pad_token_id=GPT2_EOS_TOKEN_ID,
        ignore_index=-100,
        allowed_max_length=allowed_max_length,
        device="cpu",
        mask_prompt=True,
    )

    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_opt_steps = max(1, (len(train_loader) // grad_accum_steps) * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_opt_steps, eta_min=1e-5
    )

    initial_train_loss, initial_val_loss = evaluate_model(
        model, train_loader, val_loader, device, eval_iter=eval_iter
    )
    print(
        f"Initial -> Train loss: {initial_train_loss:.4f}, Val loss: {initial_val_loss:.4f}"
    )

    best_val_loss = float("inf")
    train_losses: List[float] = []
    val_losses: List[float] = []
    global_step = 0
    start_time = time.time()

    sample_short = {"instruction": "Explain briefly what an API is.", "input": ""}
    sample_detail = {"instruction": "Explain step by step how a transformer language model generates text.", "input": ""}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step, (input_batch, target_batch) in enumerate(train_loader):
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            is_accum_boundary = (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader)
            if is_accum_boundary:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % eval_freq == 0 or global_step == 1:
                    t_loss, v_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter=eval_iter
                    )
                    train_losses.append(t_loss)
                    val_losses.append(v_loss)
                    print(
                        f"Epoch {epoch + 1}/{epochs} (OptStep {global_step:04d}): "
                        f"Train loss {t_loss:.4f}, Val loss {v_loss:.4f}"
                    )
                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        save_checkpoint(
                            model,
                            output_checkpoint,
                            config=config,
                            metadata={
                                "model_variant": "instruction-finetuned",
                                "model_size": "124M",
                                "instruction_tuned": True,
                                "epoch": epoch + 1,
                                "global_step": global_step,
                                "val_loss": round(v_loss, 4),
                                "train_loss": round(t_loss, 4),
                            },
                        )

        # End-of-epoch evaluation & sample print
        t_loss, v_loss = evaluate_model(
            model, train_loader, val_loader, device, eval_iter=min(20, len(val_loader))
        )
        print(
            f"End of Epoch {epoch + 1}/{epochs} -> Train loss {t_loss:.4f}, Val loss {v_loss:.4f}"
        )
        if v_loss <= best_val_loss:
            best_val_loss = v_loss
            save_checkpoint(
                model,
                output_checkpoint,
                config=config,
                metadata={
                    "model_variant": "instruction-finetuned",
                    "model_size": "124M",
                    "instruction_tuned": True,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "val_loss": round(v_loss, 4),
                    "train_loss": round(t_loss, 4),
                },
            )

        print("Sample (brief):", generate_sample_response(model, tokenizer, device, sample_short, max_new_tokens=80))
        print("Sample (detail):", generate_sample_response(model, tokenizer, device, sample_detail, max_new_tokens=160))

    # Ensure checkpoint exists even if eval_freq didn't trigger improvement check
    if not Path(output_checkpoint).exists():
        save_checkpoint(
            model,
            output_checkpoint,
            config=config,
            metadata={
                "model_variant": "instruction-finetuned",
                "model_size": "124M",
                "instruction_tuned": True,
                "epoch": epochs,
                "global_step": global_step,
                "val_loss": round(best_val_loss, 4),
            },
        )

    elapsed_sec = time.time() - start_time
    print(f"Instruction fine-tuning completed in {elapsed_sec:.1f}s. Best val loss: {best_val_loss:.4f}")
    return {
        "initial_train_loss": initial_train_loss,
        "initial_val_loss": initial_val_loss,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1] if train_losses else initial_train_loss,
        "elapsed_seconds": round(elapsed_sec, 2),
        "checkpoint": str(output_checkpoint),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Instruction fine-tune GPT-2 124M")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--eval-freq", type=int, default=30)
    parser.add_argument("--eval-iter", type=int, default=15)
    parser.add_argument("--output", type=Path, default=INSTRUCT_CHECKPOINT_PATH)
    args = parser.parse_args()

    train_instruction_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        allowed_max_length=args.max_length,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        output_checkpoint=args.output,
    )


if __name__ == "__main__":
    main()

