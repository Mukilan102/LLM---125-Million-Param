"""Checkpoint saving, loading, and OpenAI GPT-2 TensorFlow weight conversion."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
import torch

from model.gpt import GPT_PRETRAINED_CONFIG_124M, GPTModel


def assign(left: torch.Tensor, right: np.ndarray) -> torch.nn.Parameter:
    """Validate tensor shape and convert a NumPy weight array into an nn.Parameter."""
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(
        torch.tensor(right, dtype=left.dtype, device=left.device)
    )


def load_weights_into_gpt(gpt: GPTModel, params: Dict[str, Any]) -> None:
    """Load OpenAI TensorFlow GPT-2 parameter dictionary into GPTModel (from GPT.ipynb)."""
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        block = gpt.trf_blocks[b]

        q_w, k_w, v_w = np.split(
            params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1
        )
        block.att.W_query.weight = assign(block.att.W_query.weight, q_w.T)
        block.att.W_key.weight = assign(block.att.W_key.weight, k_w.T)
        block.att.W_value.weight = assign(block.att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1
        )
        block.att.W_query.bias = assign(block.att.W_query.bias, q_b)
        block.att.W_key.bias = assign(block.att.W_key.bias, k_b)
        block.att.W_value.bias = assign(block.att.W_value.bias, v_b)

        block.att.out_projection.weight = assign(
            block.att.out_projection.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        block.att.out_projection.bias = assign(
            block.att.out_projection.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        block.ff.layers[0].weight = assign(
            block.ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        block.ff.layers[0].bias = assign(
            block.ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"],
        )
        block.ff.layers[2].weight = assign(
            block.ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        block.ff.layers[2].bias = assign(
            block.ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        block.norm1.scale = assign(
            block.norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        block.norm1.shift = assign(
            block.norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        block.norm2.scale = assign(
            block.norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        block.norm2.shift = assign(
            block.norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def _download_file(url: str, destination: str) -> None:
    """Download a file from OpenAI public blob storage if not already cached locally."""
    from tqdm import tqdm

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    file_size = int(response.headers.get("content-length", 0))

    if os.path.exists(destination) and os.path.getsize(destination) == file_size and file_size > 0:
        return

    desc = url.split("/")[-1]
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    progress_bar.update(len(chunk))
                    f.write(chunk)


def _load_gpt2_params_from_tf_ckpt(ckpt_path: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Read GPT-2 TensorFlow checkpoint variables into nested NumPy dict (lazy TF import)."""
    import tensorflow as tf

    params: Dict[str, Any] = {"blocks": [{} for _ in range(settings["n_layer"])]}
    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]  # Skip 'model/' prefix

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        target_dict[variable_name_parts[-1]] = variable_array

    return params


def download_and_load_gpt2_tf(
    model_size: str = "124M",
    models_dir: str = "gpt2",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load GPT-2 TF checkpoint from local dir or download if missing."""
    import tensorflow as tf

    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size must be one of {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    for filename in filenames:
        file_path = os.path.join(model_dir, filename)
        if not os.path.exists(file_path):
            file_url = f"{base_url}/{model_size}/{filename}"
            _download_file(file_url, file_path)

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    if not tf_ckpt_path:
        raise FileNotFoundError(f"No TensorFlow checkpoint found in {model_dir}")

    with open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8") as f:
        settings = json.load(f)

    params = _load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    return settings, params


def save_checkpoint(
    model: GPTModel,
    path: str | Path,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Path:
    """Save GPTModel weights, config, and optional metadata/optimizer state to disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": config or getattr(model, "cfg", GPT_PRETRAINED_CONFIG_124M),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, out_path)
    return out_path


def load_checkpoint(
    path: str | Path,
    device: torch.device,
) -> Tuple[GPTModel, Dict[str, Any], Dict[str, Any]]:
    """Load GPTModel, its config, and metadata from a PyTorch .pth checkpoint."""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", dict(GPT_PRETRAINED_CONFIG_124M))
    metadata = checkpoint.get("metadata", {})

    model = GPTModel(config)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config, metadata


def load_or_convert_pretrained_124m(
    pretrained_pth_path: str | Path,
    tf_models_dir: str | Path,
    device: torch.device,
) -> Tuple[GPTModel, Dict[str, Any], Dict[str, Any]]:
    """Load pretrained 124M PyTorch checkpoint or convert from OpenAI TF checkpoint."""
    pth_path = Path(pretrained_pth_path)
    if pth_path.exists():
        return load_checkpoint(pth_path, device)

    _, params = download_and_load_gpt2_tf(
        model_size="124M",
        models_dir=str(tf_models_dir),
    )
    config = dict(GPT_PRETRAINED_CONFIG_124M)
    model = GPTModel(config)
    load_weights_into_gpt(model, params)
    model.to(device)
    model.eval()

    metadata = {
        "model_variant": "pretrained",
        "model_size": "124M",
        "instruction_tuned": False,
    }
    save_checkpoint(model, pth_path, config=config, metadata=metadata)
    return model, config, metadata

