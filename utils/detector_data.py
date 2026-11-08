"""Safe, reproducible data preparation for the Model X-Ray-style detector.

Modified examples contain only seeded random bytes written into float32 LSBs.
No executable or malware payload is read or generated here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.gf_image import weights_to_gf_image


CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin"}
_WRAPPER_KEYS = ("state_dict", "model_state_dict", "model", "weights")


def list_checkpoints(directory: str | Path) -> list[Path]:
    """Find supported checkpoints recursively in a clean-model directory."""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Clean checkpoint directory not found: {root}")
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in CHECKPOINT_SUFFIXES
    )


def load_state_dict(path: str | Path) -> dict[str, Any]:
    """Load a PyTorch state dict or a common training-checkpoint wrapper."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch
        checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, torch.nn.Module):
        checkpoint = checkpoint.state_dict()
    for key in _WRAPPER_KEYS:
        if isinstance(checkpoint, Mapping) and key in checkpoint:
            candidate = checkpoint[key]
            if isinstance(candidate, torch.nn.Module):
                candidate = candidate.state_dict()
            if isinstance(candidate, Mapping) and any(
                isinstance(v, torch.Tensor) for v in candidate.values()
            ):
                checkpoint = candidate
                break
    if not isinstance(checkpoint, Mapping) or not any(
        isinstance(v, torch.Tensor) for v in checkpoint.values()
    ):
        raise ValueError(f"No tensor state_dict found in checkpoint: {path}")
    return dict(checkpoint)


def _is_parameter_tensor(name: str, value: Any) -> bool:
    # State dicts include BatchNorm running statistics; the reference attack
    # targets model parameters, so omit these non-parameter buffers.
    return (
        isinstance(value, torch.Tensor)
        and value.is_floating_point()
        and value.dtype == torch.float32
        and not name.endswith(("running_mean", "running_var"))
    )


def count_attackable_parameters(state_dict: Mapping[str, Any]) -> int:
    """Count float parameters covered by the float32 LSB payload surrogate."""
    return sum(
        int(value.numel()) for name, value in state_dict.items()
        if _is_parameter_tensor(name, value)
    )


def embed_random_lsb_payload(
    state_dict: Mapping[str, Any],
    *,
    severity: int,
    payload_bytes: int,
    seed: int,
) -> dict[str, Any]:
    """Return a copy with a benign random bitstring tiled into float32 LSBs.

    `severity` is the number of least significant mantissa bits replaced per
    parameter (1..23), as in the paper's X-LSB stress-test. The payload is a
    seeded random byte sequence, repeated if the model has more bit slots than
    payload bits; if it has fewer slots, only the fitting prefix is written.
    This is a benign detector surrogate, not malware.
    """
    if not 1 <= severity <= 23:
        raise ValueError("severity must be between 1 and 23 for float32 mantissas")
    if payload_bytes < 1:
        raise ValueError("payload_bytes must be positive")

    result = dict(state_dict)
    names = [
        name for name, value in state_dict.items()
        if _is_parameter_tensor(name, value)
    ]
    if not names:
        raise ValueError("Checkpoint has no floating-point parameter tensors")

    # Match the paper's float32 weight representation while preserving other
    # values, keys and tensor shapes in the source state dict.
    flat_parts = [state_dict[name].detach().cpu().float().contiguous().numpy().reshape(-1)
                  for name in names]
    lengths = [part.size for part in flat_parts]
    flat = np.concatenate(flat_parts).astype(np.float32, copy=False)
    if not np.isfinite(flat).all():
        raise ValueError("Checkpoint parameters contain NaN/Inf; refusing LSB rewrite")

    payload = np.random.default_rng(seed).bytes(payload_bytes)
    payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    raw = flat.view(np.uint32)
    mask = np.uint32((1 << severity) - 1)
    chunk_size = 1_000_000
    for start in range(0, flat.size, chunk_size):
        end = min(start + chunk_size, flat.size)
        indices = np.arange(start, end, dtype=np.int64)
        low_bits = np.zeros(end - start, dtype=np.uint32)
        for bit_index in range(severity):
            # Stream one random payload sequence through consecutive X-LSB
            # slots, matching the paper's payload-fill procedure.
            positions = (indices * severity + bit_index) % payload_bits.size
            low_bits |= payload_bits[positions].astype(np.uint32) << (severity - bit_index - 1)
        raw[start:end] = (raw[start:end] & ~mask) | low_bits

    offset = 0
    for name, length in zip(names, lengths):
        original = state_dict[name]
        chunk = raw[offset:offset + length].view(np.float32).copy()
        restored = torch.from_numpy(chunk.reshape(tuple(original.shape)))
        result[name] = restored.to(dtype=original.dtype)
        offset += length
    return result


def state_dict_to_gf(state_dict: Mapping[str, Any], *, imsize: int) -> np.ndarray:
    """Convert floating state-dict tensors to the paper's resized GF image."""
    tensors = [
        value.detach().cpu().float().reshape(-1)
        for name, value in state_dict.items()
        if isinstance(value, torch.Tensor)
        and value.is_floating_point()
        and not name.endswith(("running_mean", "running_var"))
    ]
    if not tensors:
        raise ValueError("Checkpoint has no floating-point tensors for GF conversion")
    return weights_to_gf_image(torch.cat(tensors), imsize=imsize)
