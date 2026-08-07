"""Utilities for extracting and restoring PyTorch model weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class WeightTensor:
    """A floating-point tensor extracted from a model state dictionary.

    Attributes:
        name: State-dictionary key for the tensor.
        shape: Original tensor shape.
        dtype: Original tensor dtype.
        values: Tensor values stored as detached CPU `float32`.
    """

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    values: torch.Tensor


def extract_weights(model: nn.Module) -> list[WeightTensor]:
    """Extract floating-point tensors from a PyTorch model in exact state order.

    Args:
        model: PyTorch module to inspect.

    Returns:
        Ordered metadata and float32 values for each floating-point parameter or
        persistent buffer in `model.state_dict()`.
    """

    extracted: list[WeightTensor] = []
    for name, tensor in model.state_dict().items():
        if not tensor.is_floating_point():
            continue

        extracted.append(
            WeightTensor(
                name=name,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                values=tensor.detach().cpu().to(dtype=torch.float32).clone(),
            )
        )

    return extracted


def flatten_weights(weights: Sequence[WeightTensor]) -> torch.Tensor:
    """Flatten extracted weights into one contiguous float32 vector.

    Args:
        weights: Ordered weight tensor records.

    Returns:
        One-dimensional CPU `float32` tensor containing all values in order.
    """

    if not weights:
        return torch.empty(0, dtype=torch.float32)

    return torch.cat([item.values.reshape(-1).to(dtype=torch.float32) for item in weights])


def restore_weights(
    flattened: torch.Tensor,
    reference_weights: Sequence[WeightTensor],
) -> list[WeightTensor]:
    """Restore structured weight records from a flat vector and metadata.

    Args:
        flattened: One-dimensional tensor containing replacement values.
        reference_weights: Ordered metadata returned by `extract_weights()`.

    Returns:
        Weight records with original names, shapes, and dtypes, but updated
        float32 values.

    Raises:
        ValueError: If `flattened` does not contain exactly the expected number
            of elements.
    """

    flat = flattened.detach().cpu().to(dtype=torch.float32).reshape(-1)
    expected = sum(item.values.numel() for item in reference_weights)
    if flat.numel() != expected:
        raise ValueError(
            f"Flattened tensor has {flat.numel()} values, expected {expected}."
        )

    restored: list[WeightTensor] = []
    offset = 0
    for item in reference_weights:
        size = item.values.numel()
        values = flat[offset : offset + size].reshape(item.shape).clone()
        restored.append(
            WeightTensor(
                name=item.name,
                shape=item.shape,
                dtype=item.dtype,
                values=values,
            )
        )
        offset += size

    return restored


def load_modified_weights(
    model: nn.Module,
    modified_weights: Sequence[WeightTensor],
    *,
    strict: bool = True,
) -> nn.Module:
    """Load modified floating-point tensors into a model.

    Non-floating state such as BatchNorm counters is preserved from the target
    model. The input ordering is respected, and tensor names/shapes are
    validated before loading.

    Args:
        model: Model to update in place.
        modified_weights: Ordered replacement records, typically produced by
            `restore_weights()`.
        strict: Passed to `nn.Module.load_state_dict()`.

    Returns:
        The updated model.

    Raises:
        KeyError: If a modified tensor name is absent from the target model.
        ValueError: If a modified tensor shape does not match the target state.
        TypeError: If a target state entry is not floating-point.
    """

    state = model.state_dict()
    for item in modified_weights:
        if item.name not in state:
            raise KeyError(f"Model has no state entry named '{item.name}'.")

        target = state[item.name]
        if not target.is_floating_point():
            raise TypeError(f"State entry '{item.name}' is not floating-point.")
        if tuple(target.shape) != item.shape:
            raise ValueError(
                f"Shape mismatch for '{item.name}': got {item.shape}, "
                f"expected {tuple(target.shape)}."
            )

        state[item.name] = item.values.to(device=target.device, dtype=target.dtype)

    model.load_state_dict(state, strict=strict)
    return model
