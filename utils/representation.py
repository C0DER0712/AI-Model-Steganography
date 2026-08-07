"""Model X-Ray-style weight image representations."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ChannelArray = np.ndarray


def weights_to_channels(weights: torch.Tensor | np.ndarray) -> ChannelArray:
    """Convert float32 weights to four IEEE754 grayscale channels.

    Each float32 value is interpreted as its 32-bit IEEE754 bit pattern and
    split into four unsigned bytes:
    `p0 = bits[31:24]`, `p1 = bits[23:16]`, `p2 = bits[15:8]`,
    `p3 = bits[7:0]`.

    Args:
        weights: Tensor or array of float-compatible weight values.

    Returns:
        A `uint8` array with shape `(4, side, side)`, where each channel is
        padded with zeros after the final real weight if needed.
    """

    values = _to_float32_numpy(weights).reshape(-1)
    side = _square_side(values.size)
    padded_size = side * side

    bits = values.view(np.uint32)
    if padded_size != bits.size:
        bits = np.pad(bits, (0, padded_size - bits.size), constant_values=0)

    channels = np.empty((4, padded_size), dtype=np.uint8)
    channels[0] = ((bits >> 24) & 0xFF).astype(np.uint8)
    channels[1] = ((bits >> 16) & 0xFF).astype(np.uint8)
    channels[2] = ((bits >> 8) & 0xFF).astype(np.uint8)
    channels[3] = (bits & 0xFF).astype(np.uint8)

    return channels.reshape(4, side, side)


def channels_to_weights(
    channels: ChannelArray,
    *,
    num_values: int | None = None,
) -> torch.Tensor:
    """Reconstruct float32 weights from four IEEE754 grayscale channels.

    Args:
        channels: `uint8`-compatible array with shape `(4, height, width)`.
        num_values: Optional number of real, unpadded values to return.

    Returns:
        One-dimensional CPU `torch.float32` tensor.

    Raises:
        ValueError: If channel shape or `num_values` is invalid.
    """

    channel_array = np.asarray(channels)
    if channel_array.ndim != 3 or channel_array.shape[0] != 4:
        raise ValueError("Expected channels with shape (4, height, width).")

    flat = channel_array.astype(np.uint8, copy=False).reshape(4, -1).astype(np.uint32)
    bits = (
        (flat[0] << 24)
        | (flat[1] << 16)
        | (flat[2] << 8)
        | flat[3]
    )

    if num_values is not None:
        if num_values < 0:
            raise ValueError("num_values must be non-negative.")
        if num_values > bits.size:
            raise ValueError(
                f"num_values={num_values} exceeds available values {bits.size}."
            )
        bits = bits[:num_values]

    values = bits.astype(np.uint32, copy=False).view(np.float32).copy()
    return torch.from_numpy(values)


def visualize_channels(
    channels: ChannelArray,
    *,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Visualize four grayscale byte channels side by side.

    Args:
        channels: `uint8`-compatible array with shape `(4, height, width)`.
        output_path: Optional path for saving the rendered visualization.
        title: Optional figure title.

    Returns:
        The Matplotlib figure.
    """

    channel_array = np.asarray(channels)
    if channel_array.ndim != 3 or channel_array.shape[0] != 4:
        raise ValueError("Expected channels with shape (4, height, width).")

    figure, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    if title:
        figure.suptitle(title)

    for index, axis in enumerate(axes):
        axis.imshow(channel_array[index].astype(np.uint8), cmap="gray", vmin=0, vmax=255)
        axis.set_title(f"p{index}")
        axis.axis("off")

    if output_path is not None:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200)

    return figure


def _to_float32_numpy(weights: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(weights, torch.Tensor):
        return weights.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)

    return np.asarray(weights, dtype=np.float32)


def _square_side(num_values: int) -> int:
    if num_values == 0:
        return 0

    return math.ceil(math.sqrt(num_values))
