"""Model X-Ray-style weight image representations."""

from __future__ import annotations

import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class FloatImageStats:
    """Normalisation statistics for a float32 weight image.

    Produced by :func:`weights_to_float_image` and consumed by
    :func:`float_image_to_weights` to invert the normalisation *exactly*.

    Attributes:
        mean: Mean subtracted from the raw weights before scaling.
        scale: Positive divisor applied after centring (max absolute
            centred weight), so the normalised image lies in ``[-1, 1]``.
        num_values: Number of real (unpadded) weight values.
        side: Side length of the square spatial image.
    """

    mean: float
    scale: float
    num_values: int
    side: int


def weights_to_float_image(
    weights: torch.Tensor | np.ndarray,
) -> tuple[np.ndarray, FloatImageStats]:
    """Flatten and normalise float32 weights into a 1-channel spatial image.

    Unlike :func:`weights_to_channels` (which byte-decomposes each float into
    four IEEE754 planes), this keeps the weights in continuous float32 space:
    one normalised value per weight parameter, laid out on a square grid.
    Normalisation is a symmetric mean-centred max-abs affine::

        image = (weights - mean) / scale        with  scale = max|weights - mean|

    which maps the real weights exactly into ``[-1, 1]`` and is exactly
    invertible via :func:`float_image_to_weights` (no precision loss, no
    dangerous byte planes to zero out). Padding cells added to square the
    grid are filled with ``0.0`` (the normalised mean) and are dropped again
    on inversion using ``stats.num_values``.

    This is the representation the encoder/decoder operate in. The byte
    representation from :func:`weights_to_channels` is kept separate and is
    only used for the SRNet-style detector forward pass.

    Args:
        weights: Tensor or array of float-compatible weight values.

    Returns:
        Tuple ``(image, stats)`` where ``image`` is a ``float32`` array with
        shape ``(1, side, side)`` in ``[-1, 1]`` and ``stats`` carries the
        ``mean``/``scale``/``num_values``/``side`` needed to invert it.
    """

    values = _to_float32_numpy(weights).reshape(-1)
    num_values = int(values.size)
    side = _square_side(num_values)
    padded_size = side * side

    if num_values == 0:
        image = np.zeros((1, side, side), dtype=np.float32)
        return image, FloatImageStats(mean=0.0, scale=1.0, num_values=0, side=side)

    mean = float(values.mean())
    centred = values - mean
    scale = float(np.abs(centred).max())
    if scale < 1e-12:
        # All weights identical: avoid divide-by-zero; the (zero) residual is
        # representable and inversion still returns the constant weights.
        scale = 1.0

    normalised = centred / scale
    if padded_size != num_values:
        normalised = np.pad(
            normalised, (0, padded_size - num_values), constant_values=0.0
        )

    image = normalised.reshape(1, side, side).astype(np.float32, copy=False)
    return image, FloatImageStats(
        mean=mean, scale=scale, num_values=num_values, side=side
    )


def float_image_to_weights(
    image: torch.Tensor | np.ndarray,
    *,
    mean: float = 0.0,
    scale: float = 1.0,
    num_values: int | None = None,
) -> torch.Tensor:
    """Invert :func:`weights_to_float_image` back to flat float32 weights.

    The inverse is the affine ``weights = image * scale + mean`` applied to
    the flattened image. When ``image`` is a ``torch.Tensor`` this is fully
    differentiable and preserves the autograd graph, so the classification
    objective can push gradients through weight reconstruction straight to
    the encoder — no straight-through estimator needed (the byte-packing
    path is the only non-differentiable one, and it is used solely for the
    detector).

    Args:
        image: ``(1, side, side)`` / ``(side, side)`` / flat float image, as a
            torch tensor (keeps grad) or numpy array.
        mean: ``FloatImageStats.mean`` used during normalisation.
        scale: ``FloatImageStats.scale`` used during normalisation.
        num_values: Optional number of real, unpadded weights to return
            (drops the square-padding cells).

    Returns:
        One-dimensional ``torch.float32`` tensor of weight values.
    """

    if isinstance(image, torch.Tensor):
        flat = image.reshape(-1)
        if num_values is not None:
            flat = flat[:num_values]
        return flat.to(dtype=torch.float32) * scale + mean

    flat_np = np.asarray(image, dtype=np.float32).reshape(-1)
    if num_values is not None:
        flat_np = flat_np[:num_values]
    weights = flat_np * np.float32(scale) + np.float32(mean)
    return torch.from_numpy(weights.astype(np.float32, copy=False))


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
