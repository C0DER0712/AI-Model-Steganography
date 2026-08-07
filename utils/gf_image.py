"""Grayscale-Fourpart (GF) image utilities for Model X-Ray detection.

The Model X-Ray paper represents AI model weights as **Grayscale-Fourpart
(GF)** images: the four IEEE754 byte channels produced by
:func:`~utils.representation.weights_to_channels` are tiled into a single
``(2S × 2S)`` grayscale image in a 2×2 quad layout::

    ┌─────┬─────┐
    │ p0  │ p1  │   p0 = bits[31:24]  (sign + exponent MSB)
    ├─────┼─────┤   p1 = bits[23:16]
    │ p2  │ p3  │   p2 = bits[15:8]
    └─────┴─────┘   p3 = bits[7:0]    (mantissa LSByte)

This single-channel format is the direct input to SRNet and the OSL CNN
in the paper.  The standalone SRNet detector (``models/srnet_detector.py``)
expects ``(B, 1, H, W)`` tensors produced by this module.

Functions
---------
channels_to_gf_image(channels)
    ``(4, S, S)`` uint8 → ``(2S, 2S)`` uint8 grayscale GF image.

gf_image_to_channels(gf_image)
    ``(2S, 2S)`` uint8 GF image → ``(4, S, S)`` uint8 channels (inverse).

resize_gf_image(gf_image, size)
    Resize a GF image to ``(size, size)`` pixels (nearest-neighbour to
    preserve byte values).

weights_to_gf_image(weights, imsize)
    Convenience: flat weight tensor → GF image of ``(imsize, imsize)``
    pixels, ready for SRNet.

gf_image_to_tensor(gf_image)
    ``(H, W)`` uint8 numpy → ``(1, H, W)`` float32 torch tensor in
    ``[0, 255]`` (SRNet input format).
"""

from __future__ import annotations

from typing import Union

import cv2
import numpy as np
import torch

from utils.representation import weights_to_channels


# ---------------------------------------------------------------------------
# Core GF conversions
# ---------------------------------------------------------------------------


def channels_to_gf_image(channels: np.ndarray) -> np.ndarray:
    """Convert four byte channels to a single Grayscale-Fourpart image.

    The four ``(S, S)`` channel planes are tiled into a ``(2S, 2S)``
    single-channel image using the canonical 2×2 quad layout described in
    the Model X-Ray paper.

    Args:
        channels: ``uint8``-compatible array with shape ``(4, S, S)``.

    Returns:
        ``uint8`` array with shape ``(2S, 2S)``.

    Raises:
        ValueError: If ``channels`` does not have shape ``(4, S, S)`` with
            equal height and width.
    """
    channels = np.asarray(channels, dtype=np.uint8)
    if channels.ndim != 3 or channels.shape[0] != 4:
        raise ValueError(
            f"Expected channels with shape (4, S, S), got {channels.shape}."
        )
    _, h, w = channels.shape
    if h != w:
        raise ValueError(
            f"Channel spatial dimensions must be square, got ({h}, {w})."
        )
    s = h
    gf = np.empty((2 * s, 2 * s), dtype=np.uint8)
    gf[:s, :s] = channels[0]   # p0 — top-left
    gf[:s, s:] = channels[1]   # p1 — top-right
    gf[s:, :s] = channels[2]   # p2 — bottom-left
    gf[s:, s:] = channels[3]   # p3 — bottom-right
    return gf


def gf_image_to_channels(gf_image: np.ndarray) -> np.ndarray:
    """Invert :func:`channels_to_gf_image` — recover the four byte channels.

    Args:
        gf_image: ``uint8``-compatible array with shape ``(2S, 2S)``.

    Returns:
        ``uint8`` array with shape ``(4, S, S)``.

    Raises:
        ValueError: If ``gf_image`` is not a 2-D square array of even side.
    """
    gf_image = np.asarray(gf_image, dtype=np.uint8)
    if gf_image.ndim != 2:
        raise ValueError(
            f"Expected 2-D GF image, got {gf_image.ndim}-D array."
        )
    h, w = gf_image.shape
    if h != w:
        raise ValueError(
            f"GF image must be square, got ({h}, {w})."
        )
    if h % 2 != 0:
        raise ValueError(
            f"GF image side must be even (got {h}); it should be 2× the "
            "channel side S."
        )
    s = h // 2
    channels = np.empty((4, s, s), dtype=np.uint8)
    channels[0] = gf_image[:s, :s]   # p0
    channels[1] = gf_image[:s, s:]   # p1
    channels[2] = gf_image[s:, :s]   # p2
    channels[3] = gf_image[s:, s:]   # p3
    return channels


# ---------------------------------------------------------------------------
# Resize helpers
# ---------------------------------------------------------------------------


def resize_gf_image(gf_image: np.ndarray, size: int) -> np.ndarray:
    """Resize a GF image to ``(size, size)`` using nearest-neighbour interpolation.

    Nearest-neighbour is preferred over bilinear because the GF image encodes
    raw byte values — sub-pixel blending would corrupt the bit-level statistics
    that steganalysis features rely on.

    Args:
        gf_image: ``uint8``-compatible 2-D array.
        size: Target side length in pixels (square output).

    Returns:
        Resized ``uint8`` array with shape ``(size, size)``.
    """
    gf_image = np.asarray(gf_image, dtype=np.uint8)
    if gf_image.shape == (size, size):
        return gf_image
    resized = cv2.resize(
        gf_image, (size, size), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(np.uint8)


# ---------------------------------------------------------------------------
# Convenience: flat weights → ready-to-use GF tensor
# ---------------------------------------------------------------------------


def weights_to_gf_image(
    weights: Union[torch.Tensor, np.ndarray],
    imsize: int = 256,
) -> np.ndarray:
    """Convert flat model weights to a resized GF image.

    Combines :func:`~utils.representation.weights_to_channels`,
    :func:`channels_to_gf_image`, and :func:`resize_gf_image` in one call.

    Args:
        weights: 1-D float tensor or array of model weights.
        imsize: Target side length of the output GF image in pixels.
            Use 256 for SRNet (default) or 100 for the OSL CNN.

    Returns:
        ``uint8`` array with shape ``(imsize, imsize)``.
    """
    channels = weights_to_channels(weights)   # (4, S, S) uint8
    gf = channels_to_gf_image(channels)       # (2S, 2S) uint8
    return resize_gf_image(gf, imsize)


def state_dict_to_gf_image(
    state_dict: dict,
    imsize: int = 256,
) -> np.ndarray:
    """Convert a PyTorch state_dict to a resized GF image.

    Extracts all floating-point tensors in state-dict order, concatenates
    them into a flat weight vector, and calls :func:`weights_to_gf_image`.

    Args:
        state_dict: A ``model.state_dict()`` dictionary.
        imsize: Target side length of the output GF image.

    Returns:
        ``uint8`` array with shape ``(imsize, imsize)``.
    """
    tensors = []
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            tensors.append(
                tensor.detach().cpu().to(dtype=torch.float32).reshape(-1)
            )
    flat = torch.cat(tensors) if tensors else torch.empty(0, dtype=torch.float32)
    return weights_to_gf_image(flat, imsize=imsize)


# ---------------------------------------------------------------------------
# Tensor conversion for SRNet input
# ---------------------------------------------------------------------------


def gf_image_to_tensor(gf_image: np.ndarray) -> torch.Tensor:
    """Convert a GF image array to a float32 torch tensor for SRNet input.

    Args:
        gf_image: ``uint8``-compatible 2-D array with shape ``(H, W)``.

    Returns:
        Float32 tensor with shape ``(1, H, W)`` with values in ``[0, 255]``.
        The channel dimension is the single grayscale channel expected by
        :class:`~models.srnet_detector.SRNetDetector`.
    """
    arr = np.asarray(gf_image, dtype=np.uint8)
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
