"""PyTorch device selection helpers."""

from __future__ import annotations

import torch


def get_device(preferred: str | None = None) -> torch.device:
    """Return a usable PyTorch device.

    Args:
        preferred: Optional device name. Supported values are `cpu`, `cuda`,
            `mps`, or `auto`.

    Returns:
        A `torch.device` selected from available hardware.

    Raises:
        ValueError: If a requested explicit device is unsupported or
            unavailable.
    """

    choice = (preferred or "auto").lower()

    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if choice == "cpu":
        return torch.device("cpu")

    if choice == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return torch.device("cuda")

    if choice == "mps":
        if not _mps_available():
            raise ValueError("MPS was requested but is not available.")
        return torch.device("mps")

    raise ValueError(f"Unsupported device preference: {preferred}")


def _mps_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
