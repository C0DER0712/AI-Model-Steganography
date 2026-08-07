"""Benign random payload utilities.

This module only creates random byte payloads for defensive research tests. It
does not generate, transform, execute, or otherwise handle malware.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


PayloadSize = Literal["128KB", "256KB", "512KB", "1MB"]

SUPPORTED_PAYLOAD_SIZES: dict[PayloadSize, int] = {
    "128KB": 128 * 1024,
    "256KB": 256 * 1024,
    "512KB": 512 * 1024,
    "1MB": 1024 * 1024,
}


def generate_payload(size: PayloadSize | int, seed: int | None = None) -> bytes:
    """Generate a benign random payload.

    Args:
        size: One of `128KB`, `256KB`, `512KB`, `1MB`, or an exact supported
            size in bytes.
        seed: Optional seed for reproducible research fixtures.

    Returns:
        Random payload bytes.

    Raises:
        ValueError: If the requested size is unsupported.
    """

    num_bytes = _resolve_payload_size(size)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=num_bytes, dtype=np.uint8).tobytes()


def payload_to_tensor(payload: bytes | bytearray | memoryview) -> torch.Tensor:
    """Convert payload bytes to a 1D bit tensor.

    Args:
        payload: Raw payload bytes.

    Returns:
        CPU `torch.uint8` tensor containing one bit per element in stable
        most-significant-bit-first order.
    """

    byte_array = np.frombuffer(bytes(payload), dtype=np.uint8)
    bit_array = np.unpackbits(byte_array, bitorder="big")
    return torch.from_numpy(bit_array.astype(np.uint8, copy=False))


def tensor_to_payload(tensor: torch.Tensor) -> bytes:
    """Convert a 1D bit tensor back to payload bytes.

    Args:
        tensor: Tensor containing binary values. Floating tensors are accepted
            when all values are exactly 0 or 1.

    Returns:
        Reconstructed payload bytes.

    Raises:
        ValueError: If the tensor is not one-dimensional, has non-binary
            values, or its length is not divisible by 8.
    """

    bits = tensor.detach().cpu().reshape(-1)
    if tensor.ndim != 1:
        raise ValueError("Payload tensor must be one-dimensional.")
    if bits.numel() % 8 != 0:
        raise ValueError("Payload tensor length must be divisible by 8.")

    if not torch.all((bits == 0) | (bits == 1)):
        raise ValueError("Payload tensor must contain only binary values 0 or 1.")

    bit_array = bits.to(dtype=torch.uint8).numpy()
    return np.packbits(bit_array, bitorder="big").tobytes()


def save_payload(payload: bytes | bytearray | memoryview, path: str | Path) -> Path:
    """Save payload bytes to disk.

    Args:
        payload: Payload bytes to persist.
        path: Destination file path.

    Returns:
        Resolved destination path.
    """

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(bytes(payload))
    return destination


def load_payload(path: str | Path) -> bytes:
    """Load payload bytes from disk.

    Args:
        path: Source file path.

    Returns:
        Payload bytes.

    Raises:
        FileNotFoundError: If the payload file does not exist.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Payload file not found: {source}")

    return source.read_bytes()


def bit_error_rate(original: bytes | torch.Tensor, recovered: bytes | torch.Tensor) -> float:
    """Measure payload bit error rate.

    Args:
        original: Original payload as bytes or a bit tensor.
        recovered: Recovered payload as bytes or a bit tensor.

    Returns:
        Fraction of differing bits in `[0.0, 1.0]`.

    Raises:
        ValueError: If the payloads do not contain the same number of bits.
    """

    original_bits = _as_bit_tensor(original)
    recovered_bits = _as_bit_tensor(recovered)

    if original_bits.numel() != recovered_bits.numel():
        raise ValueError(
            "Payloads must contain the same number of bits to compare integrity."
        )
    if original_bits.numel() == 0:
        return 0.0

    errors = torch.count_nonzero(original_bits != recovered_bits).item()
    return errors / original_bits.numel()


def payload_dataset(
    count: int,
    size: PayloadSize | int,
    *,
    seed: int | None = None,
) -> "RandomPayloadDataset":
    """Create a dataset of benign random payload tensors.

    Args:
        count: Number of payload samples.
        size: Supported payload size.
        seed: Optional base seed for reproducible payload generation.

    Returns:
        Dataset returning 1D `torch.uint8` bit tensors.
    """

    return RandomPayloadDataset(count=count, size=size, seed=seed)


@dataclass(frozen=True)
class RandomPayloadDataset(Dataset[torch.Tensor]):
    """Dataset of generated benign random payload tensors."""

    count: int
    size: PayloadSize | int
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("count must be non-negative.")
        _resolve_payload_size(self.size)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= self.count:
            raise IndexError(index)

        item_seed = None if self.seed is None else self.seed + index
        return payload_to_tensor(generate_payload(self.size, seed=item_seed))


def _resolve_payload_size(size: PayloadSize | int) -> int:
    if isinstance(size, str):
        if size not in SUPPORTED_PAYLOAD_SIZES:
            supported = ", ".join(SUPPORTED_PAYLOAD_SIZES)
            raise ValueError(f"Unsupported payload size '{size}'. Use one of: {supported}.")
        return SUPPORTED_PAYLOAD_SIZES[size]

    if size not in set(SUPPORTED_PAYLOAD_SIZES.values()):
        supported = ", ".join(str(value) for value in SUPPORTED_PAYLOAD_SIZES.values())
        raise ValueError(f"Unsupported payload size {size}. Use one of: {supported} bytes.")

    return size


def _as_bit_tensor(payload: bytes | torch.Tensor) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        return payload.detach().cpu().reshape(-1).to(dtype=torch.uint8)

    return payload_to_tensor(payload)
