"""Training datasets for the steganographic embedding pipeline.

Provides two dataset types:

1. :class:`SteganographyDataset` — wraps a classification image dataset and
   attaches a fresh random payload to each sample.  The payload size is fixed
   at construction time.

2. :class:`SyntheticWeightDataset` — generates synthetic random weight
   representations paired with random payloads.  This enables pipeline
   testing without a real image dataset or pretrained host model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.payload import SUPPORTED_PAYLOAD_SIZES, PayloadSize, generate_payload, payload_to_tensor


@dataclass(frozen=True)
class SteganographyBatch:
    """A single training batch for the steganographic pipeline.

    Attributes:
        images: Float image tensor, shape ``(B, 3, H, W)``.
        labels: Class index tensor, shape ``(B,)``.
        payload_bits: Binary bit tensor, shape ``(B, num_bits)``.
    """

    images: torch.Tensor
    labels: torch.Tensor
    payload_bits: torch.Tensor


class SteganographyDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Image classification dataset augmented with random payload tensors.

    Each ``__getitem__`` call returns ``(image, label, payload_bits)`` where
    ``payload_bits`` is a freshly generated (or deterministically seeded)
    binary bit tensor of length ``payload_bits``.

    Args:
        image_dataset: Any ``Dataset`` that yields ``(image, label)`` pairs.
        payload_size: One of the supported payload sizes or an exact byte count.
        payload_seed: Base seed for reproducible payload generation.  Sample
            ``i`` uses seed ``payload_seed + i``.  Pass ``None`` for random.

    Example::

        from torchvision.datasets import CIFAR10
        from torchvision import transforms

        ds = SteganographyDataset(
            CIFAR10(root="/tmp", transform=transforms.ToTensor()),
            payload_size="128KB",
        )
        image, label, bits = ds[0]
    """

    def __init__(
        self,
        image_dataset: Dataset[tuple[torch.Tensor, Any]],
        payload_size: PayloadSize | int,
        *,
        payload_seed: int | None = None,
    ) -> None:
        _validate_payload_size(payload_size)
        self.image_dataset = image_dataset
        self.payload_size = payload_size
        self.payload_seed = payload_seed

    def __len__(self) -> int:
        return len(self.image_dataset)  # type: ignore[arg-type]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, label = self.image_dataset[index]
        seed = None if self.payload_seed is None else self.payload_seed + index
        payload = generate_payload(self.payload_size, seed=seed)
        bits = payload_to_tensor(payload)
        return image, torch.tensor(label, dtype=torch.long), bits


class SyntheticWeightDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Synthetic dataset of random weight representations and payloads.

    Useful for unit-testing and smoke-running the full pipeline without a
    real image dataset or host model.  Each sample returns:

    * ``weight_repr``: 4-channel float32 representation of shape
      ``(4, image_side, image_side)`` with pixel values in ``[0, 255]``.
    * ``labels``: Zero-filled label tensor (shape ``(1,)``).
    * ``payload_bits``: Binary bit tensor of length ``payload_bits``.

    Args:
        count: Number of synthetic samples.
        num_weights: Number of synthetic model weights to simulate.
        payload_size: Payload size per sample.
        num_classes: Number of fake output classes.
        seed: Global random seed for reproducibility.
    """

    def __init__(
        self,
        count: int,
        num_weights: int = 11_689_512,  # Approx ResNet18 parameter count
        payload_size: PayloadSize | int = "128KB",
        num_classes: int = 1000,
        seed: int | None = None,
    ) -> None:
        if count <= 0:
            raise ValueError("count must be positive.")
        if num_weights <= 0:
            raise ValueError("num_weights must be positive.")
        _validate_payload_size(payload_size)
        self.count = count
        self.num_weights = num_weights
        self.payload_size = payload_size
        self.num_classes = num_classes
        self.seed = seed
        self._side = math.ceil(math.sqrt(num_weights))

    def __len__(self) -> int:
        return self.count

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.count:
            raise IndexError(index)

        rng = np.random.default_rng(
            None if self.seed is None else self.seed + index
        )

        # Synthetic weight representation: uniform bytes
        side = self._side
        channels = rng.integers(0, 256, size=(4, side, side), dtype=np.uint8)
        weight_repr = torch.from_numpy(channels.astype(np.float32))

        # Random class label
        label = torch.tensor(
            rng.integers(0, self.num_classes), dtype=torch.long
        )

        # Random payload
        payload_seed = None if self.seed is None else self.seed * 10_000 + index
        payload = generate_payload(self.payload_size, seed=payload_seed)
        bits = payload_to_tensor(payload)

        return weight_repr, label, bits


class SyntheticImageDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Synthetic dataset of random RGB images and payloads.

    Generates ``(3, image_size, image_size)`` float32 images with pixel values
    in ``[0, 1]``, random class labels, and random binary payload tensors.
    This is the correct synthetic dataset to use with the training pipeline,
    because the pipeline expects classification images as input (not weight
    representations).

    Args:
        count: Number of synthetic samples.
        payload_size: Payload size per sample.
        num_classes: Number of fake output classes.
        image_size: Spatial size of each synthetic image (square).
        seed: Global random seed for reproducibility.
    """

    def __init__(
        self,
        count: int,
        payload_size: PayloadSize | int = "128KB",
        num_classes: int = 1000,
        image_size: int = 32,
        seed: int | None = None,
    ) -> None:
        if count <= 0:
            raise ValueError("count must be positive.")
        _validate_payload_size(payload_size)
        self.count = count
        self.payload_size = payload_size
        self.num_classes = num_classes
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.count

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.count:
            raise IndexError(index)

        rng = np.random.default_rng(
            None if self.seed is None else self.seed + index
        )

        # Synthetic RGB image — uniform random pixels in [0, 1].
        image = torch.from_numpy(
            rng.random((3, self.image_size, self.image_size), dtype=np.float32)
        )

        # Random class label.
        label = torch.tensor(
            rng.integers(0, self.num_classes), dtype=torch.long
        )

        # Random binary payload.
        payload_seed = None if self.seed is None else self.seed * 10_000 + index
        payload = generate_payload(self.payload_size, seed=payload_seed)
        bits = payload_to_tensor(payload)

        return image, label, bits


def build_data_loaders(
    train_dataset: Dataset[Any],
    val_dataset: Dataset[Any] | None = None,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    collate_fn: Callable[..., Any] | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any] | None]:
    """Build train and optional validation data loaders.

    Args:
        train_dataset: Training dataset.
        val_dataset: Optional validation dataset.
        batch_size: Number of samples per batch.
        num_workers: DataLoader worker processes.
        pin_memory: Whether to pin memory for faster GPU transfers.
        collate_fn: Optional custom collation function.

    Returns:
        Tuple of ``(train_loader, val_loader)``.  ``val_loader`` is ``None``
        when no validation dataset is provided.
    """
    common_kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if collate_fn is not None:
        common_kwargs["collate_fn"] = collate_fn

    train_loader: DataLoader[Any] = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **common_kwargs,
    )

    val_loader: DataLoader[Any] | None = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            **common_kwargs,
        )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_payload_size(size: PayloadSize | int) -> None:
    if isinstance(size, str) and size not in SUPPORTED_PAYLOAD_SIZES:
        supported = ", ".join(SUPPORTED_PAYLOAD_SIZES)
        raise ValueError(
            f"Unsupported payload size '{size}'. Supported: {supported}."
        )
