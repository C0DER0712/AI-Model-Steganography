"""Host model accuracy-drop evaluation.

Measures the classification accuracy of a host model before and after
steganographic weight modification, reporting the accuracy drop as one of
the four key research metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccuracyResult:
    """Classification accuracy measurement for one model checkpoint.

    Attributes:
        top1_accuracy: Top-1 classification accuracy in ``[0, 1]``.
        top5_accuracy: Top-5 classification accuracy in ``[0, 1]``.
        num_samples: Number of evaluation samples.
        num_correct_top1: Raw count of top-1 correct predictions.
        num_correct_top5: Raw count of top-5 correct predictions.
    """

    top1_accuracy: float
    top5_accuracy: float
    num_samples: int
    num_correct_top1: int
    num_correct_top5: int

    def __str__(self) -> str:
        return (
            f"AccuracyResult("
            f"top1={self.top1_accuracy:.4f}, "
            f"top5={self.top5_accuracy:.4f}, "
            f"n={self.num_samples})"
        )


@dataclass(frozen=True)
class AccuracyDropResult:
    """Accuracy drop between original and steganographic host model.

    Attributes:
        original: Accuracy on the unmodified host model.
        modified: Accuracy on the host model with embedded payload.
        top1_drop: Absolute top-1 accuracy drop (original - modified).
        top5_drop: Absolute top-5 accuracy drop (original - modified).
    """

    original: AccuracyResult
    modified: AccuracyResult
    top1_drop: float
    top5_drop: float

    def to_dict(self) -> dict[str, float]:
        """Return scalar metrics as a flat dictionary."""
        return {
            "original_top1": self.original.top1_accuracy,
            "original_top5": self.original.top5_accuracy,
            "modified_top1": self.modified.top1_accuracy,
            "modified_top5": self.modified.top5_accuracy,
            "top1_drop": self.top1_drop,
            "top5_drop": self.top5_drop,
        }

    def __str__(self) -> str:
        return (
            f"AccuracyDropResult(\n"
            f"  original  top1={self.original.top1_accuracy:.4f}  "
            f"top5={self.original.top5_accuracy:.4f}\n"
            f"  modified  top1={self.modified.top1_accuracy:.4f}  "
            f"top5={self.modified.top5_accuracy:.4f}\n"
            f"  drop      top1={self.top1_drop:.4f}  top5={self.top5_drop:.4f}"
            f"\n)"
        )


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device | str | None = None,
    verbose: bool = True,
    max_batches: int | None = None,
) -> AccuracyResult:
    """Evaluate top-1 and top-5 classification accuracy.

    Args:
        model: Classification model returning logits.
        data_loader: Iterable yielding ``(images, labels)`` batches.
        device: Target device.  Auto-detected if ``None``.
        verbose: Whether to show a progress bar.
        max_batches: Optional limit on the number of batches evaluated.

    Returns:
        :class:`AccuracyResult` with top-1 and top-5 accuracy.
    """
    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    iterable = tqdm(data_loader, desc="Evaluating accuracy", disable=not verbose)
    for batch_idx, (images, labels) in enumerate(iterable):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(resolved_device)
        labels = labels.to(resolved_device)

        logits = model(images)
        top1, top5 = _topk_accuracy(logits, labels, topk=(1, 5))

        batch_size = labels.size(0)
        correct_top1 += top1
        correct_top5 += top5
        total += batch_size

    if total == 0:
        return AccuracyResult(
            top1_accuracy=0.0,
            top5_accuracy=0.0,
            num_samples=0,
            num_correct_top1=0,
            num_correct_top5=0,
        )

    return AccuracyResult(
        top1_accuracy=correct_top1 / total,
        top5_accuracy=correct_top5 / total,
        num_samples=total,
        num_correct_top1=correct_top1,
        num_correct_top5=correct_top5,
    )


def evaluate_accuracy_drop(
    original_model: nn.Module,
    modified_model: nn.Module,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device | str | None = None,
    verbose: bool = True,
    max_batches: int | None = None,
) -> AccuracyDropResult:
    """Measure accuracy drop between original and modified host models.

    Args:
        original_model: The unmodified host model.
        modified_model: The host model with steganographic weights.
        data_loader: Shared evaluation data loader.
        device: Target device.
        verbose: Whether to show progress bars.
        max_batches: Optional batch limit for fast evaluation.

    Returns:
        :class:`AccuracyDropResult` summarising accuracy before and after
        embedding.
    """
    logger.info("Evaluating original model accuracy…")
    original_result = evaluate_accuracy(
        original_model,
        data_loader,
        device=device,
        verbose=verbose,
        max_batches=max_batches,
    )

    logger.info("Evaluating modified model accuracy…")
    modified_result = evaluate_accuracy(
        modified_model,
        data_loader,
        device=device,
        verbose=verbose,
        max_batches=max_batches,
    )

    drop1 = original_result.top1_accuracy - modified_result.top1_accuracy
    drop5 = original_result.top5_accuracy - modified_result.top5_accuracy

    result = AccuracyDropResult(
        original=original_result,
        modified=modified_result,
        top1_drop=drop1,
        top5_drop=drop5,
    )

    if verbose:
        logger.info(str(result))

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk: tuple[int, ...] = (1, 5),
) -> list[int]:
    """Return per-topk counts of correct predictions."""
    max_k = max(topk)
    num_classes = logits.shape[-1]
    effective_k = min(max_k, num_classes)

    _, predicted = logits.topk(effective_k, dim=1, largest=True, sorted=True)
    predicted = predicted.t()
    correct = predicted.eq(labels.unsqueeze(0))

    results = []
    for k in topk:
        k_eff = min(k, num_classes)
        correct_k = correct[:k_eff].reshape(-1).float().sum(0)
        results.append(int(correct_k.sum().item()))

    return results


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)
