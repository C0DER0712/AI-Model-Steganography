"""Tests for accuracy-drop evaluation."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.accuracy import (
    AccuracyDropResult,
    AccuracyResult,
    evaluate_accuracy,
    evaluate_accuracy_drop,
)


def _make_loader(num_samples: int = 20, num_classes: int = 5) -> DataLoader:
    images = torch.randn(num_samples, 3, 8, 8)
    labels = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=4)


def _make_perfect_model(num_classes: int = 5) -> nn.Module:
    """A model that always predicts label 0 perfectly for the first class."""
    class AlwaysFirst(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            logits = torch.full((batch, num_classes), -100.0)
            logits[:, 0] = 100.0
            return logits
    return AlwaysFirst()


def _make_random_model(num_classes: int = 5) -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, num_classes))


class TestEvaluateAccuracy:
    def test_returns_accuracy_result(self) -> None:
        loader = _make_loader()
        model = _make_random_model()
        result = evaluate_accuracy(model, loader, verbose=False)
        assert isinstance(result, AccuracyResult)

    def test_top1_in_range(self) -> None:
        loader = _make_loader()
        model = _make_random_model()
        result = evaluate_accuracy(model, loader, verbose=False)
        assert 0.0 <= result.top1_accuracy <= 1.0

    def test_top5_geq_top1(self) -> None:
        loader = _make_loader(num_samples=20, num_classes=5)
        model = _make_random_model(num_classes=5)
        result = evaluate_accuracy(model, loader, verbose=False)
        assert result.top5_accuracy >= result.top1_accuracy

    def test_num_samples_matches_dataset(self) -> None:
        loader = _make_loader(num_samples=20)
        model = _make_random_model()
        result = evaluate_accuracy(model, loader, verbose=False)
        assert result.num_samples == 20

    def test_max_batches_limits_evaluation(self) -> None:
        loader = _make_loader(num_samples=40)
        model = _make_random_model()
        result = evaluate_accuracy(model, loader, verbose=False, max_batches=2)
        assert result.num_samples <= 40

    def test_empty_loader_returns_zeros(self) -> None:
        loader = DataLoader(TensorDataset(torch.empty(0, 3, 8, 8), torch.empty(0, dtype=torch.long)), batch_size=4)
        model = _make_random_model()
        result = evaluate_accuracy(model, loader, verbose=False)
        assert result.num_samples == 0
        assert result.top1_accuracy == 0.0


class TestEvaluateAccuracyDrop:
    def test_returns_accuracy_drop_result(self) -> None:
        loader = _make_loader()
        orig = _make_random_model()
        modified = _make_random_model()
        result = evaluate_accuracy_drop(orig, modified, loader, verbose=False)
        assert isinstance(result, AccuracyDropResult)

    def test_identical_models_zero_drop(self) -> None:
        loader = _make_loader()
        model = _make_random_model()
        result = evaluate_accuracy_drop(model, model, loader, verbose=False)
        assert abs(result.top1_drop) < 1e-6

    def test_drop_computed_correctly(self) -> None:
        loader = _make_loader()
        orig = _make_random_model()
        modified = _make_random_model()
        result = evaluate_accuracy_drop(orig, modified, loader, verbose=False)
        expected_drop1 = result.original.top1_accuracy - result.modified.top1_accuracy
        assert abs(result.top1_drop - expected_drop1) < 1e-6

    def test_to_dict_keys(self) -> None:
        loader = _make_loader()
        orig = _make_random_model()
        modified = _make_random_model()
        result = evaluate_accuracy_drop(orig, modified, loader, verbose=False)
        d = result.to_dict()
        assert "original_top1" in d
        assert "modified_top1" in d
        assert "top1_drop" in d
        assert "top5_drop" in d
