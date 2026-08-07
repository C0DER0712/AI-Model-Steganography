"""Tests for embedding capacity calculations."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from evaluation.capacity import CapacityResult, compute_capacity, max_payload_bits, capacity_summary


def _make_model(num_params: int = 1024) -> nn.Module:
    """Create a simple linear model with approximately num_params parameters."""
    side = max(1, int(num_params ** 0.5))
    return nn.Linear(side, side)


class TestComputeCapacity:
    def test_returns_capacity_result(self) -> None:
        model = _make_model()
        result = compute_capacity(model, payload_bits=64)
        assert isinstance(result, CapacityResult)

    def test_bits_per_parameter_positive(self) -> None:
        model = _make_model()
        result = compute_capacity(model, payload_bits=64)
        assert result.bits_per_parameter > 0

    def test_embedding_rate_in_range(self) -> None:
        model = _make_model()
        result = compute_capacity(model, payload_bits=64)
        assert 0.0 < result.embedding_rate <= 1.0

    def test_payload_bits_too_large_raises(self) -> None:
        model = nn.Linear(2, 2)  # small model
        num_params = sum(p.numel() for p in model.parameters() if p.is_floating_point())
        total_bits = num_params * 32
        with pytest.raises(ValueError, match="payload_bits"):
            compute_capacity(model, payload_bits=total_bits + 1)

    def test_zero_payload_bits(self) -> None:
        model = _make_model()
        result = compute_capacity(model, payload_bits=0)
        assert result.bits_per_parameter == 0.0
        assert result.embedding_rate == 0.0

    def test_ber_affects_effective_bits(self) -> None:
        model = _make_model()
        result_no_ber = compute_capacity(model, payload_bits=64, bit_error_rate=0.0)
        result_with_ber = compute_capacity(model, payload_bits=64, bit_error_rate=0.1)
        assert result_no_ber.effective_payload_bits > result_with_ber.effective_payload_bits

    def test_ber_none_preserved(self) -> None:
        model = _make_model()
        result = compute_capacity(model, payload_bits=64, bit_error_rate=None)
        assert result.bit_error_rate is None

    def test_to_dict_has_required_keys(self) -> None:
        model = _make_model()
        result = compute_capacity(model, payload_bits=64)
        d = result.to_dict()
        required = {
            "num_float_parameters",
            "num_float_bits",
            "payload_bits",
            "bits_per_parameter",
            "embedding_rate",
            "effective_payload_bits",
        }
        assert required.issubset(d.keys())


class TestMaxPayloadBits:
    def test_returns_positive_int(self) -> None:
        model = _make_model(1024)
        result = max_payload_bits(model, target_ber=0.01)
        assert isinstance(result, int)
        assert result > 0

    def test_smaller_ber_gives_larger_capacity(self) -> None:
        model = _make_model(1024)
        cap_strict = max_payload_bits(model, target_ber=0.001)
        cap_loose = max_payload_bits(model, target_ber=0.1)
        assert cap_strict > cap_loose

    def test_invalid_ber_raises(self) -> None:
        model = _make_model()
        with pytest.raises(ValueError, match="target_ber"):
            max_payload_bits(model, target_ber=1.5)


class TestCapacitySummary:
    def test_returns_list_of_correct_length(self) -> None:
        model = _make_model(2048)
        payload_list = [64, 128, 256]
        results = capacity_summary(model, payload_list)
        assert len(results) == 3

    def test_mismatched_lengths_raise(self) -> None:
        model = _make_model()
        with pytest.raises(ValueError):
            capacity_summary(model, [64, 128], bit_error_rates=[0.01])
