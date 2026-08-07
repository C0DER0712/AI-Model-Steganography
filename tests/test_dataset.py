"""Tests for steganography training datasets."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from training.dataset import (
    SyntheticWeightDataset,
    build_data_loaders,
)


class TestSyntheticWeightDataset:
    def test_len(self) -> None:
        ds = SyntheticWeightDataset(count=8, payload_size="128KB", seed=0)
        assert len(ds) == 8

    def test_item_types(self) -> None:
        ds = SyntheticWeightDataset(count=4, payload_size="128KB", seed=0)
        weight_repr, label, bits = ds[0]
        assert isinstance(weight_repr, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert isinstance(bits, torch.Tensor)

    def test_weight_repr_channels(self) -> None:
        ds = SyntheticWeightDataset(count=2, payload_size="128KB", seed=0)
        weight_repr, _, _ = ds[0]
        assert weight_repr.ndim == 3
        assert weight_repr.shape[0] == 4

    def test_bits_are_binary(self) -> None:
        ds = SyntheticWeightDataset(count=4, payload_size="128KB", seed=0)
        for i in range(4):
            _, _, bits = ds[i]
            assert torch.all((bits == 0) | (bits == 1))

    def test_bits_length_matches_payload(self) -> None:
        from utils.payload import SUPPORTED_PAYLOAD_SIZES
        ds = SyntheticWeightDataset(count=2, payload_size="128KB", seed=0)
        _, _, bits = ds[0]
        expected = SUPPORTED_PAYLOAD_SIZES["128KB"] * 8
        assert bits.numel() == expected

    def test_deterministic_with_seed(self) -> None:
        ds1 = SyntheticWeightDataset(count=4, payload_size="128KB", seed=42)
        ds2 = SyntheticWeightDataset(count=4, payload_size="128KB", seed=42)
        for i in range(4):
            r1, l1, b1 = ds1[i]
            r2, l2, b2 = ds2[i]
            assert torch.equal(r1, r2)
            assert torch.equal(b1, b2)

    def test_different_seeds_differ(self) -> None:
        ds1 = SyntheticWeightDataset(count=4, payload_size="128KB", seed=1)
        ds2 = SyntheticWeightDataset(count=4, payload_size="128KB", seed=2)
        _, _, b1 = ds1[0]
        _, _, b2 = ds2[0]
        assert not torch.equal(b1, b2)

    def test_index_out_of_bounds(self) -> None:
        ds = SyntheticWeightDataset(count=4, payload_size="128KB", seed=0)
        with pytest.raises(IndexError):
            ds[4]
        with pytest.raises(IndexError):
            ds[-1]

    def test_count_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="count"):
            SyntheticWeightDataset(count=0, payload_size="128KB")


class TestBuildDataLoaders:
    def test_returns_train_loader(self) -> None:
        ds = SyntheticWeightDataset(count=8, payload_size="128KB", seed=0)
        train, val = build_data_loaders(ds, batch_size=4)
        assert train is not None
        assert val is None

    def test_returns_val_loader_when_provided(self) -> None:
        ds_train = SyntheticWeightDataset(count=8, payload_size="128KB", seed=0)
        ds_val = SyntheticWeightDataset(count=4, payload_size="128KB", seed=1)
        train, val = build_data_loaders(ds_train, ds_val, batch_size=4)
        assert val is not None

    def test_batch_shape(self) -> None:
        ds = SyntheticWeightDataset(count=8, payload_size="128KB", seed=0)
        train, _ = build_data_loaders(ds, batch_size=4)
        batch = next(iter(train))
        weight_repr, labels, bits = batch
        assert weight_repr.shape[0] == 4
        assert labels.shape == (4,)
        assert bits.ndim == 2
        assert bits.shape[0] == 4
