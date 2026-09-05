"""Smoke tests for the experiment orchestration module.

Uses the ``"tiny"`` synthetic backbone and a tiny random image dataset so
the full training loop can complete in seconds without OOM.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from evaluation.differentiable_detector import DifferentiableDetectorConfig
from models.decoder import DecoderConfig
from models.encoder import EncoderConfig
from models.pipeline import PipelineConfig
from training.experiment import ExperimentConfig, ExperimentResult, SteganographyExperiment
from training.losses import LossWeights


PAYLOAD_BITS = 64  # 8 bytes — keeps tensors tiny


# ---------------------------------------------------------------------------
# Tiny synthetic image dataset (3-channel, required by the host model)
# ---------------------------------------------------------------------------

class _TinyImageDataset(Dataset):
    """Synthetic (3, 8, 8) image dataset with random payload bits."""

    def __init__(self, count: int, payload_bits: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self._images = torch.from_numpy(
            rng.random((count, 3, 8, 8)).astype(np.float32)
        )
        self._labels = torch.zeros(count, dtype=torch.long)
        self._bits = torch.from_numpy(
            rng.integers(0, 2, (count, payload_bits)).astype(np.float32)
        )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        return self._images[idx], self._labels[idx], self._bits[idx]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _tiny_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        host_model_name="tiny",
        host_model_num_classes=10,
        host_model_pretrained=False,
        payload_bits=PAYLOAD_BITS,
        encoder=EncoderConfig(
            payload_dim=PAYLOAD_BITS, base_channels=8, num_residual_blocks=1
        ),
        decoder=DecoderConfig(
            base_channels=8, num_residual_blocks=1
        ),
        detector=DifferentiableDetectorConfig(
            hpf_channels=4, base_channels=4, num_type1_blocks=1,
            num_type2_blocks=1, fc_hidden_dim=8,
        ),
    )


def _tiny_experiment_config(output_dir: str, max_epochs: int = 1) -> ExperimentConfig:
    return ExperimentConfig(
        output_dir=output_dir,
        max_epochs=max_epochs,
        batch_size=2,
        learning_rate=1e-3,
        early_stopping_patience=None,
        scheduler="cosine",
        scheduler_t_max=max_epochs,
        device="cpu",
        pipeline=_tiny_pipeline_config(),
        loss_weights=LossWeights(
            classification=1.0, payload=1.0, distortion=1.0, detector=1.0
        ),
        log_every_n_steps=1,
        num_workers=0,
    )


def _tiny_loaders(count: int = 4) -> tuple[DataLoader, DataLoader]:
    ds_train = _TinyImageDataset(count, PAYLOAD_BITS, seed=0)
    ds_val = _TinyImageDataset(count // 2, PAYLOAD_BITS, seed=99)
    train_loader = DataLoader(ds_train, batch_size=2, shuffle=False, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=2, shuffle=False, drop_last=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSteganographyExperiment:
    def test_run_returns_experiment_result(self, tmp_path) -> None:
        cfg = _tiny_experiment_config(str(tmp_path / "out1"), max_epochs=1)
        train_loader, val_loader = _tiny_loaders()
        exp = SteganographyExperiment(cfg)
        result = exp.run(train_loader, val_loader)
        assert isinstance(result, ExperimentResult)

    def test_history_has_epochs(self, tmp_path) -> None:
        cfg = _tiny_experiment_config(str(tmp_path / "out2"), max_epochs=2)
        train_loader, val_loader = _tiny_loaders()
        exp = SteganographyExperiment(cfg)
        result = exp.run(train_loader, val_loader)
        assert len(result.history) == 2

    def test_output_dirs_created(self, tmp_path) -> None:
        cfg = _tiny_experiment_config(str(tmp_path / "out3"), max_epochs=1)
        SteganographyExperiment(cfg)
        output_dir = tmp_path / "out3"
        for sub in ("checkpoints", "figures", "metrics", "logs"):
            assert (output_dir / sub).exists(), f"Missing subdir: {sub}"

    def test_result_output_dir_matches_config(self, tmp_path) -> None:
        out = str(tmp_path / "out4")
        cfg = _tiny_experiment_config(out, max_epochs=1)
        train_loader, val_loader = _tiny_loaders()
        exp = SteganographyExperiment(cfg)
        result = exp.run(train_loader, val_loader)
        assert result.output_dir.resolve() == (tmp_path / "out4").resolve()
