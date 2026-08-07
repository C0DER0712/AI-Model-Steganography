"""Focused tests for the standalone SRNet/FSL detection path."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from evaluation.fsl_detector import (
    FSLConfig,
    FSLDataset,
    FSLDetector,
    NearestCentroidClassifier,
    weighted_metric,
)
from models.srnet_detector import SRNetConfig, SRNetDetector
from utils.gf_image import channels_to_gf_image, gf_image_to_channels


def test_gf_conversion_round_trip() -> None:
    channels = np.arange(4 * 8 * 8, dtype=np.uint8).reshape(4, 8, 8)
    gf = channels_to_gf_image(channels)

    assert gf.shape == (16, 16)
    np.testing.assert_array_equal(gf_image_to_channels(gf), channels)


def test_srnet_returns_normalized_embedding() -> None:
    model = SRNetDetector(SRNetConfig(embedding_dim=32))
    embeddings = model.embed(torch.randint(0, 256, (2, 1, 32, 32)).float())

    assert embeddings.shape == (2, 32)
    assert torch.allclose(
        torch.linalg.vector_norm(embeddings, dim=1),
        torch.ones(2),
        atol=1e-5,
    )


def test_fsl_detector_fit_and_predict() -> None:
    rng = np.random.default_rng(7)
    benign = [rng.integers(0, 256, (16, 16), dtype=np.uint8) for _ in range(2)]
    malicious = [rng.integers(0, 256, (16, 16), dtype=np.uint8) for _ in range(2)]
    detector = FSLDetector(
        FSLConfig(
            imsize=16,
            embedding_dim=16,
            num_epochs=1,
            num_triplets_per_epoch=2,
            batch_size=2,
            device="cpu",
        ),
        srnet=SRNetDetector(SRNetConfig(embedding_dim=16)),
    )

    losses = detector.fit(benign, malicious)
    result = detector.predict_gf(benign[0])

    assert len(losses) == 1
    assert result.dist_to_benign >= 0
    assert result.dist_to_malicious >= 0
    assert 0 <= result.anomaly_score <= 1


def test_centroid_and_weighted_metric() -> None:
    classifier = NearestCentroidClassifier(embedding_dim=2)
    classifier.fit(torch.tensor([[0.0, 0.0], [1.0, 1.0]]), [0, 1])
    predicted, benign_distance, malicious_distance = classifier.predict(
        torch.tensor([0.1, 0.1])
    )

    assert predicted == 0
    assert benign_distance < malicious_distance
    assert weighted_metric({0: 1.0, 1: 1.0, 2: 1.0}, max_severity=2) == 1.0
    with pytest.raises(ValueError):
        weighted_metric({0: 1.1}, max_severity=2)