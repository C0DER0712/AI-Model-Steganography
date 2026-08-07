"""Persistence tests for the SRNet/FSL detector artefacts."""

from __future__ import annotations

import numpy as np
import torch

from evaluation.fsl_detector import FSLConfig, FSLDetector
from models.srnet_detector import SRNetConfig, SRNetDetector


def test_srnet_save_and_load_preserves_embeddings(tmp_path) -> None:
    model = SRNetDetector(SRNetConfig(embedding_dim=8))
    inputs = torch.randint(0, 256, (1, 1, 16, 16)).float()

    model.eval()
    with torch.no_grad():
        expected = model.embed(inputs)

    path = tmp_path / "srnet.pt"
    model.save(path)
    restored = SRNetDetector.load(path, map_location="cpu")
    restored.eval()
    with torch.no_grad():
        actual = restored.embed(inputs)

    assert torch.allclose(expected, actual)


def test_fsl_save_and_load_predicts(tmp_path) -> None:
    rng = np.random.default_rng(11)
    benign = [rng.integers(0, 256, (16, 16), dtype=np.uint8) for _ in range(2)]
    malicious = [rng.integers(0, 256, (16, 16), dtype=np.uint8) for _ in range(2)]
    detector = FSLDetector(
        FSLConfig(
            imsize=16,
            embedding_dim=8,
            num_epochs=1,
            num_triplets_per_epoch=2,
            batch_size=2,
            device="cpu",
        ),
        srnet=SRNetDetector(SRNetConfig(embedding_dim=8)),
    )
    detector.fit(benign, malicious)
    detector.save(tmp_path)

    restored = FSLDetector.load(
        tmp_path / "srnet.pt",
        tmp_path / "classifier.pt",
        config=FSLConfig(imsize=16, embedding_dim=8, device="cpu"),
        map_location="cpu",
    )
    result = restored.predict_gf(benign[0])

    assert restored.is_fitted
    assert 0.0 <= result.anomaly_score <= 1.0