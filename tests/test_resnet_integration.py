"""Integration test: BatchNorm-backed pipeline forward + backward.

Validates the BatchNorm host-model path (Conv → BN → ReLU, as found in
ResNet18/MobileNetV2/VGG16) using the ``"tiny_bn"`` synthetic backbone so
the test completes in seconds without OOMing on the full 11 M-parameter model.

The ``"tiny_bn"`` backbone is architecturally representative of the production
backbones for the property under test: it contains ``BatchNorm2d`` layers whose
``running_mean`` and ``running_var`` buffers appear in ``state_dict()`` as
float tensors.

Regression guards:
* ``_rebuild_params`` must **not** make ``running_mean`` / ``running_var``
  differentiable — ``F.batch_norm`` raises ``RuntimeError`` if they are.
* The encoder must receive non-zero gradients after a composite loss backward.
* Detector parameters must accumulate no gradient (frozen by design).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from evaluation.differentiable_detector import DifferentiableDetectorConfig
from models.decoder import DecoderConfig
from models.encoder import EncoderConfig
from models.pipeline import EmbeddingPipeline, PipelineConfig, _rebuild_params
from training.losses import CompositeLoss, LossWeights
from utils.weights import WeightTensor, extract_weights, flatten_weights


PAYLOAD_BITS = 64  # small but non-trivial payload for speed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_bn_pipeline() -> EmbeddingPipeline:
    """Build a tiny_bn (BatchNorm) pipeline, shared across this module."""
    cfg = PipelineConfig(
        host_model_name="tiny_bn",
        host_model_num_classes=10,
        host_model_pretrained=False,
        payload_bits=PAYLOAD_BITS,
        encoder=EncoderConfig(payload_dim=PAYLOAD_BITS, base_channels=8, num_residual_blocks=1),
        decoder=DecoderConfig(base_channels=8, num_residual_blocks=1, chunk_size=PAYLOAD_BITS),
        detector=DifferentiableDetectorConfig(
            hpf_channels=4, base_channels=4, num_type1_blocks=1,
            num_type2_blocks=1, fc_hidden_dim=8,
        ),
    )
    pipeline = EmbeddingPipeline(cfg)
    # eval() is required for BatchNorm to use stored running stats; also
    # prevents the single-sample batch-stat ValueError.
    pipeline.host_model.eval()
    return pipeline


# ---------------------------------------------------------------------------
# Unit test: _rebuild_params does not make BN buffers differentiable
# ---------------------------------------------------------------------------

class TestRebuildParamsBatchNormBuffers:
    """Direct unit test of _rebuild_params — no full pipeline needed."""

    def test_running_mean_not_differentiable(self) -> None:
        """running_mean must never carry requires_grad=True into functional_call."""
        # Tiny model with BatchNorm, matching the structural property of
        # ResNet18 / MobileNetV2 / VGG16.
        model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 2),
        )
        model.eval()

        records = extract_weights(model)
        flat = flatten_weights(records)

        # Simulate a differentiable STE output (as returned by the encoder).
        source = flat.clone().requires_grad_(True)
        modified_flat = source * 1.0  # non-leaf, carries grad_fn

        params = _rebuild_params(modified_flat, records, model)

        for name, tensor in params.items():
            if "running_mean" in name or "running_var" in name:
                assert not tensor.requires_grad, (
                    f"Buffer '{name}' must not require grad — "
                    "F.batch_norm rejects differentiable running statistics."
                )

    def test_learnable_params_are_differentiable(self) -> None:
        """Weight and bias tensors must retain grad_fn from the STE output."""
        model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 2),
        )
        model.eval()

        records = extract_weights(model)
        flat = flatten_weights(records)
        source = flat.clone().requires_grad_(True)
        modified_flat = source * 1.0

        params = _rebuild_params(modified_flat, records, model)
        learnable = {n for n, _ in model.named_parameters()}

        differentiable_count = sum(
            1 for name, t in params.items()
            if name in learnable and t.requires_grad
        )
        assert differentiable_count > 0, "No learnable parameter received a differentiable replacement."


# ---------------------------------------------------------------------------
# Forward pass tests on tiny_bn pipeline
# ---------------------------------------------------------------------------

class TestTinyBnForwardPass:
    def test_forward_does_not_raise(self, tiny_bn_pipeline: EmbeddingPipeline) -> None:
        """Regression: forward must not raise RuntimeError from BN grad check."""
        images = torch.randn(2, 3, 8, 8)
        labels = torch.zeros(2, dtype=torch.long)
        payload = torch.randint(0, 2, (2, PAYLOAD_BITS), dtype=torch.float32)
        # Must not raise RuntimeError about non-differentiable running statistics.
        result = tiny_bn_pipeline(images, labels, payload)
        from training.losses import LossInputs
        assert isinstance(result, LossInputs)

    def test_classification_logits_shape(self, tiny_bn_pipeline: EmbeddingPipeline) -> None:
        images = torch.randn(2, 3, 8, 8)
        labels = torch.zeros(2, dtype=torch.long)
        payload = torch.randint(0, 2, (2, PAYLOAD_BITS), dtype=torch.float32)
        result = tiny_bn_pipeline(images, labels, payload)
        assert result.classification_logits.shape == (2, 10)


# ---------------------------------------------------------------------------
# Gradient flow tests on tiny_bn pipeline
# ---------------------------------------------------------------------------

class TestTinyBnGradientFlow:
    def test_encoder_receives_gradient(self, tiny_bn_pipeline: EmbeddingPipeline) -> None:
        tiny_bn_pipeline.encoder.zero_grad()
        images = torch.randn(2, 3, 8, 8)
        labels = torch.zeros(2, dtype=torch.long)
        payload = torch.randint(0, 2, (2, PAYLOAD_BITS), dtype=torch.float32)

        loss_fn = CompositeLoss(
            LossWeights(classification=1.0, payload=1.0, distortion=1.0, detector=1.0)
        )
        result = tiny_bn_pipeline(images, labels, payload)
        loss_fn(result).total.backward()

        enc_grads = [
            p.grad for p in tiny_bn_pipeline.encoder.parameters()
            if p.grad is not None
        ]
        assert len(enc_grads) > 0, "Encoder received no gradients."

    def test_detector_stays_frozen_after_backward(
        self, tiny_bn_pipeline: EmbeddingPipeline
    ) -> None:
        tiny_bn_pipeline.encoder.zero_grad()
        images = torch.randn(2, 3, 8, 8)
        labels = torch.zeros(2, dtype=torch.long)
        payload = torch.randint(0, 2, (2, PAYLOAD_BITS), dtype=torch.float32)

        result = tiny_bn_pipeline(images, labels, payload)
        CompositeLoss()(result).total.backward()

        for name, param in tiny_bn_pipeline.detector.named_parameters():
            assert param.grad is None, (
                f"Detector param '{name}' accumulated a gradient — must stay frozen."
            )
