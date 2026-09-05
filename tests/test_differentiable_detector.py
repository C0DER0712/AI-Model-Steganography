"""Tests for the differentiable frozen detector."""

from __future__ import annotations

import pytest
import torch

from evaluation.differentiable_detector import (
    DifferentiableDetector,
    DifferentiableDetectorConfig,
    build_differentiable_detector,
)


@pytest.fixture
def small_config() -> DifferentiableDetectorConfig:
    return DifferentiableDetectorConfig(
        input_channels=4,
        hpf_channels=8,
        base_channels=8,
        num_type1_blocks=1,
        num_type2_blocks=1,
        fc_hidden_dim=16,
        dropout=0.0,
    )


@pytest.fixture
def detector(small_config: DifferentiableDetectorConfig) -> DifferentiableDetector:
    return DifferentiableDetector(small_config)


class TestFrozenGuarantee:
    """The detector must never receive gradient updates."""

    def test_all_parameters_frozen_at_init(self, detector: DifferentiableDetector) -> None:
        for name, param in detector.named_parameters():
            assert not param.requires_grad, f"Parameter {name} is not frozen."

    def test_is_frozen_returns_true(self, detector: DifferentiableDetector) -> None:
        assert detector.is_frozen()

    def test_freeze_restores_frozen_state(self, detector: DifferentiableDetector) -> None:
        # Temporarily unfreeze, then re-freeze.
        for param in detector.parameters():
            param.requires_grad_(True)
        assert not detector.is_frozen()
        detector.freeze()
        assert detector.is_frozen()


class TestForwardPass:
    def test_output_shape_single_item(self, detector: DifferentiableDetector) -> None:
        x = torch.randn(1, 4, 32, 32) * 127 + 128  # simulate byte values
        logits = detector(x)
        assert logits.shape == (1, 1)

    def test_output_shape_batch(self, detector: DifferentiableDetector) -> None:
        x = torch.randn(4, 4, 64, 64) * 50 + 128
        logits = detector(x)
        assert logits.shape == (4, 1)

    def test_output_is_scalar_logit(self, detector: DifferentiableDetector) -> None:
        x = torch.zeros(1, 4, 32, 32)
        logits = detector(x)
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all()

    def test_variable_spatial_size(self, detector: DifferentiableDetector) -> None:
        for side in (16, 32, 64, 128):
            x = torch.randn(1, 4, side, side)
            logits = detector(x)
            assert logits.shape == (1, 1), f"Failed for spatial size {side}"


class TestDifferentiability:
    """Gradients must flow to the input (encoder output) despite frozen params."""

    def test_gradient_flows_to_input(self, detector: DifferentiableDetector) -> None:
        # Create a leaf tensor so .grad is populated after backward.
        x = (torch.randn(1, 4, 32, 32) * 50 + 128).detach().requires_grad_(True)
        logits = detector(x)
        loss = logits.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.all(x.grad == 0), "All input gradients are zero."

    def test_detector_params_have_no_grad_after_backward(
        self, detector: DifferentiableDetector
    ) -> None:
        x = (torch.randn(1, 4, 32, 32) * 50 + 128).detach().requires_grad_(True)
        logits = detector(x)
        logits.sum().backward()
        for name, param in detector.named_parameters():
            assert param.grad is None, (
                f"Parameter {name} accumulated a gradient — detector must stay frozen."
            )

    def test_encoder_receives_gradient_through_detector(
        self, small_config: DifferentiableDetectorConfig
    ) -> None:
        """Simulate the adversarial training scenario end-to-end.

        The encoder now operates in single-channel float weight space, while
        the detector still consumes the 4-channel IEEE754 byte image. The
        pipeline bridges the two with a straight-through estimator
        (``_WeightsToChannelsSTE``), so a detector-evasion gradient must still
        reach the encoder across that non-differentiable byte-packing step.
        """
        from models.encoder import EncoderConfig, WeightPayloadEncoder
        from models.pipeline import _WeightsToChannelsSTE

        enc_cfg = EncoderConfig(
            payload_dim=16,
            base_channels=8,
            num_residual_blocks=1,
        )
        encoder = WeightPayloadEncoder(enc_cfg)
        det = DifferentiableDetector(small_config)

        side = 32
        weight_repr = torch.randn(1, 1, side, side)  # single-channel float image
        payload = torch.randint(0, 2, (1, 16), dtype=torch.float32)

        modified, _gate = encoder(weight_repr, payload)  # (1, 1, side, side)
        # Bridge float weight space -> 4-channel detector bytes via the STE,
        # exactly as EmbeddingPipeline.forward does before the detector call.
        modified_flat = modified.reshape(1, side * side)
        det_image = _WeightsToChannelsSTE.apply(modified_flat, side)  # (1, 4, side, side)
        logits = det(det_image)
        benign_target = torch.zeros_like(logits)

        import torch.nn.functional as F
        loss = F.binary_cross_entropy_with_logits(logits, benign_target)
        loss.backward()

        # Encoder must have received gradients across the STE bridge.
        enc_grads = [p.grad for p in encoder.parameters() if p.grad is not None]
        assert len(enc_grads) > 0, "Encoder received no gradients from detector loss."


class TestBuilderFunction:
    def test_build_default(self) -> None:
        det = build_differentiable_detector()
        assert det.is_frozen()

    def test_build_with_config(self, small_config: DifferentiableDetectorConfig) -> None:
        det = build_differentiable_detector(small_config)
        assert det.config.base_channels == small_config.base_channels


class TestConfigValidation:
    def test_invalid_input_channels(self) -> None:
        with pytest.raises(ValueError, match="input_channels"):
            DifferentiableDetector(
                DifferentiableDetectorConfig(input_channels=0)
            )

    def test_invalid_dropout(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            DifferentiableDetector(
                DifferentiableDetectorConfig(dropout=1.5)
            )
