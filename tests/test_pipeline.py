"""Tests for the end-to-end EmbeddingPipeline.

All tests use the ``"tiny"`` synthetic backbone (~1.7 K float params) so
that weight extraction, the 4-channel representation, and the functional
forward pass run in milliseconds without exhausting test-environment RAM.
"""

from __future__ import annotations

import pytest
import torch

from evaluation.differentiable_detector import DifferentiableDetectorConfig
from models.decoder import DecoderConfig
from models.encoder import EncoderConfig
from models.pipeline import EmbeddingPipeline, PipelineConfig, build_pipeline
from training.losses import LossInputs


# ---------------------------------------------------------------------------
# Small configuration — keeps the test suite fast
# ---------------------------------------------------------------------------

PAYLOAD_BITS = 64  # 8 bytes — small enough for the tiny backbone's weight image


def _tiny_enc() -> EncoderConfig:
    return EncoderConfig(payload_dim=PAYLOAD_BITS, base_channels=8, num_residual_blocks=1)


def _tiny_dec() -> DecoderConfig:
    return DecoderConfig(base_channels=8, num_residual_blocks=1, chunk_size=PAYLOAD_BITS)


def _tiny_det() -> DifferentiableDetectorConfig:
    return DifferentiableDetectorConfig(
        hpf_channels=4, base_channels=4, num_type1_blocks=1, num_type2_blocks=1, fc_hidden_dim=8
    )


@pytest.fixture
def small_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        host_model_name="tiny",
        host_model_num_classes=10,
        host_model_pretrained=False,
        payload_bits=PAYLOAD_BITS,
        encoder=_tiny_enc(),
        decoder=_tiny_dec(),
        detector=_tiny_det(),
    )


@pytest.fixture
def pipeline(small_pipeline_config: PipelineConfig) -> EmbeddingPipeline:
    return EmbeddingPipeline(small_pipeline_config)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestPipelineConstruction:
    def test_detector_is_frozen(self, pipeline: EmbeddingPipeline) -> None:
        assert pipeline.detector.is_frozen()

    def test_host_model_frozen_by_default(self, pipeline: EmbeddingPipeline) -> None:
        for param in pipeline.host_model.parameters():
            assert not param.requires_grad

    def test_encoder_trainable(self, pipeline: EmbeddingPipeline) -> None:
        assert any(p.requires_grad for p in pipeline.encoder.parameters())

    def test_decoder_trainable(self, pipeline: EmbeddingPipeline) -> None:
        assert any(p.requires_grad for p in pipeline.decoder.parameters())

    def test_host_model_trainable_when_configured(
        self, small_pipeline_config: PipelineConfig
    ) -> None:
        cfg = PipelineConfig(
            **{**small_pipeline_config.__dict__, "train_host_model": True}
        )
        pipe = EmbeddingPipeline(cfg)
        assert any(p.requires_grad for p in pipe.host_model.parameters())


# ---------------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_returns_loss_inputs(self, pipeline: EmbeddingPipeline) -> None:
        images = torch.randn(2, 3, 8, 8)
        labels = torch.zeros(2, dtype=torch.long)
        payload = torch.randint(0, 2, (2, PAYLOAD_BITS), dtype=torch.float32)
        result = pipeline(images, labels, payload)
        assert isinstance(result, LossInputs)

    def test_classification_logits_shape(self, pipeline: EmbeddingPipeline) -> None:
        images = torch.randn(2, 3, 8, 8)
        labels = torch.zeros(2, dtype=torch.long)
        payload = torch.randint(0, 2, (2, PAYLOAD_BITS), dtype=torch.float32)
        result = pipeline(images, labels, payload)
        assert result.classification_logits.shape == (2, 10)

    def test_payload_logits_shape(self, pipeline: EmbeddingPipeline) -> None:
        images = torch.randn(2, 3, 8, 8)
        labels = torch.zeros(2, dtype=torch.long)
        payload = torch.randint(0, 2, (2, PAYLOAD_BITS), dtype=torch.float32)
        result = pipeline(images, labels, payload)
        assert result.payload_logits.shape == (2, PAYLOAD_BITS)

    def test_detector_logits_shape(self, pipeline: EmbeddingPipeline) -> None:
        images = torch.randn(2, 3, 8, 8)
        labels = torch.zeros(2, dtype=torch.long)
        payload = torch.randint(0, 2, (2, PAYLOAD_BITS), dtype=torch.float32)
        result = pipeline(images, labels, payload)
        assert result.detector_logits.shape == (2,)

    def test_detector_targets_are_benign(self, pipeline: EmbeddingPipeline) -> None:
        """The encoder is trained to fool the detector toward label 0 (benign)."""
        images = torch.randn(1, 3, 8, 8)
        labels = torch.zeros(1, dtype=torch.long)
        payload = torch.randint(0, 2, (1, PAYLOAD_BITS), dtype=torch.float32)
        result = pipeline(images, labels, payload)
        assert torch.all(result.detector_targets == 0)

    def test_weight_representations_populated(self, pipeline: EmbeddingPipeline) -> None:
        images = torch.randn(1, 3, 8, 8)
        labels = torch.zeros(1, dtype=torch.long)
        payload = torch.randint(0, 2, (1, PAYLOAD_BITS), dtype=torch.float32)
        result = pipeline(images, labels, payload)
        assert result.modified_weights is not None
        assert result.original_weights is not None
        assert result.modified_weights.shape == result.original_weights.shape

    def test_shared_payload_broadcast(self, pipeline: EmbeddingPipeline) -> None:
        """A 1D payload should be broadcast across the batch dimension."""
        images = torch.randn(3, 3, 8, 8)
        labels = torch.zeros(3, dtype=torch.long)
        payload = torch.randint(0, 2, (PAYLOAD_BITS,), dtype=torch.float32)
        result = pipeline(images, labels, payload)
        assert result.payload_logits.shape[0] == 3


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_encoder_receives_composite_gradient(
        self, pipeline: EmbeddingPipeline
    ) -> None:
        from training.losses import CompositeLoss, LossWeights

        images = torch.randn(1, 3, 8, 8)
        labels = torch.zeros(1, dtype=torch.long)
        payload = torch.randint(0, 2, (1, PAYLOAD_BITS), dtype=torch.float32)

        loss_fn = CompositeLoss(
            LossWeights(classification=1.0, payload=1.0, distortion=1.0, detector=1.0)
        )
        loss_inputs = pipeline(images, labels, payload)
        loss_output = loss_fn(loss_inputs)
        loss_output.total.backward()

        enc_grads = [p.grad for p in pipeline.encoder.parameters() if p.grad is not None]
        assert len(enc_grads) > 0, "Encoder received no gradients."

    def test_detector_params_untouched_after_backward(
        self, pipeline: EmbeddingPipeline
    ) -> None:
        from training.losses import CompositeLoss

        images = torch.randn(1, 3, 8, 8)
        labels = torch.zeros(1, dtype=torch.long)
        payload = torch.randint(0, 2, (1, PAYLOAD_BITS), dtype=torch.float32)

        loss_fn = CompositeLoss()
        loss_inputs = pipeline(images, labels, payload)
        loss_output = loss_fn(loss_inputs)
        loss_output.total.backward()

        for name, param in pipeline.detector.named_parameters():
            assert param.grad is None, f"Detector param '{name}' accumulated a gradient."


# ---------------------------------------------------------------------------
# Encode / decode helper tests
# ---------------------------------------------------------------------------

class TestEncodeDecodeMethods:
    def test_encode_returns_correct_shapes(self, pipeline: EmbeddingPipeline) -> None:
        bits = torch.randint(0, 2, (PAYLOAD_BITS,), dtype=torch.float32)
        modified, original = pipeline.encode(bits)
        assert modified.shape == original.shape
        assert modified.ndim == 4
        assert modified.shape[1] == 4  # 4-channel representation

    def test_decode_returns_bit_tensor(self, pipeline: EmbeddingPipeline) -> None:
        bits = torch.randint(0, 2, (PAYLOAD_BITS,), dtype=torch.float32)
        modified, _ = pipeline.encode(bits)
        decoded = pipeline.decode(modified, PAYLOAD_BITS)
        assert decoded.shape == (1, PAYLOAD_BITS)
        assert decoded.dtype == torch.uint8
        assert torch.all((decoded == 0) | (decoded == 1))


# ---------------------------------------------------------------------------
# Builder function
# ---------------------------------------------------------------------------

class TestBuilderFunction:
    def test_build_pipeline_returns_embedding_pipeline(self) -> None:
        pipe = build_pipeline(
            PipelineConfig(
                host_model_name="tiny",
                host_model_num_classes=10,
                host_model_pretrained=False,
                payload_bits=PAYLOAD_BITS,
                encoder=_tiny_enc(),
                decoder=_tiny_dec(),
                detector=_tiny_det(),
            )
        )
        assert isinstance(pipe, EmbeddingPipeline)
        assert pipe.detector.is_frozen()
