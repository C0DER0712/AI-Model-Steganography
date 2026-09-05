import pytest
import torch

from models.encoder import EncoderConfig, WeightPayloadEncoder, build_encoder


def test_encoder_returns_modified_representation_with_configured_shape() -> None:
    config = EncoderConfig(
        payload_dim=16,
        base_channels=8,
        num_residual_blocks=2,
        attention_reduction=4,
        max_delta=0.25,
    )
    encoder = WeightPayloadEncoder(config)
    weights = torch.rand(2, 1, 8, 8)
    payload = torch.randint(0, 2, (2, 16), dtype=torch.uint8)

    output, gate = encoder(weights, payload)

    assert output.shape == weights.shape
    assert output.dtype == weights.dtype
    assert gate is None
    assert torch.max(torch.abs(output - weights)) <= config.max_delta + 1e-6


def test_encoder_is_differentiable() -> None:
    encoder = build_encoder(
        EncoderConfig(
            payload_dim=8,
            base_channels=4,
            num_residual_blocks=1,
            attention_reduction=2,
        )
    )
    weights = torch.rand(1, 1, 4, 4, requires_grad=True)
    payload = torch.randint(0, 2, (1, 8), dtype=torch.float32)

    output, _ = encoder(weights, payload)
    loss = output.mean()
    loss.backward()

    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()


def test_encoder_validates_input_shapes() -> None:
    encoder = build_encoder(EncoderConfig(payload_dim=8, base_channels=4))

    with pytest.raises(ValueError, match="weight_representation"):
        encoder(torch.rand(4, 8, 8), torch.rand(1, 8))

    with pytest.raises(ValueError, match="input channels"):
        encoder(torch.rand(1, 3, 8, 8), torch.rand(1, 8))

    with pytest.raises(ValueError, match="payload_dim"):
        encoder(torch.rand(1, 1, 8, 8), torch.rand(1, 7))
