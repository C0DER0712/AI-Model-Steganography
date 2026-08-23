"""Tests for the v2 DensePayloadDecoder."""

import pytest
import torch

from models.decoder import (
    DecoderConfig,
    DensePayloadDecoder,
    build_decoder,
    bit_error_rate,
    decode,
    payload_reconstruction_accuracy,
    reconstruct_payload,
)
from utils.payload import generate_payload, payload_to_tensor


def _small_config() -> DecoderConfig:
    return DecoderConfig(
        base_channels=8,
        num_residual_blocks=2,
        attention_reduction=4,
    )


def test_decoder_forward_returns_flat_logits() -> None:
    """forward() returns (batch, grid_side*grid_side) logits."""
    import math
    decoder = build_decoder(_small_config())
    representation = torch.rand(2, 4, 32, 32)
    num_bits = 64

    logits = decoder(representation, num_bits=num_bits)

    grid_side = math.ceil(math.sqrt(num_bits))
    assert logits.shape == (2, grid_side * grid_side)


def test_decoder_forward_various_payload_sizes() -> None:
    """forward() works for several different num_bits values."""
    import math
    decoder = build_decoder(_small_config())
    representation = torch.rand(1, 4, 64, 64)

    for num_bits in [8, 64, 128, 1024]:
        logits = decoder(representation, num_bits=num_bits)
        grid_side = math.ceil(math.sqrt(num_bits))
        assert logits.shape == (1, grid_side * grid_side), f"Failed for num_bits={num_bits}"


def test_decode_thresholds_and_trims_to_num_bits() -> None:
    """decode() trims padded grid cells and thresholds correctly."""
    logits = torch.tensor([[2.0, -1.0, 0.5, -0.5, 3.0, -2.0]])  # 6 logits, num_bits=4

    bits = decode(logits, num_bits=4)

    assert bits.shape == (1, 4)
    assert bits.tolist() == [[1, 0, 1, 0]]


def test_reconstruct_payload_packs_decoded_bits() -> None:
    target_payload = bytes([0b10110010])
    target_bits = payload_to_tensor(target_payload)
    logits = torch.where(
        target_bits.reshape(1, -1) == 1,
        torch.ones(1, 8),
        -torch.ones(1, 8),
    )

    assert reconstruct_payload(logits, num_bits=8) == [target_payload]


def test_decoder_reconstruct_payload_method() -> None:
    decoder = build_decoder(_small_config())
    representation = torch.rand(1, 4, 64, 64)

    payloads = decoder.reconstruct_payload(representation, num_bits=128 * 8)

    assert len(payloads) == 1
    assert len(payloads[0]) == 128


def test_bit_error_rate_and_reconstruction_accuracy() -> None:
    target = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0], dtype=torch.uint8)
    predicted = torch.tensor([1, 1, 1, 0, 0, 0, 1, 0], dtype=torch.uint8)

    assert bit_error_rate(predicted, target) == pytest.approx(2 / 8)
    assert payload_reconstruction_accuracy(predicted, target) == pytest.approx(6 / 8)


def test_decoder_validates_inputs() -> None:
    decoder = build_decoder(_small_config())

    with pytest.raises(ValueError, match="weight_representation"):
        decoder(torch.rand(4, 8, 8), num_bits=8)

    with pytest.raises(ValueError, match="input channels"):
        decoder(torch.rand(1, 3, 8, 8), num_bits=8)

    with pytest.raises(ValueError, match="num_bits"):
        decoder(torch.rand(1, 4, 8, 8), num_bits=0)


def test_parameter_count_independent_of_num_bits() -> None:
    """DensePayloadDecoder parameter count must not grow with payload size."""
    decoder = build_decoder(_small_config())
    before = sum(p.numel() for p in decoder.parameters())

    decoder(torch.rand(1, 4, 32, 32), num_bits=64)
    decoder(torch.rand(1, 4, 32, 32), num_bits=4096)

    after = sum(p.numel() for p in decoder.parameters())
    assert after == before