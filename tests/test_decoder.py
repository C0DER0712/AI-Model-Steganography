import math

import pytest
import torch

from models.decoder import (
    DecoderConfig,
    build_decoder,
    bit_error_rate,
    decode,
    payload_reconstruction_accuracy,
    reconstruct_payload,
    sinusoidal_chunk_positions,
)
from utils.payload import generate_payload, payload_to_tensor


def _small_config() -> DecoderConfig:
    return DecoderConfig(
        base_channels=8,
        num_residual_blocks=2,
        attention_reduction=4,
        chunk_size=16,
        chunk_position_dim=8,
        hidden_dim=16,
    )


def test_decoder_forward_returns_chunked_logits_for_random_payload_size() -> None:
    decoder = build_decoder(_small_config())
    representation = torch.rand(2, 4, 8, 8)
    payload = torch.stack(
        [
            payload_to_tensor(generate_payload("128KB", seed=1)),
            payload_to_tensor(generate_payload("128KB", seed=2)),
        ]
    )

    logits = decoder(representation, num_bits=payload.shape[1])

    assert logits.shape == (
        2,
        math.ceil(payload.shape[1] / decoder.config.chunk_size),
        decoder.config.chunk_size,
    )


def test_shared_chunk_head_parameter_count_does_not_grow_with_payload_length() -> None:
    decoder = build_decoder(_small_config())
    parameter_count_before = sum(parameter.numel() for parameter in decoder.parameters())
    representation = torch.rand(1, 4, 8, 8)

    decoder(representation, num_bits=1024)
    decoder(representation, num_bits=(1024 * 1024 * 8) + 4096)

    parameter_count_after = sum(parameter.numel() for parameter in decoder.parameters())
    assert parameter_count_after == parameter_count_before


def test_decode_thresholds_and_trims_padded_chunk_bits() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, -1.0, 0.5, -0.5],
                [-2.0, 3.0, 4.0, -4.0],
            ]
        ]
    )

    bits = decode(logits, num_bits=6)

    assert bits.tolist() == [[1, 0, 1, 0, 0, 1]]


def test_reconstruct_payload_packs_decoded_bits() -> None:
    target_payload = bytes([0b10110010])
    target_bits = payload_to_tensor(target_payload)
    logits = torch.where(
        target_bits.reshape(1, 1, -1) == 1,
        torch.ones(1, 1, 8),
        -torch.ones(1, 1, 8),
    )

    assert reconstruct_payload(logits, num_bits=8) == [target_payload]


def test_decoder_reconstruct_payload_returns_bytes_for_random_payload_length() -> None:
    decoder = build_decoder(_small_config())
    representation = torch.rand(1, 4, 8, 8)

    payloads = decoder.reconstruct_payload(representation, num_bits=128 * 1024 * 8)

    assert len(payloads) == 1
    assert len(payloads[0]) == 128 * 1024


def test_bit_error_rate_and_payload_reconstruction_accuracy() -> None:
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


def test_sinusoidal_chunk_positions_are_dynamic_and_deterministic() -> None:
    first = sinusoidal_chunk_positions(num_chunks=3, dim=5)
    second = sinusoidal_chunk_positions(num_chunks=3, dim=5)

    assert first.shape == (3, 5)
    assert torch.equal(first, second)
