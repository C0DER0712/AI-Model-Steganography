import pytest
import torch

from utils.payload import (
    SUPPORTED_PAYLOAD_SIZES,
    bit_error_rate,
    generate_payload,
    load_payload,
    payload_dataset,
    payload_to_tensor,
    save_payload,
    tensor_to_payload,
)


@pytest.mark.parametrize("label,num_bytes", SUPPORTED_PAYLOAD_SIZES.items())
def test_generate_payload_supports_required_sizes(label: str, num_bytes: int) -> None:
    payload = generate_payload(label, seed=7)

    assert len(payload) == num_bytes


def test_generate_payload_rejects_unsupported_sizes() -> None:
    with pytest.raises(ValueError, match="Unsupported payload size"):
        generate_payload("64KB")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported payload size"):
        generate_payload(64 * 1024)


def test_payload_tensor_round_trip_is_bit_exact() -> None:
    payload = generate_payload("128KB", seed=123)

    tensor = payload_to_tensor(payload)
    reconstructed = tensor_to_payload(tensor)

    assert tensor.dtype == torch.uint8
    assert tensor.shape == (len(payload) * 8,)
    assert reconstructed == payload
    assert bit_error_rate(payload, reconstructed) == 0.0


def test_tensor_to_payload_rejects_invalid_tensors() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        tensor_to_payload(torch.zeros((1, 8), dtype=torch.uint8))

    with pytest.raises(ValueError, match="divisible by 8"):
        tensor_to_payload(torch.zeros(7, dtype=torch.uint8))

    with pytest.raises(ValueError, match="binary"):
        tensor_to_payload(torch.tensor([0, 1, 2, 0, 0, 0, 0, 0], dtype=torch.uint8))


def test_payload_save_and_load_round_trip(tmp_path) -> None:
    payload = generate_payload("128KB", seed=42)
    path = save_payload(payload, tmp_path / "payload.bin")

    assert load_payload(path) == payload


def test_payload_dataset_returns_reproducible_bit_tensors() -> None:
    first = payload_dataset(2, "128KB", seed=99)
    second = payload_dataset(2, "128KB", seed=99)

    assert len(first) == 2
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert not torch.equal(first[0], first[1])


def test_payload_dataset_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="count must be non-negative"):
        payload_dataset(-1, "128KB")


def test_bit_error_rate_measures_bit_integrity() -> None:
    original = bytes([0b10101010, 0b11110000])
    recovered = bytes([0b10101011, 0b11110010])

    assert bit_error_rate(original, recovered) == pytest.approx(2 / 16)


def test_bit_error_rate_rejects_size_mismatch() -> None:
    with pytest.raises(ValueError, match="same number of bits"):
        bit_error_rate(bytes([0]), bytes([0, 1]))
