import numpy as np
import torch

from utils.representation import (
    channels_to_weights,
    visualize_channels,
    weights_to_channels,
)


def _bits(values: torch.Tensor) -> torch.Tensor:
    return values.detach().cpu().contiguous().view(torch.int32)


def test_weights_to_channels_uses_ieee754_most_significant_byte_first() -> None:
    weights = torch.tensor([1.0, -2.5], dtype=torch.float32)

    channels = weights_to_channels(weights)

    assert channels.shape == (4, 2, 2)
    assert channels[:, 0, 0].tolist() == [0x3F, 0x80, 0x00, 0x00]
    assert channels[:, 0, 1].tolist() == [0xC0, 0x20, 0x00, 0x00]
    assert channels[:, 1, 0].tolist() == [0x00, 0x00, 0x00, 0x00]
    assert channels.dtype == np.uint8


def test_channels_to_weights_reconstructs_bits_exactly() -> None:
    weights = torch.tensor(
        [
            0.0,
            -0.0,
            1.0,
            -2.5,
            torch.finfo(torch.float32).tiny,
            torch.finfo(torch.float32).max,
            float("inf"),
            float("-inf"),
        ],
        dtype=torch.float32,
    )

    channels = weights_to_channels(weights)
    reconstructed = channels_to_weights(channels, num_values=weights.numel())

    assert torch.equal(_bits(reconstructed), _bits(weights))


def test_channels_round_trip_preserves_flattened_weight_order() -> None:
    weights = torch.arange(17, dtype=torch.float32).reshape(1, 17)

    channels = weights_to_channels(weights)
    reconstructed = channels_to_weights(channels, num_values=weights.numel())

    assert torch.equal(reconstructed, weights.reshape(-1))


def test_channels_to_weights_validates_shape_and_count() -> None:
    bad_channels = np.zeros((3, 2, 2), dtype=np.uint8)
    good_channels = np.zeros((4, 2, 2), dtype=np.uint8)

    try:
        channels_to_weights(bad_channels)
    except ValueError as exc:
        assert "(4, height, width)" in str(exc)
    else:
        raise AssertionError("channels_to_weights should reject non-four-channel input")

    try:
        channels_to_weights(good_channels, num_values=5)
    except ValueError as exc:
        assert "exceeds" in str(exc)
    else:
        raise AssertionError("channels_to_weights should reject oversized num_values")


def test_visualize_channels_returns_figure_and_writes_file(tmp_path) -> None:
    channels = weights_to_channels(torch.arange(4, dtype=torch.float32))
    output_path = tmp_path / "channels.png"

    figure = visualize_channels(channels, output_path=output_path, title="GF")

    assert output_path.is_file()
    assert len(figure.axes) == 4
