"""Scalable chunk-based payload decoder.

The decoder maps a modified Model-XRay weight representation back to benign
payload bits. It avoids a single huge output layer by reconstructing payloads
chunk by chunk with a shared prediction head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from utils.payload import tensor_to_payload


@dataclass(frozen=True)
class DecoderConfig:
    """Configuration for `ChunkedPayloadDecoder`.

    Attributes:
        input_channels: Number of input Model-XRay representation channels.
        base_channels: Width of the convolutional feature extractor.
        num_residual_blocks: Number of residual CNN blocks in the extractor.
        attention_reduction: Reduction ratio for channel attention.
        chunk_size: Number of payload bits reconstructed per shared-head call.
        chunk_position_dim: Size of sinusoidal chunk-position features.
        hidden_dim: Hidden width of the shared chunk reconstruction head.
        dropout: Dropout probability in the chunk reconstruction head.
    """

    input_channels: int = 4
    base_channels: int = 64
    num_residual_blocks: int = 4
    attention_reduction: int = 8
    chunk_size: int = 1024
    chunk_position_dim: int = 64
    hidden_dim: int = 256
    dropout: float = 0.0


class ConvNormActivation(nn.Module):
    """Convolution block: Conv2d -> GroupNorm -> GELU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2

        # Conv2d learns local structure in the four IEEE754 byte planes.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        # GroupNorm keeps behavior stable for small research batch sizes.
        self.norm = nn.GroupNorm(num_groups=_num_groups(out_channels), num_channels=out_channels)
        # GELU provides a smooth nonlinearity for payload evidence extraction.
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and activation."""

        return self.activation(self.norm(self.conv(inputs)))


class ChannelAttention(nn.Module):
    """Channel attention for emphasizing useful decoded signal features."""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)

        # AdaptiveAvgPool2d summarizes each channel across the full weight image.
        self.pool = nn.AdaptiveAvgPool2d(1)
        # First 1x1 convolution compresses global channel statistics.
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        # GELU models nonlinear channel relationships.
        self.activation = nn.GELU()
        # Second 1x1 convolution expands attention logits back to all channels.
        self.expand = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply learned channel attention."""

        attention = self.pool(features)
        attention = self.reduce(attention)
        attention = self.activation(attention)
        attention = torch.sigmoid(self.expand(attention))
        return features * attention


class ResidualDecoderBlock(nn.Module):
    """Residual CNN block for extracting payload-reconstruction evidence."""

    def __init__(self, channels: int, attention_reduction: int) -> None:
        super().__init__()

        # First convolution extracts local evidence from byte-plane features.
        self.conv1 = ConvNormActivation(channels, channels)
        # Second convolution refines evidence before residual addition.
        self.conv2 = ConvNormActivation(channels, channels)
        # ChannelAttention learns which feature channels matter for decoding.
        self.attention = ChannelAttention(channels, attention_reduction)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run one residual feature-extraction block."""

        residual = features
        features = self.conv1(features)
        features = self.conv2(features)
        features = self.attention(features)
        return residual + features


class ChunkHead(nn.Module):
    """Shared MLP head that predicts one payload chunk at a time."""

    def __init__(
        self,
        feature_dim: int,
        position_dim: int,
        hidden_dim: int,
        chunk_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # Linear layer fuses global image features with one chunk position.
        self.input_projection = nn.Linear(feature_dim + position_dim, hidden_dim)
        # GELU gives the shared head nonlinear reconstruction capacity.
        self.activation = nn.GELU()
        # Dropout optionally regularizes chunk predictions.
        self.dropout = nn.Dropout(dropout)
        # Final linear layer predicts logits for one fixed-size bit chunk.
        self.output_projection = nn.Linear(hidden_dim, chunk_size)

    def forward(
        self,
        global_features: torch.Tensor,
        chunk_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict chunk logits for every requested chunk position."""

        batch_size = global_features.shape[0]
        num_chunks = chunk_positions.shape[0]
        features = global_features.unsqueeze(1).expand(batch_size, num_chunks, -1)
        positions = chunk_positions.unsqueeze(0).expand(batch_size, num_chunks, -1)
        hidden = torch.cat([features, positions], dim=-1)
        hidden = self.input_projection(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return self.output_projection(hidden)


class ChunkedPayloadDecoder(nn.Module):
    """Decode payload bits from modified Model-XRay weight representations."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config

        # Stem projects four Model-XRay channels into decoder feature space.
        self.stem = ConvNormActivation(config.input_channels, config.base_channels)
        # Residual blocks extract evidence without tying parameters to payload length.
        self.blocks = nn.ModuleList(
            [
                ResidualDecoderBlock(
                    channels=config.base_channels,
                    attention_reduction=config.attention_reduction,
                )
                for _ in range(config.num_residual_blocks)
            ]
        )
        # AdaptiveAvgPool2d converts arbitrary spatial sizes to one feature vector.
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # ChunkHead is shared across all chunks, keeping parameter count scalable.
        self.chunk_head = ChunkHead(
            feature_dim=config.base_channels,
            position_dim=config.chunk_position_dim,
            hidden_dim=config.hidden_dim,
            chunk_size=config.chunk_size,
            dropout=config.dropout,
        )

    def forward(self, weight_representation: torch.Tensor, num_bits: int) -> torch.Tensor:
        """Return chunked payload logits.

        Args:
            weight_representation: Tensor with shape `(batch, 4, height, width)`.
            num_bits: Number of payload bits to reconstruct.

        Returns:
            Logits with shape `(batch, num_chunks, chunk_size)`. The final chunk
            may include padded logits beyond `num_bits`.

        Raises:
            ValueError: If inputs do not match the decoder configuration.
        """

        self._validate_inputs(weight_representation, num_bits)
        num_chunks = math.ceil(num_bits / self.config.chunk_size)

        features = self.stem(weight_representation.to(dtype=torch.float32))
        for block in self.blocks:
            features = block(features)

        global_features = self.global_pool(features).flatten(start_dim=1)
        positions = sinusoidal_chunk_positions(
            num_chunks=num_chunks,
            dim=self.config.chunk_position_dim,
            device=weight_representation.device,
        )
        return self.chunk_head(global_features, positions)

    @torch.no_grad()
    def decode(
        self,
        weight_representation: torch.Tensor,
        num_bits: int,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """Decode payload bits by thresholding reconstructed logits.

        Args:
            weight_representation: Tensor with shape `(batch, 4, height, width)`.
            num_bits: Number of real payload bits to return.
            threshold: Logit threshold used to map logits to binary bits.

        Returns:
            CPU `torch.uint8` tensor with shape `(batch, num_bits)`.
        """

        logits = self.forward(weight_representation, num_bits)
        bits = (logits.reshape(logits.shape[0], -1)[:, :num_bits] > threshold)
        return bits.to(dtype=torch.uint8).cpu()

    @torch.no_grad()
    def reconstruct_payload(
        self,
        weight_representation: torch.Tensor,
        num_bits: int,
        threshold: float = 0.0,
    ) -> list[bytes]:
        """Decode and pack payload bits back into byte strings.

        Args:
            weight_representation: Tensor with shape `(batch, 4, height, width)`.
            num_bits: Number of real payload bits to return. Must be divisible
                by 8 to pack complete bytes.
            threshold: Logit threshold used to map logits to binary bits.

        Returns:
            List of reconstructed payload byte strings, one per batch item.

        Raises:
            ValueError: If `num_bits` is not divisible by 8.
        """

        if num_bits % 8 != 0:
            raise ValueError("num_bits must be divisible by 8 to reconstruct bytes.")

        decoded = self.decode(weight_representation, num_bits, threshold=threshold)
        return [tensor_to_payload(bits) for bits in decoded]

    def _validate_inputs(self, weight_representation: torch.Tensor, num_bits: int) -> None:
        if weight_representation.ndim != 4:
            raise ValueError(
                "weight_representation must have shape "
                "(batch, channels, height, width)."
            )
        if weight_representation.shape[1] != self.config.input_channels:
            raise ValueError(
                f"Expected {self.config.input_channels} input channels, "
                f"got {weight_representation.shape[1]}."
            )
        if num_bits <= 0:
            raise ValueError("num_bits must be positive.")


def build_decoder(config: DecoderConfig | None = None) -> ChunkedPayloadDecoder:
    """Build a `ChunkedPayloadDecoder` from an optional configuration."""

    return ChunkedPayloadDecoder(config or DecoderConfig())


def decode(logits: torch.Tensor, num_bits: int, threshold: float = 0.0) -> torch.Tensor:
    """Threshold chunked logits into payload bits.

    Args:
        logits: Tensor with shape `(batch, num_chunks, chunk_size)`.
        num_bits: Number of real bits to keep after trimming padded chunk space.
        threshold: Logit threshold for binary reconstruction.

    Returns:
        CPU `torch.uint8` tensor with shape `(batch, num_bits)`.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape (batch, num_chunks, chunk_size).")
    flat = logits.reshape(logits.shape[0], -1)
    if num_bits <= 0 or num_bits > flat.shape[1]:
        raise ValueError("num_bits must be positive and fit within logits.")

    return (flat[:, :num_bits] > threshold).to(dtype=torch.uint8).cpu()


def reconstruct_payload(logits: torch.Tensor, num_bits: int, threshold: float = 0.0) -> list[bytes]:
    """Convert chunked logits into reconstructed payload bytes."""

    if num_bits % 8 != 0:
        raise ValueError("num_bits must be divisible by 8 to reconstruct bytes.")

    return [tensor_to_payload(bits) for bits in decode(logits, num_bits, threshold)]


def bit_error_rate(predicted_bits: torch.Tensor, target_bits: torch.Tensor) -> float:
    """Compute bit error rate between reconstructed and target payload bits."""

    predicted = predicted_bits.detach().cpu().reshape(-1).to(dtype=torch.uint8)
    target = target_bits.detach().cpu().reshape(-1).to(dtype=torch.uint8)
    if predicted.numel() != target.numel():
        raise ValueError("predicted_bits and target_bits must have the same length.")
    if predicted.numel() == 0:
        return 0.0

    errors = torch.count_nonzero(predicted != target).item()
    return errors / predicted.numel()


def payload_reconstruction_accuracy(
    predicted_bits: torch.Tensor,
    target_bits: torch.Tensor,
) -> float:
    """Compute payload reconstruction accuracy as `1 - bit_error_rate`."""

    return 1.0 - bit_error_rate(predicted_bits, target_bits)


def sinusoidal_chunk_positions(
    num_chunks: int,
    dim: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create deterministic sinusoidal chunk-position encodings."""

    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive.")
    if dim <= 0:
        raise ValueError("dim must be positive.")

    positions = torch.arange(num_chunks, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    encodings = torch.zeros(num_chunks, dim, device=device, dtype=torch.float32)
    encodings[:, 0::2] = torch.sin(positions * frequencies)
    if dim > 1:
        encodings[:, 1::2] = torch.cos(positions * frequencies[: encodings[:, 1::2].shape[1]])
    return encodings


def _num_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _validate_config(config: DecoderConfig) -> None:
    if config.input_channels <= 0:
        raise ValueError("input_channels must be positive.")
    if config.base_channels <= 0:
        raise ValueError("base_channels must be positive.")
    if config.num_residual_blocks < 0:
        raise ValueError("num_residual_blocks must be non-negative.")
    if config.attention_reduction <= 0:
        raise ValueError("attention_reduction must be positive.")
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if config.chunk_position_dim <= 0:
        raise ValueError("chunk_position_dim must be positive.")
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive.")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in the range [0.0, 1.0).")
