"""Dense fully-convolutional payload decoder.

DESIGN NOTE (architecture v2 — dense/fully-convolutional):
The previous version pooled each payload chunk's spatial region down to a
single averaged feature vector, then a shared `ChunkHead` MLP had to
reconstruct many independent bits (e.g. 64+) from that one vector. That is
a genuine information bottleneck: averaging over a region and then asking
an MLP to disentangle multiple unrelated bits from the result throws away
most of the spatial detail the encoder could have used, and caps
achievable accuracy regardless of training time (see models/encoder.py's
design note for the matching bottleneck on the encoder side, and the full
reasoning for why this task needs a Baluja-2017-style dense architecture
rather than a HiDDeN-style pooled/redundant one).

This version removes chunking, region pooling into shared vectors, and the
shared MLP head entirely. The decoder is a plain convolutional feature
extractor followed by a single 1x1 convolution down to ONE output channel
— a dense bit-logit map, exactly matching Baluja's reveal network shape.
That single-channel map is then mildly average-pooled down to the payload
grid resolution (the same grid the encoder used to lay out its bitmap),
which acts as light antialiasing over each bit's local pixel neighborhood
rather than aggregating many unrelated bits into one shared vector. No
chunk-position encoding is needed: spatial correspondence between a decoded
grid cell and its bit is now direct and positional, not something a shared
head has to reconstruct.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from utils.payload import tensor_to_payload


@dataclass(frozen=True)
class DecoderConfig:
    """Configuration for `DensePayloadDecoder`.

    Attributes:
        input_channels: Number of input Model-XRay representation channels.
        base_channels: Width of the convolutional feature extractor.
        num_residual_blocks: Number of residual CNN blocks in the extractor.
        attention_reduction: Reduction ratio for channel attention.
        gradient_checkpointing: If true, wraps each residual block in
            `torch.utils.checkpoint.checkpoint` to cut peak activation
            memory at the cost of extra compute during backward. See
            `EncoderConfig.gradient_checkpointing` for rationale.
    """

    input_channels: int = 4
    base_channels: int = 64
    num_residual_blocks: int = 4
    attention_reduction: int = 8
    gradient_checkpointing: bool = False


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


class DensePayloadDecoder(nn.Module):
    """Decode payload bits from modified Model-XRay weight representations.

    Fully convolutional, dense, per-bit prediction — see module docstring.
    Previously named `ChunkedPayloadDecoder`; renamed because there is no
    chunking left in this design.
    """

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config

        # Stem projects four byte channels into a learned feature space.
        self.stem = ConvNormActivation(config.input_channels, config.base_channels)
        # Residual blocks extract payload-reconstruction evidence.
        self.blocks = nn.ModuleList(
            [
                ResidualDecoderBlock(
                    channels=config.base_channels,
                    attention_reduction=config.attention_reduction,
                )
                for _ in range(config.num_residual_blocks)
            ]
        )
        # Single 1x1 conv down to ONE channel: a dense bit-logit map, the
        # same spatial size as the weight image. No shared MLP head, no
        # multi-bit-per-vector reconstruction — each spatial location's
        # pooled value (see forward()) directly IS one bit's logit.
        self.logit_projection = nn.Conv2d(config.base_channels, 1, kernel_size=1)

    def forward(self, weight_representation: torch.Tensor, num_bits: int) -> torch.Tensor:
        """Reconstruct payload logits from a weight representation.

        Args:
            weight_representation: Tensor with shape `(batch, 4, height, width)`.
            num_bits: Number of real payload bits to reconstruct (before
                padding to a square grid).

        Returns:
            Tensor with shape `(batch, grid_side * grid_side)` — bit logits
            in row-major grid order, matching the encoder's bitmap layout.
            Callers slice to `[:, :num_bits]` to drop grid padding.
        """

        self._validate_inputs(weight_representation, num_bits)
        original_dtype = weight_representation.dtype

        features = self.stem(weight_representation.to(dtype=torch.float32))

        use_checkpoint = self.config.gradient_checkpointing and self.training
        for block in self.blocks:
            if use_checkpoint:
                features = torch.utils.checkpoint.checkpoint(
                    block, features, use_reentrant=False
                )
            else:
                features = block(features)

        logit_map = self.logit_projection(features)  # (batch, 1, H, W)

        # Mild average-pool down to the payload grid resolution: this
        # antialiases over each bit's own local pixel neighborhood (the
        # weight image is naturally higher-resolution than the payload
        # grid — see encoder module docstring), NOT an aggregation of many
        # unrelated bits into a shared vector like the previous design.
        grid_side = math.ceil(math.sqrt(num_bits))
        pooled = F.adaptive_avg_pool2d(logit_map, output_size=(grid_side, grid_side))
        logits = pooled.reshape(pooled.shape[0], grid_side * grid_side)

        return logits.to(dtype=original_dtype) if original_dtype.is_floating_point else logits

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
        bits = logits[:, :num_bits] > threshold
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


def build_decoder(config: DecoderConfig | None = None) -> DensePayloadDecoder:
    """Build a `DensePayloadDecoder` from an optional configuration."""

    return DensePayloadDecoder(config or DecoderConfig())


def decode(logits: torch.Tensor, num_bits: int, threshold: float = 0.0) -> torch.Tensor:
    """Threshold dense logits into payload bits.

    Args:
        logits: Tensor with shape `(batch, grid_side * grid_side)`.
        num_bits: Number of real bits to keep after trimming grid padding.
        threshold: Logit threshold for binary reconstruction.

    Returns:
        CPU `torch.uint8` tensor with shape `(batch, num_bits)`.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, grid_side * grid_side).")
    if num_bits <= 0 or num_bits > logits.shape[1]:
        raise ValueError("num_bits must be positive and fit within logits.")

    return (logits[:, :num_bits] > threshold).to(dtype=torch.uint8).cpu()


def reconstruct_payload(logits: torch.Tensor, num_bits: int, threshold: float = 0.0) -> list[bytes]:
    """Convert dense logits into reconstructed payload bytes."""

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