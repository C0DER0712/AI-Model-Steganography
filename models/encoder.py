"""Configurable encoder for Model-XRay weight representations.

The encoder receives a four-channel Model-XRay-style weight image and a benign
payload tensor, then predicts a continuous residual update to the weight image.
It does not perform handcrafted bit or LSB replacement; all modifications are
learned through convolution, payload conditioning, and attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for `WeightPayloadEncoder`.

    Attributes:
        input_channels: Number of input weight-representation channels. Model
            X-Ray GF representations use four channels: p0, p1, p2, p3.
        output_channels: Number of output representation channels.
        payload_dim: Number of payload tensor elements expected per sample.
        base_channels: Width of the convolutional feature backbone.
        num_residual_blocks: Number of payload-conditioned residual CNN blocks.
        payload_embedding_dim: Hidden size used to encode payload tensors.
        payload_chunk_size: Chunk width used by the payload embedding's shared
            linear layer. Keeping this fixed (independent of `payload_dim`)
            is what keeps parameter count from scaling with payload size —
            see `PayloadEncoder` for details.
        attention_reduction: Reduction ratio in channel-attention MLPs.
        dropout: Dropout probability applied to payload embeddings.
        max_delta: Scale applied to the predicted residual update.
    """

    input_channels: int = 4
    output_channels: int = 4
    payload_dim: int = 1024
    base_channels: int = 64
    num_residual_blocks: int = 4
    payload_embedding_dim: int = 256
    payload_chunk_size: int = 1024
    attention_reduction: int = 8
    dropout: float = 0.0
    max_delta: float = 1.0


class ConvNormActivation(nn.Module):
    """Convolution block: Conv2d -> GroupNorm -> GELU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2

        # Conv2d learns local correlations across byte-plane neighborhoods.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        # GroupNorm is stable for small batches common in large weight images.
        self.norm = nn.GroupNorm(num_groups=_num_groups(out_channels), num_channels=out_channels)
        # GELU provides a smooth nonlinearity for continuous residual prediction.
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and activation."""

        return self.activation(self.norm(self.conv(inputs)))


class PayloadEncoder(nn.Module):
    """MLP that converts a payload bit tensor into a dense conditioning vector.

    A naive `nn.Linear(payload_dim, embedding_dim)` scales with payload size —
    for a 128KB payload (1,048,576 bits) at embedding_dim=256 that alone is
    ~268M parameters, dwarfing everything else in the model.

    Instead, the payload is split into fixed-size chunks and a single shared
    linear layer embeds every chunk with the *same* weights (parameter count
    depends only on `chunk_size`, never on `payload_dim`). Chunk embeddings
    are then mean-pooled into one global conditioning vector. This mirrors
    the decoder's `ChunkHead`, which is shared across chunks by design.
    """

    def __init__(
        self,
        payload_dim: int,
        embedding_dim: int,
        dropout: float,
        chunk_size: int = 1024,
    ) -> None:
        super().__init__()

        self.payload_dim = payload_dim
        self.chunk_size = min(chunk_size, payload_dim)
        self.num_chunks = math.ceil(payload_dim / self.chunk_size)
        self.padded_dim = self.num_chunks * self.chunk_size

        # Shared linear layer embeds every payload chunk with identical
        # weights — this is what keeps the parameter count independent of
        # payload_dim, unlike a single dense payload_dim x embedding_dim layer.
        self.chunk_projection = nn.Linear(self.chunk_size, embedding_dim)
        # GELU keeps the payload pathway differentiable and expressive.
        self.activation = nn.GELU()
        # Dropout optionally regularizes payload conditioning.
        self.dropout = nn.Dropout(dropout)
        # Linear layer refines the pooled payload embedding used by FiLM blocks.
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, payload: torch.Tensor) -> torch.Tensor:
        """Encode payload tensors with shape `(batch, payload_dim)`."""

        payload = payload.to(dtype=torch.float32)
        if self.padded_dim != self.payload_dim:
            payload = F.pad(payload, (0, self.padded_dim - self.payload_dim))

        # (batch, payload_dim) -> (batch, num_chunks, chunk_size)
        chunks = payload.view(payload.shape[0], self.num_chunks, self.chunk_size)

        # Shared linear layer applied identically to every chunk, then
        # mean-pooled into one global conditioning vector.
        chunk_embeddings = self.activation(self.chunk_projection(chunks))
        pooled = chunk_embeddings.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.output_projection(pooled)


class FiLM(nn.Module):
    """Feature-wise linear modulation generated from a payload embedding."""

    def __init__(self, embedding_dim: int, channels: int) -> None:
        super().__init__()

        # Linear layer predicts per-channel scale and bias from the payload.
        self.modulation = nn.Linear(embedding_dim, channels * 2)

    def forward(self, features: torch.Tensor, payload_embedding: torch.Tensor) -> torch.Tensor:
        """Apply payload-conditioned scale and shift to feature maps."""

        gamma, beta = self.modulation(payload_embedding).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return features * (1.0 + gamma) + beta


class ChannelSpatialAttention(nn.Module):
    """Attention block that learns which channels and spatial locations to edit."""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)

        # AdaptiveAvgPool2d summarizes each feature channel globally.
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        # First 1x1 convolution compresses channel statistics.
        self.channel_reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        # GELU models nonlinear channel interactions.
        self.channel_activation = nn.GELU()
        # Second 1x1 convolution predicts channel attention logits.
        self.channel_expand = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        # 7x7 convolution predicts spatial attention from average and max maps.
        self.spatial_projection = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply channel attention followed by spatial attention."""

        channel_attention = self.channel_pool(features)
        channel_attention = self.channel_reduce(channel_attention)
        channel_attention = self.channel_activation(channel_attention)
        channel_attention = torch.sigmoid(self.channel_expand(channel_attention))
        features = features * channel_attention

        average_map = torch.mean(features, dim=1, keepdim=True)
        maximum_map = torch.amax(features, dim=1, keepdim=True)
        spatial_attention = torch.sigmoid(
            self.spatial_projection(torch.cat([average_map, maximum_map], dim=1))
        )
        return features * spatial_attention


class PayloadConditionedResidualBlock(nn.Module):
    """Residual CNN block conditioned on payload embeddings with attention."""

    def __init__(self, channels: int, embedding_dim: int, attention_reduction: int) -> None:
        super().__init__()

        # First convolution extracts local byte-plane features.
        self.conv1 = ConvNormActivation(channels, channels)
        # Second convolution refines features before residual addition.
        self.conv2 = ConvNormActivation(channels, channels)
        # FiLM injects payload information without fixed bit-placement rules.
        self.film = FiLM(embedding_dim, channels)
        # Attention learns useful channels and positions for representation edits.
        self.attention = ChannelSpatialAttention(channels, attention_reduction)

    def forward(
        self,
        features: torch.Tensor,
        payload_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Run a payload-conditioned residual update."""

        residual = features
        features = self.conv1(features)
        features = self.film(features, payload_embedding)
        features = self.conv2(features)
        features = self.attention(features)
        return residual + features


class WeightPayloadEncoder(nn.Module):
    """Encoder that maps weights plus payload to a modified weight representation."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config

        # PayloadEncoder turns random payload bits into global conditioning.
        self.payload_encoder = PayloadEncoder(
            payload_dim=config.payload_dim,
            embedding_dim=config.payload_embedding_dim,
            dropout=config.dropout,
            chunk_size=config.payload_chunk_size,
        )
        # Stem projects four byte channels into a learned feature space.
        self.stem = ConvNormActivation(config.input_channels, config.base_channels)
        # Residual blocks learn content-aware, payload-conditioned edit features.
        self.blocks = nn.ModuleList(
            [
                PayloadConditionedResidualBlock(
                    channels=config.base_channels,
                    embedding_dim=config.payload_embedding_dim,
                    attention_reduction=config.attention_reduction,
                )
                for _ in range(config.num_residual_blocks)
            ]
        )
        # Head maps features back to the four-channel representation domain.
        self.output_projection = nn.Conv2d(
            config.base_channels,
            config.output_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, weight_representation: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        """Produce a modified weight representation.

        Args:
            weight_representation: Tensor with shape `(batch, 4, height, width)`.
            payload: Tensor with shape `(batch, payload_dim)`.

        Returns:
            Tensor with the same spatial shape as `weight_representation` and
            `config.output_channels` channels.

        Raises:
            ValueError: If input shapes do not match the encoder configuration.
        """

        self._validate_inputs(weight_representation, payload)

        original_dtype = weight_representation.dtype
        features = self.stem(weight_representation.to(dtype=torch.float32))
        payload_embedding = self.payload_encoder(payload)

        for block in self.blocks:
            features = block(features, payload_embedding)

        delta = torch.tanh(self.output_projection(features)) * self.config.max_delta
        output = weight_representation.to(dtype=torch.float32) + delta
        return output.to(dtype=original_dtype) if original_dtype.is_floating_point else output

    def _validate_inputs(
        self,
        weight_representation: torch.Tensor,
        payload: torch.Tensor,
    ) -> None:
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
        if payload.ndim != 2:
            raise ValueError("payload must have shape (batch, payload_dim).")
        if payload.shape[0] != weight_representation.shape[0]:
            raise ValueError("payload batch size must match weight batch size.")
        if payload.shape[1] != self.config.payload_dim:
            raise ValueError(
                f"Expected payload_dim={self.config.payload_dim}, "
                f"got {payload.shape[1]}."
            )


def build_encoder(config: EncoderConfig | None = None) -> WeightPayloadEncoder:
    """Build a `WeightPayloadEncoder` from an optional configuration."""

    return WeightPayloadEncoder(config or EncoderConfig())


def _num_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _validate_config(config: EncoderConfig) -> None:
    if config.input_channels <= 0:
        raise ValueError("input_channels must be positive.")
    if config.output_channels <= 0:
        raise ValueError("output_channels must be positive.")
    if config.payload_dim <= 0:
        raise ValueError("payload_dim must be positive.")
    if config.base_channels <= 0:
        raise ValueError("base_channels must be positive.")
    if config.num_residual_blocks < 0:
        raise ValueError("num_residual_blocks must be non-negative.")
    if config.payload_embedding_dim <= 0:
        raise ValueError("payload_embedding_dim must be positive.")
    if config.payload_chunk_size <= 0:
        raise ValueError("payload_chunk_size must be positive.")
    if config.attention_reduction <= 0:
        raise ValueError("attention_reduction must be positive.")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in the range [0.0, 1.0).")
    if config.max_delta <= 0:
        raise ValueError("max_delta must be positive.")
