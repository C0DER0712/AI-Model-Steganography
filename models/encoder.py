"""Configurable encoder for Model-XRay weight representations.

DESIGN NOTE (architecture v2 — dense/fully-convolutional):
The previous version chunked the payload into fixed-size groups, projected
each chunk through one shared linear layer into a single embedding, then
broadcast that ONE embedding uniformly across an entire spatial region via
FiLM. That forces many independent bits (e.g. 64+ per chunk) to share one
bottleneck vector on the way in, mirrored by a matching bottleneck on the
decoder side (see models/decoder.py's design note). Empirically this capped
payload-recovery accuracy well below 100% (~66-76% observed at 128KB-1MB
scale) no matter how long training ran or how the chunk/region-pooling
parameters were tuned — it's a structural ceiling, not a training issue.

This version follows Baluja et al. 2017 ("Hiding Images in Plain Sight")
rather than HiDDeN (Zhu et al. 2018): HiDDeN is built for tiny payloads
(30-100 bits) that must survive lossy image transforms, and achieves
robustness by spreading one small message redundantly across an entire
image, decoded via global pooling. That is the opposite of what a
1MB-scale, no-lossy-channel payload needs. Baluja's reveal network is
fully convolutional end-to-end with no pooling-to-a-vector step anywhere —
every output element is predicted from its own local receptive field,
never mixed with unrelated message content through a shared bottleneck.

Concretely: the payload is reshaped into a 2D bitmap, given local context
by a small conv stack (`MessagePreparationNetwork`), then upsampled
(nearest, to preserve hard bit boundaries) to the weight image's native
resolution. Every residual block is modulated by this per-pixel feature
map via `SpatialFiLM` — a 1x1 conv predicting scale/shift AT EACH
LOCATION, never a single pooled vector broadcast everywhere. No chunk
grouping, no shared per-chunk projection, no chunk-position encoding is
needed anymore: spatial correspondence between a bit and its evidence is
now direct and structural, not something a shared MLP has to reconstruct.
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
        message_channels: Width of the message-preparation conv stack that
            processes the payload bitmap before it modulates the weight
            features. Independent of `base_channels` — this only needs
            enough capacity to give each bit useful local context, not to
            carry the whole weight-image feature representation.
        message_prep_layers: Number of conv layers in the message-preparation
            stack (minimum 1, which is just the input stem).
        attention_reduction: Reduction ratio in channel-attention MLPs.
        max_delta: Scale applied to the predicted residual update.
        gradient_checkpointing: If true, wraps each residual block in
            `torch.utils.checkpoint.checkpoint`, trading extra compute
            (recomputing activations during backward) for a large cut in
            peak activation memory. Since every layer here operates on the
            full weight-image resolution, activation memory dominates GPU
            usage — this is the single most effective memory lever short of
            shrinking the model itself, and preserves capacity/quality
            unlike reducing `base_channels`. Recommended on GPUs under
            ~8-12GB VRAM.
    """

    input_channels: int = 4
    output_channels: int = 4
    payload_dim: int = 1024
    base_channels: int = 64
    num_residual_blocks: int = 4
    message_channels: int = 32
    message_prep_layers: int = 2
    attention_reduction: int = 8
    max_delta: float = 1.0
    gradient_checkpointing: bool = False


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


class MessagePreparationNetwork(nn.Module):
    """Gives each payload bit local spatial context before it modulates features.

    Takes the payload reshaped as a 2D bitmap (one bit per grid cell) and
    processes it through a small conv stack, entirely independently per
    location aside from the receptive field grown by the conv kernels
    themselves — there is no pooling, flattening, or shared-vector step
    here. Output has the SAME spatial layout as the input bitmap (one
    feature vector per bit, at that bit's own grid location), just enriched
    with local context. This mirrors the "prep network" in Baluja et al.
    2017, which processes the secret image with conv layers before it ever
    touches the cover image.
    """

    def __init__(self, message_channels: int, num_layers: int) -> None:
        super().__init__()
        num_layers = max(1, num_layers)

        layers = [ConvNormActivation(1, message_channels)]
        layers.extend(
            ConvNormActivation(message_channels, message_channels)
            for _ in range(num_layers - 1)
        )
        self.net = nn.Sequential(*layers)

    def forward(self, bitmap: torch.Tensor) -> torch.Tensor:
        """Process a payload bitmap with shape `(batch, 1, grid, grid)`."""

        return self.net(bitmap)


class SpatialFiLM(nn.Module):
    """Feature-wise linear modulation predicted independently AT EACH PIXEL.

    Unlike the previous per-chunk FiLM (one embedding broadcast uniformly
    across an entire region, shared by every bit in that chunk), this
    predicts scale and shift from the message feature map at every spatial
    location via a 1x1 convolution — every bit gets its own modulation,
    never shared with unrelated bits. The message feature map passed in is
    already upsampled to the weight representation's resolution (see
    `WeightPayloadEncoder.forward`), so this is a genuinely local,
    per-position operation, not a global broadcast.
    """

    def __init__(self, message_channels: int, channels: int) -> None:
        super().__init__()

        # 1x1 convolution predicts per-pixel scale and bias from local
        # message features — no spatial mixing here, kernel_size=1 keeps
        # each output location dependent only on that same location's
        # message evidence, preserving the one-bit-one-location guarantee.
        self.modulation = nn.Conv2d(message_channels, channels * 2, kernel_size=1)

    def forward(self, features: torch.Tensor, message_features: torch.Tensor) -> torch.Tensor:
        """Apply per-pixel scale and shift.

        Args:
            features: Shape `(batch, channels, height, width)`.
            message_features: Shape `(batch, message_channels, height, width)`,
                already upsampled to match `features`' spatial size.
        """

        gamma_beta = self.modulation(message_features)
        gamma, beta = gamma_beta.chunk(2, dim=1)
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
    """Residual CNN block conditioned on per-pixel message features with attention."""

    def __init__(self, channels: int, message_channels: int, attention_reduction: int) -> None:
        super().__init__()

        # First convolution extracts local byte-plane features.
        self.conv1 = ConvNormActivation(channels, channels)
        # SpatialFiLM injects per-pixel payload information at this depth —
        # re-injecting at every block (not just once at the input) mirrors
        # HiDDeN's finding that repeated conditioning at multiple depths
        # improves recoverability, applied here per-pixel rather than
        # per-chunk-vector.
        self.film = SpatialFiLM(message_channels, channels)
        # Second convolution refines features before residual addition.
        self.conv2 = ConvNormActivation(channels, channels)
        # Attention learns useful channels and positions for representation edits.
        self.attention = ChannelSpatialAttention(channels, attention_reduction)

    def forward(
        self,
        features: torch.Tensor,
        message_features: torch.Tensor,
    ) -> torch.Tensor:
        """Run a payload-conditioned residual update."""

        residual = features
        features = self.conv1(features)
        features = self.film(features, message_features)
        features = self.conv2(features)
        features = self.attention(features)
        return residual + features


class WeightPayloadEncoder(nn.Module):
    """Encoder that maps weights plus payload to a modified weight representation."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.grid_side = math.ceil(math.sqrt(config.payload_dim))
        self.padded_dim = self.grid_side * self.grid_side

        # MessagePreparationNetwork gives every bit local context, keeping
        # its own dedicated grid location throughout (no pooling/flattening).
        self.message_prep = MessagePreparationNetwork(
            message_channels=config.message_channels,
            num_layers=config.message_prep_layers,
        )
        # Stem projects four byte channels into a learned feature space.
        self.stem = ConvNormActivation(config.input_channels, config.base_channels)
        # Residual blocks learn content-aware, payload-conditioned edit features.
        self.blocks = nn.ModuleList(
            [
                PayloadConditionedResidualBlock(
                    channels=config.base_channels,
                    message_channels=config.message_channels,
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
        weight_representation = weight_representation.to(dtype=torch.float32)
        _, _, height, width = weight_representation.shape

        # Reshape payload into a 2D bitmap: one bit per grid cell, zero-padded
        # to a perfect square grid. This is the ONLY reshape the payload ever
        # undergoes — every subsequent step preserves this spatial layout.
        payload = payload.to(dtype=torch.float32)
        if self.padded_dim != self.config.payload_dim:
            payload = F.pad(payload, (0, self.padded_dim - self.config.payload_dim))
        bitmap = payload.view(payload.shape[0], 1, self.grid_side, self.grid_side)

        message_features = self.message_prep(bitmap)
        # Nearest upsampling (not bilinear) preserves hard bit boundaries —
        # blending two different bits' feature values together at the
        # boundary would reintroduce exactly the kind of cross-bit mixing
        # this architecture is designed to avoid.
        message_features = F.interpolate(
            message_features, size=(height, width), mode="nearest"
        )

        features = self.stem(weight_representation)

        use_checkpoint = self.config.gradient_checkpointing and self.training
        for block in self.blocks:
            if use_checkpoint:
                features = torch.utils.checkpoint.checkpoint(
                    block, features, message_features, use_reentrant=False
                )
            else:
                features = block(features, message_features)

        delta = torch.tanh(self.output_projection(features)) * self.config.max_delta
        output = weight_representation + delta
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
    if config.message_channels <= 0:
        raise ValueError("message_channels must be positive.")
    if config.message_prep_layers <= 0:
        raise ValueError("message_prep_layers must be positive.")
    if config.attention_reduction <= 0:
        raise ValueError("attention_reduction must be positive.")
    if config.max_delta <= 0:
        raise ValueError("max_delta must be positive.")