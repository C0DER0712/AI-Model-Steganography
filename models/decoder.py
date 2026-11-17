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

from models.encoder import AdaptiveCapacityGate
from utils.representation import weights_to_float_image
from utils.payload import tensor_to_payload


@dataclass(frozen=True)
class DecoderConfig:
    """Configuration for `DensePayloadDecoder`.

    Attributes:
        input_channels: Decoder feature channels. Reference decoding uses one
            channel (modified minus clean representation). Blind decoding uses
            normalized modified weights and a local high-pass detail channel;
            the pipeline selects the input layout.
        base_channels: Width of the convolutional feature extractor.
        num_residual_blocks: Number of residual CNN blocks in the extractor.
        attention_reduction: Reduction ratio for channel attention.
        gradient_checkpointing: If true, wraps each residual block in
            `torch.utils.checkpoint.checkpoint` to cut peak activation
            memory at the cost of extra compute during backward. See
            `EncoderConfig.gradient_checkpointing` for rationale.
    """

    input_channels: int = 1
    base_channels: int = 64
    num_residual_blocks: int = 4
    attention_reduction: int = 8
    gradient_checkpointing: bool = False
    # Number of independent payload bits encoded per spatial position (weight
    # pixel). With reference decoding, the decoder sees modified minus clean
    # representation. With blind decoding, the reveal network must separate
    # payload evidence from the host-weight carrier itself. Higher values
    # increase the bit density in either mode and make recovery harder.
    # Requires encoder.bits_per_pixel to match this value exactly.
    bits_per_pixel: int = 1
    # Experimental blind decoder: local gate-biased cross-attention on the
    # payload grid. Queries are shared by bit plane, so params do not scale
    # with payload length. Its gate is estimated from modified weights only.
    gate_conditioned_local_attention: bool = False
    local_attention_dim: int = 16


class ConvNormActivation(nn.Module):
    """Convolution block: Conv2d -> GroupNorm -> GELU."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)

        # Conv2d learns local structure in the weight-image input features.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
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

    def __init__(
        self, channels: int, attention_reduction: int, dilation: int = 1
    ) -> None:
        super().__init__()

        # First convolution extracts local evidence from the decoder features.
        self.conv1 = ConvNormActivation(channels, channels, dilation=dilation)
        # Second convolution refines evidence before residual addition.
        self.conv2 = ConvNormActivation(channels, channels, dilation=dilation)
        # ChannelAttention learns which feature channels matter for decoding.
        self.attention = ChannelAttention(channels, attention_reduction)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run one residual feature-extraction block."""

        residual = features
        features = self.conv1(features)
        features = self.conv2(features)
        features = self.attention(features)
        return residual + features


class GateConditionedLocalAttention(nn.Module):
    """Attend over nearby payload-grid cells using a blind estimated gate.

    The module first compresses full-resolution CNN features to the payload
    grid, then each cell attends to its 3x3 neighborhood. Queries are shared
    across cells and learned per bit plane; there is no N-by-HW attention
    matrix or per-payload-bit parameter table.
    """

    def __init__(self, channels: int, bits_per_pixel: int, attention_dim: int):
        super().__init__()
        gate_width = max(8, channels // 4)
        self.capacity_gate = AdaptiveCapacityGate(
            input_channels=1, hidden_channels=gate_width
        )
        self.key = nn.Conv2d(channels, attention_dim, kernel_size=1, bias=False)
        self.value = nn.Conv2d(channels, attention_dim, kernel_size=1, bias=False)
        self.queries = nn.Parameter(torch.empty(bits_per_pixel, attention_dim))
        nn.init.normal_(self.queries, std=attention_dim ** -0.5)
        hidden = max(8, attention_dim)
        self.bit_head = nn.Sequential(
            nn.Linear(attention_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.attention_dim = attention_dim

    @staticmethod
    def _neighborhoods(features: torch.Tensor) -> torch.Tensor:
        """Return (batch, cells, 9, channels) 3x3 local feature tokens."""
        batch, channels, height, width = features.shape
        unfolded = F.unfold(features, kernel_size=3, padding=1)
        return (
            unfolded.reshape(batch, channels, 9, height * width)
            .permute(0, 3, 2, 1)
            .contiguous()
        )

    def forward(
        self,
        features: torch.Tensor,
        modified_weights: torch.Tensor,
        grid_side: int,
    ) -> torch.Tensor:
        batch = features.shape[0]
        grid_features = F.adaptive_avg_pool2d(
            features, output_size=(grid_side, grid_side)
        )
        key_tokens = self._neighborhoods(self.key(grid_features))
        value_tokens = self._neighborhoods(self.value(grid_features))

        # This mask is recomputed from the stego weights. The clean host and
        # encoder-side mask are never inputs to this blind decoder path.
        gate = self.capacity_gate(modified_weights.float())
        gate = F.adaptive_avg_pool2d(gate, output_size=(grid_side, grid_side))
        gate_tokens = self._neighborhoods(gate).squeeze(-1).clamp_min(1e-6)
        gate_bias = gate_tokens.log().unsqueeze(2)  # (B, cells, 1, 9)

        scale = self.attention_dim ** -0.5
        attention_logits = torch.einsum(
            "bcnd,pd->bcpn", key_tokens, self.queries
        ) * scale
        attention = torch.softmax(attention_logits + gate_bias, dim=-1)
        contexts = torch.einsum("bcpn,bcnd->bcpd", attention, value_tokens)
        logits = self.bit_head(contexts).squeeze(-1)
        return logits.reshape(batch, grid_side * grid_side * self.queries.shape[0])


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

        # Stem projects the single float channel into a learned feature space.
        self.stem = ConvNormActivation(config.input_channels, config.base_channels)
        # Residual blocks extract payload-reconstruction evidence.
        self.blocks = nn.ModuleList(
            [
                ResidualDecoderBlock(
                    channels=config.base_channels,
                    attention_reduction=config.attention_reduction,
                    # Blind decoding must distinguish payload edits from
                    # structured host-weight content. Exponentially spaced
                    # dilation adds broad context without extra parameters.
                    dilation=(min(2 ** index, 8) if config.input_channels == 2 else 1),
                )
                for index in range(config.num_residual_blocks)
            ]
        )
        self.local_attention = (
            GateConditionedLocalAttention(
                channels=config.base_channels,
                bits_per_pixel=config.bits_per_pixel,
                attention_dim=config.local_attention_dim,
            )
            if config.gate_conditioned_local_attention else None
        )
        # 1x1 conv down to bits_per_pixel channels: each channel is an
        # independent dense logit map for one bit-plane.  With bpp=1
        # (default) this is identical to the original single-channel design.
        # Higher bpp values let each spatial position carry multiple
        # independent bits. The encoder and decoder share the same bit-plane
        # layout in both reference and blind modes.
        self.logit_projection = nn.Conv2d(
            config.base_channels, config.bits_per_pixel, kernel_size=1
        )

    def forward(self, weight_representation: torch.Tensor, num_bits: int) -> torch.Tensor:
        """Reconstruct payload logits from a weight representation.

        Args:
            weight_representation: Tensor with shape `(batch, 1, height, width)`.
            num_bits: Number of real payload bits to reconstruct (before
                padding to a square grid).

        Returns:
            Tensor with shape `(batch, grid_side * grid_side)` — bit logits
            in row-major grid order, matching the encoder's bitmap layout.
            Callers slice to `[:, :num_bits]` to drop grid padding.
        """

        weight_representation = self._prepare_input(weight_representation)
        self._validate_inputs(weight_representation, num_bits)

        # Mirror the encoder guard: disable autocast so that the large
        # weight-image feature maps never run in float16.
        if torch.is_autocast_enabled():
            with torch.amp.autocast(device_type="cuda", enabled=False):
                return self.forward(weight_representation, num_bits)

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

        bpp = self.config.bits_per_pixel

        # Pool down to the per-bit-plane grid resolution.  For bpp=1 the
        # total payload bits equal grid_side².  For bpp>1, each grid cell
        # carries bpp bits, so grid_side = ceil(sqrt(num_bits / bpp)).
        grid_side = math.ceil(math.sqrt(num_bits / bpp))
        if self.local_attention is not None:
            logits = self.local_attention(
                features,
                weight_representation[:, :1],
                grid_side,
            )
            return logits.to(dtype=original_dtype)
        logit_map = self.logit_projection(features)  # (batch, bpp, H, W)
        pooled = F.adaptive_avg_pool2d(logit_map, output_size=(grid_side, grid_side))
        # pooled: (batch, bpp, grid_side, grid_side)
        # Interleave: each spatial cell's bpp logits are adjacent in the
        # output, matching the encoder's bitmap layout — see WeightPayloadEncoder.
        # (batch, bpp, grid_side, grid_side)
        #   -> (batch, grid_side, grid_side, bpp)   [move bpp to last]
        #   -> (batch, grid_side * grid_side * bpp) [flatten row-major]
        logits = pooled.permute(0, 2, 3, 1).reshape(pooled.shape[0], grid_side * grid_side * bpp)

        return logits.to(dtype=original_dtype) if original_dtype.is_floating_point else logits

    def _prepare_input(self, weight_representation: torch.Tensor) -> torch.Tensor:
        """Build blind-decoder features from a single normalized weight image.

        The two-channel blind input retains the normalized modified weights
        and adds a fixed 5x5 local-detail channel (weight minus local mean).
        This gives the reveal network both carrier context and a feature that
        emphasizes small local edits. It is not modified-minus-clean
        subtraction and requires no clean reference model.
        """
        if weight_representation.ndim != 4:
            return weight_representation
        expected = self.config.input_channels
        actual = weight_representation.shape[1]
        if actual == expected:
            return weight_representation
        if expected == 2 and actual == 1:
            local_mean = F.avg_pool2d(
                weight_representation.float(), kernel_size=5, stride=1,
                padding=2, count_include_pad=False,
            )
            local_detail = weight_representation.float() - local_mean
            return torch.cat(
                (weight_representation.float(), local_detail),
                dim=1,
            )
        return weight_representation

    @torch.no_grad()
    def decode_flat_weights(
        self,
        modified_weights: torch.Tensor,
        num_bits: int,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """Blindly decode directly from flattened modified model weights.

        ``modified_weights`` must contain only the learnable float32 weights
        in the same deterministic order used by ``extract_parameters``. No
        clean/reference checkpoint or reference normalization statistics are
        accepted or needed.
        """
        image, _ = weights_to_float_image(modified_weights)
        device = next(self.parameters()).device
        representation = torch.from_numpy(image).unsqueeze(0).to(device)
        return self.decode(representation, num_bits, threshold=threshold)

    @torch.no_grad()
    def decode(
        self,
        weight_representation: torch.Tensor,
        num_bits: int,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """Decode payload bits by thresholding reconstructed logits.

        Args:
            weight_representation: Tensor with shape `(batch, 1, height, width)`.
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
            weight_representation: Tensor with shape `(batch, 1, height, width)`.
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
    if config.local_attention_dim <= 0:
        raise ValueError("local_attention_dim must be positive.")
