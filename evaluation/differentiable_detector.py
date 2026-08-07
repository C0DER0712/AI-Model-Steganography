"""Differentiable frozen detector for adversarial steganography training.

Implements an SRNet-inspired binary classifier that operates on the four-channel
Model X-Ray weight representation.  All parameters are frozen immediately after
construction; the module never receives gradient updates.  However, because the
*inputs* to the detector (produced by the encoder) still carry ``requires_grad``,
gradients from the detector loss flow back through the network's activations to
the encoder, enabling end-to-end adversarial training without modifying the
detector weights.

Architecture (SRNet-inspired, adapted for 4-channel weight images):
  - Normalization + preprocessing high-pass filter (fixed Laplacian kernels)
  - Truncation / clamp nonlinearity
  - Four Type-I residual blocks (constant spatial resolution)
  - Three Type-II strided blocks (spatial downsampling by 2 each)
  - Global average pooling
  - Fully-connected binary classification head

Reference: Fridrich & Kodovsky, "Rich Models for Steganalysis of Digital
Images," IEEE Trans. Info. Forensics and Security, 2012.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DifferentiableDetectorConfig:
    """Configuration for :class:`DifferentiableDetector`.

    Attributes:
        input_channels: Number of input representation channels (default 4).
        hpf_channels: Number of high-pass filter output channels.
        base_channels: Feature width for residual backbone.
        num_type1_blocks: Number of residual blocks at full resolution.
        num_type2_blocks: Number of strided downsampling blocks.
        truncation_threshold: Clamping bound applied after HPF (tanh scale).
        fc_hidden_dim: Hidden width of the classification head.
        dropout: Dropout probability in the classification head.
    """

    input_channels: int = 4
    hpf_channels: int = 30
    base_channels: int = 32
    num_type1_blocks: int = 4
    num_type2_blocks: int = 3
    truncation_threshold: float = 3.0
    fc_hidden_dim: int = 128
    dropout: float = 0.5


class _FixedHPFLayer(nn.Module):
    """Fixed (non-learnable) high-pass filter applied channel-wise.

    Uses the SRNet KV (Kernels from Variance) set approximated with
    Laplacian and first/second-derivative filters.  All weights are
    registered as buffers so they are included in ``state_dict`` but never
    updated.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        kernels = _build_hpf_kernels(out_channels)
        # Expand to cover all input channels by repeating.
        # Shape: (out_channels, in_channels, 3, 3)
        kernels_expanded = kernels.unsqueeze(1).expand(-1, in_channels, -1, -1) / in_channels
        self.register_buffer("weight", kernels_expanded.clone().contiguous())
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=1)


class _TruncationLayer(nn.Module):
    """Absolute value followed by symmetric clamp."""

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.abs(x), max=self.threshold)


class _SRNetResidualBlock(nn.Module):
    """Type-I residual block: preserves spatial size."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class _SRNetDownBlock(nn.Module):
    """Type-II strided block: halves spatial resolution and doubles channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # 1x1 projection shortcut for the residual connection.
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.pool(self.bn2(self.conv2(out)))
        return F.relu(out + shortcut, inplace=True)


class DifferentiableDetector(nn.Module):
    """Frozen SRNet-inspired detector for adversarial steganography training.

    All parameters are set to ``requires_grad=False`` during construction and
    remain frozen for the lifetime of the module.  The forward pass is fully
    differentiable with respect to its *input* tensor; the encoder therefore
    receives a meaningful gradient signal from the detector loss even though the
    detector itself is never updated.

    Args:
        config: Detector architecture configuration.

    Example::

        detector = DifferentiableDetector()
        # detector.parameters() has requires_grad=False for all tensors.

        modified_repr = encoder(weight_repr, payload)  # requires_grad=True
        logits = detector(modified_repr)
        loss = F.binary_cross_entropy_with_logits(logits, benign_target)
        loss.backward()
        # Gradients accumulate in encoder.parameters(), not detector.parameters().
    """

    def __init__(self, config: DifferentiableDetectorConfig | None = None) -> None:
        super().__init__()
        cfg = config or DifferentiableDetectorConfig()
        _validate_detector_config(cfg)
        self.config = cfg

        # --- Preprocessing ---
        self.hpf = _FixedHPFLayer(cfg.input_channels, cfg.hpf_channels)
        self.truncation = _TruncationLayer(cfg.truncation_threshold)

        # Projection from HPF output into backbone feature width.
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.hpf_channels, cfg.base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.base_channels),
            nn.ReLU(inplace=True),
        )

        # --- Type-I residual blocks (constant spatial size) ---
        self.type1_blocks = nn.Sequential(
            *[_SRNetResidualBlock(cfg.base_channels) for _ in range(cfg.num_type1_blocks)]
        )

        # --- Type-II downsampling blocks ---
        channels = cfg.base_channels
        down_blocks: list[nn.Module] = []
        for _ in range(cfg.num_type2_blocks):
            out_ch = min(channels * 2, 512)
            down_blocks.append(_SRNetDownBlock(channels, out_ch))
            channels = out_ch
        self.type2_blocks = nn.Sequential(*down_blocks)

        # --- Classification head ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, cfg.fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fc_hidden_dim, 1),
        )

        self._init_weights()
        self.freeze()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, weight_representation: torch.Tensor) -> torch.Tensor:
        """Compute binary detection logits from a weight representation.

        Args:
            weight_representation: Float tensor with shape
                ``(batch, 4, height, width)``.  Values should be in the
                range ``[0, 255]`` (raw byte-channel scale).

        Returns:
            Logit tensor with shape ``(batch, 1)``.  Positive values indicate
            that the detector classifies the input as steganographic.
        """
        # Normalize to [0, 1] for the HPF layer.
        x = weight_representation.float() / 255.0
        x = self.hpf(x)
        x = self.truncation(x)
        x = self.stem(x)
        x = self.type1_blocks(x)
        x = self.type2_blocks(x)
        x = self.global_pool(x)
        return self.classifier(x)

    def freeze(self) -> None:
        """Set all parameters to ``requires_grad=False``.

        Called automatically during construction.  Call again if
        parameters were modified externally.
        """
        for param in self.parameters():
            param.requires_grad_(False)
        # Buffers (HPF weights) are already non-differentiable by nature.

    def is_frozen(self) -> bool:
        """Return True if every parameter has ``requires_grad=False``."""
        return all(not p.requires_grad for p in self.parameters())

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        # HPF layer uses fixed buffer kernels; do not re-initialize.


def build_differentiable_detector(
    config: DifferentiableDetectorConfig | None = None,
) -> DifferentiableDetector:
    """Build a frozen :class:`DifferentiableDetector`."""
    return DifferentiableDetector(config)


# ---------------------------------------------------------------------------
# HPF kernel generation
# ---------------------------------------------------------------------------


def _build_hpf_kernels(num_kernels: int) -> torch.Tensor:
    """Generate ``num_kernels`` 3×3 high-pass filter kernels.

    Uses a set of Laplacian, gradient, and second-derivative filters that
    emphasize high-frequency residuals — the primary signal exploited by
    steganalysis detectors.  If more kernels are requested than the fixed bank
    has, they are generated from sinusoidal patterns.

    Returns:
        Float32 tensor with shape ``(num_kernels, 3, 3)``.
    """
    fixed_kernels: list[list[list[float]]] = [
        # Laplacian
        [[ 0, -1,  0], [-1,  4, -1], [ 0, -1,  0]],
        [[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]],
        [[ 1, -2,  1], [-2,  4, -2], [ 1, -2,  1]],
        # Horizontal / vertical gradients
        [[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]],
        [[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]],
        # Diagonal gradients
        [[-1, -1,  2], [-1,  2, -1], [ 2, -1, -1]],
        [[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]],
        # Second derivatives
        [[ 0,  0,  0], [-1,  2, -1], [ 0,  0,  0]],
        [[ 0, -1,  0], [ 0,  2,  0], [ 0, -1,  0]],
        [[ 0,  0,  0], [ 0, -1,  0], [ 0,  1,  0]],
        # Compass operators
        [[-1,  1, -1], [ 1,  0,  1], [-1,  1, -1]],
        [[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]],
        # Kirsch compass kernels (subset)
        [[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]],
        [[ 5,  5, -3], [ 5,  0, -3], [-3, -3, -3]],
        [[ 5, -3, -3], [ 5,  0, -3], [ 5, -3, -3]],
        # Prewitt
        [[ 1,  1,  1], [ 0,  0,  0], [-1, -1, -1]],
        [[ 1,  0, -1], [ 1,  0, -1], [ 1,  0, -1]],
        # Frei-Chen
        [[ 1,  math.sqrt(2), 1], [ 0, 0, 0], [-1, -math.sqrt(2), -1]],
        [[ 1, 0, -1], [math.sqrt(2), 0, -math.sqrt(2)], [1, 0, -1]],
        # Box-difference
        [[ 1,  1,  1], [ 1, -8,  1], [ 1,  1,  1]],
        # Identity minus mean (acts like unsharp mask)
        [[-1/9, -1/9, -1/9], [-1/9, 8/9, -1/9], [-1/9, -1/9, -1/9]],
        # High-pass cross
        [[ 0, -1,  0], [-1,  5, -1], [ 0, -1,  0]],
        # Emboss
        [[-2, -1,  0], [-1,  1,  1], [ 0,  1,  2]],
        [[ 2,  1,  0], [ 1,  1, -1], [ 0, -1, -2]],
        # Spot
        [[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]],
        # Simple edge
        [[ 0,  0,  0], [-1,  1,  0], [ 0,  0,  0]],
        [[ 0, -1,  0], [ 0,  1,  0], [ 0,  0,  0]],
        [[ 0,  0,  0], [ 0,  1, -1], [ 0,  0,  0]],
        [[ 0,  0,  0], [ 0,  1,  0], [ 0, -1,  0]],
        [[ 1, -1,  0], [ 0,  0,  0], [ 0,  0,  0]],
    ]

    kernels_np = fixed_kernels[:num_kernels]
    if len(kernels_np) < num_kernels:
        # Supplement with sinusoidal high-frequency kernels.
        existing = len(kernels_np)
        for idx in range(existing, num_kernels):
            freq = (idx - existing + 1) * math.pi / 3
            k = [[math.sin(freq * r) * math.cos(freq * c) for c in range(3)] for r in range(3)]
            kernels_np.append(k)

    tensor = torch.tensor(kernels_np, dtype=torch.float32)  # (K, 3, 3)

    # Normalize each kernel to zero-sum (high-pass property).
    for i in range(tensor.shape[0]):
        s = tensor[i].sum()
        if abs(s.item()) > 1e-6:
            tensor[i] = tensor[i] - s / 9.0

    return tensor


def _validate_detector_config(cfg: DifferentiableDetectorConfig) -> None:
    if cfg.input_channels <= 0:
        raise ValueError("input_channels must be positive.")
    if cfg.hpf_channels <= 0:
        raise ValueError("hpf_channels must be positive.")
    if cfg.base_channels <= 0:
        raise ValueError("base_channels must be positive.")
    if cfg.num_type1_blocks < 0:
        raise ValueError("num_type1_blocks must be non-negative.")
    if cfg.num_type2_blocks < 0:
        raise ValueError("num_type2_blocks must be non-negative.")
    if cfg.fc_hidden_dim <= 0:
        raise ValueError("fc_hidden_dim must be positive.")
    if not 0.0 <= cfg.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
