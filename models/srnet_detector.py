"""Standalone SRNet PyTorch architecture for Model X-Ray detection.

Faithful PyTorch port of the SRNet steganalysis network (Boroumand et al., 2018),
adapted for AI model weight steganography detection as used in the Model X-Ray paper.

This module is the *evaluation* SRNet — a trainable model used to build the
Few-Shot Learning detector.  It is separate from the frozen
:class:`~evaluation.differentiable_detector.DifferentiableDetector`, which is
an SRNet-inspired module used only to provide adversarial gradient signal during
encoder training.

Architecture:
  - Type 1 (2 blocks): Conv-BN-ReLU noise residual extraction (1 → 64 → 16 ch)
  - Type 2 (5 blocks): Residual blocks preserving spatial resolution (16 ch)
  - Type 3 (4 blocks): Downsampling blocks with AveragePooling (16→32→64→128→256)
  - Type 4 (1 block):  Global Average Pooling → Linear projection → 512-d embedding

Input:  ``(B, 1, H, W)`` grayscale Grayscale-Fourpart (GF) images, values in
        ``[0, 255]``.
Output: 512-d L2-normalised embedding vector (for FSL) or class logits when a
        classification head is attached.

Reference:
    Boroumand, M. et al. "Deep Residual Network for Steganalysis of Digital
    Images." IEEE Trans. Info. Forensics and Security, 2018.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SRNetConfig:
    """Configuration for :class:`SRNetDetector`.

    Attributes:
        in_channels: Number of input image channels.  GF images are
            single-channel grayscale, so the default is 1.
        embedding_dim: Dimensionality of the L2-normalised output embedding.
        num_classes: When set, a linear classification head is attached and
            :meth:`~SRNetDetector.forward` returns class logits instead of
            embeddings.  Set to ``None`` for embedding-only (FSL) mode.
    """

    in_channels: int = 1
    embedding_dim: int = 512
    num_classes: Optional[int] = None


# ---------------------------------------------------------------------------
# Building-block layers
# ---------------------------------------------------------------------------


class _Type1Block(nn.Module):
    """Type-1 block: Conv 3×3 → BN → ReLU (noise-residual extraction)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class _Type2Block(nn.Module):
    """Type-2 block: residual block preserving spatial resolution.

    Two sequential Conv-BN layers with an additive identity shortcut.
    """

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


class _Type3Block(nn.Module):
    """Type-3 block: downsampling residual block with AveragePooling.

    Halves the spatial resolution and expands the channel width.  A 1×1
    projection shortcut (with the same pooling) preserves the residual
    connection across the channel expansion.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
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


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class SRNetDetector(nn.Module):
    """Standalone SRNet-based detector for Model X-Ray few-shot evaluation.

    Use :meth:`embed` to obtain L2-normalised 512-d representations suitable
    for triplet-loss FSL training.  Use :meth:`forward` to obtain logits when
    a classification head is configured (i.e. ``config.num_classes`` is set).

    Args:
        config: Architecture configuration.

    Example — embedding mode::

        model = SRNetDetector()
        gf_image = torch.randint(0, 256, (4, 1, 256, 256)).float()
        emb = model.embed(gf_image)  # (4, 512), L2-normalised

    Example — classification mode::

        model = SRNetDetector(SRNetConfig(num_classes=2))
        logits = model(gf_image)  # (4, 2)
    """

    def __init__(self, config: Optional[SRNetConfig] = None) -> None:
        super().__init__()
        cfg = config or SRNetConfig()
        self.config = cfg

        # ---- Type-1: noise residual extraction (2 blocks) ----
        # in_channels → 64 → 16
        self.type1 = nn.Sequential(
            _Type1Block(cfg.in_channels, 64),
            _Type1Block(64, 16),
        )

        # ---- Type-2: residual blocks at constant resolution (5 blocks) ----
        self.type2 = nn.Sequential(*[_Type2Block(16) for _ in range(5)])

        # ---- Type-3: downsampling blocks (4 blocks) ----
        # 16 → 32 → 64 → 128 → 256
        type3_dims = [(16, 32), (32, 64), (64, 128), (128, 256)]
        self.type3 = nn.Sequential(
            *[_Type3Block(in_ch, out_ch) for in_ch, out_ch in type3_dims]
        )

        # ---- Type-4: global average pool → embedding ----
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.embedding_proj = nn.Linear(256, cfg.embedding_dim)

        # ---- Optional classification head ----
        self.classifier: Optional[nn.Linear] = None
        if cfg.num_classes is not None:
            self.classifier = nn.Linear(cfg.embedding_dim, cfg.num_classes)

        self._init_weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return an L2-normalised embedding for each input image.

        Args:
            x: Float tensor with shape ``(B, 1, H, W)``.  Values should be
                in the range ``[0, 255]`` (raw byte scale).

        Returns:
            Float tensor with shape ``(B, embedding_dim)``, L2-normalised
            along the feature dimension.
        """
        x = x.float() / 255.0
        x = self.type1(x)
        x = self.type2(x)
        x = self.type3(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.embedding_proj(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits (with head) or embeddings (without).

        Args:
            x: Float tensor with shape ``(B, 1, H, W)``.

        Returns:
            When ``config.num_classes`` is set: logit tensor ``(B, num_classes)``.
            Otherwise: L2-normalised embedding ``(B, embedding_dim)``.
        """
        emb = self.embed(x)
        if self.classifier is not None:
            return self.classifier(emb)
        return emb

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save model weights and config to ``path``.

        Args:
            path: Destination ``.pt`` file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> "SRNetDetector":
        """Load a previously saved :class:`SRNetDetector`.

        Args:
            path: Path to a ``.pt`` file saved with :meth:`save`.
            map_location: Optional device remapping for ``torch.load``.

        Returns:
            A :class:`SRNetDetector` with weights loaded.
        """
        # The checkpoint intentionally contains the SRNetConfig dataclass.
        # These are locally-generated research artefacts, so opt into the
        # full trusted checkpoint format on PyTorch versions that support the
        # weights_only switch.  Keep compatibility with older PyTorch releases.
        try:
            ckpt = torch.load(
                path,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            ckpt = torch.load(path, map_location=map_location)
        model = cls(ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_srnet_detector(
    config: Optional[SRNetConfig] = None,
) -> SRNetDetector:
    """Build a :class:`SRNetDetector` from an optional configuration.

    Args:
        config: Architecture configuration.  Defaults to
            :class:`SRNetConfig` with single-channel input and 512-d embedding.

    Returns:
        A new :class:`SRNetDetector` instance.
    """
    return SRNetDetector(config or SRNetConfig())
