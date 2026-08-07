"""Host model adapters for supported architectures.

Provides pretrained ResNet18, ResNet50, MobileNetV2, and VGG16 wrappers that
expose the model for steganographic weight modification research.  Each adapter
supports classification forward passes and integrates with `utils.weights` for
weight extraction and restoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torchvision.models as tvm
from torch import nn
from torch.func import functional_call

HostModelName = Literal["resnet18", "resnet50", "mobilenet_v2", "vgg16", "tiny", "tiny_bn"]

_HOST_MODEL_NAMES: frozenset[HostModelName] = frozenset(
    {"resnet18", "resnet50", "mobilenet_v2", "vgg16", "tiny", "tiny_bn"}
)


@dataclass(frozen=True)
class HostModelConfig:
    """Configuration for a host model adapter.

    Attributes:
        name: One of the four supported backbone names.
        num_classes: Output class count; default 1000 (ImageNet).
        pretrained: Whether to load official ImageNet pre-trained weights.
    """

    name: HostModelName
    num_classes: int = 1000
    pretrained: bool = True


class HostModelAdapter(nn.Module):
    """Thin wrapper around a torchvision backbone used as the host model.

    The adapter keeps the full backbone as a child module so that
    `utils.weights.extract_weights` sees every floating-point parameter in a
    stable, deterministic order.  A differentiable classification forward is
    available via :meth:`functional_forward` for use with modified weight
    tensors without mutating the stored parameters.
    """

    def __init__(self, config: HostModelConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.model: nn.Module = _build_backbone(config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run a standard forward pass and return class logits.

        Args:
            images: Float image batch with shape `(batch, 3, H, W)`.

        Returns:
            Logit tensor with shape `(batch, num_classes)`.
        """
        return self.model(images)

    def functional_forward(
        self,
        images: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run classification with externally supplied parameter tensors.

        This method uses :func:`torch.func.functional_call` so that modified
        weights produced by the encoder can be used for the classification
        forward pass without in-place mutation of the host model's stored
        parameters.  The computation graph is preserved, enabling gradients to
        flow back to the encoder through the classification loss.

        Args:
            images: Float image batch with shape `(batch, 3, H, W)`.
            params: Mapping from parameter name to replacement tensor.  Any
                parameter not listed here is taken from `self.model`.

        Returns:
            Logit tensor with shape `(batch, num_classes)`.
        """
        return functional_call(self.model, params, (images,))

    @property
    def num_float_parameters(self) -> int:
        """Total floating-point values in the host backbone's state dict.

        This includes both learnable parameters and persistent floating-point
        buffers (e.g. BatchNorm running statistics), matching the count
        returned by :func:`~utils.weights.flatten_weights` so that callers
        can use this value to size buffers without a separate extraction step.
        """
        return sum(
            t.numel()
            for t in self.model.state_dict().values()
            if t.is_floating_point()
        )

    @property
    def backbone_name(self) -> HostModelName:
        """Name of the underlying backbone architecture."""
        return self.config.name


def build_host_model(
    name: HostModelName,
    num_classes: int = 1000,
    pretrained: bool = True,
) -> HostModelAdapter:
    """Build a :class:`HostModelAdapter` for the requested backbone.

    Args:
        name: One of ``"resnet18"``, ``"resnet50"``, ``"mobilenet_v2"``,
            ``"vgg16"``.
        num_classes: Number of output classes.
        pretrained: Whether to load official pretrained weights.

    Returns:
        Configured :class:`HostModelAdapter`.
    """
    return HostModelAdapter(HostModelConfig(name=name, num_classes=num_classes, pretrained=pretrained))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_backbone(config: HostModelConfig) -> nn.Module:
    name = config.name

    if name == "tiny":
        # Minimal two-conv backbone for unit tests.  No BatchNorm so it works
        # with batch_size=1.  ~1.7 K float parameters for any num_classes.
        return _TinyBackbone(config.num_classes)

    if name == "tiny_bn":
        # Like "tiny" but with BatchNorm layers.  Used to regression-test that
        # _rebuild_params does not make running_mean / running_var differentiable.
        # ~400 learnable params + 16 float buffers (running_mean/var).
        return _TinyBnBackbone(config.num_classes)

    weights_arg = "DEFAULT" if config.pretrained else None

    if name == "resnet18":
        model = tvm.resnet18(weights=weights_arg)
    elif name == "resnet50":
        model = tvm.resnet50(weights=weights_arg)
    elif name == "mobilenet_v2":
        model = tvm.mobilenet_v2(weights=weights_arg)
    elif name == "vgg16":
        model = tvm.vgg16(weights=weights_arg)
    else:
        raise ValueError(f"Unsupported host model name: '{name}'.")

    if config.num_classes != 1000:
        model = _replace_classifier_head(model, name, config.num_classes)

    return model


class _TinyBackbone(nn.Module):
    """Minimal 2-conv backbone (~1.7 K float params) for fast unit tests.

    Accepts any ``(B, 3, H, W)`` input with ``H, W >= 3``.  No BatchNorm so
    single-sample inference works without special handling.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class _TinyBnBackbone(nn.Module):
    """Minimal backbone with BatchNorm — for regression-testing that
    ``_rebuild_params`` does not make ``running_mean`` / ``running_var``
    differentiable.

    Architecture mirrors the relevant structural property of ResNet/MobileNet
    (Conv → BN → ReLU) without the memory cost of a full torchvision model.
    Has ~400 learnable float params plus 16 float buffers
    (``running_mean`` / ``running_var`` for two BN layers).

    Must be used in ``eval()`` mode when batch_size == 1.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def _replace_classifier_head(model: nn.Module, name: HostModelName, num_classes: int) -> nn.Module:
    """Replace the final classifier layer for a non-ImageNet class count."""
    if name in ("resnet18", "resnet50"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == "mobilenet_v2":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif name == "vgg16":
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def _validate_config(config: HostModelConfig) -> None:
    if config.name not in _HOST_MODEL_NAMES:
        supported = ", ".join(sorted(_HOST_MODEL_NAMES))
        raise ValueError(
            f"Unsupported host model name '{config.name}'. "
            f"Supported names: {supported}."
        )
    if config.num_classes <= 0:
        raise ValueError("num_classes must be positive.")
