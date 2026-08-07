"""Tests for host model adapters."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from models.host_models import (
    HostModelAdapter,
    HostModelConfig,
    build_host_model,
)
from utils.weights import extract_weights, flatten_weights


@pytest.fixture(params=["resnet18", "mobilenet_v2"])
def host_model(request) -> HostModelAdapter:
    """Instantiate a small untrained host model for testing."""
    return build_host_model(request.param, num_classes=10, pretrained=False)


class TestHostModelConfig:
    def test_invalid_name_raises(self) -> None:
        # Validation happens in _validate_config, called by HostModelAdapter.
        with pytest.raises(ValueError, match="Unsupported"):
            build_host_model("vgg_unknown", num_classes=10)  # type: ignore[arg-type]

    def test_invalid_num_classes_raises(self) -> None:
        with pytest.raises(ValueError, match="num_classes"):
            build_host_model("resnet18", num_classes=0)


class TestHostModelAdapter:
    def test_is_nn_module(self, host_model: HostModelAdapter) -> None:
        assert isinstance(host_model, nn.Module)

    def test_forward_returns_correct_shape(self, host_model: HostModelAdapter) -> None:
        host_model.eval()  # Avoid BatchNorm single-sample error.
        images = torch.randn(2, 3, 32, 32)
        logits = host_model(images)
        assert logits.shape == (2, 10)

    def test_num_float_parameters_positive(self, host_model: HostModelAdapter) -> None:
        assert host_model.num_float_parameters > 0

    def test_backbone_name_matches(self) -> None:
        model = build_host_model("resnet18", num_classes=10, pretrained=False)
        assert model.backbone_name == "resnet18"

    def test_weight_extraction_round_trip(self, host_model: HostModelAdapter) -> None:
        """extract_weights + flatten_weights must match num_float_parameters."""
        records = extract_weights(host_model.model)
        flat = flatten_weights(records)
        assert flat.numel() == host_model.num_float_parameters


class TestFunctionalForward:
    def test_functional_forward_matches_standard(self) -> None:
        model = build_host_model("resnet18", num_classes=10, pretrained=False)
        model.eval()  # Required for batch-size-1 BatchNorm.
        images = torch.randn(2, 3, 32, 32)

        # Collect all params and buffers for functional_call.
        params = {
            name: tensor
            for name, tensor in model.model.state_dict().items()
        }
        logits_standard = model(images)
        logits_functional = model.functional_forward(images, params)

        assert torch.allclose(logits_standard, logits_functional, atol=1e-5)

    def test_functional_forward_allows_gradient(self) -> None:
        model = build_host_model("resnet18", num_classes=10, pretrained=False)
        model.eval()
        images = torch.randn(2, 3, 32, 32)

        # Only learnable parameters (not buffers like running_mean/var) get
        # requires_grad=True; functional_call forbids grad on running stats.
        learnable_names = {n for n, _ in model.model.named_parameters()}
        params: dict[str, torch.Tensor] = {}
        for name, tensor in model.model.state_dict().items():
            cloned = tensor.clone().detach()
            if name in learnable_names and tensor.is_floating_point():
                cloned.requires_grad_(True)
            params[name] = cloned

        logits = model.functional_forward(images, params)
        loss = logits.sum()
        loss.backward()

        # At least some learnable parameters must have received gradients.
        grad_count = sum(
            1 for name, t in params.items()
            if name in learnable_names and t.is_floating_point() and t.grad is not None
        )
        assert grad_count > 0


class TestAllBackbones:
    """Smoke test that all four supported backbones instantiate correctly."""

    @pytest.mark.parametrize(
        "name", ["resnet18", "resnet50", "mobilenet_v2", "vgg16"]
    )
    def test_instantiation(self, name: str) -> None:
        model = build_host_model(name, num_classes=5, pretrained=False)
        model.eval()  # Disable BatchNorm training mode for single-sample tests.
        assert model.backbone_name == name
        images = torch.randn(2, 3, 32, 32)
        logits = model(images)
        assert logits.shape == (2, 5)
