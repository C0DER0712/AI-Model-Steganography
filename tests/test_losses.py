import pytest
import torch
from torch.nn import functional as F

from training.losses import (
    ClassificationLoss,
    CompositeLoss,
    DetectorLoss,
    LossInputs,
    LossWeights,
    PayloadReconstructionLoss,
    WeightDistortionLoss,
)


def test_classification_loss_matches_cross_entropy() -> None:
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.2]])
    targets = torch.tensor([0, 1])

    loss = ClassificationLoss()(logits, targets)

    assert loss == pytest.approx(F.cross_entropy(logits, targets).item())


def test_payload_reconstruction_loss_matches_bce_with_logits() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.5, -0.5]])
    targets = torch.tensor([[1, 0, 1, 0]], dtype=torch.uint8)

    loss = PayloadReconstructionLoss()(logits, targets)

    assert loss == pytest.approx(
        F.binary_cross_entropy_with_logits(logits, targets.float()).item()
    )


def test_weight_distortion_loss_matches_mse() -> None:
    modified = torch.tensor([1.0, 2.0, 5.0])
    original = torch.tensor([1.0, 1.0, 3.0])

    loss = WeightDistortionLoss()(modified, original)

    assert loss == pytest.approx(F.mse_loss(modified, original).item())


def test_detector_loss_supports_binary_and_multiclass_outputs() -> None:
    detector = DetectorLoss()
    binary_logits = torch.tensor([[2.0], [-2.0]])
    binary_targets = torch.tensor([[1.0], [0.0]])
    multiclass_logits = torch.tensor([[2.0, 0.5], [0.1, 1.2]])
    multiclass_targets = torch.tensor([0, 1])

    assert detector(binary_logits, binary_targets) == pytest.approx(
        F.binary_cross_entropy_with_logits(binary_logits, binary_targets).item()
    )
    assert detector(multiclass_logits, multiclass_targets) == pytest.approx(
        F.cross_entropy(multiclass_logits, multiclass_targets).item()
    )


def test_composite_loss_applies_configurable_weights() -> None:
    inputs = LossInputs(
        classification_logits=torch.tensor([[2.0, 0.5], [0.1, 1.2]]),
        classification_targets=torch.tensor([0, 1]),
        payload_logits=torch.tensor([[2.0, -1.0, 0.5, -0.5]]),
        payload_targets=torch.tensor([[1, 0, 1, 0]], dtype=torch.uint8),
        modified_weights=torch.tensor([1.0, 2.0, 5.0]),
        original_weights=torch.tensor([1.0, 1.0, 3.0]),
        detector_logits=torch.tensor([[2.0], [-2.0]]),
        detector_targets=torch.tensor([[1.0], [0.0]]),
    )
    weights = LossWeights(classification=0.5, payload=2.0, distortion=0.25, detector=3.0)

    output = CompositeLoss(weights)(inputs)

    expected_classification = F.cross_entropy(
        inputs.classification_logits,
        inputs.classification_targets,
    )
    expected_payload = F.binary_cross_entropy_with_logits(
        inputs.payload_logits,
        inputs.payload_targets.float(),
    )
    expected_distortion = F.mse_loss(inputs.modified_weights, inputs.original_weights)
    expected_detector = F.binary_cross_entropy_with_logits(
        inputs.detector_logits,
        inputs.detector_targets,
    )
    expected_total = (
        weights.classification * expected_classification
        + weights.payload * expected_payload
        + weights.distortion * expected_distortion
        + weights.detector * expected_detector
    )

    assert output.total == pytest.approx(expected_total.item())
    assert set(output.components) == {
        "classification",
        "payload",
        "distortion",
        "detector",
    }


def test_composite_loss_skips_zero_weight_terms() -> None:
    inputs = LossInputs(
        payload_logits=torch.tensor([[2.0, -1.0]]),
        payload_targets=torch.tensor([[1, 0]], dtype=torch.uint8),
    )
    composite = CompositeLoss(
        LossWeights(classification=0.0, payload=1.0, distortion=0.0, detector=0.0)
    )

    output = composite(inputs)

    assert set(output.components) == {"payload"}


def test_composite_loss_does_not_evaluate_zero_weight_classification() -> None:
    inputs = LossInputs(
        classification_logits=torch.full((2, 2), float("nan")),
        classification_targets=torch.tensor([0, 1]),
        payload_logits=torch.tensor([[2.0, -1.0]]),
        payload_targets=torch.tensor([[1, 0]], dtype=torch.uint8),
    )

    output = CompositeLoss(
        LossWeights(classification=0.0, payload=1.0, distortion=0.0, detector=0.0)
    )(inputs)

    assert torch.isfinite(output.total)
    assert set(output.components) == {"payload"}


def test_composite_loss_requires_inputs_for_enabled_terms() -> None:
    composite = CompositeLoss(
        LossWeights(classification=1.0, payload=0.0, distortion=0.0, detector=0.0)
    )

    with pytest.raises(ValueError, match="classification loss requires"):
        composite(LossInputs())


def test_loss_weights_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CompositeLoss(LossWeights(classification=-1.0))
