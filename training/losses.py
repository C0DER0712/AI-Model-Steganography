"""Loss functions for defensive model-steganography research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LossWeights:
    """Weights for the composite training objective.

    Attributes:
        classification: Alpha coefficient for host-task classification loss.
        payload: Beta coefficient for payload reconstruction loss.
        distortion: Gamma coefficient for weight-representation distortion loss.
        detector: Delta coefficient for frozen-detector loss.
        capacity: Eta coefficient for adaptive capacity maximisation loss.
            When > 0 and encoder.adaptive_capacity=True, the training loss
            includes -eta * mean(gate) which pushes the learned gate toward
            1 (use more pixels) unless the classification or payload gradient
            pushes it toward 0 (skip sensitive pixels). The interplay of
            these three forces causes the encoder to discover the Pareto-
            optimal set of weight pixels to embed bits in, without any
            hand-coded bpp assignment.
    """

    classification: float = 0.5
    payload: float = 10.0
    distortion: float = 200.0
    detector: float = 2.0
    capacity: float = 0.0


@dataclass(frozen=True)
class LossInputs:
    """Inputs consumed by `CompositeLoss`.

    Attributes:
        classification_logits: Host-model class logits after weight modification.
        classification_targets: Ground-truth class labels.
        payload_logits: Decoder logits for recovered payload bits.
        payload_targets: Ground-truth random payload bits.
        modified_weights: Modified Model-XRay weight representation.
        original_weights: Original Model-XRay weight representation.
        detector_logits: Frozen detector logits for modified weights.
        detector_targets: Target detector labels.
    """

    classification_logits: torch.Tensor | None = None
    classification_targets: torch.Tensor | None = None
    payload_logits: torch.Tensor | None = None
    payload_targets: torch.Tensor | None = None
    modified_weights: torch.Tensor | None = None
    original_weights: torch.Tensor | None = None
    detector_logits: torch.Tensor | None = None
    detector_targets: torch.Tensor | None = None
    # Learned per-pixel gate from AdaptiveCapacityGate; None for fixed-bpp.
    # Shape (batch, 1, H, W), values in [0,1].
    capacity_gate: torch.Tensor | None = None


@dataclass(frozen=True)
class LossOutput:
    """Composite loss result with named component losses."""

    total: torch.Tensor
    components: Mapping[str, torch.Tensor]


class ClassificationLoss(nn.Module):
    """Cross-entropy loss for preserving host-model task accuracy."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute host-task classification loss."""

        return F.cross_entropy(logits, targets)


class PayloadReconstructionLoss(nn.Module):
    """Binary cross-entropy loss for recovered random payload bits."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute payload bit reconstruction loss from logits."""

        return F.binary_cross_entropy_with_logits(
            logits.reshape_as(targets).to(dtype=torch.float32),
            targets.to(dtype=torch.float32),
        )


class WeightDistortionLoss(nn.Module):
    """Mean-squared error between original and modified weight representations."""

    def forward(self, modified_weights: torch.Tensor, original_weights: torch.Tensor) -> torch.Tensor:
        """Compute weight-representation distortion loss."""

        return F.mse_loss(modified_weights, original_weights)


class DetectorLoss(nn.Module):
    """Loss against a frozen detector's output.

    This module is intentionally generic. Binary detector outputs use
    BCE-with-logits, while multi-class detector outputs use cross entropy.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute detector objective loss from logits."""

        if logits.ndim == 1 or logits.shape[-1] == 1:
            return F.binary_cross_entropy_with_logits(
                logits.reshape_as(targets).to(dtype=torch.float32),
                targets.to(dtype=torch.float32),
            )

        return F.cross_entropy(logits, targets.to(dtype=torch.long))


class CompositeLoss(nn.Module):
    """Weighted objective:

    `L = alpha * classification + beta * payload + gamma * distortion
    + delta * detector`.
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        *,
        classification_loss: nn.Module | None = None,
        payload_loss: nn.Module | None = None,
        distortion_loss: nn.Module | None = None,
        detector_loss: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        _validate_loss_weights(self.weights)

        self.classification_loss = classification_loss or ClassificationLoss()
        self.payload_loss = payload_loss or PayloadReconstructionLoss()
        self.distortion_loss = distortion_loss or WeightDistortionLoss()
        self.detector_loss = detector_loss or DetectorLoss()
        self._last_host_accuracy = 0.0

    def set_alpha(self, alpha: float) -> None:
        """Update the classification loss weight (alpha) in-place.

        Called each epoch by AlphaCurriculumScheduler during the warm-up
        phase to linearly ramp alpha from 0 → target while keeping all other
        loss weights unchanged.  The pipeline's run_classification flag is
        NOT changed here — that is set once at pipeline build time based on
        the *final* target alpha (> 0), so classification forward passes
        always run during warmup even though their gradient contribution is
        scaled to 0 for the first few epochs.  This is intentional: it keeps
        the pipeline graph identical throughout training, avoiding any state
        discontinuity on resume.
        """
        import dataclasses
        self.weights = dataclasses.replace(self.weights, classification=float(alpha))

    def forward(self, inputs: LossInputs) -> LossOutput:
        """Compute weighted composite loss and named components.

        Args:
            inputs: Tensors required by enabled loss terms.

        Returns:
            `LossOutput` containing the scalar total and detached component
            losses for logging.

        Raises:
            ValueError: If an enabled term is missing required tensors.
        """

        total: torch.Tensor | None = None
        components: dict[str, torch.Tensor] = {}

        classification = self._maybe_classification_loss(inputs)
        payload = self._maybe_payload_loss(inputs)
        distortion = self._maybe_distortion_loss(inputs)
        detector = self._maybe_detector_loss(inputs)

        capacity = self._maybe_capacity_loss(inputs)

        weighted_terms = {
            "classification": (classification, self.weights.classification),
            "payload": (payload, self.weights.payload),
            "distortion": (distortion, self.weights.distortion),
            "detector": (detector, self.weights.detector),
            "capacity": (capacity, self.weights.capacity),
        }

        for name, (loss, weight) in weighted_terms.items():
            if loss is None:
                continue

            components[name] = loss.detach()
            term = loss * weight
            total = term if total is None else total + term

        if total is None:
            raise ValueError("At least one loss weight must be positive.")

        return LossOutput(total=total, components=components)

    def _maybe_classification_loss(self, inputs: LossInputs) -> torch.Tensor | None:
        if self.weights.classification == 0:
            return None
        if inputs.classification_logits is None or inputs.classification_targets is None:
            raise ValueError("classification loss requires logits and targets.")
        loss = self.classification_loss(inputs.classification_logits, inputs.classification_targets)
        # Store accuracy as a side-channel metric (no gradient needed)
        with torch.no_grad():
            preds = inputs.classification_logits.argmax(dim=-1)
            self._last_host_accuracy = (
                (preds == inputs.classification_targets).float().mean().item()
            )
        return loss

    def _maybe_payload_loss(self, inputs: LossInputs) -> torch.Tensor | None:
        if self.weights.payload == 0:
            return None
        if inputs.payload_logits is None or inputs.payload_targets is None:
            raise ValueError("payload loss requires logits and targets.")
        return self.payload_loss(inputs.payload_logits, inputs.payload_targets)

    def _maybe_distortion_loss(self, inputs: LossInputs) -> torch.Tensor | None:
        if self.weights.distortion == 0:
            return None
        if inputs.modified_weights is None or inputs.original_weights is None:
            raise ValueError("distortion loss requires modified and original weights.")
        return self.distortion_loss(inputs.modified_weights, inputs.original_weights)

    def _maybe_capacity_loss(self, inputs: LossInputs) -> torch.Tensor | None:
        """Adaptive capacity maximisation: penalise unused pixels.

        When the encoder's AdaptiveCapacityGate is active, ``capacity_gate``
        is a (B, 1, H, W) soft mask in [0,1].  We want to MAXIMISE the
        mean gate (use as many pixels as possible), so the loss term is
        ``-mean(gate)``.  The negative sign means minimising this loss is
        equivalent to maximising gate usage.

        The interplay with other losses gives genuine adaptive capacity:
        - Capacity loss pushes gate → 1 (use all pixels).
        - Classification loss gradient pushes delta → 0 for sensitive pixels,
          which forces gate → 0 there (or the accuracy drops).
        - Payload loss is unchanged (gate-independent; gated-out positions
          still contribute BCE, incentivising the gate to only zero out
          positions where the payload gradient is low anyway).
        Together these three forces discover the Pareto-optimal set of
        pixels to embed bits in — no hand-coded bpp required.
        """
        if self.weights.capacity == 0:
            return None
        if inputs.capacity_gate is None:
            return None  # encoder.adaptive_capacity=False, skip silently
        # -mean(gate): minimising this maximises gate usage.
        return -inputs.capacity_gate.mean()

    def _maybe_detector_loss(self, inputs: LossInputs) -> torch.Tensor | None:
        if self.weights.detector == 0:
            return None
        if inputs.detector_logits is None or inputs.detector_targets is None:
            raise ValueError("detector loss requires logits and targets.")
        return self.detector_loss(inputs.detector_logits, inputs.detector_targets)


def _validate_loss_weights(weights: LossWeights) -> None:
    values = {
        "classification": weights.classification,
        "payload": weights.payload,
        "distortion": weights.distortion,
        "detector": weights.detector,
    }
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} loss weight must be non-negative.")
