"""Training utilities for the defensive research codebase."""

from training.dataset import SyntheticImageDataset, SyntheticWeightDataset
from training.losses import (
    ClassificationLoss,
    CompositeLoss,
    DetectorLoss,
    LossInputs,
    LossOutput,
    LossWeights,
    PayloadReconstructionLoss,
    WeightDistortionLoss,
)
from training.trainer import Trainer, TrainerConfig, TrainerState

# training.dataset and training.experiment import from models.pipeline, which
# depends on training.losses.  Importing them here would create a circular
# dependency through package __init__ files.  Import them directly instead:
#   from training.dataset import SteganographyDataset
#   from training.experiment import run_experiment

__all__ = [
    "ClassificationLoss",
    "CompositeLoss",
    "DetectorLoss",
    "LossInputs",
    "LossOutput",
    "LossWeights",
    "PayloadReconstructionLoss",
    "Trainer",
    "TrainerConfig",
    "TrainerState",
    "WeightDistortionLoss",
]
