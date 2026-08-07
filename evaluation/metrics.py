"""Detection evaluation metrics.

Computes classification metrics for Model X-Ray detector evaluation,
including accuracy, precision, recall, F1-score, ROC-AUC, and error rates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

if TYPE_CHECKING:
    from evaluation.detector import ModelXRayDetector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectorMetrics:
    """Metrics for evaluating a Model X-Ray detector.

    Attributes:
        accuracy: Overall classification accuracy.
        precision: True positives / (True positives + False positives).
        recall: True positives / (True positives + False negatives).
        f1_score: Harmonic mean of precision and recall.
        roc_auc: Area under the receiver operating characteristic curve.
        false_positive_rate: False positives / (False positives + True negatives).
        false_negative_rate: False negatives / (False negatives + True positives).
        confusion_matrix: 2x2 confusion matrix [[TN, FP], [FN, TP]].
        y_true: Ground truth labels (0=clean, 1=modified).
        y_scores: Continuous anomaly scores for ROC.
        y_pred: Binary predictions.
    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    false_positive_rate: float
    false_negative_rate: float
    confusion_matrix: np.ndarray
    y_true: np.ndarray
    y_scores: np.ndarray
    y_pred: np.ndarray

    def to_dict(self) -> dict[str, float]:
        """Returns scalar metrics as a dictionary.

        Returns:
            Dictionary containing scalar metrics.
        """
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
        }

    def __str__(self) -> str:
        """Formats scalar metrics as a string.

        Returns:
            Formatted string representation.
        """
        return (
            f"DetectorMetrics:\n"
            f"  Accuracy:    {self.accuracy:.4f}\n"
            f"  Precision:   {self.precision:.4f}\n"
            f"  Recall:      {self.recall:.4f}\n"
            f"  F1-Score:    {self.f1_score:.4f}\n"
            f"  ROC-AUC:     {self.roc_auc:.4f}\n"
            f"  FPR:         {self.false_positive_rate:.4f}\n"
            f"  FNR:         {self.false_negative_rate:.4f}"
        )


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray
) -> DetectorMetrics:
    """Computes detection metrics from predictions and labels.

    Args:
        y_true: Ground truth labels.
        y_pred: Binary predictions.
        y_scores: Continuous anomaly scores.

    Returns:
        Computed DetectorMetrics.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    else:
        logger.warning("Only one class present in y_true. ROC-AUC is undefined, defaulting to 0.5.")
        roc_auc = 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return DetectorMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        roc_auc=roc_auc,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        confusion_matrix=cm,
        y_true=y_true,
        y_scores=y_scores,
        y_pred=y_pred,
    )


def evaluate_detector(
    detector: 'ModelXRayDetector',
    clean_model_paths: list[str | Path],
    modified_model_paths: list[str | Path],
    *,
    verbose: bool = True,
) -> DetectorMetrics:
    """Evaluates a detector on saved model files.

    Args:
        detector: The ModelXRayDetector instance to evaluate.
        clean_model_paths: Paths to benign model files.
        modified_model_paths: Paths to malicious model files.
        verbose: Whether to display progress bars and log metrics.

    Returns:
        Computed DetectorMetrics.
    """
    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    y_scores_list: list[float] = []

    def _process_paths(paths: list[str | Path], label: int, desc: str) -> None:
        iterable = tqdm(paths, desc=desc, disable=not verbose)
        for path in iterable:
            result = detector.predict(path)
            y_true_list.append(label)
            y_pred_list.append(1 if result.is_malicious else 0)
            y_scores_list.append(float(result.anomaly_score))

    _process_paths(clean_model_paths, 0, "Evaluating clean models")
    _process_paths(modified_model_paths, 1, "Evaluating modified models")

    metrics = _compute_metrics(
        np.array(y_true_list), np.array(y_pred_list), np.array(y_scores_list)
    )

    if verbose:
        logger.info(f"\n{metrics}")

    return metrics


def evaluate_detector_from_weights(
    detector: 'ModelXRayDetector',
    clean_weights: list[torch.Tensor],
    modified_weights: list[torch.Tensor],
    *,
    verbose: bool = True,
) -> DetectorMetrics:
    """Evaluates a detector on flat weight tensors.

    Args:
        detector: The ModelXRayDetector instance to evaluate.
        clean_weights: Benign model weights.
        modified_weights: Malicious model weights.
        verbose: Whether to display progress bars and log metrics.

    Returns:
        Computed DetectorMetrics.
    """
    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    y_scores_list: list[float] = []

    def _process_weights(weights_list: list[torch.Tensor], label: int, desc: str) -> None:
        iterable = tqdm(weights_list, desc=desc, disable=not verbose)
        for weights in iterable:
            result = detector.predict_from_weight_tensor(weights)
            y_true_list.append(label)
            y_pred_list.append(1 if result.is_malicious else 0)
            y_scores_list.append(float(result.anomaly_score))

    _process_weights(clean_weights, 0, "Evaluating clean weights")
    _process_weights(modified_weights, 1, "Evaluating modified weights")

    metrics = _compute_metrics(
        np.array(y_true_list), np.array(y_pred_list), np.array(y_scores_list)
    )

    if verbose:
        logger.info(f"\n{metrics}")

    return metrics
