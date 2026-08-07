"""Evaluation module for Model X-Ray detector integration.

Provides tools for evaluating generated models against the Model X-Ray detector,
including metrics computation and publication-quality plotting.
"""

from evaluation.accuracy import AccuracyDropResult, AccuracyResult, evaluate_accuracy, evaluate_accuracy_drop
from evaluation.capacity import CapacityResult, compute_capacity, max_payload_bits
from evaluation.detector import DetectionResult, ModelXRayDetector
from evaluation.differentiable_detector import DifferentiableDetector, DifferentiableDetectorConfig, build_differentiable_detector
from evaluation.fsl_detector import (
    FSLConfig,
    FSLDataset,
    FSLDetector,
    NNClassifier,
    NearestCentroidClassifier,
    TripletDataset,
    weighted_metric,
)
from evaluation.metrics import DetectorMetrics, evaluate_detector, evaluate_detector_from_weights
from evaluation.plotting import (
    plot_confusion_matrix,
    plot_detection_comparison,
    plot_metrics_bar,
    plot_metrics_summary,
    plot_roc_curve,
    plot_score_distribution,
)

__all__ = [
    "AccuracyDropResult",
    "AccuracyResult",
    "CapacityResult",
    "DetectionResult",
    "DetectorMetrics",
    "DifferentiableDetector",
    "DifferentiableDetectorConfig",
    "FSLConfig",
    "FSLDataset",
    "FSLDetector",
    "ModelXRayDetector",
    "NearestCentroidClassifier",
    "NNClassifier",
    "TripletDataset",
    "weighted_metric",
    "build_differentiable_detector",
    "compute_capacity",
    "evaluate_accuracy",
    "evaluate_accuracy_drop",
    "evaluate_detector",
    "evaluate_detector_from_weights",
    "max_payload_bits",
    "plot_confusion_matrix",
    "plot_detection_comparison",
    "plot_metrics_bar",
    "plot_metrics_summary",
    "plot_roc_curve",
    "plot_score_distribution",
]
