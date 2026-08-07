from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pytest
import torch

from evaluation.detector import DetectionResult, ModelXRayDetector
from evaluation.metrics import (
    DetectorMetrics,
    _compute_metrics,
    evaluate_detector,
    evaluate_detector_from_weights,
)

class MockDetector(ModelXRayDetector):
    """A mock detector for testing metrics."""
    def __init__(self, scores: List[float]):
        super().__init__()
        self.scores = scores
        self.call_idx = 0

    def predict(self, model_path: Union[str, Path]) -> DetectionResult:
        score = self.scores[self.call_idx % len(self.scores)]
        self.call_idx += 1
        return DetectionResult(
            is_malicious=score > 0.5,
            confidence=score if score > 0.5 else 1 - score,
            anomaly_score=score,
            dist_to_benign=1 - score,
            dist_to_malicious=score
        )

    def predict_from_weight_tensor(self, weights: torch.Tensor) -> DetectionResult:
        # Same as predict for mock
        return self.predict("dummy_path")


def test_compute_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    
    metrics = _compute_metrics(y_true, y_pred, y_scores)
    
    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.false_positive_rate == 0.0
    assert metrics.false_negative_rate == 0.0

def test_compute_metrics_all_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    y_scores = np.array([0.8, 0.9, 0.1, 0.2])
    
    metrics = _compute_metrics(y_true, y_pred, y_scores)
    
    assert metrics.accuracy == 0.0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.roc_auc == 0.0
    assert metrics.false_positive_rate == 1.0
    assert metrics.false_negative_rate == 1.0

def test_compute_metrics_single_class():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 1, 0, 1])
    y_scores = np.array([0.1, 0.8, 0.2, 0.9])
    
    metrics = _compute_metrics(y_true, y_pred, y_scores)
    
    # ROC-AUC should fallback to 0.5 when only one class is present
    assert metrics.roc_auc == 0.5
    assert metrics.accuracy == 0.5

def test_evaluate_detector():
    # 2 clean models (scores 0.1, 0.2), 2 modified models (scores 0.8, 0.9)
    mock_detector = MockDetector([0.1, 0.2, 0.8, 0.9])
    
    metrics = evaluate_detector(
        mock_detector,
        clean_model_paths=["clean1.pt", "clean2.pt"],
        modified_model_paths=["mod1.pt", "mod2.pt"],
        verbose=False
    )
    
    assert metrics.accuracy == 1.0
    assert metrics.roc_auc == 1.0

def test_evaluate_detector_from_weights():
    mock_detector = MockDetector([0.1, 0.8])
    
    metrics = evaluate_detector_from_weights(
        mock_detector,
        clean_weights=[torch.randn(10)],
        modified_weights=[torch.randn(10)],
        verbose=False
    )
    
    assert metrics.accuracy == 1.0

def test_metrics_to_dict():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_scores = np.array([0.2, 0.8])
    
    metrics = _compute_metrics(y_true, y_pred, y_scores)
    d = metrics.to_dict()
    
    assert isinstance(d, dict)
    assert 'accuracy' in d
    assert 'roc_auc' in d
    assert 'confusion_matrix' not in d  # Only scalars should be in dict
