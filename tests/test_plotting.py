import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from evaluation.metrics import DetectorMetrics, _compute_metrics
from evaluation.plotting import (
    plot_confusion_matrix,
    plot_detection_comparison,
    plot_metrics_bar,
    plot_metrics_summary,
    plot_roc_curve,
    plot_score_distribution,
)

@pytest.fixture
def dummy_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])
    
    return _compute_metrics(y_true, y_pred, y_scores)

@pytest.fixture
def dummy_metrics_single_class():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 1])
    y_scores = np.array([0.1, 0.2, 0.8])
    
    return _compute_metrics(y_true, y_pred, y_scores)

def test_plot_roc_curve(dummy_metrics):
    fig = plot_roc_curve(dummy_metrics, title="Test ROC")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_roc_curve_single_class(dummy_metrics_single_class):
    fig = plot_roc_curve(dummy_metrics_single_class)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_confusion_matrix(dummy_metrics):
    fig = plot_confusion_matrix(dummy_metrics, title="Test CM", normalize=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_metrics_bar(dummy_metrics):
    fig = plot_metrics_bar(dummy_metrics)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_score_distribution(dummy_metrics):
    fig = plot_score_distribution(dummy_metrics)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_detection_comparison(dummy_metrics, dummy_metrics_single_class):
    metrics_dict = {
        "ModelA": dummy_metrics,
        "ModelB": dummy_metrics_single_class
    }
    fig = plot_detection_comparison(metrics_dict)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_metrics_summary(dummy_metrics):
    with tempfile.TemporaryDirectory() as temp_dir:
        saved_paths = plot_metrics_summary(dummy_metrics, temp_dir, prefix="test")
        
        assert len(saved_paths) == 4
        for path in saved_paths:
            assert path.exists()
            assert path.stat().st_size > 0
