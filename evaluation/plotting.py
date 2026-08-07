"""Publication-quality plots for detector evaluation.

All functions return a `matplotlib.figure.Figure` and optionally save to disk.
Plots follow a consistent academic style with proper axis labels, legends,
and 300 DPI resolution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

if TYPE_CHECKING:
    from evaluation.metrics import DetectorMetrics

logger = logging.getLogger(__name__)

# Publication-quality plot style
_STYLE = {
    'figure.dpi': 300,
    'font.size': 11,
    'font.family': 'serif',
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (6, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Refined color palette
COLORS = {
    'primary': '#2563EB',
    'secondary': '#DC2626',
    'accent': '#059669',
    'neutral': '#6B7280',
    'clean': '#2563EB',
    'malicious': '#DC2626',
}

def _save_figure(fig: plt.Figure, output_path: Union[str, Path, None]) -> None:
    """Helper to save a figure if path is provided and close it."""
    if output_path is not None:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

def plot_roc_curve(
    metrics: 'DetectorMetrics',
    *,
    title: Optional[str] = None,
    output_path: Union[str, Path, None] = None
) -> plt.Figure:
    """Plots the ROC curve for the detector.

    Args:
        metrics: The computed DetectorMetrics.
        title: Optional plot title.
        output_path: Optional path to save the plot.

    Returns:
        The matplotlib Figure.
    """
    with plt.style.context(_STYLE):
        fig, ax = plt.subplots()
        
        if len(np.unique(metrics.y_true)) > 1:
            fpr, tpr, _ = roc_curve(metrics.y_true, metrics.y_scores)
            ax.plot(fpr, tpr, color=COLORS['primary'], lw=2, label=f'ROC Curve (AUC = {metrics.roc_auc:.3f})')
            ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS['primary'])
        else:
            # Handle edge case where ROC can't be computed well
            ax.plot([0, 1], [0, 1], color=COLORS['primary'], lw=2, label=f'ROC Curve (AUC = {metrics.roc_auc:.3f})')
            logger.warning("plot_roc_curve: Only one class present, plotting diagonal.")
            
        # Diagonal reference line
        ax.plot([0, 1], [0, 1], color=COLORS['neutral'], lw=1, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        if title:
            ax.set_title(title)
        ax.legend(loc="lower right")
        
    _save_figure(fig, output_path)
    return fig

def plot_confusion_matrix(
    metrics: 'DetectorMetrics',
    *,
    title: Optional[str] = None,
    normalize: bool = False,
    output_path: Union[str, Path, None] = None
) -> plt.Figure:
    """Plots the confusion matrix.

    Args:
        metrics: The computed DetectorMetrics.
        title: Optional plot title.
        normalize: Whether to normalize values to percentages.
        output_path: Optional path to save the plot.

    Returns:
        The matplotlib Figure.
    """
    cm = metrics.confusion_matrix.copy()
    
    if normalize:
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    else:
        cm_norm = cm

    with plt.style.context(_STYLE):
        # Override grid for heatmap
        with plt.rc_context({'axes.grid': False}):
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
            
            # Colorbar
            fig.colorbar(im, ax=ax)
            
            classes = ['Clean', 'Modified']
            tick_marks = np.arange(len(classes))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(classes)
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(classes)
            
            fmt = '.2f' if normalize else 'd'
            thresh = cm_norm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm_norm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm_norm[i, j] > thresh else "black")
            
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            if title:
                ax.set_title(title)
                
    _save_figure(fig, output_path)
    return fig

def plot_metrics_bar(
    metrics: 'DetectorMetrics',
    *,
    title: Optional[str] = None,
    output_path: Union[str, Path, None] = None
) -> plt.Figure:
    """Plots a horizontal bar chart of the scalar metrics.

    Args:
        metrics: The computed DetectorMetrics.
        title: Optional plot title.
        output_path: Optional path to save the plot.

    Returns:
        The matplotlib Figure.
    """
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'FPR', 'FNR']
    values = [
        metrics.accuracy, metrics.precision, metrics.recall, 
        metrics.f1_score, metrics.roc_auc, 
        metrics.false_positive_rate, metrics.false_negative_rate
    ]
    
    # Colors: Good metrics in primary, error metrics in secondary
    colors = [COLORS['primary']] * 5 + [COLORS['secondary']] * 2
    
    with plt.style.context(_STYLE):
        fig, ax = plt.subplots()
        y_pos = np.arange(len(labels))
        
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlim([0, 1.05])
        
        if title:
            ax.set_title(title)
            
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
            
    _save_figure(fig, output_path)
    return fig

def plot_score_distribution(
    metrics: 'DetectorMetrics',
    *,
    title: Optional[str] = None,
    bins: int = 30,
    output_path: Union[str, Path, None] = None
) -> plt.Figure:
    """Plots the distribution of anomaly scores.

    Args:
        metrics: The computed DetectorMetrics.
        title: Optional plot title.
        bins: Number of histogram bins.
        output_path: Optional path to save the plot.

    Returns:
        The matplotlib Figure.
    """
    y_true = metrics.y_true
    y_scores = metrics.y_scores
    
    clean_scores = y_scores[y_true == 0]
    mod_scores = y_scores[y_true == 1]
    
    with plt.style.context(_STYLE):
        fig, ax = plt.subplots()
        
        if len(clean_scores) > 0:
            ax.hist(clean_scores, bins=bins, alpha=0.6, color=COLORS['clean'], 
                    label='Clean Models', density=False)
        if len(mod_scores) > 0:
            ax.hist(mod_scores, bins=bins, alpha=0.6, color=COLORS['malicious'], 
                    label='Modified Models', density=False)
            
        # Threshold line
        ax.axvline(x=0.5, color=COLORS['neutral'], linestyle='--', lw=1.5, label='Threshold (0.5)')
        
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Count')
        if title:
            ax.set_title(title)
        ax.legend()
        
    _save_figure(fig, output_path)
    return fig

def plot_detection_comparison(
    metrics_dict: Dict[str, 'DetectorMetrics'],
    *,
    title: Optional[str] = None,
    output_path: Union[str, Path, None] = None
) -> plt.Figure:
    """Plots a comparison of metrics across multiple models/configurations.

    Args:
        metrics_dict: Dictionary mapping names to DetectorMetrics.
        title: Optional plot title.
        output_path: Optional path to save the plot.

    Returns:
        The matplotlib Figure.
    """
    labels = list(metrics_dict.keys())
    
    accs = [m.accuracy for m in metrics_dict.values()]
    precs = [m.precision for m in metrics_dict.values()]
    recs = [m.recall for m in metrics_dict.values()]
    f1s = [m.f1_score for m in metrics_dict.values()]
    aucs = [m.roc_auc for m in metrics_dict.values()]
    
    x = np.arange(len(labels))
    width = 0.15
    
    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(6, len(labels)*1.5), 5))
        
        ax.bar(x - 2*width, accs, width, label='Accuracy', color='#3B82F6')
        ax.bar(x - width, precs, width, label='Precision', color='#10B981')
        ax.bar(x, recs, width, label='Recall', color='#F59E0B')
        ax.bar(x + width, f1s, width, label='F1-Score', color='#8B5CF6')
        ax.bar(x + 2*width, aucs, width, label='ROC-AUC', color='#EC4899')
        
        ax.set_ylabel('Score')
        if title:
            ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=5)
        ax.set_ylim([0, 1.05])
        
    _save_figure(fig, output_path)
    return fig

def plot_metrics_summary(
    metrics: 'DetectorMetrics',
    output_dir: Union[str, Path],
    *,
    prefix: str = 'detector'
) -> List[Path]:
    """Generates all standard plots and saves them to a directory.

    Args:
        metrics: The computed DetectorMetrics.
        output_dir: Directory to save the plots.
        prefix: Prefix for the saved filenames.

    Returns:
        List of paths to the saved figures.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    # 1. ROC Curve
    roc_path = out_dir / f"{prefix}_roc_curve.png"
    plot_roc_curve(metrics, title="ROC Curve", output_path=roc_path)
    saved_paths.append(roc_path)
    
    # 2. Confusion Matrix
    cm_path = out_dir / f"{prefix}_confusion_matrix.png"
    plot_confusion_matrix(metrics, title="Confusion Matrix", output_path=cm_path)
    saved_paths.append(cm_path)
    
    # 3. Metrics Bar Chart
    bar_path = out_dir / f"{prefix}_metrics_bar.png"
    plot_metrics_bar(metrics, title="Evaluation Metrics", output_path=bar_path)
    saved_paths.append(bar_path)
    
    # 4. Score Distribution
    dist_path = out_dir / f"{prefix}_score_dist.png"
    plot_score_distribution(metrics, title="Anomaly Score Distribution", output_path=dist_path)
    saved_paths.append(dist_path)
    
    return saved_paths
