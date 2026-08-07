import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from evaluation.detector import DetectionResult, ModelXRayDetector

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

@pytest.fixture
def dummy_state_dict():
    model = DummyModel()
    return model.state_dict()

@pytest.fixture
def temp_model_file(dummy_state_dict):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "dummy.pt"
        torch.save(dummy_state_dict, path)
        yield path

def test_detection_result_dataclass():
    result = DetectionResult(
        is_malicious=True,
        confidence=0.8,
        anomaly_score=0.75,
        dist_to_benign=0.9,
        dist_to_malicious=0.1
    )
    assert result.is_malicious is True
    assert result.confidence == 0.8
    assert result.anomaly_score == 0.75

def test_detector_fallback_initialization():
    # Initialize without SRNet weights
    detector = ModelXRayDetector()
    assert detector.use_srnet is False

def test_predict_from_weight_tensor():
    detector = ModelXRayDetector()
    
    # Create a random weight tensor
    weights = torch.randn(1000)
    
    result = detector.predict_from_weight_tensor(weights)
    assert isinstance(result, DetectionResult)
    assert isinstance(result.is_malicious, bool)
    assert 0.0 <= result.anomaly_score <= 1.0

def test_predict_from_weights_dict(dummy_state_dict):
    detector = ModelXRayDetector()
    result = detector.predict_from_weights(dummy_state_dict)
    
    assert isinstance(result, DetectionResult)

def test_predict_from_file(temp_model_file):
    detector = ModelXRayDetector()
    result = detector.predict(temp_model_file)
    
    assert isinstance(result, DetectionResult)

def test_predict_file_not_found():
    detector = ModelXRayDetector()
    with pytest.raises(FileNotFoundError):
        detector.predict(Path("nonexistent_file_path_12345.pt"))

def test_predict_batch(temp_model_file):
    detector = ModelXRayDetector()
    results = detector.predict_batch([temp_model_file, temp_model_file])
    
    assert len(results) == 2
    assert all(isinstance(r, DetectionResult) for r in results)
