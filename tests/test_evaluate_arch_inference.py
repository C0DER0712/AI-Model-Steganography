import torch

from evaluation.differentiable_detector import DifferentiableDetector, DifferentiableDetectorConfig
from scripts.evaluate import _infer_detector_config


def test_detector_config_is_inferred_from_checkpoint_shapes():
    state = {
        "detector.hpf.weight": torch.empty(30, 4, 3, 3),
        "detector.stem.0.weight": torch.empty(32, 30, 3, 3),
        "detector.classifier.1.weight": torch.empty(128, 256),
    }
    for index in range(4):
        state[f"detector.type1_blocks.{index}.conv1.weight"] = torch.empty(32, 32, 3, 3)
    for index in range(3):
        in_channels = 32 * (2 ** index)
        out_channels = min(in_channels * 2, 512)
        state[f"detector.type2_blocks.{index}.conv1.weight"] = torch.empty(
            out_channels, in_channels, 3, 3
        )

    inferred = _infer_detector_config(state)

    assert inferred == {
        "input_channels": 4,
        "hpf_channels": 30,
        "base_channels": 32,
        "num_type1_blocks": 4,
        "num_type2_blocks": 3,
        "fc_hidden_dim": 128,
    }
    model = DifferentiableDetector(DifferentiableDetectorConfig(**inferred))
    assert model.hpf.weight.shape == state["detector.hpf.weight"].shape
    assert model.stem[0].weight.shape == state["detector.stem.0.weight"].shape
    assert len(model.type1_blocks) == 4
    assert len(model.type2_blocks) == 3


def test_detector_inference_uses_default_architecture_without_checkpoint_state():
    inferred = _infer_detector_config({})
    defaults = DifferentiableDetectorConfig()

    assert inferred["num_type1_blocks"] == defaults.num_type1_blocks
    assert inferred["num_type2_blocks"] == defaults.num_type2_blocks
    assert inferred["hpf_channels"] == defaults.hpf_channels
    assert inferred["base_channels"] == defaults.base_channels
