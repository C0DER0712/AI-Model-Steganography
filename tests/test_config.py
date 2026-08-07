from pathlib import Path

import pytest

from utils.config import load_config


def test_load_default_config() -> None:
    config = load_config(Path("configs/default.toml"))

    assert config["safety"]["payload_type"] == "benign_random"
    assert config["safety"]["allow_malware_generation"] is False
    assert "accuracy_drop" in config["research"]["metrics"]


def test_load_config_rejects_unknown_extension(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project: test", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported configuration format"):
        load_config(config_path)
