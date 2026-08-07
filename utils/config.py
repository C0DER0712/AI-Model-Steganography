"""Configuration loading utilities."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any


Config = dict[str, Any]


def load_config(path: str | Path) -> Config:
    """Load a JSON or TOML configuration file.

    Args:
        path: Path to a `.json` or `.toml` configuration file.

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file extension is unsupported or the top-level
            object is not a mapping.
    """

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_path.suffix == ".json":
        with config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    elif config_path.suffix == ".toml":
        with config_path.open("rb") as file:
            data = tomllib.load(file)
    else:
        raise ValueError(
            f"Unsupported configuration format '{config_path.suffix}'. "
            "Use .json or .toml."
        )

    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping.")

    return data
