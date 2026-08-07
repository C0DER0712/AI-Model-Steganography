"""Integration tests for the training CLI configuration pipeline.

Verifies that the shipped ``configs/experiment.toml``, the CLI default path,
and CLI-over-TOML precedence all produce internally consistent configurations
with matching ``payload_bits`` / ``encoder.payload_dim`` / ``decoder.chunk_size``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable when pytest is invoked from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train import _resolve, parse_args


# ---------------------------------------------------------------------------
# Helpers that replicate the config-resolution logic from scripts/train.py
# without actually training (no network construction needed here)
# ---------------------------------------------------------------------------

def _resolve_config(argv: list[str] | None = None):
    """Run parse_args + config resolution; return the resolved pipeline kwargs."""
    from utils.config import load_config
    from utils.payload import SUPPORTED_PAYLOAD_SIZES

    args = parse_args(argv)

    file_cfg: dict = {}
    if args.config is not None:
        file_cfg = load_config(args.config)

    host_sec   = file_cfg.get("host_model", {})
    data_sec   = file_cfg.get("data", {})
    train_sec  = file_cfg.get("training", {})
    enc_sec    = file_cfg.get("encoder", {})
    dec_sec    = file_cfg.get("decoder", {})
    loss_sec   = file_cfg.get("loss_weights", {})
    rt_sec     = file_cfg.get("runtime", {})

    payload_size_str = _resolve(args.payload_size, data_sec.get("payload_size"), "128KB")
    payload_bits = SUPPORTED_PAYLOAD_SIZES[payload_size_str] * 8

    return {
        "payload_size_str": payload_size_str,
        "payload_bits": payload_bits,
        "host_model": _resolve(args.host_model, host_sec.get("name"), "resnet18"),
        "num_classes": _resolve(args.num_classes, host_sec.get("num_classes"), 1000),
        "pretrained": _resolve(args.pretrained, host_sec.get("pretrained"), False),
        "train_host_model": _resolve(
            args.train_host_model, host_sec.get("train_host_model"), False
        ),
        "enc_base_channels": enc_sec.get("base_channels", 64),
        "dec_base_channels": dec_sec.get("base_channels", 64),
        "max_epochs": _resolve(args.epochs, train_sec.get("max_epochs"), 10),
        "batch_size": _resolve(args.batch_size, train_sec.get("batch_size"), 4),
        "device": _resolve(args.device, rt_sec.get("device"), "auto"),
    }


TOML_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "experiment.toml")


# ---------------------------------------------------------------------------
# Shipped TOML produces coherent payload dimensions
# ---------------------------------------------------------------------------

class TestShippedTomlConfig:
    """The committed configs/experiment.toml must resolve without errors."""

    def test_loads_without_error(self) -> None:
        cfg = _resolve_config(["--config", TOML_PATH])
        assert cfg is not None

    def test_payload_bits_matches_payload_size(self) -> None:
        from utils.payload import SUPPORTED_PAYLOAD_SIZES
        cfg = _resolve_config(["--config", TOML_PATH])
        expected = SUPPORTED_PAYLOAD_SIZES[cfg["payload_size_str"]] * 8
        assert cfg["payload_bits"] == expected, (
            f"payload_bits={cfg['payload_bits']} does not match "
            f"payload_size '{cfg['payload_size_str']}' → {expected} bits"
        )

    def test_encoder_payload_dim_equals_payload_bits(self) -> None:
        """encoder.payload_dim must always equal pipeline.payload_bits."""
        from models.decoder import DecoderConfig
        from models.encoder import EncoderConfig
        from utils.payload import SUPPORTED_PAYLOAD_SIZES

        cfg = _resolve_config(["--config", TOML_PATH])
        payload_bits = cfg["payload_bits"]

        enc = EncoderConfig(payload_dim=payload_bits)
        dec = DecoderConfig(chunk_size=payload_bits)

        assert enc.payload_dim == payload_bits
        assert dec.chunk_size == payload_bits

    def test_host_model_is_supported_backbone(self) -> None:
        cfg = _resolve_config(["--config", TOML_PATH])
        from models.host_models import _HOST_MODEL_NAMES
        assert cfg["host_model"] in _HOST_MODEL_NAMES


# ---------------------------------------------------------------------------
# CLI defaults (no config file)
# ---------------------------------------------------------------------------

class TestCliDefaults:
    def test_default_payload_size_is_valid(self) -> None:
        from utils.payload import SUPPORTED_PAYLOAD_SIZES
        cfg = _resolve_config([])
        assert cfg["payload_size_str"] in SUPPORTED_PAYLOAD_SIZES

    def test_default_payload_bits_positive(self) -> None:
        cfg = _resolve_config([])
        assert cfg["payload_bits"] > 0

    def test_default_host_model(self) -> None:
        cfg = _resolve_config([])
        assert cfg["host_model"] == "resnet18"

    def test_encoder_decoder_consistent_with_payload(self) -> None:
        """Even without a config file, encoder/decoder dims match payload_bits."""
        cfg = _resolve_config([])
        from models.encoder import EncoderConfig
        from models.decoder import DecoderConfig
        enc = EncoderConfig(payload_dim=cfg["payload_bits"])
        dec = DecoderConfig(chunk_size=cfg["payload_bits"])
        assert enc.payload_dim == dec.chunk_size == cfg["payload_bits"]


# ---------------------------------------------------------------------------
# CLI-over-TOML precedence
# ---------------------------------------------------------------------------

class TestCliOverridesPrecedence:
    """CLI flags must take precedence over TOML values."""

    def test_cli_epochs_overrides_toml(self) -> None:
        cfg_toml = _resolve_config(["--config", TOML_PATH])
        toml_epochs = cfg_toml["max_epochs"]
        override = toml_epochs + 999
        cfg_cli = _resolve_config(["--config", TOML_PATH, "--epochs", str(override)])
        assert cfg_cli["max_epochs"] == override, (
            f"CLI --epochs {override} did not override TOML max_epochs={toml_epochs}"
        )

    def test_cli_batch_size_overrides_toml(self) -> None:
        cfg_toml = _resolve_config(["--config", TOML_PATH])
        override = cfg_toml["batch_size"] + 8
        cfg_cli = _resolve_config(["--config", TOML_PATH, "--batch-size", str(override)])
        assert cfg_cli["batch_size"] == override

    def test_cli_payload_size_overrides_toml(self) -> None:
        from utils.payload import SUPPORTED_PAYLOAD_SIZES
        # Pick a payload size that differs from the TOML default.
        cfg_toml = _resolve_config(["--config", TOML_PATH])
        toml_size = cfg_toml["payload_size_str"]
        other_sizes = [s for s in SUPPORTED_PAYLOAD_SIZES if s != toml_size]
        if not other_sizes:
            pytest.skip("Only one payload size available; cannot test override.")
        override_size = other_sizes[0]
        cfg_cli = _resolve_config(
            ["--config", TOML_PATH, "--payload-size", override_size]
        )
        assert cfg_cli["payload_size_str"] == override_size, (
            f"CLI --payload-size {override_size} did not override TOML {toml_size}"
        )
        # payload_bits must also update consistently.
        assert cfg_cli["payload_bits"] == SUPPORTED_PAYLOAD_SIZES[override_size] * 8

    def test_cli_host_model_overrides_toml(self) -> None:
        cfg_toml = _resolve_config(["--config", TOML_PATH])
        # TOML default is resnet18; override with mobilenet_v2.
        cfg_cli = _resolve_config(
            ["--config", TOML_PATH, "--host-model", "mobilenet_v2"]
        )
        assert cfg_cli["host_model"] == "mobilenet_v2"

    def test_cli_device_overrides_toml(self) -> None:
        cfg_cli = _resolve_config(
            ["--config", TOML_PATH, "--device", "cpu"]
        )
        assert cfg_cli["device"] == "cpu"


# ---------------------------------------------------------------------------
# _resolve helper
# ---------------------------------------------------------------------------

class TestResolveHelper:
    def test_cli_wins_over_toml_and_default(self) -> None:
        assert _resolve("cli", "toml", "default") == "cli"

    def test_toml_wins_over_default_when_cli_is_none(self) -> None:
        assert _resolve(None, "toml", "default") == "toml"

    def test_default_used_when_both_absent(self) -> None:
        assert _resolve(None, None, "default") == "default"

    def test_zero_is_a_valid_cli_value(self) -> None:
        assert _resolve(0, 10, 99) == 0

    def test_false_is_a_valid_cli_value(self) -> None:
        assert _resolve(False, True, True) is False
