#!/usr/bin/env python3
"""Decode a payload from modified host weights without a clean host model.

The experiment checkpoint supplies the trained reveal network/configuration;
the modified host checkpoint supplies the carrier. No clean host checkpoint or
reference normalization statistics are loaded by this script.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.decoder import DecoderConfig, DensePayloadDecoder
from models.host_models import build_host_model
from utils.payload import SUPPORTED_PAYLOAD_SIZES, tensor_to_payload
from utils.weights import extract_parameters, flatten_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Training checkpoint containing decoder.* weights and config.")
    parser.add_argument("--modified-checkpoint", type=Path, required=True,
                        help="The embedded host model state_dict (no clean host needed).")
    parser.add_argument("--host-model", default="mobilenet_v2")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--payload-size", choices=SUPPORTED_PAYLOAD_SIZES, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Path for the recovered payload bytes.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _unwrap_state_dict(value: object) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict):
        raise ValueError("Modified checkpoint must contain a state dictionary.")
    state = value
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = state.get(key)
        if isinstance(candidate, dict):
            state = candidate
            break
    if not state or not all(torch.is_tensor(tensor) for tensor in state.values()):
        raise ValueError("Could not find a tensor-only model state_dict.")
    for prefix in ("host_model.model.", "model."):
        if all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: tensor for key, tensor in state.items()}
            break
    return state


def main() -> int:
    args = parse_args()
    device = _device(args.device)
    training = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    pipeline_cfg = training.get("config", {}).get("pipeline", {})
    if pipeline_cfg.get("reference_decoding", True):
        raise ValueError(
            "This checkpoint was trained for reference decoding. Train with "
            "configs/experiment_blind.toml before using this blind decoder."
        )
    decoder_values = pipeline_cfg.get("decoder", {})
    accepted = {field.name for field in fields(DecoderConfig)}
    decoder_cfg = DecoderConfig(**{
        key: value for key, value in decoder_values.items() if key in accepted
    })
    payload_bits = SUPPORTED_PAYLOAD_SIZES[args.payload_size] * 8
    state = training.get("model", training)
    decoder_state = {
        key[len("decoder."):]: value
        for key, value in state.items()
        if key.startswith("decoder.")
    }
    if not decoder_state:
        raise ValueError("Training checkpoint has no decoder.* state entries.")
    decoder = DensePayloadDecoder(decoder_cfg)
    decoder.load_state_dict(decoder_state, strict=True)
    decoder.to(device).eval()

    host = build_host_model(args.host_model, num_classes=args.num_classes, pretrained=False)
    modified_state = _unwrap_state_dict(
        torch.load(args.modified_checkpoint, map_location="cpu", weights_only=True)
    )
    host.model.load_state_dict(modified_state, strict=True)
    modified_flat = flatten_weights(extract_parameters(host.model))
    recovered_bits = decoder.decode_flat_weights(modified_flat, payload_bits)[0]
    recovered_bytes = tensor_to_payload(recovered_bits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(recovered_bytes)
    print(f"Recovered {len(recovered_bytes):,} bytes to {args.output.resolve()}")
    print(f"Blind input: modified host weights only | decoder device: {device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
