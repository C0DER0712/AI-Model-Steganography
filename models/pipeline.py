"""End-to-end steganographic embedding pipeline.

Wires together the host model, weight extraction, Model X-Ray representation,
encoder, decoder, and frozen differentiable detector into a single
:class:`torch.nn.Module`.

Gradient flow
-------------
* **Payload recovery**: decoder loss → decoder parameters, encoder parameters.
* **Weight distortion**: MSE on representations → encoder parameters.
* **Detector evasion**: frozen-detector loss → (flows through frozen activations)
  → modified representation → encoder parameters.
* **Classification preservation**: cross-entropy on host model logits →
  (via STE through weight reconstruction) → encoder parameters.

The frozen detector's *parameters* never receive gradient updates.  The
encoder's parameters do, because the modified representation tensor produced
by the encoder carries ``requires_grad=True`` and the detector's activations
are differentiable with respect to their *inputs*.

Weight reconstruction uses a Straight-Through Estimator (STE) for the
non-differentiable byte-packing step, allowing the classification loss to
contribute gradients to the encoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from evaluation.differentiable_detector import DifferentiableDetector, DifferentiableDetectorConfig
from models.decoder import DensePayloadDecoder, DecoderConfig, build_decoder
from models.encoder import EncoderConfig, WeightPayloadEncoder, build_encoder
from models.host_models import HostModelAdapter, HostModelConfig, HostModelName, build_host_model
from training.losses import LossInputs
from utils.representation import channels_to_weights, weights_to_channels
from utils.weights import WeightTensor, extract_weights, flatten_weights


@dataclass(frozen=True)
class PipelineConfig:
    """Full pipeline configuration."""

    host_model_name: HostModelName = "resnet18"
    host_model_num_classes: int = 1000
    host_model_pretrained: bool = False
    host_model_checkpoint: str | None = None
    train_host_model: bool = False
    encoder: EncoderConfig = None          # type: ignore[assignment]
    decoder: DecoderConfig = None          # type: ignore[assignment]
    detector: DifferentiableDetectorConfig = None  # type: ignore[assignment]
    payload_bits: int = 8192  # 1024 bytes
    payload_replicas: int = 1
    run_classification: bool = True
    run_detector: bool = True
    reference_decoding: bool = True

    def __post_init__(self) -> None:
        if self.encoder is None:
            object.__setattr__(self, "encoder", EncoderConfig(payload_dim=self.payload_bits))
        if self.decoder is None:
            object.__setattr__(self, "decoder", DecoderConfig())
        if self.detector is None:
            object.__setattr__(self, "detector", DifferentiableDetectorConfig())


class EmbeddingPipeline(nn.Module):
    """Full steganographic embedding pipeline."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        super().__init__()
        cfg = config or PipelineConfig()
        self.config = cfg

        self.host_model: HostModelAdapter = build_host_model(
            cfg.host_model_name,
            num_classes=cfg.host_model_num_classes,
            pretrained=cfg.host_model_pretrained,
        )
        if cfg.host_model_checkpoint is not None:
            self._load_host_checkpoint(cfg.host_model_checkpoint)
        self.encoder: WeightPayloadEncoder = build_encoder(cfg.encoder)
        self.decoder: DensePayloadDecoder = build_decoder(cfg.decoder)
        self.detector: DifferentiableDetector = DifferentiableDetector(cfg.detector)

        if not cfg.train_host_model:
            for param in self.host_model.parameters():
                param.requires_grad_(False)

        self.detector.freeze()

        self._cached_weight_records: list[WeightTensor] | None = None
        self._cached_original_repr: torch.Tensor | None = None
        if not cfg.train_host_model:
            with torch.no_grad():
                weight_records = extract_weights(self.host_model.model)
                flat_weights = flatten_weights(weight_records)
                channels_uint8 = weights_to_channels(flat_weights)
                cached_repr = torch.from_numpy(channels_uint8.astype(np.float32))
            self._cached_weight_records = weight_records
            self.register_buffer("_cached_original_repr_buf", cached_repr, persistent=False)

    def _load_host_checkpoint(self, checkpoint_path: str) -> None:
        """Load a converged task-model checkpoint before payload embedding."""
        raw_path = checkpoint_path or "/kaggle/input/models/cs23b2013/mobile-net-pretrained/pytorch/default/1/mobilenet_v2_cifar10.pt"
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Host-model checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError("Host-model checkpoint must contain a state dictionary.")
        self.host_model.load_state_dict(state_dict, strict=True)

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        payload_bits: torch.Tensor,
    ) -> LossInputs:
        """Run the full pipeline for one training step."""
        device = images.device
        image_batch_size = images.shape[0]

        num_replicas = max(1, min(self.config.payload_replicas, image_batch_size))

        if payload_bits.ndim == 1:
            payload_bits = payload_bits.unsqueeze(0).expand(num_replicas, -1)
        else:
            payload_bits = payload_bits[:num_replicas]
        payload_bits = payload_bits.to(device=device, dtype=torch.float32)

        if self._cached_weight_records is not None:
            weight_records = self._cached_weight_records
            original_repr = self._cached_original_repr_buf.to(device)
        else:
            weight_records = extract_weights(self.host_model.model)
            flat_weights = flatten_weights(weight_records).to(device)
            channels_uint8 = weights_to_channels(flat_weights)
            original_repr = torch.from_numpy(channels_uint8.astype(np.float32)).to(device)

        num_weight_values = sum(record.values.numel() for record in weight_records)
        original_repr_batch = original_repr.unsqueeze(0).expand(num_replicas, -1, -1, -1)

        # ---- Encoder: produce modified representation ----
        modified_repr_batch = self.encoder(original_repr_batch, payload_bits)

        # Unpack tuple output FIRST if encoder returned (modified_repr, capacity_gate)
        if isinstance(modified_repr_batch, tuple):
            modified_repr_batch, capacity_gate = modified_repr_batch
        else:
            capacity_gate = None

        # Quantize the Tensor after unpacking
        modified_repr_batch = _quantize_byte_channels_ste(modified_repr_batch)

        # ---- Decoder: recover payload from each replica ----
        if self.config.reference_decoding:
            decoder_input = modified_repr_batch - original_repr_batch
        else:
            decoder_input = modified_repr_batch

        payload_logits = self.decoder(decoder_input, self.config.payload_bits)
        payload_logits_flat = payload_logits.reshape(num_replicas, -1)[
            :, : self.config.payload_bits
        ]

        # ---- Detector: score the modified representation ----
        if self.config.run_detector:
            detector_logits = self.detector(modified_repr_batch).squeeze(-1)
            detector_targets = torch.zeros_like(detector_logits)
        else:
            detector_logits = None
            detector_targets = None

        # ---- Classification with modified weights (STE) ----
        if self.config.run_classification:
            modified_repr_single = modified_repr_batch[0]
            modified_flat = _ChannelsToWeightsSTE.apply(
                modified_repr_single, len(weight_records), num_weight_values
            )
            modified_params = _rebuild_params(
                modified_flat, weight_records, self.host_model.model
            )
            classification_logits = self.host_model.functional_forward(images, modified_params)
            classification_targets = labels
        else:
            classification_logits = None
            classification_targets = None

        return LossInputs(
            classification_logits=classification_logits,
            classification_targets=classification_targets,
            payload_logits=payload_logits_flat,
            payload_targets=payload_bits.reshape(num_replicas, -1)[
                :, : self.config.payload_bits
            ],
            modified_weights=modified_repr_batch,
            original_weights=original_repr_batch.detach(),
            detector_logits=detector_logits,
            detector_targets=detector_targets,
            capacity_gate=capacity_gate,
        )

    def encode(
        self,
        payload_bits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a payload into the host model weights (inference only)."""
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            if payload_bits.ndim == 1:
                payload_bits = payload_bits.unsqueeze(0)
            payload_bits = payload_bits.to(device=device, dtype=torch.float32)
            weight_records = extract_weights(self.host_model.model)
            flat_weights = flatten_weights(weight_records).to(device)
            channels_uint8 = weights_to_channels(flat_weights)
            original_repr = torch.from_numpy(channels_uint8.astype(np.float32)).to(device)
            original_repr_batch = original_repr.unsqueeze(0)
            modified_repr = self.encoder(original_repr_batch, payload_bits)
            if isinstance(modified_repr, tuple):
                modified_repr, _ = modified_repr
            modified_repr = _quantize_byte_channels_ste(modified_repr)
        return modified_repr, original_repr_batch

    def decode(
        self,
        modified_repr: torch.Tensor,
        original_repr: torch.Tensor | None = None,
        num_bits: int | None = None,
    ) -> torch.Tensor:
        """Decode payload bits from a modified representation."""
        self.eval()
        with torch.no_grad():
            bits = num_bits or self.config.payload_bits
            if self.config.reference_decoding:
                if original_repr is None:
                    device = modified_repr.device
                    if self._cached_original_repr_buf is not None:
                        orig = self._cached_original_repr_buf.to(device).float()
                    else:
                        weight_records = extract_weights(self.host_model.model)
                        flat = flatten_weights(weight_records).to(device)
                        orig = torch.from_numpy(
                            weights_to_channels(flat).astype("float32")
                        ).to(device)
                    orig = orig.unsqueeze(0).expand_as(modified_repr)
                else:
                    orig = original_repr.to(modified_repr.device).float()
                decoder_input = modified_repr.float() - orig
            else:
                decoder_input = modified_repr
            return self.decoder.decode(decoder_input, bits)


def build_pipeline(config: PipelineConfig | None = None) -> EmbeddingPipeline:
    return EmbeddingPipeline(config)


def _quantize_byte_channels_ste(channels: torch.Tensor) -> torch.Tensor:
    """Round representation values to valid bytes with identity gradients."""
    quantized = channels.round().clamp(0, 255)
    return channels + (quantized - channels).detach()


class _ChannelsToWeightsSTE(torch.autograd.Function):
    """Convert float representation channels back to float32 weights."""

    @staticmethod
    def forward(
        ctx: Any,
        channels_float: torch.Tensor,
        weight_records: list[WeightTensor],
        num_values: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(channels_float)
        ctx.num_values = num_values
        ctx.channels_shape = tuple(channels_float.shape)
        ctx.device = channels_float.device

        channels_np = (
            channels_float.detach()
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        weights_f32 = channels_to_weights(channels_np, num_values=num_values)
        return weights_f32.to(channels_float.device)

    @staticmethod
    def backward(ctx: Any, grad_weights: torch.Tensor):
        c, h, w = ctx.channels_shape
        n = ctx.num_values
        hw = h * w

        grad_per_pixel = torch.zeros(hw, device=grad_weights.device, dtype=torch.float32)
        num_real = min(n, hw)
        grad_per_pixel[:num_real] = grad_weights.reshape(-1)[:num_real]

        grad_channels = torch.zeros(c, h, w, device=grad_weights.device, dtype=torch.float32)
        for ci in range(c):
            grad_channels[ci] = grad_per_pixel.reshape(h, w)
        return grad_channels / float(c), None, None


def _rebuild_params(
    modified_flat: torch.Tensor,
    weight_records: list[WeightTensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Reconstruct a parameter dict for ``functional_call`` from a flat tensor."""
    params: dict[str, torch.Tensor] = {}
    state = model.state_dict()
    device = modified_flat.device

    learnable_names: frozenset[str] = frozenset(
        name for name, _ in model.named_parameters()
    )

    offset = 0
    float_index = {rec.name: rec for rec in weight_records}

    for name, tensor in state.items():
        if name in float_index:
            rec = float_index[name]
            size = rec.values.numel()
            if name in learnable_names:
                values = modified_flat[offset: offset + size].reshape(rec.shape)
                params[name] = values.to(device=device, dtype=tensor.dtype)
            else:
                params[name] = tensor.to(device=device).detach()
            offset += size
        else:
            params[name] = tensor.to(device=device)

    return params