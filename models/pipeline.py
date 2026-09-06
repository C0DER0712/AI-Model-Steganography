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

Representation spaces
---------------------
The encoder and decoder operate in **float weight space**: a single-channel
image holding one normalised float32 value per weight. Reconstructing weights
for the classification objective is therefore an ordinary differentiable
affine map (``float_image_to_weights``) with no STE required.

The detector still consumes the **4-channel IEEE754 byte image**. That byte
packing is non-differentiable, so a Straight-Through Estimator
(``_WeightsToChannelsSTE``) bridges the modified float weights into the
detector's byte input, letting the detector-evasion loss still reach the
encoder.
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
from utils.representation import (
    float_image_to_weights,
    weights_to_channels,
    weights_to_float_image,
)
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
            # Keep frozen host permanently in eval so BatchNorm never runs
            # its in-place running-stat update during functional_call.
            # In train mode each forward corrupts running_mean/running_var,
            # dropping host accuracy from ~95 % to random within one epoch.
            self.host_model.eval()

        self.detector.freeze()

        self._cached_weight_records: list[WeightTensor] | None = None
        self._cached_original_repr: torch.Tensor | None = None
        # Normalisation stats for the float weight image, needed to map
        # encoder outputs back to real weights (classification) and to bytes
        # (detector). Set when the host model is frozen and its weights fixed.
        self._norm_mean: float = 0.0
        self._norm_scale: float = 1.0
        self._num_weight_values: int = 0
        if not cfg.train_host_model:
            with torch.no_grad():
                # BN running_mean/running_var are float buffers but have wildly
                # different magnitudes (running_var ≈ 100–400).  Including them in
                # extract_weights inflates norm_scale to ~380, so a normalised
                # delta of ±0.05 maps to ±19 in real weight space — enough to
                # destroy classification completely.  Restrict to learnable
                # parameters only so scale reflects actual weight magnitudes.
                _learnable_names = frozenset(
                    n for n, _ in self.host_model.model.named_parameters()
                )
                weight_records = [
                    r for r in extract_weights(self.host_model.model)
                    if r.name in _learnable_names
                ]
                flat_weights = flatten_weights(weight_records)
                float_image, stats = weights_to_float_image(flat_weights)
                cached_repr = torch.from_numpy(float_image).float()
            self._cached_weight_records = weight_records
            self._norm_mean = stats.mean
            self._norm_scale = stats.scale
            self._num_weight_values = stats.num_values
            # Buffer name kept for backward compatibility; now holds the
            # single-channel float image rather than the byte channels.
            self.register_buffer("_cached_original_repr_buf", cached_repr, persistent=False)

    def _load_host_checkpoint(self, checkpoint_path: str) -> None:
        """Load a converged task-model checkpoint before payload embedding."""
        raw_path = checkpoint_path or "/kaggle/input/models/cs23b2013/mobile-net-pretrained/pytorch/default/1/mobilenet_v2_cifar10.pt"
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Host-model checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break
        if not isinstance(state_dict, dict):
            raise ValueError("Host-model checkpoint must contain a state dictionary.")
        # The backbone lives at self.host_model.model, whose keys are NOT
        # "model."-prefixed. Strip that prefix if the checkpoint was saved
        # from the adapter so the keys line up, then load into the backbone.
        cleaned = {
            (k[len("model."):] if k.startswith("model.") else k): v
            for k, v in state_dict.items()
        }
        self.host_model.model.load_state_dict(cleaned, strict=True)

    # ------------------------------------------------------------------
    # Training-mode override
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "EmbeddingPipeline":
        """Set training mode, but keep a frozen host model in eval.

        ``nn.Module.train()`` recurses into every child including
        ``self.host_model``.  For a frozen host that must not have
        its BatchNorm running statistics corrupted by functional_call
        during training, we immediately restore eval mode afterward.
        """
        super().train(mode)
        if not self.config.train_host_model:
            self.host_model.eval()
        return self

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
            norm_mean = self._norm_mean
            norm_scale = self._norm_scale
        else:
            _learnable_names = frozenset(
                n for n, _ in self.host_model.model.named_parameters()
            )
            weight_records = [
                r for r in extract_weights(self.host_model.model)
                if r.name in _learnable_names
            ]
            flat_weights = flatten_weights(weight_records)
            float_image, stats = weights_to_float_image(flat_weights)
            original_repr = torch.from_numpy(float_image).float().to(device)
            norm_mean = stats.mean
            norm_scale = stats.scale

        num_weight_values = sum(record.values.numel() for record in weight_records)
        # original_repr is a single-channel float image (1, side, side).
        side = original_repr.shape[-1]
        original_repr_batch = original_repr.unsqueeze(0).expand(num_replicas, -1, -1, -1)

        # ---- Encoder: produce modified representation ----
        modified_repr_batch = self.encoder(original_repr_batch, payload_bits)

        # Unpack tuple output if encoder returned (modified_repr, capacity_gate).
        if isinstance(modified_repr_batch, tuple):
            modified_repr_batch, capacity_gate = modified_repr_batch
        else:
            capacity_gate = None

        # No byte quantization: encoder/decoder stay in continuous float space.

        # ---- Decoder: recover payload from each replica ----
        if self.config.reference_decoding:
            decoder_input = modified_repr_batch - original_repr_batch
        else:
            decoder_input = modified_repr_batch

        payload_logits = self.decoder(decoder_input, self.config.payload_bits)
        payload_logits_flat = payload_logits.reshape(num_replicas, -1)[
            :, : self.config.payload_bits
        ]

        # ---- Detector: score the modified weights in byte space ----
        # The detector consumes the 4-channel IEEE754 image, so denormalise
        # the modified float weights back to real values and byte-pack them
        # through the STE (the only non-differentiable hop; gradient passes
        # straight through it back to the encoder).
        if self.config.run_detector:
            modified_flat_batch = (
                modified_repr_batch.reshape(num_replicas, side * side) * norm_scale
                + norm_mean
            )
            detector_image = _WeightsToChannelsSTE.apply(modified_flat_batch, side)
            detector_logits = self.detector(detector_image).squeeze(-1)
            detector_targets = torch.zeros_like(detector_logits)
        else:
            detector_logits = None
            detector_targets = None

        # ---- Classification with modified weights (differentiable) ----
        # float_image_to_weights is a plain affine map, so reconstruction is
        # differentiable end-to-end — no STE needed on this path.
        if self.config.run_classification:
            modified_flat = float_image_to_weights(
                modified_repr_batch[0],
                mean=norm_mean,
                scale=norm_scale,
                num_values=num_weight_values,
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
            _learnable_names = frozenset(
                n for n, _ in self.host_model.model.named_parameters()
            )
            weight_records = [
                r for r in extract_weights(self.host_model.model)
                if r.name in _learnable_names
            ]
            flat_weights = flatten_weights(weight_records)
            float_image, _stats = weights_to_float_image(flat_weights)
            original_repr = torch.from_numpy(float_image).float().to(device)
            original_repr_batch = original_repr.unsqueeze(0)
            modified_repr = self.encoder(original_repr_batch, payload_bits)
            if isinstance(modified_repr, tuple):
                modified_repr, _ = modified_repr
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
                        _learnable_names = frozenset(
                            n for n, _ in self.host_model.model.named_parameters()
                        )
                        weight_records = [
                            r for r in extract_weights(self.host_model.model)
                            if r.name in _learnable_names
                        ]
                        flat = flatten_weights(weight_records)
                        float_image, _stats = weights_to_float_image(flat)
                        orig = torch.from_numpy(float_image).float().to(device)
                    orig = orig.unsqueeze(0).expand_as(modified_repr)
                else:
                    orig = original_repr.to(modified_repr.device).float()
                decoder_input = modified_repr.float() - orig
            else:
                decoder_input = modified_repr
            return self.decoder.decode(decoder_input, bits)


def build_pipeline(config: PipelineConfig | None = None) -> EmbeddingPipeline:
    return EmbeddingPipeline(config)


class _WeightsToChannelsSTE(torch.autograd.Function):
    """Byte-pack modified float32 weights into the 4-channel detector image.

    Only the detector consumes the byte representation; the encoder/decoder
    stay in float space. This STE bridges the two so the detector-evasion
    gradient still reaches the encoder.

    Forward:
        Denormalised modified weights (float32, shape ``(R, N)`` where
        ``N = side * side``) are byte-decomposed via
        :func:`utils.representation.weights_to_channels` (left unchanged) into
        the 4-channel IEEE754 image the SRNet-style detector expects, returned
        as float in ``[0, 255]`` with shape ``(R, 4, side, side)``.

    Backward (STE):
        Byte packing is a non-differentiable step function, so gradient is
        propagated straight through: each weight's gradient is the sum of the
        four byte channels that encode it, divided by the number of channels
        (4) to preserve scale.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        weights_flat: torch.Tensor,
        side: int,
    ) -> torch.Tensor:
        num_replicas, num_values = weights_flat.shape
        ctx.shape = (num_replicas, num_values)
        ctx.side = side

        weights_np = weights_flat.detach().cpu().numpy()
        channels = np.stack(
            [
                weights_to_channels(weights_np[r]).astype(np.float32)
                for r in range(num_replicas)
            ],
            axis=0,
        )  # (R, 4, side, side)
        return torch.from_numpy(channels).to(weights_flat.device)

    @staticmethod
    def backward(ctx: Any, grad_channels: torch.Tensor):  # type: ignore[override]
        num_replicas, num_values = ctx.shape
        side = ctx.side
        num_channels = grad_channels.shape[1]

        # STE: treat byte-packing as identity. Sum the per-channel gradients
        # back onto each weight pixel, then average across channels to keep
        # the gradient scale comparable to the forward magnitude.
        grad_pixels = grad_channels.sum(dim=1).reshape(num_replicas, side * side)
        grad_flat = grad_pixels[:, :num_values] / float(num_channels)
        return grad_flat, None


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