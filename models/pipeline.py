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
    """Full pipeline configuration.

    Attributes:
        host_model_name: Backbone for the host model.
        host_model_num_classes: Output class count.
        host_model_pretrained: Whether to use pretrained weights.
        train_host_model: Whether to allow the host model to be fine-tuned
            alongside the encoder/decoder.
        encoder: Encoder architecture config.
        decoder: Decoder architecture config.
        detector: Frozen detector architecture config.
        payload_bits: Number of payload bits per training sample.
        payload_replicas: Number of full-resolution encoder/decoder/detector
            forward passes run per training step, independent of the image
            batch size. Only the FIRST replica's modified weights are ever
            used for the classification loss (see `forward`), so replicating
            the encoder pass once per *image* in the batch is pure waste —
            it multiplies the (already huge, whole-weight-image) CNN
            activation memory by the image batch size for no benefit.
            `payload_replicas` instead controls how many independent
            payloads are embedded and decoded per step, which is what
            actually needs batching for payload-recovery training signal.
            Keep this small (1-2) for large host models on limited-VRAM
            GPUs; the image batch size (set by the data loader) can stay
            larger since classification reuses a single modified weight set.
    """

    host_model_name: HostModelName = "resnet18"
    host_model_num_classes: int = 1000
    host_model_pretrained: bool = False
    train_host_model: bool = False
    encoder: EncoderConfig = None          # type: ignore[assignment]
    decoder: DecoderConfig = None          # type: ignore[assignment]
    detector: DifferentiableDetectorConfig = None  # type: ignore[assignment]
    payload_bits: int = 8192  # 1024 bytes
    payload_replicas: int = 1
    # Skip the corresponding forward pass entirely (not just exclude it from
    # the loss afterward) when the matching loss weight is 0. The detector's
    # full CNN forward pass over the entire weight-image is the single most
    # expensive thing `forward()` runs — computing it when `delta=0` (its
    # loss weight) burns the same memory/compute for zero training signal.
    # Same idea for classification when `alpha=0`: skips the STE weight
    # reconstruction and the host model's functional_forward. Both default
    # True (original always-run behavior). `SteganographyExperiment.run`
    # sets these automatically from `cfg.loss_weights` each run, so turning
    # alpha/delta back on for the full 4-objective run automatically turns
    # the forward passes back on too — no manual toggling needed.
    run_classification: bool = True
    run_detector: bool = True
    # -------------------------------------------------------------------------
    # Reference-based decoding (default: True — the correct final design)
    # -------------------------------------------------------------------------
    # When True, the decoder receives `modified_repr − original_repr` (the
    # encoder's pure delta residual) instead of `modified_repr` itself.
    #
    # WHY THIS IS THE RIGHT CHOICE:
    # The decoder's 17×17-pixel receptive field (4 residual blocks × 2 3×3
    # convs = 8 hops) covers ~133 adjacent bits simultaneously at the 1.47
    # pixels/bit density of a 1507×1507 weight image. With base_channels=32
    # that gives only 0.24 features per interfering bit — the network
    # physically cannot disentangle 133 overlapping signals in 32 channels,
    # which is why training plateaus at ~76% accuracy regardless of how long
    # it runs. The natural weight texture (the IEEE754 bit patterns of the
    # original weights) acts as structured interference on top of the
    # encoder's subtle per-pixel perturbations.
    #
    # Passing modified − original eliminates the interference completely:
    # the residual is bounded in [-max_delta, +max_delta] per byte channel,
    # has zero natural-weight-texture noise, and directly IS the encoder's
    # signal. The decoder then just needs to learn the sign of the pooled
    # residual per bit — a trivially easier task that converges fast and
    # reaches near-zero BER.
    #
    # THREAT MODEL:
    # This is correct for keyed model steganography: the sender (embedder)
    # knows the original model; the receiver (extractor) also knows the
    # original model and uses it as a secret key. The SRNet malware detector
    # only ever sees the modified model weights — never the original — so
    # the evaluation scenario is unaffected. This mirrors how all practical
    # model watermarking schemes work (e.g. Uchida et al. 2017, RivaGAN,
    # DeepIPR).
    #
    # Set to False ONLY if you specifically need blind decoding (extractor
    # has no access to the original model) — but expect ~76% BER ceiling
    # even with deeper/wider architectures unless base_channels is
    # dramatically increased (≥256 to get features/bit above 1.0).
    reference_decoding: bool = True

    def __post_init__(self) -> None:
        # Allow None sentinels; replace with defaults sized to self.payload_bits.
        if self.encoder is None:
            object.__setattr__(self, "encoder", EncoderConfig(payload_dim=self.payload_bits))
        if self.decoder is None:
            # DensePayloadDecoder's parameter count depends only on
            # base_channels/num_residual_blocks, never on payload_bits — the
            # old chunk_size caveat here no longer applies (see
            # models/decoder.py's design note: chunking was removed).
            object.__setattr__(self, "decoder", DecoderConfig())
        if self.detector is None:
            object.__setattr__(self, "detector", DifferentiableDetectorConfig())


class EmbeddingPipeline(nn.Module):
    """Full steganographic embedding pipeline.

    This module owns the encoder, decoder, host model, and frozen detector.
    Its :meth:`forward` method accepts a mini-batch of classification images,
    their labels, and a payload bit tensor, runs the full embedding loop, and
    returns a :class:`~training.losses.LossInputs` ready for the composite
    loss.

    Parameters that are trained:
      - ``encoder`` (always)
      - ``decoder`` (always)
      - ``host_model`` (only if ``config.train_host_model=True``)

    Parameters that are always frozen:
      - ``detector``
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        super().__init__()
        cfg = config or PipelineConfig()
        self.config = cfg

        self.host_model: HostModelAdapter = build_host_model(
            cfg.host_model_name,
            num_classes=cfg.host_model_num_classes,
            pretrained=cfg.host_model_pretrained,
        )
        self.encoder: WeightPayloadEncoder = build_encoder(cfg.encoder)
        self.decoder: DensePayloadDecoder = build_decoder(cfg.decoder)
        self.detector: DifferentiableDetector = DifferentiableDetector(cfg.detector)

        if not cfg.train_host_model:
            for param in self.host_model.parameters():
                param.requires_grad_(False)

        # The detector is always frozen; enforce it here.
        self.detector.freeze()

        # When the host model is frozen (the default), its weights never
        # change across the whole training run, so extracting them and
        # converting to the 4-channel representation is identical work
        # every single step. That conversion round-trips through CPU (numpy
        # bit manipulation), which stalls the GPU waiting on synchronous
        # CUDA->CPU->CUDA transfers every step — this is the dominant cost
        # for large host models, not the GPU compute itself. Do it once here
        # instead of inside forward().
        self._cached_weight_records: list[WeightTensor] | None = None
        self._cached_original_repr: torch.Tensor | None = None
        if not cfg.train_host_model:
            with torch.no_grad():
                weight_records = extract_weights(self.host_model.model)
                flat_weights = flatten_weights(weight_records)
                channels_uint8 = weights_to_channels(flat_weights)
                cached_repr = torch.from_numpy(channels_uint8.astype(np.float32))
            self._cached_weight_records = weight_records
            # register_buffer so `.to(device)` / `.cuda()` on the pipeline
            # moves this cached image along with the rest of the module,
            # without it being treated as a learnable parameter.
            self.register_buffer("_cached_original_repr_buf", cached_repr, persistent=False)

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        payload_bits: torch.Tensor,
    ) -> LossInputs:
        """Run the full pipeline for one training step.

        Args:
            images: Float image batch, shape ``(B, 3, H, W)``.
            labels: Ground-truth class indices, shape ``(B,)``.
            payload_bits: Binary bit tensor, shape ``(B, payload_bits)`` or
                ``(payload_bits,)`` for a shared payload (will be broadcast).

        Returns:
            :class:`~training.losses.LossInputs` with all tensors populated.
        """
        device = images.device
        image_batch_size = images.shape[0]

        # ---- How many full-resolution encoder passes this step runs ----
        # Capped by both the configured `payload_replicas` and the number of
        # images available (never replicate beyond what the batch provides).
        num_replicas = max(1, min(self.config.payload_replicas, image_batch_size))

        # ---- Broadcast/select payload for exactly `num_replicas` samples ----
        if payload_bits.ndim == 1:
            payload_bits = payload_bits.unsqueeze(0).expand(num_replicas, -1)
        else:
            payload_bits = payload_bits[:num_replicas]
        payload_bits = payload_bits.to(device=device, dtype=torch.float32)

        # ---- Extract flat weights from host model ----
        # Skip recomputation entirely when the host model is frozen (the
        # default) — see the caching note in __init__. Only when
        # train_host_model=True do the weight VALUES actually change
        # between steps, so only then do we re-extract every call.
        if self._cached_weight_records is not None:
            weight_records = self._cached_weight_records
            original_repr = self._cached_original_repr_buf.to(device)
        else:
            weight_records = extract_weights(self.host_model.model)
            flat_weights = flatten_weights(weight_records).to(device)  # (N,)
            channels_uint8 = weights_to_channels(flat_weights)  # numpy (4, H, W) uint8
            original_repr = torch.from_numpy(channels_uint8.astype(np.float32)).to(device)
        # Total real (unpadded) element count — needed downstream to trim
        # padding when reconstructing weights from the channel image.
        num_weight_values = sum(record.values.numel() for record in weight_records)

        # Only replicate the (expensive) weight image `num_replicas` times,
        # NOT once per image in the classification batch.
        original_repr_batch = original_repr.unsqueeze(0).expand(num_replicas, -1, -1, -1)

        # ---- Encoder: produce modified representation ----
        # This CNN forward pass runs over the FULL weight image, so its cost
        # scales with num_replicas, not with the (potentially much larger)
        # image_batch_size used for classification below.
        modified_repr_batch = self.encoder(original_repr_batch, payload_bits)
        # Shape: (num_replicas, 4, H, W)

        # ---- Decoder: recover payload from each replica ----
        # Reference-based: give the decoder (modified - original) instead of
        # the raw modified representation. The residual is the encoder's pure
        # delta signal: bounded in [-max_delta, +max_delta] per byte channel,
        # with zero natural-weight-texture interference. The decoder just
        # needs to learn sign-of-pooled-residual per bit, not untangle the
        # encoder's perturbation FROM the host model's natural weight pattern.
        # See PipelineConfig.reference_decoding for the full rationale.


        # If encoder returns a tuple (modified_repr, gate) or similar:
        if isinstance(modified_repr_batch, tuple):
            modified_repr_batch, capacity_gate = modified_repr_batch
        else:
            capacity_gate = None
        
        decoder_input = modified_repr_batch - original_repr_batch

        if self.config.reference_decoding:
            decoder_input = modified_repr_batch - original_repr_batch
        else:
            decoder_input = modified_repr_batch
        payload_logits = self.decoder(decoder_input, self.config.payload_bits)
        payload_logits_flat = payload_logits.reshape(num_replicas, -1)[
            :, : self.config.payload_bits
        ]

        # ---- Detector: score the modified representation ----
        # The detector is frozen but its forward pass is differentiable w.r.t. input.
        # Skipped entirely (not just excluded from the loss) when delta=0 —
        # see PipelineConfig.run_detector. This is the single most expensive
        # forward pass in the pipeline (a full CNN over the whole weight
        # image), so running it for zero training signal is pure waste.
        if self.config.run_detector:
            detector_logits = self.detector(modified_repr_batch).squeeze(-1)  # (num_replicas,)
            # Target: predict as benign (label = 0) to fool the detector.
            detector_targets = torch.zeros_like(detector_logits)
        else:
            detector_logits = None
            detector_targets = None

        # ---- Classification with modified weights (STE) ----
        # Skipped entirely (not just excluded from the loss) when alpha=0 —
        # see PipelineConfig.run_classification. Avoids the STE weight
        # reconstruction (which round-trips through CPU/numpy) and the host
        # model's functional_forward when classification isn't contributing
        # to the loss anyway.
        if self.config.run_classification:
            # Use the first sample's modified representation for weight reconstruction.
            modified_repr_single = modified_repr_batch[0]  # (4, H, W)
            modified_flat = _ChannelsToWeightsSTE.apply(
                modified_repr_single, len(weight_records), num_weight_values
            )
            # Rebuild parameter dict for functional_call.
            modified_params = _rebuild_params(
                modified_flat, weight_records, self.host_model.model
            )
            # Run classification with modified weights; gradient flows through functional_call.
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
        """Encode a payload into the host model weights (inference only).

        Args:
            payload_bits: Binary bit tensor with shape ``(payload_bits,)``
                or ``(1, payload_bits)``.

        Returns:
            Tuple of ``(modified_repr, original_repr)`` with shape
            ``(1, 4, H, W)`` each.
        """
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
        return modified_repr, original_repr_batch

    def decode(
        self,
        modified_repr: torch.Tensor,
        original_repr: torch.Tensor | None = None,
        num_bits: int | None = None,
    ) -> torch.Tensor:
        """Decode payload bits from a modified representation.

        Args:
            modified_repr: Tensor with shape ``(B, 4, H, W)``.
            original_repr: Original (unmodified) weight representation with the
                same shape as ``modified_repr``. Required when
                ``config.reference_decoding=True`` (the default). If not
                provided and reference_decoding is True, the pipeline
                computes it from the cached host model weights automatically.
            num_bits: Number of bits to decode; defaults to ``payload_bits``
                in :attr:`config`.

        Returns:
            Uint8 decoded bit tensor with shape ``(B, num_bits)``.
        """
        self.eval()
        with torch.no_grad():
            bits = num_bits or self.config.payload_bits
            if self.config.reference_decoding:
                if original_repr is None:
                    # Compute from cache (host model weights are frozen and
                    # never change after __init__, so the cached repr is valid).
                    device = modified_repr.device
                    if self._cached_original_repr_buf is not None:
                        orig = self._cached_original_repr_buf.to(device).float()
                    else:
                        weight_records = extract_weights(self.host_model.model)
                        flat = flatten_weights(weight_records).to(device)
                        orig = torch.from_numpy(
                            weights_to_channels(flat).astype("float32")
                        ).to(device)
                    # Expand to match batch dim of modified_repr
                    orig = orig.unsqueeze(0).expand_as(modified_repr)
                else:
                    orig = original_repr.to(modified_repr.device).float()
                decoder_input = modified_repr.float() - orig
            else:
                decoder_input = modified_repr
            return self.decoder.decode(decoder_input, bits)


def build_pipeline(config: PipelineConfig | None = None) -> EmbeddingPipeline:
    """Build an :class:`EmbeddingPipeline` from an optional configuration."""
    return EmbeddingPipeline(config)


# ---------------------------------------------------------------------------
# Straight-Through Estimator for byte-channel → float weight reconstruction
# ---------------------------------------------------------------------------


class _ChannelsToWeightsSTE(torch.autograd.Function):
    """Convert float representation channels back to float32 weights.

    Forward:
        Rounds values to the nearest byte, reconstructs the IEEE754 bit
        pattern exactly, and returns the bit-exact float32 weights.

    Backward (STE):
        Treats the reconstruction as an identity function for the purpose of
        gradient propagation.  Each modified float32 weight contributes its
        gradient equally to all four channel positions that encode it, divided
        by the number of channels (4) to preserve gradient scale.  Zero
        gradient is returned for padded pixel positions beyond ``num_values``.
    """

    @staticmethod
    def forward(  # type: ignore[override]
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
    def backward(ctx: Any, grad_weights: torch.Tensor):  # type: ignore[override]
        c, h, w = ctx.channels_shape
        n = ctx.num_values
        hw = h * w

        # Distribute each weight's gradient to all 4 channel positions that
        # encode it (STE: treat quantization + bit-cast as identity).
        grad_per_pixel = torch.zeros(hw, device=grad_weights.device, dtype=torch.float32)
        num_real = min(n, hw)
        grad_per_pixel[:num_real] = grad_weights.reshape(-1)[:num_real]

        grad_channels = torch.zeros(c, h, w, device=grad_weights.device, dtype=torch.float32)
        for ci in range(c):
            grad_channels[ci] = grad_per_pixel.reshape(h, w)
        # Divide by number of channels for equal gradient sharing.
        return grad_channels / float(c), None, None


# ---------------------------------------------------------------------------
# Parameter reconstruction helpers
# ---------------------------------------------------------------------------


def _rebuild_params(
    modified_flat: torch.Tensor,
    weight_records: list[WeightTensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Reconstruct a parameter dict for ``functional_call`` from a flat tensor.

    **Learnable parameters** (in ``model.named_parameters()``) are replaced
    with the corresponding STE-modified slice so that gradients can flow back
    to the encoder.

    **Non-learnable buffers** — including BatchNorm ``running_mean`` /
    ``running_var`` / ``num_batches_tracked`` — are always taken unchanged
    from the model's current ``state_dict()`` as detached, non-differentiable
    tensors.  ``functional_call`` / ``F.batch_norm`` raises a ``RuntimeError``
    if running statistics carry ``requires_grad=True``, so this is required for
    correctness on any backbone that contains BatchNorm layers.

    The ``offset`` pointer advances through ``modified_flat`` for **every**
    floating-point entry in ``weight_records`` (buffers included), because the
    encoder processed all of them.  Buffers simply have their STE slice
    discarded and replaced by the unmodified state value.

    Args:
        modified_flat: Modified flat weight vector, shape ``(N,)``.
        weight_records: Ordered metadata from :func:`~utils.weights.extract_weights`.
        model: Reference model providing buffer values and learnable-param names.

    Returns:
        Dict mapping state-dict key → replacement tensor suitable for
        ``functional_call``.
    """
    params: dict[str, torch.Tensor] = {}
    state = model.state_dict()
    device = modified_flat.device

    # Learnable parameters — only these receive STE-modified, differentiable values.
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
                # Replace with the STE-modified, gradient-carrying slice.
                values = modified_flat[offset: offset + size].reshape(rec.shape)
                params[name] = values.to(device=device, dtype=tensor.dtype)
            else:
                # Buffer (e.g. running_mean/running_var): use current unmodified
                # state and ensure it is detached so functional_call does not
                # see requires_grad=True on BatchNorm running statistics.
                params[name] = tensor.to(device=device).detach()
            offset += size
        else:
            # Non-floating buffer (e.g. num_batches_tracked).
            params[name] = tensor.to(device=device)

    return params
