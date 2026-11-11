#!/usr/bin/env python3
"""Evaluation entry point for the steganographic embedding pipeline.

Loads a trained checkpoint, runs the full evaluation suite, and writes all
metrics and figures to an output directory.  Evaluation covers:

  1. Payload recovery — BER, reconstruction accuracy.
  2. Optional clean-vs-modified detection metrics from a trained SRNet/FSL detector.
  3. Embedding capacity — bits per parameter, embedding rate.
  4. Accuracy drop — top-1 / top-5 classification accuracy change (when an
     image dataset is available).

Usage
-----

    python scripts/evaluate.py \\
        --checkpoint outputs/run_01/checkpoints/best.pt \\
        --host-model resnet18 \\
        --payload-size 128KB \\
        --num-samples 64 \\
        --output-dir outputs/run_01/eval

The script also supports evaluating the pipeline in its initial (untrained)
state to provide a baseline comparison:

    python scripts/evaluate.py \\
        --host-model resnet18 \\
        --payload-size 128KB \\
        --baseline-only

Include real accuracy-drop metrics on CIFAR-10's held-out test set:

    python scripts/evaluate.py \\
        --checkpoint outputs/run_01/checkpoints/best.pt \\
        --host-model resnet18 \\
        --num-classes 10 \\
        --dataset cifar10 \\
        --data-root ./data \\
        --output-dir outputs/run_01/eval

"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from evaluation.accuracy import evaluate_accuracy_drop
from evaluation.capacity import compute_capacity
from evaluation.detector import ModelXRayDetector
from evaluation.metrics import evaluate_detector_from_weights
from evaluation.plotting import plot_metrics_summary
from models.decoder import DecoderConfig
from models.encoder import EncoderConfig
from models.pipeline import EmbeddingPipeline, PipelineConfig
from evaluation.differentiable_detector import DifferentiableDetectorConfig
from training.dataset import SyntheticWeightDataset
from utils.config import load_config
from utils.logging import configure_logging
from utils.payload import SUPPORTED_PAYLOAD_SIZES, generate_payload, payload_to_tensor
from utils.seed import set_seed
from utils.weights import extract_parameters, flatten_weights, load_modified_weights, restore_weights
from utils.representation import weights_to_float_image, weights_to_channels


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained steganographic embedding pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a trained pipeline checkpoint (.pt).  "
             "If omitted, evaluates the untrained baseline.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Training TOML config used to construct the matching encoder/decoder architecture.",
    )
    parser.add_argument(
        "--host-model",
        choices=["resnet18", "resnet50", "mobilenet_v2", "vgg16"],
        default="resnet18",
    )
    parser.add_argument(
        "--payload-size",
        choices=["128KB", "256KB", "512KB", "1MB"],
        default="128KB",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of host model output classes. Set to 10 when using "
             "--dataset cifar10.",
    )
    parser.add_argument(
        "--host-checkpoint", type=Path, default=None,
        help="Trained host-model checkpoint (required for task-specific class heads).",
    )
    parser.add_argument(
        "--detector-srnet", type=Path, default=None,
        help="Optional trained Model X-Ray SRNet checkpoint.",
    )
    parser.add_argument(
        "--detector-classifier", type=Path, default=None,
        help="Optional trained Model X-Ray FSL classifier/centroid checkpoint.",
    )
    parser.add_argument(
        "--detector-image-size", type=int, default=256,
        help="GF image side used by the trained detector (train_detector.py defaults to 256).",
    )
    parser.add_argument(
        "--detector-training-distribution",
        choices=["unspecified", "benign_random_lsb_surrogate", "learned_encoder_carriers", "other"],
        default="unspecified",
        help="Provenance label for the supplied detector checkpoint; this is metadata, not verified automatically.",
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "cifar10"],
        default="synthetic",
        help="Image source for the accuracy-drop metric. 'synthetic' skips "
             "accuracy-drop entirely (no real images available). 'cifar10' "
             "evaluates top-1/top-5 accuracy drop on the real CIFAR-10 test "
             "split. (default: synthetic)",
    )
    parser.add_argument(
        "--image-size", type=int, default=None,
        help="Host input image side; defaults to the training TOML value or 224.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data"),
        help="Directory to download/cache CIFAR-10 into. Ignored for "
             "'synthetic'. (default: ./data)",
    )
    parser.add_argument(
        "--accuracy-batch-size",
        type=int,
        default=64,
        help="Batch size for the CIFAR-10 accuracy-drop pass.",
    )
    parser.add_argument(
        "--accuracy-max-batches",
        type=int,
        default=None,
        help="Optional cap on the number of CIFAR-10 test batches evaluated "
             "for accuracy drop (full 10k-image test set if omitted — can "
             "be slow on CPU).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of payload samples to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval"),
        help="Directory for evaluation results and figures.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Evaluate the untrained pipeline as a baseline.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(level="DEBUG" if args.verbose else "INFO")
    logger = logging.getLogger(__name__)
    set_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    payload_size_str = args.payload_size
    payload_bits = SUPPORTED_PAYLOAD_SIZES[payload_size_str] * 8

    # Load the trainer metadata before construction so the architecture
    # (notably bits_per_pixel and adaptive-capacity layers) matches training.
    ckpt = None
    if args.checkpoint is not None and not args.baseline_only:
        logger.info("Loading checkpoint metadata from %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved_pipeline_cfg = (
        ckpt.get("config", {}).get("pipeline", {}) if isinstance(ckpt, dict) else {}
    )
    experiment_config = None
    if args.config is not None:
        experiment_config = load_config(args.config)
        pipeline_cfg = _pipeline_config_from_toml(
            experiment_config,
            host_model=args.host_model,
            num_classes=args.num_classes,
            host_checkpoint=args.host_checkpoint,
            payload_bits=payload_bits,
        )
        logger.info("Constructed evaluation architecture from %s", args.config)
    else:
        if not saved_pipeline_cfg.get("encoder"):
            logger.info(
                "Checkpoint has no saved PipelineConfig; inferring encoder/decoder "
                "structure from its tensor shapes."
            )
        pipeline_cfg = _pipeline_config_for_evaluation(
            saved_pipeline_cfg,
            state_dict=(ckpt.get("model", ckpt) if isinstance(ckpt, dict) else None),
            host_model=args.host_model,
            num_classes=args.num_classes,
            host_checkpoint=args.host_checkpoint,
            payload_bits=payload_bits,
        )
    if saved_pipeline_cfg and saved_pipeline_cfg.get("payload_bits") not in (None, payload_bits):
        raise ValueError(
            f"Requested {payload_size_str} ({payload_bits:,} bits), but the checkpoint "
            f"was trained for {saved_pipeline_cfg['payload_bits']:,} bits. Use the "
            "matching --payload-size."
        )
    pipeline = EmbeddingPipeline(pipeline_cfg)
    device = _resolve_device(args.device)
    pipeline = pipeline.to(device)
    pipeline.eval()

    if ckpt is not None:
        state = ckpt.get("model", ckpt)
        pipeline.load_state_dict(state, strict=True)
        logger.info("Checkpoint loaded.")
    else:
        logger.info("No checkpoint provided — evaluating untrained baseline.")

    # ---- Generate payloads and encode them ----
    logger.info("Encoding %d payloads…", args.num_samples)

    original_weight_list: list[torch.Tensor] = []
    modified_weight_list: list[torch.Tensor] = []
    payload_bits_list: list[torch.Tensor] = []
    decoded_bits_list: list[torch.Tensor] = []

    weight_records = extract_parameters(pipeline.host_model.model)
    flat_weights_cpu = flatten_weights(weight_records)
    _, norm_stats = weights_to_float_image(flat_weights_cpu)

    with torch.no_grad():
        for i in range(args.num_samples):
            payload = generate_payload(payload_size_str, seed=args.seed + i)
            bits = payload_to_tensor(payload).to(device=device, dtype=torch.float32)

            modified_repr, original_repr = pipeline.encode(bits)
            original_weight_list.append(flat_weights_cpu.clone())

            # Reconstruct flat modified weights for detector evaluation.
            # The representation is now a single-channel float image in
            # normalised [-1, 1] space; invert with the host model's stats.
            from utils.representation import float_image_to_weights
            modified_flat = float_image_to_weights(
                modified_repr[0].detach().cpu(),
                mean=norm_stats.mean,
                scale=norm_stats.scale,
                num_values=flat_weights_cpu.numel(),
            )
            # Simulate a saved FP32 state_dict: load the reconstructed
            # parameters into a fresh host model, then re-extract them before
            # decoding. This prevents the in-memory encoder tensor from
            # bypassing serialization/reconstruction effects.
            roundtrip_model = copy.deepcopy(pipeline.host_model.model)
            load_modified_weights(
                roundtrip_model,
                restore_weights(modified_flat, weight_records),
                strict=True,
            )
            serialized = io.BytesIO()
            torch.save(roundtrip_model.state_dict(), serialized)
            serialized.seek(0)
            saved_state = torch.load(serialized, map_location="cpu", weights_only=True)
            roundtrip_model = copy.deepcopy(pipeline.host_model.model).cpu()
            roundtrip_model.load_state_dict(saved_state, strict=True)
            roundtrip_flat = flatten_weights(extract_parameters(roundtrip_model))
            # Use the *clean reference* normalization for receiver-side
            # subtraction. Recomputing mean/scale from the modified model
            # would give the receiver an unavailable statistic and change the
            # residual seen by the decoder.
            reference_side = original_repr.shape[-1]
            normalized_flat = (
                roundtrip_flat - norm_stats.mean
            ) / norm_stats.scale
            if normalized_flat.numel() < reference_side * reference_side:
                normalized_flat = torch.nn.functional.pad(
                    normalized_flat,
                    (0, reference_side * reference_side - normalized_flat.numel()),
                )
            roundtrip_repr = normalized_flat.reshape(1, 1, reference_side, reference_side).to(device)
            roundtrip_decoded = pipeline.decode(
                roundtrip_repr, original_repr, payload_bits
            )
            decoded = roundtrip_decoded
            modified_weight_list.append(roundtrip_flat.cpu())
            payload_bits_list.append(bits.cpu().to(dtype=torch.uint8))
            decoded_bits_list.append(decoded[0])

    # ---- Payload recovery metrics ----
    bers: list[float] = []
    for orig, rec in zip(payload_bits_list, decoded_bits_list):
        orig_trim = orig[:payload_bits]
        rec_trim = rec[:payload_bits]
        n = min(orig_trim.numel(), rec_trim.numel())
        errors = int((orig_trim[:n] != rec_trim[:n]).sum().item())
        bers.append(errors / n if n > 0 else 0.0)

    mean_ber = float(np.mean(bers))
    mean_accuracy = 1.0 - mean_ber

    logger.info("Payload recovery — BER: %.4f, Accuracy: %.4f", mean_ber, mean_accuracy)

    # ---- Embedding capacity ----
    capacity = compute_capacity(
        pipeline.host_model.model,
        payload_bits,
        bit_error_rate=mean_ber,
    )
    logger.info("%s", capacity)

    # ---- Optional trained detector evaluation ----
    trained_detector = args.detector_srnet is not None or args.detector_classifier is not None
    if trained_detector and (args.detector_srnet is None or args.detector_classifier is None):
        raise ValueError("Pass both --detector-srnet and --detector-classifier for trained FSL evaluation.")
    det_metrics = None
    if trained_detector:
        logger.info("Running trained Model X-Ray SRNet/FSL detector…")
        detector = ModelXRayDetector(
            srnet_weights_path=args.detector_srnet,
            centroid_path=args.detector_classifier,
            device=args.device,
            image_size=args.detector_image_size,
        )
        # The wrapper can fall back after missing files, unavailable modules, or
        # load errors. Never report that fallback as trained-detector evidence.
        if not detector.use_srnet:
            raise RuntimeError(
                "Trained detector evaluation was requested, but SRNet/FSL did not "
                "load. Check the checkpoint paths and Model X-Ray dependencies."
            )
        try:
            det_metrics = evaluate_detector_from_weights(
                detector,
                clean_weights=original_weight_list,
                modified_weights=modified_weight_list,
                verbose=args.verbose,
            )
            logger.info("%s", det_metrics)
            det_figures = plot_metrics_summary(
                det_metrics,
                output_dir=output_dir / "figures",
                prefix="detector",
            )
            logger.info("Detection figures saved: %s", [str(p) for p in det_figures])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Trained detector evaluation failed: %s", exc)
            det_metrics = None
    else:
        logger.info(
            "Skipping detection metrics: no trained SRNet/FSL checkpoints supplied. "
            "The statistical fallback is a heuristic and is not valid detection evidence."
        )

    # ---- Accuracy drop (only when a real image dataset is available) ----
    accuracy_result = None
    if args.dataset == "cifar10":
        if args.num_classes != 10:
            logger.warning(
                "--dataset cifar10 expects --num-classes 10 (got %d); "
                "the host model's class head won't match CIFAR-10 labels.",
                args.num_classes,
            )
        logger.info("Evaluating accuracy drop on the real CIFAR-10 test set…")
        image_size = args.image_size or (
            experiment_config.get("data", {}).get("image_size", 224)
            if experiment_config is not None
            else 224
        )
        transform_ops = []
        if image_size != 32:
            transform_ops.append(transforms.Resize((image_size, image_size)))
        transform_ops.append(transforms.ToTensor())
        if args.host_model != "tiny":
            transform_ops.append(transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ))
        transform = transforms.Compose(transform_ops)
        test_ds = CIFAR10(
            root=str(args.data_root), train=False, download=True, transform=transform
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.accuracy_batch_size, shuffle=False
        )

        # Build a modified copy of the host model from the first payload
        # sample's modified weights (mirrors how pipeline.forward() reconstructs
        # weights from modified_repr_batch[0] during training).
        modified_records = restore_weights(modified_weight_list[0], weight_records)
        modified_host_model = copy.deepcopy(pipeline.host_model.model).to(device)
        load_modified_weights(modified_host_model, modified_records, strict=True)

        accuracy_result = evaluate_accuracy_drop(
            original_model=pipeline.host_model.model,
            modified_model=modified_host_model,
            data_loader=test_loader,
            device=device,
            verbose=args.verbose,
            max_batches=args.accuracy_max_batches,
        )
        logger.info("%s", accuracy_result)
    else:
        logger.info(
            "Skipping accuracy drop — pass --dataset cifar10 for real "
            "top-1/top-5 accuracy metrics."
        )

    # ---- Persist results ----
    results: dict = {
        "payload_recovery": {
            "mean_ber": mean_ber,
            "mean_accuracy": mean_accuracy,
            "num_samples": args.num_samples,
            "payload_bits": payload_bits,
        },
        "capacity": {k: v for k, v in capacity.to_dict().items() if v is not None},
        "evaluation_protocol": {
            "reference_decoding": pipeline.config.reference_decoding,
            "post_fp32_roundtrip": True,
            "detector": "trained_srnet_fsl_single_host_pilot" if trained_detector and det_metrics is not None else "not_evaluated",
            "detector_training_distribution": args.detector_training_distribution if trained_detector else None,
            "detector_test_design": "one_clean_host_repeated_across_payload_variants" if trained_detector else None,
            "fallback_detector_used_for_metrics": False,
            "detector_results_are_malware_validation": False,
        },
    }
    if det_metrics is not None:
        results["detection"] = det_metrics.to_dict()
    if accuracy_result is not None:
        results["accuracy_drop"] = accuracy_result.to_dict()

    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Evaluation results saved to %s", metrics_path)

    return 0


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _pipeline_config_for_evaluation(
    saved_config: dict,
    *,
    state_dict: dict | None,
    host_model: str,
    num_classes: int,
    host_checkpoint: Path | None,
    payload_bits: int,
) -> PipelineConfig:
    """Rebuild the pipeline architecture recorded in a trainer checkpoint."""
    def make_config(config_type, values: dict, **overrides):
        accepted = {field.name for field in fields(config_type)}
        arguments = {key: value for key, value in values.items() if key in accepted}
        arguments.update(overrides)
        return config_type(**arguments)

    saved = saved_config if isinstance(saved_config, dict) else {}
    inferred_encoder, inferred_decoder = _infer_encoder_decoder_config(state_dict or {})
    encoder_values = inferred_encoder
    encoder_values.update(saved.get("encoder") or {})
    decoder_values = inferred_decoder
    decoder_values.update(saved.get("decoder") or {})
    detector_values = dict(saved.get("detector") or {})
    encoder = make_config(EncoderConfig, encoder_values, payload_dim=payload_bits)
    decoder = make_config(DecoderConfig, decoder_values)
    if encoder.bits_per_pixel != decoder.bits_per_pixel:
        raise ValueError(
            "Checkpoint/config disagrees on bits_per_pixel: "
            f"encoder={encoder.bits_per_pixel}, decoder={decoder.bits_per_pixel}."
        )
    detector = make_config(DifferentiableDetectorConfig, detector_values)

    pipeline_fields = {field.name for field in fields(PipelineConfig)}
    arguments = {
        key: value
        for key, value in saved.items()
        if key in pipeline_fields and key not in {"encoder", "decoder", "detector"}
    }
    arguments.update({
        "host_model_name": host_model,
        "host_model_num_classes": num_classes,
        "host_model_pretrained": False,
        "host_model_checkpoint": str(host_checkpoint) if host_checkpoint else None,
        "payload_bits": payload_bits,
        "encoder": encoder,
        "decoder": decoder,
        "detector": detector,
    })
    return PipelineConfig(**arguments)


def _infer_encoder_decoder_config(state: dict) -> tuple[dict, dict]:
    """Infer structural settings from a model state dict when metadata is absent.

    Trainer checkpoints currently store optimizer/trainer metadata but not the
    experiment's PipelineConfig. Parameter shapes provide the required model
    widths, block counts, payload bits-per-location, and adaptive-gate setting.
    """
    if not state:
        return {}, {}

    def shape(key: str) -> tuple[int, ...] | None:
        value = state.get(key)
        return tuple(value.shape) if isinstance(value, torch.Tensor) else None

    def block_count(prefix: str) -> int:
        indexes = [
            int(match.group(1))
            for key in state
            if (match := re.match(rf"{re.escape(prefix)}\.blocks\.(\d+)\.conv1\.conv\.weight$", key))
        ]
        return max(indexes) + 1 if indexes else 4

    def attention_reduction(prefix: str, key_suffix: str) -> int:
        weight_shape = shape(f"{prefix}.blocks.0.{key_suffix}.weight")
        stem_shape = shape(f"{prefix}.stem.conv.weight")
        if weight_shape and stem_shape and weight_shape[0] > 0:
            return max(1, stem_shape[0] // weight_shape[0])
        return 8

    encoder_stem = shape("encoder.stem.conv.weight")
    message_first = shape("encoder.message_prep.net.0.conv.weight")
    output_projection = shape("encoder.output_projection.weight")
    message_layers = [
        int(match.group(1))
        for key in state
        if (match := re.match(r"encoder\.message_prep\.net\.(\d+)\.conv\.weight$", key))
    ]
    encoder = {
        "input_channels": encoder_stem[1] if encoder_stem else 1,
        "output_channels": output_projection[0] if output_projection else 1,
        "base_channels": encoder_stem[0] if encoder_stem else 64,
        "num_residual_blocks": block_count("encoder"),
        "message_channels": message_first[0] if message_first else 32,
        "message_prep_layers": max(message_layers) + 1 if message_layers else 2,
        "attention_reduction": attention_reduction("encoder", "attention.channel_reduce"),
        "bits_per_pixel": message_first[1] if message_first else 1,
        "adaptive_capacity": any(key.startswith("encoder.capacity_gate.") for key in state),
        "max_delta": 0.05,
    }

    decoder_stem = shape("decoder.stem.conv.weight")
    projection = shape("decoder.logit_projection.weight")
    decoder = {
        "input_channels": decoder_stem[1] if decoder_stem else 1,
        "base_channels": decoder_stem[0] if decoder_stem else 64,
        "num_residual_blocks": block_count("decoder"),
        "attention_reduction": attention_reduction("decoder", "attention.reduce"),
        "bits_per_pixel": projection[0] if projection else 1,
    }
    return encoder, decoder


def _pipeline_config_from_toml(
    config: dict,
    *,
    host_model: str,
    num_classes: int,
    host_checkpoint: Path | None,
    payload_bits: int,
) -> PipelineConfig:
    """Construct an evaluation pipeline from the same TOML as training."""
    host = config.get("host_model", {})
    encoder_values = config.get("encoder", {})
    decoder_values = config.get("decoder", {})
    detector_values = config.get("detector", {})
    training = config.get("training", {})
    pipeline_values = config.get("pipeline", {})

    bpp = encoder_values.get(
        "bits_per_pixel", decoder_values.get("bits_per_pixel", 1)
    )
    encoder = EncoderConfig(
        payload_dim=payload_bits,
        base_channels=encoder_values.get("base_channels", 64),
        num_residual_blocks=encoder_values.get("num_residual_blocks", 4),
        message_channels=encoder_values.get("message_channels", 32),
        message_prep_layers=encoder_values.get("message_prep_layers", 2),
        attention_reduction=encoder_values.get("attention_reduction", 8),
        max_delta=encoder_values.get("max_delta", 0.05),
        gradient_checkpointing=encoder_values.get("gradient_checkpointing", False),
        bits_per_pixel=bpp,
        adaptive_capacity=encoder_values.get("adaptive_capacity", False),
    )
    decoder = DecoderConfig(
        base_channels=decoder_values.get("base_channels", 64),
        num_residual_blocks=decoder_values.get("num_residual_blocks", 4),
        attention_reduction=decoder_values.get("attention_reduction", 8),
        gradient_checkpointing=decoder_values.get("gradient_checkpointing", False),
        bits_per_pixel=bpp,
    )
    detector = DifferentiableDetectorConfig(
        **{
            field.name: detector_values[field.name]
            for field in fields(DifferentiableDetectorConfig)
            if field.name in detector_values
        }
    )
    return PipelineConfig(
        host_model_name=host_model,
        host_model_num_classes=num_classes,
        host_model_pretrained=False,
        host_model_checkpoint=str(host_checkpoint) if host_checkpoint else None,
        train_host_model=host.get("train_host_model", False),
        encoder=encoder,
        decoder=decoder,
        detector=detector,
        payload_bits=payload_bits,
        payload_replicas=training.get("payload_replicas", 1),
        run_classification=True,
        run_detector=False,
        reference_decoding=pipeline_values.get("reference_decoding", True),
    )


if __name__ == "__main__":
    sys.exit(main())
