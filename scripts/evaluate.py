#!/usr/bin/env python3
"""Evaluation entry point for the steganographic embedding pipeline.

Loads a trained checkpoint, runs the full evaluation suite, and writes all
metrics and figures to an output directory.  Evaluation covers:

  1. Payload recovery — BER, reconstruction accuracy.
  2. Model X-Ray detection rate — statistical fallback detector.
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
import json
import logging
import sys
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
from models.decoder import bit_error_rate as decoder_ber
from models.pipeline import EmbeddingPipeline, PipelineConfig
from training.dataset import SyntheticWeightDataset
from utils.config import load_config
from utils.logging import configure_logging
from utils.payload import SUPPORTED_PAYLOAD_SIZES, generate_payload, payload_to_tensor
from utils.seed import set_seed
from utils.weights import extract_weights, flatten_weights, load_modified_weights, restore_weights
from utils.representation import weights_to_channels


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
        "--dataset",
        choices=["synthetic", "cifar10"],
        default="synthetic",
        help="Image source for the accuracy-drop metric. 'synthetic' skips "
             "accuracy-drop entirely (no real images available). 'cifar10' "
             "evaluates top-1/top-5 accuracy drop on the real CIFAR-10 test "
             "split. (default: synthetic)",
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

    # ---- Build or load pipeline ----
    pipeline_cfg = PipelineConfig(
        host_model_name=args.host_model,
        host_model_num_classes=args.num_classes,
        host_model_pretrained=False,
        payload_bits=payload_bits,
    )
    pipeline = EmbeddingPipeline(pipeline_cfg)

    device = _resolve_device(args.device)
    pipeline = pipeline.to(device)
    pipeline.eval()

    if args.checkpoint is not None and not args.baseline_only:
        logger.info("Loading checkpoint from %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        state = ckpt.get("model", ckpt)
        pipeline.load_state_dict(state, strict=False)
        logger.info("Checkpoint loaded.")
    else:
        logger.info("No checkpoint provided — evaluating untrained baseline.")

    # ---- Generate payloads and encode them ----
    logger.info("Encoding %d payloads…", args.num_samples)

    original_weight_list: list[torch.Tensor] = []
    modified_weight_list: list[torch.Tensor] = []
    payload_bits_list: list[torch.Tensor] = []
    decoded_bits_list: list[torch.Tensor] = []

    weight_records = extract_weights(pipeline.host_model.model)
    flat_weights_cpu = flatten_weights(weight_records)

    with torch.no_grad():
        for i in range(args.num_samples):
            payload = generate_payload(payload_size_str, seed=args.seed + i)
            bits = payload_to_tensor(payload).to(device=device, dtype=torch.float32)

            modified_repr, original_repr = pipeline.encode(bits)
            decoded = pipeline.decode(modified_repr, payload_bits)

            original_weight_list.append(flat_weights_cpu.clone())

            # Reconstruct flat modified weights for detector evaluation.
            # The representation is now a single-channel float image in
            # normalised [-1, 1] space; invert with the host model's stats.
            from utils.representation import (
                weights_to_float_image,
                float_image_to_weights,
            )
            _, stats = weights_to_float_image(flat_weights_cpu)
            modified_flat = float_image_to_weights(
                modified_repr[0].detach().cpu(),
                mean=stats.mean,
                scale=stats.scale,
                num_values=flat_weights_cpu.numel(),
            )
            modified_weight_list.append(modified_flat.cpu())
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

    # ---- Model X-Ray detection rate ----
    logger.info("Running Model X-Ray statistical detector…")
    detector = ModelXRayDetector(device=args.device)
    try:
        det_metrics = evaluate_detector_from_weights(
            detector,
            clean_weights=[w for w in original_weight_list],
            modified_weights=[w for w in modified_weight_list],
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
        logger.warning("Detector evaluation failed: %s", exc)
        det_metrics = None

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
        transform = transforms.Compose([transforms.ToTensor()])
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


if __name__ == "__main__":
    sys.exit(main())
