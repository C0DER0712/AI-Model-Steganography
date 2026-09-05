#!/usr/bin/env python3
"""End-to-end demonstration of the adaptive model steganography pipeline.

This script ties together all project components into a single reproducible run:

1. **Train the steganographic embedding pipeline** (encoder + decoder) for N
   epochs on synthetic images or CIFAR-10 with a ResNet18 host model.
2. **Generate detection datasets**: save benign (original) and
   steganographically-modified model checkpoints to disk.
3. **Train the SRNet FSL detector** on those checkpoints using triplet loss.
4. **Evaluate detection**: run the trained detector on held-out models,
   computing accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix.
5. **Report and save metrics** to ``outputs/metrics/demo_results.json``.
6. **Generate plots**: detection score distributions, ROC curve, confusion
   matrix, and training loss curves to ``outputs/figures/``.

Usage
-----
Quick smoke-test (5 embedding epochs, 20 FSL epochs, 6 total models):

    python scripts/run_demo.py --epochs 5 --fsl-epochs 20 --num-samples 6

Full run with CIFAR-10:

    python scripts/run_demo.py \\
        --dataset cifar10 \\
        --data-root ./data \\
        --epochs 50 \\
        --fsl-epochs 50 \\
        --num-samples 20 \\
        --device auto \\
        --output-dir outputs/demo

Configuration precedence (highest to lowest)
---------------------------------------------
1. CLI flags
2. ``configs/experiment.toml`` values
3. Hard-coded defaults in this script
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

from evaluation.detector import ModelXRayDetector
from evaluation.fsl_detector import FSLDetector, FSLConfig
from evaluation.metrics import DetectorMetrics, evaluate_detector
from evaluation.plotting import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_score_distribution,
)
from models.pipeline import EmbeddingPipeline, PipelineConfig
from models.decoder import DecoderConfig
from models.encoder import EncoderConfig
from training.dataset import SyntheticImageDataset, SteganographyDataset, build_data_loaders
from training.experiment import ExperimentConfig, SteganographyExperiment
from training.losses import LossWeights
from utils.config import load_config
from utils.device import get_device
from utils.gf_image import state_dict_to_gf_image
from utils.logging import configure_logging
from utils.payload import SUPPORTED_PAYLOAD_SIZES
from utils.seed import set_seed
from utils.weights import extract_weights, flatten_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end steganography + FSL detection demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment.toml"),
        help="TOML configuration file.",
    )
    parser.add_argument(
        "--dataset", choices=["synthetic", "cifar10"], default=None,
        help="Image dataset for embedding training.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("./data"),
        help="CIFAR-10 download directory.",
    )
    parser.add_argument(
        "--host-model",
        choices=["resnet18", "resnet50", "mobilenet_v2", "vgg16", "tiny", "tiny_bn"],
        default=None,
        help="Host model backbone (default: resnet18; use tiny for a low-memory smoke run).",
    )
    parser.add_argument(
        "--num-classes", type=int, default=None,
        help="Host model output classes (default: 10 for CIFAR-10, 1000 for synthetic).",
    )
    parser.add_argument(
        "--synthetic-samples", type=int, default=None,
        help="Number of synthetic image samples for embedding training.",
    )
    parser.add_argument(
        "--payload-size",
        choices=["128KB", "256KB", "512KB", "1MB"],
        default=None,
        help="Payload size per training sample.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Embedding pipeline training epochs (default: 5).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Training batch size (default: 4).",
    )
    parser.add_argument(
        "--fsl-epochs", type=int, default=None,
        help="SRNet FSL training epochs (default: 50).",
    )
    parser.add_argument(
        "--imsize", type=int, default=None,
        help="SRNet GF image side length (default: config value, 256). "
             "Use a smaller value for a fast smoke run.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Number of benign + malicious model checkpoints to generate "
             "for FSL training (total; split equally). (default: 10)",
    )
    parser.add_argument(
        "--test-samples", type=int, default=None,
        help="Number of held-out benign + malicious checkpoints for evaluation "
             "(default: 4).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Compute device: auto/cpu/cuda/mps.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Root output directory (default: outputs).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Global random seed (default: 42).",
    )
    parser.add_argument(
        "--skip-training", action="store_true", default=False,
        help="Skip embedding training; load the most recent checkpoint from "
             "outputs/checkpoints/ if available.",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def _resolve(cli_val, toml_val, default):
    if cli_val is not None:
        return cli_val
    if toml_val is not None:
        return toml_val
    return default


# ---------------------------------------------------------------------------
# Model checkpoint generation helpers
# ---------------------------------------------------------------------------


def _generate_benign_checkpoints(
    pipeline: EmbeddingPipeline,
    num_samples: int,
    output_dir: Path,
    seed: int = 42,
) -> List[Path]:
    """Save *num_samples* benign model checkpoints.

    Each checkpoint is the original host-model state_dict perturbed with a
    tiny Gaussian noise (std = 1e-4) so that the FSL detector sees variation
    across benign samples even with a single base model.

    Args:
        pipeline: Fitted :class:`~models.pipeline.EmbeddingPipeline`.
        num_samples: Number of checkpoints to generate.
        output_dir: Directory for saved ``.pt`` files.
        seed: RNG seed for reproducible noise.

    Returns:
        List of paths to the saved checkpoints.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(seed)
    base_state = pipeline.host_model.model.state_dict()
    paths: List[Path] = []

    for i in range(num_samples):
        perturbed = {}
        for k, v in base_state.items():
            if v.is_floating_point():
                noise = torch.randn(v.shape, generator=rng) * 1e-4
                perturbed[k] = v + noise
            else:
                perturbed[k] = v.clone()
        path = output_dir / f"benign_{i:03d}.pt"
        torch.save(perturbed, path)
        paths.append(path)
        logger.debug("Saved benign checkpoint: %s", path)

    logger.info("Generated %d benign checkpoints in %s.", num_samples, output_dir)
    return paths


def _generate_malicious_checkpoints(
    pipeline: EmbeddingPipeline,
    num_samples: int,
    payload_bits: int,
    output_dir: Path,
    device: torch.device,
    seed: int = 42,
) -> List[Path]:
    """Save *num_samples* malicious model checkpoints produced by the encoder.

    Each checkpoint contains the host-model state_dict with weights modified
    by the encoder to embed a unique random payload.

    Args:
        pipeline: Fitted :class:`~models.pipeline.EmbeddingPipeline`.
        num_samples: Number of checkpoints to generate.
        payload_bits: Number of payload bits per embedding.
        output_dir: Directory for saved ``.pt`` files.
        device: Compute device.
        seed: RNG seed for payload generation.

    Returns:
        List of paths to the saved checkpoints.
    """
    from utils.representation import weights_to_float_image, float_image_to_weights
    from utils.weights import extract_weights, flatten_weights, restore_weights, load_modified_weights
    import copy

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.eval()
    paths: List[Path] = []

    for i in range(num_samples):
        rng = torch.Generator().manual_seed(seed + i)
        payload = torch.randint(0, 2, (payload_bits,), generator=rng).float()

        with torch.no_grad():
            modified_repr, _ = pipeline.encode(payload.to(device))
            # modified_repr: (1, 1, H, W) float32 image in normalised [-1, 1]

        # Convert the modified float image back to flat float32 weights using
        # the same normalisation stats the pipeline cached for this host model.
        weight_records = extract_weights(pipeline.host_model.model)
        num_values = sum(r.values.numel() for r in weight_records)
        _, stats = weights_to_float_image(flatten_weights(weight_records))
        flat_modified = float_image_to_weights(
            modified_repr.squeeze(0).detach().cpu(),
            mean=stats.mean,
            scale=stats.scale,
            num_values=num_values,
        )

        # Rebuild the full state_dict with modified floating-point parameters.
        restored = restore_weights(flat_modified, weight_records)
        model_copy = copy.deepcopy(pipeline.host_model.model)
        load_modified_weights(model_copy, restored, strict=False)
        state = model_copy.state_dict()

        path = output_dir / f"malicious_{i:03d}.pt"
        torch.save(state, path)
        paths.append(path)
        logger.debug("Saved malicious checkpoint: %s", path)

    logger.info("Generated %d malicious checkpoints in %s.", num_samples, output_dir)
    return paths


# ---------------------------------------------------------------------------
# GF image extraction
# ---------------------------------------------------------------------------


def _load_gf_images(
    paths: List[Path],
    imsize: int = 256,
) -> List[np.ndarray]:
    """Load model checkpoints and convert each to a GF image.

    Args:
        paths: List of checkpoint ``.pt`` file paths.
        imsize: Target GF image side length.

    Returns:
        List of ``(imsize, imsize)`` uint8 numpy arrays.
    """
    gf_images = []
    for p in paths:
        state_dict = torch.load(p, map_location="cpu")
        gf = state_dict_to_gf_image(state_dict, imsize=imsize)
        gf_images.append(gf)
    return gf_images


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_epoch_losses(
    losses: List[float],
    output_dir: Path,
    title: str = "FSL Triplet Training Loss",
    filename: str = "fsl_training_loss.png",
) -> None:
    """Plot and save FSL training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), losses, color="#2563EB")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Triplet Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    dest = output_dir / filename
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved FSL loss curve to %s", dest)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    log_level = "DEBUG" if args.verbose else "INFO"
    configure_logging(level=log_level, log_file=None)

    # ---- Resolve configuration ----
    file_cfg: dict = {}
    if args.config is not None and Path(args.config).exists():
        file_cfg = load_config(args.config)
        logger.info("Loaded config from %s", args.config)

    host_sec   = file_cfg.get("host_model", {})
    data_sec   = file_cfg.get("data", {})
    train_sec  = file_cfg.get("training", {})
    enc_sec    = file_cfg.get("encoder", {})
    dec_sec    = file_cfg.get("decoder", {})
    loss_sec   = file_cfg.get("loss_weights", {})
    rt_sec     = file_cfg.get("runtime", {})
    srnet_sec  = file_cfg.get("srnet_detector", {})

    seed          = _resolve(args.seed, rt_sec.get("seed"), 42)
    device_str    = _resolve(args.device, rt_sec.get("device"), "auto")
    output_dir    = Path(_resolve(args.output_dir, None, Path("outputs")))
    dataset_choice = _resolve(args.dataset, data_sec.get("dataset"), "synthetic")
    host_model_name = _resolve(args.host_model, host_sec.get("name"), "resnet18")
    payload_size_str = _resolve(args.payload_size, data_sec.get("payload_size"), "128KB")
    payload_bits  = SUPPORTED_PAYLOAD_SIZES[payload_size_str] * 8
    default_classes = 10 if dataset_choice == "cifar10" else 1000
    num_classes   = _resolve(args.num_classes, host_sec.get("num_classes"), default_classes)

    # Demo-specific knobs
    embed_epochs  = _resolve(args.epochs, train_sec.get("max_epochs"), 5)
    batch_size    = _resolve(args.batch_size, train_sec.get("batch_size"), 4)
    fsl_epochs    = _resolve(args.fsl_epochs, srnet_sec.get("fsl_epochs"), 50)
    # ``--num-samples`` means the total training set size.  The TOML config
    # exposes per-class values, so retain that distinction when no CLI value
    # is supplied.
    if args.num_samples is not None:
        num_benign_samples = max(1, args.num_samples // 2)
        num_malicious_samples = max(1, args.num_samples - num_benign_samples)
    else:
        num_benign_samples = int(srnet_sec.get("num_benign_samples", 5))
        num_malicious_samples = int(srnet_sec.get("num_malicious_samples", 5))
    total_train_samples = num_benign_samples + num_malicious_samples
    test_samples  = _resolve(args.test_samples, None, max(2, total_train_samples // 3))
    imsize        = int(_resolve(args.imsize, srnet_sec.get("imsize"), 256))
    triplet_margin = float(srnet_sec.get("triplet_margin", 0.5))
    fsl_lr        = float(srnet_sec.get("fsl_lr", 6e-5))

    set_seed(seed)
    device = get_device(device_str)
    logger.info("Device: %s | Seed: %d", device, seed)

    # ---- Output dirs ----
    checkpoints_dir = output_dir / "checkpoints"
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    benign_dir  = output_dir / "checkpoints" / "benign"
    malicious_dir = output_dir / "checkpoints" / "malicious"
    benign_test_dir = output_dir / "checkpoints" / "benign_test"
    malicious_test_dir = output_dir / "checkpoints" / "malicious_test"
    fsl_model_dir = output_dir / "checkpoints" / "fsl_detector"

    for d in (checkpoints_dir, figures_dir, metrics_dir,
              benign_dir, malicious_dir, benign_test_dir, malicious_test_dir,
              fsl_model_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ====================================================================
    # Phase 1: Train the steganographic embedding pipeline
    # ====================================================================

    logger.info("=" * 60)
    logger.info("Phase 1: Embedding pipeline training (%d epochs)", embed_epochs)
    logger.info("=" * 60)

    encoder_cfg = EncoderConfig(
        payload_dim=payload_bits,
        base_channels=enc_sec.get("base_channels", 64),
        num_residual_blocks=enc_sec.get("num_residual_blocks", 4),
        payload_embedding_dim=enc_sec.get("payload_embedding_dim", 256),
        payload_chunk_size=enc_sec.get("payload_chunk_size", 1024),
        attention_reduction=enc_sec.get("attention_reduction", 8),
        dropout=enc_sec.get("dropout", 0.0),
        max_delta=enc_sec.get("max_delta", 1.0),
    )
    decoder_cfg = DecoderConfig(
        chunk_size=dec_sec.get("chunk_size", 1024),
        base_channels=dec_sec.get("base_channels", 64),
        num_residual_blocks=dec_sec.get("num_residual_blocks", 4),
        attention_reduction=dec_sec.get("attention_reduction", 8),
        chunk_position_dim=dec_sec.get("chunk_position_dim", 64),
        hidden_dim=dec_sec.get("hidden_dim", 256),
        dropout=dec_sec.get("dropout", 0.0),
    )
    pipeline_cfg = PipelineConfig(
        host_model_name=host_model_name,
        host_model_num_classes=num_classes,
        host_model_pretrained=host_sec.get("pretrained", False),
        train_host_model=False,
        payload_bits=payload_bits,
        encoder=encoder_cfg,
        decoder=decoder_cfg,
    )
    loss_weights = LossWeights(
        classification=loss_sec.get("alpha", 1.0),
        payload=loss_sec.get("beta", 1.0),
        distortion=loss_sec.get("gamma", 1.0),
        detector=loss_sec.get("delta", 1.0),
    )
    exp_cfg = ExperimentConfig(
        output_dir=str(output_dir),
        max_epochs=embed_epochs,
        batch_size=batch_size,
        learning_rate=train_sec.get("learning_rate", 1e-4),
        weight_decay=train_sec.get("weight_decay", 1e-5),
        gradient_clip_norm=train_sec.get("gradient_clip_norm", 1.0) or 1.0,
        early_stopping_patience=None,
        scheduler="cosine",
        scheduler_t_max=embed_epochs,
        device=device_str,
        pipeline=pipeline_cfg,
        loss_weights=loss_weights,
        num_workers=data_sec.get("num_workers", 0),
        log_every_n_steps=10,
        save_best_only=False,
    )

    # Build data loaders
    total_img_samples = int(
        _resolve(args.synthetic_samples, data_sec.get("synthetic_samples"), 64)
    )
    val_split = data_sec.get("val_split", 0.1)

    if dataset_choice == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
        cifar_full = CIFAR10(
            root=str(args.data_root), train=True, download=True, transform=transform
        )
        cap = min(total_img_samples, len(cifar_full))
        cifar_full = torch.utils.data.Subset(cifar_full, range(cap))
        val_n = max(1, int(cap * val_split))
        train_n = cap - val_n
        cifar_train, cifar_val = random_split(
            cifar_full, [train_n, val_n],
            generator=torch.Generator().manual_seed(seed),
        )
        train_ds = SteganographyDataset(cifar_train, payload_size=payload_size_str, payload_seed=seed)
        val_ds = SteganographyDataset(cifar_val, payload_size=payload_size_str, payload_seed=seed + 100_000)
    else:
        val_n = max(1, int(total_img_samples * val_split))
        train_n = total_img_samples - val_n
        train_ds = SyntheticImageDataset(
            count=train_n, payload_size=payload_size_str,
            num_classes=num_classes, image_size=32, seed=seed,
        )
        val_ds = SyntheticImageDataset(
            count=val_n, payload_size=payload_size_str,
            num_classes=num_classes, image_size=32, seed=seed + 100_000,
        )

    train_loader, val_loader = build_data_loaders(
        train_ds, val_ds,
        batch_size=batch_size,
        num_workers=data_sec.get("num_workers", 0),
    )

    embedding_pipeline: Optional[EmbeddingPipeline] = None
    embed_history: list = []

    if args.skip_training:
        # Try to find the latest checkpoint.
        ckpt_files = sorted(checkpoints_dir.glob("*.pt"))
        if ckpt_files:
            logger.info("--skip-training: loading checkpoint %s", ckpt_files[-1])
            pipeline_obj = EmbeddingPipeline(pipeline_cfg)
            ckpt = torch.load(ckpt_files[-1], map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt)
            pipeline_obj.load_state_dict(state, strict=False)
            embedding_pipeline = pipeline_obj
        else:
            logger.warning("No checkpoint found; training from scratch.")

    if embedding_pipeline is None:
        t0 = time.time()
        experiment = SteganographyExperiment(exp_cfg)
        result = experiment.run(train_loader, val_loader)
        embed_history = result.history
        logger.info(
            "Embedding training complete in %.1f s. "
            "Best checkpoint: %s",
            time.time() - t0,
            result.best_checkpoint,
        )
        # Re-build pipeline to use in checkpoint generation.
        embedding_pipeline = EmbeddingPipeline(pipeline_cfg)
        if result.best_checkpoint and result.best_checkpoint.exists():
            ckpt = torch.load(result.best_checkpoint, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt)
            embedding_pipeline.load_state_dict(state, strict=False)

    embedding_pipeline = embedding_pipeline.to(device)

    # ====================================================================
    # Phase 2: Generate detection datasets
    # ====================================================================

    logger.info("=" * 60)
    logger.info(
        "Phase 2: Generating %d benign + %d malicious checkpoints",
        num_benign_samples, num_malicious_samples,
    )
    logger.info("=" * 60)

    benign_train_paths = _generate_benign_checkpoints(
        embedding_pipeline, num_benign_samples, benign_dir, seed=seed
    )
    malicious_train_paths = _generate_malicious_checkpoints(
        embedding_pipeline, num_malicious_samples, payload_bits, malicious_dir, device, seed=seed
    )
    benign_test_paths = _generate_benign_checkpoints(
        embedding_pipeline, test_samples, benign_test_dir, seed=seed + 9_999
    )
    malicious_test_paths = _generate_malicious_checkpoints(
        embedding_pipeline, test_samples, payload_bits,
        malicious_test_dir, device, seed=seed + 9_999,
    )

    # ====================================================================
    # Phase 3: Train the SRNet FSL detector
    # ====================================================================

    logger.info("=" * 60)
    logger.info("Phase 3: Training SRNet FSL detector (%d epochs)", fsl_epochs)
    logger.info("=" * 60)

    logger.info("Loading GF images for FSL training…")
    benign_gf = _load_gf_images(benign_train_paths, imsize=imsize)
    malicious_gf = _load_gf_images(malicious_train_paths, imsize=imsize)

    fsl_config = FSLConfig(
        imsize=imsize,
        embedding_dim=512,
        triplet_margin=triplet_margin,
        learning_rate=fsl_lr,
        num_epochs=fsl_epochs,
        batch_size=int(srnet_sec.get("fsl_batch_size", 32)),
        num_triplets_per_epoch=max(
            200, (num_benign_samples + num_malicious_samples) * 20
        ),
        classifier="centroid",
        device=device_str,
        seed=seed,
    )

    fsl_detector = FSLDetector(config=fsl_config)
    t0 = time.time()
    fsl_losses = fsl_detector.fit(benign_gf, malicious_gf)
    logger.info("FSL training complete in %.1f s.", time.time() - t0)

    fsl_detector.save(fsl_model_dir)
    logger.info("FSL detector saved to %s.", fsl_model_dir)

    _plot_epoch_losses(fsl_losses, figures_dir)

    # ====================================================================
    # Phase 4: Evaluate detection on held-out models
    # ====================================================================

    logger.info("=" * 60)
    logger.info(
        "Phase 4: Evaluating on %d benign + %d malicious test models",
        test_samples, test_samples,
    )
    logger.info("=" * 60)

    # Wrap FSLDetector in the standard ModelXRayDetector interface.
    wrapped_detector = ModelXRayDetector(
        srnet_weights_path=fsl_model_dir / "srnet.pt",
        centroid_path=fsl_model_dir / "classifier.pt",
        device=device_str,
        image_size=imsize,
    )

    metrics = evaluate_detector(
        wrapped_detector,
        clean_model_paths=benign_test_paths,
        modified_model_paths=malicious_test_paths,
        verbose=True,
    )

    logger.info("\n%s", metrics)

    # ====================================================================
    # Phase 5: Save metrics and plots
    # ====================================================================

    logger.info("=" * 60)
    logger.info("Phase 5: Saving metrics and generating plots")
    logger.info("=" * 60)

    results_dict = {
        "embedding_epochs": embed_epochs,
        "fsl_epochs": fsl_epochs,
        "num_train_samples": total_train_samples,
        "num_test_samples": test_samples,
        "host_model": host_model_name,
        "payload_size": payload_size_str,
        "detection_metrics": metrics.to_dict(),
        "fsl_training_losses": fsl_losses,
    }

    results_path = metrics_dir / "demo_results.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # Plots
    try:
        fig = plot_score_distribution(metrics, title="Detection Score Distribution")
        fig.savefig(figures_dir / "demo_score_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not generate score distribution plot: %s", exc)

    try:
        fig = plot_roc_curve(metrics, title="Detection ROC Curve")
        fig.savefig(figures_dir / "demo_roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not generate ROC curve plot: %s", exc)

    try:
        fig = plot_confusion_matrix(metrics, title="Detection Confusion Matrix")
        fig.savefig(figures_dir / "demo_confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not generate confusion matrix plot: %s", exc)

    # ====================================================================
    # Summary
    # ====================================================================

    logger.info("=" * 60)
    logger.info("DEMO COMPLETE")
    logger.info("  Detection accuracy : %.4f", metrics.accuracy)
    logger.info("  Precision          : %.4f", metrics.precision)
    logger.info("  Recall             : %.4f", metrics.recall)
    logger.info("  F1-Score           : %.4f", metrics.f1_score)
    logger.info("  ROC-AUC            : %.4f", metrics.roc_auc)
    logger.info("  FPR                : %.4f", metrics.false_positive_rate)
    logger.info("  FNR                : %.4f", metrics.false_negative_rate)
    logger.info("Results in: %s", metrics_dir / "demo_results.json")
    logger.info("Plots in  : %s", figures_dir)
    logger.info("=" * 60)

    return 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_detections(detector: ModelXRayDetector, paths: list):
    """Yield (path, DetectionResult) tuples."""
    for p in paths:
        yield p, detector.predict(p)


if __name__ == "__main__":
    sys.exit(main())
