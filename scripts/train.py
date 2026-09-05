#!/usr/bin/env python3
"""Training entry point for the steganographic embedding pipeline.

Usage
-----
Basic run with all defaults (synthetic data, ResNet18 host, 10 epochs):

    python scripts/train.py

Custom configuration via TOML (CLI flags always override file values):

    python scripts/train.py \\
        --config configs/experiment.toml \\
        --host-model resnet18 \\
        --epochs 100 \\
        --batch-size 32 \\
        --lr 1e-4 \\
        --payload-size 128KB \\
        --output-dir outputs/run_01 \\
        --device auto

Real CIFAR-10 data instead of synthetic noise images:

    python scripts/train.py \\
        --dataset cifar10 \\
        --data-root ./data \\
        --num-classes 10 \\
        --host-model resnet18 \\
        --epochs 20

All four training objectives are always active:
  α · classification_loss   (preserve host model accuracy)
  β · payload_loss          (recover the random payload)
  γ · distortion_loss       (minimise weight-representation distance)
  δ · detector_loss         (fool the frozen Model X-Ray detector)

Configuration precedence (highest to lowest)
--------------------------------------------
1. CLI flags supplied at invocation time
2. Values from the TOML / JSON config file (``--config``)
3. Hard-coded defaults in this script
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, TypeVar

# Ensure the project root is on the path when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

from models.decoder import DecoderConfig
from models.encoder import EncoderConfig
from models.pipeline import PipelineConfig
from training.dataset import SteganographyDataset, SyntheticImageDataset, build_data_loaders
from training.experiment import ExperimentConfig, SteganographyExperiment
from training.losses import LossWeights
from utils.config import load_config
from utils.logging import configure_logging
from utils.seed import set_seed

T = TypeVar("T")


def _resolve(cli_val: T | None, toml_val: Any, default: T) -> T:
    """Return the highest-precedence non-None value.

    Priority: explicit CLI flag > TOML file value > hard-coded default.
    """
    if cli_val is not None:
        return cli_val  # type: ignore[return-value]
    if toml_val is not None:
        return toml_val
    return default


def _cifar_root(path: Path) -> Path:
    """Return torchvision's root when given its batches directory directly."""
    return path.parent if path.name == "cifar-10-batches-py" else path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the steganographic embedding pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (optional; CLI flags override TOML values)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a TOML or JSON experiment configuration file.",
    )

    # Data source
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "cifar10"],
        default=None,
        help="Image source for training. 'synthetic' uses random noise "
             "images; 'cifar10' downloads/uses real CIFAR-10 images. "
             "(default: synthetic)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data"),
        help="Directory to download/cache CIFAR-10 into. Ignored for "
             "'synthetic'. (default: ./data)",
    )
    parser.add_argument(
        "--download-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download CIFAR-10 when it is not already present.",
    )

    # Pipeline — default=None so we can detect explicit CLI supply
    parser.add_argument(
        "--host-model",
        choices=["resnet18", "resnet50", "mobilenet_v2", "vgg16"],
        default=None,
        help="Host model backbone. (default: resnet18)",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load official pretrained ImageNet weights for the host model. (default: False)",
    )
    parser.add_argument(
        "--train-host-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow the host model weights to be fine-tuned. (default: False)",
    )
    parser.add_argument(
        "--host-checkpoint",
        type=Path,
        default=None,
        help="Path to a separately fine-tuned host-model checkpoint.",
    )
    parser.add_argument(
        "--payload-size",
        choices=["128KB", "256KB", "512KB", "1MB"],
        default=None,
        help="Payload size embedded per training sample. (default: 128KB)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of host model output classes. "
             "(default: 1000 for synthetic, 10 for cifar10)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Spatial size of synthetic training images in pixels. (default: 32)",
    )

    # Training — all default=None for correct CLI-over-TOML precedence
    parser.add_argument("--epochs", type=int, default=None,
                        help="Maximum training epochs. (default: 10)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size. (default: 4)")
    parser.add_argument(
        "--payload-replicas", type=int, default=None,
        help=(
            "Number of full-resolution encoder/decoder/detector forward "
            "passes per step, independent of --batch-size (which only "
            "affects the classification loss). Keep this small (1-2) on "
            "limited-VRAM GPUs — the encoder runs its CNN over the ENTIRE "
            "weight image once per replica, so memory scales with this "
            "value, not with --batch-size. (default: 1)"
        ),
    )
    parser.add_argument("--lr", type=float, default=None,
                        help="Initial learning rate. (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="AdamW weight decay. (default: 1e-5)")
    parser.add_argument(
        "--gradient-clip", type=float, default=None,
        help="Gradient clipping max norm; 0 to disable. (default: 1.0)",
    )
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "step", "reduce_on_plateau"],
        default=None,
        help="Learning-rate scheduler. (default: cosine)",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Early stopping patience in epochs; omit to disable. (default: None)",
    )
    parser.add_argument("--mixed-precision", action="store_true", default=False,
                        help="Enable AMP on CUDA.")

    # Loss weights
    parser.add_argument("--alpha", type=float, default=None,
                        help="Classification loss weight α. (default: 1.0)")
    parser.add_argument("--beta", type=float, default=None,
                        help="Payload recovery loss weight β. (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Weight distortion loss weight γ. (default: 1.0)")
    parser.add_argument("--delta", type=float, default=None,
                        help="Detector evasion loss weight δ. (default: 1.0)")

    # Data
    parser.add_argument(
        "--synthetic-samples", type=int, default=None,
        help="Number of synthetic training images. (default: 128)",
    )
    parser.add_argument(
        "--val-split", type=float, default=None,
        help="Fraction of synthetic samples reserved for validation. (default: 0.1)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="DataLoader worker processes. (default: 0)",
    )

    # Infrastructure
    parser.add_argument("--device", default=None,
                        help="Compute device: auto/cpu/cuda/mps. (default: auto)")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Root directory for checkpoints, logs, and figures. (default: outputs)",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Global random seed. (default: 42)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to a checkpoint to resume.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable DEBUG logging.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log_level = "DEBUG" if args.verbose else "INFO"
    configure_logging(level=log_level, log_file=None)
    logger = logging.getLogger(__name__)

    seed = _resolve(args.seed, None, 42)
    set_seed(seed)
    logger.info("Seed set to %d.", seed)

    # ---- Load base config from TOML/JSON ----
    # The TOML uses nested sections: [host_model], [data], [training], etc.
    # CLI flags take precedence over all TOML values.
    file_cfg: dict = {}
    if args.config is not None:
        file_cfg = load_config(args.config)
        logger.info("Loaded config from %s", args.config)

    host_sec   = file_cfg.get("host_model", {})
    data_sec   = file_cfg.get("data", {})
    train_sec  = file_cfg.get("training", {})
    enc_sec    = file_cfg.get("encoder", {})
    dec_sec    = file_cfg.get("decoder", {})
    loss_sec   = file_cfg.get("loss_weights", {})
    rt_sec     = file_cfg.get("runtime", {})

    # ---- Resolve payload bits ----
    # payload_dim for the encoder is ALWAYS derived from payload_size.
    # It must never be taken from the TOML [encoder] section, because the TOML
    # value can be stale and any mismatch causes an immediate shape error.
    from utils.payload import SUPPORTED_PAYLOAD_SIZES
    payload_size_str: str = _resolve(
        args.payload_size, data_sec.get("payload_size"), "128KB"
    )
    payload_bits: int = SUPPORTED_PAYLOAD_SIZES[payload_size_str] * 8

    # ---- Resolve dataset choice ----
    dataset_choice: str = _resolve(args.dataset, data_sec.get("dataset"), "synthetic")
    default_num_classes = 10 if dataset_choice == "cifar10" else 1000

    # ---- Build encoder / decoder configs ----
    # payload_dim MUST equal payload_bits (it's the actual input length) — it
    # is NOT read from the TOML [encoder] section, to prevent mismatches.
    # Architecture v2 (dense/fully-convolutional): no chunking, no shared
    # per-chunk MLP head. Payload is reshaped into a square bitmap and
    # processed/decoded with full spatial correspondence throughout — see
    # models/encoder.py and models/decoder.py module docstrings for why.
    # bits_per_pixel: how many independent payload bits each weight pixel
    # carries.  1 = original design (ceiling ~277KB for MobileNetV2).
    # Higher values multiply capacity: bpp=4 -> ~1.1MB / 12.5% embed rate.
    # Must be identical in encoder and decoder configs.
    bits_per_pixel = enc_sec.get("bits_per_pixel", dec_sec.get("bits_per_pixel", 1))

    encoder_cfg = EncoderConfig(
        payload_dim=payload_bits,
        base_channels=enc_sec.get("base_channels", 64),
        num_residual_blocks=enc_sec.get("num_residual_blocks", 4),
        message_channels=enc_sec.get("message_channels", 32),
        message_prep_layers=enc_sec.get("message_prep_layers", 2),
        attention_reduction=enc_sec.get("attention_reduction", 8),
        # max_delta is now the bound in normalised [-1, 1] float weight space
        # (see EncoderConfig.max_delta); small by default for stealth.
        max_delta=enc_sec.get("max_delta", 0.05),
        gradient_checkpointing=enc_sec.get("gradient_checkpointing", False),
        bits_per_pixel=bits_per_pixel,
        adaptive_capacity=enc_sec.get("adaptive_capacity", False),
    )
    decoder_cfg = DecoderConfig(
        base_channels=dec_sec.get("base_channels", 64),
        num_residual_blocks=dec_sec.get("num_residual_blocks", 4),
        attention_reduction=dec_sec.get("attention_reduction", 8),
        gradient_checkpointing=dec_sec.get("gradient_checkpointing", False),
        bits_per_pixel=bits_per_pixel,
    )

    # ---- Build pipeline config ----
    pipeline_sec = file_cfg.get("pipeline", {})
    host_checkpoint = _resolve(args.host_checkpoint, host_sec.get("checkpoint"), None)
    pipeline_cfg = PipelineConfig(
        host_model_name=_resolve(args.host_model, host_sec.get("name"), "resnet18"),
        host_model_num_classes=_resolve(args.num_classes, host_sec.get("num_classes"), default_num_classes),
        host_model_pretrained=_resolve(args.pretrained, host_sec.get("pretrained"), False),
        host_model_checkpoint=str(host_checkpoint) if host_checkpoint is not None else None,
        train_host_model=_resolve(args.train_host_model, host_sec.get("train_host_model"), False),
        payload_bits=payload_bits,
        payload_replicas=_resolve(args.payload_replicas, train_sec.get("payload_replicas"), 1),
        encoder=encoder_cfg,
        decoder=decoder_cfg,
        # Reference-based decoding: decoder receives (modified - original) instead
        # of the raw modified representation. See PipelineConfig.reference_decoding.
        # Default True — the correct permanent design for keyed model steganography.
        reference_decoding=pipeline_sec.get("reference_decoding", True),
    )

    # ---- Build loss weights ----
    loss_weights = LossWeights(
        classification=_resolve(args.alpha, loss_sec.get("alpha"), 1.0),
        payload=_resolve(args.beta, loss_sec.get("beta"), 1.0),
        distortion=_resolve(args.gamma, loss_sec.get("gamma"), 1.0),
        detector=_resolve(args.delta, loss_sec.get("delta"), 1.0),
    )
    # Explicit confirmation of the ACTUALLY-resolved weights (as opposed to
    # what's on disk in the config file, which CLI flags override silently)
    # — printed unconditionally, not gated by --verbose, since a mismatch
    # here has caused real confusion before.
    logger.info(
        "Resolved loss weights: alpha(classification)=%s beta(payload)=%s "
        "gamma(distortion)=%s delta(detector)=%s",
        loss_weights.classification,
        loss_weights.payload,
        loss_weights.distortion,
        loss_weights.detector,
    )

    # ---- Build experiment config ----
    gradient_clip_raw = _resolve(args.gradient_clip, train_sec.get("gradient_clip_norm"), 1.0)
    gradient_clip = gradient_clip_raw if gradient_clip_raw and gradient_clip_raw > 0 else None
    max_epochs = _resolve(args.epochs, train_sec.get("max_epochs"), 10)
    exp_cfg = ExperimentConfig(
        output_dir=str(_resolve(args.output_dir, None, Path("outputs"))),
        max_epochs=max_epochs,
        batch_size=_resolve(args.batch_size, train_sec.get("batch_size"), 4),
        learning_rate=_resolve(args.lr, train_sec.get("learning_rate"), 1e-4),
        weight_decay=_resolve(args.weight_decay, train_sec.get("weight_decay"), 1e-5),
        gradient_clip_norm=gradient_clip,
        mixed_precision=_resolve(None, train_sec.get("mixed_precision"), args.mixed_precision),
        early_stopping_patience=_resolve(
            args.patience, train_sec.get("early_stopping_patience"), None
        ),
        # Curriculum: alpha ramps 0 → target over first N epochs.
        # See ExperimentConfig.alpha_warmup_epochs for the rationale.
        alpha_warmup_epochs=_resolve(
            None, train_sec.get("alpha_warmup_epochs"), 0
        ),
        scheduler=_resolve(args.scheduler, train_sec.get("scheduler"), "cosine"),
        scheduler_t_max=_resolve(None, train_sec.get("scheduler_t_max"), max_epochs),
        device=_resolve(args.device, rt_sec.get("device"), "auto"),
        pipeline=pipeline_cfg,
        loss_weights=loss_weights,
        num_workers=_resolve(args.num_workers, data_sec.get("num_workers"), 0),
        log_every_n_steps=_resolve(None, train_sec.get("log_every_n_steps"), 10),
        save_best_only=_resolve(None, train_sec.get("save_best_only"), False),
    )

    # ---- Build image datasets ----
    val_split = _resolve(args.val_split, data_sec.get("val_split"), 0.1)
    image_size = _resolve(args.image_size, data_sec.get("image_size"), 32)

    if dataset_choice == "cifar10":
        # Real CIFAR-10 images/labels, wrapped with a fresh random payload
        # per sample via SteganographyDataset.  Downloads to --data-root on
        # first run (~170MB) and is cached for subsequent runs.
        tfms = []
        if image_size != 32:
            tfms.append(transforms.Resize((image_size, image_size)))
        tfms.append(transforms.ToTensor())
        if pipeline_cfg.host_model_pretrained:
            tfms.append(transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ))
        transform = transforms.Compose(tfms)

        data_root = _cifar_root(args.data_root)
        cifar_train_full = CIFAR10(
            root=str(data_root), train=True, download=args.download_dataset, transform=transform
        )
        cifar_test = CIFAR10(
            root=str(data_root), train=False, download=args.download_dataset, transform=transform
        )

        # A calibration subset limits costly full-weight-image passes. The
        # host itself was already fine-tuned on the full training split.
        train_cap = data_sec.get("max_train_samples")
        if train_cap is not None and train_cap < len(cifar_train_full):
            cifar_train_full = torch.utils.data.Subset(cifar_train_full, range(train_cap))

        # Validate on CIFAR-10's official held-out test split, never on data
        # the host saw during pretraining.
        val_cap = data_sec.get("max_val_samples")
        if val_cap is not None and val_cap < len(cifar_test):
            cifar_test = torch.utils.data.Subset(cifar_test, range(val_cap))

        cifar_train = cifar_train_full
        cifar_val = cifar_test
        train_n = len(cifar_train)
        val_n = len(cifar_val)

        logger.info(
            "CIFAR-10 dataset: %d calibration-train / %d held-out-test val "
            "— %dx%d — payload %s (%d bits).",
            train_n, val_n, image_size, image_size, payload_size_str, payload_bits,
        )

        train_ds = SteganographyDataset(cifar_train, payload_size=payload_size_str, payload_seed=seed)
        val_ds = SteganographyDataset(cifar_val, payload_size=payload_size_str, payload_seed=seed + 100_000)
    else:
        # SyntheticImageDataset yields (3, H, W) random-noise RGB images —
        # useful for smoke-testing the pipeline only. Do NOT use
        # SyntheticWeightDataset here: that yields 4-channel weight
        # representations, not classification images.
        total = _resolve(args.synthetic_samples, data_sec.get("synthetic_samples"), 128)
        val_n = max(1, int(total * val_split))
        train_n = total - val_n

        logger.info(
            "Synthetic image dataset: %d train / %d val — %dx%d — payload %s (%d bits).",
            train_n, val_n, image_size, image_size, payload_size_str, payload_bits,
        )

        train_ds = SyntheticImageDataset(
            count=train_n,
            payload_size=payload_size_str,
            num_classes=pipeline_cfg.host_model_num_classes,
            image_size=image_size,
            seed=seed,
        )
        val_ds = SyntheticImageDataset(
            count=val_n,
            payload_size=payload_size_str,
            num_classes=pipeline_cfg.host_model_num_classes,
            image_size=image_size,
            seed=seed + 100_000,
        )

    train_loader, val_loader = build_data_loaders(
        train_ds,
        val_ds,
        batch_size=exp_cfg.batch_size,
        num_workers=exp_cfg.num_workers,
    )

    # ---- Run experiment ----
    experiment = SteganographyExperiment(exp_cfg)

    logger.info("Starting experiment…")
    result = experiment.run(train_loader, val_loader, resume_from=args.resume)

    logger.info("Experiment finished in %.1f seconds.", result.elapsed_seconds)
    if result.capacity is not None:
        logger.info("%s", result.capacity)
    if result.best_checkpoint is not None:
        logger.info("Best checkpoint: %s", result.best_checkpoint)

    return 0


if __name__ == "__main__":
    sys.exit(main())
