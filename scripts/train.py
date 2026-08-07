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
from torch.utils.data import random_split
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
    # chunk_size / payload_chunk_size are intentionally NOT tied to
    # payload_bits: they control the width of a shared, reused linear layer
    # (see models/encoder.py::PayloadEncoder and models/decoder.py::ChunkHead).
    # Setting them equal to payload_bits collapses "chunked" into one giant
    # dense layer — hundreds of millions of params for large payloads.
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

    # ---- Build pipeline config ----
    pipeline_cfg = PipelineConfig(
        host_model_name=_resolve(args.host_model, host_sec.get("name"), "resnet18"),
        host_model_num_classes=_resolve(args.num_classes, host_sec.get("num_classes"), default_num_classes),
        host_model_pretrained=_resolve(args.pretrained, host_sec.get("pretrained"), False),
        train_host_model=_resolve(args.train_host_model, host_sec.get("train_host_model"), False),
        payload_bits=payload_bits,
        encoder=encoder_cfg,
        decoder=decoder_cfg,
    )

    # ---- Build loss weights ----
    loss_weights = LossWeights(
        classification=_resolve(args.alpha, loss_sec.get("alpha"), 1.0),
        payload=_resolve(args.beta, loss_sec.get("beta"), 1.0),
        distortion=_resolve(args.gamma, loss_sec.get("gamma"), 1.0),
        detector=_resolve(args.delta, loss_sec.get("delta"), 1.0),
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
    image_size = _resolve(args.image_size, None, 32)

    if dataset_choice == "cifar10":
        # Real CIFAR-10 images/labels, wrapped with a fresh random payload
        # per sample via SteganographyDataset.  Downloads to --data-root on
        # first run (~170MB) and is cached for subsequent runs.
        tfms = [transforms.ToTensor()]
        if image_size != 32:
            tfms.insert(0, transforms.Resize((image_size, image_size)))
        transform = transforms.Compose(tfms)

        cifar_train_full = CIFAR10(
            root=str(args.data_root), train=True, download=True, transform=transform
        )
        cifar_test = CIFAR10(
            root=str(args.data_root), train=False, download=True, transform=transform
        )

        # Optionally cap CIFAR-10's 50k training images with --synthetic-samples
        # (reused as a generic "num samples" cap here for quick smoke tests).
        cap = _resolve(args.synthetic_samples, data_sec.get("synthetic_samples"), None)
        if cap is not None and cap < len(cifar_train_full):
            cifar_train_full = torch.utils.data.Subset(cifar_train_full, range(cap))

        val_n = max(1, int(len(cifar_train_full) * val_split))
        train_n = len(cifar_train_full) - val_n
        cifar_train, cifar_val = random_split(
            cifar_train_full,
            [train_n, val_n],
            generator=torch.Generator().manual_seed(seed),
        )

        logger.info(
            "CIFAR-10 dataset: %d train / %d val (held-out test set unused for now) "
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

    if args.resume is not None:
        logger.info("Checkpoint resumption requested from %s.", args.resume)
        logger.warning(
            "Automatic checkpoint resumption is not yet supported in the "
            "experiment runner.  Start training from scratch or call "
            "trainer.load_checkpoint() manually."
        )

    logger.info("Starting experiment…")
    result = experiment.run(train_loader, val_loader)

    logger.info("Experiment finished in %.1f seconds.", result.elapsed_seconds)
    if result.capacity is not None:
        logger.info("%s", result.capacity)
    if result.best_checkpoint is not None:
        logger.info("Best checkpoint: %s", result.best_checkpoint)

    return 0


if __name__ == "__main__":
    sys.exit(main())
