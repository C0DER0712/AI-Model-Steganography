"""Train and evaluate the Model X-Ray-style SRNet few-shot detector.

The positive class is a clean checkpoint with a seeded random byte payload
written into X least significant float32 bits. This is a benign surrogate for
the paper's payload-bearing class; it does not create or execute malware.

Example::

    python scripts/train_detector.py --clean-dir data/model_zoo \
        --output-dir outputs/detector --payload-size 1MB --train-severity auto
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from evaluation.fsl_detector import FSLConfig, FSLDetector, weighted_metric
from utils.detector_data import (
    count_attackable_parameters,
    embed_random_lsb_payload,
    list_checkpoints,
    load_state_dict,
    state_dict_to_gf,
)


LOG = logging.getLogger("train_detector")
_PAYLOAD_BYTES = {"128KB": 128 * 1024, "256KB": 256 * 1024,
                  "512KB": 512 * 1024, "1MB": 1024 * 1024}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SRNet FSL on clean vs benign-random-payload model checkpoints."
    )
    parser.add_argument("--clean-dir", type=Path, required=True,
                        help="Directory of independent clean model checkpoints (.pt/.pth/.ckpt/.bin).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/detector"))
    parser.add_argument("--heldout-dir", type=Path, default=None,
                        help="Optional disjoint model zoo for OOD testing; variants are created after the split.")
    parser.add_argument("--payload-size", choices=tuple(_PAYLOAD_BYTES), default="1MB")
    parser.add_argument("--train-severity", default="auto",
                        help="Number of float32 LSBs (1..23), or 'auto' to fit the requested payload on every source model.")
    parser.add_argument("--eval-severities", default=",".join(map(str, range(1, 24))),
                        help="Comma-separated X-LSB severities evaluated on held-out source models.")
    parser.add_argument("--test-fraction", type=float, default=0.25,
                        help="Fraction of source checkpoints held out before making any variants.")
    parser.add_argument("--imsize", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100,
                        help="Upper bound for paper-style threshold-stopped FSL training.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--triplets-per-epoch", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--loss-threshold-lower", type=float, default=0.1)
    parser.add_argument("--loss-threshold-upper", type=float, default=0.4)
    parser.add_argument("--classifier", choices=("centroid", "1nn"), default="centroid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _scores(detector: FSLDetector, images: list[np.ndarray], labels: list[int]):
    y_pred: list[int] = []
    y_score: list[float] = []
    for image in images:
        result = detector.predict_gf(image)
        y_pred.append(int(result.is_malicious))
        y_score.append(float(result.anomaly_score))
    y_true = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    score = np.asarray(y_score, dtype=np.float64)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else None,
        "n_clean": int((y_true == 0).sum()),
        "n_payload_modified": int((y_true == 1).sum()),
    }


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.train_severity != "auto":
        try:
            args.train_severity = int(args.train_severity)
        except ValueError as exc:
            raise SystemExit("--train-severity must be 'auto' or an integer in 1..23") from exc
        if not 1 <= args.train_severity <= 23:
            raise SystemExit("--train-severity must be in 1..23")
    if not 0.1 <= args.test_fraction <= 0.5:
        raise SystemExit("--test-fraction must be between 0.1 and 0.5")
    eval_severities = sorted(set(int(s.strip()) for s in args.eval_severities.split(",") if s.strip()))
    if not eval_severities or any(s < 1 or s > 23 for s in eval_severities):
        raise SystemExit("--eval-severities must contain values in 1..23")

    train_paths = list_checkpoints(args.clean_dir)
    if args.heldout_dir is not None:
        test_paths = list_checkpoints(args.heldout_dir)
        overlap = {p.resolve() for p in train_paths} & {p.resolve() for p in test_paths}
        if len(train_paths) < 2 or len(test_paths) < 2 or overlap:
            raise SystemExit(
                "--clean-dir needs at least 2 checkpoints and --heldout-dir needs at least 2 disjoint checkpoints."
            )
        paths = train_paths + test_paths
        split_kind = "user_supplied_disjoint_model_zoo"
    else:
        paths = train_paths.copy()
        if len(paths) < 5:
            raise SystemExit(
                f"Found {len(paths)} checkpoint(s); at least 5 independent clean source models are required. "
                "A valid held-out-source evaluation cannot be made from repeated variants of one model."
            )
        random.seed(args.seed)
        rng = random.Random(args.seed)
        shuffled = paths.copy()
        rng.shuffle(shuffled)
        n_test = max(2, min(len(paths) - 2, math.ceil(len(paths) * args.test_fraction)))
        test_paths = shuffled[:n_test]
        train_paths = shuffled[n_test:]
        split_kind = "random_source_checkpoint_holdout"
    LOG.info("Source-model split: %d train / %d held-out test", len(train_paths), len(test_paths))
    LOG.info("Labels are clean vs random-payload-modified (benign surrogate).")

    payload_bytes = _PAYLOAD_BYTES[args.payload_size]
    source_parameter_counts = {
        path: count_attackable_parameters(load_state_dict(path)) for path in paths
    }
    if min(source_parameter_counts.values()) < 1:
        raise SystemExit("At least one float32 parameter is required in every source model.")
    if args.train_severity == "auto":
        min_parameters = min(source_parameter_counts.values())
        args.train_severity = max(1, math.ceil(payload_bytes * 8 / min_parameters))
        if args.train_severity > 23:
            raise SystemExit(
                "Requested payload does not fit in the smallest source model's 23 float32 mantissa bits."
            )
        LOG.info("Auto-selected training severity X=%d to fit %s on every source model",
                 args.train_severity, args.payload_size)

    train_clean: list[np.ndarray] = []
    train_modified: list[np.ndarray] = []
    for index, path in enumerate(train_paths):
        clean = load_state_dict(path)
        modified = embed_random_lsb_payload(
            clean, severity=args.train_severity,
            payload_bytes=payload_bytes, seed=args.seed + index,
        )
        train_clean.append(state_dict_to_gf(clean, imsize=args.imsize))
        train_modified.append(state_dict_to_gf(modified, imsize=args.imsize))
        LOG.info("Prepared training source %s", path.name)

    config = FSLConfig(
        imsize=args.imsize,
        embedding_dim=args.embedding_dim,
        triplet_margin=args.margin,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_triplets_per_epoch=args.triplets_per_epoch,
        classifier=args.classifier,
        device=args.device,
        seed=args.seed,
        stop_when_loss_in_range=True,
        loss_threshold_lower=args.loss_threshold_lower,
        loss_threshold_upper=args.loss_threshold_upper,
        exhaustive_triplets=True,
    )
    detector = FSLDetector(config=config)
    losses = detector.fit(train_clean, train_modified)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector.save(args.output_dir / "fsl_detector")

    heldout_clean_states = [load_state_dict(path) for path in test_paths]
    clean_images = [state_dict_to_gf(s, imsize=args.imsize) for s in heldout_clean_states]
    clean_accuracy = float(np.mean([
        detector.predict_gf(image).is_malicious is False for image in clean_images
    ]))
    # Evaluate each X separately with the same held-out clean source models.
    per_severity: dict[int, dict] = {}
    for severity in eval_severities:
        images = clean_images.copy()
        labels = [0] * len(images)
        for index, (path, clean_state) in enumerate(zip(test_paths, heldout_clean_states)):
            carrier = embed_random_lsb_payload(
                clean_state, severity=severity,
                payload_bytes=payload_bytes,
                seed=args.seed + 100_000 + severity * 1_000 + index,
            )
            images.append(state_dict_to_gf(carrier, imsize=args.imsize))
            labels.append(1)
        per_severity[severity] = _scores(detector, images, labels)
        LOG.info("Held-out severity X=%d: %s", severity, per_severity[severity])

    # Paper's weighted metric is only reported for a complete severity sweep.
    max_eval = max(eval_severities)
    paper_metric = weighted_metric(
        {0: clean_accuracy,
         **{x: per_severity[x]["accuracy"] for x in eval_severities}},
        max_severity=max_eval,
    ) if eval_severities == list(range(1, 24)) else None

    report = {
        "task": "clean_vs_benign_random_payload_modified",
        "payload_type": "seeded_random_bytes_tiled_into_float32_lsb_slots",
        "payload_size_bytes": payload_bytes,
        "train_payload_capacity_bits_by_source": {
            str(path): source_parameter_counts[path] * args.train_severity
            for path in train_paths
        },
        "train_payload_bytes_fully_embedded_by_source": {
            str(path): min(payload_bytes, source_parameter_counts[path] * args.train_severity // 8)
            for path in train_paths
        },
        "training_lsb_severity": args.train_severity,
        "evaluated_lsb_severities": eval_severities,
        "split_unit": split_kind + ":source_checkpoint_before_variant_generation",
        "train_sources": [str(p) for p in train_paths],
        "heldout_sources": [str(p) for p in test_paths],
        "clean_only_accuracy": clean_accuracy,
        "gf_representation": "Model X-Ray interleaved 2x2 float-byte tiles",
        "classifier": args.classifier,
        "triplet_loss": "mean(relu(||a-p||2_squared - ||a-n||2_squared + margin))",
        "training_triplet_loss": losses,
        "per_severity_metrics": per_severity,
        "paper_weighted_metric": paper_metric,
        "note": "Results measure detection of benign random LSB modifications, not malware detection.",
    }
    report_path = args.output_dir / "detector_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("Saved detector and results under %s", args.output_dir)
    LOG.info("Held-out source identities: %s", ", ".join(p.name for p in test_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
