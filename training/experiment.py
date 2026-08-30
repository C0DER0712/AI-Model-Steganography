"""Experiment orchestration for steganographic embedding training.

Assembles the complete training pipeline — host model, encoder, decoder,
frozen detector, composite loss, optimizer, scheduler, and trainer — then
runs training, evaluates the trained model, generates plots, and persists
all artefacts to the configured output directory.

This module is the single authoritative entry point for running a full
experiment.  CLI scripts in ``scripts/`` delegate to :func:`run_experiment`.
"""

from __future__ import annotations

import json
import logging
import time
import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluation.capacity import CapacityResult, compute_capacity
from evaluation.differentiable_detector import DifferentiableDetectorConfig
from evaluation.plotting import plot_metrics_summary
from models.decoder import DecoderConfig
from models.encoder import EncoderConfig
from models.pipeline import EmbeddingPipeline, PipelineConfig
from training.losses import CompositeLoss, LossInputs, LossOutput, LossWeights
from training.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Full experiment configuration.

    Attributes:
        output_dir: Root output directory for checkpoints, logs, and figures.
        max_epochs: Maximum training epochs.
        batch_size: Training and validation batch size.
        learning_rate: Initial optimiser learning rate.
        weight_decay: AdamW weight-decay coefficient.
        gradient_clip_norm: Optional gradient clipping max norm.
        mixed_precision: Enable AMP on CUDA.
        early_stopping_patience: Epochs without improvement before stopping.
        scheduler: LR scheduler type (``"cosine"``, ``"step"``, or
            ``"reduce_on_plateau"``).
        scheduler_step_size: Step size for ``StepLR``.
        scheduler_gamma: Decay factor for ``StepLR``.
        scheduler_t_max: Period for ``CosineAnnealingLR``.
        device: Compute device (``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``).
        pipeline: Pipeline architecture configuration.
        loss_weights: Composite loss term weights.
        log_every_n_steps: TensorBoard step logging interval.
        save_best_only: Only save the best checkpoint.
        num_workers: DataLoader worker count.
    """

    output_dir: str = "outputs"
    max_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float | None = 1.0
    mixed_precision: bool = False
    early_stopping_patience: int | None = 10
    scheduler: str = "cosine"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5
    scheduler_t_max: int = 50
    device: str = "auto"
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    log_every_n_steps: int = 10
    save_best_only: bool = False
    num_workers: int = 0
    # Curriculum learning: number of epochs over which alpha (classification
    # loss weight) is linearly ramped from 0 up to its configured target value.
    # 0 = no warmup (alpha is fixed at its configured value from epoch 1).
    #
    # WHY: With a large payload and max_delta, the encoder's payload gradient
    # is strong and fast-converging. Introducing the classification gradient
    # simultaneously from epoch 1 creates a tug-of-war that slows both
    # objectives. The warmup lets the encoder first learn WHERE and HOW to
    # embed bits (unconstrained, strong signal), then gradually introduces
    # accuracy pressure that forces the encoder to find more efficient,
    # subtle embeddings — exactly the "embed aggressively first, then learn
    # stealth" curriculum.
    alpha_warmup_epochs: int = 0


@dataclass(frozen=True)
class ExperimentResult:
    """Summary of a completed experiment run.

    Attributes:
        history: Per-epoch metric dictionaries.
        capacity: Embedding capacity computed at the end of training.
        best_checkpoint: Path to the best saved checkpoint.
        output_dir: Root output directory.
        elapsed_seconds: Wall-clock training duration.
    """

    history: list[dict[str, float]]
    capacity: CapacityResult | None
    best_checkpoint: Path | None
    output_dir: Path
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary dictionary."""
        cap_dict = self.capacity.to_dict() if self.capacity is not None else {}
        return {
            "history": self.history,
            "capacity": cap_dict,
            "best_checkpoint": str(self.best_checkpoint) if self.best_checkpoint else None,
            "output_dir": str(self.output_dir),
            "elapsed_seconds": self.elapsed_seconds,
        }


class SteganographyExperiment:
    """Manages the full lifecycle of a steganography training experiment.

    Usage::

        exp = SteganographyExperiment(config)
        result = exp.run(train_loader, val_loader)
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self._setup_output_dirs()
        self._setup_logging()

    def run(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any] | None = None,
        resume_from: str | Path | None = None,
    ) -> ExperimentResult:
        """Build all components, train, evaluate, and save artefacts.

        Args:
            train_loader: Training data loader yielding
                ``(images, labels, payload_bits)`` tuples.
            val_loader: Optional validation data loader with the same schema.
            resume_from: Optional path to a checkpoint (as saved by
                `Trainer` to `checkpoints/latest.pt` or `best.pt`). When
                given, model/optimizer/scheduler/scaler state and the
                trainer's epoch counter are restored before training
                resumes, continuing from the next epoch rather than
                starting over.

        Returns:
            :class:`ExperimentResult` summarising the completed run.
        """
        start_time = time.time()
        cfg = self.config

        # ---- Build pipeline ----
        logger.info("Building pipeline…")
        # Skip the detector's and/or classification's forward pass entirely
        # when their loss weight is 0 (see PipelineConfig.run_classification/
        # run_detector) — no point paying for a full CNN forward pass over
        # the whole weight image (the detector) or a functional_call through
        # the host model (classification) when neither contributes to the
        # loss. This is derived fresh from cfg.loss_weights on every run, so
        # switching alpha/delta back on for the full 4-objective run
        # automatically re-enables both forward passes — nothing to revert
        # by hand.
        pipeline_cfg = dataclasses.replace(
            cfg.pipeline,
            run_classification=cfg.loss_weights.classification > 0,
            run_detector=cfg.loss_weights.detector > 0,
        )
        logger.info(
            "Pipeline forward passes: classification=%s (alpha=%s), detector=%s (delta=%s)",
            pipeline_cfg.run_classification,
            cfg.loss_weights.classification,
            pipeline_cfg.run_detector,
            cfg.loss_weights.detector,
        )
        pipeline = EmbeddingPipeline(pipeline_cfg)
        logger.info(
            "Pipeline built: host=%s, encoder params=%d, decoder params=%d",
            cfg.pipeline.host_model_name,
            sum(p.numel() for p in pipeline.encoder.parameters()),
            sum(p.numel() for p in pipeline.decoder.parameters()),
        )

        # ---- Build loss ----
        loss_fn = CompositeLoss(weights=cfg.loss_weights)

        # ---- Build optimizer ----
        trainable_params = list(pipeline.encoder.parameters()) + list(
            pipeline.decoder.parameters()
        )
        if cfg.pipeline.train_host_model:
            trainable_params += list(pipeline.host_model.parameters())

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # ---- Build scheduler ----
        scheduler = _build_scheduler(optimizer, cfg)

        # ---- Wrap scheduler with curriculum (alpha warmup) if configured ----
        if cfg.alpha_warmup_epochs > 0 and cfg.loss_weights.classification > 0:
            logger.info(
                "Alpha curriculum: ramping classification weight 0 → %.4f over %d epochs.",
                cfg.loss_weights.classification,
                cfg.alpha_warmup_epochs,
            )
            # Start alpha at 0 (the ramp-up starts from epoch 1 in step())
            loss_fn.set_alpha(0.0)
            scheduler = AlphaCurriculumScheduler(
                scheduler, loss_fn, cfg.alpha_warmup_epochs, cfg.loss_weights.classification
            )

        # ---- Build trainer ----
        trainer_config = TrainerConfig(
            max_epochs=cfg.max_epochs,
            device=cfg.device,
            mixed_precision=cfg.mixed_precision,
            gradient_clip_norm=cfg.gradient_clip_norm,
            checkpoint_dir=str(self.output_dir / "checkpoints"),
            log_dir=str(self.output_dir / "tensorboard"),
            save_best_only=cfg.save_best_only,
            monitor="val_loss" if val_loader is not None else "train_loss",
            monitor_mode="min",
            early_stopping_patience=cfg.early_stopping_patience,
            log_every_n_steps=cfg.log_every_n_steps,
            scheduler_interval=_scheduler_interval(cfg.scheduler),
        )

        trainer = Trainer(
            model=pipeline,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            config=trainer_config,
            val_loader=val_loader,
            scheduler=scheduler,
            batch_to_loss_inputs=_pipeline_batch_adapter,
        )

        # ---- Resume from checkpoint, if requested ----
        if resume_from is not None:
            logger.info("Resuming from checkpoint: %s", resume_from)
            trainer.load_checkpoint(resume_from)
            logger.info(
                "Restored model/optimizer/scheduler/scaler state. "
                "Resuming at epoch %d of %d.",
                trainer.state.epoch + 1,
                cfg.max_epochs,
            )

        # ---- Train ----
        logger.info("Starting training for %d epochs…", cfg.max_epochs)
        history = trainer.train()
        trainer.close()
        elapsed = time.time() - start_time
        logger.info("Training complete in %.1f seconds.", elapsed)

        # ---- Compute capacity ----
        capacity: CapacityResult | None = None
        try:
            final_ber = _extract_final_ber(history)
            capacity = compute_capacity(
                pipeline.host_model.model,
                cfg.pipeline.payload_bits,
                bit_error_rate=final_ber,
            )
            logger.info("Capacity: %s", capacity)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not compute capacity: %s", exc)

        # ---- Persist artefacts ----
        best_ckpt = self.output_dir / "checkpoints" / trainer_config.best_checkpoint_name
        if not best_ckpt.exists():
            best_ckpt = None  # type: ignore[assignment]

        self._save_history(history)
        self._save_capacity(capacity)
        self._generate_training_plots(history)

        return ExperimentResult(
            history=history,
            capacity=capacity,
            best_checkpoint=best_ckpt,
            output_dir=self.output_dir,
            elapsed_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Artefact helpers
    # ------------------------------------------------------------------

    def _setup_output_dirs(self) -> None:
        for sub in ("checkpoints", "tensorboard", "figures", "metrics", "logs"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        log_path = self.output_dir / "logs" / "experiment.log"
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(handler)

    def _save_history(self, history: list[dict[str, float]]) -> None:
        dest = self.output_dir / "metrics" / "training_history.json"
        with open(dest, "w") as f:
            json.dump(history, f, indent=2)
        logger.info("Training history saved to %s", dest)

    def _save_capacity(self, capacity: CapacityResult | None) -> None:
        if capacity is None:
            return
        dest = self.output_dir / "metrics" / "capacity.json"
        with open(dest, "w") as f:
            json.dump(
                {k: v for k, v in capacity.to_dict().items() if v is not None},
                f,
                indent=2,
            )
        logger.info("Capacity metrics saved to %s", dest)

    def _generate_training_plots(self, history: list[dict[str, float]]) -> None:
        """Generate loss-curve plots from the training history."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            epochs = list(range(1, len(history) + 1))

            metric_pairs = [
                ("train_loss", "val_loss", "Total Loss"),
                ("train_payload", "val_payload", "Payload Recovery Loss"),
                ("train_distortion", "val_distortion", "Distortion Loss"),
                ("train_detector", "val_detector", "Detector Loss"),
            ]

            for ax, (train_key, val_key, title) in zip(axes.flat, metric_pairs):
                train_vals = [e.get(train_key) for e in history]
                val_vals = [e.get(val_key) for e in history]

                if any(v is not None for v in train_vals):
                    ax.plot(
                        epochs,
                        [v or 0 for v in train_vals],
                        label="train",
                        color="#2563EB",
                    )
                if any(v is not None for v in val_vals):
                    ax.plot(
                        epochs,
                        [v or 0 for v in val_vals],
                        label="val",
                        color="#DC2626",
                        linestyle="--",
                    )

                ax.set_title(title, fontsize=11)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend()
                ax.grid(alpha=0.3)

            fig.tight_layout()
            dest = self.output_dir / "figures" / "training_curves.png"
            fig.savefig(dest, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Training curves saved to %s", dest)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not generate training plots: %s", exc)




# ---------------------------------------------------------------------------
# Curriculum scheduler
# ---------------------------------------------------------------------------


class AlphaCurriculumScheduler:
    """Wraps a PyTorch LR scheduler and linearly ramps alpha each epoch.

    The Trainer calls ``scheduler.step()`` (or ``scheduler.step(metrics)``
    for ReduceLROnPlateau) at the end of every epoch.  This wrapper
    intercepts those calls to update ``loss_fn.weights.classification``
    from 0 → ``target_alpha`` linearly over ``warmup_epochs`` epochs,
    then delegates to the real scheduler for the LR update.

    Checkpoint-safe: ``state_dict`` / ``load_state_dict`` persist the
    current epoch counter so alpha is restored correctly on resume.
    """

    def __init__(
        self,
        real_scheduler: Any,
        loss_fn: "CompositeLoss",
        warmup_epochs: int,
        target_alpha: float,
    ) -> None:
        self._sched = real_scheduler
        self._loss_fn = loss_fn
        self._warmup = warmup_epochs
        self._target = target_alpha
        self._epoch: int = 0

    def _update_alpha(self) -> None:
        if self._warmup <= 0:
            return
        alpha = min(self._target, self._target * self._epoch / self._warmup)
        self._loss_fn.set_alpha(alpha)
        if self._epoch % max(1, self._warmup // 10) == 0 or self._epoch == self._warmup:
            logger.info(
                "Curriculum: epoch %d/%d — alpha ramped to %.4f / %.4f",
                self._epoch, self._warmup, alpha, self._target,
            )

    def step(self, metrics: float | None = None) -> None:
        self._epoch += 1
        self._update_alpha()
        if metrics is not None:
            self._sched.step(metrics)
        else:
            self._sched.step()

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self._epoch, "real": self._sched.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._epoch = state["epoch"]
        self._sched.load_state_dict(state["real"])
        # Restore alpha to wherever we were in the curriculum
        self._update_alpha()

    # Proxy everything else the Trainer might call
    def get_last_lr(self) -> list[float]:
        return getattr(self._sched, "get_last_lr", lambda: [])()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sched, name)


def run_experiment(
    config: ExperimentConfig,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any] | None = None,
) -> ExperimentResult:
    """Convenience function: build and run a :class:`SteganographyExperiment`.

    Args:
        config: Full experiment configuration.
        train_loader: Training data loader.
        val_loader: Optional validation data loader.

    Returns:
        :class:`ExperimentResult`.
    """
    experiment = SteganographyExperiment(config)
    return experiment.run(train_loader, val_loader)


# ---------------------------------------------------------------------------
# Batch adapter
# ---------------------------------------------------------------------------


def _pipeline_batch_adapter(
    model: nn.Module,
    batch: Any,
    device: torch.device,
) -> LossInputs:
    """Convert a raw training batch into ``LossInputs`` via the pipeline.

    The ``model`` argument must be an :class:`~models.pipeline.EmbeddingPipeline`
    instance (the trainer wraps it transparently).

    Batch schema: ``(images, labels, payload_bits)`` — the output of
    :class:`~training.dataset.SteganographyDataset`.
    """
    from models.pipeline import EmbeddingPipeline  # local import to avoid circularity

    if not isinstance(model, EmbeddingPipeline):
        raise TypeError(
            f"Expected EmbeddingPipeline, got {type(model).__name__}. "
            "Use a SteganographyExperiment or provide a custom batch adapter."
        )

    if not (isinstance(batch, (tuple, list)) and len(batch) == 3):
        raise ValueError(
            "Expected batch of (images, labels, payload_bits). "
            "Use SteganographyDataset to produce correctly shaped batches."
        )

    images, labels, payload_bits = batch
    images = images.to(device, dtype=torch.float32)
    labels = labels.to(device)
    payload_bits = payload_bits.to(device)

    return model(images, labels, payload_bits)


# ---------------------------------------------------------------------------
# Scheduler helpers
# ---------------------------------------------------------------------------


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
) -> Any:
    sched = cfg.scheduler.lower()
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.scheduler_t_max
        )
    if sched == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.scheduler_step_size,
            gamma=cfg.scheduler_gamma,
        )
    if sched in ("reduce_on_plateau", "plateau"):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.scheduler_gamma, patience=5
        )
    raise ValueError(
        f"Unknown scheduler '{cfg.scheduler}'. "
        "Choose from 'cosine', 'step', or 'reduce_on_plateau'."
    )


def _scheduler_interval(scheduler: str) -> str:
    return "step" if scheduler.lower() == "step" else "epoch"


def _extract_final_ber(history: list[dict[str, float]]) -> float | None:
    """Extract the most recent bit-error-rate from training history.

    Prior to this fix, nothing in the training loop ever logged a
    ``val_ber``/``train_ber``/``bit_error_rate`` key, so this always
    returned ``None`` and capacity reports always showed "N/A" regardless
    of how training actually went. `Trainer` now logs
    ``val_payload_accuracy`` each epoch (see `training/trainer.py`), so
    derive BER from that as `1 - accuracy` when the direct keys aren't
    present.
    """
    for entry in reversed(history):
        for key in ("val_ber", "train_ber", "bit_error_rate"):
            if key in entry:
                return float(entry[key])
        if "val_payload_accuracy" in entry:
            return 1.0 - float(entry["val_payload_accuracy"])
    return None