"""Configurable training framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from models.decoder import payload_reconstruction_accuracy
from training.losses import LossInputs, LossOutput


MonitorMode = Literal["min", "max"]
SchedulerInterval = Literal["epoch", "step"]
BatchToLossInputs = Callable[[nn.Module, Any, torch.device], LossInputs]


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for `Trainer`.

    Attributes:
        max_epochs: Number of full training epochs.
        device: Device string such as `cpu`, `cuda`, or `auto`.
        mixed_precision: Whether to enable automatic mixed precision on CUDA.
        gradient_clip_norm: Optional max norm for gradient clipping.
        checkpoint_dir: Directory for checkpoint files.
        log_dir: Directory for TensorBoard event files.
        checkpoint_name: File name used for the latest checkpoint.
        best_checkpoint_name: File name used for the best monitored checkpoint.
        save_best_only: If true, only writes the best checkpoint.
        monitor: Metric key used for best-checkpoint and early-stopping logic.
        monitor_mode: Whether lower or higher monitor values are better.
        early_stopping_patience: Optional number of non-improving validation
            epochs before stopping.
        log_every_n_steps: Training scalar logging interval in optimizer steps.
        scheduler_interval: Whether the scheduler steps after each epoch or
            optimizer step.
    """

    max_epochs: int
    device: str = "auto"
    mixed_precision: bool = False
    gradient_clip_norm: float | None = None
    checkpoint_dir: str | Path = "outputs/checkpoints"
    log_dir: str | Path = "outputs/tensorboard"
    checkpoint_name: str = "latest.pt"
    best_checkpoint_name: str = "best.pt"
    save_best_only: bool = False
    monitor: str = "val_loss"
    monitor_mode: MonitorMode = "min"
    early_stopping_patience: int | None = None
    log_every_n_steps: int = 1
    scheduler_interval: SchedulerInterval = "epoch"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainerState:
    """Serializable trainer state."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float | None = None
    epochs_without_improvement: int = 0
    stopped_early: bool = False


class Trainer:
    """Configurable PyTorch trainer with validation and checkpointing."""

    def __init__(
        self,
        *,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: Iterable[Any],
        config: TrainerConfig,
        val_loader: Iterable[Any] | None = None,
        scheduler: Any | None = None,
        batch_to_loss_inputs: BatchToLossInputs | None = None,
        writer: SummaryWriter | None = None,
    ) -> None:
        """Initialize the training framework.

        Args:
            model: Model or pipeline module being optimized.
            loss_fn: Loss module returning either `LossOutput` or a scalar.
            optimizer: PyTorch optimizer.
            train_loader: Iterable training batches.
            config: Fully configurable trainer settings.
            val_loader: Optional iterable validation batches.
            scheduler: Optional learning-rate scheduler.
            batch_to_loss_inputs: Adapter from raw batches to `LossInputs`.
            writer: Optional injected TensorBoard writer for tests.
        """

        _validate_config(config)
        self.config = config
        self.device = _resolve_device(config.device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.batch_to_loss_inputs = batch_to_loss_inputs or default_batch_to_loss_inputs
        self.checkpoint_dir = Path(config.checkpoint_dir).expanduser().resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = writer or SummaryWriter(log_dir=str(Path(config.log_dir).expanduser()))
        self.state = TrainerState()
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=config.mixed_precision and self.device.type == "cuda",
        )

    def train(self) -> list[dict[str, float]]:
        """Run training and optional validation.

        Returns:
            Per-epoch history dictionaries with training, validation, and
            learning-rate values.
        """

        history: list[dict[str, float]] = []
        for epoch in range(self.state.epoch, self.config.max_epochs):
            train_metrics = self._run_training_epoch(epoch)
            val_metrics = self.validate(epoch) if self.val_loader is not None else {}
            metrics = {**train_metrics, **val_metrics, "lr": self._current_lr()}
            history.append(metrics)

            if self.config.scheduler_interval == "epoch" and self.scheduler is not None:
                self._step_scheduler(metrics)

            self._update_checkpointing(epoch, metrics)
            self._log_epoch(epoch, metrics)
            self._print_epoch_summary(epoch, metrics)

            if self.state.stopped_early:
                break

        self.writer.flush()
        return history

    @torch.no_grad()
    def validate(self, epoch: int | None = None) -> dict[str, float]:
        """Run the validation loop.

        Args:
            epoch: Optional epoch index for TensorBoard logging.

        Returns:
            Dictionary containing `val_loss` and detached component means.
        """

        if self.val_loader is None:
            return {}

        self.model.eval()
        totals: dict[str, float] = {}
        count = 0

        for batch in self.val_loader:
            loss_inputs = self.batch_to_loss_inputs(self.model, batch, self.device)
            loss_output = self._compute_loss_from_inputs(loss_inputs)
            self._accumulate(totals, "val_loss", loss_output.total.detach())
            for name, value in loss_output.components.items():
                self._accumulate(totals, f"val_{name}", value)
            if loss_inputs.payload_logits is not None and loss_inputs.payload_targets is not None:
                predicted_bits = (torch.sigmoid(loss_inputs.payload_logits) > 0.5).float()
                accuracy = payload_reconstruction_accuracy(
                    predicted_bits, loss_inputs.payload_targets
                )
                self._accumulate(totals, "val_payload_accuracy", torch.tensor(accuracy))
            if loss_inputs.classification_logits is not None and loss_inputs.classification_targets is not None:
                preds = loss_inputs.classification_logits.argmax(dim=-1)
                host_acc = (preds == loss_inputs.classification_targets).float().mean()
                self._accumulate(totals, "val_host_accuracy", host_acc)
            count += 1

        metrics = {key: value / count for key, value in totals.items()} if count else {}
        if epoch is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, epoch)
        return metrics

    def save_checkpoint(self, path: str | Path, metrics: Mapping[str, float] | None = None) -> Path:
        """Save model, optimizer, scheduler, and trainer state."""

        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict(),
            "trainer_state": asdict(self.state),
            "config": asdict(self.config),
            "metrics": dict(metrics or {}),
        }
        torch.save(checkpoint, destination)
        return destination

    def load_checkpoint(self, path: str | Path) -> Mapping[str, Any]:
        """Load a checkpoint into the trainer."""

        checkpoint = torch.load(
            Path(path).expanduser().resolve(),
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])

        state = checkpoint.get("trainer_state", {})
        self.state = TrainerState(**state)
        return checkpoint

    def close(self) -> None:
        """Close TensorBoard resources."""

        self.writer.close()

    def _run_training_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        count = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.config.mixed_precision and self.device.type == "cuda",
            ):
                loss_output = self._compute_loss(batch)

            self.scaler.scale(loss_output.total).backward()

            if self.config.gradient_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.gradient_clip_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.state = TrainerState(
                epoch=epoch,
                global_step=self.state.global_step + 1,
                best_metric=self.state.best_metric,
                epochs_without_improvement=self.state.epochs_without_improvement,
                stopped_early=self.state.stopped_early,
            )

            if self.config.scheduler_interval == "step" and self.scheduler is not None:
                self._step_scheduler({})

            self._accumulate(totals, "train_loss", loss_output.total.detach())
            for name, value in loss_output.components.items():
                self._accumulate(totals, f"train_{name}", value)
            count += 1

            if self.state.global_step % self.config.log_every_n_steps == 0:
                self.writer.add_scalar(
                    "train/loss_step",
                    float(loss_output.total.detach().cpu()),
                    self.state.global_step,
                )

        return {key: value / count for key, value in totals.items()} if count else {}

    def _compute_loss(self, batch: Any) -> LossOutput:
        loss_inputs = self.batch_to_loss_inputs(self.model, batch, self.device)
        return self._compute_loss_from_inputs(loss_inputs)

    def _compute_loss_from_inputs(self, loss_inputs: LossInputs) -> LossOutput:
        output = self.loss_fn(loss_inputs)
        if isinstance(output, LossOutput):
            return output
        if isinstance(output, torch.Tensor):
            return LossOutput(total=output, components={"loss": output.detach()})
        raise TypeError("loss_fn must return LossOutput or torch.Tensor.")

    def _update_checkpointing(self, epoch: int, metrics: Mapping[str, float]) -> None:
        monitor_value = metrics.get(self.config.monitor)
        improved = monitor_value is not None and self._is_improved(monitor_value)

        if improved:
            self.state = TrainerState(
                epoch=epoch + 1,
                global_step=self.state.global_step,
                best_metric=monitor_value,
                epochs_without_improvement=0,
                stopped_early=False,
            )
            self.save_checkpoint(self.checkpoint_dir / self.config.best_checkpoint_name, metrics)
        else:
            self.state = TrainerState(
                epoch=epoch + 1,
                global_step=self.state.global_step,
                best_metric=self.state.best_metric,
                epochs_without_improvement=self.state.epochs_without_improvement + 1,
                stopped_early=self.state.stopped_early,
            )

        if not self.config.save_best_only:
            self.save_checkpoint(self.checkpoint_dir / self.config.checkpoint_name, metrics)

        patience = self.config.early_stopping_patience
        if patience is not None and self.state.epochs_without_improvement >= patience:
            self.state = TrainerState(
                epoch=self.state.epoch,
                global_step=self.state.global_step,
                best_metric=self.state.best_metric,
                epochs_without_improvement=self.state.epochs_without_improvement,
                stopped_early=True,
            )

    def _current_lr(self) -> float:
        """Return the current learning rate from the first param group."""
        return self.optimizer.param_groups[0]["lr"]

    def _is_improved(self, value: float) -> bool:
        if self.state.best_metric is None:
            return True
        if self.config.monitor_mode == "min":
            return value < self.state.best_metric
        return value > self.state.best_metric

    def _step_scheduler(self, metrics: Mapping[str, float]) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            metric = metrics.get(self.config.monitor)
            if metric is None:
                raise ValueError(
                    f"Scheduler requires monitored metric '{self.config.monitor}'."
                )
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def _log_epoch(self, epoch: int, metrics: Mapping[str, float]) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, epoch)

    @staticmethod
    def _print_epoch_summary(epoch: int, metrics: Mapping[str, float]) -> None:
        """Print a concise, one-line-per-epoch summary to console.

        Only fires once per epoch (not per step) so it's usable on
        platforms like Kaggle where per-step output is easy to lose track
        of in a long-running cell.
        """

        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")
        val_payload_accuracy = metrics.get("val_payload_accuracy")
        # Raw (unweighted) payload BCE loss is a far more sensitive signal
        # than thresholded accuracy for large payloads: with e.g. ~53M total
        # bit predictions in a 128KB-payload validation set, small real
        # improvements can be invisible in accuracy at 2-decimal precision
        # for many epochs, while the continuous BCE loss moves immediately.
        val_payload_loss = metrics.get("val_payload")
        train_payload_loss = metrics.get("train_payload")
        val_host_accuracy = metrics.get("val_host_accuracy")

        parts = [f"epoch {epoch + 1}"]
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.4f}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        if train_payload_loss is not None:
            parts.append(f"train_payload_bce={train_payload_loss:.6f}")
        if val_payload_loss is not None:
            parts.append(f"val_payload_bce={val_payload_loss:.6f}")
        if val_payload_accuracy is not None:
            parts.append(f"val_payload_recovery_accuracy={val_payload_accuracy * 100:.4f}%")
        if val_host_accuracy is not None:
            parts.append(f"val_host_accuracy={val_host_accuracy * 100:.4f}%")
        print(" | ".join(parts), flush=True)

    @staticmethod
    def _accumulate(totals: dict[str, float], key: str, value: torch.Tensor) -> None:
        totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())


def default_batch_to_loss_inputs(
    model: nn.Module,
    batch: Any,
    device: torch.device,
) -> LossInputs:
    """Convert a batch mapping into `LossInputs`.

    This default adapter intentionally does not assume a specific encoder,
    decoder, detector, or host-model wiring. Project training scripts can inject
    a custom adapter once the full pipeline is assembled.
    """

    del model
    if isinstance(batch, LossInputs):
        return _move_loss_inputs(batch, device)
    if not isinstance(batch, Mapping):
        raise TypeError(
            "Default batch adapter expects LossInputs or a mapping. "
            "Provide batch_to_loss_inputs for custom batches."
        )

    allowed = LossInputs.__dataclass_fields__.keys()
    kwargs = {
        key: _move_to_device(value, device)
        for key, value in batch.items()
        if key in allowed
    }
    return LossInputs(**kwargs)


def _move_loss_inputs(inputs: LossInputs, device: torch.device) -> LossInputs:
    kwargs = {
        field_name: _move_to_device(getattr(inputs, field_name), device)
        for field_name in LossInputs.__dataclass_fields__
    }
    return LossInputs(**kwargs)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _resolve_device(device: str) -> torch.device:
    choice = device.lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def _validate_config(config: TrainerConfig) -> None:
    if config.max_epochs <= 0:
        raise ValueError("max_epochs must be positive.")
    if config.gradient_clip_norm is not None and config.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive when set.")
    if config.monitor_mode not in {"min", "max"}:
        raise ValueError("monitor_mode must be 'min' or 'max'.")
    if (
        config.early_stopping_patience is not None
        and config.early_stopping_patience <= 0
    ):
        raise ValueError("early_stopping_patience must be positive when set.")
    if config.log_every_n_steps <= 0:
        raise ValueError("log_every_n_steps must be positive.")
    if config.scheduler_interval not in {"epoch", "step"}:
        raise ValueError("scheduler_interval must be 'epoch' or 'step'.")
