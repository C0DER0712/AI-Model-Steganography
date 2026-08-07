"""Logging configuration helpers."""

from __future__ import annotations

import logging
from pathlib import Path


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure application logging.

    Args:
        level: Logging verbosity accepted by `logging`.
        log_file: Optional file path for persistent logs.

    Returns:
        The configured root logger.
    """

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        path = Path(log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )

    return logging.getLogger()
