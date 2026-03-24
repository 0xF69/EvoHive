"""Structured logging for EvoHive engine.

Provides file-based DEBUG logging (rotating, 10 MB) and console WARNING+ logging
so that Rich user-facing output is not disrupted.
"""

import json
import logging
import logging.handlers
from typing import Any


_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s | %(message)s"
_LOG_FILE = "evohive_runs.log"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 3

# Keep track of whether root handlers are already wired up.
_initialized = False


def _ensure_handlers() -> None:
    """Attach console + rotating-file handlers to the 'evohive' root logger once."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger("evohive")
    root.setLevel(logging.DEBUG)
    # Don't propagate to the root logger (avoids duplicate output / LiteLLM noise).
    root.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT)

    # ── File handler: captures everything (DEBUG+) ──
    fh = logging.handlers.RotatingFileHandler(
        _LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # ── Console handler: WARNING+ only (don't clutter Rich output) ──
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)
    root.addHandler(ch)


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``evohive.*`` hierarchy.

    Usage::

        logger = get_logger("evohive.engine.evolution")
        logger.info("Generation %d complete", gen)

    The first call also wires up the shared handlers (file + console).
    """
    _ensure_handlers()
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event_type: str, **kwargs: Any) -> None:
    """Log a structured JSON event at INFO level.

    Produces a single-line JSON payload so the log file is easy to parse
    programmatically::

        log_event(logger, "generation_end", generation=3, best_fitness=8.7)
        # => [2026-03-23 ...] INFO evohive.engine.evolution | {"event": "generation_end", "generation": 3, "best_fitness": 8.7}
    """
    payload = {"event": event_type, **kwargs}
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))
