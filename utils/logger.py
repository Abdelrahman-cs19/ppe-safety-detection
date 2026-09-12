"""
Shared logging configuration.

Every module gets a logger via `get_logger(__name__)` so log lines are
attributable to their source (detection, tracking, rules, etc.) and
consistently formatted, both on the console and in a rotating log file.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.config import LOGS_DIR, settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root() -> None:
    """Attach console + rotating file handlers to the root logger exactly once."""
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(settings.log_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    log_file = Path(LOGS_DIR) / "ppe_safety.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring handlers on first use."""
    _configure_root()
    return logging.getLogger(name)
