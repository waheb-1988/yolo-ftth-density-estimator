"""Centralised logging setup for FALCON."""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_to_file: bool = True,
                  log_dir: Path | None = None) -> None:
    """
    Configure root logger once at application startup.
    Safe to call multiple times — only attaches handlers on the first call.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    level_num = getattr(logging, level.upper(), logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)-30s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root.setLevel(level_num)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level_num)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file
    if log_to_file:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "falcon.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(level_num)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Quiet noisy third-party libraries
    for lib in ("ultralytics", "PIL", "urllib3", "matplotlib",
                "httpx", "uvicorn.access"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Call after setup_logging()."""
    return logging.getLogger(name)
