"""
RiskLens AI — Logging Configuration
=====================================
Structured JSON logging using Loguru.

Why structured logging:
    1. JSON logs are machine-parseable (ELK stack, CloudWatch)
    2. Each log entry has timestamp, level, module, message
    3. In production, you grep logs by field, not by regex

Why Loguru over stdlib logging:
    1. Zero-config — works out of the box
    2. Colored output in development
    3. JSON serialization for production
    4. Automatic exception formatting
"""

import sys
import logging
from pathlib import Path

try:
    from loguru import logger as _loguru_logger
    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """Configure logging for the application."""
    if HAS_LOGURU:
        _loguru_logger.remove()
        _loguru_logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan> — <level>{message}</level>",
            colorize=True,
        )
        if log_file:
            _loguru_logger.add(
                log_file,
                level=log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} — {message}",
                rotation="10 MB",
                retention="7 days",
            )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )


def get_logger(name: str):
    """
    Get a logger instance.

    Returns Loguru logger if available, otherwise stdlib logger.
    Both support: logger.info(), logger.warning(), logger.error()
    """
    if HAS_LOGURU:
        return _loguru_logger.bind(name=name)
    else:
        return logging.getLogger(name)
