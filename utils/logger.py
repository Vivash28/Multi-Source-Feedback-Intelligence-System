"""
utils/logger.py
===============
Centralised logging configuration for the entire system.

Usage
-----
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing started")
"""

import logging
import logging.handlers
import sys
import os


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    The logger writes to both stdout and a rotating file.
    Log level is controlled via the LOG_LEVEL environment variable
    (default: INFO).

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_level_str: str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level: int = getattr(logging, log_level_str, logging.INFO)
    log_file: str = os.getenv(
        "LOG_FILE",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "feedback_system.log",
        ),
    )

    logger = logging.getLogger(name)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("Could not create file log handler: %s", exc)

    logger.propagate = False
    return logger