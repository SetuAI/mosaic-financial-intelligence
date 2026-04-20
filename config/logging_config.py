"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
config/logging_config.py

Centralised logging setup for the entire system.

Why not just use print() or basicConfig():
    - basicConfig() called in multiple modules produces duplicate handlers
      and messy interleaved output — a known Python logging gotcha.
    - print() gives you nothing in production — no timestamps, no severity,
      no way to filter by component.
    - This setup gives every component its own named logger that inherits
      from the root, so you can silence a noisy module without touching others.

Usage in any module:
    from config.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Fetching filings for AMZN")
"""

import logging
import sys
from typing import Optional


# The log format that actually tells you what you need to know:
# when it happened, how bad it is, which module fired it, and what it says.
# %(name)s is the logger name — set to __name__ in each module, so you
# always know exactly where a log line came from.
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """
    Configures the root logger once at application startup.

    Call this exactly once — in run_ingestion.py or your main entry point.
    Every subsequent get_logger() call inherits this config automatically.

    Args:
        level: Logging level as a string — DEBUG, INFO, WARNING, ERROR.
               INFO is the right default for production runs.
               Switch to DEBUG when chasing down a parsing bug.
    """
    # Convert string level to the logging module's integer constant.
    # logging.getLevelName("INFO") returns 20 — the int logging needs.
    numeric_level = logging.getLevelName(level.upper())

    # Guard against someone passing a typo like "INFOO" — getLevelName
    # returns the string back if it cannot find the level, not an int.
    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Invalid log level: '{level}'. "
            f"Valid options are: DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )

    # StreamHandler writes to stdout — plays nicely with CloudWatch when
    # we deploy to Lambda, which captures stdout automatically.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))

    # Configure the root logger — all named loggers in child modules
    # bubble up to this one unless explicitly told not to.
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear any existing handlers before adding ours — prevents duplicate
    # log lines if setup_logging() is accidentally called more than once.
    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.addHandler(handler)

    # Silence noisy third-party loggers that we do not care about.
    # httpx logs every single request at INFO level — useful for debugging
    # but overwhelming in normal operation.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Returns a named logger for the calling module.

    The name should always be __name__ — this gives you the full module
    path in log output (e.g. ingestion.edgar_client) which makes it
    immediately obvious where a log line came from.

    Args:
        name: Logger name — pass __name__ from the calling module.

    Returns:
        A configured Logger instance.
    """
    return logging.getLogger(name or "mosaic")