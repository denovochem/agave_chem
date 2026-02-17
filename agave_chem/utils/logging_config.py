import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

ERROR_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>\n"
    "<red>Exception type: {exception.__class__.__name__}</red>\n"
    "<red>Traceback (most recent call last):\n"
    "{exception}</red>"
)

LOG_LEVELS = {
    "development": "DEBUG",
    "testing": "INFO",
    "production": "WARNING",
    "default": "WARNING",
}


def configure_logging(
    level: str | None = None,
    log_dir: Path | None = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
    serialize: bool = False,
    **kwargs: Any,
) -> None:
    """
    Configure logging for the application with enhanced error reporting.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        rotation: Log rotation configuration (e.g., "10 MB", "1 week")
        retention: Log retention period (e.g., "30 days")
        serialize: Whether to serialize log records as JSON
        **kwargs: Additional arguments to pass to logger.add()
    """
    log_file: Path | None = None
    error_log_file: Path | None = None
    if log_dir:
        log_file = log_dir / "cholla_chem.log"
        error_log_file = log_dir / "cholla_chem_errors.log"

        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.touch()

        error_log_file.parent.mkdir(parents=True, exist_ok=True)
        error_log_file.touch()

    # Determine log level from environment if not specified
    if level is None:
        env = os.getenv("loguru_level", "default").lower()
        level = LOG_LEVELS.get(env, LOG_LEVELS["default"])

    # Configure exception hook for uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.opt(exception=(exc_type, exc_value, exc_traceback)).error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception

    logger.remove()

    logger.add(
        sys.stderr,
        level=level,
        format=DEFAULT_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=True,
        enqueue=False,
        catch=True,
    )

    if log_file:
        logger.add(
            str(log_file),
            rotation=rotation,
            retention=retention,
            level=level,
            format=DEFAULT_FORMAT,
            enqueue=False,
            backtrace=True,
            diagnose=True,
            catch=True,
            **kwargs,
        )

    if error_log_file:
        logger.add(
            str(error_log_file),
            rotation=rotation,
            retention=retention,
            level="WARNING",
            format=ERROR_FORMAT,
            enqueue=False,
            backtrace=True,
            diagnose=True,
            catch=True,
            **kwargs,
        )

    # Add structured logging if enabled
    if serialize and sys.__stdout__ is not None:
        logger.add(
            sys.__stdout__,
            level=level,
            format=DEFAULT_FORMAT,
            serialize=True,
            enqueue=False,
            backtrace=True,
            diagnose=True,
            **kwargs,
        )


def disable_library_logging() -> None:
    logger.disable("cholla_chem")


def enable_library_logging() -> None:
    logger.enable("cholla_chem")


__all__ = ["logger", "configure_logging"]
