import os
from pathlib import Path
import sys
from typing import Any, Dict

from loguru import logger

# Default log format with more detailed traceback
DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# More detailed format for error logs
ERROR_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>\n"
    "<red>Exception type: {exception.__class__.__name__}</red>\n"
    "<red>Traceback (most recent call last):\n"
    "{exception}</red>"
)

# Log levels for different environments
LOG_LEVELS = {
    "development": "DEBUG",
    "testing": "INFO",
    "production": "WARNING",
    "default": "production",
}

# Default log directory - using absolute path to project root
PROJECT_ROOT = Path(__file__).parent.parent  # Go up three dirs
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists

# Default log files
LOG_FILE = LOG_DIR / "info.log"
ERROR_LOG_FILE = LOG_DIR / "errors.log"

LOG_FILE.touch()
ERROR_LOG_FILE.touch()


def configure_logging(
    level: str = None,
    log_file: Path = LOG_FILE,
    error_log_file: Path = ERROR_LOG_FILE,
    rotation: str = "10 MB",
    retention: str = "30 days",
    serialize: bool = False,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Configure logging for the application with enhanced error reporting.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to the main log file
        error_log_file: Path to the error log file
        rotation: Log rotation configuration (e.g., "10 MB", "1 week")
        retention: Log retention period (e.g., "30 days")
        serialize: Whether to serialize log records as JSON
        **kwargs: Additional arguments to pass to logger.add()
    """
    # Ensure log directory exists
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.touch()
    if error_log_file:
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

    # Remove default handler
    logger.remove()

    # Add console handler with enhanced error formatting
    logger.add(
        sys.stderr,
        level=level,
        format=DEFAULT_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=True,  # Enable variable values in traceback
        enqueue=True,  # Make logging thread-safe
        catch=True,  # Catch and log errors in logging system itself
    )

    # Add file handler for all logs
    if log_file:
        logger.add(
            str(log_file),
            rotation=rotation,
            retention=retention,
            level=level,
            format=DEFAULT_FORMAT,
            enqueue=True,
            backtrace=True,
            diagnose=True,
            catch=True,
            **kwargs,
        )

    # Add file handler for error logs only with enhanced formatting
    if error_log_file:
        logger.add(
            str(error_log_file),
            rotation=rotation,
            retention=retention,
            level="WARNING",
            format=ERROR_FORMAT,  # Use the more detailed format for errors
            enqueue=True,
            backtrace=True,
            diagnose=True,
            catch=True,
            **kwargs,
        )

    # Add structured logging if enabled
    if serialize:
        logger.add(
            sys.stdout,
            level=level,
            format=DEFAULT_FORMAT,
            serialize=True,
            enqueue=True,
            backtrace=True,
            diagnose=True,
            **kwargs,
        )


# Export logger for easy access
__all__ = ["logger", "configure_logging"]
