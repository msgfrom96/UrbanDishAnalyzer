"""Logging configuration module."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Configuration object with proper type annotation
config: Dict[str, Any] = {}


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    file_level: Optional[int] = None,
) -> None:
    """Set up logging configuration.

    Args:
        level: Logging level to use
        log_file: Optional path to log file
        file_level: Optional logging level for the file handler
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if log_file:
        add_file_handler(log_file, level=file_level or level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_level(level: int) -> None:
    """Set logging level.

    Args:
        level: Logging level to use
    """
    logging.getLogger().setLevel(level)


def add_file_handler(
    log_file: str, level: int = logging.INFO, format_str: Optional[str] = None
) -> None:
    """Add a file handler to the root logger.

    Args:
        log_file: Path to log file
        level: Logging level for the handler
        format_str: Optional format string for the handler
    """
    # Create directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Create handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(level)

    # Set formatter
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handler.setFormatter(logging.Formatter(format_str))

    # Add handler to root logger
    logging.getLogger().addHandler(handler)


class LogManager:
    """Manager for contextual logging."""

    def __init__(self, name: str) -> None:
        """Initialize the log manager.

        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(f"uda.{name}")

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message.

        Args:
            msg: Message to log
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments
        """
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log info message.

        Args:
            msg: Message to log
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments
        """
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message.

        Args:
            msg: Message to log
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments
        """
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log error message.

        Args:
            msg: Message to log
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments
        """
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log exception with traceback.

        Args:
            msg: Message to log
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments
        """
        self.logger.exception(msg, *args, **kwargs)

    def profile(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log profile-related message.

        Args:
            msg: Message to log
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments
        """
        self.logger.info(f"[PROFILE] {msg}", *args, **kwargs)

    def analysis(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log analysis-related message.

        Args:
            msg: Message to log
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments
        """
        self.logger.info(f"[ANALYSIS] {msg}", *args, **kwargs)

    def geo(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log geography-related message.

        Args:
            msg: Message to log
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments
        """
        self.logger.info(f"[GEO] {msg}", *args, **kwargs)

    def data(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log data-related message.

        Args:
            msg: Message to log
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments
        """
        self.logger.info(f"[DATA] {msg}", *args, **kwargs)


# Set up default logging configuration
setup_logging(
    level=config.get("logging.level", logging.INFO),
    log_file=config.get("logging.file", None),
    file_level=config.get("logging.file_level", None),
)
