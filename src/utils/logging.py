from loguru import logger
import sys


def configure_logging() -> None:
    """Configure Loguru logger with a clean format to stdout."""
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )


__all__ = ["configure_logging", "logger"]


