"""
Centralized logging configuration for Temporal-QBER pipeline.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes through tqdm.write to avoid interleaving."""
    
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(
    log_level: int = logging.INFO,
    log_dir: Optional[str] = None,
    experiment_id: Optional[str] = None,
    disable_console: bool = False,
) -> logging.Logger:
    """
    Set up centralized logging with console and file handlers.
    
    Args:
        log_level: Logging level (default: INFO)
        log_dir: Directory for log files (default: results/logs/)
        experiment_id: Unique experiment identifier for log filename
        disable_console: If True, disable console logging
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("temporal_qber")
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler - use tqdm.write to avoid interleaving with progress bars
    if not disable_console:
        console_handler = TqdmLoggingHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir is None:
        log_dir = "results/logs"
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    if experiment_id is None:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = log_path / f"experiment_{experiment_id}.log"
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # File gets DEBUG and above
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger() -> logging.Logger:
    """Get the temporal_qber logger instance."""
    return logging.getLogger("temporal_qber")


def log_experiment_config(logger: logging.Logger, config: dict):
    """Log complete experiment configuration."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 60)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)


def log_exception(logger: logging.Logger, msg: str, exc: Exception):
    """Log exception with traceback."""
    logger.exception(f"{msg}: {exc}")