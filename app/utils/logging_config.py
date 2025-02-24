"""
Logging configuration for CyberImage
"""
import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

def setup_logging(app):
    """Configure application-wide logging"""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create formatters
    verbose_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # Create handlers
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # File handlers
    today = datetime.now().strftime("%Y-%m-%d")

    # Application log
    app_handler = logging.handlers.RotatingFileHandler(
        log_dir / f"app-{today}.log",
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(verbose_formatter)

    # Error log
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / f"error-{today}.log",
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(verbose_formatter)

    # Model log
    model_handler = logging.handlers.RotatingFileHandler(
        log_dir / f"model-{today}.log",
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    model_handler.setLevel(logging.DEBUG)
    model_handler.setFormatter(verbose_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(app_handler)
    root_logger.addHandler(error_handler)

    # Configure specific loggers
    model_logger = logging.getLogger("app.models")
    model_logger.addHandler(model_handler)

    # Disable propagation for model logger to avoid duplicate logs
    model_logger.propagate = False

    # Log startup
    app.logger.info("Logging system initialized")