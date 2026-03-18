"""
Centralized logging configuration for the RAG agent.
"""

import logging
import os
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Configure and return a logger instance with both file and console handlers.
    
    Args:
        name: Logger name (typically __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Create logs directory if it doesn't exist
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler
    log_file = log_dir / f"rag_agent_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger (avoid duplicate logs)
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger
