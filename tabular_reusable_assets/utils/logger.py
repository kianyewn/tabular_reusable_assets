import sys
from pathlib import Path
from loguru import logger

def setup_logger(output_dir: Path = None, log_level: str = "INFO"):
    """Configure loguru logger"""
    
    # Remove default logger
    logger.remove()
    
    # Add stdout handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file handler if output_dir is provided
    if output_dir:
        log_file = Path(output_dir) / "train.log"
        logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="100 MB",
            retention="10 days"
        )

    return logger

    
# Create a default logger instance
default_logger = setup_logger()