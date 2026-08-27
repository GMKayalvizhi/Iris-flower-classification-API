import logging
import sys
from logging.handlers import RotatingFileHandler
import os

# Make sure the logs folder exists before we try to write to it
os.makedirs("logs", exist_ok=True)

def setup_logging():
    logger = logging.getLogger("iris_api")
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if this gets called more than once
    # (e.g. during --reload restarts)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — prints to your terminal, same as print() did
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler — writes to a file, but caps its size
    # maxBytes=1_000_000 means ~1MB per file before it rotates
    # backupCount=3 means it keeps 3 old versions before deleting the oldest
    file_handler = RotatingFileHandler(
        "logs/app.log", maxBytes=1_000_000, backupCount=3
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()