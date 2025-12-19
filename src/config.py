from pathlib import Path
import logging
from logging import Logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def log(name: str) -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()

    fmt = "%(asctime)s [%(levelname)s] - %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
