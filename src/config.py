from pathlib import Path
import logging
from logging import Logger

try:
    from rich.logging import RichHandler
except ImportError:  # pragma: no cover - optional dependency
    RichHandler = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def log(name: str) -> Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    if RichHandler is not None:
        handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s [%(levelname)s] - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
