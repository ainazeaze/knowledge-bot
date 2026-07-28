import logging
import logging.handlers
import sys
from pathlib import Path

_LOG_FILE = Path(__file__).parent.parent / "logs" / "retrieval.log"


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("brainbot")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_FILE, maxBytes=1_000_000, backupCount=3
    )
    file_handler.setFormatter(fmt)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stderr_handler)
    return logger


logger = _get_logger()
