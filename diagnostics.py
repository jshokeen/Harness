"""Console and file diagnostics. Never pass credentials to this logger."""
import logging
from pathlib import Path


def get_logger():
    logger = logging.getLogger("harness")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (
            logging.StreamHandler(),
            logging.FileHandler(Path(__file__).with_name("harness.log"), encoding="utf-8"),
        ):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.propagate = False
    return logger
