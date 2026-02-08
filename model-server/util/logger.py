import logging
import colorlog

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("pitchcraft-model")
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            "%(bold_black)s%(asctime)s %(log_color)s%(levelname)-9s%(reset)s%(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "blue",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger

logger = setup_logger()
