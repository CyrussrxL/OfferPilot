import logging
import os
from datetime import datetime


def setup_logger(name: str = "OfferPilot", level: str = "INFO") -> logging.Logger:
    """
    配置并返回一个统一的 Logger 实例。
    同时输出到控制台和文件，文件按日期命名。
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = os.getenv("LOG_DIR", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"companion_ai_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()
