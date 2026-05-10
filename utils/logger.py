import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

from utils.path_util import LOGS_DIR


# 日志保存的根目录并确保其存在
os.makedirs(LOGS_DIR,exist_ok=True)

# 日志格式配置
DEFAULT_LOG_FORMAT = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')


def get_logger(
        name:str = "RagService",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file = None,
) ->logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(console_handler)

    # 文件handler
    if not log_file:
        log_file = os.path.join(LOGS_DIR,f"{name}_{datetime.now().strftime('%y%m%d')}.log")

    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        backupCount=365,
        encoding="utf-8"
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handler)

    return logger


logger = get_logger()


