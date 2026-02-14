"""
日志模块 - 为 LyricSync 提供统一的日志管理。

使用 Python logging 模块实现分级日志，同时输出到控制台和文件。
日志文件按日期滚动存储，便于追踪转录、分离、格式化等各环节的执行状态。
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# 日志目录：项目根目录下的 logs 文件夹
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

# 日志格式：时间 | 等级 | 模块名 | 消息
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    创建并返回一个命名的 Logger 实例。

    每个模块调用此函数获取自己的 logger，保证日志输出格式统一。
    日志同时写入控制台（INFO 级别以上）和文件（DEBUG 级别以上）。

    Args:
        name: 日志记录器名称，一般传入模块名（如 'transcriber'、'separator'）
        level: 日志记录的最低级别，默认 DEBUG

    Returns:
        logging.Logger: 配置好的日志记录器实例
    """
    logger = logging.getLogger(f"lyric_sync.{name}")

    # 避免重复添加 handler（模块被多次导入时）
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # --- 控制台 Handler：INFO 级别以上 ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    # --- 文件 Handler：DEBUG 级别以上，单文件最大 10MB，保留 5 个备份 ---
    file_path = os.path.join(_LOG_DIR, "lyric_sync.log")
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
