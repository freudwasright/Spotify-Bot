import logging
from typing import Union


class TrainingLogger:
    def __init__(self, logger=None):
        self.logger = logger

    @classmethod
    def construct_logger(cls, name: str, log_file_path: str, logger_level: Union[int, str], formatter):
        logger = logging.getLogger(name)
        logger.setLevel(logger_level)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return cls(logger)

    def log_info(self, message: str) -> None:
        self.logger.info(message)
