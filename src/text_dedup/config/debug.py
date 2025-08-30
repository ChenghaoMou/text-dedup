import logging

from pydantic_settings import BaseSettings


class DebugConfig(BaseSettings):
    logging_level: int = logging.DEBUG
    enable_profiling: bool = False
