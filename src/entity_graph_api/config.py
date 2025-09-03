import logging
import logging.config
import sys
import warnings
from enum import StrEnum
from typing import Annotated

from loguru import logger as logger_loguru
from pydantic import AfterValidator, AnyUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console

from entity_graph_api.config_logger import config_logger

console = Console()
logger_loguru.level("INFO", color="<blue>")
logger_loguru.level("DEBUG", color="<cyan>")

def setup_logger(name: str) -> logging.Logger:

    logging.config.dictConfig(config_logger)

    warnings.filterwarnings("ignore")

    logger_loguru.configure(
        handlers=[
            {
                "sink": sys.stdout,
                "level": CONFIG.LOGGING_LEVEL,
            }
        ]
    )

    return logger_loguru.bind(name=name)


URL = Annotated[AnyUrl, AfterValidator(str)]

UPLOAD_DIR = "/app/uploads"

# Custom StrEnum is more handy than using integers that represent logging levels
class LoggingLevels(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env"), env_nested_delimiter="__", extra="ignore"
    )

    API_BASE_URL: URL | None = None

    LOGGING_LEVEL: LoggingLevels = LoggingLevels.DEBUG

    API_KEY: str | None = None


CONFIG = Settings()
