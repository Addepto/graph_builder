import functools
import logging
import logging.config
import sys
import warnings

from loguru import logger as logger_loguru
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console

console = Console()
# logger_loguru.remove(0)
# Format for REQUEST logs
logger_loguru.add(
    sys.stdout,
    level=1,  # we are filtering anyway
    colorize=True,
    filter=lambda x: x["level"].name in ("REQUEST"),
    format="<level>{message}</level> ({process.name}::{thread.name})",
    # enqueue=True, # Enable in case of problems with multi-process support
)
# Format for CCTRACE logs
logger_loguru.add(
    lambda x: console.print(
        x, end="", no_wrap=True, soft_wrap=True
    ),  # prevent extra new line from print and log
    level=1,  # we are filtering anyway
    colorize=False,
    filter=lambda x: x["level"].name in ("CCTRACE"),
    format="{message} [italic grey]({process.name}::{thread.name})[/]",
    # enqueue=True, # Enable in case of problems with multi-process support
)
logger_loguru.level("INFO", color="<blue>")
logger_loguru.level("DEBUG", color="<cyan>")
# level no shouuld be in between TRACE (5) and DEBUG (10)
logger_loguru.level("CCTRACE", no=9)
logger_loguru.level("REQUEST", no=8, color="<white>")


def setup_logger(name: str) -> logging.Logger:

    def logger_wraps():

        def wrapper(func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                logger_ = logger_loguru.opt(depth=1, ansi=False)
                logger_.log(
                    "CCTRACE",
                    f"[bold italic blue on green]Entering[/] [bold yellow]' {func.__module__}.{func.__name__}'[/]",
                )
                logger_.log("CCTRACE", f"{args=}, {kwargs=}")
                result = func(*args, **kwargs)
                logger_.log(
                    "CCTRACE",
                    f"[bold italic blue on red]Exiting[/]  [bold yellow]' {func.__module__}.{func.__name__}'[/]",
                )
                logger_.log("CCTRACE", f"{result=}")

                return result

            return wrapped

        return wrapper

    warnings.filterwarnings("ignore")

    logger_loguru.wrap = logger_wraps

    return logger_loguru


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env.common", ".env"), env_nested_delimiter="__", extra="ignore"
    )


CONFIG = Settings()
