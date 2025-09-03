import importlib.util

config_logger = {
    "version": 1,
    "disable_existing_loggers": False,
    "root": {"handlers": ["console"], "level": "DEBUG"},
    "handlers": {
        "console": {
            "formatter": "std_out",
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
        "console_uvi": (
            {
                "formatter": "uvi",
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "stream": "ext://sys.stdout",
            }
        ),
    },
    "formatters": {
        "std_out": {
            "format": "%(name)s : %(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(lineno)d : %(message)s",
            "datefmt": "%d-%m-%Y %I:%M:%S",
        },
        "uvi": (
            {
                "()": ("uvicorn.logging.DefaultFormatter"),
                "fmt": "%(levelprefix)s %(asctime)s - %(message)s",
                "use_colors": True,
            }
        ),
    },
    "loggers": {
        "entity_graph": {
            "level": "DEBUG",
        },
        "openai": {
            "level": "WARNING",
        },
        "watchfiles": {
            "level": "WARNING",
        },
        "httpcore": {
            "level": "WARNING",
        },
        "uvicorn": {
            "level": "DEBUG",
            "propagate": False,
            "handlers": ["console_uvi"]
        },
    },
}

# Removing uvicorn-related logger config if there is no uvicorn module
if not importlib.util.find_spec("uvicorn"):
    config_logger["handlers"].pop("console_uvi")
    config_logger["loggers"].pop("uvicorn")
    config_logger["formatters"].pop("uvi")
