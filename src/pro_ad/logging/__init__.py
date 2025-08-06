import logging

LOGGERS = {}


class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelname == "WARNING":
            record.levelname = "WARN"
        return super().format(record)


def get_logger(name: str, level: int = logging.INFO):
    if name in LOGGERS:
        return LOGGERS[name]

    fmt = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = CustomFormatter(fmt=fmt, datefmt=datefmt)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(level)

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)

    LOGGERS[name] = logger
    return logger
