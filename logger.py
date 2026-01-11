import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_LOGGERS: dict[str, logging.Logger] = {}

def _ensure_logs_dir(logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)

def get_logger(
        name: str,
        logs_dir: str | Path = "logs",
        level: int = logging.INFO,
        filename: str | None = None,
        when: str = "midnight",
        backup_count: int = 14,
        ) -> logging.Logger:

    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        _LOGGERS[name] = logger
        return logger

    logs_path = Path(logs_dir)
    _ensure_logs_dir(logs_path)

    if filename is None:
        filename = f"{name}.log"
    log_file = logs_path / filename

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when=when,
        interval=1,
        backupCount=backup_count,
        encoding="utf-8",
        utc=False,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    _LOGGERS[name] = logger
    return logger