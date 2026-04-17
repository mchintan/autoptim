"""Rich-based console logger used by the CLI and orchestrator."""
from __future__ import annotations

import logging
from rich.console import Console
from rich.logging import RichHandler

_console: Console | None = None


def console() -> Console:
    global _console
    if _console is None:
        _console = Console(highlight=False)
    return _console


def configure(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console(), show_path=False, rich_tracebacks=True)],
        force=True,
    )
    return logging.getLogger("autoptim")


def logger(name: str = "autoptim") -> logging.Logger:
    return logging.getLogger(name)
