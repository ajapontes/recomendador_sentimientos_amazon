# src/utils/logging_setup.py
"""
Configuración de logging para la API:
- Formato consistente (nivel, ts, logger, mensaje).
- Integra logs de uvicorn.
- Permite elegir nivel desde settings (INFO por defecto).
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str | None = None) -> None:
    """
    Configura logging de la app y ajusta loggers de uvicorn.

    Parameters
    ----------
    level : str | None
        Nivel de logging ("DEBUG", "INFO", "WARNING", "ERROR"). Por defecto "INFO".
    """
    log_level = (level or "INFO").upper()
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Root logger
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Afinar uvicorn para que no duplique ni cambie formato
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.setLevel(getattr(logging, log_level, logging.INFO))
        # No cambiar handlers, dejamos que herede el formateador root

    logging.getLogger(__name__).info("Logging inicializado (level=%s)", log_level)
