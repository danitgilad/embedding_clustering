"""Central logging configuration. Call configure_logging() once at program start.

No module should call logging.basicConfig or use print() for operational output.
"""
from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging idempotently to the given level.

    Replaces existing handlers so repeated calls don't stack duplicates.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper()))
