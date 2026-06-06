"""Small filesystem helpers shared across parts."""
from __future__ import annotations

from pathlib import Path


def sanitize_id(s: str) -> str:
    """Make a string safe for a filename: keep alnum/-_. , replace the rest with '_'."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
