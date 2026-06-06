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


def write_json(obj: object, path: str | Path) -> Path:
    """Write obj as pretty JSON, coercing numpy scalars/arrays and NaN -> null.

    Used to persist clustering results (metrics + cluster profiles) as a machine-readable
    findings artifact alongside the figures.
    """
    import json
    import math

    import numpy as np

    def clean(o: object) -> object:
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, (np.floating, float)):
            f = float(o)
            return None if math.isnan(f) else f
        return o

    p = Path(path)
    p.write_text(json.dumps(clean(obj), indent=2))
    return p
