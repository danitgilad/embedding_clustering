"""Self-contained interactive Plotly viewer for embedding clusters.

Render-only: coordinates, cluster labels, metrics, and base64 thumbnails are passed in.
One toggle button per encoder; one Plotly trace per cluster; hover shows the point's
thumbnail + id + any extra meta. For small sets (always_show_thumbs=True) thumbnails are
placed on the plot as cluster-coloured cards (umap_viewer style); for large sets they
appear on hover only. Adapted from glasses_3d_umap/src/glasses_umap/umap_view.py.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path


def image_to_data_uri(path: str | Path, max_px: int = 96) -> str:
    """Downscale an image so its long side is <= max_px; return a base64 PNG data URI.

    Returns "" if the file is missing (the viewer then renders that point thumbnail-less).
    """
    from PIL import Image

    p = Path(path)
    if not p.exists():
        return ""
    img = Image.open(p).convert("RGBA")
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
