"""Render a GLB's TRIANGULATED MESH SURFACE to shaded PNGs (mesh, not point cloud).

Uses matplotlib Poly3DCollection (no GPU/EGL). Renders at supersample x size then
LANCZOS-downscales for smooth edges. One PNG per view angle.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from src.utils.io import ensure_dir, sanitize_id

log = logging.getLogger(__name__)
_LIGHT = np.array([0.3, 0.2, 0.9]) / np.linalg.norm([0.3, 0.2, 0.9])


def _vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Return (V,3) float colors in [0,1] from the mesh, or None if unavailable."""
    try:
        vc = mesh.visual.vertex_colors
        if vc is not None and len(vc) == len(mesh.vertices):
            return np.asarray(vc[:, :3], dtype=float) / 255.0
    except Exception as exc:  # noqa: BLE001 - colors optional; degrade to gray, but log
        log.debug("vertex_colors unavailable; using gray fallback: %s", exc)
    return None


def _render_one(mesh: trimesh.Trimesh, size_px: int, elev: float, azim: float,
                ss: int) -> Image.Image:
    """Rasterize one view to an RGBA PIL image (shaded, anti-aliased)."""
    v = np.asarray(mesh.vertices, dtype=float)
    v = v - v.mean(axis=0)
    max_norm = float(np.linalg.norm(v, axis=1).max()) or 1.0
    v = v / max_norm
    f = np.asarray(mesh.faces)

    vc = _vertex_colors(mesh)
    face_rgb = vc[f].mean(axis=1) if vc is not None else np.full((len(f), 3), 0.6)
    shade = 0.45 + 0.55 * np.clip(np.abs(np.asarray(mesh.face_normals) @ _LIGHT), 0, 1)
    face_rgb = np.clip(face_rgb * shade[:, None], 0, 1)

    dpi = 100
    fig = plt.figure(figsize=(size_px * ss / dpi, size_px * ss / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(v[f], facecolors=face_rgb, edgecolors="none"))
    for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        lim(-0.6, 0.6)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)   # no bbox_inches="tight": keep square canvas
    plt.close(fig); buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    return img.resize((size_px, size_px), Image.LANCZOS)  # exact, consistent output size


def render_views(mesh: trimesh.Trimesh, asset_id: str, out_dir: str | Path,
                 size_px: int, supersample: int,
                 views: list[tuple[float, float]]) -> list[Path]:
    """Render `mesh` from each (elev, azim) view; save <out_dir>/<id>_v<i>.png. Return paths."""
    out = ensure_dir(out_dir)
    ss = max(1, int(supersample))
    paths: list[Path] = []
    for i, (elev, azim) in enumerate(views):
        img = _render_one(mesh, size_px, elev, azim, ss)
        p = out / f"{sanitize_id(asset_id)}_v{i}.png"
        img.save(p); paths.append(p)
    log.info("Rendered %d views for %s", len(paths), asset_id)
    return paths
