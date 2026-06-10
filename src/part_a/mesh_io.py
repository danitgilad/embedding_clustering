"""Load GLB glasses meshes and derive geometry inputs.

GLBs load as a trimesh.Scene (multiple mesh components + materials). to_single_mesh
concatenates them into one Trimesh. sample_surface_points draws uniform points from the
triangulated surface (input to the Point-MAE encoder; surface sampling, not vertex cloud).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

log = logging.getLogger(__name__)


def load_glb(path: str | Path) -> trimesh.Scene | trimesh.Trimesh:
    """Load a .glb via trimesh. May return a Scene (multi-mesh) or a single Trimesh."""
    return trimesh.load(str(path), process=False)


def to_single_mesh(obj: trimesh.Scene | trimesh.Trimesh,
                   bake_texture_color: bool = False) -> trimesh.Trimesh:
    """Flatten a Scene into one Trimesh, APPLYING each node's scene-graph transform;
    pass a Trimesh through. Raises ValueError if there is no mesh geometry.

    Uses Scene.dump(concatenate=True) — NOT trimesh.util.concatenate — because the latter
    merges geometries in their local frames and drops the scene-graph node transforms.

    bake_texture_color: when True, bake each geometry's texture into vertex colors
    (visual.to_color()) before flattening, so the merged mesh carries the real material
    colours. Used for the coloured hover renders; the default (False) leaves the mesh
    uncoloured (the renderer then shades it grey).
    """
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if not obj.geometry:
        raise ValueError("GLB contains no mesh geometry")
    merged = _colored_single_mesh(obj) if bake_texture_color else obj.dump(concatenate=True)
    if not isinstance(merged, trimesh.Trimesh):
        raise ValueError("GLB did not concatenate to a single mesh")
    return merged


def _colored_single_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    """Flatten a Scene to one Trimesh with baked vertex colours, applying each node's
    scene-graph transform manually and merging with ``trimesh.util.concatenate``.

    We avoid ``Scene.dump(concatenate=True)`` on the coloured path because its visual-merge
    overflows on some GLBs (`index N out of bounds for axis 0 with size 4`). Applying the
    transforms ourselves keeps geometry correct (the reason we normally avoid util.concatenate)
    while sidestepping the buggy colour merge.
    """
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        geo = scene.geometry[geom_name].copy()
        geo.apply_transform(transform)
        geo.visual = _bake_geometry_color(geo)
        parts.append(geo)
    return trimesh.util.concatenate(parts)


def _bake_geometry_color(geo: trimesh.Trimesh):
    """Convert one geometry's texture to vertex colours (best-effort, merge-safe).

    Clamps UVs into [0, 1) first — trimesh's ``to_color`` indexes an H×W texture and overflows
    when a UV is exactly 1.0 (`index 1024 ... size 1024`). If sampling still fails (e.g. a 1-D /
    greyscale texture), falls back to a flat ColorVisuals from the material's main colour, so
    EVERY geometry becomes ColorVisuals and ``Scene.dump(concatenate=True)`` merges cleanly
    instead of aborting or flattening to a single colour.
    """
    vis = geo.visual
    try:
        uv = getattr(vis, "uv", None)
        if uv is not None:
            vis.uv = np.clip(np.asarray(uv, dtype=float), 0.0, 1.0 - 1e-6)
        # to_color() is LAZY — force vertex_colors here so a sampling failure is caught now
        # (not later on access) and we return a concrete (non-lazy) ColorVisuals.
        colors = np.asarray(vis.to_color().vertex_colors)
        if colors.ndim != 2 or len(colors) != len(geo.vertices):
            raise ValueError("degenerate vertex colours")
        return trimesh.visual.ColorVisuals(geo, vertex_colors=colors)
    except Exception as exc:  # noqa: BLE001 - per-geometry colour bake is best-effort
        log.debug("texture->colour bake failed (%s); flat material-colour fallback", exc)
        mc = getattr(getattr(vis, "material", None), "main_color", None)
        rgba = np.asarray(mc if mc is not None else (150, 150, 150, 255), dtype=np.uint8)
        return trimesh.visual.ColorVisuals(
            geo, vertex_colors=np.tile(rgba, (len(geo.vertices), 1)))


def sample_surface_points(mesh: trimesh.Trimesh, n_points: int, seed: int) -> np.ndarray:
    """Uniformly sample n_points (N,3) from the mesh surface, normalized to a unit sphere.

    Centering + scaling makes the descriptor translation/scale invariant.
    """
    rng = np.random.RandomState(seed)
    # trimesh >= 3.9 accepts seed= kwarg; using it here for reproducibility
    pts, _ = trimesh.sample.sample_surface(mesh, n_points, seed=rng.randint(2**31 - 1))
    pts = np.asarray(pts, dtype=float)
    pts -= pts.mean(axis=0)
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 0:
        pts /= scale
    return pts
