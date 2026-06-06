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
    if bake_texture_color:
        for geo in obj.geometry.values():
            try:
                geo.visual = geo.visual.to_color()
            except Exception as exc:  # noqa: BLE001 - colour baking is best-effort
                log.debug("texture->colour bake failed (%s); leaving uncoloured", exc)
    dumped = obj.dump(concatenate=True)
    if not isinstance(dumped, trimesh.Trimesh):
        raise ValueError("GLB did not concatenate to a single mesh")
    return dumped


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
