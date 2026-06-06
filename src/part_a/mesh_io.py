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


def to_single_mesh(obj: trimesh.Scene | trimesh.Trimesh) -> trimesh.Trimesh:
    """Concatenate a Scene's geometries into one Trimesh; pass a Trimesh through.

    Raises ValueError if the GLB had no geometry (caller skips such files).
    """
    if isinstance(obj, trimesh.Trimesh):
        return obj
    geoms = list(obj.geometry.values())
    if not geoms:
        raise ValueError("GLB contains no mesh geometry")
    if len(geoms) == 1:
        return geoms[0]
    return trimesh.util.concatenate(geoms)


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
