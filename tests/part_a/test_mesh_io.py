import numpy as np
import trimesh
from src.part_a.mesh_io import to_single_mesh, sample_surface_points

def _scene_with_two_boxes():
    a = trimesh.creation.box(extents=(1, 1, 1))
    b = trimesh.creation.box(extents=(1, 1, 1)); b.apply_translation([3, 0, 0])
    return trimesh.Scene([a, b])

def test_to_single_mesh_concatenates_scene():
    mesh = to_single_mesh(_scene_with_two_boxes())
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.vertices) > 0 and len(mesh.faces) > 0

def test_sample_surface_points_shape():
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    pts = sample_surface_points(mesh, n_points=1024, seed=0)
    assert pts.shape == (1024, 3)
    pts2 = sample_surface_points(mesh, n_points=1024, seed=0)
    assert np.allclose(pts, pts2)
