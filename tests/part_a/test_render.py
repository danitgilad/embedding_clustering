import trimesh
from PIL import Image
from src.part_a.render import render_views

def test_render_views_produces_images(tmp_path):
    mesh = trimesh.creation.box(extents=(2, 1, 0.2))
    paths = render_views(mesh, "box", tmp_path, size_px=128, supersample=1,
                         views=[(80, -90), (20, 0)])
    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        assert Image.open(p).width > 0
