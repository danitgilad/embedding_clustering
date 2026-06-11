import numpy as np
import trimesh

from src.core.types import Asset
from src.part_a.explore import explore_assets, write_exploration_report


def _make_glb(path, n_components=2):
    """Write a small multi-component GLB (stacked boxes) to `path`."""
    scene = trimesh.Scene()
    for k in range(n_components):
        box = trimesh.creation.box(extents=(1, 1, 1))
        box.apply_translation((2.0 * k, 0, 0))
        scene.add_geometry(box, geom_name=f"part_{k}")
    path.write_bytes(scene.export(file_type="glb"))


def test_explore_assets_reports_structure(tmp_path):
    glb = tmp_path / "specs.glb"
    _make_glb(glb, n_components=2)
    rows = explore_assets([Asset(id="specs", path=glb)])
    assert len(rows) == 1
    r = rows[0]
    assert r["id"] == "specs"
    assert r["components"] == 2
    assert r["vertices"] == 16 and r["faces"] == 24   # two unit cubes
    assert r["extent"] is not None and len(r["extent"]) == 3


def test_write_exploration_report_writes_markdown(tmp_path):
    glb = tmp_path / "specs.glb"
    _make_glb(glb, n_components=1)
    out = write_exploration_report([Asset(id="specs", path=glb)], tmp_path / "dataset_exploration.md")
    assert out.exists()
    text = out.read_text()
    assert "Dataset Exploration" in text and "| specs |" in text and "Observations" in text
