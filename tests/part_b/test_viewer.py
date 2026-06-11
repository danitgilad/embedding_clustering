import json
import numpy as np
from PIL import Image
from src.config import load_config
from src.part_b.viewer import build_part_b_overview, build_part_b_viewer


def _fixture(tmp_path, n=12):
    out = tmp_path / "outputs"; out.mkdir()
    faces = tmp_path / "faces"; faces.mkdir()
    ids = [f"face_{i:04d}" for i in range(n)]
    np.save(out / "arcface.npy", np.random.RandomState(1).rand(n, 32))
    (out / "arcface.ids.json").write_text(json.dumps(ids))
    (out / "arcface_attributes.json").write_text(json.dumps(
        {i: {"age": 20 + 4 * k, "gender": "F" if k % 2 else "M", "pose_yaw": 0.0}
         for k, i in enumerate(ids)}))
    for i in ids:
        Image.new("RGB", (64, 64), (90, 90, 10)).save(faces / f"{i}.jpg")
    return out, faces, ids


def test_build_part_b_viewer_writes_html(tmp_path):
    out, faces, _ = _fixture(tmp_path)
    cfg = load_config("config/default.yaml")
    html_path = build_part_b_viewer(cfg, out_dir=out, faces_dir=faces)
    assert html_path.exists()
    txt = html_path.read_text()
    assert "arcface" in txt and "face_0000" in txt and "cdn.plot.ly" in txt


def test_build_part_b_overview_writes_png(tmp_path):
    out, faces, _ = _fixture(tmp_path)
    cfg = load_config("config/default.yaml")
    png = build_part_b_overview(cfg, out_dir=out, faces_dir=faces)
    assert png.exists() and png.stat().st_size > 0
