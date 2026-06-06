import json
import numpy as np
from PIL import Image
from src.config import load_config
from src.part_b.viewer import build_part_b_viewer

def test_build_part_b_viewer_writes_html(tmp_path):
    out = tmp_path / "outputs"; out.mkdir()
    faces = tmp_path / "faces"; faces.mkdir()
    ids = [f"face_{i:04d}" for i in range(8)]
    np.save(out / "arcface.npy", np.random.RandomState(0).rand(8, 32))
    (out / "arcface.ids.json").write_text(json.dumps(ids))
    (out / "arcface_attributes.json").write_text(json.dumps(
        {i: {"age": 30, "gender": "F", "pose_yaw": 1.0} for i in ids}))
    for i in ids:
        Image.new("RGB", (64, 64), (90, 90, 10)).save(faces / f"{i}.jpg")
    cfg = load_config("config/default.yaml")
    html_path = build_part_b_viewer(cfg, out_dir=out, faces_dir=faces)
    assert html_path.exists()
    txt = html_path.read_text()
    assert "arcface" in txt and "face_0000" in txt and "cdn.plot.ly" in txt
