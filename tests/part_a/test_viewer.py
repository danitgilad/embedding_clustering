import json
import numpy as np
from PIL import Image
from src.config import load_config
from src.part_a.viewer import build_part_a_viewer

def test_build_part_a_viewer_writes_html(tmp_path):
    out = tmp_path / "outputs"; out.mkdir()
    renders = tmp_path / "renders"; renders.mkdir()
    ids = [f"g{i}" for i in range(6)]
    np.save(out / "dinov2.npy", np.random.RandomState(0).rand(6, 32))
    (out / "dinov2.ids.json").write_text(json.dumps(ids))
    for i in ids:
        Image.new("RGB", (40, 40), (10, 90, 10)).save(renders / f"{i}_v0.png")
    cfg = load_config("config/default.yaml")
    html_path = build_part_a_viewer(cfg, out_dir=out, render_dir=renders)
    assert html_path.exists()
    txt = html_path.read_text()
    assert "dinov2" in txt and "g0" in txt and "cdn.plot.ly" in txt
