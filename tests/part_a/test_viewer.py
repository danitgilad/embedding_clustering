import json
import numpy as np
from PIL import Image
from src.config import load_config
from src.part_a.viewer import (build_feature_distribution_figure, build_part_a_overview,
                               build_part_a_viewer)


def _fixture(tmp_path):
    out = tmp_path / "outputs"; out.mkdir()
    renders = tmp_path / "renders"; renders.mkdir()
    ids = [f"g{i}" for i in range(6)]
    for enc, dim in (("dinov2", 32), ("point_mae", 24)):
        np.save(out / f"{enc}.npy", np.random.RandomState(0).rand(6, dim))
        (out / f"{enc}.ids.json").write_text(json.dumps(ids))
    for i in ids:
        Image.new("RGB", (40, 40), (10, 90, 10)).save(renders / f"{i}_v0.png")
    return out, renders, ids


def test_build_part_a_viewer_writes_html(tmp_path):
    out, renders, _ = _fixture(tmp_path)
    cfg = load_config("config/default.yaml")
    html_path = build_part_a_viewer(cfg, out_dir=out, render_dir=renders)
    assert html_path.exists()
    txt = html_path.read_text()
    assert "dinov2" in txt and "g0" in txt and "cdn.plot.ly" in txt


def test_build_part_a_overview_writes_png(tmp_path):
    out, renders, _ = _fixture(tmp_path)
    cfg = load_config("config/default.yaml")
    png = build_part_a_overview(cfg, out_dir=out, render_dir=renders)
    assert png.exists() and png.stat().st_size > 0


def test_build_feature_distribution_figure_writes_png(tmp_path):
    out, _, _ = _fixture(tmp_path)
    cfg = load_config("config/default.yaml")
    png = build_feature_distribution_figure(cfg, out_dir=out)
    assert png.exists() and png.stat().st_size > 0
