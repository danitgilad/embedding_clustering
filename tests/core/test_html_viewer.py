from PIL import Image
from src.core.html_viewer import image_to_data_uri

def test_image_to_data_uri_downscales_and_encodes(tmp_path):
    p = tmp_path / "x.png"
    Image.new("RGB", (512, 256), (200, 10, 10)).save(p)
    uri = image_to_data_uri(p, max_px=96)
    assert uri.startswith("data:image/png;base64,")
    import base64, io
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert max(Image.open(io.BytesIO(raw)).size) == 96

def test_image_to_data_uri_missing_returns_empty(tmp_path):
    assert image_to_data_uri(tmp_path / "nope.png") == ""

import numpy as np
from src.core.html_viewer import build_viewer_html

def _proj(n, k):
    rng = np.random.RandomState(0)
    return {"coords2d": rng.rand(n, 2), "labels": np.arange(n) % k,
            "metrics": {"silhouette": 0.5, "davies_bouldin": 0.8, "calinski_harabasz": 4.0}}

def test_build_viewer_html_contains_encoders_ids_and_plotly():
    ids = [f"a{i}" for i in range(6)]
    thumbs = ["data:image/png;base64,AAAA"] * 6
    projections = {"dinov2": _proj(6, 3), "point_mae": _proj(6, 2)}
    html = build_viewer_html(projections, ids, thumbs, hover_meta=None,
                             title="Part A", intro="hello", always_show_thumbs=True)
    assert "cdn.plot.ly" in html
    assert "dinov2" in html and "point_mae" in html
    assert "a0" in html and "a5" in html
    assert "<table" in html
