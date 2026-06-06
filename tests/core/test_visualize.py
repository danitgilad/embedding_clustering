import numpy as np
from src.core.visualize import scatter_2d, metric_table_png, cluster_montage

def test_scatter_writes_png(tmp_path):
    pts = np.random.RandomState(0).rand(20, 2)
    labels = np.array([0, 1] * 10)
    out = scatter_2d(pts, labels, tmp_path / "s.png", title="t")
    assert out.exists() and out.stat().st_size > 0

def test_metric_table_writes_png(tmp_path):
    rows = {"dinov2": {"silhouette": 0.5}, "point_mae": {"silhouette": 0.3}}
    out = metric_table_png(rows, tmp_path / "tbl.png", title="cmp")
    assert out.exists()

def test_cluster_montage_writes_png(tmp_path):
    from PIL import Image
    imgs = []
    for i in range(4):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (8, 8), (i * 10, 0, 0)).save(p)
        imgs.append(p)
    labels = np.array([0, 0, 1, 1])
    out = cluster_montage(imgs, labels, tmp_path / "m.png")
    assert out.exists()
