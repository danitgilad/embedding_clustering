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

def test_cluster_montage_accepts_row_titles(tmp_path):
    from PIL import Image
    import numpy as np
    from src.core.visualize import cluster_montage
    imgs = []
    for i in range(4):
        p = tmp_path / f"{i}.png"; Image.new("RGB", (8, 8), (i*10, 0, 0)).save(p); imgs.append(p)
    out = cluster_montage(imgs, np.array([0, 0, 1, 1]), tmp_path / "m.png",
                          row_titles={0: "C0 n=2", 1: "C1 n=2"}, caption="demo")
    assert out.exists()

def test_cluster_montage_ids_crop_summary(tmp_path):
    from PIL import Image
    import numpy as np
    imgs, ids = [], []
    for i in range(4):
        p = tmp_path / f"g{i}.png"; Image.new("RGBA", (16, 16), (0, i * 20, 0, 255)).save(p)
        imgs.append(p); ids.append(f"glb_{i}")
    out = cluster_montage(imgs, np.array([0, 0, 1, 1]), tmp_path / "m.png",
                          ids=ids, crop=True, summary=True, caption="glasses")
    assert out.exists() and out.stat().st_size > 0

def test_scatter_with_point_ids(tmp_path):
    import numpy as np
    pts = np.random.RandomState(0).rand(6, 2)
    out = scatter_2d(pts, np.array([0, 0, 1, 1, 2, 2]), tmp_path / "s2.png",
                     title="t", point_ids=[f"a{i}" for i in range(6)])
    assert out.exists() and out.stat().st_size > 0
