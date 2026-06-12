import numpy as np
from src.core.types import Asset, Embeddings
from src.part_a.pipeline import run_clustering_stage


class FakeExtractor:
    name = "fake"
    def extract(self, items):
        v = np.array([[0, 0], [0.1, 0], [5, 5], [5.1, 5]], dtype=float)
        return Embeddings(v, [a.id for a in items], self.name)


def test_run_clustering_stage_writes_outputs(tmp_path):
    assets = [Asset(id=f"a{i}", path=tmp_path / f"a{i}.glb") for i in range(4)]
    results = run_clustering_stage(
        extractor=FakeExtractor(), assets=assets, out_dir=tmp_path,
        algorithms=["kmeans"], k_min=2, k_max=3,
        preprocess=["standardize"], pca_components=None, seed=0,
    )
    assert "kmeans" in results
    assert (tmp_path / "fake.npy").exists()
    assert (tmp_path / "figures").exists()
    # Part A (KMeans-only) writes no per-algorithm scatter and no standalone metrics table —
    # the montage carries the metrics and part_a_overview.png carries the comparison.
    assert not (tmp_path / "figures" / "fake_kmeans_umap.png").exists()
    assert not (tmp_path / "figures" / "fake_metrics.png").exists()
