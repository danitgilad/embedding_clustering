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
        preprocess=["standardize"], pca_components=None,
        umap_cfg={"n_neighbors": 3, "min_dist": 0.1, "metric": "euclidean"}, seed=0,
    )
    assert "kmeans" in results
    assert (tmp_path / "fake.npy").exists()
    assert (tmp_path / "figures").exists()
