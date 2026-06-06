import numpy as np
from src.core.types import Asset, Embeddings
from src.part_b.pipeline import characterize_clusters, run_clustering_stage


def test_characterize_clusters_profiles_attributes():
    labels = np.array([0, 0, 1, 1])
    attrs = {"a": {"age": 25, "gender": "F"}, "b": {"age": 27, "gender": "F"},
             "c": {"age": 60, "gender": "M"}, "d": {"age": 64, "gender": "M"}}
    ids = ["a", "b", "c", "d"]
    profile = characterize_clusters(labels, ids, attrs)
    assert profile[0]["mean_age"] < profile[1]["mean_age"]
    assert profile[0]["top_gender"] == "F" and profile[1]["top_gender"] == "M"


class FakeFaceExtractor:
    name = "arcface"
    def __init__(self):
        self.attributes = {f"a{i}": {"age": 20 + 40 * (i // 2), "gender": "F" if i < 2 else "M"}
                           for i in range(4)}
        self.skipped = {}
    def extract(self, items):
        v = np.array([[0, 0], [0.1, 0], [5, 5], [5.1, 5]], dtype=float)
        return Embeddings(v, [a.id for a in items], self.name)


def test_run_clustering_stage_writes_outputs(tmp_path):
    assets = [Asset(id=f"a{i}", path=tmp_path / f"a{i}.jpg") for i in range(4)]
    res = run_clustering_stage(
        extractor=FakeFaceExtractor(), assets=assets, out_dir=tmp_path,
        algorithms=["kmeans"], k_min=2, k_max=3, preprocess=["l2norm"],
        pca_components=None, umap_cfg={"n_neighbors": 3, "min_dist": 0.1, "metric": "cosine"},
        seed=0)
    assert "kmeans" in res
    assert (tmp_path / "figures").exists()
