import json
import numpy as np
from src.core.types import Asset, Embeddings
from src.part_b.pipeline import run_clustering_stage

class FakeFaceExtractor:
    name = "arcface"
    def __init__(self):
        self.attributes = {f"a{i}": {"age": 30 + i, "gender": "F" if i % 2 else "M",
                                     "pose_yaw": float(i)} for i in range(4)}
        self.skipped = {}
    def extract(self, items):
        return Embeddings(np.array([[0,0],[0.1,0],[5,5],[5.1,5]], float),
                          [a.id for a in items], self.name)

def test_attributes_persisted(tmp_path):
    assets = [Asset(id=f"a{i}", path=tmp_path / f"a{i}.jpg") for i in range(4)]
    run_clustering_stage(FakeFaceExtractor(), assets, tmp_path, ["kmeans"], 2, 3,
                         ["l2norm"], None,
                         {"n_neighbors": 3, "min_dist": 0.1, "metric": "cosine"}, 0)
    attrs = json.loads((tmp_path / "arcface_attributes.json").read_text())
    assert attrs["a0"]["gender"] == "M" and "pose_yaw" in attrs["a0"]
