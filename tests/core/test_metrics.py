import numpy as np
from sklearn.datasets import make_blobs
from src.core.metrics import internal_metrics, external_metrics

def test_internal_metrics_keys_and_ranges():
    X, y = make_blobs(n_samples=60, centers=3, cluster_std=0.5, random_state=0)
    m = internal_metrics(X, y)
    assert set(m) == {"silhouette", "davies_bouldin", "calinski_harabasz"}
    assert -1.0 <= m["silhouette"] <= 1.0
    assert m["davies_bouldin"] >= 0.0

def test_external_metrics_perfect_match():
    labels = np.array([0, 0, 1, 1, 2, 2])
    truth = np.array(["a", "a", "b", "b", "c", "c"])
    m = external_metrics(labels, truth)
    assert m["nmi"] == 1.0 and m["ari"] == 1.0 and m["purity"] == 1.0
