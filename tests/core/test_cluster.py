import numpy as np
from sklearn.datasets import make_blobs
from src.core.cluster import cluster, ClusterResult

def _blobs(k=3, n=90, seed=0):
    X, _ = make_blobs(n_samples=n, centers=k, cluster_std=0.6, random_state=seed)
    return X

def test_kmeans_recovers_k_via_silhouette():
    X = _blobs(k=3)
    res = cluster(X, algorithm="kmeans", k_min=2, k_max=6, seed=0)
    assert isinstance(res, ClusterResult)
    assert res.n_clusters == 3
    assert res.labels.shape == (90,)

def test_agglomerative_runs():
    X = _blobs(k=4)
    res = cluster(X, algorithm="agglomerative", k_min=2, k_max=6, seed=0)
    assert res.n_clusters == 4

def test_hdbscan_returns_labels():
    X = _blobs(k=3, n=120)
    res = cluster(X, algorithm="hdbscan", k_min=2, k_max=6, seed=0)
    assert res.labels.shape == (120,)
    assert res.n_clusters >= 1
