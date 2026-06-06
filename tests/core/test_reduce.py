import numpy as np
from src.core.reduce import preprocess, umap_2d

def test_standardize_then_l2norm():
    X = np.array([[1.0, 2.0], [3.0, 5.0], [5.0, 6.0]])
    out = preprocess(X, ["standardize", "l2norm"])
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)

def test_pca_reduces_dim():
    X = np.random.RandomState(0).rand(20, 10)
    out = preprocess(X, [], pca_components=3)
    assert out.shape == (20, 3)

def test_umap_2d_shape():
    X = np.random.RandomState(0).rand(30, 8)
    emb = umap_2d(X, n_neighbors=5, min_dist=0.1, metric="euclidean", seed=0)
    assert emb.shape == (30, 2)
