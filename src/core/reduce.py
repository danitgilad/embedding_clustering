"""Embedding preprocessing + 2D projection for visualization.

preprocess() applies a configurable, ordered chain (standardize, l2norm) then optional
PCA. umap_2d() projects to 2D for scatter plots only — clustering runs on preprocess()
output, not on the UMAP coords.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

log = logging.getLogger(__name__)


def preprocess(
    X: np.ndarray,
    steps: Sequence[str],
    pca_components: int | None = None,
) -> np.ndarray:
    """Apply ordered steps ('standardize', 'l2norm') then optional PCA. Returns new array."""
    out = np.asarray(X, dtype=float)
    for step in steps:
        if step == "standardize":
            out = StandardScaler().fit_transform(out)
        elif step == "l2norm":
            out = normalize(out, norm="l2", axis=1)
        else:
            raise ValueError(f"unknown preprocess step: {step!r}")
    if pca_components:
        n = min(pca_components, *out.shape)
        out = PCA(n_components=n, random_state=0).fit_transform(out)
        log.info("PCA -> %d dims", n)
    return out


def umap_2d(
    X: np.ndarray, n_neighbors: int, min_dist: float, metric: str, seed: int
) -> np.ndarray:
    """Project X to 2D with UMAP for plotting. n_neighbors auto-capped at N-1."""
    import umap

    n = max(2, min(n_neighbors, X.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n, min_dist=min_dist, metric=metric, random_state=seed
    )
    return reducer.fit_transform(X)
