"""Clustering with automatic k-selection.

KMeans and Agglomerative sweep k in [k_min, k_max] and pick the k with the best cosine
silhouette. HDBSCAN needs no k (density-based) and may mark noise as label -1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusterResult:
    """Labels per row plus metadata about the chosen clustering."""
    labels: np.ndarray
    n_clusters: int
    algorithm: str
    k_selected: int | None  # None for HDBSCAN


def _best_k(X: np.ndarray, make, k_min: int, k_max: int,
            score_fn: Callable[[np.ndarray], float] | None = None) -> tuple[np.ndarray, int]:
    """Sweep k, return (labels, k) maximizing a score. k_max capped at N-1.

    Default score is cosine silhouette (geometric separation). Pass `score_fn(labels)->float`
    to select k by a different objective — e.g. attribute alignment (NMI vs pseudo-labels).
    """
    best_labels, best_k, best_score = None, None, -np.inf
    hi = min(k_max, X.shape[0] - 1)
    for k in range(max(2, k_min), hi + 1):
        labels = make(k).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = score_fn(labels) if score_fn else silhouette_score(X, labels, metric="cosine")
        log.debug("k=%d score=%.4f", k, score)
        if score > best_score:
            best_labels, best_k, best_score = labels, k, score
    if best_labels is None:
        best_labels, best_k = make(2).fit_predict(X), 2
    return best_labels, best_k


def cluster(
    X: np.ndarray, algorithm: str, k_min: int, k_max: int, seed: int,
    score_fn: Callable[[np.ndarray], float] | None = None,
) -> ClusterResult:
    """Cluster X with the named algorithm. Returns a ClusterResult.

    score_fn (optional) overrides silhouette as the k-selection objective for KMeans /
    Agglomerative (HDBSCAN chooses no k).
    """
    if algorithm == "kmeans":
        labels, k = _best_k(
            X, lambda k: KMeans(n_clusters=k, n_init=10, random_state=seed), k_min, k_max,
            score_fn,
        )
        return ClusterResult(labels, len(set(labels)), "kmeans", k)
    if algorithm == "agglomerative":
        labels, k = _best_k(
            X, lambda k: AgglomerativeClustering(n_clusters=k), k_min, k_max, score_fn
        )
        return ClusterResult(labels, len(set(labels)), "agglomerative", k)
    if algorithm == "hdbscan":
        import hdbscan

        labels = hdbscan.HDBSCAN(min_cluster_size=max(5, X.shape[0] // 20)).fit_predict(X)
        n = len(set(labels) - {-1})
        return ClusterResult(labels, n, "hdbscan", None)
    raise ValueError(f"unknown algorithm: {algorithm!r}")
