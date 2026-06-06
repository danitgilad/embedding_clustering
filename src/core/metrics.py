"""Cluster-quality metrics.

internal_metrics: no labels needed (silhouette/DB/CH). external_metrics: compare cluster
labels against pseudo-labels (Part B's InsightFace age/gender) via NMI/ARI/purity.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Silhouette (cosine), Davies-Bouldin, Calinski-Harabasz on X given labels.

    Noise points (label -1, HDBSCAN) are excluded from the computation.
    """
    mask = labels != -1
    Xv, lv = X[mask], labels[mask]
    if len(set(lv)) < 2:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan")}
    return {
        "silhouette": float(silhouette_score(Xv, lv, metric="cosine")),
        "davies_bouldin": float(davies_bouldin_score(Xv, lv)),
        "calinski_harabasz": float(calinski_harabasz_score(Xv, lv)),
    }


def _purity(labels: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of points in the majority truth-class of their assigned cluster."""
    total, correct = len(labels), 0
    for c in set(labels):
        members = truth[labels == c]
        if len(members):
            vals, counts = np.unique(members, return_counts=True)
            correct += counts.max()
    return correct / total


def external_metrics(labels: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """NMI / ARI / purity of cluster labels vs categorical pseudo-labels."""
    return {
        "nmi": float(normalized_mutual_info_score(truth, labels)),
        "ari": float(adjusted_rand_score(truth, labels)),
        "purity": float(_purity(labels, truth)),
    }
