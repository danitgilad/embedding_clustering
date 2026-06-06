"""Part B orchestration: generate -> embed -> cluster -> CHARACTERIZE -> evaluate -> visualize.

The core deliverable is characterize_clusters: describe each cluster in human terms using
the InsightFace attributes (mean age, dominant gender, mean pose). External metrics validate
the embedding clusters against gender/age-bucket pseudo-labels.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from src.core import metrics as M
from src.core.cluster import cluster
from src.core.embedding_store import save_embeddings
from src.core.reduce import preprocess as _pre, umap_2d
from src.core.types import Asset
from src.core.visualize import metric_table_png, scatter_2d
from src.utils.io import ensure_dir, write_json

log = logging.getLogger(__name__)


def characterize_clusters(labels: np.ndarray, ids: Sequence[str],
                          attributes: dict[str, dict]) -> dict[int, dict]:
    """Per cluster: size, mean age, dominant gender, % dominant gender. Human-readable profile."""
    profile: dict[int, dict] = {}
    labels = np.asarray(labels)
    for c in sorted(set(labels.tolist())):
        members = [ids[i] for i in range(len(ids)) if labels[i] == c]
        ages = [attributes[m]["age"] for m in members if m in attributes]
        genders = [attributes[m]["gender"] for m in members if m in attributes]
        top_gender = max(set(genders), key=genders.count) if genders else "?"
        profile[int(c)] = {
            "size": len(members),
            "mean_age": float(np.mean(ages)) if ages else float("nan"),
            "top_gender": top_gender,
            "pct_top_gender": (genders.count(top_gender) / len(genders)) if genders else 0.0,
        }
    return profile


def _age_bucket(age: float) -> str:
    """Coarse age bucket used as a pseudo-label for cluster validation."""
    return "young" if age < 35 else ("middle" if age < 55 else "old")


def run_clustering_stage(extractor, assets: Sequence[Asset], out_dir: str | Path,
                         algorithms: Sequence[str], k_min: int, k_max: int,
                         preprocess: Sequence[str], pca_components: int | None,
                         umap_cfg: dict, seed: int) -> dict:
    """Embed, cluster per algorithm, characterize clusters, validate vs pseudo-labels, plot."""
    out = ensure_dir(out_dir)
    fig_dir = ensure_dir(out / "figures")
    emb = extractor.extract(assets)
    save_embeddings(emb, out)
    X = _pre(emb.vectors, list(preprocess), pca_components=pca_components)
    coords = umap_2d(X, umap_cfg["n_neighbors"], umap_cfg["min_dist"], umap_cfg["metric"], seed)

    attrs = getattr(extractor, "attributes", {})
    gender_truth = np.array([attrs.get(i, {}).get("gender", "?") for i in emb.ids])
    age_truth = np.array([_age_bucket(attrs.get(i, {}).get("age", 0.0)) for i in emb.ids])

    results: dict[str, dict] = {}
    for algo in algorithms:
        res = cluster(X, algo, k_min, k_max, seed)
        row = {"n_clusters": res.n_clusters, **M.internal_metrics(X, res.labels)}
        if attrs:
            row.update({f"gender_{k}": v for k, v in
                        M.external_metrics(res.labels, gender_truth).items()})
            row.update({f"age_{k}": v for k, v in
                        M.external_metrics(res.labels, age_truth).items()})
            results[f"{algo}__profile"] = characterize_clusters(res.labels, emb.ids, attrs)
        results[algo] = row
        scatter_2d(coords, res.labels, fig_dir / f"arcface_{algo}_umap.png",
                   title=f"arcface · {algo} (k={res.n_clusters})")
    metric_table_png({a: {k: v for k, v in r.items() if isinstance(v, float)}
                      for a, r in results.items() if not a.endswith("__profile")},
                     fig_dir / "arcface_metrics.png", title="Face clustering metrics")
    write_json(results, out / f"{emb.name}_results.json")
    return results
