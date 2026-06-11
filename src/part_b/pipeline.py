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
from src.core.visualize import cluster_montage, metric_table_png, scatter_2d
from src.utils.io import ensure_dir, write_json

log = logging.getLogger(__name__)


def build_extractors(cfg):
    """Instantiate Part B encoders in config order (arcface first = attribute source)."""
    exts = []
    for name in cfg.part_b.encoders:
        if name == "arcface":
            from src.part_b.extractors.arcface import ArcFaceExtractor
            exts.append(ArcFaceExtractor(cfg.part_b.insightface.model_name,
                                         cfg.part_b.insightface.det_size))
        elif name == "dinov2_generic":
            from src.part_b.extractors.dinov2_generic import DINOv2GenericExtractor
            exts.append(DINOv2GenericExtractor(cfg.part_b.dinov2_generic.hf_model))
        else:
            raise ValueError(f"unknown Part B encoder {name!r}")
    return exts


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


def attribute_score_fn(gender_truth: np.ndarray, age_truth: np.ndarray):
    """A k-selection objective: gender AMI + age AMI of a candidate labelling.

    Used so the k-sweep picks the number of clusters that best aligns with the attributes we
    actually care about (gender, age), rather than pure geometric silhouette. We use *adjusted*
    mutual information (AMI), which is corrected for chance — unlike raw NMI/purity it does not
    creep upward with more clusters, so it won't reward over-splitting to k_max.
    """
    from sklearn.metrics import adjusted_mutual_info_score as ami

    def score(labels: np.ndarray) -> float:
        return float(ami(gender_truth, labels) + ami(age_truth, labels))

    return score


def run_clustering_stage(extractor, assets: Sequence[Asset], out_dir: str | Path,
                         algorithms: Sequence[str], k_min: int, k_max: int,
                         preprocess: Sequence[str], pca_components: int | None,
                         umap_cfg: dict, seed: int,
                         montage_images: dict[str, Path] | None = None,
                         k_selection: str = "silhouette") -> dict:
    """Embed, cluster per algorithm, characterize clusters, validate vs pseudo-labels, plot.

    If `montage_images` maps face id -> image path, also writes a per-cluster face montage
    for the first (primary) algorithm. `k_selection="attribute"` (when attributes exist)
    picks k by maximizing gender+age AMI (chance-adjusted) instead of silhouette.
    """
    out = ensure_dir(out_dir)
    fig_dir = ensure_dir(out / "figures")
    emb = extractor.extract(assets)
    save_embeddings(emb, out)
    X = _pre(emb.vectors, list(preprocess), pca_components=pca_components)
    coords = umap_2d(X, umap_cfg["n_neighbors"], umap_cfg["min_dist"], umap_cfg["metric"], seed)

    attrs = getattr(extractor, "attributes", {})
    if attrs:
        write_json(attrs, ensure_dir(out_dir) / f"{emb.name}_attributes.json")
    gender_truth = np.array([attrs.get(i, {}).get("gender", "?") for i in emb.ids])
    age_truth = np.array([_age_bucket(attrs.get(i, {}).get("age", 0.0)) for i in emb.ids])
    score_fn = attribute_score_fn(gender_truth, age_truth) \
        if (k_selection == "attribute" and attrs) else None

    results: dict[str, dict] = {}
    primary_labels = None
    for algo in algorithms:
        res = cluster(X, algo, k_min, k_max, seed, score_fn=score_fn)
        if primary_labels is None:
            primary_labels = res.labels
        row = {"n_clusters": res.n_clusters, **M.internal_metrics(X, res.labels)}
        if attrs:
            row.update({f"gender_{k}": v for k, v in
                        M.external_metrics(res.labels, gender_truth).items()})
            row.update({f"age_{k}": v for k, v in
                        M.external_metrics(res.labels, age_truth).items()})
            results[f"{algo}__profile"] = characterize_clusters(res.labels, emb.ids, attrs)
        results[algo] = row
        note = ""
        if algo == "hdbscan" and res.n_clusters == 0:
            mcs = max(5, X.shape[0] // 20)
            note = ("HDBSCAN is density-based (no preset k): it forms a cluster only where a region "
                    f"of ≥{mcs} points is denser than its surroundings, separated by lower-density "
                    "gaps. It found none → every point is noise (k=0) — the honest signature of "
                    "a continuous embedding manifold, not a failure.")
        scatter_2d(coords, res.labels, fig_dir / f"{emb.name}_{algo}_umap.png",
                   title=f"{emb.name} · {algo} (k={res.n_clusters})", note=note)
    metric_table_png({a: {k: v for k, v in r.items() if isinstance(v, float)}
                      for a, r in results.items() if not a.endswith("__profile")},
                     fig_dir / f"{emb.name}_metrics.png",
                     title=f"{emb.name} face clustering metrics")
    if montage_images and primary_labels is not None:
        sel = [(montage_images[i], int(primary_labels[j]))
               for j, i in enumerate(emb.ids) if i in montage_images]
        if sel:
            prof = results.get(f"{algorithms[0]}__profile", {})
            titles = {int(c): f"C{c} · n={d['size']} · {d['pct_top_gender']*100:.0f}% "
                              f"{d['top_gender']} · age {d['mean_age']:.0f}"
                      for c, d in prof.items()}
            cluster_montage([p for p, _ in sel], [lab for _, lab in sel],
                            fig_dir / f"{emb.name}_clusters_montage.png",
                            row_titles=titles,
                            caption="Faces grouped by KMeans cluster (sample per cluster)")
    write_json(results, out / f"{emb.name}_results.json")
    return results
