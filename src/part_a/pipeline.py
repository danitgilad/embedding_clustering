"""Part A orchestration: discover assets, render, extract, cluster, evaluate, visualize.

The clustering stage is extractor-agnostic (reused per encoder) — the shared pipeline.
Heavy stages (render/extract) are split out so they can run on the GPU box and cache to
disk; cluster/viz run on the cached .npy anywhere.
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Sequence

from src.config import Config
from src.core import metrics as M
from src.core.cluster import cluster
from src.core.embedding_store import save_embeddings
from src.core.reduce import preprocess as _pre, umap_2d
from src.core.types import Asset, FeatureExtractor
from src.core.visualize import cluster_montage, metric_table_png, scatter_2d
from src.utils.io import ensure_dir, write_json

log = logging.getLogger(__name__)


def discover_assets(assets_dir: str | Path) -> list[Asset]:
    """Build an Asset per .glb in assets_dir (id = filename stem)."""
    d = Path(assets_dir)
    return [Asset(id=p.stem, path=p) for p in sorted(d.glob("*.glb"))]


def build_extractors(cfg: Config, render_dir: Path) -> list[FeatureExtractor]:
    """Instantiate the configured Part A extractors (2D from renders, 3D from mesh)."""
    exts: list[FeatureExtractor] = []
    for name in cfg.part_a.encoders_2d:
        if name == "dinov2":
            from src.part_a.extractors.dinov2 import DINOv2Extractor
            exts.append(DINOv2Extractor(cfg.part_a.dinov2.hf_model, render_dir))
        else:
            raise ValueError(f"unknown 2D encoder {name!r}")
    for name in cfg.part_a.encoders_3d:
        if name == "point_mae":
            from src.part_a.extractors.point_mae import PointMAEExtractor
            exts.append(PointMAEExtractor(cfg.part_a.point_mae.checkpoint,
                                          cfg.part_a.point_sampling.n_points, cfg.seed))
        else:
            raise ValueError(f"unknown 3D encoder {name!r}")
    return exts


def run_clustering_stage(extractor: FeatureExtractor, assets: Sequence[Asset],
                         out_dir: str | Path, algorithms: Sequence[str], k_min: int,
                         k_max: int, preprocess: Sequence[str], pca_components: int | None,
                         umap_cfg: dict, seed: int,
                         montage_images: dict[str, Path] | None = None) -> dict:
    """Extract (or reuse) embeddings, cluster with each algorithm, write figures + metrics.

    If `montage_images` maps asset id -> a representative image path, also writes a
    per-cluster thumbnail montage for the first (primary) algorithm.
    """
    out = ensure_dir(out_dir)
    fig_dir = ensure_dir(out / "figures")
    emb = extractor.extract(assets)
    save_embeddings(emb, out)
    X = _pre(emb.vectors, list(preprocess), pca_components=pca_components)
    coords = umap_2d(X, umap_cfg["n_neighbors"], umap_cfg["min_dist"], umap_cfg["metric"], seed)
    results: dict[str, dict] = {}
    primary_labels = None
    for algo in algorithms:
        res = cluster(X, algo, k_min, k_max, seed)
        if primary_labels is None:
            primary_labels = res.labels
        m = M.internal_metrics(X, res.labels)
        results[algo] = {"n_clusters": res.n_clusters, **m}
        scatter_2d(coords, res.labels, fig_dir / f"{extractor.name}_{algo}_umap.png",
                   title=f"{extractor.name} · {algo} (k={res.n_clusters})")
    metric_table_png({a: {k: v for k, v in r.items() if k != "n_clusters"}
                      for a, r in results.items()},
                     fig_dir / f"{extractor.name}_metrics.png",
                     title=f"{extractor.name} clustering metrics")
    if montage_images and primary_labels is not None:
        sel = [(montage_images[i], int(primary_labels[j]))
               for j, i in enumerate(emb.ids) if i in montage_images]
        if sel:
            sizes = Counter(int(l) for l in primary_labels)
            titles = {c: f"cluster {c} (n={n})" for c, n in sizes.items()}
            cluster_montage([p for p, _ in sel], [lab for _, lab in sel],
                            fig_dir / f"{extractor.name}_clusters_montage.png",
                            row_titles=titles,
                            caption=f"{extractor.name}: glasses grouped by primary-algorithm cluster")
    write_json(results, out / f"{extractor.name}_results.json")
    return results
