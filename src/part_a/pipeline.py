"""Part A orchestration: discover assets, render, extract, cluster, evaluate, visualize.

The clustering stage is extractor-agnostic (reused per encoder) — the shared pipeline.
Heavy stages (render/extract) are split out so they can run on the GPU box and cache to
disk; cluster/viz run on the cached .npy anywhere.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from src.config import Config
from src.core import metrics as M
from src.core.cluster import cluster
from src.core.embedding_store import save_embeddings
from src.core.reduce import preprocess as _pre
from src.core.types import Asset, FeatureExtractor
from src.core.visualize import cluster_montage
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
        elif name == "clip":
            from src.part_a.extractors.clip import CLIPExtractor
            exts.append(CLIPExtractor(cfg.part_a.clip.hf_model, render_dir))
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
                         seed: int,
                         montage_images: dict[str, Path] | None = None) -> dict:
    """Extract embeddings, cluster with each algorithm, write the metrics table + montage.

    If `montage_images` maps asset id -> a representative image path, also writes a per-cluster
    thumbnail montage for the first (primary) algorithm. (UMAP for the report comes from the
    viewer stage / part_a_overview.png, so no per-algorithm scatter is written here.)
    """
    out = ensure_dir(out_dir)
    fig_dir = ensure_dir(out / "figures")
    emb = extractor.extract(assets)
    save_embeddings(emb, out)
    X = _pre(emb.vectors, list(preprocess), pca_components=pca_components)
    results: dict[str, dict] = {}
    primary_labels = None
    for algo in algorithms:
        res = cluster(X, algo, k_min, k_max, seed)
        if primary_labels is None:
            primary_labels = res.labels
        results[algo] = {"n_clusters": res.n_clusters, **M.internal_metrics(X, res.labels)}
    # Part A clusters with KMeans only. The per-cluster montage carries the metrics in its header,
    # and the cross-encoder comparison table lives in part_a_overview.png — so no standalone
    # metrics table or per-algorithm UMAP scatter is written here.
    if montage_images and primary_labels is not None:
        sel = [(montage_images[i], int(primary_labels[j]), i)
               for j, i in enumerate(emb.ids) if i in montage_images]
        if sel:
            km = results[algorithms[0]]
            cluster_montage([p for p, _, _ in sel], [lab for _, lab, _ in sel],
                            fig_dir / f"{extractor.name}_clusters_montage.png",
                            ids=[i for _, _, i in sel], crop=True, summary=True,
                            caption=(f"{extractor.name} · KMeans (k={km['n_clusters']})\n"
                                     f"silhouette={km['silhouette']:.3f} · "
                                     f"DB={km['davies_bouldin']:.2f} · CH={km['calinski_harabasz']:.2f}"))
    write_json(results, out / f"{extractor.name}_results.json")
    return results
