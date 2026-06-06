"""Build the Part A interactive HTML viewer from cached embeddings (decoupled from encoding).

For each outputs/part_a/<encoder>.npy it recomputes UMAP + KMeans + internal metrics
(deterministic, reusing core.*), uses the front-view renders as always-visible thumbnails,
and writes outputs/part_a/viewer.html.
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.config import Config
from src.core import metrics as M
from src.core.cluster import cluster
from src.core.embedding_store import load_embeddings
from src.core.html_viewer import build_viewer_html, image_to_data_uri
from src.core.reduce import preprocess, umap_2d
from src.utils.io import sanitize_id

log = logging.getLogger(__name__)


def build_part_a_viewer(cfg: Config, out_dir: str | Path, render_dir: str | Path) -> Path:
    """Assemble outputs/part_a/viewer.html from every <encoder>.npy in out_dir."""
    out_dir, render_dir = Path(out_dir), Path(render_dir)
    npys = sorted(p for p in out_dir.glob("*.npy"))
    if not npys:
        raise FileNotFoundError(f"no embeddings (*.npy) in {out_dir}")
    ids = None
    projections: dict[str, dict] = {}
    um = cfg.reduce.umap
    algo = cfg.part_a.clustering.algorithms[0]
    for npy in npys:
        emb = load_embeddings(npy.stem, out_dir)
        if ids is None:
            ids = emb.ids
        elif emb.ids != ids:
            raise ValueError(f"id mismatch for {npy.stem}; viewer needs a shared id order")
        X = preprocess(emb.vectors, list(cfg.reduce.preprocess), pca_components=cfg.reduce.pca_components)
        coords = umap_2d(X, um.n_neighbors, um.min_dist, um.metric, cfg.seed)
        res = cluster(X, algo, cfg.part_a.clustering.k_min, cfg.part_a.clustering.k_max, cfg.seed)
        projections[npy.stem] = {"coords2d": coords, "labels": res.labels,
                                 "metrics": M.internal_metrics(X, res.labels)}
    thumbs = [image_to_data_uri(render_dir / f"{sanitize_id(i)}_v0.png", max_px=128) for i in ids]
    html = build_viewer_html(
        projections, ids, thumbs, hover_meta=None,
        title="Part A — 3D glasses: 2D-vs-3D feature clusters",
        intro=("Each point is one glasses asset, shown as its rendered thumbnail on a "
               "cluster-coloured card. Buttons switch the feature/encoder. Hover for id."),
        always_show_thumbs=True, page_title="Part A — Glasses Cluster Viewer")
    out_html = out_dir / "viewer.html"
    out_html.write_text(html)
    log.info("Wrote %s", out_html)
    return out_html
