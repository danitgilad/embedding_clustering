"""Build the Part B interactive HTML viewer from cached embeddings (decoupled from encoding).

For each outputs/part_b/<encoder>.npy it recomputes UMAP + KMeans + internal metrics, uses
downscaled face images as HOVER thumbnails (too many points for always-visible), and shows
age/gender/pose in the hover from <encoder>_attributes.json. Writes outputs/part_b/viewer.html.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.config import Config
from src.core import metrics as M
from src.core.cluster import cluster
from src.core.embedding_store import load_embeddings
from src.core.html_viewer import build_viewer_html, image_to_data_uri
from src.core.reduce import preprocess, umap_2d
from src.part_b.pipeline import _age_bucket, attribute_score_fn

log = logging.getLogger(__name__)


def build_part_b_viewer(cfg: Config, out_dir: str | Path, faces_dir: str | Path) -> Path:
    """Assemble outputs/part_b/viewer.html from every <encoder>.npy in out_dir."""
    out_dir, faces_dir = Path(out_dir), Path(faces_dir)
    npys = sorted(p for p in out_dir.glob("*.npy"))
    if not npys:
        raise FileNotFoundError(f"no embeddings (*.npy) in {out_dir}")
    ids = None
    projections: dict[str, dict] = {}
    hover_meta: dict[str, dict] = {}
    um = cfg.reduce.umap
    algo = cfg.part_b.clustering.algorithms[0]
    use_attr = cfg.part_b.clustering.k_selection == "attribute"
    for npy in npys:
        emb = load_embeddings(npy.stem, out_dir)
        if ids is None:
            ids = emb.ids
        elif emb.ids != ids:
            raise ValueError(f"id mismatch for {npy.stem}; viewer needs a shared id order")
        X = preprocess(emb.vectors, list(cfg.reduce.preprocess), pca_components=cfg.reduce.pca_components)
        coords = umap_2d(X, um.n_neighbors, um.min_dist, um.metric, cfg.seed)
        # encoders with an attributes file (e.g. arcface) drive hover meta and, when enabled,
        # attribute-based k-selection so the viewer's k matches the pipeline's results.
        score_fn = None
        attr_file = out_dir / f"{npy.stem}_attributes.json"
        if attr_file.exists():
            raw = json.loads(attr_file.read_text())
            if not hover_meta:
                hover_meta = {i: {"age": f"{a['age']:.0f}", "gender": a["gender"],
                                  "pose_yaw": f"{a['pose_yaw']:.0f}"} for i, a in raw.items()}
            if use_attr:
                gender = np.array([raw.get(i, {}).get("gender", "?") for i in emb.ids])
                age = np.array([_age_bucket(raw.get(i, {}).get("age", 0.0)) for i in emb.ids])
                score_fn = attribute_score_fn(gender, age)
        res = cluster(X, algo, cfg.part_b.clustering.k_min, cfg.part_b.clustering.k_max,
                      cfg.seed, score_fn=score_fn)
        projections[npy.stem] = {"coords2d": coords, "labels": res.labels,
                                 "metrics": M.internal_metrics(X, res.labels)}
    thumbs = [image_to_data_uri(faces_dir / f"{i}.jpg", max_px=96, fmt="jpeg") for i in ids]
    html = build_viewer_html(
        projections, ids, thumbs, hover_meta=hover_meta or None,
        title="Part B — Faces: attribute clusters",
        intro=("Each point is one generated face, coloured by its KMeans cluster. Hover a "
               "point to see the face plus predicted age / gender / pose. Buttons switch the "
               "embedding model."),
        always_show_thumbs=False, page_title="Part B — Face Cluster Viewer")
    out_html = out_dir / "viewer.html"
    out_html.write_text(html)
    log.info("Wrote %s", out_html)
    return out_html
