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
    # which encoders are the 2D (render-based) feature vs the 3D (mesh-based) feature — this
    # is the whole point of Part A, so tag each toggle/metric-row with its modality.
    modality = {n: "2D · render" for n in cfg.part_a.encoders_2d}
    modality.update({n: "3D · mesh" for n in cfg.part_a.encoders_3d})
    for npy in npys:
        emb = load_embeddings(npy.stem, out_dir)
        if ids is None:
            ids = emb.ids
        elif emb.ids != ids:
            raise ValueError(f"id mismatch for {npy.stem}; viewer needs a shared id order")
        X = preprocess(emb.vectors, list(cfg.reduce.preprocess), pca_components=cfg.reduce.pca_components)
        coords = umap_2d(X, um.n_neighbors, um.min_dist, um.metric, cfg.seed)
        res = cluster(X, algo, cfg.part_a.clustering.k_min, cfg.part_a.clustering.k_max, cfg.seed)
        label = f"{npy.stem} · {modality.get(npy.stem, '?')}"
        projections[label] = {"coords2d": coords, "labels": res.labels,
                              "metrics": M.internal_metrics(X, res.labels)}
    thumbs = [image_to_data_uri(render_dir / f"{sanitize_id(i)}_v0.png", max_px=128) for i in ids]
    # coloured renders (textures baked) used only in the larger hover popup; fall back to the
    # grey on-plot thumbnail if a coloured render is missing.
    colored_dir = render_dir / "colored"
    hover_thumbs = [image_to_data_uri(colored_dir / f"{sanitize_id(i)}_v0.png", max_px=256)
                    or thumbs[k] for k, i in enumerate(ids)]
    twod = ", ".join(cfg.part_a.encoders_2d)
    threed = ", ".join(cfg.part_a.encoders_3d)
    html = build_viewer_html(
        projections, ids, thumbs, hover_meta=None,
        title="Part A — 3D glasses: 2D-vs-3D feature clusters",
        intro=(f"Each point is one glasses asset (grey render on the plot; hover for a larger "
               f"colour render + id). <b>The point of Part A is 2D vs 3D features:</b> the "
               f"<b>2D · render</b> features (<b>{twod}</b>) come from the rendered images; the "
               f"<b>3D · mesh</b> feature (<b>{threed}</b>) is computed directly from the mesh, "
               f"with no rendering. Each toggle button and metric-table row is tagged 2D/3D — "
               f"switch between them to compare."),
        always_show_thumbs=True, thumb_scale=2.0, hover_thumbs=hover_thumbs,
        page_title="Part A — Glasses Cluster Viewer")
    out_html = out_dir / "viewer.html"
    out_html.write_text(html)
    log.info("Wrote %s", out_html)
    return out_html
