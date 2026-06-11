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
from src.utils.io import ensure_dir

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
    k_min, k_max = cfg.part_b.clustering.k_min, cfg.part_b.clustering.k_max
    attr_ctx = None   # (X, gender, age) of the first encoder with attributes, for the k-table
    for npy in npys:
        emb = load_embeddings(npy.stem, out_dir)
        if ids is None:
            ids = emb.ids
        elif emb.ids != ids:
            raise ValueError(f"id mismatch for {npy.stem}; viewer needs a shared id order")
        X = preprocess(emb.vectors, list(cfg.reduce.preprocess), pca_components=cfg.reduce.pca_components)
        coords = umap_2d(X, um.n_neighbors, um.min_dist, um.metric, cfg.seed)
        attr_file = out_dir / f"{npy.stem}_attributes.json"
        if attr_file.exists():
            # Encoder with attributes (arcface): expose BOTH k-selections as separate toggles
            # so the two clusterings can be compared visually on the same UMAP layout.
            raw = json.loads(attr_file.read_text())
            if not hover_meta:
                hover_meta = {i: {"age": f"{a['age']:.0f}", "gender": a["gender"],
                                  "pose_yaw": f"{a['pose_yaw']:.0f}"} for i, a in raw.items()}
            gender = np.array([raw.get(i, {}).get("gender", "?") for i in emb.ids])
            age = np.array([_age_bucket(raw.get(i, {}).get("age", 0.0)) for i in emb.ids])
            if attr_ctx is None:
                attr_ctx = (X, gender, age)
            variants = [
                ("attribute", attribute_score_fn(gender, age)),
                ("silhouette", None),
            ]
            for tag, sfn in variants:
                res = cluster(X, algo, k_min, k_max, cfg.seed, score_fn=sfn)
                m = M.internal_metrics(X, res.labels)
                m["gender_purity"] = M.external_metrics(res.labels, gender)["purity"]
                projections[f"{npy.stem} · {tag} k={res.n_clusters}"] = {
                    "coords2d": coords, "labels": res.labels, "metrics": m}
        else:
            res = cluster(X, algo, k_min, k_max, cfg.seed)
            projections[npy.stem] = {"coords2d": coords, "labels": res.labels,
                                     "metrics": M.internal_metrics(X, res.labels)}
    extra_html = ""
    if attr_ctx is not None:
        extra_html = _k_selection_table(*attr_ctx, algo, k_min, k_max, cfg.seed)
    thumbs = [image_to_data_uri(faces_dir / f"{i}.jpg", max_px=96, fmt="jpeg") for i in ids]
    html = build_viewer_html(
        projections, ids, thumbs, hover_meta=hover_meta or None,
        title="Part B — Faces: attribute clusters",
        intro=("Each point is one generated face, coloured by its KMeans cluster. Hover a "
               "point to see the face plus predicted age / gender / pose. Buttons switch the "
               "view: ArcFace under both k-selections (attribute k=3 vs silhouette k=6) and "
               "the generic-DINOv2 encoder — toggle to compare the clusterings on one layout."),
        always_show_thumbs=False, extra_html=extra_html,
        page_title="Part B — Face Cluster Viewer")
    out_html = out_dir / "viewer.html"
    out_html.write_text(html)
    log.info("Wrote %s", out_html)
    return out_html


def _k_selection_table(X: np.ndarray, gender: np.ndarray, age: np.ndarray, algo: str,
                       k_min: int, k_max: int, seed: int) -> str:
    """HTML table comparing silhouette vs attribute (AMI) k-selection for `algo` on arcface."""
    from src.core.metrics import external_metrics

    rows = []
    for mode, sfn in (("silhouette (geometric)", None),
                      ("attribute (gender+age AMI)", attribute_score_fn(gender, age))):
        res = cluster(X, algo, k_min, k_max, seed, score_fn=sfn)
        g = external_metrics(res.labels, gender)
        a = external_metrics(res.labels, age)
        rows.append((mode, res.n_clusters, g["purity"], g["nmi"], a["purity"]))
    head = ("<tr><th>k-selection</th><th>k</th><th>gender purity</th>"
            "<th>gender NMI</th><th>age purity</th></tr>")
    body = "".join(
        f"<tr><td>{m}</td><td>{k}</td><td>{gp:.3f}</td><td>{gn:.3f}</td><td>{ap:.3f}</td></tr>"
        for m, k, gp, gn, ap in rows)
    return (f'<p style="margin-top:0"><b>Choosing k — {algo} on arcface</b>: silhouette '
            f'(geometric separation) vs attribute-driven (maximise gender+age AMI). '
            f'Attribute-driven is the more gender-meaningful partition:</p>'
            f'<table class="m">{head}{body}</table>')


def build_part_b_overview(cfg: Config, out_dir: str | Path, faces_dir: str | Path,
                          out_path: str | Path | None = None, faces_per_cluster: int = 5) -> Path:
    """Static PNG: the ArcFace UMAP shown three ways — coloured by cluster (with a few sample
    faces per cluster overlaid), by predicted gender, and by predicted age — so it's visible
    that the clusters track gender + age. Axis-labelled; metrics in the cluster-panel title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from PIL import Image

    out_dir, faces_dir = Path(out_dir), Path(faces_dir)
    emb = load_embeddings("arcface", out_dir)
    ids = emb.ids
    X = preprocess(emb.vectors, list(cfg.reduce.preprocess), pca_components=cfg.reduce.pca_components)
    um = cfg.reduce.umap
    coords = umap_2d(X, um.n_neighbors, um.min_dist, um.metric, cfg.seed)
    raw = json.loads((out_dir / "arcface_attributes.json").read_text())
    gender = np.array([raw.get(i, {}).get("gender", "?") for i in ids])
    age = np.array([_age_bucket(raw.get(i, {}).get("age", 0.0)) for i in ids])
    algo = cfg.part_b.clustering.algorithms[0]
    score_fn = (attribute_score_fn(gender, age)
                if cfg.part_b.clustering.k_selection == "attribute" else None)
    res = cluster(X, algo, cfg.part_b.clustering.k_min, cfg.part_b.clustering.k_max,
                  cfg.seed, score_fn=score_fn)
    labels = np.asarray(res.labels)
    met = M.internal_metrics(X, labels)
    gp = M.external_metrics(labels, gender)["purity"]

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=130)

    def _panel(ax, values, title, faces=False):
        cats = sorted(set(values.tolist()))
        col = {c: cmap(i % 10) for i, c in enumerate(cats)}
        for c in cats:
            mm = values == c
            lab = f"{c} (n={int(mm.sum())})"
            ax.scatter(coords[mm, 0], coords[mm, 1], s=14, color=col[c], label=lab, alpha=0.7)
        if faces:
            for c in cats:
                idx = np.where(values == c)[0]
                centroid = X[idx].mean(0)
                near = idx[np.argsort(((X[idx] - centroid) ** 2).sum(1))[:faces_per_cluster]]
                for j in near:
                    p = faces_dir / f"{ids[j]}.jpg"
                    if not p.exists():
                        continue
                    im = Image.open(p).convert("RGB"); im.thumbnail((46, 46), Image.LANCZOS)
                    ab = AnnotationBbox(OffsetImage(np.asarray(im), zoom=1.0),
                                        (coords[j, 0], coords[j, 1]), frameon=True, pad=0.05,
                                        bboxprops=dict(edgecolor=col[c], lw=2))
                    ax.add_artist(ab)
        ax.legend(fontsize=8, loc="best")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_xticks([]); ax.set_yticks([])

    _panel(axes[0], labels,
           f"clusters ({algo}, k={len(set(labels.tolist()))}) — "
           f"silhouette={met['silhouette']:.3f} · gender purity={gp:.3f}", faces=True)
    _panel(axes[1], gender, "predicted gender")
    _panel(axes[2], age, "predicted age")

    fig.suptitle("Part B — ArcFace face embedding (UMAP): clusters vs gender vs age", fontsize=13)
    fig.text(0.5, 0.945,
             "ArcFace embeds the colour face crop. Same layout coloured by KMeans cluster / "
             "predicted gender / predicted age — the clusters clearly track gender (and age). "
             "Faces on the left panel are samples nearest each cluster centroid.",
             ha="center", va="top", fontsize=9, style="italic", color="#333")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = Path(out_path) if out_path else out_dir / "figures" / "part_b_overview.png"
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path
