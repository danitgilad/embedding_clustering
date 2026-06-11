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
                          encoder: str = "arcface", out_path: str | Path | None = None,
                          n_faces: int = 110) -> Path:
    """Static PNG for one face encoder: its UMAP shown by cluster / predicted gender / predicted
    age. TOP row = coloured points only (unoccluded); BOTTOM row = the same layout with a dense
    sample of face thumbnails overlaid (border = that column's category). Predicted gender/age
    are InsightFace model outputs (not a projection) — the same layout is merely recoloured."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from PIL import Image

    out_dir, faces_dir = Path(out_dir), Path(faces_dir)
    emb = load_embeddings(encoder, out_dir)
    ids = emb.ids
    X = preprocess(emb.vectors, list(cfg.reduce.preprocess), pca_components=cfg.reduce.pca_components)
    um = cfg.reduce.umap
    coords = umap_2d(X, um.n_neighbors, um.min_dist, um.metric, cfg.seed)
    # gender/age are InsightFace predictions (canonical face attributes), keyed by face id —
    # so they apply to ANY encoder's layout, including the generic-DINOv2 ablation.
    raw = json.loads((out_dir / "arcface_attributes.json").read_text())
    gender = np.array([raw.get(i, {}).get("gender", "?") for i in ids])
    age = np.array([_age_bucket(raw.get(i, {}).get("age", 0.0)) for i in ids])
    algo = cfg.part_b.clustering.algorithms[0]
    has_attr = (out_dir / f"{encoder}_attributes.json").exists()
    score_fn = (attribute_score_fn(gender, age)
                if (cfg.part_b.clustering.k_selection == "attribute" and has_attr) else None)
    res = cluster(X, algo, cfg.part_b.clustering.k_min, cfg.part_b.clustering.k_max,
                  cfg.seed, score_fn=score_fn)
    labels = np.asarray(res.labels)
    met = M.internal_metrics(X, labels)
    gp = M.external_metrics(labels, gender)["purity"]
    ksel = "attribute-k" if score_fn is not None else "silhouette-k"

    series = [
        (f"clusters ({algo}, {ksel}={len(set(labels.tolist()))}) · "
         f"sil={met['silhouette']:.3f} · gender purity={gp:.3f}", "clusters", labels),
        ("predicted gender", "predicted gender", gender),
        ("predicted age", "predicted age", age),
    ]
    stride = max(1, len(ids) // max(1, n_faces))
    face_idx = list(range(0, len(ids), stride))

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 3, figsize=(21, 13), dpi=130)
    for c_i, (title, short, values) in enumerate(series):
        cats = sorted(set(values.tolist()))
        col = {c: cmap(i % 10) for i, c in enumerate(cats)}
        ax_top, ax_bot = axes[0][c_i], axes[1][c_i]
        for c in cats:
            mm = values == c
            ax_top.scatter(coords[mm, 0], coords[mm, 1], s=14, color=col[c],
                           label=f"{c} (n={int(mm.sum())})", alpha=0.8)
        ax_top.legend(fontsize=8, loc="best")
        ax_top.set_title(title, fontsize=11)
        ax_bot.scatter(coords[:, 0], coords[:, 1], s=5, color="#dddddd", alpha=0.6)
        for j in face_idx:
            p = faces_dir / f"{ids[j]}.jpg"
            if not p.exists():
                continue
            im = Image.open(p).convert("RGB"); im.thumbnail((38, 38), Image.LANCZOS)
            ab = AnnotationBbox(OffsetImage(np.asarray(im), zoom=1.0), (coords[j, 0], coords[j, 1]),
                                frameon=True, pad=0.02, bboxprops=dict(edgecolor=col[values[j]], lw=1.4))
            ax_bot.add_artist(ab)
        ax_bot.set_title(f"{short} — with faces ({len(face_idx)} shown; border = category)", fontsize=10)
        for ax in (ax_top, ax_bot):
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Part B — {encoder} face embedding (UMAP): clusters vs gender vs age", fontsize=14)
    fig.text(0.5, 0.955,
             "Same UMAP layout recoloured three ways. Predicted gender/age are InsightFace model "
             "outputs (not axes) — colouring the fixed layout by them shows the embedding separates "
             "by gender (and age). Top = points only; bottom = a dense face sample.",
             ha="center", va="top", fontsize=9, style="italic", color="#333")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    name = "part_b_overview.png" if encoder == "arcface" else f"part_b_overview_{encoder}.png"
    out_path = Path(out_path) if out_path else out_dir / "figures" / name
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path
