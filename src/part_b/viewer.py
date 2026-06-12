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
from src.core.html_viewer import build_viewer_html, image_to_data_uri, make_hist_spec
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
    hist: dict[str, dict] = {}
    hover_meta: dict[str, dict] = {}
    um = cfg.reduce.umap
    algo = cfg.part_b.clustering.algorithms[0]
    k_min, k_max = cfg.part_b.clustering.k_min, cfg.part_b.clustering.k_max
    attr_ctx = None   # (X, gender, age) of the first encoder with attributes, for the k-table
    # shared gender/age labels (InsightFace, keyed by face id) → split the distance histograms
    shared_attr = out_dir / "arcface_attributes.json"
    raw_shared = json.loads(shared_attr.read_text()) if shared_attr.exists() else {}
    gender_by_id = {i: a.get("gender", "?") for i, a in raw_shared.items()}
    agebucket_by_id = {i: _age_bucket(a.get("age", 0.0)) for i, a in raw_shared.items()}
    for npy in npys:
        emb = load_embeddings(npy.stem, out_dir)
        if ids is None:
            ids = emb.ids
        elif emb.ids != ids:
            raise ValueError(f"id mismatch for {npy.stem}; viewer needs a shared id order")
        X = preprocess(emb.vectors, list(cfg.reduce.preprocess), pca_components=cfg.reduce.pca_components)
        coords = umap_2d(X, um.n_neighbors, um.min_dist, um.metric, cfg.seed)
        # feature-distance histograms for this encoder (independent of k-selection): split by
        # gender, then by age bucket — same two attributes as feature_distributions.png
        enc_hist = None
        if gender_by_id:
            g = np.array([gender_by_id.get(i, "?") for i in emb.ids])
            ab = np.array([agebucket_by_id.get(i, "?") for i in emb.ids])
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            iu = np.triu_indices(len(X), k=1)
            d = (1.0 - Xn @ Xn.T)[iu]
            sg, sa = g[iu[0]] == g[iu[1]], ab[iu[0]] == ab[iu[1]]
            gap_g = float(d[~sg].mean()) - float(d[sg].mean())
            gap_a = float(d[~sa].mean()) - float(d[sa].mean())
            enc_hist = [
                make_hist_spec(
                    f"Same vs different gender · Δmean = {gap_g:.3f} (larger = separates gender more)",
                    "cosine distance",
                    [("same gender", "#55A868", d[sg]), ("different gender", "#C44E52", d[~sg])]),
                make_hist_spec(
                    f"Same vs different age bucket · Δmean = {gap_a:.3f} (larger = separates age more)",
                    "cosine distance",
                    [("same age", "#55A868", d[sa]), ("different age", "#C44E52", d[~sa])]),
            ]
        keys_before = set(projections)
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
        if enc_hist is not None:
            for key in set(projections) - keys_before:
                hist[key] = enc_hist
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
               "the generic-DINOv2 encoder — toggle to compare the clusterings on one layout. "
               "Two feature-distance histograms beside the scatter (same vs different gender, "
               "and same vs different age bucket) switch with the encoder — a wider gap between "
               "the dashed means = the embedding separates that attribute more. (y-axis = "
               "density: each curve normalised to area 1, so the groups compare despite "
               "different counts.)"),
        always_show_thumbs=False, extra_html=extra_html, hist=hist,
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


def build_part_b_feature_distribution(cfg: Config, out_dir: str | Path,
                                      out_path: str | Path | None = None) -> Path:
    """Per encoder: pairwise cosine-distance distribution split by ATTRIBUTE (same vs different
    gender, and same vs different age bucket). Δmean = mean(different) − mean(same) is a
    label-free read on whether the embedding encodes that attribute — positive ⇒ same-attribute
    faces sit closer. The Part B analogue of Part A's feature_distributions, but split by the
    attributes we care about (the manifold is continuous, so a cluster split would just restate
    the low silhouette)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    raw = json.loads((out_dir / "arcface_attributes.json").read_text())
    encoders = [e for e in cfg.part_b.encoders if (out_dir / f"{e}.npy").exists()]
    summary: dict[str, tuple[float, float]] = {}
    fig, axes = plt.subplots(len(encoders), 3, figsize=(15, 3.7 * len(encoders)),
                             dpi=130, squeeze=False)

    def _split(ax, dists, same, attr) -> float:
        mean_same, mean_diff = float(dists[same].mean()), float(dists[~same].mean())
        for data, colour, lab in ((dists[same], "#55A868", f"same {attr}"),
                                  (dists[~same], "#C44E52", f"different {attr}")):
            ax.hist(data, bins=40, density=True, color=colour, alpha=0.55, label=lab)
        ax.axvline(mean_same, color="#55A868", ls="--", lw=1)
        ax.axvline(mean_diff, color="#C44E52", ls="--", lw=1)
        ax.set_title(f"by {attr} — Δmean = {mean_diff - mean_same:.3f}", fontsize=10)
        ax.set_xlabel("cosine distance"); ax.set_ylabel("density"); ax.legend(fontsize=8)
        return mean_diff - mean_same

    for row, enc in zip(axes, encoders):
        emb = load_embeddings(enc, out_dir)
        X = preprocess(emb.vectors, list(cfg.reduce.preprocess), pca_components=cfg.reduce.pca_components)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        iu = np.triu_indices(len(X), k=1)
        d = (1.0 - Xn @ Xn.T)[iu]
        gender = np.array([raw.get(i, {}).get("gender", "?") for i in emb.ids])
        age = np.array([_age_bucket(raw.get(i, {}).get("age", 0.0)) for i in emb.ids])
        row[0].hist(d, bins=40, color="#4C72B0", alpha=0.85)
        row[0].set_title(f"{enc} — all {len(d):,} pairwise cosine distances", fontsize=10)
        row[0].set_xlabel("cosine distance"); row[0].set_ylabel("pair count")
        summary[enc] = (_split(row[1], d, gender[iu[0]] == gender[iu[1]], "gender"),
                        _split(row[2], d, age[iu[0]] == age[iu[1]], "age bucket"))

    rank = " · ".join(f"{e}: gender Δ={g:.2f}, age Δ={a:.2f}" for e, (g, a) in summary.items())
    fig.suptitle("Part B — feature-distribution by attribute (does the embedding encode gender / age?)",
                 fontsize=14, y=0.99)
    fig.text(0.5, 0.005,
             "All pairwise cosine distances per encoder, split by whether the two faces share a "
             "GENDER (middle) or AGE bucket (right); dashed = means, Δmean = mean(different) − "
             "mean(same). Positive Δmean ⇒ same-attribute faces sit closer ⇒ the embedding encodes "
             "that attribute. " + rank + ". Both separate gender more than age — matching the "
             "cluster-purity results; a cluster split would be near-flat (the manifold is "
             "continuous). y-axis = density: each curve normalised so its area = 1 (bin height = "
             "fraction of pairs per unit distance), so the two groups are comparable in shape.",
             ha="center", va="bottom", fontsize=8.5, color="#444", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out_path = Path(out_path) if out_path else out_dir / "figures" / "feature_distributions.png"
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path


def build_part_b_overview(cfg: Config, out_dir: str | Path, faces_dir: str | Path,
                          encoder: str = "arcface", out_path: str | Path | None = None,
                          n_faces: int = 110, k_selection: str | None = None) -> Path:
    """Static PNG for one face encoder: its UMAP shown by cluster / predicted gender / predicted
    age. TOP row = coloured points only (unoccluded); BOTTOM row = the same layout with a dense
    sample of face thumbnails overlaid (border = that column's category). Predicted gender/age
    are InsightFace model outputs (not a projection) — the same layout is merely recoloured.

    `k_selection` ("attribute" → AMI-driven k, "silhouette" → geometric k) overrides the configured
    default, letting both partitions of the same encoder be emitted as separate figures (e.g.
    ArcFace attribute-k=3 vs silhouette-k=6)."""
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
    sel = k_selection or cfg.part_b.clustering.k_selection
    score_fn = (attribute_score_fn(gender, age) if (sel == "attribute" and has_attr) else None)
    res = cluster(X, algo, cfg.part_b.clustering.k_min, cfg.part_b.clustering.k_max,
                  cfg.seed, score_fn=score_fn)
    labels = np.asarray(res.labels)
    met = M.internal_metrics(X, labels)
    eg, ea = M.external_metrics(labels, gender), M.external_metrics(labels, age)
    k_found = len(set(labels.tolist()))
    sel_used = "attribute" if score_fn is not None else "silhouette"
    ksel = f"{sel_used}-k"

    series = [
        (f"clusters ({algo}, {ksel}={k_found}) · "
         f"sil={met['silhouette']:.3f} · gender purity={eg['purity']:.3f}", "clusters", labels),
        ("predicted gender", "predicted gender", gender),
        ("predicted age", "predicted age", age),
    ]
    stride = max(1, len(ids) // max(1, n_faces))
    face_idx = list(range(0, len(ids), stride))

    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(21, 14.5), dpi=130)
    gs = fig.add_gridspec(3, 3, height_ratios=[6, 6, 1.0], top=0.92, bottom=0.04, hspace=0.12)
    for c_i, (title, short, values) in enumerate(series):
        cats = sorted(set(values.tolist()))
        col = {c: cmap(i % 10) for i, c in enumerate(cats)}
        ax_top, ax_bot = fig.add_subplot(gs[0, c_i]), fig.add_subplot(gs[1, c_i])
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

    # metrics summary table spanning the bottom row
    tax = fig.add_subplot(gs[2, :]); tax.axis("off")
    cols = ["k", "silhouette ↑", "Davies–Bouldin ↓", "Calinski–Harabasz ↑",
            "gender purity ↑", "gender NMI ↑", "age purity ↑", "age NMI ↑"]
    body = [[str(k_found), f"{met['silhouette']:.3f}", f"{met['davies_bouldin']:.2f}",
             f"{met['calinski_harabasz']:.1f}", f"{eg['purity']:.3f}", f"{eg['nmi']:.3f}",
             f"{ea['purity']:.3f}", f"{ea['nmi']:.3f}"]]
    t = tax.table(cellText=body, colLabels=cols, loc="center", cellLoc="center",
                  rowLabels=[f"{encoder} · {ksel}"])
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.6)
    tax.text(0.5, -0.35, "↑ higher better · ↓ lower better.  gender/age purity & NMI score the "
             "clustering against InsightFace's predicted labels; low silhouette is expected on a "
             "continuous manifold.", transform=tax.transAxes, ha="center", va="top",
             fontsize=8.5, color="#555")

    sel_label = ("attribute-driven k-selection (maximise gender+age AMI)" if score_fn is not None
                 else "silhouette k-selection (geometric separation)")
    fig.suptitle(f"Part B — {encoder} · {sel_label} → k={k_found}   "
                 "(same UMAP recoloured by cluster / gender / age)", fontsize=14, y=0.975)
    fig.text(0.5, 0.945,
             "Same UMAP layout recoloured three ways. Predicted gender/age are InsightFace model "
             "outputs (not axes) — colouring the fixed layout by them shows the embedding separates "
             "by gender (and age). Top = points only; bottom = a dense face sample.",
             ha="center", va="top", fontsize=9, style="italic", color="#333")
    # Filename encodes BOTH k and the k-selection so the variants are unmistakable, e.g.
    # part_b_overview_arcface_k_3_attribute.png vs part_b_overview_arcface_k_6_silhouette.png.
    name = f"part_b_overview_{encoder}_k_{k_found}_{sel_used}.png"
    out_path = Path(out_path) if out_path else out_dir / "figures" / name
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path
