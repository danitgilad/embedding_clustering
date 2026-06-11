"""Build the Part A visualisations from cached embeddings (decoupled from encoding).

Two artifacts, both reading outputs/part_a/<encoder>.npy and recomputing UMAP + KMeans +
internal metrics deterministically (reusing core.*):
  - build_part_a_viewer: the interactive self-contained HTML (Plotly).
  - build_part_a_overview: a single static PNG with one panel per encoder — each glasses
    render placed at its UMAP point, border colour = cluster, label = GLB id, metrics in the
    title. This is the "match a glasses to its cluster at a glance" figure for the report.
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
from src.utils.io import ensure_dir, sanitize_id

log = logging.getLogger(__name__)


def _part_a_projections(cfg: Config, out_dir: Path) -> tuple[list[dict], list[str]]:
    """Per-encoder {name, modality, coords, labels, metrics} from cached .npy + the shared ids.

    2D (render) vs 3D (mesh) modality comes from cfg.part_a.encoders_2d/encoders_3d.
    """
    npys = sorted(out_dir.glob("*.npy"))
    if not npys:
        raise FileNotFoundError(f"no embeddings (*.npy) in {out_dir}")
    modality = {n: "2D · render" for n in cfg.part_a.encoders_2d}
    modality.update({n: "3D · mesh" for n in cfg.part_a.encoders_3d})
    um = cfg.reduce.umap
    algo = cfg.part_a.clustering.algorithms[0]
    items: list[dict] = []
    ids: list[str] | None = None
    for npy in npys:
        emb = load_embeddings(npy.stem, out_dir)
        if ids is None:
            ids = emb.ids
        elif emb.ids != ids:
            raise ValueError(f"id mismatch for {npy.stem}; viewer needs a shared id order")
        X = preprocess(emb.vectors, list(cfg.reduce.preprocess), pca_components=cfg.reduce.pca_components)
        coords = umap_2d(X, um.n_neighbors, um.min_dist, um.metric, cfg.seed)
        res = cluster(X, algo, cfg.part_a.clustering.k_min, cfg.part_a.clustering.k_max, cfg.seed)
        items.append({"name": npy.stem, "modality": modality.get(npy.stem, "?"),
                      "coords": coords, "labels": res.labels, "X": X,
                      "metrics": M.internal_metrics(X, res.labels)})
    return items, ids


def build_part_a_viewer(cfg: Config, out_dir: str | Path, render_dir: str | Path) -> Path:
    """Assemble outputs/part_a/viewer.html from every <encoder>.npy in out_dir."""
    out_dir, render_dir = Path(out_dir), Path(render_dir)
    items, ids = _part_a_projections(cfg, out_dir)
    projections = {f"{it['name']} · {it['modality']}":
                   {"coords2d": it["coords"], "labels": it["labels"], "metrics": it["metrics"]}
                   for it in items}
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
               f"switch between them to compare. <b>Note:</b> the renders are shown in colour "
               f"for inspection, but the 2D encoders embed <b>greyscale</b> shape renders and "
               f"Point-MAE uses mesh geometry — <b>colour &amp; texture are not used</b> by the "
               f"clustering."),
        always_show_thumbs=True, thumb_scale=2.0, hover_thumbs=hover_thumbs,
        page_title="Part A — Glasses Cluster Viewer")
    out_html = out_dir / "viewer.html"
    out_html.write_text(html)
    log.info("Wrote %s", out_html)
    return out_html


def build_feature_distribution_figure(cfg: Config, out_dir: str | Path,
                                      out_path: str | Path | None = None) -> Path:
    """One static PNG analysing each encoder's *feature distribution* and *discriminative power*.

    Per encoder (row): (left) histogram of all pairwise cosine distances — the spread of the
    embedding space; (right) the same distances split into intra-cluster vs inter-cluster, whose
    gap (mean_inter − mean_intra) is a label-free measure of how separable the features are. This
    is the "explore distributions and discriminative properties / compare expressiveness" view —
    complementary to the clustering metrics, working on the embeddings directly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = Path(out_dir)
    items, _ = _part_a_projections(cfg, out_dir)
    n = len(items)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.4 * n), dpi=130, squeeze=False)
    for row, it in zip(axes, items):
        X = np.asarray(it["X"], dtype=float)
        labels = np.asarray(it["labels"])
        # cosine distance on the (preprocess already l2-normalised) embeddings
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        dist = 1.0 - Xn @ Xn.T
        iu = np.triu_indices(len(X), k=1)
        alld = dist[iu]
        same = labels[iu[0]] == labels[iu[1]]
        intra, inter = alld[same], alld[~same]

        ax_l, ax_r = row
        ax_l.hist(alld, bins=24, color="#4C72B0", alpha=0.85)
        ax_l.set_title(f"{it['name']} · {it['modality']} — pairwise cosine distances", fontsize=10)
        ax_l.set_xlabel("cosine distance"); ax_l.set_ylabel("pair count")

        gap = (float(inter.mean()) - float(intra.mean())) if len(intra) and len(inter) else float("nan")
        for data, colour, lab in ((intra, "#55A868", "intra-cluster"), (inter, "#C44E52", "inter-cluster")):
            if len(data):
                ax_r.hist(data, bins=20, density=True, color=colour, alpha=0.55, label=lab)
        ax_r.set_title(f"intra vs inter — separation (Δmean) = {gap:.3f}", fontsize=10)
        ax_r.set_xlabel("cosine distance"); ax_r.set_ylabel("density"); ax_r.legend(fontsize=8)

    fig.suptitle("Part A — feature-distribution & discriminability analysis "
                 "(larger intra↔inter gap = more discriminative features)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = Path(out_path) if out_path else out_dir / "figures" / "feature_distributions.png"
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path


def build_part_a_overview(cfg: Config, out_dir: str | Path, render_dir: str | Path,
                          out_path: str | Path | None = None, thumb_px: int = 72) -> Path:
    """One static PNG: a panel per encoder, each glasses render placed at its UMAP point with
    a cluster-coloured border + its GLB id, and the encoder's metrics in the panel title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from PIL import Image

    out_dir, render_dir = Path(out_dir), Path(render_dir)
    items, ids = _part_a_projections(cfg, out_dir)
    colored = render_dir / "colored"

    def thumb(i: str) -> "np.ndarray | None":
        for p in (colored / f"{sanitize_id(i)}_v0.png", render_dir / f"{sanitize_id(i)}_v0.png"):
            if not p.exists():
                continue
            im = Image.open(p).convert("RGBA")
            bbox = im.getbbox()          # drop the transparent margin so the glasses are big
            if bbox:
                im = im.crop(bbox)
            w, h = im.size               # scale the cropped glasses to fill a fixed square box
            scale = (thumb_px * 0.95) / max(w, h)
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
            canvas = Image.new("RGBA", (thumb_px, thumb_px), (0, 0, 0, 0))
            canvas.paste(im, ((thumb_px - im.size[0]) // 2, (thumb_px - im.size[1]) // 2))
            return np.asarray(canvas)
        return None

    imgs = {i: thumb(i) for i in ids}
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 8.2), dpi=130, squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, it in zip(axes[0], items):
        coords, labels = it["coords"], it["labels"]
        uniq = sorted({int(v) for v in labels})
        col = {c: cmap(k % 10) for k, c in enumerate(uniq)}
        xs, ys = coords[:, 0], coords[:, 1]
        padx = (float(xs.max() - xs.min()) or 1.0) * 0.18
        pady = (float(ys.max() - ys.min()) or 1.0) * 0.20
        ax.set_xlim(xs.min() - padx, xs.max() + padx)
        ax.set_ylim(ys.min() - pady, ys.max() + pady)
        for j, i in enumerate(ids):
            c = col[int(labels[j])]
            im = imgs.get(i)
            if im is not None:
                ab = AnnotationBbox(OffsetImage(im, zoom=1.0), (xs[j], ys[j]), frameon=True,
                                    pad=0.1, bboxprops=dict(edgecolor=c, lw=2.5))
                ab.set_clip_on(False)
                ax.add_artist(ab)
            else:
                ax.scatter([xs[j]], [ys[j]], color=c, s=80)
            ax.annotate(i, (xs[j], ys[j]), textcoords="offset points", xytext=(0, -30),
                        ha="center", va="top", fontsize=6.5, color="#222",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        m = it["metrics"]
        ax.set_title(f"{it['name']} · {it['modality']}\n"
                     f"k={len(uniq)} · silhouette={m['silhouette']:.3f} · "
                     f"DB={m['davies_bouldin']:.2f} · CH={m['calinski_harabasz']:.2f}",
                     fontsize=11)
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Part A — glasses clustered per encoder   "
                 "(thumbnail = rendered glasses · border colour = cluster · label = GLB id)",
                 fontsize=13)
    fig.text(0.5, 0.945,
             "Renders are shown in colour for inspection only — the 2D encoders (DINOv2, CLIP) "
             "embed GREYSCALE shape renders and Point-MAE uses mesh geometry, so colour & "
             "texture are NOT used by the clustering.",
             ha="center", va="top", fontsize=9, style="italic", color="#b00000")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = Path(out_path) if out_path else out_dir / "figures" / "part_a_overview.png"
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path
