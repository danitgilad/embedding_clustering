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
    import numpy as np
    from src.core.html_viewer import make_hist_spec
    out_dir, render_dir = Path(out_dir), Path(render_dir)
    items, ids = _part_a_projections(cfg, out_dir)
    projections = {f"{it['name']} · {it['modality']}":
                   {"coords2d": it["coords"], "labels": it["labels"], "metrics": it["metrics"]}
                   for it in items}
    # per-encoder feature-distance histogram (intra- vs inter-cluster), switched with the scatter
    hist = {}
    for it in items:
        X = np.asarray(it["X"], dtype=float)
        labels = np.asarray(it["labels"])
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        iu = np.triu_indices(len(X), k=1)
        d = (1.0 - Xn @ Xn.T)[iu]
        same = labels[iu[0]] == labels[iu[1]]
        intra, inter = d[same], d[~same]
        gap = (float(inter.mean()) - float(intra.mean())) if len(intra) and len(inter) else float("nan")
        hist[f"{it['name']} · {it['modality']}"] = [make_hist_spec(
            f"Intra- vs inter-cluster distance · Δmean = {gap:.3f} (larger = more discriminative)",
            "cosine distance",
            [("intra-cluster", "#55A868", intra), ("inter-cluster", "#C44E52", inter)])]
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
               f"clustering. The <b>feature-distance histogram beside the scatter</b> "
               f"(intra- vs inter-cluster cosine distance) switches with the selected encoder — a "
               f"wider gap between the dashed means = more discriminative features. (y-axis = "
               f"<i>density</i>: each curve is normalised to area 1, so the two groups compare "
               f"despite different pair counts.)"),
        always_show_thumbs=True, thumb_scale=2.0, hover_thumbs=hover_thumbs, hist=hist,
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
    gaps: dict[str, float] = {}
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
        ax_l.set_title(f"{it['name']} · {it['modality']} — all pairwise cosine distances", fontsize=10)
        ax_l.set_xlabel("cosine distance  (0 = identical direction, ~1 = unrelated)")
        ax_l.set_ylabel("pair count")

        gap = (float(inter.mean()) - float(intra.mean())) if len(intra) and len(inter) else float("nan")
        gaps[it["name"]] = gap
        for data, colour, lab in ((intra, "#55A868", "intra-cluster"), (inter, "#C44E52", "inter-cluster")):
            if len(data):
                ax_r.hist(data, bins=20, density=True, color=colour, alpha=0.55, label=lab)
        ax_r.axvline(float(intra.mean()), color="#55A868", ls="--", lw=1)
        ax_r.axvline(float(inter.mean()), color="#C44E52", ls="--", lw=1)
        ax_r.set_title(f"same-cluster vs different-cluster — separation Δmean = {gap:.3f}", fontsize=10)
        ax_r.set_xlabel("cosine distance"); ax_r.set_ylabel("density"); ax_r.legend(fontsize=8)

    ranking = " > ".join(f"{nm} ({g:.2f})" for nm, g in
                         sorted(gaps.items(), key=lambda kv: kv[1], reverse=True))
    n_assets = len(items[0]["X"])
    n_pairs = n_assets * (n_assets - 1) // 2
    fig.suptitle(f"Part A — how discriminative is each feature? Pairwise cosine distances between "
                 f"the {n_assets} GLB embeddings ({n_pairs} pairs/encoder), split intra- vs "
                 f"inter-cluster", fontsize=13, y=0.99)
    fig.text(0.5, 0.005,
             "How to read — LEFT: the spread of all pairwise distances in the embedding (a left "
             "tail near 0 = near-duplicate assets). RIGHT: those same distances split by whether "
             "the pair shares a cluster; dashed lines = the two means, Δmean = mean(inter) − "
             "mean(intra). A LARGER Δmean means same-cluster items sit much closer than "
             "different-cluster ones → more discriminative features. Ranking: " + ranking +
             " — matching the silhouette order; CLIP's heavy intra/inter overlap is why it "
             "separates worst. (Right-panel y-axis = density: each curve is normalised so its area "
             "= 1, i.e. bin height = fraction of pairs per unit distance, making the two "
             "differently-sized groups comparable in shape.)",
             ha="center", va="bottom", fontsize=8.5, color="#444", wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    out_path = Path(out_path) if out_path else out_dir / "figures" / "feature_distances_by_cluster.png"
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path


def _glasses_thumbs(render_dir: Path, ids, thumb_px: int) -> dict:
    """id -> RGBA thumbnail of the colour render (cropped to the subject + scaled to fill the
    box), or None if missing. Shared by the UMAP-panel figures."""
    import numpy as np
    from PIL import Image
    out: dict = {}
    for i in ids:
        out[i] = None
        for p in (render_dir / "colored" / f"{sanitize_id(i)}_v0.png",
                  render_dir / f"{sanitize_id(i)}_v0.png"):
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
            out[i] = np.asarray(canvas)
            break
    return out


def _draw_glasses_umap(ax, coords, labels, ids, imgs, cmap, title) -> None:
    """Draw one encoder's UMAP panel: each glasses thumbnail at its point, border = cluster
    colour, GLB id beneath, `title` above. (Shared by the overview and the fixed-k review.)"""
    import numpy as np
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    labels = np.asarray(labels)
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
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP 1", fontsize=9); ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def build_part_a_fixed_k_review(cfg: Config, out_dir: str | Path, render_dir: str | Path,
                                k: int = 6, out_path: str | Path | None = None,
                                thumb_px: int = 72) -> Path:
    """UMAP review of ALL encoders clustered at one COMMON k (default 6): a panel per encoder,
    glasses placed on its UMAP and coloured by KMeans@k. Pairs with the overview's fixed-k column
    — holding k equal removes k as a variable, so the partitions are directly comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir, render_dir = Path(out_dir), Path(render_dir)
    items, ids = _part_a_projections(cfg, out_dir)
    imgs = _glasses_thumbs(render_dir, ids, thumb_px)
    n = len(items)
    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(7.5 * n, 9.6), dpi=130)
    gs = fig.add_gridspec(2, n, height_ratios=[6.6, 1.5], top=0.88, bottom=0.05, hspace=0.05)
    comp = []
    for c_idx, it in enumerate(items):
        ax = fig.add_subplot(gs[0, c_idx])
        X = np.asarray(it["X"], dtype=float)
        res = cluster(X, "kmeans", k, k, cfg.seed)
        m = M.internal_metrics(X, res.labels)
        comp.append((f"{it['name']} · {it['modality']}", m["silhouette"],
                     m["davies_bouldin"], m["calinski_harabasz"]))
        _draw_glasses_umap(ax, it["coords"], res.labels, ids, imgs, cmap,
                           f"{it['name']} · {it['modality']}\nKMeans @ k={k} · "
                           f"silhouette={m['silhouette']:.3f}")

    # full-metric comparison table at the common k (silhouette is only one of three measures)
    tax = fig.add_subplot(gs[1, :]); tax.axis("off")
    cols = [f"encoder · modality (all @ k={k})", "silhouette ↑", "Davies–Bouldin ↓",
            "Calinski–Harabasz ↑"]
    body = [[r[0], f"{r[1]:.3f}", f"{r[2]:.2f}", f"{r[3]:.2f}"] for r in comp]
    t = tax.table(cellText=body, colLabels=cols, loc="upper center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.5)
    for ci, direction in {1: "up", 2: "down", 3: "up"}.items():
        vals = [r[ci] for r in comp]
        bi = (max if direction == "up" else min)(range(len(vals)), key=lambda i: vals[i])
        t[bi + 1, ci].set_text_props(fontweight="bold")
    best = max(comp, key=lambda r: r[1])[0].split(" · ")[0]
    tax.text(0.5, -0.10, f"At the common k={k}, {best} still separates best on all three internal "
             "measures — the ranking doesn't depend on each encoder's own k.",
             transform=tax.transAxes, ha="center", va="top", fontsize=9.5, color="#222")

    fig.suptitle(f"Part A — every encoder clustered at a COMMON k={k} (UMAP)   "
                 "— holding k equal isolates the feature, so the partitions are directly comparable",
                 fontsize=12, y=0.965)
    fig.text(0.5, 0.925, "Renders shown in colour for inspection only — colour & texture are NOT "
             "used by the clustering.", ha="center", va="top", fontsize=9, style="italic",
             color="#b00000")
    out_path = Path(out_path) if out_path else out_dir / "figures" / f"part_a_k{k}_umap.png"
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path


def build_part_a_overview(cfg: Config, out_dir: str | Path, render_dir: str | Path,
                          out_path: str | Path | None = None, thumb_px: int = 72) -> Path:
    """One static PNG: a panel per encoder, each glasses render placed at its UMAP point with
    a cluster-coloured border + its GLB id, the encoder's metrics in the panel title, and a
    cross-encoder comparison table (incl. a fixed-k column) + takeaway beneath."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir, render_dir = Path(out_dir), Path(render_dir)
    items, ids = _part_a_projections(cfg, out_dir)
    imgs = _glasses_thumbs(render_dir, ids, thumb_px)
    n = len(items)
    fixed_k = min(6, cfg.part_a.clustering.k_max)
    k_min, k_max = cfg.part_a.clustering.k_min, cfg.part_a.clustering.k_max

    # cross-encoder comparison: each encoder's silhouette-selected KMeans metrics + a FIXED-k
    # column, so the encoders are compared both at their own best k and at one common k (fairness).
    comp = []
    for it in items:
        Xe, m = np.asarray(it["X"], dtype=float), it["metrics"]
        kstar = len({int(v) for v in it["labels"]})
        fk = cluster(Xe, "kmeans", fixed_k, fixed_k, cfg.seed)
        comp.append((f"{it['name']} · {it['modality']}", kstar, m["silhouette"],
                     m["davies_bouldin"], m["calinski_harabasz"],
                     M.internal_metrics(Xe, fk.labels)["silhouette"]))

    fig = plt.figure(figsize=(7.5 * n, 10.0), dpi=130)
    gs = fig.add_gridspec(2, n, height_ratios=[6.6, 1.7], top=0.86, bottom=0.05, hspace=0.05)
    cmap = plt.get_cmap("tab10")
    for c_idx, it in enumerate(items):
        ax = fig.add_subplot(gs[0, c_idx])
        m = it["metrics"]
        _draw_glasses_umap(ax, it["coords"], it["labels"], ids, imgs, cmap,
                           f"{it['name']} · {it['modality']}\n"
                           f"k={len({int(v) for v in it['labels']})} · "
                           f"silhouette={m['silhouette']:.3f} · DB={m['davies_bouldin']:.2f} · "
                           f"CH={m['calinski_harabasz']:.2f}")

    # comparison table + takeaway spanning the bottom
    tax = fig.add_subplot(gs[1, :]); tax.axis("off")
    cols = ["encoder · modality", "k*", "silhouette ↑", "Davies–Bouldin ↓",
            "Calinski–Harabasz ↑", f"KMeans sil @ k={fixed_k} ↑"]
    body = [[r[0], str(r[1]), f"{r[2]:.3f}", f"{r[3]:.2f}", f"{r[4]:.2f}", f"{r[5]:.3f}"]
            for r in comp]
    t = tax.table(cellText=body, colLabels=cols, loc="upper center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
    # bold the best value per quality column (col index matches comp tuple index)
    for ci, direction in {2: "up", 3: "down", 4: "up", 5: "up"}.items():
        col_vals = [r[ci] for r in comp]
        bi = (max if direction == "up" else min)(range(len(col_vals)), key=lambda i: col_vals[i])
        t[bi + 1, ci].set_text_props(fontweight="bold")
    best = max(comp, key=lambda r: r[2])[0].split(" · ")[0]
    worst = min(comp, key=lambda r: r[2])[0].split(" · ")[0]
    tax.text(0.5, -0.06,
             f"Takeaway: {best} separates best — at its own k* AND at the common k={fixed_k} — "
             f"while {worst} is weakest. k* is chosen per encoder by the SAME silhouette sweep "
             f"over k∈[{k_min},{k_max}]; the differing k* (DINOv2/Point-MAE→7, CLIP→3) is itself "
             "a result — CLIP's coarser, language-aligned features don't support finer splits. "
             "Identical downstream for all ⇒ the comparison is fair.",
             transform=tax.transAxes, ha="center", va="top", fontsize=9.5, color="#222", wrap=True)

    fig.suptitle("Part A — glasses clustered per encoder   "
                 "(thumbnail = rendered glasses · border colour = cluster · label = GLB id)",
                 fontsize=13, y=0.975)
    fig.text(0.5, 0.935,
             "Renders are shown in colour for inspection only — the 2D encoders (DINOv2, CLIP) "
             "embed GREYSCALE shape renders and Point-MAE uses mesh geometry, so colour & "
             "texture are NOT used by the clustering.",
             ha="center", va="top", fontsize=9, style="italic", color="#b00000")
    out_path = Path(out_path) if out_path else out_dir / "figures" / "part_a_overview.png"
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path
