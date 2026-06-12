"""Visualization helpers — all save to image files (the assignment requires saved images).

Functions are part-agnostic: they take arrays/paths and write a PNG, returning its path.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# Whether higher ("up") or lower ("down") is better, for annotating metric tables.
_BETTER = {"silhouette": "up", "davies_bouldin": "down", "calinski_harabasz": "up"}


def _metric_direction(metric: str) -> str | None:
    """'up' if higher is better, 'down' if lower, None if not a quality score."""
    if metric in _BETTER:
        return _BETTER[metric]
    if metric.endswith(("purity", "nmi", "ari")):
        return "up"
    return None


def scatter_2d(points: np.ndarray, labels: np.ndarray, out_path: str | Path,
               title: str = "", point_ids: Sequence[str] | None = None,
               note: str = "") -> Path:
    """Scatter 2D UMAP points coloured by cluster, with a DISCRETE legend (not a colorbar).

    HDBSCAN noise (label -1) is drawn grey and labelled "noise". If `point_ids` is given and
    the set is small (<=30, i.e. Part A), each point is annotated with its id. `note` adds a
    wrapped explanatory caption beneath the plot (e.g. why HDBSCAN returned k=0).
    """
    out_path = Path(out_path)
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(6.8, 5.8), dpi=120)
    cmap = plt.get_cmap("tab10")
    for k, c in enumerate(sorted({int(v) for v in labels})):
        m = labels == c
        col = (0.7, 0.7, 0.7) if c == -1 else cmap(k % 10)
        lbl = f"noise (n={int(m.sum())})" if c == -1 else f"cluster {c} (n={int(m.sum())})"
        ax.scatter(points[m, 0], points[m, 1], color=col, s=42, alpha=0.85, label=lbl)
    if point_ids is not None and len(points) <= 30:
        for (x, y), i in zip(points, point_ids):
            ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=6, color="#333")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, -0.08, "UMAP projection — for display only; clustering runs on the "
            "full-dimensional embeddings", transform=ax.transAxes, ha="center",
            fontsize=7, color="#777")
    if note:
        ax.text(0.5, -0.15, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color="#444", wrap=True)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


def algorithm_comparison_png(coords: np.ndarray, algo_labels: Mapping[str, np.ndarray],
                             metrics: Mapping[str, Mapping[str, float]], out_path: str | Path,
                             title: str = "", notes: Mapping[str, str] | None = None) -> Path:
    """One figure comparing clustering algorithms on a shared UMAP: a scatter per algorithm
    (same layout, coloured by that algorithm's clusters, discrete legend) on top, and a metrics
    table (rows = algorithms, best per column bold) beneath. Replaces the separate per-algorithm
    `_<algo>_umap.png` scatters and the standalone `_metrics.png`."""
    out_path = Path(out_path)
    coords = np.asarray(coords, dtype=float)
    notes = notes or {}
    algos = list(algo_labels)
    n = len(algos)
    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(5.4 * n, 6.4), dpi=120)
    gs = fig.add_gridspec(2, n, height_ratios=[5, 1.7], hspace=0.32, top=0.9, bottom=0.06)
    for j, algo in enumerate(algos):
        ax = fig.add_subplot(gs[0, j])
        labels = np.asarray(algo_labels[algo])
        for k, c in enumerate(sorted({int(v) for v in labels})):
            mm = labels == c
            col = (0.7, 0.7, 0.7) if c == -1 else cmap(k % 10)
            lbl = f"noise (n={int(mm.sum())})" if c == -1 else f"cluster {c} (n={int(mm.sum())})"
            ax.scatter(coords[mm, 0], coords[mm, 1], color=col, s=16, alpha=0.85, label=lbl)
        n_clusters = len(set(int(v) for v in labels) - {-1})
        ax.set_title(f"{algo} (k={n_clusters})", fontsize=11)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, loc="best", framealpha=0.9)
        if algo in notes:
            ax.text(0.5, -0.12, notes[algo], transform=ax.transAxes, ha="center", va="top",
                    fontsize=7.5, color="#444", wrap=True)
    tax = fig.add_subplot(gs[1, :]); tax.axis("off")
    keys = sorted({k for r in metrics.values() for k in r})
    arrow = {"up": " ↑", "down": " ↓"}
    col_labels = [k + arrow.get(_metric_direction(k) or "", "") for k in keys]
    body = [[("nan" if not np.isfinite(metrics[a].get(k, float("nan"))) else f"{metrics[a][k]:.3f}")
             for k in keys] for a in algos]
    t = tax.table(cellText=body, rowLabels=algos, colLabels=col_labels, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.45)
    for ci, k in enumerate(keys):
        d = _metric_direction(k)
        finite = [(i, metrics[a][k]) for i, a in enumerate(algos)
                  if isinstance(metrics[a].get(k), (int, float)) and np.isfinite(metrics[a].get(k))]
        if d and finite:
            bi = (max if d == "up" else min)(finite, key=lambda t: t[1])[0]
            t[bi + 1, ci].set_text_props(fontweight="bold")
    tax.text(0.5, -0.04, "↑ higher better · ↓ lower better · best per column bold.  silhouette "
             "needs ≥2 clusters (nan for HDBSCAN noise-only).", transform=tax.transAxes,
             ha="center", va="top", fontsize=7.5, color="#555")
    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


def metric_table_png(rows: Mapping[str, Mapping[str, float]], out_path: str | Path,
                     title: str = "") -> Path:
    """Render a {row_name: {metric: value}} table as a PNG, with good/bad context.

    Column headers get a ↑/↓ arrow (which direction is better) and the best value per metric
    is bold. A caption explains the ranges so the reader can judge "is this good?".
    """
    out_path = Path(out_path)
    metrics = sorted({m for r in rows.values() for m in r})
    row_names = list(rows)
    arrow = {"up": " ↑", "down": " ↓"}
    col_labels = [m + arrow.get(_metric_direction(m) or "", "") for m in metrics]

    # best value per metric column (max if higher-is-better, min if lower) -> bold that cell
    best_row: dict[str, int] = {}
    for m in metrics:
        vals = [rows[r].get(m, float("nan")) for r in row_names]
        finite = [(i, v) for i, v in enumerate(vals) if np.isfinite(v)]
        if finite and _metric_direction(m):
            best = (max if _metric_direction(m) == "up" else min)(finite, key=lambda t: t[1])
            best_row[m] = best[0]

    cell_text = [[("nan" if not np.isfinite(rows[r].get(m, float("nan")))
                   else f"{rows[r][m]:.3f}") for m in metrics] for r in row_names]
    fig, ax = plt.subplots(figsize=(2.5 + 1.7 * len(metrics), 1.4 + 0.5 * len(row_names)), dpi=120)
    ax.axis("off"); ax.set_title(title, pad=12)
    tbl = ax.table(cellText=cell_text, rowLabels=row_names, colLabels=col_labels, loc="center")
    tbl.scale(1, 1.5)
    for ci, m in enumerate(metrics):                      # bold the best cell per column
        if m in best_row:
            tbl[best_row[m] + 1, ci].set_text_props(fontweight="bold")
    ax.text(0.5, -0.02,
            "↑ higher is better · ↓ lower is better · best per column in bold.  "
            "silhouette ∈[−1,1] (cohesion vs separation) · Davies–Bouldin ≥0 (0 ideal) · "
            "Calinski–Harabasz (variance ratio, ↑) · purity/NMI/ARI ∈[0,1] vs pseudo-labels.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#555", wrap=True)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


def _subject_thumb(path: Path, thumb_px: int, crop: bool) -> np.ndarray:
    """Load an image as an RGBA thumbnail; if `crop`, drop the transparent margin and scale
    the subject to fill the square (so small renders aren't lost in whitespace)."""
    im = Image.open(path).convert("RGBA")
    if crop:
        bbox = im.getbbox()
        if bbox:
            im = im.crop(bbox)
        w, h = im.size
        scale = (thumb_px * 0.97) / max(w, h)
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        canvas = Image.new("RGBA", (thumb_px, thumb_px), (255, 255, 255, 0))
        canvas.paste(im, ((thumb_px - im.size[0]) // 2, (thumb_px - im.size[1]) // 2), im)
        im = canvas
    else:
        im = im.resize((thumb_px, thumb_px))
    return np.asarray(im)


def cluster_montage(image_paths: Sequence[str | Path], labels: np.ndarray,
                    out_path: str | Path, thumb_px: int = 120, max_per_cluster: int = 12,
                    row_titles: dict[int, str] | None = None, caption: str = "",
                    ids: Sequence[str] | None = None, crop: bool = False,
                    summary: bool = False) -> Path:
    """Grid of thumbnails grouped by cluster — one row per cluster, with a coloured header
    band ("Cluster k · n=…") matching the scatter/overview colours.

    `ids` (aligned with image_paths) annotates each thumbnail with its id; `crop` fills the
    cell with the subject (Part A renders have transparent margins); `summary=True` adds a
    member table below. HDBSCAN noise (label -1) is skipped. `row_titles` overrides headers.
    """
    out_path = Path(out_path)
    labels = np.asarray(labels)
    ids = list(ids) if ids is not None else [None] * len(image_paths)

    by_cluster: dict[int, list[tuple[Path, object]]] = {}
    for p, lab, i in zip(image_paths, labels, ids):
        if int(lab) == -1:
            continue
        by_cluster.setdefault(int(lab), []).append((Path(p), i))
    clusters = sorted(by_cluster)
    sizes = {c: len(v) for c, v in by_cluster.items()}
    ncols = max(min(len(v), max_per_cluster) for v in by_cluster.values()) + 1  # +1 header col
    cmap = plt.get_cmap("tab10")

    nrows = len(clusters)
    fig_h = nrows * (thumb_px / 70) + (nrows * 0.28 + 0.8 if summary else 0.6)
    fig = plt.figure(figsize=(ncols * (thumb_px / 78), fig_h), dpi=120)
    if summary:
        outer = fig.add_gridspec(2, 1, height_ratios=[nrows, max(1.0, nrows * 0.42)], hspace=0.18)
        grid = outer[0].subgridspec(nrows, ncols, wspace=0.05, hspace=0.32)
        table_ax = fig.add_subplot(outer[1]); table_ax.axis("off")
    else:
        grid = fig.add_gridspec(nrows, ncols, wspace=0.05, hspace=0.32)
        table_ax = None

    for r, c_lab in enumerate(clusters):
        colour = cmap(r % 10)
        head = fig.add_subplot(grid[r, 0]); head.axis("off")
        text = (row_titles or {}).get(c_lab) or f"Cluster {c_lab}\nn = {sizes[c_lab]}"
        head.text(0.5, 0.5, text.replace(" · ", "\n"), ha="center", va="center", fontsize=8.5,
                  color="white", bbox=dict(boxstyle="round,pad=0.4", fc=colour, ec="none"))
        for col in range(1, ncols):
            ax = fig.add_subplot(grid[r, col]); ax.axis("off")
            idx = col - 1
            if idx < len(by_cluster[c_lab]):
                path, iid = by_cluster[c_lab][idx]
                ax.imshow(_subject_thumb(path, thumb_px, crop))
                for spine in ax.spines.values():
                    spine.set_visible(True); spine.set_edgecolor(colour); spine.set_linewidth(2)
                ax.set_xticks([]); ax.set_yticks([]); ax.axis("on")
                if iid is not None:
                    ax.text(0.5, -0.04, str(iid), transform=ax.transAxes, ha="center",
                            va="top", fontsize=5.5, color="#333")

    if table_ax is not None:
        body = [[f"{c}", str(sizes[c]),
                 ", ".join(str(i) for _, i in by_cluster[c])] for c in clusters]
        t = table_ax.table(cellText=body, colLabels=["cluster", "size", "member ids"],
                           cellLoc="left", colLoc="left", loc="upper center",
                           colWidths=[0.08, 0.08, 0.84])
        t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1, 1.3)

    fig.suptitle("Per-cluster sample thumbnails" + (f" — {caption}" if caption else ""),
                 fontsize=11)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path
