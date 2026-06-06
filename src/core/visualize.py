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


def scatter_2d(points: np.ndarray, labels: np.ndarray, out_path: str | Path,
               title: str = "") -> Path:
    """Scatter 2D points colored by integer label; save PNG. Returns the path."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    sc = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=40)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


def metric_table_png(rows: Mapping[str, Mapping[str, float]], out_path: str | Path,
                     title: str = "") -> Path:
    """Render a {row_name: {metric: value}} table as a PNG (for the README/report)."""
    out_path = Path(out_path)
    metrics = sorted({m for r in rows.values() for m in r})
    cell_text = [[f"{rows[r].get(m, float('nan')):.3f}" for m in metrics] for r in rows]
    fig, ax = plt.subplots(figsize=(2 + 1.6 * len(metrics), 1 + 0.5 * len(rows)), dpi=120)
    ax.axis("off"); ax.set_title(title)
    tbl = ax.table(cellText=cell_text, rowLabels=list(rows), colLabels=metrics,
                   loc="center")
    tbl.scale(1, 1.4)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


def cluster_montage(image_paths: Sequence[str | Path], labels: np.ndarray,
                    out_path: str | Path, thumb_px: int = 96,
                    max_per_cluster: int = 12) -> Path:
    """Grid of thumbnails grouped by cluster (one row per cluster). Save PNG.

    Shows at most `max_per_cluster` examples per cluster so the montage stays readable for
    large clusters (cluster label -1, HDBSCAN noise, is skipped).
    """
    out_path = Path(out_path)
    by_cluster: dict[int, list[Path]] = {}
    for p, lab in zip(image_paths, labels):
        if int(lab) == -1:
            continue
        bucket = by_cluster.setdefault(int(lab), [])
        if len(bucket) < max_per_cluster:
            bucket.append(Path(p))
    rows = sorted(by_cluster)
    ncols = max(len(v) for v in by_cluster.values())
    fig, axes = plt.subplots(len(rows), ncols,
                             figsize=(ncols * 1.3, len(rows) * 1.3), dpi=120,
                             squeeze=False)
    for r, c_lab in enumerate(rows):
        for col in range(ncols):
            ax = axes[r][col]; ax.axis("off")
            if col < len(by_cluster[c_lab]):
                img = Image.open(by_cluster[c_lab][col]).convert("RGB").resize(
                    (thumb_px, thumb_px))
                ax.imshow(np.asarray(img))
        axes[r][0].set_ylabel(f"c{c_lab}", rotation=0, labelpad=18, va="center")
    fig.suptitle("Clusters")
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path
