"""Part A dataset exploration: examine each GLB's internal structure and materials.

Addresses the assignment's "Dataset Exploration" step — load each asset, inspect how mesh
components and materials are organised, and write a committed Markdown report. This is what
motivates two downstream decisions documented in the README: GLBs are multi-component Scenes
(so scene-graph node transforms must be applied when flattening), and their materials carry
real colours that the encoders deliberately ignore (greyscale shape / pure geometry).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh

from src.core.types import Asset
from src.part_a.mesh_io import load_glb
from src.utils.io import ensure_dir

log = logging.getLogger(__name__)


def _rgba_str(color) -> str | None:
    """Render a trimesh main_color (RGBA array) as a compact hex string, or None."""
    if color is None:
        return None
    arr = np.asarray(color).astype(int).tolist()
    if len(arr) < 3:
        return None
    return "#{:02x}{:02x}{:02x}".format(*arr[:3])


def explore_asset(asset: Asset) -> dict:
    """Inspect one GLB: component/vertex/face counts, materials, texturing, bounding extent."""
    obj = load_glb(asset.path)
    geoms = {"<mesh>": obj} if isinstance(obj, trimesh.Trimesh) else dict(obj.geometry)
    materials: list[dict] = []
    textured = False
    for geom in geoms.values():
        vis = getattr(geom, "visual", None)
        mat = getattr(vis, "material", None)
        has_tex = getattr(mat, "image", None) is not None or getattr(vis, "uv", None) is not None
        textured = textured or bool(has_tex)
        materials.append({
            "name": getattr(mat, "name", None),
            "main_color": _rgba_str(getattr(mat, "main_color", None)),
            "textured": bool(has_tex),
        })
    extent = None
    if obj.bounds is not None:
        extent = np.round(obj.bounds[1] - obj.bounds[0], 3).tolist()
    return {
        "id": asset.id,
        "components": len(geoms),
        "vertices": int(sum(len(g.vertices) for g in geoms.values())),
        "faces": int(sum(len(g.faces) for g in geoms.values())),
        "n_materials": len(materials),
        "textured": textured,
        "main_colors": sorted({m["main_color"] for m in materials if m["main_color"]}),
        "extent": extent,
        "materials": materials,
    }


def explore_assets(assets: Sequence[Asset]) -> list[dict]:
    """Inspect every asset (sorted by id) — one structured row each."""
    return [explore_asset(a) for a in assets]


def _exploration_markdown(rows: list[dict]) -> str:
    """Render the per-asset rows into a Markdown report with a short observations summary."""
    comps = [r["components"] for r in rows]
    verts = [r["vertices"] for r in rows]
    n_textured = sum(r["textured"] for r in rows)
    header = (
        "# Part A — Dataset Exploration (GLB internal structure)\n\n"
        f"{len(rows)} `.glb` glasses assets. Each loads via `trimesh` as a **Scene** of one or "
        "more mesh components (frame / lenses / hinges …) with PBR materials. The table reports "
        "each file's internal organisation; the observations below motivate two pipeline "
        "decisions documented in the README.\n\n"
        "| asset | components | vertices | faces | materials | textured | main colours | extent (w×h×d) |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    body = ""
    for r in rows:
        colours = ", ".join(r["main_colors"]) or "—"
        extent = "×".join(f"{v:g}" for v in r["extent"]) if r["extent"] else "—"
        body += (f"| {r['id']} | {r['components']} | {r['vertices']:,} | {r['faces']:,} | "
                 f"{r['n_materials']} | {'yes' if r['textured'] else 'no'} | {colours} | {extent} |\n")
    summary = (
        "\n**Observations**\n"
        f"- Mesh components per asset range **{min(comps)}–{max(comps)}** — most GLBs are "
        "multi-component Scenes, so flattening must **apply each node's scene-graph transform** "
        "(`Scene.dump(concatenate=True)`), not merge in local frames (see README → Challenges).\n"
        f"- Vertex counts span **{min(verts):,}–{max(verts):,}**; we sample a fixed 1024 surface "
        "points for the 3D (Point-MAE) feature so geometry detail is comparable across assets.\n"
        f"- **{n_textured}/{len(rows)}** assets carry texture/material colour. That colour is real "
        "(baked into the coloured *hover* renders) but the encoders embed **greyscale** shape "
        "renders / pure xyz geometry — colour is intentionally **not** a clustering signal.\n"
    )
    return header + body + summary


def write_exploration_report(assets: Sequence[Asset], out_path: str | Path) -> Path:
    """Explore all assets and write the Markdown report to out_path."""
    rows = explore_assets(assets)
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    out_path.write_text(_exploration_markdown(rows))
    log.info("Wrote %s (%d assets)", out_path, len(rows))
    return out_path
