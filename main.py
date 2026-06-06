"""CLI entry point. Runs each part and each stage (clear entry point per part).

Examples:
  python main.py part-a all
  python main.py part-a render
  python main.py part-a cluster
  python main.py part-b generate --n 500
  python main.py part-b all
Global: --config, --log-level, --set k=v (dotted config override, repeatable).
"""
from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path

from src.config import load_config
from src.logging_setup import configure_logging
from src.utils.io import ensure_dir
from src.utils.seeding import seed_everything

log = logging.getLogger(__name__)


def parse_overrides(pairs: list[str]) -> dict:
    """Parse ['a.b=1', 'c=x'] into {'a.b': 1, 'c': 'x'} (literal-eval values when possible)."""
    out: dict = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        try:
            val = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            val = raw
        out[key.strip()] = val
    return out


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser: positional part + stage, global config/override flags."""
    p = argparse.ArgumentParser(description="Embedding clustering (Part A + Part B)")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="dotted config override, e.g. --set part_b.n_images=200")
    p.add_argument("part", choices=["part-a", "part-b"])
    p.add_argument("stage", choices=["render", "generate", "extract", "cluster", "viewer", "all"])
    p.add_argument("--n", type=int, default=None, help="Part B: number of faces")
    return p


def _run_part_a(cfg, stage: str) -> None:
    """Run the requested Part A stage(s)."""
    from src.part_a import pipeline as A

    out = ensure_dir(Path(cfg.paths.outputs_dir) / "part_a")
    render_dir = ensure_dir(Path(cfg.paths.data_dir) / "part_a_renders")
    assets = A.discover_assets(cfg.paths.assets_dir)
    if stage in ("render", "all"):
        from src.part_a.mesh_io import load_glb, to_single_mesh
        from src.part_a.render import render_views
        views = [tuple(v) for v in cfg.part_a.render.views]
        for a in assets:
            mesh = to_single_mesh(load_glb(a.path))
            render_views(mesh, a.id, render_dir, cfg.part_a.render.size_px,
                         cfg.part_a.render.supersample, views)
            # one colour-baked front-view render per asset, for the viewer's hover popup
            color_mesh = to_single_mesh(load_glb(a.path), bake_texture_color=True)
            render_views(color_mesh, a.id, render_dir / "colored", cfg.part_a.render.size_px,
                         cfg.part_a.render.supersample, views[:1])
    if stage in ("extract", "cluster", "all"):
        import dataclasses

        from src.utils.io import sanitize_id
        umap_cfg = dataclasses.asdict(cfg.reduce.umap)
        # front-view render per asset, for the per-cluster montage
        montage = {a.id: render_dir / f"{sanitize_id(a.id)}_v0.png" for a in assets}
        for ext in A.build_extractors(cfg, render_dir):
            res = A.run_clustering_stage(
                ext, assets, out, cfg.part_a.clustering.algorithms,
                cfg.part_a.clustering.k_min, cfg.part_a.clustering.k_max,
                cfg.reduce.preprocess, cfg.reduce.pca_components, umap_cfg, cfg.seed,
                montage_images=montage)
            log.info("Part A %s: %s", ext.name, res)
    if stage in ("viewer", "all"):
        from src.part_a.viewer import build_part_a_viewer
        build_part_a_viewer(cfg, out, render_dir)


def _run_part_b(cfg, stage: str) -> None:
    """Run the requested Part B stage(s)."""
    from src.part_b import pipeline as B
    from src.part_b.generate import generate_faces

    data_dir = ensure_dir(Path(cfg.paths.data_dir) / "faces")
    out = ensure_dir(Path(cfg.paths.outputs_dir) / "part_b")
    if stage in ("generate", "all"):
        generate_faces(cfg.part_b.n_images, cfg.part_b.tpdne_url, data_dir,
                       cfg.part_b.request_delay_s, cfg.part_b.max_retries)
    if stage in ("extract", "cluster", "all"):
        import dataclasses
        from src.core.types import Asset
        assets = [Asset(id=p.stem, path=p) for p in sorted(data_dir.glob("*.jpg"))]
        for ext in B.build_extractors(cfg):
            res = B.run_clustering_stage(
                ext, assets, out, cfg.part_b.clustering.algorithms,
                cfg.part_b.clustering.k_min, cfg.part_b.clustering.k_max,
                cfg.reduce.preprocess, cfg.reduce.pca_components,
                dataclasses.asdict(cfg.reduce.umap), cfg.seed,
                montage_images={a.id: a.path for a in assets})
            log.info("Part B %s: %s", ext.name,
                     {k: v for k, v in res.items() if not k.endswith("__profile")})
    if stage in ("viewer", "all"):
        from src.part_b.viewer import build_part_b_viewer
        build_part_b_viewer(cfg, out, data_dir)


def main(argv: list[str] | None = None) -> None:
    """Parse args, load config, seed, and dispatch to the requested part/stage."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    overrides = parse_overrides(args.overrides)
    if args.part == "part-b" and args.n is not None:
        overrides["part_b.n_images"] = args.n
    cfg = load_config(args.config, overrides)
    seed_everything(cfg.seed)
    if args.part == "part-a":
        _run_part_a(cfg, args.stage)
    else:
        _run_part_b(cfg, args.stage)


if __name__ == "__main__":
    main()
