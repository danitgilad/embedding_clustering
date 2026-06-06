"""Typed configuration: YAML -> frozen dataclasses, with dotted-key CLI overrides.

No other module reads YAML or hardcodes paths; they consume a Config object.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, get_type_hints

import yaml


@dataclass(frozen=True)
class Paths:
    assets_dir: str = "assets"
    outputs_dir: str = "outputs"
    data_dir: str = "data"


@dataclass(frozen=True)
class UMAPCfg:
    n_neighbors: int = 10
    min_dist: float = 0.1
    metric: str = "cosine"


@dataclass(frozen=True)
class ReduceCfg:
    preprocess: tuple[str, ...] = ("standardize", "l2norm")
    pca_components: int | None = None
    umap: UMAPCfg = field(default_factory=UMAPCfg)


@dataclass(frozen=True)
class RenderCfg:
    size_px: int = 512
    supersample: int = 3
    views: tuple[tuple[float, float], ...] = ((80.0, -90.0),)


@dataclass(frozen=True)
class PointSamplingCfg:
    n_points: int = 8192


@dataclass(frozen=True)
class DinoCfg:
    hf_model: str = "facebook/dinov2-base"


@dataclass(frozen=True)
class PointMAECfg:
    checkpoint: str = "vendor/Point-MAE/checkpoints/pretrain.pth"


@dataclass(frozen=True)
class ClusteringCfg:
    algorithms: tuple[str, ...] = ("kmeans", "agglomerative")
    k_min: int = 2
    k_max: int = 8


@dataclass(frozen=True)
class PartACfg:
    encoders_2d: tuple[str, ...] = ("dinov2",)
    encoders_3d: tuple[str, ...] = ("point_mae",)
    render: RenderCfg = field(default_factory=RenderCfg)
    point_sampling: PointSamplingCfg = field(default_factory=PointSamplingCfg)
    dinov2: DinoCfg = field(default_factory=DinoCfg)
    point_mae: PointMAECfg = field(default_factory=PointMAECfg)
    clustering: ClusteringCfg = field(default_factory=ClusteringCfg)


@dataclass(frozen=True)
class InsightFaceCfg:
    model_name: str = "buffalo_l"
    det_size: int = 640


@dataclass(frozen=True)
class PartBCfg:
    n_images: int = 500
    tpdne_url: str = "https://thispersondoesnotexist.com/"
    request_delay_s: float = 1.0
    max_retries: int = 5
    insightface: InsightFaceCfg = field(default_factory=InsightFaceCfg)
    clustering: ClusteringCfg = field(
        default_factory=lambda: ClusteringCfg(
            algorithms=("kmeans", "agglomerative", "hdbscan"), k_min=2, k_max=12
        )
    )


@dataclass(frozen=True)
class Config:
    seed: int = 42
    paths: Paths = field(default_factory=Paths)
    reduce: ReduceCfg = field(default_factory=ReduceCfg)
    part_a: PartACfg = field(default_factory=PartACfg)
    part_b: PartBCfg = field(default_factory=PartBCfg)


def _build(dc_type: type, data: Mapping[str, Any] | None) -> Any:
    """Recursively build a (frozen) dataclass from a mapping, applying defaults.

    Resolves string annotations via get_type_hints so nested dataclasses are detected
    even under `from __future__ import annotations`. YAML lists become tuples.
    """
    if data is None:
        return dc_type()
    hints = get_type_hints(dc_type)
    kwargs: dict[str, Any] = {}
    for f in fields(dc_type):
        if f.name not in data:
            continue
        val = data[f.name]
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(val, Mapping):
            kwargs[f.name] = _build(ftype, val)
        elif isinstance(val, list):
            kwargs[f.name] = tuple(tuple(x) if isinstance(x, list) else x for x in val)
        else:
            kwargs[f.name] = val
    return dc_type(**kwargs)


def _apply_override(cfg: Config, dotted: str, value: Any) -> Config:
    """Return a new Config with one dotted-path field replaced. Raises KeyError if absent."""
    parts = dotted.split(".")

    def walk(obj: Any, segs: list[str]) -> Any:
        if not is_dataclass(obj) or not any(f.name == segs[0] for f in fields(obj)):
            raise KeyError(dotted)
        if len(segs) == 1:
            return replace(obj, **{segs[0]: value})
        head, *rest = segs
        return replace(obj, **{head: walk(getattr(obj, head), rest)})

    return walk(cfg, parts)


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Config:
    """Load YAML into a frozen Config, then apply dotted-key overrides (CLI flags)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = _build(Config, raw)
    for key, val in (overrides or {}).items():
        cfg = _apply_override(cfg, key, val)
    return cfg
