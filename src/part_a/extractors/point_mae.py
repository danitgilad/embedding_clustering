"""3D geometric feature: Point-MAE embedding over points sampled from the mesh surface.

(Part A primary.) Self-supervised, pure-geometry — mirrors DINOv2. The vendored Point-MAE
repo (scripts/setup_encoders.sh) provides the model + checkpoint. We load the pretrained
encoder, feed (n_points, 3) surface samples, and take the pooled feature.

NOTE FOR THE GPU-BOX RUN: the exact import path + forward signature depend on the vendored
repo. Confirm the encoder class + how to obtain the global feature when running setup on the
box; the repo-specific bits are isolated to `_load_model` and `_encode`.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from src.core.types import Asset, Embeddings
from src.part_a.mesh_io import load_glb, sample_surface_points, to_single_mesh

log = logging.getLogger(__name__)
_VENDOR = Path("vendor/Point-MAE")


class PointMAEExtractor:
    """Pretrained Point-MAE encoder over mesh-surface point clouds."""

    def __init__(self, checkpoint: str, n_points: int, seed: int) -> None:
        self.name = "point_mae"
        self.checkpoint = checkpoint
        self.n_points = n_points
        self.seed = seed
        self._model = None
        self._device = "cpu"

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch

        if str(_VENDOR) not in sys.path:
            sys.path.insert(0, str(_VENDOR))
        from models.Point_MAE import Point_MAE  # type: ignore  # vendored repo

        model = Point_MAE(self._default_cfg())
        state = torch.load(self.checkpoint, map_location="cpu")
        model.load_state_dict(state.get("base_model", state), strict=False)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = model.eval().to(self._device)

    @staticmethod
    def _default_cfg() -> SimpleNamespace:
        """Minimal config the vendored Point_MAE expects. Confirm fields from the repo yaml."""
        return SimpleNamespace(
            mask_ratio=0.6, mask_type="rand", trans_dim=384, encoder_dims=384,
            depth=12, drop_path_rate=0.1, num_heads=6, group_size=32, num_group=64,
        )

    def _encode(self, pts: np.ndarray) -> np.ndarray:
        import torch

        x = torch.from_numpy(pts).float().unsqueeze(0).to(self._device)  # (1, N, 3)
        with torch.no_grad():
            feats = self._model.forward_eval(x) if hasattr(self._model, "forward_eval") \
                else self._model(x)
        feats = feats if isinstance(feats, torch.Tensor) else feats[0]
        return feats.squeeze(0).float().cpu().numpy().reshape(-1)

    def extract(self, items: Sequence[Asset]) -> Embeddings:
        """Sample surface points per asset and encode to one vector each."""
        self._load_model()
        vecs, ids = [], []
        for asset in items:
            try:
                mesh = to_single_mesh(load_glb(asset.path))
                pts = sample_surface_points(mesh, self.n_points, self.seed)
                vecs.append(self._encode(pts))
                ids.append(asset.id)
            except Exception:  # noqa: BLE001 - isolate per-item failures
                log.exception("Point-MAE failed on %s; skipping", asset.id)
        return Embeddings(np.vstack(vecs), ids, self.name)
