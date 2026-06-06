"""3D geometric feature: Point-MAE embedding over points sampled from the mesh surface.

(Part A primary.) Self-supervised, pure-geometry — the 3D counterpart to DINOv2 on the 2D
side. We sample a point cloud from the triangulated mesh surface, run it through a
pretrained Point-MAE encoder, and pool the per-group tokens into one vector per asset.

The encoder is a self-contained, CPU-friendly reimplementation in
``_point_mae_backbone.py`` that loads the official Point-MAE ShapeNet pretrained weights
(see ``scripts/setup_encoders.sh``). This avoids the upstream repo's CUDA-only ops/extensions
so the whole pipeline runs CPU-only and stays reproducible.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from src.core.types import Asset, Embeddings
from src.part_a.mesh_io import load_glb, sample_surface_points, to_single_mesh

log = logging.getLogger(__name__)


class PointMAEExtractor:
    """Pretrained Point-MAE encoder over mesh-surface point clouds (768-d embedding)."""

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

        from src.part_a.extractors._point_mae_backbone import load_point_mae_encoder

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = load_point_mae_encoder(self.checkpoint).to(self._device)

    def _encode(self, pts: np.ndarray) -> np.ndarray:
        import torch

        x = torch.from_numpy(pts).float().unsqueeze(0).to(self._device)  # (1, N, 3)
        with torch.no_grad():
            feat = self._model(x)
        return feat.squeeze(0).float().cpu().numpy().reshape(-1)

    def extract(self, items: Sequence[Asset]) -> Embeddings:
        """Sample surface points per asset and encode to one vector each.

        Per-item failures are isolated and logged so one bad mesh can't abort the batch.
        Raises ValueError if every item failed (no silent empty result).
        """
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
        if not vecs:
            raise ValueError("Point-MAE produced no embeddings (all items failed)")
        return Embeddings(np.vstack(vecs), ids, self.name)
