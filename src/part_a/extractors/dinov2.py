"""2D visual feature: render-based DINOv2 embedding (Part A primary).

For each asset, load its pre-rendered views, run them through a frozen DINOv2 ViT, and
mean-pool the per-view CLS embeddings into one vector. Renders are produced beforehand by
render.py and live under render_dir as <id>_v<k>.png.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from src.core.types import Asset, Embeddings
from src.utils.io import sanitize_id

log = logging.getLogger(__name__)


class DINOv2Extractor:
    """Frozen DINOv2 image embedding over multi-view renders, mean-pooled per asset."""

    def __init__(self, hf_model: str, render_dir: str | Path) -> None:
        self.name = "dinov2"
        self.hf_model = hf_model
        self.render_dir = Path(render_dir)
        self._model = None
        self._processor = None
        self._device = "cpu"

    def _ensure_model(self) -> None:
        if self._model is None:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            self._processor = AutoImageProcessor.from_pretrained(self.hf_model)
            self._model = AutoModel.from_pretrained(self.hf_model).eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)

    def _embed_image(self, path: Path) -> np.ndarray:
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        cls = out.last_hidden_state[:, 0, :]   # CLS token = global image embedding
        return cls.squeeze(0).cpu().numpy()

    def extract(self, items: Sequence[Asset]) -> Embeddings:
        """Embed each asset's renders and mean-pool views into one vector."""
        self._ensure_model()
        vecs, ids = [], []
        for asset in items:
            views = sorted(self.render_dir.glob(f"{sanitize_id(asset.id)}_v*.png"))
            if not views:
                log.warning("no renders for %s; skipping", asset.id)
                continue
            per_view = np.stack([self._embed_image(p) for p in views])
            vecs.append(per_view.mean(axis=0))
            ids.append(asset.id)
        if not vecs:
            raise ValueError("DINOv2 produced no embeddings (no renders found for any asset)")
        return Embeddings(np.vstack(vecs), ids, self.name)
