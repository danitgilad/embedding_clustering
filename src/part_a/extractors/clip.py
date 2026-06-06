"""2D visual feature: render-based CLIP image embedding (Part A optional comparison).

Mirrors the DINOv2 extractor: load each asset's multi-view renders, embed each with a
frozen CLIP image encoder, mean-pool the views into one vector per asset.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from src.core.types import Asset, Embeddings
from src.utils.io import sanitize_id

log = logging.getLogger(__name__)


class CLIPExtractor:
    """Frozen CLIP image embedding over multi-view renders, mean-pooled per asset."""

    def __init__(self, hf_model: str, render_dir: str | Path) -> None:
        self.name = "clip"
        self.hf_model = hf_model
        self.render_dir = Path(render_dir)
        self._model = None
        self._processor = None
        self._device = "cpu"

    def _ensure_model(self) -> None:
        if self._model is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(self.hf_model)
            self._model = CLIPModel.from_pretrained(self.hf_model).eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)

    def _embed_image(self, path: Path) -> np.ndarray:
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt").to(self._device)
        with torch.no_grad():
            feat = self._model.get_image_features(**inputs)
        return feat.squeeze(0).cpu().numpy()

    def extract(self, items: Sequence[Asset]) -> Embeddings:
        """Embed each asset's renders and mean-pool views into one vector."""
        self._ensure_model()
        vecs, ids = [], []
        for asset in items:
            views = sorted(self.render_dir.glob(f"{sanitize_id(asset.id)}_v*.png"))
            if not views:
                log.warning("no renders for %s; skipping", asset.id)
                continue
            vecs.append(np.stack([self._embed_image(p) for p in views]).mean(axis=0))
            ids.append(asset.id)
        if not vecs:
            raise ValueError("CLIP produced no embeddings (no renders found for any asset)")
        return Embeddings(np.vstack(vecs), ids, self.name)
