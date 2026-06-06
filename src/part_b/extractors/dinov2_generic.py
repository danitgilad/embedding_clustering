"""Part B comparison feature: generic DINOv2 embedding of the face image (no face model).

Answers 'does a face-specialized model (ArcFace) cluster faces better than a general
backbone?'. Embeds each face JPG directly (CLS token); exposes no attributes.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from src.core.types import Asset, Embeddings

log = logging.getLogger(__name__)


class DINOv2GenericExtractor:
    """Frozen DINOv2 image embedding over single face images."""

    def __init__(self, hf_model: str) -> None:
        self.name = "dinov2_generic"
        self.hf_model = hf_model
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

    def extract(self, items: Sequence[Asset]) -> Embeddings:
        """Embed each face image (CLS token) into one vector."""
        self._ensure_model()
        import torch
        from PIL import Image

        vecs, ids = [], []
        for asset in items:
            try:
                img = Image.open(asset.path).convert("RGB")
            except Exception:  # noqa: BLE001 - skip unreadable images, keep the batch
                log.warning("unreadable image %s; skipping", asset.id)
                continue
            inputs = self._processor(images=img, return_tensors="pt").to(self._device)
            with torch.no_grad():
                out = self._model(**inputs)
            vecs.append(out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy())
            ids.append(asset.id)
        if not vecs:
            raise ValueError("DINOv2-generic produced no embeddings (no readable images)")
        return Embeddings(np.vstack(vecs), ids, self.name)
