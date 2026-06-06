"""Part B feature: InsightFace ArcFace 512-D embedding + age/gender/pose attributes.

We cluster the embedding; the attributes (collected during extraction) are exposed via
`.attributes` and later used as pseudo-labels to interpret/validate clusters. Images with
no detectable face are skipped and counted; when several faces are detected the largest
(the main subject) is used. NOTE: thispersondoesnotexist faces fill the frame, so the
detector needs a small det_size (~320) — a large det_size misses these big faces entirely.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from src.core.types import Asset, Embeddings

log = logging.getLogger(__name__)


def _gender(face: object) -> str:
    """Normalize InsightFace gender to 'M'/'F'/'?' across versions (.sex or .gender)."""
    g = getattr(face, "sex", None)
    if g in ("M", "F"):
        return g
    val = getattr(face, "gender", None)   # older insightface: 1=male, 0=female
    return "M" if val == 1 else "F" if val == 0 else "?"


class ArcFaceExtractor:
    """InsightFace FaceAnalysis wrapper producing aligned embeddings + attributes."""

    def __init__(self, model_name: str, det_size: int) -> None:
        self.name = "arcface"
        self.model_name = model_name
        self.det_size = det_size
        self._app = None
        self.attributes: dict[str, dict] = {}   # id -> {age, gender, pose_yaw}
        self.skipped: dict[str, str] = {}        # id -> reason

    def _ensure_app(self) -> None:
        if self._app is None:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name=self.model_name)
            app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))
            self._app = app

    def extract(self, items: Sequence[Asset]) -> Embeddings:
        """Detect + embed one face per image (the largest); collect age/gender/pose attributes."""
        self._ensure_app()
        import cv2

        vecs, ids = [], []
        for asset in items:
            img = cv2.imread(str(asset.path))
            if img is None:
                self.skipped[asset.id] = "unreadable"
                continue
            faces = self._app.get(img)
            if not faces:
                self.skipped[asset.id] = "no face"
                continue
            # Keep the largest face (main subject) if the detector returns spurious extras.
            face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
            vecs.append(np.asarray(face.normed_embedding, dtype=float))
            ids.append(asset.id)
            pose = getattr(face, "pose", None)
            self.attributes[asset.id] = {
                "age": float(face.age),
                "gender": _gender(face),
                "pose_yaw": float(pose[1]) if pose is not None else 0.0,
            }
        log.info("ArcFace: kept %d, skipped %d", len(ids), len(self.skipped))
        if not vecs:
            raise ValueError("ArcFace produced no embeddings (all images skipped)")
        return Embeddings(np.vstack(vecs), ids, self.name)
