"""Persist/restore Embeddings as <name>.npy + <name>.ids.json under a directory.

This cache is the seam between the GPU encode (run once on the box) and the cheap
analysis loop. load_embeddings HARD-fails on row/id misalignment because a silent desync
would invalidate every downstream comparison.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.core.types import Embeddings
from src.utils.io import ensure_dir

log = logging.getLogger(__name__)


def save_embeddings(emb: Embeddings, out_dir: str | Path) -> Path:
    """Write emb to <out_dir>/<name>.npy and <name>.ids.json. Returns the .npy path."""
    out = ensure_dir(out_dir)
    npy = out / f"{emb.name}.npy"
    np.save(npy, emb.vectors)
    (out / f"{emb.name}.ids.json").write_text(json.dumps(emb.ids))
    log.info("Saved embeddings '%s' %s -> %s", emb.name, emb.vectors.shape, npy)
    return npy


def load_embeddings(name: str, out_dir: str | Path) -> Embeddings:
    """Load embeddings by name; validates (N,D)<->ids alignment via Embeddings.__post_init__."""
    out = Path(out_dir)
    vectors = np.load(out / f"{name}.npy")
    ids = json.loads((out / f"{name}.ids.json").read_text())
    return Embeddings(vectors=vectors, ids=ids, name=name)
