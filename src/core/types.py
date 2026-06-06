"""Core data contracts shared by every part.

An Asset is one input item (a GLB mesh, or a face image). An Embeddings bundle pairs
an (N, D) matrix with its N ids. A FeatureExtractor turns Assets into Embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Asset:
    """One input item. `path` points at the source file; `id` is its stable identifier."""
    id: str
    path: Path


@dataclass(frozen=True)
class Embeddings:
    """An (N, D) embedding matrix with aligned ids and the producing extractor's name."""
    vectors: np.ndarray
    ids: list[str]
    name: str

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D, got shape {self.vectors.shape}")
        if self.vectors.shape[0] != len(self.ids):
            raise ValueError(
                f"row count {self.vectors.shape[0]} != id count {len(self.ids)}"
            )


@runtime_checkable
class FeatureExtractor(Protocol):
    """Turns a sequence of Assets into one Embeddings bundle. `name` keys the cache."""
    name: str

    def extract(self, items: Sequence[Asset]) -> Embeddings: ...
