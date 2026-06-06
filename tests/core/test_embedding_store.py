import json
import numpy as np
import pytest
from src.core.types import Embeddings
from src.core.embedding_store import save_embeddings, load_embeddings

def test_round_trip(tmp_path):
    emb = Embeddings(np.arange(6, dtype=float).reshape(3, 2), ["a", "b", "c"], "dinov2")
    save_embeddings(emb, tmp_path)
    loaded = load_embeddings("dinov2", tmp_path)
    assert loaded.ids == ["a", "b", "c"]
    assert loaded.name == "dinov2"
    assert np.allclose(loaded.vectors, emb.vectors)

def test_load_detects_corrupted_alignment(tmp_path):
    emb = Embeddings(np.zeros((2, 2)), ["a", "b"], "x")
    save_embeddings(emb, tmp_path)
    (tmp_path / "x.ids.json").write_text(json.dumps(["only_one"]))
    with pytest.raises(ValueError):
        load_embeddings("x", tmp_path)
