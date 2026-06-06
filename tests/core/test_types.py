import numpy as np
from src.core.types import Embeddings

def test_embeddings_validates_alignment():
    import pytest
    Embeddings(vectors=np.zeros((2, 4)), ids=["a", "b"], name="x")  # ok
    with pytest.raises(ValueError):
        Embeddings(vectors=np.zeros((2, 4)), ids=["a"], name="x")   # mismatch
