from src.utils.seeding import seed_everything
from src.utils.io import sanitize_id, ensure_dir

def test_sanitize_id_keeps_safe_chars():
    assert sanitize_id("00712316925280 (1)") == "00712316925280__1_"

def test_ensure_dir_creates(tmp_path):
    d = ensure_dir(tmp_path / "a" / "b")
    assert d.is_dir()

def test_seed_everything_makes_numpy_deterministic():
    import numpy as np
    seed_everything(123); a = np.random.rand(3)
    seed_everything(123); b = np.random.rand(3)
    assert (a == b).all()
