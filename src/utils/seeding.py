"""Global determinism for reproducible clusters/UMAP."""
from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch (if importable). Call once after config load."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
