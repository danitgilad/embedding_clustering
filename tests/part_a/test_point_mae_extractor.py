import pytest
from src.core.types import Asset, FeatureExtractor


def test_point_mae_implements_protocol():
    from src.part_a.extractors.point_mae import PointMAEExtractor
    ext = PointMAEExtractor(checkpoint="vendor/x.pth", n_points=1024, seed=0)
    assert ext.name == "point_mae"
    assert callable(ext.extract)
    assert isinstance(ext, FeatureExtractor)


@pytest.mark.slow
def test_point_mae_extract_real():
    """Needs the downloaded Point-MAE checkpoint (scripts/setup_encoders.sh)."""
    from src.part_a.extractors.point_mae import PointMAEExtractor
    ext = PointMAEExtractor(checkpoint="checkpoints/point_mae_pretrain.pth",
                            n_points=1024, seed=0)
    emb = ext.extract([Asset(id="m", path="assets/00686245121504.glb")])
    assert emb.vectors.shape[0] == 1 and emb.vectors.shape[1] == 768
