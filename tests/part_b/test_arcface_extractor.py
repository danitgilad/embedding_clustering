import pytest
from src.core.types import Asset, FeatureExtractor


def test_arcface_implements_protocol():
    from src.part_b.extractors.arcface import ArcFaceExtractor
    ext = ArcFaceExtractor(model_name="buffalo_l", det_size=320)
    assert ext.name == "arcface"
    assert callable(ext.extract)
    assert isinstance(ext.attributes, dict)
    assert isinstance(ext, FeatureExtractor)


@pytest.mark.slow
def test_arcface_extract_real(tmp_path):
    """Needs insightface + a real face image at tmp_path/face_0000.jpg (GPU box)."""
    from src.part_b.extractors.arcface import ArcFaceExtractor
    ext = ArcFaceExtractor(model_name="buffalo_l", det_size=320)
    emb = ext.extract([Asset(id="f0", path=tmp_path / "face_0000.jpg")])
    assert emb.vectors.shape[1] == 512
    assert "f0" in ext.attributes and "age" in ext.attributes["f0"]
