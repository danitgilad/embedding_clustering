import pytest
from src.core.types import Asset, FeatureExtractor

def test_dinov2_generic_implements_protocol():
    from src.part_b.extractors.dinov2_generic import DINOv2GenericExtractor
    ext = DINOv2GenericExtractor(hf_model="facebook/dinov2-base")
    assert ext.name == "dinov2_generic"
    assert callable(ext.extract)
    assert isinstance(ext, FeatureExtractor)

@pytest.mark.slow
def test_dinov2_generic_extract_real(tmp_path):
    from PIL import Image
    from src.part_b.extractors.dinov2_generic import DINOv2GenericExtractor
    p = tmp_path / "face_0000.jpg"; Image.new("RGB", (256, 256), (180, 150, 130)).save(p)
    ext = DINOv2GenericExtractor(hf_model="facebook/dinov2-base")
    emb = ext.extract([Asset(id="face_0000", path=p)])
    assert emb.vectors.shape[0] == 1 and emb.vectors.shape[1] > 100
