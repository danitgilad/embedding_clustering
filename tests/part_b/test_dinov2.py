import pytest
from src.core.types import Asset, FeatureExtractor

def test_dinov2_face_implements_protocol():
    from src.part_b.extractors.dinov2 import DINOv2FaceExtractor
    ext = DINOv2FaceExtractor(hf_model="facebook/dinov2-base")
    assert ext.name == "dinov2"
    assert callable(ext.extract)
    assert isinstance(ext, FeatureExtractor)

@pytest.mark.slow
def test_dinov2_face_extract_real(tmp_path):
    from PIL import Image
    from src.part_b.extractors.dinov2 import DINOv2FaceExtractor
    p = tmp_path / "face_0000.jpg"; Image.new("RGB", (256, 256), (180, 150, 130)).save(p)
    ext = DINOv2FaceExtractor(hf_model="facebook/dinov2-base")
    emb = ext.extract([Asset(id="face_0000", path=p)])
    assert emb.vectors.shape[0] == 1 and emb.vectors.shape[1] > 100
