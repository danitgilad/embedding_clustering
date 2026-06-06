import pytest
from src.core.types import Asset, FeatureExtractor


def test_dinov2_implements_protocol(tmp_path):
    from src.part_a.extractors.dinov2 import DINOv2Extractor
    ext = DINOv2Extractor(hf_model="facebook/dinov2-base", render_dir=tmp_path)
    assert ext.name == "dinov2"
    assert callable(ext.extract)
    assert isinstance(ext, FeatureExtractor)   # runtime_checkable Protocol shape


@pytest.mark.slow
def test_dinov2_extract_real(tmp_path):
    """Real model + real render. Needs torch + network for weights (box only)."""
    import trimesh
    from src.part_a.render import render_views
    from src.part_a.extractors.dinov2 import DINOv2Extractor
    mesh = trimesh.creation.box(extents=(2, 1, 0.2))
    render_views(mesh, "box", tmp_path, size_px=128, supersample=1, views=[(80, -90)])
    ext = DINOv2Extractor(hf_model="facebook/dinov2-base", render_dir=tmp_path)
    emb = ext.extract([Asset(id="box", path=tmp_path / "box.glb")])
    assert emb.vectors.shape[0] == 1 and emb.vectors.shape[1] > 100
