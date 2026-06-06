import pytest
from src.core.types import Asset, FeatureExtractor

def test_clip_implements_protocol(tmp_path):
    from src.part_a.extractors.clip import CLIPExtractor
    ext = CLIPExtractor(hf_model="openai/clip-vit-base-patch32", render_dir=tmp_path)
    assert ext.name == "clip"
    assert callable(ext.extract)
    assert isinstance(ext, FeatureExtractor)

@pytest.mark.slow
def test_clip_extract_real(tmp_path):
    import trimesh
    from src.part_a.render import render_views
    from src.part_a.extractors.clip import CLIPExtractor
    mesh = trimesh.creation.box(extents=(2, 1, 0.2))
    render_views(mesh, "box", tmp_path, size_px=128, supersample=1, views=[(80, -90)])
    ext = CLIPExtractor(hf_model="openai/clip-vit-base-patch32", render_dir=tmp_path)
    emb = ext.extract([Asset(id="box", path=tmp_path / "box.glb")])
    assert emb.vectors.shape[0] == 1 and emb.vectors.shape[1] >= 256
