from pathlib import Path
from src.config import load_config

def test_load_config_reads_yaml_and_applies_overrides(tmp_path: Path):
    yaml_text = """
seed: 7
paths: {assets_dir: assets, outputs_dir: outputs, data_dir: data}
reduce: {preprocess: [standardize], pca_components: null, umap: {n_neighbors: 5, min_dist: 0.1, metric: cosine}}
part_a:
  encoders_2d: [dinov2]
  encoders_3d: [point_mae]
  render: {size_px: 256, supersample: 2, views: [[80, -90]]}
  point_sampling: {n_points: 1024}
  dinov2: {hf_model: facebook/dinov2-base}
  point_mae: {checkpoint: vendor/x.pth}
  clustering: {algorithms: [kmeans], k_min: 2, k_max: 4}
part_b:
  n_images: 10
  tpdne_url: https://example.com/
  request_delay_s: 0.0
  max_retries: 1
  insightface: {model_name: buffalo_l, det_size: 320}
  clustering: {algorithms: [kmeans], k_min: 2, k_max: 3}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p, overrides={"part_b.n_images": 99})
    assert cfg.seed == 7
    assert cfg.part_a.render.size_px == 256
    assert cfg.part_b.n_images == 99            # override applied
    assert cfg.part_a.clustering.k_max == 4

def test_load_config_rejects_unknown_override_key(tmp_path: Path):
    import pytest
    p = tmp_path / "c.yaml"
    p.write_text("seed: 1\npaths: {assets_dir: a, outputs_dir: o, data_dir: d}\n")
    with pytest.raises(KeyError):
        load_config(p, overrides={"nonexistent.key": 1})
