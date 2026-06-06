# Embedding-Clustering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a well-structured Python project that clusters 3D glasses assets (Part A: 2D-render-derived vs 3D-mesh-derived features) and AI-generated faces (Part B: discover & characterize attribute clusters via InsightFace embeddings).

**Architecture:** A shared, part-agnostic pipeline (`reduce → cluster → evaluate → visualize`) operating on a generic `(embeddings, ids)` contract, fed by per-part `FeatureExtractor`s. Embeddings are cached to `.npy` keyed by extractor name, decoupling the GPU-heavy encode (run on elem-danit1) from cheap local analysis.

**Tech Stack:** Python 3.10+, trimesh, matplotlib (Agg), PyTorch + transformers (DINOv2), Point-MAE (vendored), insightface + onnxruntime, scikit-learn, umap-learn, hdbscan, PyYAML, pytest.

**Reference docs:** `task.md` (assignment), `DEFINITIONS.md` (acceptance checklist), `docs/superpowers/specs/2026-06-06-unsupervised-clustering-design.md` (design), `PLAN.md` (live decision log D1–D16).

**Standing rules (D10 — document continuously):** after each task, (a) append a line to `PLAN.md` Progress log, (b) tick the relevant `DEFINITIONS.md` box when a criterion is met, (c) every public function/class/module gets a docstring + type hints, (d) commit. The README grows section-by-section, not at the end.

**Compute (D11):** Heavy stages run on **elem-danit1** via the `run-on-elem-danit1` skill. Tests and pure-Python logic run anywhere. The `.npy` embedding cache is the machine boundary.

---

## File Structure

| File | Responsibility |
|---|---|
| `main.py` | argparse CLI entry point; routes `part-a`/`part-b` × stage to pipelines |
| `config/default.yaml` | All paths + tunables (single source of truth) |
| `requirements.txt` | Pinned pip deps (Point-MAE handled separately by setup script) |
| `README.md` | Setup, run, approach, findings (grown incrementally) |
| `src/config.py` | YAML → frozen dataclasses + validation; CLI override merge |
| `src/logging_setup.py` | One logging configuration used everywhere |
| `src/utils/seeding.py` | Global determinism (numpy/torch/random) |
| `src/utils/io.py` | Small filesystem helpers (sanitize id, ensure dir) |
| `src/core/types.py` | `Asset`, `Embeddings`, `FeatureExtractor` Protocol |
| `src/core/embedding_store.py` | Save/load `.npy` + `ids.json`; id-alignment guard |
| `src/core/reduce.py` | standardize / L2-norm / PCA / UMAP |
| `src/core/cluster.py` | KMeans, Agglomerative, HDBSCAN + silhouette-swept k |
| `src/core/metrics.py` | silhouette, Davies–Bouldin, Calinski–Harabasz, NMI/ARI/purity |
| `src/core/visualize.py` | UMAP scatter, per-cluster montages, dist plots, metric tables |
| `src/part_a/mesh_io.py` | trimesh load `.glb` → single mesh; surface point sampling; colors |
| `src/part_a/render.py` | Multi-view triangulated-mesh render (matplotlib Agg) |
| `src/part_a/extractors/dinov2.py` | renders → DINOv2 embedding (2D primary) |
| `src/part_a/extractors/point_mae.py` | sampled points → Point-MAE embedding (3D primary) |
| `src/part_a/pipeline.py` | Orchestrate Part A stages |
| `src/part_b/generate.py` | Rate-limited TPDNE download + hash-dedup + retry |
| `src/part_b/extractors/arcface.py` | InsightFace → 512-D embedding + age/gender/pose |
| `src/part_b/pipeline.py` | Orchestrate Part B stages |
| `scripts/setup_encoders.sh` | Bootstrap Point-MAE repo + checkpoint (idempotent) |
| `tests/...` | pytest for deterministic logic + Protocol contract |

---

## Phase 0 — Infra verification & scaffolding

### Task 0.1: Verify elem-danit1 access, outbound internet, and Point-MAE source

**Files:** none (verification only; record findings in `PLAN.md`).

- [ ] **Step 1: Confirm SSH + GPU on the box.** Use the `run-on-elem-danit1` skill to run:
  `nvidia-smi --query-gpu=name,memory.total --format=csv && python3 --version`
  Expected: an A100 line + Python 3.x.

- [ ] **Step 2: Verify outbound internet to TPDNE (resolves §11.1 / D11).** On the box:
  `python3 - <<'PY'
import urllib.request
req = urllib.request.Request("https://thispersondoesnotexist.com/", headers={"User-Agent":"Mozilla/5.0"})
data = urllib.request.urlopen(req, timeout=15).read()
print("bytes:", len(data), "jpeg:", data[:3] == b"\xff\xd8\xff")
PY`
  Expected: a few hundred KB and `jpeg: True`.
  If it FAILS (firewalled): record the fallback in `PLAN.md` — download faces locally, sync JPEGs to the box; Part B `generate` then runs locally.

- [ ] **Step 3: Confirm Point-MAE repo + checkpoint reachable (resolves §11.2).** On the box:
  `git ls-remote https://github.com/Pang-Yatian/Point-MAE.git HEAD`
  Expected: a hash. Note the pretrained checkpoint URL from the repo README (the ShapeNet/ScanObjectNN pretrain `.pth`) and record it in `PLAN.md` for `setup_encoders.sh`.

- [ ] **Step 4: Record findings + commit.** Append results to `PLAN.md` Progress log (internet yes/no, GPU name, Point-MAE checkpoint URL).
  ```bash
  git add PLAN.md && git commit -m "chore: record elem-danit1 infra verification"
  ```

### Task 0.2: Project skeleton + requirements

**Files:**
- Create: `requirements.txt`, `README.md`, `config/default.yaml`
- Create: `src/__init__.py`, `src/core/__init__.py`, `src/part_a/__init__.py`, `src/part_a/extractors/__init__.py`, `src/part_b/__init__.py`, `src/part_b/extractors/__init__.py`, `src/utils/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create package dirs with `__init__.py`.**
  ```bash
  mkdir -p src/core src/part_a/extractors src/part_b/extractors src/utils tests scripts config outputs data
  touch src/__init__.py src/core/__init__.py src/part_a/__init__.py \
        src/part_a/extractors/__init__.py src/part_b/__init__.py \
        src/part_b/extractors/__init__.py src/utils/__init__.py tests/__init__.py
  echo "*" > outputs/.gitkeep_dummy && rm outputs/.gitkeep_dummy  # outputs/ is gitignored
  ```

- [ ] **Step 2: Write `requirements.txt`** (pinned majors; Point-MAE excluded — see `setup_encoders.sh`).
  ```
  numpy==1.26.4
  scipy==1.13.1
  trimesh==4.4.3
  pygltflib==1.16.2
  matplotlib==3.9.0
  pillow==10.4.0
  scikit-learn==1.5.1
  umap-learn==0.5.6
  hdbscan==0.8.37
  torch==2.3.1
  transformers==4.43.3
  insightface==0.7.3
  onnxruntime==1.18.1
  requests==2.32.3
  pyyaml==6.0.1
  pytest==8.2.2
  ```
  > Note during execution: on the A100 box, install `onnxruntime-gpu` instead of `onnxruntime` if GPU inference for InsightFace is wanted; pin the torch build to the box's CUDA.

- [ ] **Step 3: Write `config/default.yaml`** (the single source of paths + tunables).
  ```yaml
  seed: 42

  paths:
    assets_dir: assets
    outputs_dir: outputs
    data_dir: data

  reduce:
    preprocess: [standardize, l2norm]   # applied in order before clustering
    pca_components: null                # null = skip PCA
    umap:
      n_neighbors: 10
      min_dist: 0.1
      metric: cosine

  part_a:
    encoders_2d: [dinov2]               # primary; optional later: clip, pe_core
    encoders_3d: [point_mae]            # primary; optional later: openshape
    render:
      size_px: 512
      supersample: 3
      views:                            # [elevation, azimuth] degrees
        - [80, -90]
        - [20, -90]
        - [20, 0]
        - [20, 90]
    point_sampling:
      n_points: 8192
    dinov2:
      hf_model: facebook/dinov2-base
    point_mae:
      checkpoint: vendor/Point-MAE/checkpoints/pretrain.pth   # confirmed in Task 0.1
    clustering:
      algorithms: [kmeans, agglomerative]
      k_min: 2
      k_max: 8

  part_b:
    n_images: 500
    tpdne_url: https://thispersondoesnotexist.com/
    request_delay_s: 1.0
    max_retries: 5
    insightface:
      model_name: buffalo_l
      det_size: 640
    clustering:
      algorithms: [kmeans, agglomerative, hdbscan]
      k_min: 2
      k_max: 12
  ```

- [ ] **Step 4: Write `README.md` stub** (sections filled incrementally per D10).
  ```markdown
  # Embedding Clustering

  Unsupervised clustering of (A) 3D glasses assets and (B) AI-generated faces.

  ## Setup
  _TBD — Phase 0._

  ## Usage
  _TBD._

  ## Part A — 3D glasses
  _TBD._

  ## Part B — Faces
  _TBD._

  ## Findings
  _TBD._
  ```
  > The README "TBD"s are temporary scaffolding filled by later tasks; they must all be gone before the final commit (enforced in Task 5.3).

- [ ] **Step 5: Commit.**
  ```bash
  git add -A && git commit -m "chore: project skeleton, requirements, default config"
  ```

### Task 0.3: Logging setup

**Files:**
- Create: `src/logging_setup.py`
- Test: `tests/test_logging_setup.py`

- [ ] **Step 1: Write the failing test.**
  ```python
  # tests/test_logging_setup.py
  import logging
  from src.logging_setup import configure_logging

  def test_configure_logging_sets_level_and_is_idempotent():
      configure_logging("DEBUG")
      root = logging.getLogger()
      assert root.level == logging.DEBUG
      n_handlers = len(root.handlers)
      configure_logging("INFO")            # second call must not stack handlers
      assert len(root.handlers) == n_handlers
      assert root.level == logging.INFO
  ```

- [ ] **Step 2: Run to verify it fails.** `pytest tests/test_logging_setup.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `src/logging_setup.py`.**
  ```python
  """Central logging configuration. Call configure_logging() once at program start.

  No module should call logging.basicConfig or use print() for operational output.
  """
  from __future__ import annotations

  import logging

  _FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


  def configure_logging(level: str = "INFO") -> None:
      """Configure root logging idempotently to the given level.

      Replaces existing handlers so repeated calls don't stack duplicates.
      """
      root = logging.getLogger()
      for h in list(root.handlers):
          root.removeHandler(h)
      handler = logging.StreamHandler()
      handler.setFormatter(logging.Formatter(_FORMAT))
      root.addHandler(handler)
      root.setLevel(getattr(logging, level.upper()))
  ```

- [ ] **Step 4: Run to verify it passes.** `pytest tests/test_logging_setup.py -v` → PASS.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/logging_setup.py tests/test_logging_setup.py
  git commit -m "feat: central idempotent logging setup"
  ```

### Task 0.4: Typed config loader

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test.**
  ```python
  # tests/test_config.py
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
  ```
  > The second test loads a minimal YAML; `load_config` must tolerate missing optional sections by filling dataclass defaults. Ensure dataclasses below have defaults for every nested section.

- [ ] **Step 2: Run to verify it fails.** `pytest tests/test_config.py -v` → FAIL.

- [ ] **Step 3: Implement `src/config.py`.**
  ```python
  """Typed configuration: YAML -> frozen dataclasses, with dotted-key CLI overrides.

  No other module reads YAML or hardcodes paths; they consume a Config object.
  """
  from __future__ import annotations

  from dataclasses import dataclass, field, fields, is_dataclass
  from pathlib import Path
  from typing import Any, Mapping

  import yaml


  @dataclass(frozen=True)
  class Paths:
      assets_dir: str = "assets"
      outputs_dir: str = "outputs"
      data_dir: str = "data"


  @dataclass(frozen=True)
  class UMAPCfg:
      n_neighbors: int = 10
      min_dist: float = 0.1
      metric: str = "cosine"


  @dataclass(frozen=True)
  class ReduceCfg:
      preprocess: tuple[str, ...] = ("standardize", "l2norm")
      pca_components: int | None = None
      umap: UMAPCfg = field(default_factory=UMAPCfg)


  @dataclass(frozen=True)
  class RenderCfg:
      size_px: int = 512
      supersample: int = 3
      views: tuple[tuple[float, float], ...] = ((80.0, -90.0),)


  @dataclass(frozen=True)
  class PointSamplingCfg:
      n_points: int = 8192


  @dataclass(frozen=True)
  class DinoCfg:
      hf_model: str = "facebook/dinov2-base"


  @dataclass(frozen=True)
  class PointMAECfg:
      checkpoint: str = "vendor/Point-MAE/checkpoints/pretrain.pth"


  @dataclass(frozen=True)
  class ClusteringCfg:
      algorithms: tuple[str, ...] = ("kmeans", "agglomerative")
      k_min: int = 2
      k_max: int = 8


  @dataclass(frozen=True)
  class PartACfg:
      encoders_2d: tuple[str, ...] = ("dinov2",)
      encoders_3d: tuple[str, ...] = ("point_mae",)
      render: RenderCfg = field(default_factory=RenderCfg)
      point_sampling: PointSamplingCfg = field(default_factory=PointSamplingCfg)
      dinov2: DinoCfg = field(default_factory=DinoCfg)
      point_mae: PointMAECfg = field(default_factory=PointMAECfg)
      clustering: ClusteringCfg = field(default_factory=ClusteringCfg)


  @dataclass(frozen=True)
  class InsightFaceCfg:
      model_name: str = "buffalo_l"
      det_size: int = 640


  @dataclass(frozen=True)
  class PartBCfg:
      n_images: int = 500
      tpdne_url: str = "https://thispersondoesnotexist.com/"
      request_delay_s: float = 1.0
      max_retries: int = 5
      insightface: InsightFaceCfg = field(default_factory=InsightFaceCfg)
      clustering: ClusteringCfg = field(
          default_factory=lambda: ClusteringCfg(
              algorithms=("kmeans", "agglomerative", "hdbscan"), k_min=2, k_max=12
          )
      )


  @dataclass(frozen=True)
  class Config:
      seed: int = 42
      paths: Paths = field(default_factory=Paths)
      reduce: ReduceCfg = field(default_factory=ReduceCfg)
      part_a: PartACfg = field(default_factory=PartACfg)
      part_b: PartBCfg = field(default_factory=PartBCfg)


  def _build(dc_type: type, data: Mapping[str, Any] | None) -> Any:
      """Recursively build a (frozen) dataclass from a mapping, applying defaults.

      Tuples-of-tuples (e.g. render.views) are coerced from YAML lists.
      """
      if data is None:
          return dc_type()
      kwargs: dict[str, Any] = {}
      for f in fields(dc_type):
          if f.name not in data:
              continue
          val = data[f.name]
          ftype = f.type
          if is_dataclass(ftype) and isinstance(val, Mapping):
              kwargs[f.name] = _build(ftype, val)
          elif isinstance(val, list):
              kwargs[f.name] = tuple(
                  tuple(x) if isinstance(x, list) else x for x in val
              )
          else:
              kwargs[f.name] = val
      return dc_type(**kwargs)


  def _apply_override(cfg: Config, dotted: str, value: Any) -> Config:
      """Return a new Config with one dotted-path field replaced. Raises KeyError if absent."""
      from dataclasses import replace

      parts = dotted.split(".")
      # Walk to the parent dataclass, validating each segment exists.
      def walk(obj: Any, segs: list[str]) -> Any:
          if len(segs) == 1:
              if not any(f.name == segs[0] for f in fields(obj)):
                  raise KeyError(dotted)
              return replace(obj, **{segs[0]: value})
          head, *rest = segs
          if not any(f.name == head for f in fields(obj)):
              raise KeyError(dotted)
          child = getattr(obj, head)
          return replace(obj, **{head: walk(child, rest)})

      return walk(cfg, parts)


  def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Config:
      """Load YAML into a frozen Config, then apply dotted-key overrides (CLI flags)."""
      raw = yaml.safe_load(Path(path).read_text()) or {}
      cfg = _build(Config, raw)
      for key, val in (overrides or {}).items():
          cfg = _apply_override(cfg, key, val)
      return cfg
  ```
  > Note: `f.type` may be a string under `from __future__ import annotations`. To keep `_build` robust, the dataclasses above are simple enough that during execution you may resolve types via `typing.get_type_hints(dc_type)` instead of `f.type` if string-annotation issues arise. Add that resolution if the first test run shows dataclass fields not being built.

- [ ] **Step 4: Run to verify it passes.** `pytest tests/test_config.py -v` → PASS. If nested dataclasses aren't built (string-annotation issue), switch `_build` to use `typing.get_type_hints`.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/config.py tests/test_config.py
  git commit -m "feat: typed YAML config loader with dotted overrides"
  ```

### Task 0.5: Seeding + io utils

**Files:**
- Create: `src/utils/seeding.py`, `src/utils/io.py`
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write the failing test.**
  ```python
  # tests/test_utils.py
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
  ```

- [ ] **Step 2: Run to verify it fails.** `pytest tests/test_utils.py -v` → FAIL.

- [ ] **Step 3: Implement both utils.**
  ```python
  # src/utils/seeding.py
  """Global determinism for reproducible clusters/UMAP (D8)."""
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
  ```
  ```python
  # src/utils/io.py
  """Small filesystem helpers shared across parts."""
  from __future__ import annotations

  from pathlib import Path


  def sanitize_id(s: str) -> str:
      """Make a string safe for a filename: keep alnum/-_. , replace the rest with '_'."""
      return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


  def ensure_dir(path: str | Path) -> Path:
      """Create a directory (and parents) if missing; return it as a Path."""
      p = Path(path)
      p.mkdir(parents=True, exist_ok=True)
      return p
  ```

- [ ] **Step 4: Run to verify it passes.** `pytest tests/test_utils.py -v` → PASS.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/utils tests/test_utils.py
  git commit -m "feat: seeding + io utility helpers"
  ```

---

## Phase 1 — Shared core pipeline

### Task 1.1: Core types (Asset, Embeddings, FeatureExtractor Protocol)

**Files:**
- Create: `src/core/types.py`
- Test: `tests/core/test_types.py`

- [ ] **Step 1: Write the failing test.**
  ```python
  # tests/core/test_types.py
  import numpy as np
  from src.core.types import Embeddings

  def test_embeddings_validates_alignment():
      import pytest
      Embeddings(vectors=np.zeros((2, 4)), ids=["a", "b"], name="x")  # ok
      with pytest.raises(ValueError):
          Embeddings(vectors=np.zeros((2, 4)), ids=["a"], name="x")   # mismatch
  ```
  (Create `tests/core/__init__.py` too.)

- [ ] **Step 2: Run to verify it fails.** `pytest tests/core/test_types.py -v` → FAIL.

- [ ] **Step 3: Implement `src/core/types.py`.**
  ```python
  """Core data contracts shared by every part.

  An Asset is one input item (a GLB mesh, or a face image). An Embeddings bundle pairs
  an (N, D) matrix with its N ids. A FeatureExtractor turns Assets into Embeddings.
  """
  from __future__ import annotations

  from dataclasses import dataclass
  from pathlib import Path
  from typing import Protocol, Sequence, runtime_checkable

  import numpy as np


  @dataclass(frozen=True)
  class Asset:
      """One input item. `path` points at the source file; `id` is its stable identifier."""
      id: str
      path: Path


  @dataclass(frozen=True)
  class Embeddings:
      """An (N, D) embedding matrix with aligned ids and the producing extractor's name."""
      vectors: np.ndarray
      ids: list[str]
      name: str

      def __post_init__(self) -> None:
          if self.vectors.ndim != 2:
              raise ValueError(f"vectors must be 2D, got shape {self.vectors.shape}")
          if self.vectors.shape[0] != len(self.ids):
              raise ValueError(
                  f"row count {self.vectors.shape[0]} != id count {len(self.ids)}"
              )


  @runtime_checkable
  class FeatureExtractor(Protocol):
      """Turns a sequence of Assets into one Embeddings bundle. `name` keys the cache."""
      name: str

      def extract(self, items: Sequence[Asset]) -> Embeddings: ...
  ```

- [ ] **Step 4: Run to verify it passes.** `pytest tests/core/test_types.py -v` → PASS.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/core/types.py tests/core/
  git commit -m "feat: core types (Asset, Embeddings, FeatureExtractor)"
  ```

### Task 1.2: Embedding store (cache + alignment guard)

**Files:**
- Create: `src/core/embedding_store.py`
- Test: `tests/core/test_embedding_store.py`

- [ ] **Step 1: Write the failing test.**
  ```python
  # tests/core/test_embedding_store.py
  import json
  import numpy as np
  import pytest
  from src.core.types import Embeddings
  from src.core.embedding_store import save_embeddings, load_embeddings

  def test_round_trip(tmp_path):
      emb = Embeddings(np.arange(6, dtype=float).reshape(3, 2), ["a", "b", "c"], "dinov2")
      save_embeddings(emb, tmp_path)
      loaded = load_embeddings("dinov2", tmp_path)
      assert loaded.ids == ["a", "b", "c"]
      assert loaded.name == "dinov2"
      assert np.allclose(loaded.vectors, emb.vectors)

  def test_load_detects_corrupted_alignment(tmp_path):
      emb = Embeddings(np.zeros((2, 2)), ["a", "b"], "x")
      save_embeddings(emb, tmp_path)
      # Corrupt ids.json to simulate a desync.
      (tmp_path / "x.ids.json").write_text(json.dumps(["only_one"]))
      with pytest.raises(ValueError):
          load_embeddings("x", tmp_path)
  ```

- [ ] **Step 2: Run to verify it fails.** `pytest tests/core/test_embedding_store.py -v` → FAIL.

- [ ] **Step 3: Implement `src/core/embedding_store.py`.**
  ```python
  """Persist/restore Embeddings as <name>.npy + <name>.ids.json under a directory.

  This cache is the seam between the GPU encode (run once on the box) and the cheap
  analysis loop. load_embeddings HARD-fails on row/id misalignment (D12) because a
  silent desync would invalidate every downstream comparison.
  """
  from __future__ import annotations

  import json
  import logging
  from pathlib import Path

  import numpy as np

  from src.core.types import Embeddings
  from src.utils.io import ensure_dir

  log = logging.getLogger(__name__)


  def save_embeddings(emb: Embeddings, out_dir: str | Path) -> Path:
      """Write emb to <out_dir>/<name>.npy and <name>.ids.json. Returns the .npy path."""
      out = ensure_dir(out_dir)
      npy = out / f"{emb.name}.npy"
      np.save(npy, emb.vectors)
      (out / f"{emb.name}.ids.json").write_text(json.dumps(emb.ids))
      log.info("Saved embeddings '%s' %s -> %s", emb.name, emb.vectors.shape, npy)
      return npy


  def load_embeddings(name: str, out_dir: str | Path) -> Embeddings:
      """Load embeddings by name; validates (N,D) <-> ids alignment via Embeddings.__post_init__."""
      out = Path(out_dir)
      vectors = np.load(out / f"{name}.npy")
      ids = json.loads((out / f"{name}.ids.json").read_text())
      return Embeddings(vectors=vectors, ids=ids, name=name)  # __post_init__ guards alignment
  ```

- [ ] **Step 4: Run to verify it passes.** `pytest tests/core/test_embedding_store.py -v` → PASS.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/core/embedding_store.py tests/core/test_embedding_store.py
  git commit -m "feat: embedding store with id-alignment guard"
  ```

### Task 1.3: Reduce (standardize / L2-norm / PCA / UMAP)

**Files:**
- Create: `src/core/reduce.py`
- Test: `tests/core/test_reduce.py`

- [ ] **Step 1: Write the failing test.**
  ```python
  # tests/core/test_reduce.py
  import numpy as np
  from src.core.reduce import preprocess, umap_2d

  def test_standardize_then_l2norm():
      X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      out = preprocess(X, ["standardize", "l2norm"])
      assert np.allclose(np.linalg.norm(out, axis=1), 1.0)   # rows unit-norm

  def test_pca_reduces_dim():
      X = np.random.RandomState(0).rand(20, 10)
      out = preprocess(X, [], pca_components=3)
      assert out.shape == (20, 3)

  def test_umap_2d_shape():
      X = np.random.RandomState(0).rand(30, 8)
      emb = umap_2d(X, n_neighbors=5, min_dist=0.1, metric="euclidean", seed=0)
      assert emb.shape == (30, 2)
  ```

- [ ] **Step 2: Run to verify it fails.** `pytest tests/core/test_reduce.py -v` → FAIL.

- [ ] **Step 3: Implement `src/core/reduce.py`.**
  ```python
  """Embedding preprocessing + 2D projection for visualization.

  preprocess() applies a configurable, ordered chain (standardize, l2norm) then optional
  PCA. umap_2d() projects to 2D for scatter plots only — clustering runs on preprocess()
  output, not on the UMAP coords.
  """
  from __future__ import annotations

  import logging
  from typing import Sequence

  import numpy as np
  from sklearn.decomposition import PCA
  from sklearn.preprocessing import StandardScaler, normalize

  log = logging.getLogger(__name__)


  def preprocess(
      X: np.ndarray,
      steps: Sequence[str],
      pca_components: int | None = None,
  ) -> np.ndarray:
      """Apply ordered steps ('standardize', 'l2norm') then optional PCA. Returns new array."""
      out = np.asarray(X, dtype=float)
      for step in steps:
          if step == "standardize":
              out = StandardScaler().fit_transform(out)
          elif step == "l2norm":
              out = normalize(out, norm="l2", axis=1)
          else:
              raise ValueError(f"unknown preprocess step: {step!r}")
      if pca_components:
          n = min(pca_components, *out.shape)
          out = PCA(n_components=n, random_state=0).fit_transform(out)
          log.info("PCA -> %d dims", n)
      return out


  def umap_2d(
      X: np.ndarray, n_neighbors: int, min_dist: float, metric: str, seed: int
  ) -> np.ndarray:
      """Project X to 2D with UMAP for plotting. n_neighbors auto-capped at N-1."""
      import umap

      n = max(2, min(n_neighbors, X.shape[0] - 1))
      reducer = umap.UMAP(
          n_components=2, n_neighbors=n, min_dist=min_dist, metric=metric, random_state=seed
      )
      return reducer.fit_transform(X)
  ```
  > Note: with `random_state` set, UMAP runs single-threaded and prints a numba warning — that's expected and required for determinism (D8).

- [ ] **Step 4: Run to verify it passes.** `pytest tests/core/test_reduce.py -v` → PASS.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/core/reduce.py tests/core/test_reduce.py
  git commit -m "feat: embedding preprocess + UMAP projection"
  ```

### Task 1.4: Cluster (KMeans/Agglomerative/HDBSCAN + k selection)

**Files:**
- Create: `src/core/cluster.py`
- Test: `tests/core/test_cluster.py`

- [ ] **Step 1: Write the failing test.**
  ```python
  # tests/core/test_cluster.py
  import numpy as np
  from sklearn.datasets import make_blobs
  from src.core.cluster import cluster, ClusterResult

  def _blobs(k=3, n=90, seed=0):
      X, _ = make_blobs(n_samples=n, centers=k, cluster_std=0.6, random_state=seed)
      return X

  def test_kmeans_recovers_k_via_silhouette():
      X = _blobs(k=3)
      res = cluster(X, algorithm="kmeans", k_min=2, k_max=6, seed=0)
      assert isinstance(res, ClusterResult)
      assert res.n_clusters == 3
      assert res.labels.shape == (90,)

  def test_agglomerative_runs():
      X = _blobs(k=4)
      res = cluster(X, algorithm="agglomerative", k_min=2, k_max=6, seed=0)
      assert res.n_clusters == 4

  def test_hdbscan_returns_labels():
      X = _blobs(k=3, n=120)
      res = cluster(X, algorithm="hdbscan", k_min=2, k_max=6, seed=0)
      # HDBSCAN may label some points -1 (noise); at least it produces >=1 cluster.
      assert res.labels.shape == (120,)
      assert res.n_clusters >= 1
  ```

- [ ] **Step 2: Run to verify it fails.** `pytest tests/core/test_cluster.py -v` → FAIL.

- [ ] **Step 3: Implement `src/core/cluster.py`.**
  ```python
  """Clustering with automatic k-selection.

  KMeans and Agglomerative sweep k in [k_min, k_max] and pick the k with the best cosine
  silhouette. HDBSCAN needs no k (density-based) and may mark noise as label -1.
  """
  from __future__ import annotations

  import logging
  from dataclasses import dataclass

  import numpy as np
  from sklearn.cluster import AgglomerativeClustering, KMeans
  from sklearn.metrics import silhouette_score

  log = logging.getLogger(__name__)


  @dataclass(frozen=True)
  class ClusterResult:
      """Labels per row plus metadata about the chosen clustering."""
      labels: np.ndarray
      n_clusters: int
      algorithm: str
      k_selected: int | None  # None for HDBSCAN


  def _best_k(X: np.ndarray, make, k_min: int, k_max: int) -> tuple[np.ndarray, int]:
      """Sweep k, return (labels, k) maximizing cosine silhouette. k_max capped at N-1."""
      best_labels, best_k, best_score = None, None, -1.0
      hi = min(k_max, X.shape[0] - 1)
      for k in range(max(2, k_min), hi + 1):
          labels = make(k).fit_predict(X)
          if len(set(labels)) < 2:
              continue
          score = silhouette_score(X, labels, metric="cosine")
          log.debug("k=%d silhouette=%.4f", k, score)
          if score > best_score:
              best_labels, best_k, best_score = labels, k, score
      if best_labels is None:  # degenerate fallback
          best_labels, best_k = make(2).fit_predict(X), 2
      return best_labels, best_k


  def cluster(
      X: np.ndarray, algorithm: str, k_min: int, k_max: int, seed: int
  ) -> ClusterResult:
      """Cluster X with the named algorithm. Returns a ClusterResult."""
      if algorithm == "kmeans":
          labels, k = _best_k(
              X, lambda k: KMeans(n_clusters=k, n_init=10, random_state=seed), k_min, k_max
          )
          return ClusterResult(labels, len(set(labels)), "kmeans", k)
      if algorithm == "agglomerative":
          labels, k = _best_k(
              X, lambda k: AgglomerativeClustering(n_clusters=k), k_min, k_max
          )
          return ClusterResult(labels, len(set(labels)), "agglomerative", k)
      if algorithm == "hdbscan":
          import hdbscan

          labels = hdbscan.HDBSCAN(min_cluster_size=max(5, X.shape[0] // 20)).fit_predict(X)
          n = len(set(labels) - {-1})
          return ClusterResult(labels, n, "hdbscan", None)
      raise ValueError(f"unknown algorithm: {algorithm!r}")
  ```

- [ ] **Step 4: Run to verify it passes.** `pytest tests/core/test_cluster.py -v` → PASS.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/core/cluster.py tests/core/test_cluster.py
  git commit -m "feat: clustering with silhouette-based k selection"
  ```

### Task 1.5: Metrics (internal + external)

**Files:**
- Create: `src/core/metrics.py`
- Test: `tests/core/test_metrics.py`

- [ ] **Step 1: Write the failing test.**
  ```python
  # tests/core/test_metrics.py
  import numpy as np
  from sklearn.datasets import make_blobs
  from src.core.metrics import internal_metrics, external_metrics

  def test_internal_metrics_keys_and_ranges():
      X, y = make_blobs(n_samples=60, centers=3, cluster_std=0.5, random_state=0)
      m = internal_metrics(X, y)
      assert set(m) == {"silhouette", "davies_bouldin", "calinski_harabasz"}
      assert -1.0 <= m["silhouette"] <= 1.0
      assert m["davies_bouldin"] >= 0.0

  def test_external_metrics_perfect_match():
      labels = np.array([0, 0, 1, 1, 2, 2])
      truth = np.array(["a", "a", "b", "b", "c", "c"])
      m = external_metrics(labels, truth)
      assert m["nmi"] == 1.0 and m["ari"] == 1.0 and m["purity"] == 1.0
  ```

- [ ] **Step 2: Run to verify it fails.** `pytest tests/core/test_metrics.py -v` → FAIL.

- [ ] **Step 3: Implement `src/core/metrics.py`.**
  ```python
  """Cluster-quality metrics.

  internal_metrics: no labels needed (silhouette/DB/CH). external_metrics: compare cluster
  labels against pseudo-labels (Part B's InsightFace age/gender) via NMI/ARI/purity.
  """
  from __future__ import annotations

  import numpy as np
  from sklearn.metrics import (
      adjusted_rand_score,
      calinski_harabasz_score,
      davies_bouldin_score,
      normalized_mutual_info_score,
      silhouette_score,
  )


  def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
      """Silhouette (cosine), Davies-Bouldin, Calinski-Harabasz on X given labels.

      Noise points (label -1, HDBSCAN) are excluded from the computation.
      """
      mask = labels != -1
      Xv, lv = X[mask], labels[mask]
      if len(set(lv)) < 2:
          return {"silhouette": float("nan"), "davies_bouldin": float("nan"),
                  "calinski_harabasz": float("nan")}
      return {
          "silhouette": float(silhouette_score(Xv, lv, metric="cosine")),
          "davies_bouldin": float(davies_bouldin_score(Xv, lv)),
          "calinski_harabasz": float(calinski_harabasz_score(Xv, lv)),
      }


  def _purity(labels: np.ndarray, truth: np.ndarray) -> float:
      """Fraction of points in the majority truth-class of their assigned cluster."""
      total, correct = len(labels), 0
      for c in set(labels):
          members = truth[labels == c]
          if len(members):
              vals, counts = np.unique(members, return_counts=True)
              correct += counts.max()
      return correct / total


  def external_metrics(labels: np.ndarray, truth: np.ndarray) -> dict[str, float]:
      """NMI / ARI / purity of cluster labels vs categorical pseudo-labels."""
      return {
          "nmi": float(normalized_mutual_info_score(truth, labels)),
          "ari": float(adjusted_rand_score(truth, labels)),
          "purity": float(_purity(labels, truth)),
      }
  ```

- [ ] **Step 4: Run to verify it passes.** `pytest tests/core/test_metrics.py -v` → PASS.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/core/metrics.py tests/core/test_metrics.py
  git commit -m "feat: internal + external cluster metrics"
  ```

### Task 1.6: Visualize (scatter, montages, metric table)

**Files:**
- Create: `src/core/visualize.py`
- Test: `tests/core/test_visualize.py`

- [ ] **Step 1: Write the failing test** (assert files are produced, not pixel content).
  ```python
  # tests/core/test_visualize.py
  import numpy as np
  from src.core.visualize import scatter_2d, metric_table_png, cluster_montage

  def test_scatter_writes_png(tmp_path):
      pts = np.random.RandomState(0).rand(20, 2)
      labels = np.array([0, 1] * 10)
      out = scatter_2d(pts, labels, tmp_path / "s.png", title="t")
      assert out.exists() and out.stat().st_size > 0

  def test_metric_table_writes_png(tmp_path):
      rows = {"dinov2": {"silhouette": 0.5}, "point_mae": {"silhouette": 0.3}}
      out = metric_table_png(rows, tmp_path / "tbl.png", title="cmp")
      assert out.exists()

  def test_cluster_montage_writes_png(tmp_path):
      # 4 tiny dummy thumbnails
      from PIL import Image
      imgs = []
      for i in range(4):
          p = tmp_path / f"{i}.png"
          Image.new("RGB", (8, 8), (i * 10, 0, 0)).save(p)
          imgs.append(p)
      labels = np.array([0, 0, 1, 1])
      out = cluster_montage(imgs, labels, tmp_path / "m.png")
      assert out.exists()
  ```

- [ ] **Step 2: Run to verify it fails.** `pytest tests/core/test_visualize.py -v` → FAIL.

- [ ] **Step 3: Implement `src/core/visualize.py`.**
  ```python
  """Visualization helpers — all save to image files (the assignment requires saved images).

  Functions are part-agnostic: they take arrays/paths and write a PNG, returning its path.
  """
  from __future__ import annotations

  import logging
  from pathlib import Path
  from typing import Mapping, Sequence

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import numpy as np
  from PIL import Image

  log = logging.getLogger(__name__)


  def scatter_2d(points: np.ndarray, labels: np.ndarray, out_path: str | Path,
                 title: str = "") -> Path:
      """Scatter 2D points colored by integer label; save PNG. Returns the path."""
      out_path = Path(out_path)
      fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
      sc = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=40)
      ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
      fig.colorbar(sc, ax=ax, label="cluster")
      fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
      return out_path


  def metric_table_png(rows: Mapping[str, Mapping[str, float]], out_path: str | Path,
                       title: str = "") -> Path:
      """Render a {row_name: {metric: value}} table as a PNG (for the README/report)."""
      out_path = Path(out_path)
      metrics = sorted({m for r in rows.values() for m in r})
      cell_text = [[f"{rows[r].get(m, float('nan')):.3f}" for m in metrics] for r in rows]
      fig, ax = plt.subplots(figsize=(2 + 1.6 * len(metrics), 1 + 0.5 * len(rows)), dpi=120)
      ax.axis("off"); ax.set_title(title)
      tbl = ax.table(cellText=cell_text, rowLabels=list(rows), colLabels=metrics,
                     loc="center")
      tbl.scale(1, 1.4)
      fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
      return out_path


  def cluster_montage(image_paths: Sequence[str | Path], labels: np.ndarray,
                      out_path: str | Path, thumb_px: int = 96) -> Path:
      """Grid of thumbnails grouped by cluster (one row per cluster). Save PNG."""
      out_path = Path(out_path)
      by_cluster: dict[int, list[Path]] = {}
      for p, lab in zip(image_paths, labels):
          by_cluster.setdefault(int(lab), []).append(Path(p))
      rows = sorted(by_cluster)
      ncols = max(len(v) for v in by_cluster.values())
      fig, axes = plt.subplots(len(rows), ncols,
                               figsize=(ncols * 1.3, len(rows) * 1.3), dpi=120,
                               squeeze=False)
      for r, c_lab in enumerate(rows):
          for col in range(ncols):
              ax = axes[r][col]; ax.axis("off")
              if col < len(by_cluster[c_lab]):
                  img = Image.open(by_cluster[c_lab][col]).convert("RGB").resize(
                      (thumb_px, thumb_px))
                  ax.imshow(np.asarray(img))
          axes[r][0].set_ylabel(f"c{c_lab}", rotation=0, labelpad=18, va="center")
      fig.suptitle("Clusters")
      fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
      return out_path
  ```

- [ ] **Step 4: Run to verify it passes.** `pytest tests/core/test_visualize.py -v` → PASS.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/core/visualize.py tests/core/test_visualize.py
  git commit -m "feat: visualization (scatter, montage, metric table)"
  ```

---

## Phase 2 — Part A (3D glasses)

### Task 2.1: Mesh IO (load GLB → single mesh, sample points, colors)

**Files:**
- Create: `src/part_a/mesh_io.py`
- Test: `tests/part_a/test_mesh_io.py`

- [ ] **Step 1: Write the failing test** (synthetic mesh; no real GLB needed).
  ```python
  # tests/part_a/test_mesh_io.py  (create tests/part_a/__init__.py too)
  import numpy as np
  import trimesh
  from src.part_a.mesh_io import to_single_mesh, sample_surface_points

  def _scene_with_two_boxes():
      a = trimesh.creation.box(extents=(1, 1, 1))
      b = trimesh.creation.box(extents=(1, 1, 1)); b.apply_translation([3, 0, 0])
      return trimesh.Scene([a, b])

  def test_to_single_mesh_concatenates_scene():
      mesh = to_single_mesh(_scene_with_two_boxes())
      assert isinstance(mesh, trimesh.Trimesh)
      assert len(mesh.vertices) > 0 and len(mesh.faces) > 0

  def test_sample_surface_points_shape():
      mesh = trimesh.creation.box(extents=(1, 1, 1))
      pts = sample_surface_points(mesh, n_points=1024, seed=0)
      assert pts.shape == (1024, 3)
      # deterministic with fixed seed
      pts2 = sample_surface_points(mesh, n_points=1024, seed=0)
      assert np.allclose(pts, pts2)
  ```

- [ ] **Step 2: Run to verify it fails.** `pytest tests/part_a/test_mesh_io.py -v` → FAIL.

- [ ] **Step 3: Implement `src/part_a/mesh_io.py`.**
  ```python
  """Load GLB glasses meshes and derive geometry inputs.

  GLBs load as a trimesh.Scene (multiple mesh components + materials). to_single_mesh
  concatenates them into one Trimesh. sample_surface_points draws uniform points from the
  triangulated surface (input to the Point-MAE encoder; D1/D5 — surface, not point cloud
  of vertices).
  """
  from __future__ import annotations

  import logging
  from pathlib import Path

  import numpy as np
  import trimesh

  log = logging.getLogger(__name__)


  def load_glb(path: str | Path) -> trimesh.Scene | trimesh.Trimesh:
      """Load a .glb via trimesh. May return a Scene (multi-mesh) or a single Trimesh."""
      return trimesh.load(str(path), process=False)


  def to_single_mesh(obj: trimesh.Scene | trimesh.Trimesh) -> trimesh.Trimesh:
      """Concatenate a Scene's geometries into one Trimesh; pass a Trimesh through.

      Logs a warning if the GLB had no geometry (caller skips such files).
      """
      if isinstance(obj, trimesh.Trimesh):
          return obj
      geoms = list(obj.geometry.values())
      if not geoms:
          raise ValueError("GLB contains no mesh geometry")
      if len(geoms) == 1:
          return geoms[0]
      return trimesh.util.concatenate(geoms)


  def sample_surface_points(mesh: trimesh.Trimesh, n_points: int, seed: int) -> np.ndarray:
      """Uniformly sample n_points (N,3) from the mesh surface, normalized to a unit sphere.

      Centering + scaling makes the descriptor translation/scale invariant.
      """
      rng = np.random.RandomState(seed)
      pts, _ = trimesh.sample.sample_surface(mesh, n_points, seed=rng.randint(2**31 - 1))
      pts = np.asarray(pts, dtype=float)
      pts -= pts.mean(axis=0)
      scale = np.linalg.norm(pts, axis=1).max()
      if scale > 0:
          pts /= scale
      return pts
  ```
  > Note: `trimesh.sample.sample_surface`'s `seed` kwarg exists in trimesh 4.x. If the installed version rejects it, seed via the passed `rng` by setting `np.random.seed(seed)` before the call and drop the kwarg.

- [ ] **Step 4: Run to verify it passes.** `pytest tests/part_a/test_mesh_io.py -v` → PASS.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/part_a/mesh_io.py tests/part_a/
  git commit -m "feat: GLB mesh IO + surface point sampling"
  ```

### Task 2.2: Triangulated-mesh renderer

**Files:**
- Create: `src/part_a/render.py`
- Test: `tests/part_a/test_render.py`

- [ ] **Step 1: Write the failing smoke test.**
  ```python
  # tests/part_a/test_render.py
  import trimesh
  from PIL import Image
  from src.part_a.render import render_views

  def test_render_views_produces_images(tmp_path):
      mesh = trimesh.creation.box(extents=(2, 1, 0.2))
      paths = render_views(mesh, "box", tmp_path, size_px=128, supersample=1,
                           views=[(80, -90), (20, 0)])
      assert len(paths) == 2
      for p in paths:
          assert p.exists()
          assert Image.open(p).size == (128, 128) or Image.open(p).width > 0
  ```

- [ ] **Step 2: Run to verify it fails.** `pytest tests/part_a/test_render.py -v` → FAIL.

- [ ] **Step 3: Implement `src/part_a/render.py`** (adapted from the proven umap_viewer renderer; triangulated surface, headless Agg).
  ```python
  """Render a GLB's TRIANGULATED MESH SURFACE to shaded PNGs (D1: mesh, not point cloud).

  Uses matplotlib Poly3DCollection (no GPU/EGL). Renders at supersample x size then
  LANCZOS-downscales for smooth edges. One PNG per view angle.
  """
  from __future__ import annotations

  import io
  import logging
  from pathlib import Path

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import numpy as np
  import trimesh
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection
  from PIL import Image

  from src.utils.io import ensure_dir, sanitize_id

  log = logging.getLogger(__name__)
  _LIGHT = np.array([0.3, 0.2, 0.9]) / np.linalg.norm([0.3, 0.2, 0.9])


  def _vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
      """Return (V,3) float colors in [0,1] from the mesh, or None if unavailable."""
      try:
          vc = mesh.visual.vertex_colors
          if vc is not None and len(vc) == len(mesh.vertices):
              return np.asarray(vc[:, :3], dtype=float) / 255.0
      except Exception:  # noqa: BLE001 - colors are optional; degrade gracefully
          pass
      return None


  def _render_one(mesh: trimesh.Trimesh, size_px: int, elev: float, azim: float,
                  ss: int) -> Image.Image:
      """Rasterize one view to an RGBA PIL image (shaded, anti-aliased)."""
      v = np.asarray(mesh.vertices, dtype=float)
      v = v - v.mean(axis=0)
      max_norm = float(np.linalg.norm(v, axis=1).max()) or 1.0
      v = v / max_norm
      f = np.asarray(mesh.faces)

      vc = _vertex_colors(mesh)
      face_rgb = vc[f].mean(axis=1) if vc is not None else np.full((len(f), 3), 0.6)
      shade = 0.45 + 0.55 * np.clip(np.abs(np.asarray(mesh.face_normals) @ _LIGHT), 0, 1)
      face_rgb = np.clip(face_rgb * shade[:, None], 0, 1)

      dpi = 100
      fig = plt.figure(figsize=(size_px * ss / dpi, size_px * ss / dpi), dpi=dpi)
      ax = fig.add_subplot(111, projection="3d")
      ax.add_collection3d(Poly3DCollection(v[f], facecolors=face_rgb, edgecolors="none"))
      for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
          lim(-0.6, 0.6)
      ax.view_init(elev=elev, azim=azim)
      ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
      buf = io.BytesIO()
      fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0)
      plt.close(fig); buf.seek(0)
      img = Image.open(buf).convert("RGBA")
      if ss > 1:
          img = img.resize((max(1, img.width // ss), max(1, img.height // ss)), Image.LANCZOS)
      return img


  def render_views(mesh: trimesh.Trimesh, asset_id: str, out_dir: str | Path,
                   size_px: int, supersample: int,
                   views: list[tuple[float, float]]) -> list[Path]:
      """Render `mesh` from each (elev, azim) view; save <out_dir>/<id>_v<i>.png. Return paths."""
      out = ensure_dir(out_dir)
      ss = max(1, int(supersample))
      paths: list[Path] = []
      for i, (elev, azim) in enumerate(views):
          img = _render_one(mesh, size_px, elev, azim, ss)
          p = out / f"{sanitize_id(asset_id)}_v{i}.png"
          img.save(p); paths.append(p)
      log.info("Rendered %d views for %s", len(paths), asset_id)
      return paths
  ```

- [ ] **Step 4: Run to verify it passes.** `pytest tests/part_a/test_render.py -v` → PASS.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/part_a/render.py tests/part_a/test_render.py
  git commit -m "feat: triangulated-mesh multi-view renderer"
  ```

### Task 2.3: DINOv2 extractor (2D primary)

**Files:**
- Create: `src/part_a/extractors/dinov2.py`
- Test: `tests/part_a/test_dinov2_extractor.py`

- [ ] **Step 1: Write the contract test** (uses a tiny fake to validate wiring without downloading weights; real model behind `@slow`).
  ```python
  # tests/part_a/test_dinov2_extractor.py
  import numpy as np
  import pytest
  from src.core.types import Asset, FeatureExtractor

  def test_dinov2_implements_protocol():
      from src.part_a.extractors.dinov2 import DINOv2Extractor
      # Construct without loading weights by checking the class satisfies the Protocol shape.
      assert hasattr(DINOv2Extractor, "extract")
      assert "name" in DINOv2Extractor.__annotations__ or hasattr(DINOv2Extractor, "name")

  @pytest.mark.slow
  def test_dinov2_extract_real(tmp_path):
      """Real model + real render. Marked slow: needs torch + network for weights."""
      import trimesh
      from src.part_a.render import render_views
      from src.part_a.extractors.dinov2 import DINOv2Extractor
      mesh = trimesh.creation.box(extents=(2, 1, 0.2))
      render_views(mesh, "box", tmp_path, size_px=128, supersample=1, views=[(80, -90)])
      ext = DINOv2Extractor(hf_model="facebook/dinov2-base", render_dir=tmp_path)
      emb = ext.extract([Asset(id="box", path=tmp_path / "box.glb")])
      assert emb.vectors.shape[0] == 1 and emb.vectors.shape[1] > 100
  ```
  Add to `pyproject.toml`/`pytest.ini`: register the `slow` marker (Task 5.1 sets up `pytest.ini`).

- [ ] **Step 2: Run to verify it fails.** `pytest tests/part_a/test_dinov2_extractor.py::test_dinov2_implements_protocol -v` → FAIL.

- [ ] **Step 3: Implement `src/part_a/extractors/dinov2.py`.**
  ```python
  """2D visual feature: render-based DINOv2 embedding (Part A primary, D3).

  For each asset, load its pre-rendered views, run them through a frozen DINOv2 ViT, and
  mean-pool the per-view CLS embeddings into one vector. Renders are produced beforehand by
  render.py and live under render_dir as <id>_v<k>.png.
  """
  from __future__ import annotations

  import logging
  from pathlib import Path
  from typing import Sequence

  import numpy as np

  from src.core.types import Asset, Embeddings
  from src.utils.io import sanitize_id

  log = logging.getLogger(__name__)


  class DINOv2Extractor:
      """Frozen DINOv2 image embedding over multi-view renders, mean-pooled per asset."""

      def __init__(self, hf_model: str, render_dir: str | Path) -> None:
          self.name = "dinov2"
          self.hf_model = hf_model
          self.render_dir = Path(render_dir)
          self._model = None
          self._processor = None

      def _ensure_model(self) -> None:
          if self._model is None:
              import torch
              from transformers import AutoImageProcessor, AutoModel

              self._processor = AutoImageProcessor.from_pretrained(self.hf_model)
              self._model = AutoModel.from_pretrained(self.hf_model).eval()
              self._device = "cuda" if torch.cuda.is_available() else "cpu"
              self._model.to(self._device)

      def _embed_image(self, path: Path) -> np.ndarray:
          import torch
          from PIL import Image

          img = Image.open(path).convert("RGB")
          inputs = self._processor(images=img, return_tensors="pt").to(self._device)
          with torch.no_grad():
              out = self._model(**inputs)
          # CLS token (pooler_output) is the global image embedding.
          cls = out.last_hidden_state[:, 0, :]
          return cls.squeeze(0).cpu().numpy()

      def extract(self, items: Sequence[Asset]) -> Embeddings:
          self._ensure_model()
          vecs, ids = [], []
          for asset in items:
              views = sorted(self.render_dir.glob(f"{sanitize_id(asset.id)}_v*.png"))
              if not views:
                  log.warning("no renders for %s; skipping", asset.id)
                  continue
              per_view = np.stack([self._embed_image(p) for p in views])
              vecs.append(per_view.mean(axis=0))
              ids.append(asset.id)
          return Embeddings(np.vstack(vecs), ids, self.name)
  ```

- [ ] **Step 4: Run to verify it passes.** `pytest tests/part_a/test_dinov2_extractor.py::test_dinov2_implements_protocol -v` → PASS. (Real test runs later on the box: `pytest -m slow`.)

- [ ] **Step 5: Commit.**
  ```bash
  git add src/part_a/extractors/dinov2.py tests/part_a/test_dinov2_extractor.py
  git commit -m "feat: DINOv2 render-based 2D extractor"
  ```

### Task 2.4: Point-MAE setup script + extractor (3D primary)

**Files:**
- Create: `scripts/setup_encoders.sh`, `src/part_a/extractors/point_mae.py`
- Test: `tests/part_a/test_point_mae_extractor.py`

- [ ] **Step 1: Write `scripts/setup_encoders.sh`** (idempotent bootstrap; run on the box).
  ```bash
  #!/usr/bin/env bash
  # Bootstrap the Point-MAE encoder (vendored; custom ops + checkpoint not pip-installable).
  # Idempotent: safe to re-run. Run on elem-danit1 after `pip install -r requirements.txt`.
  set -euo pipefail
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  VENDOR="$ROOT/vendor"
  mkdir -p "$VENDOR"

  if [ ! -d "$VENDOR/Point-MAE" ]; then
    git clone --depth 1 https://github.com/Pang-Yatian/Point-MAE.git "$VENDOR/Point-MAE"
  fi

  CKPT_DIR="$VENDOR/Point-MAE/checkpoints"
  mkdir -p "$CKPT_DIR"
  # CKPT_URL confirmed in Task 0.1 (Point-MAE pretrain weights). Replace if the repo moves it.
  CKPT_URL="${POINT_MAE_CKPT_URL:?Set POINT_MAE_CKPT_URL to the pretrain .pth URL from Task 0.1}"
  if [ ! -f "$CKPT_DIR/pretrain.pth" ]; then
    wget -O "$CKPT_DIR/pretrain.pth" "$CKPT_URL"
  fi
  echo "Point-MAE ready at $VENDOR/Point-MAE"
  ```
  ```bash
  chmod +x scripts/setup_encoders.sh
  ```

- [ ] **Step 2: Write the contract test.**
  ```python
  # tests/part_a/test_point_mae_extractor.py
  import pytest
  from src.core.types import Asset

  def test_point_mae_implements_protocol():
      from src.part_a.extractors.point_mae import PointMAEExtractor
      assert hasattr(PointMAEExtractor, "extract")

  @pytest.mark.slow
  def test_point_mae_extract_real():
      """Needs vendored Point-MAE + checkpoint on the box."""
      import trimesh
      from src.part_a.extractors.point_mae import PointMAEExtractor
      ext = PointMAEExtractor(checkpoint="vendor/Point-MAE/checkpoints/pretrain.pth",
                              n_points=1024, seed=0)
      mesh_path = "assets/00686245121504.glb"
      emb = ext.extract([Asset(id="m", path=mesh_path)])
      assert emb.vectors.shape[0] == 1 and emb.vectors.shape[1] >= 256
  ```

- [ ] **Step 3: Run to verify it fails.** `pytest tests/part_a/test_point_mae_extractor.py::test_point_mae_implements_protocol -v` → FAIL.

- [ ] **Step 4: Implement `src/part_a/extractors/point_mae.py`.**
  ```python
  """3D geometric feature: Point-MAE embedding over points sampled from the mesh surface.

  (Part A primary, D5.) Self-supervised, pure-geometry — mirrors DINOv2. The vendored
  Point-MAE repo (scripts/setup_encoders.sh) provides the model + checkpoint. We load the
  pretrained encoder, feed (n_points, 3) surface samples, and take the pooled feature.

  NOTE FOR EXECUTOR: the exact import path + forward signature depend on the vendored repo.
  Confirm the encoder class + how to obtain the global feature when running setup on the box;
  the structure below is the integration contract, with the repo-specific call isolated to
  `_load_model` and `_encode`.
  """
  from __future__ import annotations

  import logging
  import sys
  from pathlib import Path
  from typing import Sequence

  import numpy as np

  from src.core.types import Asset, Embeddings
  from src.part_a.mesh_io import load_glb, sample_surface_points, to_single_mesh

  log = logging.getLogger(__name__)
  _VENDOR = Path("vendor/Point-MAE")


  class PointMAEExtractor:
      """Pretrained Point-MAE encoder over mesh-surface point clouds."""

      def __init__(self, checkpoint: str, n_points: int, seed: int) -> None:
          self.name = "point_mae"
          self.checkpoint = checkpoint
          self.n_points = n_points
          self.seed = seed
          self._model = None

      def _load_model(self) -> None:
          if self._model is not None:
              return
          import torch
          if str(_VENDOR) not in sys.path:
              sys.path.insert(0, str(_VENDOR))
          # Repo-specific: build the Point-MAE encoder and load pretrain weights.
          # Confirm against the vendored repo (models/Point_MAE.py) during box setup.
          from models.Point_MAE import Point_MAE  # type: ignore
          cfg = self._default_cfg()
          model = Point_MAE(cfg)
          state = torch.load(self.checkpoint, map_location="cpu")
          model.load_state_dict(state.get("base_model", state), strict=False)
          self._device = "cuda" if torch.cuda.is_available() else "cpu"
          self._model = model.eval().to(self._device)

      @staticmethod
      def _default_cfg():
          """Minimal config object the vendored model expects. Fill from the repo's yaml."""
          from types import SimpleNamespace
          return SimpleNamespace(
              mask_ratio=0.6, mask_type="rand", trans_dim=384, encoder_dims=384,
              depth=12, drop_path_rate=0.1, num_heads=6, group_size=32, num_group=64,
          )

      def _encode(self, pts: np.ndarray) -> np.ndarray:
          import torch
          x = torch.from_numpy(pts).float().unsqueeze(0).to(self._device)  # (1, N, 3)
          with torch.no_grad():
              # Use the encoder to get token features, mean+max pooled (Point-MAE convention).
              feats = self._model.forward_eval(x) if hasattr(self._model, "forward_eval") \
                  else self._model(x)
          feats = feats if isinstance(feats, torch.Tensor) else feats[0]
          return feats.squeeze(0).float().cpu().numpy().reshape(-1)

      def extract(self, items: Sequence[Asset]) -> Embeddings:
          self._load_model()
          vecs, ids = [], []
          for asset in items:
              try:
                  mesh = to_single_mesh(load_glb(asset.path))
                  pts = sample_surface_points(mesh, self.n_points, self.seed)
                  vecs.append(self._encode(pts))
                  ids.append(asset.id)
              except Exception:  # noqa: BLE001 - isolate per-item failures (D12)
                  log.exception("Point-MAE failed on %s; skipping", asset.id)
          return Embeddings(np.vstack(vecs), ids, self.name)
  ```
  > The `_load_model`/`_encode` internals are the only repo-specific code. During box setup, open `vendor/Point-MAE/models/Point_MAE.py`, confirm the class name, the cfg fields, and how to pull a global feature (often concat of mean+max over encoder tokens). Adjust those two methods only; the Protocol/extract loop stays as-is.

- [ ] **Step 5: Run to verify it passes.** `pytest tests/part_a/test_point_mae_extractor.py::test_point_mae_implements_protocol -v` → PASS.

- [ ] **Step 6: Commit.**
  ```bash
  git add scripts/setup_encoders.sh src/part_a/extractors/point_mae.py tests/part_a/test_point_mae_extractor.py
  git commit -m "feat: Point-MAE 3D extractor + encoder setup script"
  ```

### Task 2.5: Part A pipeline + extractor registry

**Files:**
- Create: `src/part_a/pipeline.py`
- Test: `tests/part_a/test_pipeline.py`

- [ ] **Step 1: Write the test** (drives the pipeline with a fake extractor + 2 fake assets so it runs anywhere, no GPU).
  ```python
  # tests/part_a/test_pipeline.py
  import numpy as np
  from src.core.types import Asset, Embeddings
  from src.part_a.pipeline import run_clustering_stage

  class FakeExtractor:
      name = "fake"
      def extract(self, items):
          # two clearly separated groups
          v = np.array([[0, 0], [0.1, 0], [5, 5], [5.1, 5]], dtype=float)
          return Embeddings(v, [a.id for a in items], self.name)

  def test_run_clustering_stage_writes_outputs(tmp_path):
      assets = [Asset(id=f"a{i}", path=tmp_path / f"a{i}.glb") for i in range(4)]
      results = run_clustering_stage(
          extractor=FakeExtractor(), assets=assets, out_dir=tmp_path,
          algorithms=["kmeans"], k_min=2, k_max=3,
          preprocess=["standardize"], pca_components=None,
          umap_cfg={"n_neighbors": 3, "min_dist": 0.1, "metric": "euclidean"}, seed=0,
      )
      assert "kmeans" in results
      assert (tmp_path / "fake.npy").exists()
      assert (tmp_path / "figures").exists()
  ```

- [ ] **Step 2: Run to verify it fails.** `pytest tests/part_a/test_pipeline.py -v` → FAIL.

- [ ] **Step 3: Implement `src/part_a/pipeline.py`.**
  ```python
  """Part A orchestration: discover assets, render, extract, cluster, evaluate, visualize.

  The clustering stage is extractor-agnostic (reused per encoder) — this is the shared
  pipeline from the design. Heavy stages (render/extract) are split out so they can run on
  the box and cache to disk; cluster/viz run on the cached .npy anywhere.
  """
  from __future__ import annotations

  import logging
  from pathlib import Path
  from typing import Sequence

  from src.config import Config
  from src.core import metrics as M
  from src.core.cluster import cluster
  from src.core.embedding_store import load_embeddings, save_embeddings
  from src.core.reduce import preprocess, umap_2d
  from src.core.types import Asset, FeatureExtractor
  from src.core.visualize import metric_table_png, scatter_2d
  from src.utils.io import ensure_dir

  log = logging.getLogger(__name__)


  def discover_assets(assets_dir: str | Path) -> list[Asset]:
      """Build an Asset per .glb in assets_dir (id = filename stem)."""
      d = Path(assets_dir)
      return [Asset(id=p.stem, path=p) for p in sorted(d.glob("*.glb"))]


  def build_extractors(cfg: Config, render_dir: Path) -> list[FeatureExtractor]:
      """Instantiate the configured Part A extractors (2D from renders, 3D from mesh)."""
      exts: list[FeatureExtractor] = []
      for name in cfg.part_a.encoders_2d:
          if name == "dinov2":
              from src.part_a.extractors.dinov2 import DINOv2Extractor
              exts.append(DINOv2Extractor(cfg.part_a.dinov2.hf_model, render_dir))
          else:
              raise ValueError(f"unknown 2D encoder {name!r}")
      for name in cfg.part_a.encoders_3d:
          if name == "point_mae":
              from src.part_a.extractors.point_mae import PointMAEExtractor
              exts.append(PointMAEExtractor(cfg.part_a.point_mae.checkpoint,
                                            cfg.part_a.point_sampling.n_points, cfg.seed))
          else:
              raise ValueError(f"unknown 3D encoder {name!r}")
      return exts


  def run_clustering_stage(extractor, assets: Sequence[Asset], out_dir: str | Path,
                           algorithms: Sequence[str], k_min: int, k_max: int,
                           preprocess: Sequence[str], pca_components, umap_cfg: dict,
                           seed: int) -> dict:
      """Extract (or reuse cached) embeddings, cluster with each algorithm, write figures+metrics."""
      from src.core.reduce import preprocess as _pre  # avoid shadowing param name
      out = ensure_dir(out_dir)
      fig_dir = ensure_dir(out / "figures")
      emb = extractor.extract(assets)
      save_embeddings(emb, out)
      X = _pre(emb.vectors, list(preprocess), pca_components=pca_components)
      coords = umap_2d(X, umap_cfg["n_neighbors"], umap_cfg["min_dist"],
                       umap_cfg["metric"], seed)
      results: dict[str, dict] = {}
      for algo in algorithms:
          res = cluster(X, algo, k_min, k_max, seed)
          m = M.internal_metrics(X, res.labels)
          results[algo] = {"n_clusters": res.n_clusters, **m}
          scatter_2d(coords, res.labels, fig_dir / f"{extractor.name}_{algo}_umap.png",
                     title=f"{extractor.name} · {algo} (k={res.n_clusters})")
      metric_table_png({a: {k: v for k, v in r.items() if k != "n_clusters"}
                        for a, r in results.items()},
                       fig_dir / f"{extractor.name}_metrics.png",
                       title=f"{extractor.name} clustering metrics")
      return results
  ```

- [ ] **Step 4: Run to verify it passes.** `pytest tests/part_a/test_pipeline.py -v` → PASS.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/part_a/pipeline.py tests/part_a/test_pipeline.py
  git commit -m "feat: Part A pipeline + extractor registry"
  ```

---

## Phase 3 — Part B (faces)

### Task 3.1: Face generation (download + dedup + retry)

**Files:**
- Create: `src/part_b/generate.py`
- Test: `tests/part_b/test_generate.py`

- [ ] **Step 1: Write the test** (mock the HTTP fetcher — no real network in tests, D14).
  ```python
  # tests/part_b/test_generate.py  (create tests/part_b/__init__.py)
  from src.part_b.generate import generate_faces

  def test_generate_dedups_and_stops_at_n(tmp_path):
      # Fetcher returns: A, A(dup), B, C  -> should keep 3 unique then stop at n=3
      blobs = [b"AAAA", b"AAAA", b"BBBB", b"CCCC"]
      calls = {"i": 0}
      def fake_fetch(url):
          b = blobs[min(calls["i"], len(blobs) - 1)]; calls["i"] += 1
          return b
      saved = generate_faces(n=3, url="http://x", out_dir=tmp_path, delay_s=0.0,
                             max_retries=2, fetch=fake_fetch)
      assert len(saved) == 3
      assert len(set(p.read_bytes() for p in saved)) == 3   # all unique

  def test_generate_retries_on_error(tmp_path):
      seq = [RuntimeError("net"), b"AAAA"]
      def flaky_fetch(url):
          x = seq.pop(0)
          if isinstance(x, Exception):
              raise x
          return x
      saved = generate_faces(n=1, url="http://x", out_dir=tmp_path, delay_s=0.0,
                             max_retries=3, fetch=flaky_fetch)
      assert len(saved) == 1
  ```

- [ ] **Step 2: Run to verify it fails.** `pytest tests/part_b/test_generate.py -v` → FAIL.

- [ ] **Step 3: Implement `src/part_b/generate.py`.**
  ```python
  """Generate a face dataset from thispersondoesnotexist.com.

  A plain HTTP GET returns one random JPEG. We loop to N, dedup by content hash (the site
  can repeat images), and retry transient errors with backoff (D12). `fetch` is injectable
  so tests run without network.
  """
  from __future__ import annotations

  import hashlib
  import logging
  import time
  from pathlib import Path
  from typing import Callable

  from src.utils.io import ensure_dir

  log = logging.getLogger(__name__)


  def _default_fetch(url: str) -> bytes:
      """Fetch raw bytes from url with a browser-like UA (TPDNE rejects default UA)."""
      import requests

      resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
      resp.raise_for_status()
      return resp.content


  def generate_faces(n: int, url: str, out_dir: str | Path, delay_s: float,
                     max_retries: int, fetch: Callable[[str], bytes] | None = None) -> list[Path]:
      """Download n unique face JPEGs to out_dir. Returns saved paths (face_0000.jpg ...)."""
      fetch = fetch or _default_fetch
      out = ensure_dir(out_dir)
      seen: set[str] = set()
      saved: list[Path] = []
      attempts = 0
      max_attempts = n * (max_retries + 3) + 10
      while len(saved) < n and attempts < max_attempts:
          attempts += 1
          try:
              data = fetch(url)
          except Exception as exc:  # noqa: BLE001 - transient network; retry w/ backoff
              log.warning("fetch failed (%s); retrying", exc)
              time.sleep(min(2.0, delay_s + 0.2))
              continue
          digest = hashlib.sha256(data).hexdigest()
          if digest in seen:
              log.debug("duplicate image; skipping")
              continue
          seen.add(digest)
          p = out / f"face_{len(saved):04d}.jpg"
          p.write_bytes(data)
          saved.append(p)
          if delay_s:
              time.sleep(delay_s)
      log.info("Generated %d/%d unique faces (%d attempts)", len(saved), n, attempts)
      return saved
  ```

- [ ] **Step 4: Run to verify it passes.** `pytest tests/part_b/test_generate.py -v` → PASS.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/part_b/generate.py tests/part_b/
  git commit -m "feat: TPDNE face generation with dedup + retry"
  ```

### Task 3.2: ArcFace extractor (embedding + attributes)

**Files:**
- Create: `src/part_b/extractors/arcface.py`
- Test: `tests/part_b/test_arcface_extractor.py`

- [ ] **Step 1: Write the contract test** (real run behind `@slow`).
  ```python
  # tests/part_b/test_arcface_extractor.py
  import pytest

  def test_arcface_implements_protocol():
      from src.part_b.extractors.arcface import ArcFaceExtractor
      assert hasattr(ArcFaceExtractor, "extract")
      assert hasattr(ArcFaceExtractor, "attributes")

  @pytest.mark.slow
  def test_arcface_extract_real(tmp_path):
      """Needs insightface + a real face image."""
      # Place a known face image at tmp_path/face_0000.jpg before running on the box.
      from src.core.types import Asset
      from src.part_b.extractors.arcface import ArcFaceExtractor
      ext = ArcFaceExtractor(model_name="buffalo_l", det_size=320)
      emb = ext.extract([Asset(id="f0", path=tmp_path / "face_0000.jpg")])
      assert emb.vectors.shape[1] == 512
      attrs = ext.attributes
      assert "f0" in attrs and "age" in attrs["f0"]
  ```

- [ ] **Step 2: Run to verify it fails.** `pytest tests/part_b/test_arcface_extractor.py::test_arcface_implements_protocol -v` → FAIL.

- [ ] **Step 3: Implement `src/part_b/extractors/arcface.py`.**
  ```python
  """Part B feature: InsightFace ArcFace 512-D embedding + age/gender/pose attributes (D6).

  We cluster the embedding; the attributes (collected during extraction) are exposed via
  `.attributes` and later used as pseudo-labels to interpret/validate clusters.
  Images with zero or >1 detected face are skipped and counted (D12).
  """
  from __future__ import annotations

  import logging
  from pathlib import Path
  from typing import Sequence

  import numpy as np

  from src.core.types import Asset, Embeddings

  log = logging.getLogger(__name__)


  class ArcFaceExtractor:
      """InsightFace FaceAnalysis wrapper producing aligned embeddings + attributes."""

      def __init__(self, model_name: str, det_size: int) -> None:
          self.name = "arcface"
          self.model_name = model_name
          self.det_size = det_size
          self._app = None
          self.attributes: dict[str, dict] = {}   # id -> {age, gender, pose_yaw, ...}
          self.skipped: dict[str, str] = {}        # id -> reason

      def _ensure_app(self) -> None:
          if self._app is None:
              from insightface.app import FaceAnalysis

              app = FaceAnalysis(name=self.model_name)
              app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))
              self._app = app

      def extract(self, items: Sequence[Asset]) -> Embeddings:
          self._ensure_app()
          import cv2

          vecs, ids = [], []
          for asset in items:
              img = cv2.imread(str(asset.path))
              if img is None:
                  self.skipped[asset.id] = "unreadable"; continue
              faces = self._app.get(img)
              if len(faces) != 1:
                  self.skipped[asset.id] = f"{len(faces)} faces"; continue
              f = faces[0]
              vecs.append(np.asarray(f.normed_embedding, dtype=float))
              ids.append(asset.id)
              self.attributes[asset.id] = {
                  "age": float(f.age),
                  "gender": "M" if int(f.sex == "M" or getattr(f, "gender", 1)) else "F",
                  "pose_yaw": float(f.pose[1]) if getattr(f, "pose", None) is not None else 0.0,
              }
          log.info("ArcFace: kept %d, skipped %d", len(ids), len(self.skipped))
          return Embeddings(np.vstack(vecs), ids, self.name)
  ```
  > Note: InsightFace's gender field is `f.sex` ('M'/'F') in recent versions or `f.gender` (1/0) in older ones; the expression above tolerates both — confirm on the box and simplify to whichever the installed version exposes.

- [ ] **Step 4: Run to verify it passes.** `pytest tests/part_b/test_arcface_extractor.py::test_arcface_implements_protocol -v` → PASS.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/part_b/extractors/arcface.py tests/part_b/test_arcface_extractor.py
  git commit -m "feat: InsightFace ArcFace extractor with attributes"
  ```

### Task 3.3: Part B pipeline (cluster + attribute characterization)

**Files:**
- Create: `src/part_b/pipeline.py`
- Test: `tests/part_b/test_pipeline.py`

- [ ] **Step 1: Write the test** (fake extractor exposing attributes; verifies characterization + external metrics).
  ```python
  # tests/part_b/test_pipeline.py
  import numpy as np
  from src.core.types import Asset, Embeddings
  from src.part_b.pipeline import characterize_clusters, run_clustering_stage

  def test_characterize_clusters_profiles_attributes():
      labels = np.array([0, 0, 1, 1])
      attrs = {"a": {"age": 25, "gender": "F"}, "b": {"age": 27, "gender": "F"},
               "c": {"age": 60, "gender": "M"}, "d": {"age": 64, "gender": "M"}}
      ids = ["a", "b", "c", "d"]
      profile = characterize_clusters(labels, ids, attrs)
      assert profile[0]["mean_age"] < profile[1]["mean_age"]
      assert profile[0]["top_gender"] == "F" and profile[1]["top_gender"] == "M"

  class FakeFaceExtractor:
      name = "arcface"
      def __init__(self):
          self.attributes = {f"a{i}": {"age": 20 + 40 * (i // 2), "gender": "F" if i < 2 else "M"}
                             for i in range(4)}
          self.skipped = {}
      def extract(self, items):
          v = np.array([[0, 0], [0.1, 0], [5, 5], [5.1, 5]], dtype=float)
          return Embeddings(v, [a.id for a in items], self.name)

  def test_run_clustering_stage_writes_outputs(tmp_path):
      assets = [Asset(id=f"a{i}", path=tmp_path / f"a{i}.jpg") for i in range(4)]
      res = run_clustering_stage(
          extractor=FakeFaceExtractor(), assets=assets, out_dir=tmp_path,
          algorithms=["kmeans"], k_min=2, k_max=3, preprocess=["l2norm"],
          pca_components=None, umap_cfg={"n_neighbors": 3, "min_dist": 0.1, "metric": "cosine"},
          seed=0)
      assert "kmeans" in res
      assert (tmp_path / "figures").exists()
  ```

- [ ] **Step 2: Run to verify it fails.** `pytest tests/part_b/test_pipeline.py -v` → FAIL.

- [ ] **Step 3: Implement `src/part_b/pipeline.py`.**
  ```python
  """Part B orchestration: generate -> embed -> cluster -> CHARACTERIZE -> evaluate -> visualize.

  The core deliverable is characterize_clusters: describe each cluster in human terms using
  the InsightFace attributes (mean age, dominant gender, mean pose). External metrics
  validate the embedding clusters against gender/age-bucket pseudo-labels.
  """
  from __future__ import annotations

  import logging
  from pathlib import Path
  from typing import Sequence

  import numpy as np

  from src.core import metrics as M
  from src.core.cluster import cluster
  from src.core.embedding_store import save_embeddings
  from src.core.reduce import preprocess as _pre
  from src.core.reduce import umap_2d
  from src.core.types import Asset
  from src.core.visualize import metric_table_png, scatter_2d
  from src.utils.io import ensure_dir

  log = logging.getLogger(__name__)


  def characterize_clusters(labels: np.ndarray, ids: Sequence[str],
                            attributes: dict[str, dict]) -> dict[int, dict]:
      """Per cluster: size, mean age, dominant gender, mean pose. Human-readable profile."""
      profile: dict[int, dict] = {}
      labels = np.asarray(labels)
      for c in sorted(set(labels.tolist())):
          members = [ids[i] for i in range(len(ids)) if labels[i] == c]
          ages = [attributes[m]["age"] for m in members if m in attributes]
          genders = [attributes[m]["gender"] for m in members if m in attributes]
          top_gender = max(set(genders), key=genders.count) if genders else "?"
          profile[int(c)] = {
              "size": len(members),
              "mean_age": float(np.mean(ages)) if ages else float("nan"),
              "top_gender": top_gender,
              "pct_top_gender": (genders.count(top_gender) / len(genders)) if genders else 0.0,
          }
      return profile


  def _age_bucket(age: float) -> str:
      return "young" if age < 35 else ("middle" if age < 55 else "old")


  def run_clustering_stage(extractor, assets: Sequence[Asset], out_dir: str | Path,
                           algorithms: Sequence[str], k_min: int, k_max: int,
                           preprocess: Sequence[str], pca_components, umap_cfg: dict,
                           seed: int) -> dict:
      """Embed, cluster per algorithm, characterize clusters, validate vs pseudo-labels, plot."""
      out = ensure_dir(out_dir)
      fig_dir = ensure_dir(out / "figures")
      emb = extractor.extract(assets)
      save_embeddings(emb, out)
      X = _pre(emb.vectors, list(preprocess), pca_components=pca_components)
      coords = umap_2d(X, umap_cfg["n_neighbors"], umap_cfg["min_dist"], umap_cfg["metric"], seed)

      attrs = getattr(extractor, "attributes", {})
      gender_truth = np.array([attrs.get(i, {}).get("gender", "?") for i in emb.ids])
      age_truth = np.array([_age_bucket(attrs.get(i, {}).get("age", 0.0)) for i in emb.ids])

      results: dict[str, dict] = {}
      for algo in algorithms:
          res = cluster(X, algo, k_min, k_max, seed)
          row = {"n_clusters": res.n_clusters, **M.internal_metrics(X, res.labels)}
          if attrs:
              row.update({f"gender_{k}": v for k, v in
                          M.external_metrics(res.labels, gender_truth).items()})
              row.update({f"age_{k}": v for k, v in
                          M.external_metrics(res.labels, age_truth).items()})
              results[f"{algo}__profile"] = characterize_clusters(res.labels, emb.ids, attrs)
          results[algo] = row
          scatter_2d(coords, res.labels, fig_dir / f"arcface_{algo}_umap.png",
                     title=f"arcface · {algo} (k={res.n_clusters})")
      metric_table_png({a: {k: v for k, v in r.items() if isinstance(v, float)}
                        for a, r in results.items() if not a.endswith("__profile")},
                       fig_dir / "arcface_metrics.png", title="Face clustering metrics")
      return results
  ```

- [ ] **Step 4: Run to verify it passes.** `pytest tests/part_b/test_pipeline.py -v` → PASS.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/part_b/pipeline.py tests/part_b/test_pipeline.py
  git commit -m "feat: Part B pipeline with cluster characterization"
  ```

---

## Phase 4 — CLI, integration, docs

### Task 4.1: CLI entry point

**Files:**
- Create: `main.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the test** (invoke the parser; assert it dispatches without doing heavy work).
  ```python
  # tests/test_cli.py
  from main import build_parser, parse_overrides

  def test_parser_has_part_subcommands():
      parser = build_parser()
      args = parser.parse_args(["part-b", "generate", "--n", "5"])
      assert args.part == "part-b" and args.stage == "generate" and args.n == 5

  def test_parse_overrides_dotted():
      ov = parse_overrides(["part_b.n_images=42", "seed=1"])
      assert ov == {"part_b.n_images": 42, "seed": 1}
  ```

- [ ] **Step 2: Run to verify it fails.** `pytest tests/test_cli.py -v` → FAIL.

- [ ] **Step 3: Implement `main.py`.**
  ```python
  """CLI entry point. Runs each part and each stage (assignment: clear entry point per part).

  Examples:
    python main.py part-a all
    python main.py part-a render
    python main.py part-a cluster
    python main.py part-b generate --n 500
    python main.py part-b all
  Global: --config, --log-level, --set k=v (dotted config override, repeatable).
  """
  from __future__ import annotations

  import argparse
  import ast
  import logging
  from pathlib import Path

  from src.config import load_config
  from src.logging_setup import configure_logging
  from src.utils.io import ensure_dir
  from src.utils.seeding import seed_everything

  log = logging.getLogger(__name__)


  def parse_overrides(pairs: list[str]) -> dict:
      """Parse ['a.b=1', 'c=x'] into {'a.b': 1, 'c': 'x'} (literal-eval values when possible)."""
      out: dict = {}
      for pair in pairs:
          key, _, raw = pair.partition("=")
          try:
              val = ast.literal_eval(raw)
          except (ValueError, SyntaxError):
              val = raw
          out[key.strip()] = val
      return out


  def build_parser() -> argparse.ArgumentParser:
      p = argparse.ArgumentParser(description="Embedding clustering (Part A + Part B)")
      p.add_argument("--config", default="config/default.yaml")
      p.add_argument("--log-level", default="INFO")
      p.add_argument("--set", dest="overrides", action="append", default=[],
                     help="dotted config override, e.g. --set part_b.n_images=200")
      p.add_argument("part", choices=["part-a", "part-b"])
      p.add_argument("stage", choices=["render", "generate", "extract", "cluster", "all"])
      p.add_argument("--n", type=int, default=None, help="Part B: number of faces")
      return p


  def main(argv: list[str] | None = None) -> None:
      args = build_parser().parse_args(argv)
      configure_logging(args.log_level)
      overrides = parse_overrides(args.overrides)
      if args.part == "part-b" and args.n is not None:
          overrides["part_b.n_images"] = args.n
      cfg = load_config(args.config, overrides)
      seed_everything(cfg.seed)

      if args.part == "part-a":
          from src.part_a import pipeline as A
          out = ensure_dir(Path(cfg.paths.outputs_dir) / "part_a")
          render_dir = ensure_dir(Path(cfg.paths.data_dir) / "part_a_renders")
          assets = A.discover_assets(cfg.paths.assets_dir)
          if args.stage in ("render", "all"):
              from src.part_a.mesh_io import load_glb, to_single_mesh
              from src.part_a.render import render_views
              for a in assets:
                  mesh = to_single_mesh(load_glb(a.path))
                  render_views(mesh, a.id, render_dir, cfg.part_a.render.size_px,
                               cfg.part_a.render.supersample,
                               [tuple(v) for v in cfg.part_a.render.views])
          if args.stage in ("extract", "cluster", "all"):
              umap_cfg = vars(cfg.reduce.umap)
              for ext in A.build_extractors(cfg, render_dir):
                  res = A.run_clustering_stage(
                      ext, assets, out, cfg.part_a.clustering.algorithms,
                      cfg.part_a.clustering.k_min, cfg.part_a.clustering.k_max,
                      cfg.reduce.preprocess, cfg.reduce.pca_components, umap_cfg, cfg.seed)
                  log.info("Part A %s: %s", ext.name, res)

      else:  # part-b
          from src.part_b import pipeline as B
          from src.part_b.generate import generate_faces
          data_dir = ensure_dir(Path(cfg.paths.data_dir) / "faces")
          out = ensure_dir(Path(cfg.paths.outputs_dir) / "part_b")
          if args.stage in ("generate", "all"):
              generate_faces(cfg.part_b.n_images, cfg.part_b.tpdne_url, data_dir,
                             cfg.part_b.request_delay_s, cfg.part_b.max_retries)
          if args.stage in ("extract", "cluster", "all"):
              from src.core.types import Asset
              from src.part_b.extractors.arcface import ArcFaceExtractor
              assets = [Asset(id=p.stem, path=p) for p in sorted(data_dir.glob("*.jpg"))]
              ext = ArcFaceExtractor(cfg.part_b.insightface.model_name,
                                     cfg.part_b.insightface.det_size)
              res = B.run_clustering_stage(
                  ext, assets, out, cfg.part_b.clustering.algorithms,
                  cfg.part_b.clustering.k_min, cfg.part_b.clustering.k_max,
                  cfg.reduce.preprocess, cfg.reduce.pca_components,
                  vars(cfg.reduce.umap), cfg.seed)
              log.info("Part B: %s", {k: v for k, v in res.items() if not k.endswith("__profile")})


  if __name__ == "__main__":
      main()
  ```
  > Note: `vars(cfg.reduce.umap)` works because UMAPCfg is a frozen dataclass — `vars()` returns its `__dict__`. If that raises under your Python, replace with `dataclasses.asdict(cfg.reduce.umap)`.

- [ ] **Step 4: Run to verify it passes.** `pytest tests/test_cli.py -v` → PASS.

- [ ] **Step 5: Commit.**
  ```bash
  git add main.py tests/test_cli.py
  git commit -m "feat: argparse CLI entry point"
  ```

### Task 4.2: pytest config + full local suite green

**Files:**
- Create: `pytest.ini`

- [ ] **Step 1: Write `pytest.ini`** (register the `slow` marker; default-exclude it).
  ```ini
  [pytest]
  addopts = -m "not slow"
  markers =
      slow: requires GPU/network/large model downloads (run with: pytest -m slow)
  testpaths = tests
  ```

- [ ] **Step 2: Run the whole fast suite.** `pytest -v`
  Expected: ALL tests pass, slow tests deselected. Fix any breakage before continuing.

- [ ] **Step 3: Commit.**
  ```bash
  git add pytest.ini && git commit -m "test: pytest config with slow marker; full suite green"
  ```

### Task 4.3: End-to-end run on elem-danit1

**Files:** none (produces artifacts under `outputs/`).

- [ ] **Step 1: Sync code to the box** (via `run-on-elem-danit1` skill) and install:
  `pip install -r requirements.txt` then `POINT_MAE_CKPT_URL=<url from Task 0.1> bash scripts/setup_encoders.sh`.

- [ ] **Step 2: Run Part A as a detached, resume-safe job:**
  `python main.py part-a all` → produces `outputs/part_a/{dinov2,point_mae}.npy`,
  `outputs/part_a/figures/*.png`. Then run the real encoder smoke tests: `pytest -m slow tests/part_a -v`.

- [ ] **Step 3: Run Part B detached:**
  `python main.py part-b all` → `outputs/part_b/arcface.npy` + figures. `pytest -m slow tests/part_b -v`.

- [ ] **Step 4: Build Part A cluster montages** (renders grouped by cluster) — add a small
  call in `main.py` Part A `all` branch using `core.visualize.cluster_montage` with the
  per-asset front-view render (`data/part_a_renders/<id>_v0.png`) and the kmeans labels;
  re-run `part-a all`. Verify `outputs/part_a/figures/*_montage.png` exists.

- [ ] **Step 5: Pull artifacts back** to the local machine for the README (figures only).
  Append a Progress-log entry to `PLAN.md` with the metric numbers observed.

- [ ] **Step 6: Commit** any small fixes discovered during the real run + the PLAN.md update.
  ```bash
  git add -A && git commit -m "chore: end-to-end run on elem-danit1; record findings"
  ```

### Task 4.4: README — setup, usage, approach, findings

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Fill all README sections** (remove every "TBD"):
  - **Setup:** `python -m venv .venv && pip install -r requirements.txt`; note Point-MAE bootstrap (`setup_encoders.sh` + `POINT_MAE_CKPT_URL`); note weights auto-download; note elem-danit1 as the default compute and the CPU/GPU onnxruntime choice.
  - **Usage:** the `main.py` subcommands (copy from Task 4.1 docstring) + `--set` overrides + where outputs land.
  - **Part A approach:** GLB structure observations; triangulated-mesh render (D1) with view angles; DINOv2 (2D) vs Point-MAE (3D) as the feature-type comparison; clustering + metrics; embed the comparison figures.
  - **Part B approach:** TPDNE generation (size, dedup, preprocessing); InsightFace choice + justification; embedding clustering; **cluster characterization** (the discovered groups, with attribute evidence); pseudo-label validation; iteration notes.
  - **Findings:** the actual metric tables + 2-3 sentence interpretation per part; challenges/observations; note n=14 caveat for Part A.

- [ ] **Step 2: Verify no placeholders remain.** `! grep -rn "TBD\|TODO\|_TBD_" README.md` → no matches.

- [ ] **Step 3: Commit.**
  ```bash
  git add README.md && git commit -m "docs: complete README (setup, usage, approach, findings)"
  ```

### Task 4.5: Final acceptance pass against DEFINITIONS.md

**Files:**
- Modify: `DEFINITIONS.md` (tick boxes), `PLAN.md` (final status)

- [ ] **Step 1: Walk every checkbox in `DEFINITIONS.md`** and verify it's satisfied; tick `[x]`.
  Any unmet box → open a follow-up task before declaring done.

- [ ] **Step 2: Confirm submission hygiene:** `assets/` is gitignored (not in repo);
  `outputs/figures` images exist and are referenced by README; `view_glbs.py` decision made
  (relocate to `exploration/` or leave untracked — record in PLAN.md).

- [ ] **Step 3: Run the full fast suite one last time.** `pytest -v` → all green.

- [ ] **Step 4: Final commit + push.**
  ```bash
  git add -A && git commit -m "chore: final acceptance pass; tick DEFINITIONS checklist"
  # push uses a fresh credential (the original PAT should be revoked) or SSH remote
  ```

---

## Self-Review

**Spec coverage:**
- §2 shared abstraction → Tasks 1.1–1.6 (core), reused by 2.5/3.3. ✓
- §3 layout + CLI → Tasks 0.2, 4.1. ✓
- §4 Part A (render D1, DINOv2 2D, Point-MAE 3D, 2D-vs-3D compare) → Tasks 2.1–2.5. ✓
- §5 Part B (generate, ArcFace, characterize, validate) → Tasks 3.1–3.3. ✓
- §6 methodology (preprocess/cluster/metrics/viz) → Tasks 1.3–1.6, wired in 2.5/3.3. ✓
- §7 compute/reproducibility → Tasks 0.1, 2.4 (setup_encoders.sh), 4.3. ✓
- §8 error handling/logging/testing → Task 0.3, per-item try/except in extractors, 4.2. ✓
- §9 documentation discipline → standing rule + Tasks 4.4, 4.5. ✓
- §10 deliverables mapping → Task 4.5. ✓
- §12 deferred optionals → intentionally NOT in tasks (correct). ✓

**Placeholder scan:** No "TBD/implement-later" in code steps. Two encoder internals
(Point-MAE `_load_model/_encode`, InsightFace gender field) are flagged with explicit
"confirm on the box" notes — these are genuine external-repo unknowns isolated to single
methods, resolved in Task 4.3, not hand-waving over plan logic.

**Type consistency:** `Embeddings(vectors, ids, name)`, `FeatureExtractor.extract`,
`ClusterResult(labels, n_clusters, algorithm, k_selected)`, `cluster(X, algorithm, k_min,
k_max, seed)`, `preprocess(X, steps, pca_components)`, `umap_2d(X, n_neighbors, min_dist,
metric, seed)`, `run_clustering_stage(...)` signatures match across all tasks and both
pipelines. Config field names match `config/default.yaml` ↔ dataclasses in Task 0.4.
