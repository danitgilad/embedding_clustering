# Encoder Comparison — Implementation Plan (Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the optional comparison encoders as pluggable extractors so they appear as extra toggles + metric-table rows in the existing viewers: CLIP & PE-Core (Part A 2D), DINOv2-generic (Part B), OpenShape/ULIP-2 (Part A 3D, best-effort). Re-encode on elem-danit1 and regenerate the viewers.

**Architecture:** Each encoder is one extractor module implementing the existing `FeatureExtractor` protocol, registered in the part's `build_extractors`, listed in config. The viewer stage already globs `*.npy`, so new encoders surface automatically — no viewer changes. All edits are additive. Heavy encoding runs on the box; embeddings cache to `.npy`.

**Tech Stack:** transformers (CLIP, DINOv2), Perception Encoder (`perception_models`, from source — discovery), OpenShape/PointBERT (from source, best-effort CPU port).

**Spec:** `docs/superpowers/specs/2026-06-06-interactive-viewers-design.md` §5.
**Recon (2026-06-07):** CLIP available via installed `transformers`. `open_clip`, `perception_models`, `openshape` are NOT on PyPI (from-source only); PE-Core/OpenShape lean CUDA → CPU effort/risk.
**Standing rules:** docstrings + type hints; no bare `print()`; additive edits; `./.venv/bin/pytest` fast suite; commit per task; update `PLAN.md` Progress log.

**Ordering rationale:** land the guaranteed, high-value encoders first (CLIP, DINOv2-generic), then attempt the risky ones (PE-Core, OpenShape) with documented fallbacks, then regenerate viewers.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/part_a/extractors/clip.py` | **new** — CLIP image embedding over renders (2D) |
| `src/part_b/extractors/dinov2_generic.py` | **new** — DINOv2 on face images (Part B comparison) |
| `src/part_b/pipeline.py` or `main.py` | additive — Part B loops a list of encoders |
| `src/config.py`, `config/default.yaml` | additive — `part_a.clip`, `part_b.encoders`, (later) pe_core/openshape |
| `src/part_a/extractors/pe_core.py` | **new** — Perception Encoder (discovery + fallback) |
| `src/part_a/extractors/openshape.py` | **new (best-effort)** — OpenShape point encoder, CPU |
| tests | contract tests per extractor (real runs `@slow`) |

---

## Task P2.1: CLIP extractor (Part A 2D) (TDD)

**Files:** Create `src/part_a/extractors/clip.py`; Modify `src/config.py` + `config/default.yaml` + `src/part_a/pipeline.py` (additive); Test `tests/part_a/test_clip_extractor.py`.

- [ ] **Step 1: Write the contract test**
  ```python
  # tests/part_a/test_clip_extractor.py
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
  ```

- [ ] **Step 2: Run to verify it fails** — `./.venv/bin/pytest tests/part_a/test_clip_extractor.py::test_clip_implements_protocol -v` → FAIL.

- [ ] **Step 3: Create `src/part_a/extractors/clip.py`** (mirrors DINOv2Extractor; lazy torch/transformers)
  ```python
  """2D visual feature: render-based CLIP image embedding (Part A optional comparison).

  Mirrors the DINOv2 extractor: load each asset's multi-view renders, embed each with a
  frozen CLIP image encoder, mean-pool the views into one vector per asset.
  """
  from __future__ import annotations

  import logging
  from pathlib import Path
  from typing import Sequence

  import numpy as np

  from src.core.types import Asset, Embeddings
  from src.utils.io import sanitize_id

  log = logging.getLogger(__name__)


  class CLIPExtractor:
      """Frozen CLIP image embedding over multi-view renders, mean-pooled per asset."""

      def __init__(self, hf_model: str, render_dir: str | Path) -> None:
          self.name = "clip"
          self.hf_model = hf_model
          self.render_dir = Path(render_dir)
          self._model = None
          self._processor = None
          self._device = "cpu"

      def _ensure_model(self) -> None:
          if self._model is None:
              import torch
              from transformers import CLIPModel, CLIPProcessor

              self._processor = CLIPProcessor.from_pretrained(self.hf_model)
              self._model = CLIPModel.from_pretrained(self.hf_model).eval()
              self._device = "cuda" if torch.cuda.is_available() else "cpu"
              self._model.to(self._device)

      def _embed_image(self, path: Path) -> np.ndarray:
          import torch
          from PIL import Image

          img = Image.open(path).convert("RGB")
          inputs = self._processor(images=img, return_tensors="pt").to(self._device)
          with torch.no_grad():
              feat = self._model.get_image_features(**inputs)
          return feat.squeeze(0).cpu().numpy()

      def extract(self, items: Sequence[Asset]) -> Embeddings:
          """Embed each asset's renders and mean-pool views into one vector."""
          self._ensure_model()
          vecs, ids = [], []
          for asset in items:
              views = sorted(self.render_dir.glob(f"{sanitize_id(asset.id)}_v*.png"))
              if not views:
                  log.warning("no renders for %s; skipping", asset.id)
                  continue
              vecs.append(np.stack([self._embed_image(p) for p in views]).mean(axis=0))
              ids.append(asset.id)
          if not vecs:
              raise ValueError("CLIP produced no embeddings (no renders found for any asset)")
          return Embeddings(np.vstack(vecs), ids, self.name)
  ```

- [ ] **Step 4: Run to verify it passes** — `./.venv/bin/pytest tests/part_a/test_clip_extractor.py::test_clip_implements_protocol -v` → PASS.

- [ ] **Step 5: Register CLIP (additive):**
  - `src/config.py` — add a `ClipCfg` dataclass `hf_model: str = "openai/clip-vit-base-patch32"` and a field `clip: ClipCfg = field(default_factory=ClipCfg)` on `PartACfg`.
  - `config/default.yaml` — under `part_a:` add `clip:` `\n    hf_model: openai/clip-vit-base-patch32`, and change `encoders_2d: [dinov2]` → `encoders_2d: [dinov2, clip]`.
  - `src/part_a/pipeline.py` `build_extractors` — in the `encoders_2d` loop add:
    ```python
            elif name == "clip":
                from src.part_a.extractors.clip import CLIPExtractor
                exts.append(CLIPExtractor(cfg.part_a.clip.hf_model, render_dir))
    ```
    (insert as an `elif` before the `else: raise`.)

- [ ] **Step 6: Run** `./.venv/bin/pytest -q` → all pass. **Commit:**
  ```bash
  git add src/part_a/extractors/clip.py tests/part_a/test_clip_extractor.py src/config.py config/default.yaml src/part_a/pipeline.py
  git commit -m "feat: CLIP render-based 2D extractor (Part A comparison)"
  ```

---

## Task P2.2: DINOv2-generic extractor + Part B encoder loop (Part B) (TDD)

**Files:** Create `src/part_b/extractors/dinov2_generic.py`; add `build_extractors` to `src/part_b/pipeline.py`; Modify `main.py` `_run_part_b` (additive loop), `src/config.py` + `config/default.yaml`; Test `tests/part_b/test_dinov2_generic.py`.

- [ ] **Step 1: Write the contract test**
  ```python
  # tests/part_b/test_dinov2_generic.py
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
  ```

- [ ] **Step 2: Run to verify it fails** — `./.venv/bin/pytest tests/part_b/test_dinov2_generic.py::test_dinov2_generic_implements_protocol -v` → FAIL.

- [ ] **Step 3: Create `src/part_b/extractors/dinov2_generic.py`** (embeds the face image directly; no attributes)
  ```python
  """Part B comparison feature: generic DINOv2 embedding of the face image (no face model).

  Answers 'does a face-specialized model (ArcFace) cluster faces better than a general
  backbone?'. Embeds each face JPG directly (CLS token); exposes no attributes.
  """
  from __future__ import annotations

  import logging
  from typing import Sequence

  import numpy as np

  from src.core.types import Asset, Embeddings

  log = logging.getLogger(__name__)


  class DINOv2GenericExtractor:
      """Frozen DINOv2 image embedding over single face images."""

      def __init__(self, hf_model: str) -> None:
          self.name = "dinov2_generic"
          self.hf_model = hf_model
          self._model = None
          self._processor = None
          self._device = "cpu"

      def _ensure_model(self) -> None:
          if self._model is None:
              import torch
              from transformers import AutoImageProcessor, AutoModel

              self._processor = AutoImageProcessor.from_pretrained(self.hf_model)
              self._model = AutoModel.from_pretrained(self.hf_model).eval()
              self._device = "cuda" if torch.cuda.is_available() else "cpu"
              self._model.to(self._device)

      def extract(self, items: Sequence[Asset]) -> Embeddings:
          """Embed each face image (CLS token) into one vector."""
          self._ensure_model()
          import torch
          from PIL import Image

          vecs, ids = [], []
          for asset in items:
              try:
                  img = Image.open(asset.path).convert("RGB")
              except Exception:  # noqa: BLE001 - skip unreadable images, keep the batch
                  log.warning("unreadable image %s; skipping", asset.id)
                  continue
              inputs = self._processor(images=img, return_tensors="pt").to(self._device)
              with torch.no_grad():
                  out = self._model(**inputs)
              vecs.append(out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy())
              ids.append(asset.id)
          if not vecs:
              raise ValueError("DINOv2-generic produced no embeddings (no readable images)")
          return Embeddings(np.vstack(vecs), ids, self.name)
  ```

- [ ] **Step 4: Run to verify it passes** — `./.venv/bin/pytest tests/part_b/test_dinov2_generic.py::test_dinov2_generic_implements_protocol -v` → PASS.

- [ ] **Step 5: Part B encoder registry (additive).**
  - `src/config.py` — add to `PartBCfg`: `encoders: tuple[str, ...] = ("arcface",)` and a `DinoCfg`-style `dinov2_generic` cfg reusing the existing `DinoCfg` (`dinov2_generic: DinoCfg = field(default_factory=DinoCfg)`).
  - `config/default.yaml` — under `part_b:` add `encoders: [arcface, dinov2_generic]` and `dinov2_generic:` `\n    hf_model: facebook/dinov2-base`.
  - `src/part_b/pipeline.py` — add a `build_extractors(cfg)` function:
    ```python
    def build_extractors(cfg):
        """Instantiate Part B encoders in config order (arcface first = attribute source)."""
        exts = []
        for name in cfg.part_b.encoders:
            if name == "arcface":
                from src.part_b.extractors.arcface import ArcFaceExtractor
                exts.append(ArcFaceExtractor(cfg.part_b.insightface.model_name,
                                             cfg.part_b.insightface.det_size))
            elif name == "dinov2_generic":
                from src.part_b.extractors.dinov2_generic import DINOv2GenericExtractor
                exts.append(DINOv2GenericExtractor(cfg.part_b.dinov2_generic.hf_model))
            else:
                raise ValueError(f"unknown Part B encoder {name!r}")
        return exts
    ```
  - `main.py` `_run_part_b` — replace the single-extractor block (the `ext = ArcFaceExtractor(...)` + single `run_clustering_stage` call) with a loop:
    ```python
        if stage in ("extract", "cluster", "all"):
            import dataclasses
            from src.core.types import Asset
            assets = [Asset(id=p.stem, path=p) for p in sorted(data_dir.glob("*.jpg"))]
            for ext in B.build_extractors(cfg):
                res = B.run_clustering_stage(
                    ext, assets, out, cfg.part_b.clustering.algorithms,
                    cfg.part_b.clustering.k_min, cfg.part_b.clustering.k_max,
                    cfg.reduce.preprocess, cfg.reduce.pca_components,
                    dataclasses.asdict(cfg.reduce.umap), cfg.seed,
                    montage_images={a.id: a.path for a in assets})
                log.info("Part B %s: %s", ext.name,
                         {k: v for k, v in res.items() if not k.endswith("__profile")})
    ```

- [ ] **Step 6: Run** `./.venv/bin/pytest -q` → all pass. **Commit:**
  ```bash
  git add src/part_b/extractors/dinov2_generic.py tests/part_b/test_dinov2_generic.py src/part_b/pipeline.py main.py src/config.py config/default.yaml
  git commit -m "feat: DINOv2-generic Part B extractor + multi-encoder Part B loop"
  ```

---

## Task P2.3: PE-Core extractor (Part A 2D) — discovery + implement + documented fallback

**Files:** Create `src/part_a/extractors/pe_core.py`; Modify `scripts/setup_encoders.sh` (additive PE-Core install), config + pipeline registry; Test `tests/part_a/test_pe_core_extractor.py` (protocol test only; real `@slow`).

- [ ] **Step 1: DISCOVERY on the box** — find the load path. Run (box):
  ```bash
  ssh elem-danit1 'bash -lc "cd /mnt/workspace/projects/embedding_clustering && \
    ./.venv/bin/pip install -q git+https://github.com/facebookresearch/perception_models.git 2>&1 | tail -3"'
  ssh elem-danit1 'bash -lc "cd /mnt/workspace/projects/embedding_clustering && ./.venv/bin/python - <<PY
  try:
      from core.vision_encoder import pe
      m = pe.CLIP.from_config(\"PE-Core-B16-224\", pretrained=True)
      print(\"PE_OK\", type(m).__name__)
  except Exception as e:
      print(\"PE_FAIL\", type(e).__name__, str(e)[:200])
  PY"'
  ```
  - If `PE_OK`: note the exact image-preprocess + embed call (`pe.transforms.get_image_transform(m.image_size)`, `m.encode_image(tensor)`), and proceed to Step 2.
  - If `PE_FAIL` (CUDA/xformers/flash-attn deps unmet on CPU): record the failure in `PLAN.md`, write the extractor anyway behind a clear ImportError message, mark its real test `@slow`, and **document PE-Core as "attempted, not runnable CPU-only" in the README**. Skip Steps 2–5's real-run expectations (the extractor stays present but unused). This is the explicit fallback.

- [ ] **Step 2: Write the protocol test** `tests/part_a/test_pe_core_extractor.py`:
  ```python
  import pytest
  from src.core.types import Asset, FeatureExtractor

  def test_pe_core_implements_protocol(tmp_path):
      from src.part_a.extractors.pe_core import PECoreExtractor
      ext = PECoreExtractor(model_name="PE-Core-B16-224", render_dir=tmp_path)
      assert ext.name == "pe_core"
      assert callable(ext.extract)
      assert isinstance(ext, FeatureExtractor)
  ```

- [ ] **Step 3: Run to verify it fails** → FAIL.

- [ ] **Step 4: Create `src/part_a/extractors/pe_core.py`** (lazy import; structure mirrors CLIP; the repo-specific load/embed confined to `_ensure_model`/`_embed_image`, filled per Step-1 discovery):
  ```python
  """2D visual feature: render-based Perception Encoder (PE-Core) embedding (Part A optional).

  PE-Core (Meta) is loaded from the `perception_models` package (installed from source by
  scripts/setup_encoders.sh). Mirrors the CLIP/DINOv2 extractors: multi-view renders ->
  frozen PE image embedding, mean-pooled. The repo-specific load + embed calls live in
  _ensure_model/_embed_image (confirmed during box discovery).
  """
  from __future__ import annotations

  import logging
  from pathlib import Path
  from typing import Sequence

  import numpy as np

  from src.core.types import Asset, Embeddings
  from src.utils.io import sanitize_id

  log = logging.getLogger(__name__)


  class PECoreExtractor:
      """Frozen PE-Core image embedding over multi-view renders, mean-pooled per asset."""

      def __init__(self, model_name: str, render_dir: str | Path) -> None:
          self.name = "pe_core"
          self.model_name = model_name
          self.render_dir = Path(render_dir)
          self._model = None
          self._transform = None
          self._device = "cpu"

      def _ensure_model(self) -> None:
          if self._model is not None:
              return
          import torch
          from core.vision_encoder import pe  # type: ignore  # perception_models (from source)

          self._model = pe.CLIP.from_config(self.model_name, pretrained=True).eval()
          self._transform = pe.transforms.get_image_transform(self._model.image_size)
          self._device = "cuda" if torch.cuda.is_available() else "cpu"
          self._model.to(self._device)

      def _embed_image(self, path: Path) -> np.ndarray:
          import torch
          from PIL import Image

          img = self._transform(Image.open(path).convert("RGB")).unsqueeze(0).to(self._device)
          with torch.no_grad():
              feat = self._model.encode_image(img)
          return feat.squeeze(0).float().cpu().numpy()

      def extract(self, items: Sequence[Asset]) -> Embeddings:
          self._ensure_model()
          vecs, ids = [], []
          for asset in items:
              views = sorted(self.render_dir.glob(f"{sanitize_id(asset.id)}_v*.png"))
              if not views:
                  log.warning("no renders for %s; skipping", asset.id)
                  continue
              vecs.append(np.stack([self._embed_image(p) for p in views]).mean(axis=0))
              ids.append(asset.id)
          if not vecs:
              raise ValueError("PE-Core produced no embeddings (no renders found)")
          return Embeddings(np.vstack(vecs), ids, self.name)
  ```
  > If Step-1 discovery showed a different API, adjust ONLY `_ensure_model`/`_embed_image` to match; keep the rest.

- [ ] **Step 5: Run protocol test** → PASS. Register (additive, ONLY if Step 1 was PE_OK): add `pe_core` cfg + `encoders_2d += pe_core` + `build_extractors` `elif name == "pe_core"`. Add the `perception_models` install line to `scripts/setup_encoders.sh` (additive). If PE_FAIL, do NOT add to `encoders_2d` (keep it out of the default run); leave the module + protocol test in place.

- [ ] **Step 6: Run** `./.venv/bin/pytest -q` → all pass. **Commit:**
  ```bash
  git add src/part_a/extractors/pe_core.py tests/part_a/test_pe_core_extractor.py scripts/setup_encoders.sh src/config.py config/default.yaml src/part_a/pipeline.py PLAN.md
  git commit -m "feat: PE-Core extractor (Part A; gated on box availability)"
  ```

---

## Task P2.4: OpenShape/ULIP-2 (Part A 3D) — best-effort, documented fallback

**Files:** Create `src/part_a/extractors/openshape.py` (+ backbone if a CPU port is feasible); config/registry; README note. Test: protocol test only.

- [ ] **Step 1: DISCOVERY on the box** — assess CPU feasibility. Run (box):
  ```bash
  ssh elem-danit1 'bash -lc "cd /mnt/workspace/projects/embedding_clustering && \
    git ls-remote https://github.com/Colin97/OpenShape_code.git HEAD >/dev/null 2>&1 && echo REPO_OK; \
    ./.venv/bin/python -c \"import huggingface_hub as h; print([m for m in h.list_models(author=\\\"OpenShape\\\")][:3])\" 2>&1 | tail -2"'
  ```
  Inspect whether the point encoder (PointBERT) depends on CUDA-only ops (knn/pointnet2) like Point-MAE. **Decision gate:** if a clean CPU path exists (pure-torch grouping reusable from `_point_mae_backbone.farthest_point_sample`/`knn_group`), proceed to Steps 2–5. **If it requires a large CUDA-op reimplementation (likely), STOP: write a short README note** ("OpenShape/ULIP-2 attempted; its PointBERT encoder is CUDA-coupled and a full CPU port is out of scope — Point-MAE already covers the learned-3D feature") and record the decision in `PLAN.md`. Do NOT sink unbounded time.

- [ ] **Step 2 (only if feasible): protocol test** `tests/part_a/test_openshape_extractor.py`:
  ```python
  from src.core.types import FeatureExtractor

  def test_openshape_implements_protocol(tmp_path):
      from src.part_a.extractors.openshape import OpenShapeExtractor
      ext = OpenShapeExtractor(checkpoint="checkpoints/openshape.pth", n_points=1024, seed=0)
      assert ext.name == "openshape" and callable(ext.extract)
      assert isinstance(ext, FeatureExtractor)
  ```

- [ ] **Step 3 (only if feasible): implement** `src/part_a/extractors/openshape.py` reusing
  `src/part_a/extractors/_point_mae_backbone.py`'s pure-torch `farthest_point_sample`/`knn_group`
  for grouping, loading the OpenShape PointBERT weights into a self-contained encoder (same
  pattern as the Point-MAE CPU port). Register in config/`build_extractors` under `encoders_3d`.

- [ ] **Step 4 (only if feasible): commit**
  ```bash
  git add src/part_a/extractors/openshape.py tests/part_a/test_openshape_extractor.py src/config.py config/default.yaml src/part_a/pipeline.py
  git commit -m "feat: OpenShape 3D extractor (CPU port)"
  ```

- [ ] **Step 5 (fallback path): commit the documented skip**
  ```bash
  git add README.md PLAN.md
  git commit -m "docs: OpenShape/ULIP-2 deferred (CUDA-coupled; Point-MAE covers learned-3D)"
  ```

---

## Task P2.5: Re-encode on the box, regenerate viewers, pull, document

**Files:** none (artifacts → `reports/`); Modify `README.md` (additive — extend the encoder comparison sections), `PLAN.md`.

- [ ] **Step 1: Sync code to box** (rsync as before) and install new deps there:
  `./.venv/bin/pip install -q plotly==5.22.0` (already) — plus the PE-Core source install IF Task P2.3 Step 1 was PE_OK.

- [ ] **Step 2: Encode the new Part A encoders** (renders already cached):
  `./.venv/bin/python main.py part-a extract` — runs all configured encoders_2d/3d (dinov2, clip, point_mae [+ pe_core/openshape if enabled]); writes their `.npy` + metrics + montages.

- [ ] **Step 3: Encode the new Part B encoder** (faces already downloaded):
  `./.venv/bin/python main.py part-b extract` — runs arcface + dinov2_generic.

- [ ] **Step 4: Regenerate viewers:** `./.venv/bin/python main.py part-a viewer && ./.venv/bin/python main.py part-b viewer`. Each now has the extra encoder toggles + metric-table rows.

- [ ] **Step 5: Run the real `@slow` encoder tests on the box:** `./.venv/bin/pytest -m slow tests/part_a tests/part_b -v` — confirm CLIP / DINOv2-generic (and PE-Core/OpenShape if enabled) produce embeddings.

- [ ] **Step 6: Pull artifacts** into `reports/` (viewer.html for both parts; new `*_results.json`; refreshed metric/montage PNGs). Open the viewers locally; confirm the new toggles + the metrics table comparing all encoders.

- [ ] **Step 7: Extend README** (additive): add the new encoders to the Part A / Part B comparison tables with their real metrics + a sentence on what each adds; note any documented PE-Core/OpenShape fallback.

- [ ] **Step 8: Commit + push**
  ```bash
  git add reports/ README.md PLAN.md
  git commit -m "docs: encoder comparison results (CLIP, DINOv2-generic[, PE-Core]) + regenerated viewers"
  ```

---

## Self-Review

**Spec coverage (§5):** CLIP → P2.1; DINOv2-generic + Part B loop → P2.2; PE-Core (documented fallback) → P2.3; OpenShape best-effort (documented fallback) → P2.4; re-encode + regenerate viewers → P2.5. ✓

**Placeholder scan:** No TODO/TBD. PE-Core/OpenShape have explicit discovery steps with concrete commands and a defined fallback (not vague "implement later"). CLIP/DINOv2-generic are fully concrete. ✓

**Type consistency:** All extractors implement `extract(items: Sequence[Asset]) -> Embeddings` with a `name` attr (protocol). `build_extractors(cfg)` (Part B) mirrors Part A's. Registry `elif` branches match config names (`clip`, `dinov2_generic`, `pe_core`, `openshape`). Viewer stage unchanged (globs `*.npy`). Config additions (`part_a.clip`, `part_b.encoders`, `part_b.dinov2_generic`) referenced consistently in `build_extractors`. ✓

**Ordering:** guaranteed encoders (P2.1, P2.2) before risky ones (P2.3, P2.4); regenerate last (P2.5). ✓
