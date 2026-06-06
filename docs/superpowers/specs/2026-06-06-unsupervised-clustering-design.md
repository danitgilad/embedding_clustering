# Design Spec — Unsupervised Clustering of 3D Assets & Face Images

**Date:** 2026-06-06
**Status:** Approved (brainstorm) — ready for implementation planning
**Source task:** `task.md` · **Acceptance checklist:** `DEFINITIONS.md` · **Live log:** `PLAN.md`

---

## 1. Goal

Deliver a single, well-structured Python project that solves two independent unsupervised
learning tasks and is graded as much on engineering quality as on analysis:

- **Part A** — cluster 14 `.glb` glasses assets, comparing a **2D feature derived from
  rendered images** against a **3D feature derived directly from the mesh**.
- **Part B** — generate ~500 synthetic faces, embed them with a pretrained model, and
  **discover & characterize** the natural attribute groupings (gender / age / pose / facial
  hair / glasses / expression …).

Non-goals: notebooks or flat scripts (explicitly forbidden); an installable package (not
required); identity recognition in Part B (faces are synthetic — only attribute structure
exists to find).

---

## 2. Architecture — shared pipeline, per-part front-ends

Both parts are the **same pipeline**; only the data loader and feature extractor differ:

```
 raw data ─▶ FeatureExtractor ─▶ embeddings (.npy + ids) ─▶ reduce ─▶ cluster ─▶ evaluate + visualize
            (per part)            (cached, keyed by name)   PCA/UMAP   K-means/…   metrics + plots
```

Everything downstream of embeddings is **shared, part-agnostic code** operating on a generic
`(embeddings: np.ndarray[N,D], ids: list[str])` contract. The key seam is a Protocol:

```python
class FeatureExtractor(Protocol):
    name: str                                                   # "dinov2", "point_mae", "arcface"
    def extract(self, items: Sequence[Asset]) -> Embeddings: ...  # → (N, D) + ids
```

Consequences:
- Part A's **2D-vs-3D feature comparison** becomes a first-class operation: run both extractors
  over the same assets, feed each through the *identical* downstream pipeline, compare the
  metric tables. Holding the downstream byte-for-byte identical is what makes the comparison
  fair. (The same seam also makes Part B's optional second-embedding experiment trivial.)
- Adding an encoder = adding one module implementing the Protocol.
- Embeddings are cached to disk (`.npy` + `ids.json`) keyed by extractor name. This `.npy`
  cache is BOTH the re-run boundary (never re-encode to re-cluster) AND the machine boundary
  (GPU encode on the box → analysis anywhere).

---

## 3. Project layout

```
senior_task/
├── main.py                     # CLI entry point (argparse subcommands: per-part + per-stage)
├── requirements.txt
├── README.md
├── config/
│   └── default.yaml            # ALL paths + params + encoder settings (single source)
├── src/
│   ├── __init__.py
│   ├── config.py               # YAML → typed frozen dataclasses (+ validation)
│   ├── logging_setup.py        # one logging config, used everywhere (no print())
│   ├── core/                   # shared, part-agnostic pipeline
│   │   ├── types.py            # Asset, Embeddings, FeatureExtractor Protocol
│   │   ├── embedding_store.py  # cache/load .npy + ids.json (keyed by name; id-alignment guard)
│   │   ├── reduce.py           # standardize / L2-norm / PCA / UMAP
│   │   ├── cluster.py          # KMeans, Agglomerative, HDBSCAN + silhouette-swept k
│   │   ├── metrics.py          # silhouette, Davies–Bouldin, Calinski–Harabasz, NMI/ARI/purity
│   │   └── visualize.py        # UMAP scatter, per-cluster montages, dist plots, metric tables
│   ├── part_a/                 # 3D glasses
│   │   ├── mesh_io.py          # trimesh load .glb → single mesh; surface point sampling
│   │   ├── render.py           # triangulated-mesh multi-view render (matplotlib, headless Agg)
│   │   ├── extractors/
│   │   │   ├── dinov2.py       # renders → DINOv2 embedding   (2D, PRIMARY)
│   │   │   ├── clip.py         # renders → CLIP               (2D, optional)
│   │   │   ├── pe_core.py      # renders → PE-Core            (2D, optional)
│   │   │   └── point_mae.py    # sampled points → Point-MAE   (3D, PRIMARY)
│   │   └── pipeline.py
│   ├── part_b/                 # faces
│   │   ├── generate.py         # rate-limited GET + hash-dedup from TPDNE
│   │   ├── extractors/
│   │   │   ├── arcface.py      # InsightFace → 512-D embedding + age/gender/pose (PRIMARY)
│   │   │   └── dinov2_generic.py  # generic backbone           (optional ablation)
│   │   └── pipeline.py
│   └── utils/                  # io, seeding, timing helpers
├── scripts/
│   ├── setup_encoders.sh       # bootstrap Point-MAE repo + checkpoints (idempotent, documented)
│   └── sync_to_box.sh          # rsync code → elem-danit1 (convenience)
├── outputs/                    # gitignored: embeddings/, figures/, run logs (per part)
├── data/                       # gitignored: generated faces, rendered images
├── tests/                      # pytest
├── docs/superpowers/specs/     # this spec
├── DEFINITIONS.md  PLAN.md  task.md
```

**CLI shape:**
```bash
python main.py part-a all                                   # render → extract → cluster → viz
python main.py part-a extract --encoders dinov2 point_mae
python main.py part-a cluster --algo kmeans agglomerative
python main.py part-b generate --n 500
python main.py part-b all
# global: --config config/default.yaml  --log-level INFO
```
Stages are independently runnable because each persists its output. CLI flags override config.

---

## 4. Part A — 3D glasses clustering

- **Load & explore** (`mesh_io`): trimesh loads each `.glb`; scenes/multi-mesh merged to one
  mesh; vertex colors extracted when present (gray default otherwise). Explore + document
  internal structure (mesh components, materials).
- **Render = triangulated mesh, NOT point cloud** (D1, proven better in the umap_viewer side
  project): matplotlib `Poly3DCollection`, two-sided shading, supersample + LANCZOS downscale,
  headless Agg. Multi-view (a few fixed angles) per asset.
- **2D feature (from renders):** pluggable encoders → **DINOv2 (primary)**, CLIP + PE-Core
  (`facebook/PE-Core-*`) optional. Multi-view embeddings pooled to one vector per asset.
- **3D feature (from mesh, no rendering):** **Point-MAE (primary)** — self-supervised,
  pure-geometry point encoder (mirrors DINOv2) over points sampled from the triangulated mesh
  surface. OpenShape/ULIP-2 optional secondary. Handcrafted descriptors (bbox ratios, solidity,
  PCA elongation, D2 histogram) = optional enrichment.
- **Comparison axis = 2D-render-derived vs 3D-mesh-derived** (NOT learned-vs-handcrafted).
  Apples-to-apples: learned embedding either side, modality the only variable.
- **Cluster:** KMeans (silhouette-swept k) + Agglomerative (dendrogram suits n=14). Internal
  metrics only (no labels). Compare the two feature types' cluster quality + qualitative
  montages.

> n=14 is small → metrics are illustrative/relative, not absolute. State this in the README.

---

## 5. Part B — face attribute clustering

- **Generate** (`generate`): ~500 faces (config/CLI param) via rate-limited HTTP GET to
  thispersondoesnotexist.com (plain JPEG endpoint — headless-safe), content-hash dedup,
  retry w/ backoff. InsightFace detect+align; keep exactly-one-clean-face images.
- **Model = InsightFace `buffalo_l` (ArcFace)** (D6): per face → 512-D embedding + detection +
  landmarks + **age/gender/pose**. Cluster the embedding; use age/gender/pose to interpret &
  validate clusters with evidence.
- **Core task = discover & characterize clusters** in human terms ("young women, frontal";
  "bearded men"; "wearing glasses"; "by head orientation"). Faces are synthetic → only
  attribute structure exists, so identity clustering is meaningless by design.
- **Cluster:** KMeans + Agglomerative + **HDBSCAN** (density/outliers, suits ~500). Internal
  metrics PLUS validation against InsightFace age/gender as pseudo-labels (NMI/ARI/purity).
- **Iterate** to sharpen clusters (preprocessing, k, algorithm) per the task's instruction.
- **Optional follow-up (deferred):** once the basic InsightFace pipeline runs, optionally add
  a second embedding (e.g. generic DINOv2) to see whether a face-specialized model yields
  cleaner attribute clusters. Reuses Part A machinery. Decided only after the core pipeline
  works — not part of the initial build.

---

## 6. Methodology (shared, D8)

- **Preprocess embeddings:** standardize / L2-normalize → optional PCA (denoise).
- **Cluster:** KMeans (silhouette-swept k) + Agglomerative for both; HDBSCAN added for Part B.
- **Evaluate (no ground truth):** cosine silhouette, Davies–Bouldin, Calinski–Harabasz.
  Part B additionally: NMI / ARI / purity vs InsightFace age/gender pseudo-labels.
- **Visualize:** 2D UMAP scatter colored by cluster (and by attribute for B); per-cluster
  montages (renders for A, face crops for B); feature-distribution plots. Centerpiece tables:
  for **Part A**, the 2D-vs-3D cross-feature metric comparison; for **Part B**, a per-cluster
  **attribute-profile table** (mean age, % gender, dominant pose) plus internal & pseudo-label
  metrics across the clustering algorithms.
- **Determinism:** global seeding (numpy, torch, sklearn random_state, UMAP seed).

---

## 7. Compute & infra (D11)

- **Run EVERYTHING on elem-danit1 (A100)** — render, all encoders, face download, ArcFace,
  reduce/cluster/metrics/viz. Local machine is fragile → control terminal only (edit, issue
  commands, pull back figures/HTML to view). Uses the `run-on-elem-danit1` skill: sync code →
  run stages as detached, resume-safe jobs (survive VPN/SSH drops & preemption) → pull results.
- **FIRST implementation step:** verify box outbound internet to the TPDNE endpoint.
  Fallback if firewalled: download faces locally (network-only, light) + sync JPEGs up.
- Project stays fully portable; the box is the default workflow, not a hard dependency.

### Reproducibility
- **Pinned `requirements.txt`** for all pip-installable deps: trimesh, numpy, scikit-learn,
  umap-learn, hdbscan, matplotlib, pillow, requests, pyyaml, insightface, onnxruntime, torch,
  transformers/timm (2D encoders), pytest.
- **Point-MAE isolated in `scripts/setup_encoders.sh`** (custom CUDA ops + checkpoint; NOT
  pip-installable). Documented explicitly in README — more reproducible than a lying
  requirements.txt.
- Pretrained weights (DINOv2/CLIP/PE-Core/InsightFace) auto-download from HF on first use.

---

## 8. Error handling, logging, testing

- **Error handling (D12):** resilient batches, never silent. Per-item try/except + reported
  failure summary; real misconfig fails loud; no problem-hiding fallbacks. GLB loader degrades
  gracefully (warns on missing colors/multi-mesh, skips corrupt). Face-gen: retry+backoff,
  hash-dedup, drop 0/>1-face (counted). Embedding store HARD-fails on (N,D)↔ids misalignment.
- **Logging (D13):** one `logging_setup.py`, level via `--log-level`, timestamped, no bare
  `print()`. Per-stage counts + timings; run log copied under `outputs/`.
- **Testing (D14, pytest):** deterministic part-agnostic logic — mesh_io (sample shape,
  single-mesh merge, missing-color path), embedding_store (round-trip + id-alignment guard),
  cluster (k-recovery on synthetic blobs), metrics (known inputs), generate (mocked requests).
  Encoders tested via a fake FeatureExtractor (Protocol contract); real-encoder runs are
  optional `@slow` smoke tests — no GPU/network in the default suite.

---

## 9. Documentation discipline (D10)

Continuous, not deferred: append PLAN.md Progress log after every stage/file; grow README
section-by-section as each part lands; docstrings + type hints on every public
function/class/module; tick DEFINITIONS.md boxes as criteria are met; log decisions in PLAN.md
as they happen. README always reflects current state.

---

## 10. Deliverables → acceptance mapping

| Deliverable (task.md) | Where satisfied |
|---|---|
| Structured Python project (not notebooks) | §3 layout, `src/` + `__init__.py` |
| Clear entry point / CLI | `main.py` argparse subcommands (§3) |
| Config separated, no scattered paths | `config/default.yaml` + `src/config.py` (§3) |
| `requirements.txt` reproducible | §7 reproducibility |
| README (setup, run, approach, findings) | §9 discipline; final README |
| Logging, no print() | §8 / D13 |
| Type hints + docstrings | §9 / D10 |
| Part A: 2D-from-render + 3D-from-mesh + compare | §4 |
| Part B: generate → embed → cluster → characterize | §5 |
| Output visualizations as image files | §6 visualize → `outputs/figures/` |
| `assets/` excluded from submission | gitignore / packaging step |

---

## 11. Open items to resolve at implementation start

1. Verify elem-danit1 outbound internet to TPDNE (else local-download fallback).
2. Confirm Point-MAE checkpoint source + exact repo for `setup_encoders.sh`.
3. Pick render view angles (front + a few) and supersample/size defaults for config.

## 12. Deferred until the basic pipeline runs (NOT in initial build)

- **Optional 2D encoders** for Part A: PE-Core, CLIP, OpenShape/ULIP-2 — added only after the
  **primaries (DINOv2 + Point-MAE)** produce a working end-to-end comparison.
- **Optional handcrafted geometric descriptors** for Part A (bbox ratios, solidity, PCA
  elongation, D2 histogram).
- **Optional second embedding** for Part B (generic DINOv2) — see §5.

Initial build targets the two primaries per part and a complete, working pipeline first;
everything above is incremental enrichment once that baseline is green.
