# Embedding Clustering

Two independent unsupervised-learning pipelines over different data modalities:

- **Part A** — cluster 3D glasses assets (`.glb`) by appearance, comparing a **2D feature
  derived from rendered images** (DINOv2) against a **3D feature derived directly from the
  mesh** (Point-MAE).
- **Part B** — generate a dataset of AI faces, embed them with a pretrained face model
  (InsightFace / ArcFace), then **discover and characterize** the natural attribute
  groupings (gender, age, …).

Both parts share one pipeline — `extract → reduce → cluster → evaluate → visualize` — so the
only thing that differs between feature types is the extractor. Everything downstream is held
identical, which is what makes the comparisons fair.

## Project structure

```
.
├── main.py                  # CLI entry point (argparse): runs each part and each stage
├── config/default.yaml      # all paths + tunables (single source of truth)
├── requirements.txt
├── scripts/setup_encoders.sh# downloads the Point-MAE pretrained checkpoint
├── src/
│   ├── config.py            # YAML -> typed frozen dataclasses (+ dotted CLI overrides)
│   ├── logging_setup.py     # one logging config; no bare print()
│   ├── core/                # shared, part-agnostic pipeline
│   │   ├── types.py         # Asset, Embeddings, FeatureExtractor protocol
│   │   ├── embedding_store.py  # cache embeddings (.npy + ids.json), alignment-guarded
│   │   ├── reduce.py        # standardize / L2-norm / PCA / UMAP
│   │   ├── cluster.py       # KMeans, Agglomerative, HDBSCAN + silhouette-swept k
│   │   ├── metrics.py       # silhouette, Davies-Bouldin, Calinski-Harabasz, NMI/ARI/purity
│   │   └── visualize.py     # UMAP scatter, montage, metric table (saved as PNGs)
│   ├── part_a/              # 3D glasses
│   │   ├── mesh_io.py       # load .glb -> single mesh; surface point sampling
│   │   ├── render.py        # triangulated-mesh multi-view renderer (matplotlib, headless)
│   │   ├── extractors/dinov2.py        # renders -> DINOv2 embedding (2D)
│   │   ├── extractors/point_mae.py     # sampled points -> Point-MAE embedding (3D)
│   │   ├── extractors/_point_mae_backbone.py  # CPU pure-torch Point-MAE encoder
│   │   └── pipeline.py
│   ├── part_b/              # faces
│   │   ├── generate.py      # rate-limited TPDNE download + hash-dedup
│   │   ├── extractors/arcface.py       # InsightFace -> 512-d embedding + age/gender/pose
│   │   └── pipeline.py
│   └── utils/               # seeding, io helpers
├── tests/                   # pytest (fast suite; real-model tests marked @slow)
└── reports/                 # committed result figures + results.json for each part
```

## Setup

Python 3.10+. A CPU-only environment is sufficient for everything here (the datasets are
small); no GPU is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt           # CPU torch is fine: pip install torch==2.3.1
bash scripts/setup_encoders.sh            # downloads the Point-MAE pretrain checkpoint (~348 MB)
```

Notes:
- **Pretrained weights download on first use**: DINOv2 (`facebook/dinov2-base`) via
  HuggingFace; InsightFace `buffalo_l` via the insightface model zoo. Only the Point-MAE
  checkpoint is fetched explicitly (`setup_encoders.sh`).
- **Point-MAE runs CPU-only here.** The upstream repo couples its grouping to CUDA ops
  (`knn_cuda`, pointnet2 FPS) and imports a CUDA chamfer extension at import time, so it
  cannot run CPU-only as published. `src/part_a/extractors/_point_mae_backbone.py` is a
  self-contained pure-torch reimplementation of *only the encoder forward path* (with a
  torch FPS + KNN grouping) that loads the official pretrained `module.MAE_encoder.*`
  weights. This removes any CUDA-extension build and makes setup fully reproducible.
- For a GPU box, install a CUDA torch build instead; the code uses CUDA automatically if
  `torch.cuda.is_available()`.

## Usage

```bash
# Part A (3D glasses): render -> extract (DINOv2 + Point-MAE) -> cluster -> visualize
python main.py part-a all
# individual stages:
python main.py part-a render
python main.py part-a extract
python main.py part-a cluster

# Part B (faces): generate -> extract (ArcFace) -> cluster -> characterize -> visualize
python main.py part-b generate --n 500
python main.py part-b all

# global flags
python main.py --log-level DEBUG --set part_b.n_images=200 part-b all
```

Outputs land under `outputs/<part>/`: cached embeddings (`<encoder>.npy` + `ids.json`),
figures (`figures/*.png`), and `<encoder>_results.json` (metrics + per-cluster profiles).
The committed copies used in this README live under `reports/`.

Configuration lives entirely in `config/default.yaml` and is loaded into typed dataclasses;
any value can be overridden from the CLI with `--set dotted.key=value`.

## Interactive viewers

The `viewer` stage builds a **self-contained interactive HTML** explorer per part (Plotly via
CDN, thumbnails embedded — just open the file in a browser, no server):

```bash
python main.py part-a viewer   # -> outputs/part_a/viewer.html
python main.py part-b viewer   # -> outputs/part_b/viewer.html
```

Committed copies: **[`reports/part_a/viewer.html`](reports/part_a/viewer.html)** and
**[`reports/part_b/viewer.html`](reports/part_b/viewer.html)**. Each has a **button per
encoder** (toggle the feature/model) and a **per-encoder clustering-quality table** (best cell
highlighted). It's decoupled from encoding — the stage reads the cached `*.npy`, recomputes
UMAP + clusters deterministically, and renders — so the HTML can be rebuilt/restyled instantly.

- **Part A** — each point is a glasses asset shown as its **rendered thumbnail on a
  cluster-coloured card** (umap_viewer style); hover shows the id.
- **Part B** — a UMAP scatter coloured by cluster; **hover any point to see that face** plus
  its predicted **age / gender / pose** and cluster (face thumbnails are 96 px JPEGs, keeping
  the 500-point file ~2 MB).

Static PNGs (UMAP scatters, metric tables, and **annotated per-cluster montages** — each row
labelled with the cluster's stats) are also written under `reports/` for quick at-a-glance review.

---

## Part A — 3D glasses clustering

**Goal.** Group 14 eyewear `.glb` assets by appearance, and compare how well a *2D*
(render-based) feature and a *3D* (mesh-based) feature each capture that similarity.

**Pipeline.**
1. **Load** each `.glb` with `trimesh` and flatten the scene to a single mesh **applying the
   scene-graph node transforms** (`Scene.dump(concatenate=True)` — see Challenges).
2. **2D feature (DINOv2).** Render each asset from 4 fixed viewpoints off the *triangulated
   mesh surface* (matplotlib `Poly3DCollection`, headless, supersampled + LANCZOS
   downscaled). Each view is embedded with a frozen `facebook/dinov2-base` ViT (CLS token)
   and the views are mean-pooled to one 768-d vector.
3. **3D feature (Point-MAE).** Sample 1024 points from the mesh surface and encode them with
   the pretrained Point-MAE encoder (pure-torch, CPU) into a 768-d vector
   (max ++ mean pool over group tokens). No rendering involved.
4. **Cluster** each embedding identically: standardize → L2-normalize → KMeans and
   Agglomerative, with *k* chosen by best cosine silhouette over k∈[2,8].

**Findings.** All feature types produce coherent clusters; the **2D render-based DINOv2
feature separates the glasses most cleanly**. Three encoders are compared (a second 2D
encoder, CLIP, was added — see "Encoder comparison"):

| Feature (KMeans)        | k | silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|-------------------------|---|--------------|------------------|---------------------|
| **DINOv2 (2D render)**  | 7 | **0.479**    | **0.731**        | **4.369**           |
| Point-MAE (3D mesh)     | 7 | 0.407        | 1.008            | 3.002               |
| CLIP (2D render)        | 3 | 0.358        | 1.382            | 3.560               |

(Agglomerative shows the same DINOv2 > Point-MAE ordering: 0.489 vs 0.407 silhouette.)

Interpretation: DINOv2 sees colour, material and lens/rim styling from the renders, which
dominates human perception of "similar-looking glasses"; Point-MAE sees pure geometry, so it
groups by frame shape/proportion but is blind to colour and finish. Notably **CLIP — also a
2D render feature — separates *worse* than both** (and collapses to just k=3): its
language-aligned embedding is coarser/more semantic ("a pair of glasses") and less sensitive
to the fine visual differences that distinguish these similar products. So "2D beats 3D" here
is really "DINOv2's fine-grained visual features beat both pure geometry and CLIP's coarse
semantics." **n = 14 is small, so these numbers are illustrative/relative, not absolute.**

**Figures** (`reports/part_a/`): `dinov2_kmeans_umap.png`, `point_mae_kmeans_umap.png`
(UMAP scatter coloured by cluster), `*_metrics.png` (metric tables), and
`*_clusters_montage.png` — thumbnail grids of the actual glasses grouped by cluster, which
make the appearance-based grouping directly inspectable.

---

## Part B — Face attribute clustering

**Goal.** Generate a face dataset, embed it with a pretrained model, and discover what the
natural clusters represent. Because every face is a distinct synthetic identity, there is no
identity signal to recover — the only structure to find is **attributes**.

**Pipeline.**
1. **Generate** 500 faces from [thispersondoesnotexist.com](https://thispersondoesnotexist.com)
   (plain HTTP GET → JPEG), with polite rate-limiting, content-hash dedup, and retry/backoff.
2. **Model — InsightFace `buffalo_l` (ArcFace).** Chosen because it is face-specialized and,
   per face, returns a 512-d ArcFace embedding **plus** predicted age/gender/pose. We cluster
   the embedding and use the attributes as *evidence* to characterize and validate clusters.
3. **Cluster** the embeddings identically to Part A (standardize → L2-norm → KMeans /
   Agglomerative / HDBSCAN), and validate against the model's age/gender predictions as
   pseudo-labels (NMI / ARI / purity).

**Findings — the clusters organize by gender and age.** KMeans (k = 6) yields:

| Cluster | size | mean age | dominant gender | purity |
|---------|------|----------|-----------------|--------|
| 0 | 116 | 44 | **F** | 100% |
| 1 |  87 | 48 | **M** | 100% |
| 2 |  95 | 39 | M | 65% |
| 3 |  91 | 37 | F | 84% |
| 4 |  53 | 19 | F | 58% (youngest) |
| 5 |  58 | 57 | M | 55% (oldest) |

Two clusters are perfectly gender-pure (0 = women, 1 = men), and the remaining clusters
stratify by age (cluster 4 ≈ teens/young adults, cluster 5 ≈ older adults). Quantitatively,
the clustering agrees strongly with predicted gender and moderately with age:

| Algorithm     | k | silhouette | gender purity | gender NMI | age purity |
|---------------|---|------------|---------------|------------|------------|
| **KMeans**    | 6 | 0.045      | **0.808**     | 0.270      | 0.602      |
| Agglomerative | 2 | 0.033      | 0.724         | 0.258      | 0.430      |
| HDBSCAN       | 0 | —          | —             | —          | —          |

Observations:
- **Gender is the dominant axis** of the ArcFace embedding space (purity 0.81, NMI 0.27);
  age is a secondary, gradual axis.
- **Clusters also capture attributes we never labelled**: inspecting the viewer, one of the
  k=6 clusters is visibly **people wearing glasses**. ArcFace encodes eyewear (and likely
  pose/expression) too — so the finer k=6 split surfaces *real* structure beyond gender/age,
  which our gender+age pseudo-labels can't reward. A nice reminder that the internal metrics
  only measure the *labelled* axes, not everything the embedding actually represents.
- **Low silhouette is expected and informative**: face embeddings lie on a *continuous*
  manifold (identity/attribute space), not in well-separated blobs — so partitional methods
  impose useful but soft boundaries. **HDBSCAN found no dense clusters at all** (everything
  labelled noise), which is the honest signature of a continuous space rather than a bug.
- **Face-specialized vs generic backbone** (ablation): embedding the same 500 faces with a
  *generic* DINOv2 gives more geometrically separated clusters (KMeans silhouette **0.195** vs
  ArcFace's 0.045), but those clusters key on coarse image factors (pose, lighting, framing)
  rather than identity attributes. ArcFace's lower-silhouette clusters are the ones that
  *mean* something — validated to align with gender (purity 0.81) and age. So higher silhouette
  ≠ more useful clustering: the face-specialized model earns its place by producing
  *attribute-meaningful* groups. Both are toggles in the Part B viewer.

**Choosing k — silhouette vs attribute-driven.** The detailed k=6 result above is the
*silhouette* selection. Because the structure is continuous and what we care about is
*attributes*, Part B also supports **attribute-driven k-selection** (`clustering.k_selection:
attribute`): sweep k and pick the k that maximizes **gender + age AMI** (adjusted mutual
information). On KMeans this collapses to **k=3** (women 97% F · men 86% M · a young cohort)
with **gender purity 0.864 > 0.808** and gender NMI 0.398 > 0.270 — a *more* gender-meaningful
partition than silhouette's k=6. (This is now the default; silhouette is one flag away.)
Agglomerative under AMI still climbs to k_max because its finer splits stay gender-coherent —
the same continuous-manifold signature HDBSCAN showed.

**Figures** (`reports/part_b/`): `arcface_kmeans_umap.png` (clusters), `arcface_metrics.png`,
`arcface_clusters_montage.png` (sample faces per cluster — the gender/age grouping is visible
at a glance), plus the agglomerative/hdbscan scatters. Per-cluster profiles are in
`reports/part_b/arcface_results.json`.

---

## Encoder comparison

The pipeline's pluggable `FeatureExtractor` design makes adding an encoder a one-file change,
and the viewers + metric tables pick it up automatically. Beyond the two primaries, additional
encoders were run as comparisons:

| Encoder | Part | Status | Result |
|---|---|---|---|
| DINOv2 (render) | A 2D | primary | best — silhouette 0.479 |
| Point-MAE (mesh) | A 3D | primary | 0.407 |
| **CLIP (render)** | A 2D | **added** | 0.358 — coarser/semantic, weakest here |
| ArcFace | B | primary | gender purity 0.81 (attribute-meaningful) |
| **DINOv2-generic** | B | **added** | higher silhouette (0.195) but not attribute-aligned |
| PE-Core (render) | A 2D | **deferred** | `perception_models` needs Python ≥3.11; box runs 3.10 |
| OpenShape/ULIP-2 | A 3D | **deferred** | CUDA-coupled PointBERT; Point-MAE already covers learned-3D |

The two deferred encoders are documented rather than stubbed — each is a one-line re-enable
(`encoders_2d`/`encoders_3d` in `config/default.yaml`) once its environment constraint is met.

---

## Key decisions & challenges

- **Comparison axis is 2D-vs-3D, not learned-vs-handcrafted.** Both Part A features are
  learned embeddings; holding the downstream pipeline identical isolates the *modality* as
  the only variable.
- **Render from the triangulated mesh, not a point cloud** — gives smooth, solid images that
  the 2D encoder reads far better than sparse points.
- **GLB scene-graph transforms.** Early on, flattening scenes with
  `trimesh.util.concatenate` silently dropped each component's node transform, displacing
  parts (e.g. temple arms) by nearly the whole object size. Fixed by
  `Scene.dump(concatenate=True)`, which bakes the transforms. (Regression-tested.)
- **Point-MAE on CPU.** Reimplemented the encoder in pure torch to avoid its CUDA-only
  ops/extensions, loading the official pretrained weights (verified: 0 missing/unexpected
  keys). This is *more* reproducible than depending on the CUDA build.
- **InsightFace detection size.** The frame-filling synthetic faces were *missed* by the
  detector at `det_size=640` (only 123/500 detected); a sweep showed `det_size=320`
  detects 40/40. Set to 320, and we keep the largest detected face for robustness → 500/500.

## Reproducibility & engineering

- **Determinism:** global seeding (numpy / torch / sklearn `random_state` / UMAP seed);
  Point-MAE FPS starts from a fixed point.
- **Config:** one YAML, typed dataclasses, no hardcoded paths; CLI `--set` overrides.
- **Logging:** central config, no bare `print()` for operational output.
- **Testing:** `pytest` covers the deterministic logic (mesh IO, embedding store + alignment
  guard, clustering k-recovery, metrics, face-generation dedup/retry with mocked network).
  Real-model runs are marked `@slow` and excluded from the default suite:
  ```bash
  pytest            # fast suite (no GPU/network)
  pytest -m slow    # real DINOv2 / Point-MAE / InsightFace (needs weights)
  ```

## Deliverables

- Structured Python project (not notebooks); `requirements.txt`; this README.
- Result visualizations saved as image files under `reports/`.
- The `assets/` folder is intentionally git-ignored (not part of the submission).
