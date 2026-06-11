# Embedding Clustering

Two independent unsupervised-learning pipelines over different data modalities:

- **Part A** — cluster 3D glasses assets (`.glb`) by appearance, comparing **2D features
  derived from rendered images** (DINOv2, CLIP) against a **3D feature derived directly from
  the mesh** (Point-MAE).
- **Part B** — generate a dataset of AI faces, embed them with a pretrained face model
  (InsightFace / ArcFace, plus a generic-DINOv2 ablation), then **discover and characterize**
  the natural attribute groupings (gender, age, …).

Both parts are built on one pipeline — `extract → reduce → cluster → evaluate → visualize`.
Within each part we compare several encoders (Part A: DINOv2 / CLIP / Point-MAE; Part B:
ArcFace / generic-DINOv2). Because every encoder feeds the **identical** downstream — same
preprocessing, UMAP, clustering algorithm + k-selection, and metrics — the *only* variable is
the extractor, so any difference in cluster quality is attributable to the embedding itself.
That controlled setup is what makes the **encoder-vs-encoder comparison within a part** fair.
(Part A and Part B are independent tasks on different data and are not compared to each other.)

This allows code reuse: the downstream is the shared `src/core/` package
(`reduce`/`cluster`/`metrics`/`visualize`), consumed by both parts through one
`FeatureExtractor` protocol. The per-part orchestrators `src/part_a/pipeline.py` and
`src/part_b/pipeline.py` are separate thin wrappers that call those same core modules — Part B's
just additionally does attribute characterization + pseudo-label validation that Part A doesn't need.

> 📖 **Method background** — what each embedding / clustering / metric method *is* and why we use
> it (DINOv2, CLIP, Point-MAE, ArcFace, KMeans, silhouette, AMI, …) lives in
> **[`METHODS.md`](METHODS.md)**. This README stays focused on approach, decisions, and findings.

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
│   │   ├── extractors/dinov2.py        # renders -> DINOv2 embedding (2D, primary)
│   │   ├── extractors/clip.py          # renders -> CLIP embedding (2D, comparison)
│   │   ├── extractors/point_mae.py     # sampled points -> Point-MAE embedding (3D)
│   │   ├── extractors/_point_mae_backbone.py  # CPU pure-torch Point-MAE encoder
│   │   ├── viewer.py                   # build the Part A interactive HTML
│   │   └── pipeline.py
│   ├── part_b/              # faces
│   │   ├── generate.py      # rate-limited TPDNE download + hash-dedup
│   │   ├── extractors/arcface.py       # InsightFace -> 512-d embedding + age/gender/pose
│   │   ├── extractors/dinov2_generic.py  # DINOv2 on faces (generic-backbone ablation)
│   │   ├── viewer.py                   # build the Part B interactive HTML
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
- **Pretrained weights download on first use**: DINOv2 (`facebook/dinov2-base`) and CLIP
  (`openai/clip-vit-base-patch32`) via HuggingFace; InsightFace `buffalo_l` via the insightface
  model zoo. Only the Point-MAE checkpoint is fetched explicitly (`setup_encoders.sh`).
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
# Part A (3D glasses): explore -> render -> extract (DINOv2 + Point-MAE) -> cluster -> visualize
python main.py part-a all
# individual stages:
python main.py part-a explore   # inspect GLB structure/materials -> dataset_exploration.md
python main.py part-a render
python main.py part-a extract
python main.py part-a cluster

# Part B (faces): generate -> extract (ArcFace + generic-DINOv2) -> cluster -> characterize -> visualize
python main.py part-b generate --n 500
python main.py part-b all

# global flags
python main.py --log-level DEBUG --set part_b.n_images=200 part-b all
```

Outputs land under `outputs/<part>/`: cached embeddings (`<encoder>.npy` + `ids.json`),
figures (`figures/*.png`), and `<encoder>_results.json` (metrics + per-cluster profiles).
**`outputs/` (and `data/`) are git-ignored** — they're generated when you run a stage, so a
fresh checkout has them empty. Our runs were executed on a remote CPU box, and the result
files this README references were copied into the committed **`reports/`** folder (the only
result artifacts in git).

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

**Goal.** Group 14 eyewear `.glb` assets by appearance, and compare how well **2D**
(render-based) features and a **3D** (mesh-based) feature each capture that similarity. We run
two 2D encoders (DINOv2, CLIP) and one 3D encoder (Point-MAE).

**Dataset exploration.** `part-a explore` inspects every GLB's internal structure (mesh
components, vertex/face counts, materials, texturing, bounding extent) and writes
[`reports/part_a/dataset_exploration.md`](reports/part_a/dataset_exploration.md). It shows the
assets are **multi-component Scenes** (4–7 mesh parts each → node transforms must be applied when
flattening), span **~18k–55k vertices** (we sample a fixed 1024 surface points so detail is
comparable), and are **14/14 textured** — materials carry **real colour** that the encoders
deliberately ignore (greyscale shape / xyz geometry only).

**Pipeline.**
1. **Load** each `.glb` with `trimesh` and flatten the scene to a single mesh **applying the
   scene-graph node transforms** (`Scene.dump(concatenate=True)` — see Challenges).
2. **2D features (DINOv2 + CLIP).** Render each asset from 4 fixed viewpoints off the
   *triangulated mesh surface* — **greyscale** (shape + shading, no colour/texture) — and embed
   each view with a frozen vision encoder, mean-pooling the views to one vector per asset. We
   use **DINOv2** (`facebook/dinov2-base`, CLS token → 768-d) as the primary 2D feature and
   **CLIP** (`openai/clip-vit-base-patch32`, image features → 512-d) as a second 2D encoder, to
   test whether *any* 2D render feature wins or DINOv2 specifically. (Colour is not used — see
   the note under Findings.)
3. **3D feature (Point-MAE).** Sample 1024 points from the mesh surface and encode them with
   the pretrained Point-MAE encoder (pure-torch, CPU) into a 768-d vector
   (max ++ mean pool over group tokens). No rendering involved.
4. **Cluster** each embedding identically: standardize → L2-normalize → KMeans and
   Agglomerative, with *k* chosen by best cosine silhouette over k∈[2,8].

**Findings.** All three encoders produce coherent clusters; the **2D render-based DINOv2
feature separates the glasses most cleanly**:

| Feature (KMeans)        | k | silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|-------------------------|---|--------------|------------------|---------------------|
| **DINOv2 (2D render)**  | 7 | **0.479**    | **0.731**        | **4.369**           |
| Point-MAE (3D mesh)     | 7 | 0.407        | 1.008            | 3.002               |
| CLIP (2D render)        | 3 | 0.358        | 1.382            | 3.560               |

(Agglomerative shows the same DINOv2 > Point-MAE ordering: 0.489 vs 0.407 silhouette.)

Interpretation: the 2D features (DINOv2, CLIP) embed **greyscale** renders, so they capture
the glasses' **form and silhouette as projected to 2D** (rim shape, lens curvature, frame
proportions revealed by shading) — *not* colour or material. Point-MAE captures the **raw 3D
geometry** of the surface points. So this is really a **shape-from-2D-render vs
shape-from-3D-points** comparison, and **colour/texture is not a factor for any encoder** (see
the note below). DINOv2's fine-grained 2D features resolve the form best; **CLIP — also a 2D
render feature — separates *worse*** (and collapses to k=3): its language-aligned embedding is
coarser/semantic ("a pair of glasses") and under-resolves these similar products. Point-MAE's
point geometry lands in between. **n = 14 is small, so these numbers are illustrative.**

**Feature-distribution cross-check.** To confirm the ranking isn't an artifact of one clustering,
[`feature_distributions.png`](reports/part_a/feature_distributions.png) measures discriminability *straight from the embeddings*: for each
encoder we take all pairwise **cosine distances** and split them into *intra-cluster* vs
*inter-cluster*; the gap between their means (**Δmean = mean_inter − mean_intra**) says how
cleanly same-cluster items sit closer than different-cluster items. The ordering is **identical to
silhouette — DINOv2 0.69 > Point-MAE 0.53 > CLIP 0.38** — and the histograms make CLIP's weakness
visible: its intra- and inter-cluster distances **overlap heavily** (little geometric structure to
cluster), whereas DINOv2's barely touch. An independent, *k*-agnostic view agreeing with silhouette
is good evidence the DINOv2 > Point-MAE > CLIP result is real. (The left-hand histograms also show a
small near-zero tail = the near-duplicate left/right frame variants.)

> **Note — colour and texture are intentionally NOT used.** The renders fed to DINOv2/CLIP are
> **greyscale** (form + shading only) and Point-MAE consumes **xyz surface points** — so all
> three encoders cluster by **shape/geometry, never colour or material**. The *coloured*
> glasses you see in the viewer and `part_a_overview.png` are rendered that way **for human
> inspection only**; the algorithms never see them. Trade-off: for true *appearance* similarity
> colour would matter (a black vs a red copy of the same frame look different but would embed
> almost identically here) — feeding texture-coloured renders into the 2D encoders would be a
> natural extension.

**Figures** (`reports/part_a/`): the main one is **[`part_a_overview.png`](reports/part_a/part_a_overview.png)** — a single panel
per encoder where each glasses **render is placed at its UMAP point**, framed in its
**cluster colour**, labelled with its **GLB id**, and titled with the encoder's metrics. It
makes "which glasses landed in which cluster, for each feature" inspectable at a glance (and
shows CLIP's coarse 3-cluster grouping next to DINOv2/Point-MAE's 7).
**[`feature_distributions.png`](reports/part_a/feature_distributions.png)** is the figure behind
the *Feature-distribution cross-check* above (pairwise-distance spread + the intra/inter Δmean per
encoder). Also written: `*_kmeans_umap.png` (plain scatters), `*_metrics.png` (metric tables),
`*_clusters_montage.png` (per-cluster thumbnail grids). The interactive
**[`viewer.html`](reports/part_a/viewer.html)** is the richest view (hover for a large colour render + id).

---

## Part B — Face attribute clustering

**Goal.** Generate a face dataset, embed it with a pretrained model, and discover what the
natural clusters represent. Because every face is a distinct synthetic identity, there is no
identity signal to recover — the only structure to find is **attributes**.

**Pipeline.**
1. **Generate** 500 faces from [thispersondoesnotexist.com](https://thispersondoesnotexist.com)
   (plain HTTP GET → JPEG, ~1024², one identity per request), with polite rate-limiting,
   content-hash dedup, and retry/backoff. **Preprocessing** is model-side: ArcFace detects,
   aligns and crops each face (largest face kept) before embedding; the DINOv2 ablation resizes
   + centre-crops to 224². No manual cleaning is needed (TPDNE faces are frontal and frame-filling).
2. **Model — InsightFace `buffalo_l` (ArcFace).** Chosen because it is face-specialized and,
   per face, returns a 512-d ArcFace embedding **plus** predicted age/gender/pose. We cluster
   the embedding and use the attributes as *evidence* to characterize and validate clusters.
   As an **ablation**, we also embed the same faces with a **generic DINOv2** backbone
   (`facebook/dinov2-base`) to test whether a face-specialized model clusters faces better
   than a general one (it has no attributes, so it's clustered by silhouette only).
3. **Cluster** the embeddings identically to Part A (standardize → L2-norm → KMeans /
   Agglomerative / HDBSCAN). For ArcFace, *k* can be chosen by silhouette or by **attribute
   alignment** (see "Choosing k"); clusters are validated against the model's age/gender
   predictions as pseudo-labels (NMI / ARI / purity / AMI).

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
  *generic* DINOv2 gives clusters that are **more separated *and* more gender-aligned** than
  ArcFace (KMeans silhouette **0.195** vs 0.045; gender purity **0.896** vs 0.808, NMI 0.488 vs
  0.270) — a general self-supervised backbone already groups faces by gender, no
  face-specialization required. ArcFace's edge is narrower: it tracks **age** slightly better
  (purity 0.602 vs 0.546) and, crucially, *returns* the per-face predicted age/gender/pose —
  the pseudo-labels we use to interpret and validate every clustering (DINOv2 gives none). So
  the specialization buys the attribute **read-outs**, not better gender grouping. See
  [`part_b_overview_dinov2_generic.png`](reports/part_b/part_b_overview_dinov2_generic.png); both
  encoders are toggles in the Part B viewer.

**Choosing k — silhouette vs attribute-driven.** The detailed k=6 result above is the
*silhouette* selection. Because the structure is continuous and what we care about is
*attributes*, Part B also supports **attribute-driven k-selection** (`clustering.k_selection:
attribute`): sweep k and pick the k that maximizes **gender + age AMI** (adjusted mutual
information). On KMeans this collapses to **k=3** (women 97% F · men 86% M · a young cohort)
with **gender purity 0.864 > 0.808** and gender NMI 0.398 > 0.270 — a *more* gender-meaningful
partition than silhouette's k=6. (This is now the default; silhouette is one flag away.)
Agglomerative under AMI still climbs to k_max because its finer splits stay gender-coherent —
the same continuous-manifold signature HDBSCAN showed.

**Figures** (`reports/part_b/`): the main one is **[`part_b_overview.png`](reports/part_b/part_b_overview.png)** — the ArcFace UMAP
shown three ways (by **cluster**, **gender**, **age**) in two rows: top = coloured points only
(unoccluded), bottom = the same layout with a dense face sample overlaid, so you can *see* the
clusters tracking gender + age. The same figure is generated per encoder and per k-selection:
**[`part_b_overview_arcface_silhouette.png`](reports/part_b/part_b_overview_arcface_silhouette.png)**
is the silhouette **k=6** partition (vs the default attribute **k=3**), and
**[`part_b_overview_dinov2_generic.png`](reports/part_b/part_b_overview_dinov2_generic.png)** is the
generic-backbone ablation. Also: [`arcface_clusters_montage.png`](reports/part_b/arcface_clusters_montage.png)
(sample faces per cluster), `*_umap.png` scatters, [`arcface_metrics.png`](reports/part_b/arcface_metrics.png).
Per-cluster profiles in [`arcface_results.json`](reports/part_b/arcface_results.json). (Unlike Part A, the encoders here embed the **colour** face crop —
colour *is* used.)

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
| **DINOv2-generic** | B | **added** | more separated *and* more gender-aligned (purity 0.896 > ArcFace 0.808) |
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
