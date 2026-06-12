This is a senior task
submitted by Danit Gilad,
June 2026,
https://github.com/danitgilad/embedding_clustering


# Embedding Clustering

Two independent unsupervised-learning pipelines over different data modalities:

- **Part A** — cluster 3D glasses assets (`.glb`) by appearance, comparing **2D features
  derived from rendered images** (DINOv2, CLIP) against a **3D feature derived directly from
  the mesh** (Point-MAE).
- **Part B** — generate a dataset of AI faces, embed them with a pretrained face model
  (InsightFace / ArcFace), then **discover and characterize** the natural attribute groupings
  (gender, age, …). We also run **DINOv2** on the faces as a baseline.

Both parts are built on one pipeline — `extract → reduce → cluster → evaluate → visualize`.
Within each part we compare several encoders (Part A: DINOv2 / CLIP / Point-MAE; Part B:
ArcFace / DINOv2). Because every encoder feeds the **identical** downstream — same
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
│   │   ├── extractors/dinov2.py          # DINOv2 on the face crops (Part B baseline)
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

# Part B (faces): generate -> extract (ArcFace + DINOv2) -> cluster -> characterize -> visualize
python main.py part-b generate --n 500
python main.py part-b all

# global flags
python main.py --log-level DEBUG --set part_b.n_images=200 part-b all
```

Each run writes to `outputs/<part>/` (cached `*.npy`, `*_results.json`, figures); **`outputs/` and
`data/` are git-ignored** and regenerated on demand. The committed result figures live in **`reports/`**.

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
**[`reports/part_b/viewer.html`](reports/part_b/viewer.html)** — a button per encoder plus a
switchable feature-distance histogram beside the scatter.

- **Part A** — each point is a glasses asset as a **rendered thumbnail on a cluster-coloured card**
  (hover for the id); side histogram = intra- vs inter-cluster distances.
- **Part B** — UMAP coloured by cluster; **hover any point to see the face** + predicted
  age/gender/pose; two side histograms = same- vs different- gender / age distances.

(Static PNGs — overviews, per-cluster montages, and Part B's algorithm-comparison figure — are also
under `reports/`.)

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
   *triangulated mesh surface* — **greyscale** (shape + shading, no colour/texture) — embed
   **each view independently** with a frozen vision encoder, then **mean-pool** (average) the 4
   per-view vectors into one vector per asset — an order-invariant, fixed-size aggregation (*not*
   concatenation). We use **DINOv2** (`facebook/dinov2-base`, CLS token → 768-d) as the primary 2D feature and
   **CLIP** (`openai/clip-vit-base-patch32`, image features → 512-d) as a second 2D encoder, to
   test whether *any* 2D render feature wins or DINOv2 specifically. (Colour is not used — see
   the note under Findings.)
3. **3D feature (Point-MAE).** Sample 1024 points from the mesh surface, **centre them and scale
   to the unit sphere** (so the descriptor is translation/scale-invariant), then encode with the
   pretrained Point-MAE encoder (pure-torch, CPU) into a 768-d vector (max ++ mean pool over group
   tokens). No rendering involved.
4. **Cluster** each embedding identically: standardize → L2-normalize → **KMeans**, with *k*
   chosen by best cosine silhouette over k∈[2,8]. (Part A uses KMeans only; the cross-encoder
   comparison lives in `part_a_overview.png`.)

**Findings.** All three encoders produce coherent clusters; the **2D render-based DINOv2
feature separates the glasses most cleanly**:

| Feature (KMeans)        | k | silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|-------------------------|---|--------------|------------------|---------------------|
| **DINOv2 (2D render)**  | 7 | **0.479**    | **0.731**        | **4.369**           |
| Point-MAE (3D mesh)     | 7 | 0.407        | 1.008            | 3.002               |
| CLIP (2D render)        | 3 | 0.358        | 1.382            | 3.560               |

**Is the different k fair?** Yes — *k* is chosen per encoder by the **same** silhouette sweep over
k∈[2,8]. DINOv2/Point-MAE peak at k=7, CLIP at k=3; that difference is *itself a result* (CLIP's
coarser, language-aligned embedding can't support finer splits), not a confound. At a **fixed k=6**
the ranking is unchanged — **DINOv2 0.471 > Point-MAE 0.404 > CLIP 0.302** (see the comparison table
in `part_a_overview.png`).

**Interpretation:** this is really **shape-from-2D-render vs shape-from-3D-points** — colour/texture
isn't a factor (see the note). DINOv2's fine-grained 2D features resolve the form best; **CLIP — also
2D — separates *worst*** (its language-aligned embedding is coarser and under-resolves these similar
products); Point-MAE's point geometry lands in between. **n = 14 → illustrative, not absolute.**

**Feature-distribution cross-check.** [`feature_distances_by_cluster.png`](reports/part_a/feature_distances_by_cluster.png)
checks the ranking *straight from the embeddings*: per encoder, the C(14,2)=91 pairwise cosine
distances split into intra- vs inter-cluster, gap **Δmean = mean_inter − mean_intra**. The order
matches silhouette — **DINOv2 0.69 > Point-MAE 0.53 > CLIP 0.38** — and CLIP's intra/inter histograms
overlap heavily (DINOv2's barely touch), so an independent, *k*-agnostic view confirms the result.

> **Note — colour & texture are intentionally NOT used.** DINOv2/CLIP embed **greyscale** renders
> and Point-MAE uses **xyz points**, so all three cluster by **shape only**. The coloured glasses in
> the viewer/overview are for human inspection; the algorithms never see them. *(Trade-off: for true
> appearance similarity colour would matter — a natural extension.)*

**Figures** (`reports/part_a/`):
- **[`part_a_overview.png`](reports/part_a/part_a_overview.png)** *(main)* — all encoders side-by-side:
  each glasses render at its UMAP point, cluster-coloured, GLB-id-labelled, plus a cross-encoder
  metrics table (incl. the fixed-k=6 column) and a one-line takeaway.
- **[`part_a_k6_umap.png`](reports/part_a/part_a_k6_umap.png)** — every encoder clustered at the
  common **k=6** (visual companion to the fixed-k column).
- **[`feature_distances_by_cluster.png`](reports/part_a/feature_distances_by_cluster.png)** — the discriminability
  cross-check (pairwise-distance spread + intra/inter Δmean, with a "how to read" caption).
- **`*_clusters_montage.png`** — glasses grouped by cluster, with GLB ids, a member table, and the
  metrics in the header (Part A clusters with KMeans only, so no standalone metrics/scatter files).
- **[`viewer.html`](reports/part_a/viewer.html)** — interactive (hover for a large colour render + id).

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
   We picked **`buffalo_l`** specifically because it's a SOTA face-recognition stack that
   **bundles detection + recognition + attribute prediction** in one CPU-only (onnxruntime)
   package; whether a *face-specialized* model is even needed is then tested empirically against
   the generic-backbone DINOv2 ablation below.
   As an **ablation**, we also embed the same faces with **DINOv2** — the *same*
   `facebook/dinov2-base` from Part A. It's a **generic** model (not face-specific, unlike
   ArcFace), so it's a baseline for "does a face-specialized model cluster faces better than a
   general backbone?". The **only difference from Part A's DINOv2 is the input**: face crops here
   vs glasses renders there. It has no attribute outputs, so it's clustered by silhouette only.
3. **Cluster** the embeddings identically to Part A (standardize → L2-norm → KMeans /
   Agglomerative / HDBSCAN). For ArcFace, *k* can be chosen by silhouette or by **attribute
   alignment** (see "Choosing k"); clusters are validated against the model's age/gender
   predictions as pseudo-labels (NMI / ARI / purity / AMI).
   > **Caveat — mild circularity:** for ArcFace, the same model supplies *both* the embedding and
   > the gender/age labels we validate it against, so that check is partly self-referential. The
   > DINOv2 ablation (a different embedding scored against ArcFace's *independent* labels)
   > is the cleaner test — and it still groups by gender, which corroborates the finding.

**Findings — the clusters organize by gender and age.** Characterizing the **ArcFace** clusters
(KMeans, k = 6) — *DINOv2 has no attribute outputs, so it isn't profiled per-cluster; its metrics
are in the summary table below* — yields:

| Cluster | size | mean age | dominant gender | purity |
|---------|------|----------|-----------------|--------|
| 0 | 116 | 44 | **F** | 100% |
| 1 |  87 | 48 | **M** | 100% |
| 2 |  95 | 39 | M | 65% |
| 3 |  91 | 37 | F | 84% |
| 4 |  53 | 19 | F | 58% (youngest) |
| 5 |  58 | 57 | M | 55% (oldest) |

Two clusters are perfectly gender-pure (0 = women, 1 = men), and the rest stratify by age
(cluster 4 ≈ teens/young adults, cluster 5 ≈ older adults).

**Bottom line across all Part B experiments** (the markdown of `part_b_summary.png`; best gender
purity in bold — note **DINOv2** wins it):

| experiment | k | silhouette ↑ | gender purity ↑ | gender NMI ↑ | age purity ↑ |
|---|---|---|---|---|---|
| arcface · KMeans (attribute-k) | 3 | 0.044 | 0.864 | 0.398 | 0.546 |
| arcface · KMeans (silhouette-k) | 6 | 0.045 | 0.808 | 0.270 | **0.602** |
| arcface · HDBSCAN | 0 | — | 0.564 | 0.000 | 0.418 |
| **dinov2 · KMeans** | 3 | **0.195** | **0.896** | **0.488** | 0.546 |

So every partition is highly gender-pure, **DINOv2 separates gender best** (and is more
geometrically separated), ArcFace edges age, and HDBSCAN finds no dense clusters (k=0).

Observations:
- **Gender is the dominant axis** of the ArcFace embedding (purity 0.81, NMI 0.27); age is a
  secondary, gradual axis.
- **Clusters also capture unlabelled attributes** — one k=6 cluster is visibly *people wearing
  glasses*, so the finer split surfaces real structure the gender+age pseudo-labels can't reward.
- **HDBSCAN finds no dense clusters** (k=0, all noise — see `arcface_algorithms.png`): the honest
  signature of a *continuous* embedding manifold, which is also why the silhouettes are low.
  (Density mechanics in [`METHODS.md`](METHODS.md).)
- **Face-specialized vs general backbone:** DINOv2 on the same faces clusters **more gender-aligned**
  than ArcFace (silhouette 0.195 vs 0.045; gender purity 0.896 vs 0.808) — a general backbone already
  groups by gender. ArcFace's edge is **age** (0.602 vs 0.546) and that it *returns* the age/gender/
  pose pseudo-labels we validate against; specialization buys the read-outs, not better gender
  grouping. (`part_b_overview_dinov2_k_3_silhouette.png`)

**Choosing k — silhouette vs attribute-driven.** The k=6 above is the *silhouette* pick. Since the
structure is continuous and we care about *attributes*, Part B also offers **attribute-driven**
k-selection (`clustering.k_selection: attribute`): pick the k maximizing **gender + age AMI**. On
KMeans this collapses to **k=3** (women / men / a young cohort) with **gender purity 0.864 > 0.808**
— a more gender-meaningful partition (now the default; silhouette is one flag away).

**Figures** (`reports/part_b/`):
- **[`part_b_summary.png`](reports/part_b/part_b_summary.png)**  — the bottom-line table
  comparing every experiment (encoder × k-selection + HDBSCAN) on the key metrics, conclusions beneath.
- **[`part_b_overview_arcface_k_3_attribute.png`](reports/part_b/part_b_overview_arcface_k_3_attribute.png)**
  — ArcFace UMAP (default attribute-k=3) recoloured by cluster / gender / age in two rows (points,
  then a dense face sample) + a metrics table. Variants name their k & selection:
  [`..._arcface_k_6_silhouette.png`](reports/part_b/part_b_overview_arcface_k_6_silhouette.png) and
  [`..._dinov2_k_3_silhouette.png`](reports/part_b/part_b_overview_dinov2_k_3_silhouette.png) (DINOv2 baseline).
- **[`feature_distances_by_attribute.png`](reports/part_b/feature_distances_by_attribute.png)** — pairwise distances
  split by same/different gender & age (DINOv2 separates gender more, Δ0.15, than ArcFace, Δ0.03).
- **[`arcface_algorithms.png`](reports/part_b/arcface_algorithms.png)** — a UMAP per algorithm
  (KMeans/Agglomerative/HDBSCAN, the last k=0 all-noise) + the metrics table.
- **[`arcface_clusters_montage.png`](reports/part_b/arcface_clusters_montage.png)** (sample faces per
  cluster); per-cluster profiles in [`arcface_results.json`](reports/part_b/arcface_results.json).

(Unlike Part A, the Part B encoders embed the **colour** face crop — colour *is* used.)

---

## Encoder comparison

The pipeline's pluggable `FeatureExtractor` design makes adding an encoder a one-file change,
and the viewers + metric tables pick it up automatically. Beyond the two primaries, additional
encoders were run as comparisons:

| Encoder | Part | Result |
|---|---|---|
| DINOv2 (on renders) | A 2D | best — silhouette 0.479 |
| Point-MAE (mesh) | A 3D | 0.407 |
| CLIP (on renders) | A 2D | 0.358 — coarser/semantic, weakest here |
| ArcFace | B | gender purity 0.81 (attribute-meaningful) |
| DINOv2 (on faces) | B | more separated *and* more gender-aligned (purity 0.896 > ArcFace 0.808) |

> The two DINOv2 rows are the **same model** (`facebook/dinov2-base`) on different inputs — glasses
> renders (Part A) vs face crops (Part B).

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

**Testing:** `pytest` runs a fast, hermetic suite over the pipeline logic (synthetic/mocked
inputs — no model weights, GPU, or network); `pytest -m slow` additionally runs the real
DINOv2 / Point-MAE / InsightFace. The pipeline (`main.py`) itself always uses the real models.
