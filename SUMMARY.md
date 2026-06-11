# Project Summary

Two unsupervised clustering pipelines over different modalities, sharing one backbone:
**extract embeddings → preprocess → cluster → evaluate → visualize**. Only the feature
extractor changes between tasks; everything downstream is held identical so comparisons are fair.

---

## Part A — cluster 14 3D glasses by appearance

**Question:** does a *2D* feature (from rendered images) or a *3D* feature (from the mesh)
better capture "similar-looking glasses"?

**Flow & choices**
- **Explore** each `.glb` first (`dataset_exploration.md`): they're **multi-component Scenes** (→ apply node transforms when flattening) with textured materials the encoders deliberately ignore.
- Render each `.glb` from 4 views off the **triangulated mesh**, **greyscale** (form + shading, no colour).
- **2D feature:** frozen **DINOv2** on the grey renders, views mean-pooled (+ **CLIP** as a 2nd 2D encoder, to test whether *any* 2D encoder wins or DINOv2 specifically).
- **3D feature:** **Point-MAE** on *xyz* points sampled from the mesh surface (self-supervised, pure geometry — the 3D analogue of DINOv2).
- Two further optional encoders were **attempted but deferred** (documented, not stubbed): **PE-Core** (2D) needs Python ≥3.11 (the box runs 3.10); **OpenShape/ULIP-2** (3D) is CUDA-coupled PointBERT and Point-MAE already covers the learned-3D feature.
- **Colour & texture are NOT used by any encoder** — the grey renders and xyz points carry only *shape*. (The colour shown in the viewer/`part_a_overview.png` is for human inspection only; the algorithms never see it. Trade-off: for true *appearance* similarity colour would matter.)

**Results (KMeans, cosine silhouette ↑)**

| Feature | silhouette |
|---|---|
| DINOv2 (2D) | **0.479** |
| Point-MAE (3D) | 0.407 |
| CLIP (2D) | 0.358 |

**Conclusion:** this is really **shape-from-2D-render vs shape-from-3D-points** (colour is not a
factor). DINOv2 separates best, but **CLIP — also 2D — does worst**, so it isn't "2D beats 3D";
it's that **DINOv2's fine-grained features** beat both pure geometry and CLIP's coarser,
language-aligned semantics. (n=14 → relative, not absolute.)

**Cross-check (`feature_distributions.png`).** To make sure this isn't an artifact of one
clustering, we histogram every pairwise **cosine distance** per encoder and split it into
*intra-cluster* vs *inter-cluster*; the gap between their means (Δmean) measures how
discriminative the features are without depending on a chosen *k*. The ranking is **identical to
silhouette — DINOv2 0.69 > Point-MAE 0.53 > CLIP 0.38** — and CLIP's intra/inter histograms
visibly **overlap** (little structure to cluster). An independent view agreeing with silhouette is
strong evidence the ordering is real.

---

## Part B — cluster 500 AI faces by attribute

**Question:** the faces are all distinct synthetic identities, so there's no identity to
recover — what *attribute* structure do the embeddings hold?

**Flow & choices**
- Generate 500 faces from thispersondoesnotexist.com.
- **Model: InsightFace / ArcFace** — face-specialized, and returns a 512-d embedding **plus**
  predicted age/gender/pose. The embedding is clustered; the attributes become *evidence* to
  interpret and validate clusters.
- Added a **generic DINOv2** on the same faces as an ablation (face-specialized vs general).
- **Colour *is* used here** — both encoders embed the colour face crop (the contrast with Part A's greyscale renders). Visual check: `part_b_overview_arcface_k_3_attribute.png` shows the UMAP three ways (cluster / gender / age) in two rows — points-only on top, a dense face sample below — with a metrics summary table. Filename + headline both name k and the selection, so `..._arcface_k_6_silhouette.png` and `..._dinov2_generic_k_3_silhouette.png` are unambiguous; `feature_distributions.png` splits pairwise distances by same/different gender/age.

**Results**
- KMeans k=6 splits cleanly by **gender + age** (two clusters 100% gender-pure; others stratify by age). Validated against the model's own gender labels: **purity 0.81**.
- **HDBSCAN found no dense clusters** — the honest signature of a *continuous* embedding manifold, not a bug.
- Generic DINOv2 (ablation) gave clusters that are **more separated *and* more gender-aligned** than ArcFace (silhouette 0.195 vs 0.045; gender purity 0.896 vs 0.808) — a general backbone already groups faces by gender. ArcFace's real edge is **age** (purity 0.602 vs 0.546) and that it *returns* the predicted age/gender/pose we validate against (DINOv2 gives none). Specialization buys the attribute **read-outs**, not better gender grouping.

**Conclusion:** ArcFace embeddings are organized primarily by **gender**, secondarily by
**age**; the space is continuous (so partitional methods impose soft, useful boundaries).
Finer splits also surface attributes we never labelled (e.g. an **eyewear** cluster) — the
metrics only score the labelled axes, not everything the embedding encodes.

---

## Why these metrics
No ground-truth labels, so we use **internal** metrics — **cosine silhouette** (cohesion vs
separation), Davies–Bouldin ↓, Calinski–Harabasz ↑. Part B additionally has **external**
validation: the model's predicted age/gender act as pseudo-labels, scored by **NMI / ARI /
purity**. This turns "is this clustering good?" from a guess into a measurement.

## How we chose the number of clusters
**Automatically, two ways (configurable per part).**
- **Silhouette sweep (default):** for KMeans/Agglomerative, sweep *k* and pick the *k* with the
  best cosine silhouette. HDBSCAN needs no *k*.
- **Attribute-driven (Part B):** sweep *k* and pick the *k* that maximizes **gender + age AMI**
  (adjusted mutual information — chance-corrected, so it doesn't reward over-splitting). This
  lets *the attributes we care about* drive *k*, not geometric separation.

**Why it matters — Part B comparison:**

| k-selection | k | gender purity | gender NMI |
|---|---|---|---|
| silhouette | 6 | 0.808 | 0.270 |
| **attribute (AMI)** | **3** | **0.864** | **0.398** |

Attribute-driven selection collapses to **k=3 (women / men / a young cohort)** and aligns
*better* with gender than silhouette's k=6 — confirming **gender is the dominant structure**.
(Agglomerative under AMI still rails to k_max=12: its finer splits stay gender-coherent so AMI
keeps rising — an honest signature of a continuous manifold, not a degenerate artifact.)

**Caveats:** *Part A (n=14)* "best k" is unstable (silhouette picked 7 for DINOv2, 3 for CLIP) —
illustrative only. *k is chosen per encoder by the same silhouette rule, so the differing k is itself
a result, not a confound; at a fixed k=6 the ranking is unchanged (DINOv2 0.471 > Point-MAE 0.404 >
CLIP 0.302).* *Part B* structure is continuous (gender × age × pose), so no single k is
"correct"; the attribute-driven k is the most *interpretable* choice for this goal.

---

## Reading the UMAP plots (and how they can mislead)
UMAP projects the high-dim embeddings to 2D so we can *see* neighbourhood structure — which
items sit together, whether groups are separable, how clusters relate. It's a lens, not the
analysis. Things it does **not** faithfully preserve:

- **Distance & gaps are not metric.** The space *between* clusters, and how far apart two
  clusters look, is largely arbitrary — don't read "these two clusters are far apart / similar"
  off the plot.
- **Cluster size & density are distorted.** UMAP equalises densities, so a big blob isn't
  necessarily a bigger or looser group than a small one.
- **It can manufacture or exaggerate clusters.** On a *continuous* manifold (exactly Part B),
  UMAP tends to tear the cloud into tidy islands that look more discrete than the data is —
  which is why we trust HDBSCAN's "no dense clusters" and the metrics over the picture.
- **Layout depends on hyperparameters & seed** (`n_neighbors`, `min_dist`); we fix the seed
  for reproducibility, but a different `n_neighbors` would redraw it.

**Most important caveat for this project:** we **cluster on the full-dimensional embeddings**,
not on the 2D coordinates — UMAP is only for display. So a point can sit visually *inside*
another cluster's region yet be correctly labelled by the high-D clustering. The colours
(cluster labels) are the truth; the 2D positions are an approximation. Treat the UMAP scatter
as a qualitative map to explore alongside the metrics, never as the evidence itself.

---

## Engineering highlights (brief)
- **Pluggable `FeatureExtractor` protocol** — adding an encoder is one module; viewers + metric
  tables pick it up automatically.
- Embeddings **cached to `.npy`** → decouples heavy GPU/CPU encoding from instant analysis/replots.
- **Point-MAE reimplemented in pure-torch** to run CPU-only (upstream is CUDA-coupled).
- **Interactive HTML viewers** (Plotly, self-contained): Part A glasses as cluster-coloured
  cards (colour render on hover); Part B faces on hover with age/gender/pose.
- Caught via review: GLB scene-graph transform bug; InsightFace missing frame-filling faces
  (det_size 640→320 recovered 123→500).
