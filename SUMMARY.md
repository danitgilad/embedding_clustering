# Project Summary

Two unsupervised clustering pipelines over different modalities, sharing one backbone:
**extract embeddings → preprocess → cluster → evaluate → visualize**. Only the feature
extractor changes between tasks; everything downstream is held identical so comparisons are fair.

---

## Part A — cluster 14 3D glasses by appearance

**Question:** does a *2D* feature (from rendered images) or a *3D* feature (from the mesh)
better capture "similar-looking glasses"?

**Flow & choices**
- Render each `.glb` from 4 views off the **triangulated mesh** (smoother than a point cloud).
- **2D feature:** frozen **DINOv2** on the renders (strong general visual embedding), views mean-pooled.
- **3D feature:** **Point-MAE** on points sampled from the mesh surface (self-supervised, pure geometry — the 3D analogue of DINOv2). No rendering.
- Later added **CLIP** as a 2nd 2D encoder to test whether *any* 2D encoder wins, or DINOv2 specifically.

**Results (KMeans, cosine silhouette ↑)**

| Feature | silhouette |
|---|---|
| DINOv2 (2D) | **0.479** |
| Point-MAE (3D) | 0.407 |
| CLIP (2D) | 0.358 |

**Conclusion:** DINOv2 separates best, but **CLIP — also 2D — does worst**. So it isn't
"2D beats 3D"; it's that **DINOv2's fine-grained visual features** beat both pure geometry
and CLIP's coarser, language-aligned semantics. (n=14 → relative, not absolute.)

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

**Results**
- KMeans k=6 splits cleanly by **gender + age** (two clusters 100% gender-pure; others stratify by age). Validated against the model's own gender labels: **purity 0.81**.
- **HDBSCAN found no dense clusters** — the honest signature of a *continuous* embedding manifold, not a bug.
- Generic DINOv2 gave more *separated* clusters (silhouette 0.195 vs 0.045) but they key on pose/lighting, not attributes — **higher silhouette ≠ more meaningful**. ArcFace earns its place by producing attribute-meaningful groups.

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
illustrative only. *Part B* structure is continuous (gender × age × pose), so no single k is
"correct"; the attribute-driven k is the most *interpretable* choice for this goal.

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
