# Methods Reference

Background on every embedding, reduction, clustering, and metric method used in this project.
The **README** covers *approach and findings*; this file is the *glossary* — for each method:
**what** it is, **why** it fits here, and the key parameter or caveat. All parameters live in
`config/default.yaml`.

| Stage | Methods |
|---|---|
| Feature extractors | [DINOv2](#dinov2) · [CLIP](#clip) · [Point-MAE](#point-mae) · [ArcFace](#arcface-insightface-buffalo_l) |
| Reduction | [standardize + L2-norm](#standardize--l2-normalize) · [PCA](#pca-optional) · [UMAP](#umap) |
| Clustering | [KMeans](#kmeans) · [Agglomerative](#agglomerative) · [HDBSCAN](#hdbscan) |
| Choosing *k* | [silhouette sweep](#silhouette-sweep) · [attribute-driven (AMI)](#attribute-driven-ami) |
| Internal metrics | [silhouette](#silhouette-cosine) · [Davies–Bouldin](#daviesbouldin) · [Calinski–Harabasz](#calinskiharabasz) |
| External metrics | [purity](#purity) · [NMI](#nmi) · [ARI](#ari) · [AMI](#ami) |

---

## Feature extractors (embeddings)

### DINOv2
Self-supervised vision transformer (`facebook/dinov2-base`). We take the **CLS token → 768-d**
per rendered view and mean-pool the 4 views into one vector. *Used:* Part A primary 2D feature;
Part B generic-backbone ablation (on the colour face crop). *Why:* its self-distilled features
are known to encode fine-grained shape/structure, which is exactly what separates similar frames.

### CLIP
Image–text contrastive model (`openai/clip-vit-base-patch32`); we use the **image tower → 512-d**
(same render → mean-pool pipeline as DINOv2). *Used:* Part A second 2D encoder. *Why:* a control
— it tests whether *any* 2D render feature wins or DINOv2 specifically. Its language-aligned
embedding is **coarser/semantic** ("a pair of glasses"), so it under-resolves near-identical
products (collapses to k=3 here).

### Point-MAE
Masked-autoencoder transformer pretrained on 3D point clouds (ShapeNet). We sample **1024 surface
points** from the mesh and encode them with the pretrained encoder, **max ++ mean pooling** the
group tokens → 768-d. *Used:* Part A 3D feature (operates on geometry directly, **no rendering**).
*Note:* reimplemented in pure-torch to run CPU-only (upstream couples grouping to CUDA ops);
loads the official weights with 0 missing keys.

### ArcFace (InsightFace `buffalo_l`)
Face-recognition model trained with the **additive-angular-margin (ArcFace) loss**, which makes
identities linearly separable on a hypersphere. Per face it returns a **512-d embedding plus
predicted age / gender / pose**. *Used:* Part B primary. *Why:* the embedding gives a strong
face representation to cluster, and the predicted attributes double as **pseudo-labels** to
interpret and validate clusters (no other encoder provides these). *Preprocessing:* InsightFace
**detects, aligns, and crops** the face internally; we keep the largest detected face and use
`det_size=320` (large det_size misses the frame-filling synthetic faces — see README Challenges).

---

## Preprocessing & dimensionality reduction

### Standardize + L2-normalize
Per-feature zero-mean/unit-variance (`StandardScaler`) then row L2-normalization, applied to
every embedding before clustering. *Why:* puts all encoders on a comparable scale and makes
Euclidean geometry on the normalized rows equivalent to **cosine** similarity — the metric we
cluster and score with. Order is configurable (`reduce.preprocess`).

### PCA (optional)
Linear projection to `reduce.pca_components` dimensions. **Off by default** (`null`) — the
datasets are small and the embeddings already compact — but available as a denoising/whitening
step for higher-dimensional inputs.

### UMAP
Nonlinear 2D projection (`n_neighbors=10`, `min_dist=0.1`, **cosine**) used **for display only**
— every scatter/overview is a UMAP layout. *Caveat:* UMAP distances and gaps are **not metric**
and it can exaggerate clusters on a continuous manifold, so **all clustering and metrics run on
the full-dimensional embeddings**, never on the 2D coordinates. Seed-fixed for reproducibility.

---

## Clustering algorithms

### KMeans
Partitions points into *k* spherical clusters by minimizing within-cluster variance. *Used:*
primary algorithm in both parts. *Why:* fast, deterministic (fixed seed), and a good fit when we
*want* a fixed number of groups; *k* is chosen by sweep (below). On a continuous manifold it
imposes useful but soft boundaries.

### Agglomerative
Bottom-up hierarchical merging (Ward linkage). *Used:* second algorithm in both parts as a
cross-check. *Why:* makes no centroid/sphericity assumption, so agreement with KMeans is evidence
the structure is real, not an artifact of one algorithm.

### HDBSCAN
Density-based clustering (no preset *k*). It declares a cluster only where a region of at least
`min_cluster_size` points is **denser than its surroundings** and separated by lower-density gaps;
anything else is **noise** (label −1). We use `min_cluster_size = max(5, N/20)` (= 25 for 500 faces).
*Used:* Part B diagnostic. *Why:* it can answer "are there genuinely dense, separable clusters?" —
on the continuous ArcFace manifold it finds none and labels **everything noise → k=0**. That is a
*legitimate result* meaning **"no density-separated groups exist"** (KMeans/Agglomerative can't say
this — they always return the *k* you ask for), and the honest signature of a continuous embedding
space rather than discrete blobs.

---

## Choosing *k*

### Silhouette sweep
Default selector: cluster for each *k* in a range and keep the *k* with the best mean **cosine
silhouette**. *Used:* both parts (Part A k∈[2,8], Part B k∈[2,12]). Picks the most
**geometrically separated** partition.

### Attribute-driven (AMI)
Part B alternative (`clustering.k_selection: attribute`): pick the *k* maximizing
**gender AMI + age AMI**. *Why:* when the goal is *attribute* structure, this lets the attributes
we care about choose *k* instead of geometry. Uses **adjusted** MI so it **doesn't reward
over-splitting** to k_max. On KMeans it collapses to a cleaner gender split (k=3).

---

## Cluster-quality metrics

### Internal (no labels needed)

#### Silhouette (cosine)
Per point, `(b − a) / max(a, b)` where `a` = mean intra-cluster distance, `b` = mean nearest-other-
cluster distance; averaged over points. Range −1…1, **higher = better** separation. Our headline
internal metric (computed with **cosine** distance to match preprocessing).

#### Davies–Bouldin
Average over clusters of the worst within-vs-between dispersion ratio. **Lower = better**
(0 is ideal). A separation check that, unlike silhouette, is unbounded and penalizes overlap.

#### Calinski–Harabasz
Ratio of between-cluster to within-cluster dispersion (a variance-ratio "F-statistic"). **Higher =
better**. Complements the other two; tends to favor compact, well-separated clusters.

### External (cluster labels vs pseudo-labels) — Part B only

These score the clustering against InsightFace's predicted **gender** and **age-bucket** labels
(e.g. *gender purity*, *age purity*). They turn "is this clustering meaningful?" into a number.

#### Purity
Fraction of points that fall in the **majority pseudo-class of their cluster**. Intuitive but
**not chance-corrected** — it rises as clusters get smaller, so read it alongside NMI/AMI.

#### NMI
Normalized mutual information between cluster labels and pseudo-labels (0…1). Measures shared
information regardless of label naming; **higher = more aligned**.

#### ARI
Adjusted Rand Index — pair-agreement between the two labelings, **corrected for chance** (0 ≈
random, 1 = identical). Stricter than purity/NMI.

#### AMI
Adjusted mutual information — NMI's **chance-corrected** sibling. Used as the **k-selection
objective** (gender AMI + age AMI) precisely because it does not creep upward with more clusters.
