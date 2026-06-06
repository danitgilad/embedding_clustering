# Design Spec — Interactive Viewers, Richer Figures & Encoder Comparison

**Date:** 2026-06-06
**Status:** Approved (brainstorm) — extends the completed base project.
**Builds on:** `docs/superpowers/specs/2026-06-06-unsupervised-clustering-design.md`
**Reference:** ports the approach of `~/projects/umap_viewer/glasses_3d_umap`
(`src/glasses_umap/umap_view.py`, `scripts/build_umap_viewer.py`).

## Goal

Make the clustering results visually inspectable and interactive, and run the optional
encoder comparison we deferred:

1. **Interactive self-contained HTML viewers** for Part A and Part B (Plotly), so a point can
   be correlated to the image it represents.
2. **Richer/verbose PNGs** (headlines, per-cluster stats, explanations) kept alongside.
3. The **optional encoders** as comparison toggles: CLIP & PE-Core (Part A 2D),
   DINOv2-generic (Part B), OpenShape/ULIP-2 (Part A 3D, best-effort).

**Working principle:** all edits to existing files are **additive and surgical** — extend,
do not rewrite or remove prior behaviour.

## Phasing (deliver value early)

- **Phase 1 — viewers for CURRENT results.** Build the viewer infrastructure + verbose PNGs
  and generate HTMLs from the encoders already run (A: DINOv2 + Point-MAE; B: ArcFace).
  Ship working interactive viewers.
- **Phase 2 — optional encoders + regenerate.** Add CLIP, DINOv2-generic, PE-Core,
  OpenShape (best-effort); re-encode on the box; regenerate the viewers/tables with the
  fuller comparison.

## Components

### 1. `src/core/html_viewer.py` (new, shared, render-only)
Adapted from `umap_view.py`. Pure rendering — coords/labels/metrics/thumbnails passed in.
- `build_viewer_html(projections, ids, thumbs, hover_meta, *, title, intro, always_show_thumbs, metric_cols) -> str`
  - `projections`: `{encoder_name: {"coords2d": (n,2), "labels": (n,), "metrics": {..}}}` — one
    toggle button each; first is shown by default.
  - One Plotly trace per cluster (stable cluster→colour), legend, hover tooltip.
  - **Hover tooltip** shows the point's thumbnail + id + any `hover_meta[id]` fields.
  - `always_show_thumbs=True` (Part A): place each thumbnail at its UMAP coord on a
    cluster-coloured card (umap_viewer style). `False` (Part B): hover-only.
  - **Metrics comparison table** across encoders, best cell highlighted per column.
  - Self-contained: Plotly via CDN, thumbnails base64-embedded.
- Unit-testable: returns an HTML string containing each encoder name, each id, and a
  `<script src="...plotly...">` tag; no network/model needed.

### 2. Viewer assembly = a new CLI `viewer` stage (consumes cached `.npy`)
Mirrors umap_viewer's Stage C: decoupled from encoding, so the HTML can be rebuilt instantly
without re-encoding.
- `main.py` gains stage `viewer` for both parts (and `all` runs it last).
- **Part A** (`src/part_a/viewer.py`): for each `outputs/part_a/<enc>.npy` + `<enc>.ids.json`,
  recompute UMAP + KMeans + internal metrics (reusing `core.reduce`/`cluster`/`metrics`,
  deterministic via seed); thumbnails = front-view renders
  (`data/part_a_renders/<id>_v0.png`, base64); `always_show_thumbs=True`; hover meta = none
  extra (id + cluster suffice). Write `outputs/part_a/viewer.html`.
- **Part B** (`src/part_b/viewer.py`): same recompute over `outputs/part_b/<enc>.npy`;
  thumbnails = face images **downscaled to ~96 px** (keeps 500-point HTML ~1–2 MB);
  `always_show_thumbs=False`; hover meta = **age, gender, pose** from
  `outputs/part_b/arcface_attributes.json`. Write `outputs/part_b/viewer.html`.
- Committed copies land in `reports/part_a/viewer.html`, `reports/part_b/viewer.html`.

### 3. Persist per-face attributes (small, additive)
`ArcFaceExtractor` already collects `self.attributes` (id → age/gender/pose_yaw). Add: the
Part B pipeline writes `outputs/part_b/arcface_attributes.json` after extract. Needed so the
viewer hover has real per-face data without re-detecting.

### 4. Verbose PNGs (additive to `visualize.py`)
- `cluster_montage`: add an optional `row_titles: dict[int, str]` so callers can label each
  cluster row (Part B: `C0 · n=116 · 100% F · age 44`; Part A: `cluster k (n=…)`), plus a
  descriptive suptitle/caption. Default behaviour (no titles) unchanged → existing test passes.
- Pipelines pass `row_titles` built from the cluster profiles (Part B) / sizes (Part A).
- Scatter/metric-table titles get a one-line explanation.

### 5. Optional encoders (Phase 2; each = one extractor module + registry entry)
- `src/part_a/extractors/clip.py` — CLIP image features on renders (`transformers`,
  `openai/clip-vit-base-patch32` or similar). config `part_a.encoders_2d += [clip]`.
- `src/part_b/extractors/dinov2_generic.py` — DINOv2 on face images; Part B loops encoders
  like Part A. config `part_b.encoders += [arcface, dinov2_generic]` (arcface stays primary;
  dinov2_generic has no attributes → hover falls back to id+cluster for that toggle).
- `src/part_a/extractors/pe_core.py` — Meta Perception Encoder (PE-Core); setup documented
  (`perception_models`/timm). Moderate risk; if unavailable, documented skip.
- `openshape` (Part A 3D) — **best-effort** pure-torch CPU port (like Point-MAE). If the port
  is disproportionate, log a documented skip rather than burn unbounded time.
- All heavy encoding runs on elem-danit1; embeddings cached as `.npy` so the viewer stage is
  unchanged.

## Outputs

- `outputs/<part>/viewer.html` (+ `reports/<part>/viewer.html` committed).
- Upgraded `*_clusters_montage.png` and titled scatter/metric PNGs.
- `outputs/part_b/arcface_attributes.json`.
- README gains a short **Viewers** section (how to open, what hover shows).

## Testing

- `html_viewer`: HTML contains expected encoder names + ids + Plotly script (no network).
- thumbnail downscaling helper: output size correct.
- attribute persistence: round-trips `{id: {age,gender,pose}}`.
- `cluster_montage` with `row_titles`: still writes a PNG (existing test unaffected).
- Real encoders remain `@slow` (box only).

## Non-goals

- No server/app — static self-contained HTML only.
- No change to the core clustering/encoding contracts beyond the additive items above.
