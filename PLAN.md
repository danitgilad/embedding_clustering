# Project Plan & Session-Restore Log

> Living document. Updated continuously so the session can be **restored after a crash**.
> If you're a fresh session: read this top-to-bottom, then `DEFINITIONS.md`, then `task.md`,
> and resume from the "Current status" marker below.

---

## Current status

**Phase:** COMPLETE. All 23 tasks (0.1–4.5) done on branch `feature/implementation` (pushed
to github.com/danitgilad/embedding_clustering). Final whole-project review verdict: SHIP
(no blocking issues); all 4 cosmetic findings fixed (config defaults, README CLIP, DINOv2
guard, wired cluster montages). DEFINITIONS.md all 30 boxes ticked. Box e2e succeeded:
Part A DINOv2 (sil 0.479) vs Point-MAE (0.407); Part B 500 faces → KMeans k=6 clusters by
gender+age (gender purity 0.81). Fast suite 34 passed / 3 @slow deselected; @slow real-model
tests pass on box. Figures + results.json + montages in reports/.
**REMAINING:** finishing-the-branch (merge/PR decision). Token used for pushes — REVOKE it.
**Date:** 2026-06-06
**Spec:** `docs/superpowers/specs/2026-06-06-unsupervised-clustering-design.md`
**Plan:** `docs/superpowers/plans/2026-06-06-embedding-clustering.md` (Tasks 0.1 → 4.5)
**Next action:** Task 4.3 — sync to elem-danit1, `pip install -r requirements.txt`, find
Point-MAE pretrain checkpoint URL + confirm Point_MAE class API, `setup_encoders.sh`, run
`part-a all` + `part-b all` as detached jobs, run `pytest -m slow`, pull figures back.
**Branch not yet pushed** (PAT exposed → revoke; use SSH remote for future pushes).

---

## Key facts / environment

- Working dir: `/home/danit/projects/senior_task` (NOT a git repo yet).
- Data: 14 `.glb` glasses meshes in `assets/` (~1.3–4 MB each). `assets/` is EXCLUDED
  from final submission.
- Two independent tasks: **Part A** (3D glasses clustering) + **Part B** (face-image
  embedding clustering). See `DEFINITIONS.md` for the full acceptance checklist.

### Compute / infra
- Heavy compute → **elem-danit1** (A100 GPU, 12 CPU, ~83GB RAM) over SSH, via the
  `run-on-elem-danit1` skill. Resilient to VPN/SSH drops & box preemption.
- GCS/Vertex available there (project `zeekit-deep`) via login-shell ADC.

### Decisions locked in
- D1: Render Part-A images from the **triangulated mesh** (not point cloud).
      Source of insight: `~/projects/umap_viewer/glasses_3d_umap`.
- D2: **Fresh, clean build** — standalone project. Borrow proven techniques/snippets
      from `glasses_3d_umap` (mesh render via matplotlib Poly3DCollection, elem-danit1
      workflow, metric choices) but write self-contained code structured around the
      assignment's 2D-vs-3D feature-type comparison. No dependency on the old project.
- D3: Part-A **2D visual feature** = multi-view renders → frozen pretrained vision
      encoder, pooled to one vector per asset. **Pluggable encoder interface**:
      **DINOv2 = primary**; **CLIP** and **PE-Core** (Meta Perception Encoder,
      `facebook/PE-Core-*`) added if time allows, to enrich the feature comparison.
- D4: Part-A comparison axis is **2D-render-derived vs 3D-mesh-derived features**
      (NOT learned-vs-handcrafted — that was a mis-framing; task only contrasts
      "2D visual from renders" vs "3D geometric from mesh"). Therefore the **3D feature
      = a learned point-cloud encoder** on points sampled from the triangulated mesh
      surface (no rendering). Apples-to-apples: learned 2D embedding vs learned 3D
      embedding, modality being the only variable. Handcrafted geometric descriptors
      (bbox ratios, solidity, PCA elongation, D2 histogram) = optional enrichment if
      time allows. The old `glasses_3d_umap` project may overlap freely, but we assume
      NO additional knowledge/setup from it — this project stands alone.
- D5: Part-A **3D encoder = Point-MAE** (self-supervised, pure-geometry; mirrors DINOv2).
      Points sampled from the triangulated mesh surface. OpenShape/ULIP-2 optional secondary
      if time allows. Encoders need a documented bootstrap (custom CUDA ops + checkpoints) +
      pinned requirements.txt; heavy encode runs on elem-danit1 A100.
- D6: Part-B model = **InsightFace `buffalo_l` (ArcFace)**. Cluster the 512-D ArcFace
      embedding; use the model's auxiliary **age/gender/pose + landmarks** to interpret &
      validate clusters with evidence. Clean pip install (ONNX). Key insight: faces are
      AI-generated → identity clustering is meaningless, so clusters surface ATTRIBUTES.
      Optional DINOv2 generic comparison if time allows (reuses Part A code).
- D7: Part-B dataset ~**500 faces** from thispersondoesnotexist.com (count is a
      config/CLI param). Generation: polite rate-limited GET loop + hash-dedup;
      InsightFace detect+align per face. Preprocessing documented.
- D8: **Methodology = right-tool-per-part.** Preprocess (standardize/L2-norm → optional
      PCA). Cluster: KMeans (silhouette-swept k) + Agglomerative (dendrogram) for both;
      HDBSCAN added for Part B. Eval: internal metrics (cosine silhouette, Davies–Bouldin,
      Calinski–Harabasz) for both; Part B also validated vs InsightFace age/gender
      pseudo-labels (NMI/ARI/purity). Viz: UMAP scatter (by cluster & attribute), per-cluster
      montages, feature-distribution plots, cross-feature-type metric comparison table.
- D9: **Architecture = shared part-agnostic core + per-part front-ends.** A
      `FeatureExtractor` Protocol is the key seam; downstream (reduce→cluster→eval→viz) is
      identical across extractors so comparisons are fair. Embeddings cached as `.npy`+ids
      keyed by extractor name → expensive GPU encode decoupled from cheap local analysis.
      Layout: `main.py` argparse CLI (per-part + per-stage subcommands), `config/default.yaml`
      + typed-dataclass loader (no scattered paths), one-file-per-encoder under `extractors/`,
      pytest in `tests/`. See Section 2 of the brainstorm for the full tree.
- D10: **Documentation is continuous, not deferred.** Standing rule for the whole build:
      (a) append to PLAN.md Progress log after every stage/file; (b) update README sections
      incrementally as each part lands (not a big-bang at the end); (c) docstrings + type
      hints on every public function/class/module; (d) tick DEFINITIONS.md boxes as criteria
      are met; (e) log decisions here as they happen. README must always reflect current state.
- D11: **Run EVERYTHING on elem-danit1** (render, all encoders, Part-B face download,
      ArcFace, reduce/cluster/metrics/viz). Local machine is fragile/gets stuck → used only
      as a control terminal (edit code, issue commands, pull back final figures/HTML to view).
      Mirrors the old project's "everything executes on the box" workflow.
      - Part-B download is a plain HTTP GET (JPEG) → works headless on the box if it has
        outbound internet. ✅ VERIFIED 2026-06-06: box reaches TPDNE (548KB JPEG). No fallback
        needed — faces download directly on the box.
      - Project stays fully portable; box is the default workflow, not a hard dependency.
- D12: **Error handling = resilient batches, never silent.** Per-item try/except + a
      reported failure summary; real misconfig fails loud; no problem-hiding fallbacks.
      GLB loader degrades gracefully on missing colors/multi-mesh (warns, skips corrupt).
      Face-gen retries w/ backoff, content-hash dedup, drops 0/>1-face images (counts them).
      Embedding store hard-fails on (N,D)↔ids misalignment.
- D13: **Logging** via one `logging_setup.py` (level via `--log-level`), timestamped,
      no bare print(); per-stage counts+timings; run log copied under `outputs/`.
- D14: **Testing (pytest)** targets deterministic part-agnostic logic: mesh_io,
      embedding_store (incl. id-alignment guard), cluster (k-recovery on blobs), metrics,
      generate (mocked requests). Encoders tested via a fake FeatureExtractor; real-encoder
      runs are optional @slow smoke tests (no GPU/network in default suite).
- D15: **Optional encoders/descriptors DEFERRED** until the two primaries per part
      (DINOv2 + Point-MAE for A; InsightFace for B) produce a working end-to-end pipeline.
      See spec §12. Initial build = primaries + complete working pipeline first.
- D16: **Git remote** = `github.com/danitgilad/embedding_clustering` (user's personal GH).
      Remote configured WITHOUT the token in the URL; token used only for one-off pushes and
      never written to any file. Token was exposed in chat → user to revoke/rotate it.
      `.gitignore` excludes `assets/` (task says exclude from submission), `outputs/`, `data/`,
      `vendor/`, venvs, caches, `.remember/`.
- D17: **Point-MAE runs CPU-only via a self-contained pure-torch reimplementation.**
      Upstream Point-MAE is CUDA-coupled at import time (knn_cuda, pointnet2 FPS, chamfer
      extension) so it can't run CPU as-is. Solution: `src/part_a/extractors/_point_mae_backbone.py`
      reimplements only the encoder forward path (PointNet Encoder + ViT blocks, all pure torch)
      with torch FPS+KNN grouping, and loads the official `module.MAE_encoder.*` pretrained
      weights (348MB pretrain.pth). Submodule names mirror upstream so weights load by name
      (verified 0 missing/unexpected). Global feature = concat(max,mean) over group tokens = 768-d.
      `setup_encoders.sh` now ONLY downloads the checkpoint (no clone/CUDA build) → more
      reproducible. n_points=1024 (matches pretrain). This SUPERSEDES the plan's "import vendored
      repo" approach (Task 2.4). Commit e69e890.
- _(brainstorm complete & spec written; next: user reviews spec, then writing-plans)_

---

## Open questions (brainstorm)

- [ ] Part A 2D-visual feature: which approach? (render → CNN/CLIP embedding vs.
      classic image descriptors)
- [ ] Part A 3D-geometric feature: which descriptor(s)?
- [ ] Part B pretrained model: which one? (face-recognition embedding vs. generic
      vision backbone)
- [ ] Part B dataset size & generation strategy.
- [ ] Clustering algorithm(s) + how we compare feature types.
- [ ] Where compute runs (local vs. elem-danit1) per stage.

---

## Decision log
_(chronological; append as we decide)_

- 2026-06-06: Created `DEFINITIONS.md` + `PLAN.md`. Entered brainstorming.

---

## Progress log
_(append-only; what was actually built/done, with file paths)_

- 2026-06-06: Project inspected; `task.md` + 14 `.glb` assets present. No code yet.
- 2026-06-06: Created DEFINITIONS.md + PLAN.md. Brainstormed design (D1–D14).
- 2026-06-06: Wrote design spec → `docs/superpowers/specs/2026-06-06-unsupervised-clustering-design.md`.
  Spec self-review passed. Revised per user: removed "face-vs-generic" framing; added §12
  (optional encoders deferred until primaries work).
- 2026-06-06: `git init`; first commit `9acdb80` (docs only, no solution code). Remote
  `github.com/danitgilad/embedding_clustering` (tokenless URL). Pushed `main`. `view_glbs.py`
  (user's FiftyOne exploration script) left UNTRACKED on disk — decide later whether to
  relocate under `exploration/` or keep local-only. ACTION FOR USER: revoke the exposed PAT.
- 2026-06-06: Wrote implementation plan → `docs/superpowers/plans/2026-06-06-embedding-clustering.md`.
  Phases: 0 infra/scaffold, 1 shared core, 2 Part A, 3 Part B, 4 CLI/integration/docs.
  TDD throughout; heavy encoders behind @slow + run on box. Self-review passed.
- 2026-06-06: Started subagent-driven execution on branch `feature/implementation`.
  **Task 0.1 DONE** (infra verified): elem-danit1 reachable passwordless, A100-40GB,
  Python 3.10.12; TPDNE reachable from box (548KB JPEG); Point-MAE repo reachable
  (github.com/Pang-Yatian/Point-MAE HEAD 7445a68). Point-MAE checkpoint URL: TBD — look up
  from repo README during Task 2.4 setup (env POINT_MAE_CKPT_URL).
- 2026-06-06: **Phase 0 foundation DONE** (Tasks 0.2–0.5), APPROVED by review (6 tests pass).
  Commits 148dd7a, add4e53, 5b36a1e, aae8e9c. Created: skeleton + __init__.py tree,
  requirements.txt, config/default.yaml, README stub, src/logging_setup.py, src/config.py
  (typed dataclasses + dotted overrides), src/utils/{seeding,io}.py. Local `.venv` has the
  lightweight test deps. Minor non-blocking notes: logging level validation, inline numpy
  import, missing tests/utils/ pkg dir — deferred.
- 2026-06-06: **Phase 1 shared core DONE** (Tasks 1.1–1.6), APPROVED by review (20 tests pass).
  Commits 7c48048, f3c12d6, f4e993b, 02dec73, f2e8452, f00cc6b. Created core/{types,
  embedding_store, reduce, cluster, metrics, visualize}.py. NOTE: plan's test_reduce fixture
  had an impossible assertion ([3,4] == column mean → [0,0] un-normalizable); implementer
  correctly fixed the FIXTURE to [3,5] (impl unchanged). Optional low-sev TODOs: type `make`
  param in cluster._best_k; document frozen+ndarray unhashable; add ndim!=2 + unknown-step tests.
- 2026-06-06: **Part A geometry DONE** (Tasks 2.1–2.2). Commits 25c9402, 9c5874e, then fix
  adb97ab. Review (with real-GLB smoke test) caught a CRITICAL bug: `trimesh.util.concatenate`
  drops scene-graph node transforms → multi-part meshes (temple arms) misplaced by ~object
  size. FIXED to `obj.dump(concatenate=True)` (bakes transforms) + regression test. Also fixed
  render output to exact size_px×size_px (was cropped by bbox_inches=tight) and added log on
  vertex-color fallback. Verified on real asset: 17890 verts, 256×256 render. 24 tests pass.
  LEARNING (for README): always flatten GLB scenes with dump(concatenate=True), not util.concatenate.
- 2026-06-06: **Part A extractors + pipeline DONE** (Tasks 2.3–2.5 + pytest.ini early).
  Commits a924291 (pytest.ini slow marker), f5042da (DINOv2), 00cf7a1 (Point-MAE + setup
  script), 273f777 (pipeline). APPROVED. 27 passed, 2 @slow deselected. torch/transformers
  imported lazily (verified). Point-MAE `_load_model`/`_encode` repo-specific bits deferred
  to box run (Task 4.3/2.4 real). FOLLOW-UPS (non-blocking, do in box-run hardening):
  guard np.vstack([]) when all assets skipped (dinov2+point_mae); tighten run_clustering_stage
  return type to dict[str,dict].
- 2026-06-06: **Task 4.3 IN PROGRESS** (box run). Env built on elem-danit1
  (/mnt/workspace/projects/embedding_clustering, CPU torch 2.3.1+cpu, all deps). Assets seeded
  (byte-identical to local). Point-MAE CPU backbone written + verified (slow tests pass: 768-d).
  Full driver (part-a all → part-b all) launched detached (PID 59524, run.log, run_state/.done
  markers). Polling for completion. Point-MAE checkpoint at checkpoints/point_mae_pretrain.pth.
- 2026-06-06: **Part B DONE** (Tasks 3.1–3.3), APPROVED. Commits f91835a (generate),
  59d4b6b (arcface), 8001236 (pipeline). 32 passed, 3 @slow deselected. arcface imports
  insightface/cv2 lazily; raises on all-skipped (no empty vstack); _gender handles .sex/.gender
  version diffs. pytest.ini (Task 4.2) was done early. Minor cosmetic notes (a few missing
  type annots/docstrings on privates) — non-blocking.
  REMAINING: 4.1 CLI, 4.3 box e2e run, 4.4 README, 4.5 acceptance.
- 2026-06-06: **Task 4.1 CLI DONE**, APPROVED (wiring cross-checked exact vs both pipeline
  signatures). Commit d9e3072. `main.py` argparse: `part {part-a,part-b} stage {render,
  generate,extract,cluster,all}` + --config/--log-level/--set/--n. Used dataclasses.asdict for
  umap dict. 34 passed, 3 @slow deselected. **All 19 code tasks complete.** pytest.ini=Task4.2 done.
