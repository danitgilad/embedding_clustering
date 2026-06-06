# Assignment Definitions & Acceptance Checklist

> Single source of truth distilled from `task.md`. We check our deliverable against
> every box here before calling the task done. Do **not** edit the intent of these
> items — only tick them off (`[x]`) or add clarifying sub-notes.

## Global / Project-structure requirements

- [ ] Delivered as a **well-structured Python project** — NOT notebooks, NOT flat scripts.
- [ ] `src/` directory with meaningful separation of concerns, using `__init__.py` modules.
      (Does NOT need to be an installable package.)
- [ ] Clear entry point: `main.py` and/or a CLI (`argparse`) that runs each part.
- [ ] Configuration separated from logic — **no hardcoded paths scattered** through code.
- [ ] `requirements.txt` listing all dependencies (reproducible setup).
- [ ] `README.md` with:
  - [ ] Setup & installation instructions.
  - [ ] How to run each part of the assignment.
  - [ ] Summary of approach, findings, and key decisions for each part.
- [ ] Proper **logging** — no bare `print()` for operational output.
- [ ] Meaningful function / class / module names.
- [ ] Docstrings and type hints (encouraged → we treat as expected).

## Part A — Unsupervised Clustering of 3D Assets (glasses)

**Objective:** cluster the `.glb` glasses by geometric & visual similarity. No single
correct answer; justification matters as much as result.

- [ ] **Dataset exploration**: load `.glb` files from `assets/`; examine internal
      structure (mesh components, materials organization).
- [ ] **Feature extraction — at least two types:**
  - [ ] One **2D visual feature** derived from **rendered images** of the assets.
        (Decision: render from the **triangulated mesh**, not a point cloud.)
  - [ ] One **3D geometric feature** computed **directly from the mesh** (no rendering).
  - [ ] Analyze feature distributions and discriminative properties.
- [ ] **Clustering analysis**: apply clustering / unsupervised techniques; compare
      expressiveness of the different feature types and how each affects clusters.
- [ ] **Visualization & reporting**: visualize clusters; summarize similarities found;
      document process, challenges, observations in README.
- [ ] Output visualizations saved as **image files**.

## Part B — Unsupervised Classification of Face Images via Pretrained Embeddings

**Objective:** cluster/analyze AI-generated faces using embeddings from a pretrained model.

- [ ] **Dataset creation**: programmatically generate & save faces locally from
      thispersondoesnotexist.com (or equivalent). Document generation approach,
      dataset size, preprocessing.
- [ ] **Pretrained model selection**: choose a suitable pretrained model
      (classification / detection / segmentation); justify choice in README.
- [ ] **Embedding extraction**: feed dataset through model, extract a representative
      embedding per image.
- [ ] **Feature analysis & classification**: analyze embeddings (properties, patterns);
      apply clustering / unsupervised techniques; evaluate and **iterate** to improve.

## Evaluation criteria (graded dimensions)

| Criterion | What they look for |
|---|---|
| Code quality | Modularity, readability, separation of concerns, DRY |
| Project structure | Organization, clear entry points, config management |
| Engineering practices | Logging, error handling, type hints, docstrings |
| Research ability | Thoughtful tool/model/technique selection with justification |
| Analysis depth | Feature extraction, clustering methodology, evaluation quality |
| Visualization | Clear, informative visualizations supporting the analysis |
| Documentation | README: setup, usage, approach, findings |

## Deliverables checklist

- [ ] zip archive **or** git repo.
- [ ] Structured Python project (not notebooks/flat scripts).
- [ ] `README.md` (setup, run, approach, findings).
- [ ] `requirements.txt`.
- [ ] Output visualizations saved as image files.
- [ ] `assets/` folder **EXCLUDED** from submission (they have the data).

## Project-specific decisions (ours, agreed during brainstorm)

- Rendering uses the **triangulated mesh**, not a 3D point cloud
  (learned from `~/projects/umap_viewer/glasses_3d_umap` — gives better images).
- Heavy compute may be offloaded to **elem-danit1** (A100 GPU) over SSH via the
  `run-on-elem-danit1` skill.
- _(More to be appended as the design is finalized.)_
