# Interactive Viewers — Implementation Plan (Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add self-contained interactive HTML viewers (Plotly) for Part A and Part B that correlate each UMAP point to the image it represents, plus richer annotated PNGs — generated from the CURRENT results (Part A: DINOv2 + Point-MAE; Part B: ArcFace). (Phase 2 — optional encoders — is planned separately later.)

**Architecture:** A new render-only module `src/core/html_viewer.py` (adapted from the user's `umap_view.py`) builds the HTML from coords/labels/metrics/thumbnails. Per-part `viewer.py` modules + a new CLI `viewer` stage consume the cached `*.npy` (decoupled from encoding, like the umap_viewer's Stage C), recompute UMAP+clusters+metrics deterministically, build thumbnails, and write `viewer.html`. All edits to existing files are **additive/surgical**.

**Tech Stack:** Plotly (CDN, via `plotly` python to emit figure JSON), Pillow (thumbnails), numpy, existing `core.reduce`/`cluster`/`metrics`.

**Reference:** `~/projects/umap_viewer/glasses_3d_umap/src/glasses_umap/umap_view.py`.
**Spec:** `docs/superpowers/specs/2026-06-06-interactive-viewers-design.md`.
**Standing rules:** docstrings + type hints; no bare `print()` in `src/`; `./.venv/bin/pytest` for the fast suite; commit per task; append a line to `PLAN.md` Progress log after each task.

> NOTE: `plotly` is not yet in `requirements.txt` or the local `.venv`. Task P1.0 adds it.

---

## File Structure

| File | Responsibility |
|---|---|
| `requirements.txt` | add `plotly==5.22.0` (additive) |
| `src/core/html_viewer.py` | **new** — render-only Plotly HTML builder + thumbnail helper |
| `src/part_a/viewer.py` | **new** — assemble Part A viewer from cached `outputs/part_a/*.npy` |
| `src/part_b/viewer.py` | **new** — assemble Part B viewer; hover age/gender/pose |
| `src/part_b/pipeline.py` | additive: persist `arcface_attributes.json` |
| `src/core/visualize.py` | additive: `cluster_montage(row_titles=...)` + caption |
| `src/part_a/pipeline.py`, `src/part_b/pipeline.py` | additive: pass `row_titles` to montage |
| `main.py` | additive: `viewer` stage for both parts; include in `all` |
| `tests/core/test_html_viewer.py`, `tests/part_b/test_attributes.py` | new tests |

---

## Task P1.0: Add plotly dependency

**Files:** Modify `requirements.txt`; install into `.venv`.

- [ ] **Step 1:** Append `plotly==5.22.0` to `requirements.txt` (additive — do not reorder existing lines).
- [ ] **Step 2:** Install locally: `./.venv/bin/pip install -q plotly==5.22.0`
- [ ] **Step 3:** Verify: `./.venv/bin/python -c "import plotly; print(plotly.__version__)"` → `5.22.0`
- [ ] **Step 4: Commit**
  ```bash
  git add requirements.txt && git commit -m "build: add plotly for interactive viewers"
  ```

---

## Task P1.1: html_viewer — thumbnail helper (TDD)

**Files:** Create `src/core/html_viewer.py`; Test `tests/core/test_html_viewer.py`.

- [ ] **Step 1: Write the failing test**
  ```python
  # tests/core/test_html_viewer.py
  from PIL import Image
  from src.core.html_viewer import image_to_data_uri

  def test_image_to_data_uri_downscales_and_encodes(tmp_path):
      p = tmp_path / "x.png"
      Image.new("RGB", (512, 256), (200, 10, 10)).save(p)
      uri = image_to_data_uri(p, max_px=96)
      assert uri.startswith("data:image/png;base64,")
      # decode and check the long side was capped at 96
      import base64, io
      raw = base64.b64decode(uri.split(",", 1)[1])
      assert max(Image.open(io.BytesIO(raw)).size) == 96

  def test_image_to_data_uri_missing_returns_empty(tmp_path):
      assert image_to_data_uri(tmp_path / "nope.png") == ""
  ```

- [ ] **Step 2: Run to verify it fails** — `./.venv/bin/pytest tests/core/test_html_viewer.py -v` → FAIL (module missing).

- [ ] **Step 3: Create `src/core/html_viewer.py` with the helper (rest added next task)**
  ```python
  """Self-contained interactive Plotly viewer for embedding clusters.

  Render-only: coordinates, cluster labels, metrics, and base64 thumbnails are passed in.
  One toggle button per encoder; one Plotly trace per cluster; hover shows the point's
  thumbnail + id + any extra meta. For small sets (always_show_thumbs=True) thumbnails are
  placed on the plot as cluster-coloured cards (umap_viewer style); for large sets they
  appear on hover only. Adapted from glasses_3d_umap/src/glasses_umap/umap_view.py.
  """
  from __future__ import annotations

  import base64
  import io
  from pathlib import Path


  def image_to_data_uri(path: str | Path, max_px: int = 96) -> str:
      """Downscale an image so its long side is <= max_px; return a base64 PNG data URI.

      Returns "" if the file is missing (the viewer then renders that point thumbnail-less).
      """
      from PIL import Image

      p = Path(path)
      if not p.exists():
          return ""
      img = Image.open(p).convert("RGBA")
      img.thumbnail((max_px, max_px), Image.LANCZOS)
      buf = io.BytesIO()
      img.save(buf, format="PNG")
      return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
  ```

- [ ] **Step 4: Run to verify it passes** — `./.venv/bin/pytest tests/core/test_html_viewer.py -v` → PASS.

- [ ] **Step 5: Commit**
  ```bash
  git add src/core/html_viewer.py tests/core/test_html_viewer.py
  git commit -m "feat: html_viewer image_to_data_uri thumbnail helper"
  ```

---

## Task P1.2: html_viewer — build_viewer_html (TDD)

**Files:** Modify `src/core/html_viewer.py`; extend `tests/core/test_html_viewer.py`.

- [ ] **Step 1: Add the failing test**
  ```python
  # append to tests/core/test_html_viewer.py
  import numpy as np
  from src.core.html_viewer import build_viewer_html

  def _proj(n, k):
      rng = np.random.RandomState(0)
      return {"coords2d": rng.rand(n, 2), "labels": np.arange(n) % k,
              "metrics": {"silhouette": 0.5, "davies_bouldin": 0.8, "calinski_harabasz": 4.0}}

  def test_build_viewer_html_contains_encoders_ids_and_plotly():
      ids = [f"a{i}" for i in range(6)]
      thumbs = ["data:image/png;base64,AAAA"] * 6
      projections = {"dinov2": _proj(6, 3), "point_mae": _proj(6, 2)}
      html = build_viewer_html(projections, ids, thumbs, hover_meta=None,
                               title="Part A", intro="hello", always_show_thumbs=True)
      assert "cdn.plot.ly" in html
      assert "dinov2" in html and "point_mae" in html
      assert "a0" in html and "a5" in html
      assert "<table" in html  # metrics comparison table
  ```

- [ ] **Step 2: Run to verify it fails** — `./.venv/bin/pytest tests/core/test_html_viewer.py::test_build_viewer_html_contains_encoders_ids_and_plotly -v` → FAIL.

- [ ] **Step 3: Append the builder to `src/core/html_viewer.py`**
  ```python
  import numpy as np

  _TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

  # metric -> direction for best-cell highlight (max = higher better, min = lower better)
  _METRIC_DIR = {"silhouette": max, "davies_bouldin": min, "calinski_harabasz": max,
                 "gender_purity": max, "age_purity": max, "gender_nmi": max, "age_nmi": max}


  def _fmt(v: object) -> str:
      try:
          return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.3f}"
      except (TypeError, ValueError):
          return str(v)


  def _figure_json(proj: dict, ids, thumbs, hover_meta, always_show_thumbs: bool) -> str:
      """One Plotly figure (JSON spec) for a single encoder. customdata=[id, cluster, thumb, meta]."""
      import plotly.graph_objects as go

      coords = np.asarray(proj["coords2d"], dtype=float)
      labels = np.asarray(proj["labels"])
      uniq = sorted(set(labels.tolist()))
      color = {c: _TAB10[i % len(_TAB10)] for i, c in enumerate(uniq)}

      def meta_str(i: str) -> str:
          if not hover_meta or i not in hover_meta:
              return ""
          return " · ".join(f"{k}:{v}" for k, v in hover_meta[i].items())

      fig = go.Figure()
      for c in uniq:
          m = labels == c
          idx = np.where(m)[0]
          cd = [[ids[j], int(c), thumbs[j], meta_str(ids[j])] for j in idx]
          fig.add_trace(go.Scatter(
              x=coords[m, 0].tolist(), y=coords[m, 1].tolist(), mode="markers",
              name=f"cluster {c} ({int(m.sum())})", legendgroup="c",
              marker=dict(color=color[c], size=11, line=dict(color="white", width=1)),
              customdata=cd,
              hovertemplate="<b>%{customdata[0]}</b><br>cluster %{customdata[1]}"
                            "<br>%{customdata[3]}<extra></extra>",
          ))

      shapes, images = [], []
      if always_show_thumbs:
          xs, ys = coords[:, 0], coords[:, 1]
          span = float(max(xs.ptp(), ys.ptp())) or 1.0
          s = 0.18 * span
          for j in range(len(coords)):
              if not thumbs[j]:
                  continue
              col = color[int(labels[j])]
              xi, yi = float(coords[j, 0]), float(coords[j, 1])
              shapes.append(dict(type="rect", xref="x", yref="y",
                                 x0=xi - s / 2, x1=xi + s / 2, y0=yi - s / 2, y1=yi + s / 2,
                                 fillcolor=col, opacity=0.45, line=dict(color=col, width=2),
                                 layer="below"))
              images.append(dict(source=thumbs[j], xref="x", yref="y", x=xi, y=yi,
                                 sizex=s, sizey=s, xanchor="center", yanchor="middle",
                                 sizing="contain", layer="above"))
      fig.update_layout(width=960, height=760, plot_bgcolor="#f8f8f8",
                        xaxis=dict(title="UMAP 1", zeroline=False),
                        yaxis=dict(title="UMAP 2", zeroline=False),
                        legend=dict(itemsizing="constant"),
                        margin=dict(l=55, r=160, t=10, b=45),
                        images=images, shapes=shapes)
      return fig.to_json().replace("\\u002f", "/")


  def _metrics_table(projections: dict) -> str:
      """Cross-encoder metrics table; best cell per column highlighted."""
      keys: list[str] = []
      for p in projections.values():
          for k in p["metrics"]:
              if k not in keys:
                  keys.append(k)
      best = {}
      for k in keys:
          agg = _METRIC_DIR.get(k, max)
          vals = {name: p["metrics"][k] for name, p in projections.items()
                  if isinstance(p["metrics"].get(k), (int, float)) and not np.isnan(p["metrics"][k])}
          best[k] = agg(vals, key=vals.get) if vals else None
      head = "<tr><th>encoder</th>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr>"
      rows = ""
      for name, p in projections.items():
          cells = f"<td>{name}</td>"
          for k in keys:
              win = ' class="win"' if best.get(k) == name else ""
              cells += f"<td{win}>{_fmt(p['metrics'].get(k))}</td>"
          rows += f"<tr>{cells}</tr>"
      return f'<table class="m">{head}{rows}</table>'


  def build_viewer_html(projections: dict[str, dict], ids: list[str], thumbs: list[str],
                        hover_meta: dict[str, dict] | None, *, title: str, intro: str,
                        always_show_thumbs: bool,
                        page_title: str = "Embedding Cluster Viewer") -> str:
      """Render the full self-contained explorer HTML.

      projections: {encoder_name: {"coords2d": (n,2), "labels": (n,), "metrics": {..}}}.
      ids/thumbs: length-n, aligned to every projection's row order.
      hover_meta: optional {id: {field: value}} shown in the hover tooltip.
      always_show_thumbs: True places thumbnails on the plot (small sets); False = hover-only.
      """
      names = list(projections)
      specs = {n: _figure_json(projections[n], ids, thumbs, hover_meta, always_show_thumbs)
               for n in names}
      specs_js = ",\n".join(f"'{n}': {s}" for n, s in specs.items())
      keys_js = ", ".join(f"'{n}'" for n in names)
      btns = "".join(
          f'<button id="b_{n}" onclick="show(\'{n}\')" class="tb{ " act" if i==0 else "" }">{n}</button>'
          for i, n in enumerate(names))
      divs = "".join(
          f'<div id="v_{n}" class="view"{"" if i==0 else " style=display:none"}></div>'
          for i, n in enumerate(names))
      table = _metrics_table(projections)
      return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
  <title>{page_title}</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>body{{font-family:sans-serif;margin:0;padding:14px 18px}}
  h2{{margin:0 0 4px}} p{{color:#555;font-size:13px;margin:2px 0 8px}}
  .tb{{padding:7px 15px;margin-right:7px;border:1px solid #aaa;border-radius:4px;cursor:pointer;background:#f0f0f0}}
  .tb.act{{background:#2c3e50;color:#fff}}
  table.m{{border-collapse:collapse;font-size:12px;margin:6px 0 12px}}
  table.m th,table.m td{{border:1px solid #ddd;padding:4px 9px;text-align:right}}
  table.m th{{background:#0f3460;color:#fff}} table.m td:first-child,table.m th:first-child{{text-align:left}}
  table.m td.win{{background:#cdebcd;font-weight:700}}</style></head>
  <body><h2>{title}</h2><p>{intro}</p>
  <p style="margin-top:0"><b>Clustering quality per encoder</b> (silhouette ↑, Davies–Bouldin ↓,
  Calinski–Harabasz ↑; green = best):</p>{table}
  <div style="margin:8px 0">{btns}</div>{divs}
  <div id="tip" style="position:fixed;display:none;z-index:9;background:#fff;border:1px solid #ccc;
  border-radius:6px;padding:7px;box-shadow:2px 4px 14px rgba(0,0,0,.25);font:12px/1.4 sans-serif;
  text-align:center;max-width:160px"></div>
  <script>
  var S={{ {specs_js} }}, K=[{keys_js}], R={{}};
  function tip(el){{var t=document.getElementById('tip');
    el.on('plotly_hover',function(e){{var d=e.points[0].customdata;var h='';
      if(d[2])h+='<img src="'+d[2]+'" style="max-width:140px;max-height:140px;display:block;margin:0 auto 4px">';
      h+='<b>'+d[0]+'</b><br>cluster '+d[1]+(d[3]?'<br>'+d[3]:'');t.innerHTML=h;
      var b=e.points[0].bbox||{{}};t.style.left=((b.x1||0)+12)+'px';t.style.top=((b.y0||0)-8)+'px';t.style.display='block';}});
    el.on('plotly_unhover',function(){{t.style.display='none';}});}}
  function show(n){{K.forEach(function(k){{var d=document.getElementById('v_'+k),b=document.getElementById('b_'+k);
    d.style.display=(k===n?'':'none');b.className='tb'+(k===n?' act':'');}});
    var div=document.getElementById('v_'+n);if(!R[n]){{Plotly.newPlot(div,S[n].data,S[n].layout,{{responsive:false}});R[n]=1;setTimeout(function(){{tip(div);}},400);}}}}
  show(K[0]);
  </script></body></html>"""
  ```

- [ ] **Step 4: Run to verify it passes** — `./.venv/bin/pytest tests/core/test_html_viewer.py -v` → PASS (both tests).

- [ ] **Step 5: Commit**
  ```bash
  git add src/core/html_viewer.py tests/core/test_html_viewer.py
  git commit -m "feat: html_viewer build_viewer_html (toggles, clusters, hover, metrics table)"
  ```

---

## Task P1.3: Persist per-face attributes (additive to Part B pipeline) (TDD)

**Files:** Modify `src/part_b/pipeline.py` (additive); Test `tests/part_b/test_attributes.py`.

- [ ] **Step 1: Write the failing test**
  ```python
  # tests/part_b/test_attributes.py
  import json
  import numpy as np
  from src.core.types import Asset, Embeddings
  from src.part_b.pipeline import run_clustering_stage

  class FakeFaceExtractor:
      name = "arcface"
      def __init__(self):
          self.attributes = {f"a{i}": {"age": 30 + i, "gender": "F" if i % 2 else "M",
                                       "pose_yaw": float(i)} for i in range(4)}
          self.skipped = {}
      def extract(self, items):
          return Embeddings(np.array([[0,0],[0.1,0],[5,5],[5.1,5]], float),
                            [a.id for a in items], self.name)

  def test_attributes_persisted(tmp_path):
      assets = [Asset(id=f"a{i}", path=tmp_path / f"a{i}.jpg") for i in range(4)]
      run_clustering_stage(FakeFaceExtractor(), assets, tmp_path, ["kmeans"], 2, 3,
                           ["l2norm"], None,
                           {"n_neighbors": 3, "min_dist": 0.1, "metric": "cosine"}, 0)
      attrs = json.loads((tmp_path / "arcface_attributes.json").read_text())
      assert attrs["a0"]["gender"] == "M" and "pose_yaw" in attrs["a0"]
  ```

- [ ] **Step 2: Run to verify it fails** — `./.venv/bin/pytest tests/part_b/test_attributes.py -v` → FAIL (no file written).

- [ ] **Step 3: Add attribute persistence in `src/part_b/pipeline.py`** — additive, immediately AFTER the existing `attrs = getattr(extractor, "attributes", {})` line in `run_clustering_stage` (do not change surrounding logic):
  ```python
      if attrs:
          from src.utils.io import write_json
          write_json(attrs, ensure_dir(out_dir) / f"{emb.name}_attributes.json")
  ```
  (`emb` and `out_dir` are already in scope; `write_json`/`ensure_dir` already imported at top — if `write_json` import is missing, it was added in the base project; keep the local import shown to be safe.)

- [ ] **Step 4: Run to verify it passes** — `./.venv/bin/pytest tests/part_b/test_attributes.py -v` → PASS.

- [ ] **Step 5: Commit**
  ```bash
  git add src/part_b/pipeline.py tests/part_b/test_attributes.py
  git commit -m "feat: persist per-face attributes (arcface_attributes.json) for viewer hover"
  ```

---

## Task P1.4: Verbose montage row titles (additive to visualize.py) (TDD)

**Files:** Modify `src/core/visualize.py` (additive); Test extends `tests/core/test_visualize.py`.

- [ ] **Step 1: Add the failing test**
  ```python
  # append to tests/core/test_visualize.py
  def test_cluster_montage_accepts_row_titles(tmp_path):
      from PIL import Image
      import numpy as np
      from src.core.visualize import cluster_montage
      imgs = []
      for i in range(4):
          p = tmp_path / f"{i}.png"; Image.new("RGB", (8, 8), (i*10, 0, 0)).save(p); imgs.append(p)
      out = cluster_montage(imgs, np.array([0, 0, 1, 1]), tmp_path / "m.png",
                            row_titles={0: "C0 n=2", 1: "C1 n=2"}, caption="demo")
      assert out.exists()
  ```

- [ ] **Step 2: Run to verify it fails** — `./.venv/bin/pytest tests/core/test_visualize.py::test_cluster_montage_accepts_row_titles -v` → FAIL (unexpected kwargs).

- [ ] **Step 3: Make `cluster_montage` accept `row_titles` + `caption`** — additive params with defaults, in `src/core/visualize.py`. Change the signature and the row-label + caption lines only:
  - Signature → add `row_titles: dict[int, str] | None = None, caption: str = ""` after `max_per_cluster`.
  - Replace the per-row ylabel line
    `axes[r][0].set_ylabel(f"c{c_lab}", rotation=0, labelpad=18, va="center")`
    with:
    ```python
            label = (row_titles or {}).get(c_lab, f"c{c_lab}")
            axes[r][0].set_ylabel(label, rotation=0, labelpad=42, ha="right", va="center",
                                  fontsize=8)
    ```
  - Replace `fig.suptitle("Clusters")` with:
    ```python
        fig.suptitle("Per-cluster sample thumbnails" + (f"\n{caption}" if caption else ""),
                     fontsize=11)
    ```

- [ ] **Step 4: Run to verify it passes** — `./.venv/bin/pytest tests/core/test_visualize.py -v` → PASS (new + existing montage test).

- [ ] **Step 5: Commit**
  ```bash
  git add src/core/visualize.py tests/core/test_visualize.py
  git commit -m "feat: cluster_montage row titles + caption (richer figures)"
  ```

---

## Task P1.5: Wire montage row-titles into pipelines (additive) (TDD via existing tests)

**Files:** Modify `src/part_a/pipeline.py`, `src/part_b/pipeline.py` (additive only in the montage block).

- [ ] **Step 1: Part A** — in `run_clustering_stage`, where `cluster_montage(...)` is called, build sizes and pass titles. Replace the existing montage call with:
  ```python
          from collections import Counter
          sizes = Counter(int(l) for l in primary_labels)
          titles = {c: f"cluster {c} (n={n})" for c, n in sizes.items()}
          cluster_montage([p for p, _ in sel], [lab for _, lab in sel],
                          fig_dir / f"{extractor.name}_clusters_montage.png",
                          row_titles=titles,
                          caption=f"{extractor.name}: glasses grouped by primary-algorithm cluster")
  ```

- [ ] **Step 2: Part B** — in `run_clustering_stage`, replace the existing montage call with one that uses the characterize profile of the primary algorithm:
  ```python
          prof = results.get(f"{algorithms[0]}__profile", {})
          titles = {int(c): f"C{c} · n={d['size']} · {d['pct_top_gender']*100:.0f}% "
                            f"{d['top_gender']} · age {d['mean_age']:.0f}"
                    for c, d in prof.items()}
          cluster_montage([p for p, _ in sel], [lab for _, lab in sel],
                          fig_dir / f"{emb.name}_clusters_montage.png",
                          row_titles=titles,
                          caption="Faces grouped by KMeans cluster (sample per cluster)")
  ```
  (`results`, `algorithms`, `sel`, `primary_labels`, `emb` are already in scope at that point.)

- [ ] **Step 3: Run the fast suite** — `./.venv/bin/pytest -q` → all pass (pipeline tests use fakes; montage now titled).

- [ ] **Step 4: Commit**
  ```bash
  git add src/part_a/pipeline.py src/part_b/pipeline.py
  git commit -m "feat: annotate cluster montages with per-cluster stats"
  ```

---

## Task P1.6: Part A viewer assembly + CLI `viewer` stage (TDD)

**Files:** Create `src/part_a/viewer.py`; Modify `main.py` (additive). Test `tests/part_a/test_viewer.py`.

- [ ] **Step 1: Write the failing test** (drives the builder with a fake on-disk embedding + a render)
  ```python
  # tests/part_a/test_viewer.py
  import json
  import numpy as np
  from PIL import Image
  from src.config import load_config
  from src.part_a.viewer import build_part_a_viewer

  def test_build_part_a_viewer_writes_html(tmp_path):
      out = tmp_path / "outputs"; out.mkdir()
      renders = tmp_path / "renders"; renders.mkdir()
      ids = [f"g{i}" for i in range(6)]
      np.save(out / "dinov2.npy", np.random.RandomState(0).rand(6, 32))
      (out / "dinov2.ids.json").write_text(json.dumps(ids))
      for i in ids:
          Image.new("RGB", (40, 40), (10, 90, 10)).save(renders / f"{i}_v0.png")
      cfg = load_config("config/default.yaml")
      html_path = build_part_a_viewer(cfg, out_dir=out, render_dir=renders)
      assert html_path.exists()
      txt = html_path.read_text()
      assert "dinov2" in txt and "g0" in txt and "cdn.plot.ly" in txt
  ```

- [ ] **Step 2: Run to verify it fails** — `./.venv/bin/pytest tests/part_a/test_viewer.py -v` → FAIL.

- [ ] **Step 3: Create `src/part_a/viewer.py`**
  ```python
  """Build the Part A interactive HTML viewer from cached embeddings (decoupled from encoding).

  For each outputs/part_a/<encoder>.npy it recomputes UMAP + KMeans + internal metrics
  (deterministic, reusing core.*), uses the front-view renders as always-visible thumbnails,
  and writes outputs/part_a/viewer.html.
  """
  from __future__ import annotations

  import logging
  from pathlib import Path

  from src.config import Config
  from src.core import metrics as M
  from src.core.cluster import cluster
  from src.core.embedding_store import load_embeddings
  from src.core.html_viewer import build_viewer_html, image_to_data_uri
  from src.core.reduce import preprocess, umap_2d
  from src.utils.io import sanitize_id

  log = logging.getLogger(__name__)


  def build_part_a_viewer(cfg: Config, out_dir: str | Path, render_dir: str | Path) -> Path:
      """Assemble outputs/part_a/viewer.html from every <encoder>.npy in out_dir."""
      out_dir, render_dir = Path(out_dir), Path(render_dir)
      npys = sorted(p for p in out_dir.glob("*.npy"))
      if not npys:
          raise FileNotFoundError(f"no embeddings (*.npy) in {out_dir}")
      ids = None
      projections: dict[str, dict] = {}
      um = cfg.reduce.umap
      algo = cfg.part_a.clustering.algorithms[0]
      for npy in npys:
          emb = load_embeddings(npy.stem, out_dir)
          if ids is None:
              ids = emb.ids
          elif emb.ids != ids:
              raise ValueError(f"id mismatch for {npy.stem}; viewer needs a shared id order")
          X = preprocess(emb.vectors, list(cfg.reduce.preprocess), pca_components=cfg.reduce.pca_components)
          coords = umap_2d(X, um.n_neighbors, um.min_dist, um.metric, cfg.seed)
          res = cluster(X, algo, cfg.part_a.clustering.k_min, cfg.part_a.clustering.k_max, cfg.seed)
          projections[npy.stem] = {"coords2d": coords, "labels": res.labels,
                                   "metrics": M.internal_metrics(X, res.labels)}
      thumbs = [image_to_data_uri(render_dir / f"{sanitize_id(i)}_v0.png", max_px=128) for i in ids]
      html = build_viewer_html(
          projections, ids, thumbs, hover_meta=None,
          title="Part A — 3D glasses: 2D-vs-3D feature clusters",
          intro=("Each point is one glasses asset, shown as its rendered thumbnail on a "
                 "cluster-coloured card. Buttons switch the feature/encoder. Hover for id."),
          always_show_thumbs=True, page_title="Part A — Glasses Cluster Viewer")
      out_html = out_dir / "viewer.html"
      out_html.write_text(html)
      log.info("Wrote %s", out_html)
      return out_html
  ```

- [ ] **Step 4: Run to verify it passes** — `./.venv/bin/pytest tests/part_a/test_viewer.py -v` → PASS.

- [ ] **Step 5: Wire CLI `viewer` stage (additive) in `main.py`:**
  - In `build_parser`, add `"viewer"` to the `stage` choices list (additive).
  - In `_run_part_a`, append:
    ```python
      if stage in ("viewer", "all"):
          from src.part_a.viewer import build_part_a_viewer
          build_part_a_viewer(cfg, out, render_dir)
    ```
    (place after the extract/cluster block; `out` and `render_dir` already in scope.)

- [ ] **Step 6: Run** — `./.venv/bin/pytest -q` → all pass; `./.venv/bin/python main.py --help` shows `viewer` in stage choices.

- [ ] **Step 7: Commit**
  ```bash
  git add src/part_a/viewer.py tests/part_a/test_viewer.py main.py
  git commit -m "feat: Part A interactive viewer + CLI viewer stage"
  ```

---

## Task P1.7: Part B viewer assembly + CLI (TDD)

**Files:** Create `src/part_b/viewer.py`; Modify `main.py` (additive). Test `tests/part_b/test_viewer.py`.

- [ ] **Step 1: Write the failing test**
  ```python
  # tests/part_b/test_viewer.py
  import json
  import numpy as np
  from PIL import Image
  from src.config import load_config
  from src.part_b.viewer import build_part_b_viewer

  def test_build_part_b_viewer_writes_html(tmp_path):
      out = tmp_path / "outputs"; out.mkdir()
      faces = tmp_path / "faces"; faces.mkdir()
      ids = [f"face_{i:04d}" for i in range(8)]
      np.save(out / "arcface.npy", np.random.RandomState(0).rand(8, 32))
      (out / "arcface.ids.json").write_text(json.dumps(ids))
      (out / "arcface_attributes.json").write_text(json.dumps(
          {i: {"age": 30, "gender": "F", "pose_yaw": 1.0} for i in ids}))
      for i in ids:
          Image.new("RGB", (64, 64), (90, 90, 10)).save(faces / f"{i}.jpg")
      cfg = load_config("config/default.yaml")
      html_path = build_part_b_viewer(cfg, out_dir=out, faces_dir=faces)
      assert html_path.exists()
      txt = html_path.read_text()
      assert "arcface" in txt and "face_0000" in txt and "cdn.plot.ly" in txt
  ```

- [ ] **Step 2: Run to verify it fails** — `./.venv/bin/pytest tests/part_b/test_viewer.py -v` → FAIL.

- [ ] **Step 3: Create `src/part_b/viewer.py`**
  ```python
  """Build the Part B interactive HTML viewer from cached embeddings (decoupled from encoding).

  For each outputs/part_b/<encoder>.npy it recomputes UMAP + KMeans + internal metrics, uses
  downscaled face images as HOVER thumbnails (too many points for always-visible), and shows
  age/gender/pose in the hover from <encoder>_attributes.json. Writes outputs/part_b/viewer.html.
  """
  from __future__ import annotations

  import json
  import logging
  from pathlib import Path

  from src.config import Config
  from src.core import metrics as M
  from src.core.cluster import cluster
  from src.core.embedding_store import load_embeddings
  from src.core.html_viewer import build_viewer_html, image_to_data_uri
  from src.core.reduce import preprocess, umap_2d

  log = logging.getLogger(__name__)


  def build_part_b_viewer(cfg: Config, out_dir: str | Path, faces_dir: str | Path) -> Path:
      """Assemble outputs/part_b/viewer.html from every <encoder>.npy in out_dir."""
      out_dir, faces_dir = Path(out_dir), Path(faces_dir)
      npys = sorted(p for p in out_dir.glob("*.npy"))
      if not npys:
          raise FileNotFoundError(f"no embeddings (*.npy) in {out_dir}")
      ids = None
      projections: dict[str, dict] = {}
      hover_meta: dict[str, dict] = {}
      um = cfg.reduce.umap
      algo = cfg.part_b.clustering.algorithms[0]
      for npy in npys:
          emb = load_embeddings(npy.stem, out_dir)
          if ids is None:
              ids = emb.ids
          elif emb.ids != ids:
              raise ValueError(f"id mismatch for {npy.stem}; viewer needs a shared id order")
          X = preprocess(emb.vectors, list(cfg.reduce.preprocess), pca_components=cfg.reduce.pca_components)
          coords = umap_2d(X, um.n_neighbors, um.min_dist, um.metric, cfg.seed)
          res = cluster(X, algo, cfg.part_b.clustering.k_min, cfg.part_b.clustering.k_max, cfg.seed)
          projections[npy.stem] = {"coords2d": coords, "labels": res.labels,
                                   "metrics": M.internal_metrics(X, res.labels)}
          attr_file = out_dir / f"{npy.stem}_attributes.json"
          if attr_file.exists() and not hover_meta:
              raw = json.loads(attr_file.read_text())
              hover_meta = {i: {"age": f"{a['age']:.0f}", "gender": a["gender"],
                                "pose_yaw": f"{a['pose_yaw']:.0f}"} for i, a in raw.items()}
      thumbs = [image_to_data_uri(faces_dir / f"{i}.jpg", max_px=96) for i in ids]
      html = build_viewer_html(
          projections, ids, thumbs, hover_meta=hover_meta or None,
          title="Part B — Faces: attribute clusters",
          intro=("Each point is one generated face, coloured by its KMeans cluster. Hover a "
                 "point to see the face plus predicted age / gender / pose. Buttons switch the "
                 "embedding model."),
          always_show_thumbs=False, page_title="Part B — Face Cluster Viewer")
      out_html = out_dir / "viewer.html"
      out_html.write_text(html)
      log.info("Wrote %s", out_html)
      return out_html
  ```

- [ ] **Step 4: Run to verify it passes** — `./.venv/bin/pytest tests/part_b/test_viewer.py -v` → PASS.

- [ ] **Step 5: Wire CLI (additive) in `main.py` `_run_part_b`:**
  ```python
      if stage in ("viewer", "all"):
          from src.part_b.viewer import build_part_b_viewer
          build_part_b_viewer(cfg, out, data_dir)
  ```
  (`out` and `data_dir` already in scope.)

- [ ] **Step 6: Run** — `./.venv/bin/pytest -q` → all pass.

- [ ] **Step 7: Commit**
  ```bash
  git add src/part_b/viewer.py tests/part_b/test_viewer.py main.py
  git commit -m "feat: Part B interactive viewer (hover face + age/gender/pose) + CLI"
  ```

---

## Task P1.8: Generate viewers on the box from current results; pull + document

**Files:** none (artifacts → `reports/`); Modify `README.md` (additive Viewers section), `DEFINITIONS.md`/`PLAN.md` logs.

- [ ] **Step 1: Sync code to elem-danit1** (rsync, excludes as before) and install plotly there:
  `ssh elem-danit1 'bash -lc "cd /mnt/workspace/projects/embedding_clustering && ./.venv/bin/pip install -q plotly==5.22.0"'`

- [ ] **Step 2: Regenerate Part B attributes + both viewers + verbose montages on the box:**
  ```bash
  ssh elem-danit1 'bash -lc "cd /mnt/workspace/projects/embedding_clustering && \
    ./.venv/bin/python main.py part-b extract && \
    ./.venv/bin/python main.py part-a viewer && \
    ./.venv/bin/python main.py part-b viewer"'
  ```
  (`part-b extract` re-runs to write `arcface_attributes.json` + titled montage; `part-a viewer`/`part-b viewer` build the HTML from cached `.npy`.)

- [ ] **Step 3: Pull artifacts back** into `reports/`:
  ```bash
  scp 'elem-danit1:/mnt/workspace/projects/embedding_clustering/outputs/part_a/viewer.html' reports/part_a/
  scp 'elem-danit1:/mnt/workspace/projects/embedding_clustering/outputs/part_b/viewer.html' reports/part_b/
  scp 'elem-danit1:/mnt/workspace/projects/embedding_clustering/outputs/part_b/arcface_attributes.json' reports/part_b/
  scp 'elem-danit1:/mnt/workspace/projects/embedding_clustering/outputs/part_*/figures/*_clusters_montage.png' /tmp/ # then copy into reports/part_a, reports/part_b
  ```
  Open `reports/part_a/viewer.html` and `reports/part_b/viewer.html` locally; confirm toggles, hover thumbnails, and (Part A) always-visible cards work.

- [ ] **Step 4: Add a "Viewers" section to `README.md`** (additive): how to open the HTML, what the toggles/hover show, that they're self-contained.

- [ ] **Step 5: Commit**
  ```bash
  git add reports/ README.md DEFINITIONS.md PLAN.md
  git commit -m "docs: interactive viewers (Part A cards, Part B hover) + verbose montages from current results"
  ```

---

## Phase 2 (planned later — outline only, do NOT implement in Phase 1)

Add optional encoders, each = one extractor module + a config registry entry; re-encode on
the box; rerun the `viewer` stages (they pick up the new `.npy` automatically as extra toggles):
- `src/part_a/extractors/clip.py` (CLIP on renders) + `part_a.encoders_2d += clip`.
- `src/part_b/extractors/dinov2_generic.py` (DINOv2 on faces); make Part B loop encoders +
  `part_b.encoders` config list.
- `src/part_a/extractors/pe_core.py` (Perception Encoder; setup documented).
- OpenShape/ULIP-2 (Part A 3D) best-effort pure-torch CPU port; documented skip if disproportionate.
A separate detailed plan will be written for Phase 2 before implementing it.

---

## Self-Review

**Spec coverage (Phase 1 items):** html_viewer module → P1.1/P1.2; viewer stage consuming
`.npy` → P1.6/P1.7; attribute persistence → P1.3; verbose PNGs → P1.4/P1.5; generate from
current results + README → P1.8; plotly dep → P1.0. Phase 2 (optional encoders) explicitly
deferred. ✓

**Placeholder scan:** No TODO/TBD/"implement later" in steps; all code blocks complete. The
Phase 2 outline is intentionally high-level and marked do-not-implement. ✓

**Type consistency:** `build_viewer_html(projections, ids, thumbs, hover_meta, *, title,
intro, always_show_thumbs, page_title)` and `image_to_data_uri(path, max_px)` are used
identically in P1.6/P1.7. `projections[name] = {"coords2d","labels","metrics"}` shape matches
`_figure_json`/`_metrics_table` consumers. `cluster_montage(..., row_titles=, caption=)`
matches the P1.5 callers. Viewer builders reuse existing `preprocess`/`umap_2d`/`cluster`/
`internal_metrics`/`load_embeddings`/`sanitize_id` signatures. ✓
