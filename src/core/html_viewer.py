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

import numpy as np


def image_to_data_uri(path: str | Path, max_px: int = 96, fmt: str = "png") -> str:
    """Downscale an image so its long side is <= max_px; return a base64 data URI.

    fmt "png" preserves transparency (use for the glasses renders); "jpeg" is far smaller for
    photos (use for the face thumbnails — keeps the 500-point viewer small). Returns "" if the
    file is missing (the viewer then renders that point thumbnail-less).
    """
    from PIL import Image

    p = Path(path)
    if not p.exists():
        return ""
    is_jpeg = fmt.lower() in ("jpeg", "jpg")
    img = Image.open(p).convert("RGB" if is_jpeg else "RGBA")
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG" if is_jpeg else "PNG", **({"quality": 80} if is_jpeg else {}))
    mime = "jpeg" if is_jpeg else "png"
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode()


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


def _figure_json(proj: dict, ids, thumbs, hover_meta, always_show_thumbs: bool,
                 thumb_scale: float = 1.0, hover_thumbs=None) -> str:
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
        # hover image may differ from the on-plot image (e.g. coloured hover, grey on plot)
        hth = hover_thumbs if hover_thumbs is not None else thumbs
        cd = [[ids[j], int(c), hth[j], meta_str(ids[j])] for j in idx]
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
        span = float(max(np.ptp(xs), np.ptp(ys))) or 1.0
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
            isz = s * thumb_scale   # enlarge the rendered image without growing the card
            images.append(dict(source=thumbs[j], xref="x", yref="y", x=xi, y=yi,
                               sizex=isz, sizey=isz, xanchor="center", yanchor="middle",
                               sizing="contain", layer="above"))
    fig.update_layout(width=840, height=650, plot_bgcolor="#f8f8f8",
                      xaxis=dict(title="UMAP 1", zeroline=False),
                      yaxis=dict(title="UMAP 2", zeroline=False),
                      legend=dict(itemsizing="constant", orientation="v", x=1.02, y=1),
                      margin=dict(l=50, r=130, t=10, b=45),
                      images=images, shapes=shapes)
    return fig.to_json().replace("\\u002f", "/")


def make_hist_spec(title: str, xlabel: str, named_arrays, n_bins: int = 30,
                   density: bool = True) -> dict:
    """Pre-bin several distance arrays onto a shared axis → a hist spec for the viewer.

    named_arrays: list of (name, color, values). Binning here (not client-side) keeps the HTML
    small even for Part B's ~125k pairs. density=True normalises each series so its area = 1
    (matching the static feature_distributions.png), making two differently-sized groups
    comparable in shape; density=False shows raw pair counts.
    """
    arrs = [(n, c, np.asarray(v, dtype=float)) for n, c, v in named_arrays]
    allv = np.concatenate([v for _, _, v in arrs if len(v)]) if arrs else np.array([0.0])
    edges = np.linspace(float(allv.min()), float(allv.max()) or 1.0, n_bins + 1)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    series = [{"name": n, "color": c, "x": centers,
               "y": (np.histogram(v, bins=edges, density=density)[0].tolist()
                     if len(v) else [0] * n_bins),
               "mean": (float(v.mean()) if len(v) else None)}
              for n, c, v in arrs]
    return {"title": title, "xlabel": xlabel,
            "ylabel": "density (area = 1)" if density else "pair count", "series": series}


def _hist_figure_json(spec: dict) -> str:
    """One Plotly figure (JSON) for a pre-binned feature-distance histogram. `spec` =
    {title, xlabel, series:[{name,color,x,y}]} where x=bin centres, y=counts."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for s in spec["series"]:
        fig.add_trace(go.Bar(x=s["x"], y=s["y"], name=s["name"],
                             marker_color=s["color"], opacity=0.6))
    # dashed vertical line at each series' mean (matching feature_distributions.png)
    shapes, annos = [], []
    for s in spec["series"]:
        if s.get("mean") is None:
            continue
        shapes.append(dict(type="line", xref="x", yref="paper", x0=s["mean"], x1=s["mean"],
                           y0=0, y1=1, line=dict(color=s["color"], width=2, dash="dash")))
        annos.append(dict(x=s["mean"], y=1.02, xref="x", yref="paper", showarrow=False,
                          text=f"mean {s['mean']:.2f}", font=dict(size=10, color=s["color"])))
    fig.update_layout(width=560, height=330, barmode="overlay",
                      title=dict(text=spec["title"], font=dict(size=12)),
                      xaxis=dict(title=spec["xlabel"], zeroline=False),
                      yaxis=dict(title=spec.get("ylabel", "density"), zeroline=False),
                      bargap=0, plot_bgcolor="#f8f8f8", shapes=shapes, annotations=annos,
                      legend=dict(orientation="h", y=-0.18, x=0),
                      margin=dict(l=55, r=20, t=42, b=70))
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
                      always_show_thumbs: bool, thumb_scale: float = 1.0,
                      hover_thumbs: list[str] | None = None, extra_html: str = "",
                      hist: dict[str, list[dict]] | None = None,
                      page_title: str = "Embedding Cluster Viewer") -> str:
    """Render the full self-contained explorer HTML.

    projections: {encoder_name: {"coords2d": (n,2), "labels": (n,), "metrics": {..}}}.
    ids/thumbs: length-n, aligned to every projection's row order.
    hover_meta: optional {id: {field: value}} shown in the hover tooltip.
    always_show_thumbs: True places thumbnails on the plot (small sets); False = hover-only.
    thumb_scale: multiplier on the always-visible thumbnail image size (card stays fixed).
    extra_html: optional HTML block injected below the metrics table (e.g. a k-selection
    comparison); empty by default.
    hist: optional {encoder_name: [hist_spec, ...]} — one or more pre-binned feature-distance
    histograms shown in a column beside each scatter and switched together with it (see
    _hist_figure_json for the spec).
    """
    names = list(projections)
    specs = {n: _figure_json(projections[n], ids, thumbs, hover_meta, always_show_thumbs,
                             thumb_scale, hover_thumbs)
             for n in names}
    specs_js = ",\n".join(f"'{n}': {s}" for n, s in specs.items())
    keys_js = ", ".join(f"'{n}'" for n in names)
    def _hist_list(specs):
        return "[" + ",".join(_hist_figure_json(s) for s in specs) + "]"
    hist_js = ",\n".join(f"'{n}': {_hist_list(hist[n])}" for n in names if hist and hist.get(n))
    btns = "".join(
        f'<button id="b_{n}" onclick="show(\'{n}\')" class="tb{ " act" if i==0 else "" }">{n}</button>'
        for i, n in enumerate(names))

    def _view_div(i, n):
        inline = "" if i == 0 else " style=display:none"
        specs_h = (hist or {}).get(n) or []
        hcol = "".join(f'<div id="h_{n}_{j}" class="card"></div>' for j in range(len(specs_h)))
        hcol = f'<div class="hcol">{hcol}</div>' if specs_h else ""
        return (f'<div id="view_{n}" class="view"{inline}>'
                f'<div id="v_{n}" class="card"></div>{hcol}</div>')
    divs = "".join(_view_div(i, n) for i, n in enumerate(names))
    table = _metrics_table(projections)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>{page_title}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>body{{font-family:sans-serif;margin:0;padding:14px 22px;color:#222}}
h2{{margin:0 0 4px}} p{{color:#555;font-size:13px;margin:2px 0 8px;max-width:1180px;line-height:1.5}}
.tb{{padding:7px 15px;margin-right:7px;border:1px solid #aaa;border-radius:4px;cursor:pointer;background:#f0f0f0;font-size:13px}}
.tb.act{{background:#2c3e50;color:#fff;border-color:#2c3e50}}
table.m{{border-collapse:collapse;font-size:12px;margin:6px 0 10px}}
table.m th,table.m td{{border:1px solid #ddd;padding:4px 9px;text-align:right}}
table.m th{{background:#0f3460;color:#fff}} table.m td:first-child,table.m th:first-child{{text-align:left}}
table.m td.win{{background:#cdebcd;font-weight:700}}
.bar{{margin:10px 0 6px}}
.view{{display:flex;flex-wrap:wrap;gap:20px;align-items:flex-start;margin-top:6px}}
.hcol{{display:flex;flex-direction:column;gap:18px}}
.card{{box-shadow:0 1px 4px rgba(0,0,0,.12);border-radius:6px;background:#fff;padding:6px}}</style></head>
<body><h2>{title}</h2><p>{intro}</p>
<p style="margin-top:0"><b>Clustering quality per encoder</b> (silhouette ↑, Davies–Bouldin ↓,
Calinski–Harabasz ↑; green = best):</p>{table}
{extra_html}
<div class="bar">{btns}</div>{divs}
<div id="tip" style="position:fixed;display:none;z-index:9;background:#fff;border:1px solid #ccc;
border-radius:6px;padding:7px;box-shadow:2px 4px 14px rgba(0,0,0,.25);font:12px/1.4 sans-serif;
text-align:center;max-width:300px"></div>
<script>
var S={{ {specs_js} }}, H={{ {hist_js} }}, K=[{keys_js}], R={{}};
function tip(el){{var t=document.getElementById('tip');
  el.on('plotly_hover',function(e){{var d=e.points[0].customdata;var h='';
    if(d[2])h+='<img src="'+d[2]+'" style="max-width:280px;max-height:280px;display:block;margin:0 auto 4px">';
    h+='<b>'+d[0]+'</b><br>cluster '+d[1]+(d[3]?'<br>'+d[3]:'');t.innerHTML=h;
    var b=e.points[0].bbox||{{}};t.style.left=((b.x1||0)+12)+'px';t.style.top=((b.y0||0)-8)+'px';t.style.display='block';}});
  el.on('plotly_unhover',function(){{t.style.display='none';}});}}
function show(n){{K.forEach(function(k){{var d=document.getElementById('view_'+k),b=document.getElementById('b_'+k);
  d.style.display=(k===n?'':'none');b.className='tb'+(k===n?' act':'');}});
  if(!R[n]){{var sv=document.getElementById('v_'+n);Plotly.newPlot(sv,S[n].data,S[n].layout,{{responsive:false,displayModeBar:false}});
    if(H[n]){{H[n].forEach(function(hf,i){{Plotly.newPlot(document.getElementById('h_'+n+'_'+i),hf.data,hf.layout,{{responsive:false,displayModeBar:false}});}});}}
    R[n]=1;setTimeout(function(){{tip(sv);}},400);}}}}
show(K[0]);
</script></body></html>"""
