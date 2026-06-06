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
