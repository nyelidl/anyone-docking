#!/usr/bin/env python3
"""
app.py — Streamlit UI layer for Anyone Can Dock.
All computation is delegated to core.py — this file contains only
layout, widgets, session state, and 3D/2D visualisation.
"""

import io
import os
import tempfile
import zipfile
import re as _re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core import (
    prepare_receptor,
    prepare_ligand,
    smiles_from_file,
    run_vina,
    get_vina_binary,
    check_obabel,
    fix_sdf_bond_orders,
    load_mols_from_sdf,
    write_single_pose,
    get_interacting_residues,
    calc_rmsd_heavy,
    call_poseview_v1,
    svg_to_png,
    stamp_png,
    is_cif_file,
    convert_cif_to_pdb,
)

# Graceful fallbacks for functions added in newer core.py versions
try:
    from core import call_poseview2_ref
except ImportError:
    def call_poseview2_ref(pdb_code, ligand_id):
        return None, "call_poseview2_ref not available — please update core.py"

try:
    from core import warm_poseview_cache, clear_poseview_cache
except ImportError:
    def warm_poseview_cache(path): return False, "core.py not updated yet"
    def clear_poseview_cache(): pass

try:
    from core import (
        draw_interactions_rdkit,
        draw_interaction_diagram,
        draw_interactions_rdkit_classic,
        draw_interaction_diagram_data,
    )
except ImportError:
    draw_interactions_rdkit = None
    draw_interaction_diagram = None
    draw_interactions_rdkit_classic = None

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="anyone can dock",
    page_icon="https://raw.githubusercontent.com/nyelidl/anyone-docking/main/any-L.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  THEME + COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

import json as _json

def _render_interactive_diagram(data: dict, height: int = 800) -> str:
    W       = data["W"]
    H       = data["H"]
    title   = data["title"]
    lig_svg = data["ligand_svg"]
    placements = data["placements"]

    # Escape title for JS
    title_esc = title.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")

    # Title pill dimensions
    tw = min(len(title) * 14 + 48, W - 40)
    pill_x = (W - tw) / 2

    TYPE_CFG = {
        "hbond":            {"fill": "#80dd80", "stroke": "#1a7a1a", "lineclr": "#1a7a1a", "dash": "5 3",       "lw": "1.6"},
        "hbond_to_halogen": {"fill": "#c4a0ff", "stroke": "#6633aa", "lineclr": "#6633aa", "dash": "4 2 1 2",   "lw": "1.6"},
        "pi_pi":            {"fill": "#f0a0ff", "stroke": "#e200e8", "lineclr": "#e200e8", "dash": "5 3",       "lw": "1.6"},
        "cation_pi":        {"fill": "#f0a0ff", "stroke": "#e200e8", "lineclr": "#e200e8", "dash": "5 3",       "lw": "1.6"},
        "hydrophobic":      {"fill": "#a0c8ff", "stroke": "#2287ff", "lineclr": None,      "dash": "",          "lw": "0"},
        "ionic":            {"fill": "#ffb0d0", "stroke": "#cc2277", "lineclr": "#cc2277", "dash": "6 2 2 2",   "lw": "1.8"},
        "metal":            {"fill": "#ffe090", "stroke": "#cc8800", "lineclr": "#cc8800", "dash": "3 2",       "lw": "1.8"},
        "halogen":          {"fill": "#ffb0d0", "stroke": "#cc2277", "lineclr": "#cc2277", "dash": "5 2",       "lw": "1.6"},
    }

    # Legend items (only active types)
    active_types = list(dict.fromkeys(p["itype"] for p in placements))
    LEG_LABEL = {
        "hbond": "H-bond", "hbond_to_halogen": "H···Halogen",
        "pi_pi": "π-π", "cation_pi": "Cation-π",
        "hydrophobic": "Hydrophobic", "ionic": "Ionic",
        "metal": "Metal", "halogen": "Halogen",
    }
    legend_items = []
    for it in active_types:
        cfg = TYPE_CFG.get(it, TYPE_CFG["hbond"])
        legend_items.append({
            "label":   LEG_LABEL.get(it, it),
            "fill":    cfg["fill"],
            "stroke":  cfg["stroke"],
            "lineclr": cfg["lineclr"],
            "dash":    cfg["dash"],
        })

    # Legend SVG (static, at bottom)
    LEG_Y = H - 45
    leg_entry_w = 115
    leg_total = len(legend_items) * leg_entry_w
    leg_x0 = (W - leg_total) / 2
    leg_parts = [
        f'<rect x="{leg_x0-8:.0f}" y="{LEG_Y-5}" width="{leg_total+16:.0f}" height="40"'
        f' fill="white" stroke="#e0e0e0" stroke-width="0.8" rx="6"/>'
    ]
    for k, li in enumerate(legend_items):
        ix = leg_x0 + k * leg_entry_w + 14
        leg_parts.append(
            f'<circle cx="{ix:.0f}" cy="{LEG_Y+12}" r="8"'
            f' fill="{li["fill"]}" opacity="0.5" stroke="{li["stroke"]}" stroke-width="1"/>'
        )
        if li["lineclr"]:
            leg_parts.append(
                f'<line x1="{ix+10:.0f}" y1="{LEG_Y+12}" x2="{ix+26:.0f}" y2="{LEG_Y+12}"'
                f' stroke="{li["lineclr"]}" stroke-width="1.8" stroke-dasharray="{li["dash"]}"/>'
            )
            leg_parts.append(
                f'<text x="{ix+30:.0f}" y="{LEG_Y+12}" dominant-baseline="central"'
                f' font-family="Arial,sans-serif" font-size="12" font-weight="700"'
                f' fill="#555">{li["label"]}</text>'
            )
        else:
            leg_parts.append(
                f'<text x="{ix+12:.0f}" y="{LEG_Y+12}" dominant-baseline="central"'
                f' font-family="Arial,sans-serif" font-size="12" font-weight="700"'
                f' fill="#555">{li["label"]}</text>'
            )
    legend_svg = "\n".join(leg_parts)

    placements_json = _json.dumps(placements)
    type_cfg_json   = _json.dumps(TYPE_CFG)

    html = f"""
<div style="font-family:Arial,sans-serif;background:white;border-radius:8px;
            border:1px solid #e0e0e0;overflow:hidden;">

  <!-- Toolbar -->
  <div style="display:flex;align-items:center;gap:8px;padding:8px 12px;
              border-bottom:1px solid #eee;flex-wrap:wrap;background:#fafafa;">
    <span style="font-size:12px;color:#555;flex:1;">
      🧬 <strong>Interactive mode</strong> — drag any residue label to reposition it
    </span>
    <button onclick="resetLayout()"
      style="font-size:12px;padding:4px 10px;border:1px solid #ccc;
             border-radius:4px;background:#f8f8f8;cursor:pointer;">
      ↺ Reset
    </button>
    <button onclick="exportSVG()"
      style="font-size:12px;padding:4px 10px;border:1px solid #ccc;
             border-radius:4px;background:#f8f8f8;cursor:pointer;">
      ⬇ SVG
    </button>
    <button onclick="exportPNG()"
      style="font-size:12px;padding:4px 10px;border:1px solid #4a90d9;
             border-radius:4px;background:#e8f4ff;color:#1a5fa8;cursor:pointer;font-weight:700;">
      ⬇ PNG
    </button>
    <select id="iac-dpi"
      style="font-size:12px;padding:4px 6px;border:1px solid #ccc;
             border-radius:4px;background:#fff;cursor:pointer;"
      title="PNG export resolution">
      <option value="1">Screen (1×)</option>
      <option value="2" selected>150 dpi (2×)</option>
      <option value="3">300 dpi (3×)</option>
      <option value="4">600 dpi (4×)</option>
    </select>
  </div>

  <!-- SVG canvas -->
  <svg id="iac-svg" viewBox="0 0 {W} {H}"
       style="width:100%;display:block;cursor:default;user-select:none;">

    <rect width="{W}" height="{H}" fill="white"/>

    <!-- Title pill -->
    <rect x="{pill_x:.1f}" y="12" width="{tw:.0f}" height="44"
          rx="22" ry="22" fill="#f2f2f2" stroke="none"/>
    <text x="{W/2:.1f}" y="34" text-anchor="middle" dominant-baseline="central"
          font-family="Arial,sans-serif" font-size="20" font-weight="700"
          fill="#1a1a1a">{title}</text>

    <!-- Interaction lines (updated by JS) -->
    <g id="iac-lines"></g>

    <!-- Ligand structure (static) -->
    <g id="iac-ligand">{lig_svg}</g>

    <!-- Residue circles (draggable, added by JS) -->
    <g id="iac-residues"></g>

    <!-- Legend (static) -->
    <g id="iac-legend">{legend_svg}</g>

  </svg>
</div>

<script>
(function() {{
  const PLACEMENTS = {placements_json};
  const TYPE_CFG   = {type_cfg_json};

  const svg      = document.getElementById("iac-svg");
  const linesG   = document.getElementById("iac-lines");
  const residuesG= document.getElementById("iac-residues");

  // Current positions (mutable)
  const pos = {{}};
  PLACEMENTS.forEach(p => {{ pos[p.id] = {{ x: p.bx, y: p.by }}; }});

  // DOM element caches
  const els = {{}};

  function SVG(tag, attrs) {{
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  }}

  function toSVGCoords(clientX, clientY) {{
    const rect = svg.getBoundingClientRect();
    const vb   = svg.viewBox.baseVal;
    return {{
      x: (clientX - rect.left) * (vb.width  / rect.width)  + vb.x,
      y: (clientY - rect.top)  * (vb.height / rect.height) + vb.y,
    }};
  }}

  function buildAll() {{
    linesG.innerHTML = "";
    residuesG.innerHTML = "";

    PLACEMENTS.forEach(p => {{
      const cfg = TYPE_CFG[p.itype] || TYPE_CFG["hbond"];
      const cache = {{ line: null, distRect: null, distTxt: null,
                       circ: null, txt: null }};
      els[p.id] = cache;

      // — LINE (non-hydrophobic only) —
      if (cfg.lineclr) {{
        const line = SVG("line", {{
          x1: p.lx, y1: p.ly,
          x2: pos[p.id].x, y2: pos[p.id].y,
          stroke: cfg.lineclr,
          "stroke-width": cfg.lw,
          "stroke-dasharray": cfg.dash,
          opacity: "0.85",
        }});
        linesG.appendChild(line);
        cache.line = line;

        // Distance label
        if (p.distance != null) {{
          const ds  = p.distance + "\u00c5";
          const tw2 = ds.length * 7 + 8;
          const dr  = SVG("rect", {{
            width: tw2, height: 17, rx: 4,
            fill: "white", stroke: cfg.lineclr, "stroke-width": "0.5",
          }});
          const dt = SVG("text", {{
            "text-anchor": "middle", "dominant-baseline": "central",
            "font-family": "Arial,sans-serif", "font-size": "14",
            "font-weight": "700", fill: cfg.lineclr,
          }});
          dt.textContent = ds;
          linesG.appendChild(dr);
          linesG.appendChild(dt);
          cache.distRect = dr; cache.distTxt = dt; cache.tw2 = tw2;
        }}
      }}

      // — RESIDUE CIRCLE + LABEL —
      const g = SVG("g", {{ style: "cursor:grab;" }});

      const circ = SVG("circle", {{
        cx: pos[p.id].x, cy: pos[p.id].y,
        r: "24.55",
        fill: cfg.fill, opacity: "0.5",
        stroke: cfg.stroke, "stroke-width": "1.5",
      }});
      g.appendChild(circ);
      cache.circ = circ;

      const txt = SVG("text", {{
        x: pos[p.id].x, y: pos[p.id].y,
        "text-anchor": "middle", "dominant-baseline": "central",
        "font-family": "Arial,sans-serif",
        "font-size": "13", "font-weight": "700",
        fill: cfg.stroke,
      }});
      txt.textContent = p.label;
      g.appendChild(txt);
      cache.txt = txt;

      residuesG.appendChild(g);
      makeDraggable(g, p.id);
      updateElement(p.id);  // position distance label
    }});
  }}

  function updateElement(id) {{
    const p   = PLACEMENTS.find(d => d.id === id);
    const cfg = TYPE_CFG[p.itype] || TYPE_CFG["hbond"];
    const {{ x, y }} = pos[id];
    const cache = els[id];

    if (cache.line) {{
      cache.line.setAttribute("x2", x);
      cache.line.setAttribute("y2", y);
    }}
    if (cache.circ) {{ cache.circ.setAttribute("cx", x); cache.circ.setAttribute("cy", y); }}
    if (cache.txt)  {{ cache.txt.setAttribute("x", x);   cache.txt.setAttribute("y", y);  }}

    if (cache.distRect && cache.distTxt) {{
      const t  = 0.4;
      const mx = p.lx + (x - p.lx) * t;
      const my = p.ly + (y - p.ly) * t;
      const dx = x - p.lx, dy = y - p.ly;
      const len = Math.sqrt(dx*dx + dy*dy) + 0.001;
      const px  = -dy/len * 14, py = dx/len * 14;
      const lx  = mx + px, ly = my + py;
      const tw2 = cache.tw2;
      cache.distRect.setAttribute("x",  lx - tw2/2);
      cache.distRect.setAttribute("y",  ly - 8);
      cache.distTxt.setAttribute("x",  lx);
      cache.distTxt.setAttribute("y",  ly);
    }}
  }}

  function makeDraggable(el, id) {{
    let dragging = false, startMouse = null, startPos = null;

    function onStart(clientX, clientY) {{
      dragging   = true;
      startMouse = toSVGCoords(clientX, clientY);
      startPos   = {{ ...pos[id] }};
      el.style.cursor = "grabbing";
    }}
    function onMove(clientX, clientY) {{
      if (!dragging) return;
      const cur  = toSVGCoords(clientX, clientY);
      pos[id].x  = startPos.x + cur.x - startMouse.x;
      pos[id].y  = startPos.y + cur.y - startMouse.y;
      updateElement(id);
    }}
    function onEnd() {{
      dragging = false;
      el.style.cursor = "grab";
    }}

    el.addEventListener("mousedown",  e => {{ onStart(e.clientX, e.clientY); e.preventDefault(); }});
    el.addEventListener("touchstart", e => {{ onStart(e.touches[0].clientX, e.touches[0].clientY); e.preventDefault(); }}, {{passive:false}});

    window.addEventListener("mousemove",  e => onMove(e.clientX, e.clientY));
    window.addEventListener("touchmove",  e => {{ if(dragging) {{ onMove(e.touches[0].clientX, e.touches[0].clientY); e.preventDefault(); }} }}, {{passive:false}});
    window.addEventListener("mouseup",  onEnd);
    window.addEventListener("touchend", onEnd);
  }}

  window.resetLayout = function() {{
    PLACEMENTS.forEach(p => {{ pos[p.id] = {{ x: p.bx, y: p.by }}; }});
    PLACEMENTS.forEach(p => updateElement(p.id));
    // rebuild circles at reset positions
    buildAll();
  }};

  window.exportSVG = function() {{
    const clone = svg.cloneNode(true);
    clone.setAttribute("xmlns","http://www.w3.org/2000/svg");
    const blob = new Blob([clone.outerHTML], {{type:"image/svg+xml"}});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "interaction_diagram.svg";
    a.click();
  }};

  window.exportPNG = function() {{
    const scale = parseInt(document.getElementById("iac-dpi").value) || 2;
    const W = {W}, H = {H};
    const clone = svg.cloneNode(true);
    clone.setAttribute("xmlns","http://www.w3.org/2000/svg");
    clone.setAttribute("width", W);
    clone.setAttribute("height", H);
    clone.removeAttribute("style");
    const svgStr = new XMLSerializer().serializeToString(clone);
    const svgBlob = new Blob([svgStr], {{type:"image/svg+xml;charset=utf-8"}});
    const url = URL.createObjectURL(svgBlob);
    const img = new Image();
    img.onload = function() {{
      const canvas = document.createElement("canvas");
      canvas.width  = W * scale;
      canvas.height = H * scale;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0, W, H);
      URL.revokeObjectURL(url);
      const a = document.createElement("a");
      a.download = "interaction_diagram.png";
      a.href = canvas.toDataURL("image/png");
      a.click();
    }};
    img.onerror = function() {{
      URL.revokeObjectURL(url);
      alert("PNG export failed — use SVG export instead.");
    }};
    img.src = url;
  }};

  buildAll();
}})();
</script>
"""
    return html

def _chart_colors():
    theme = st.get_option("theme.base") if hasattr(st, "get_option") else "light"
    dark  = (theme == "dark")
    return {
        "bg":        "#0d1117" if dark else "#FFFFFF",
        "bg_sub":    "#161b22" if dark else "#F6F8FA",
        "border":    "#30363d" if dark else "#D0D7DE",
        "text":      "#c9d1d9" if dark else "#24292F",
        "muted":     "#8b949e" if dark else "#57606A",
        "legend_bg": "#21262d" if dark else "#F6F8FA",
    }

def _viewer_bg():
    return _chart_colors()["bg"]

def _png_to_b64_img(png_bytes, style="width:100%;height:auto;display:block;border-radius:6px;"):
    import base64
    b64 = base64.b64encode(png_bytes).decode()
    st.markdown(
        f'<img src="data:image/png;base64,{b64}" style="{style}">',
        unsafe_allow_html=True,
    )

def _pill(text, kind="info"):
    cls = {
        "info":    "result-pill",
        "success": "success-pill",
        "warn":    "warn-pill",
    }.get(kind, "result-pill")
    return f'<span class="{cls}">{text}</span>'


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
:root {
    --bg:#FFFFFF; --bg-subtle:#F6F8FA; --bg-card:#F0F4F8; --bg-input:#FFFFFF;
    --border:#D0D7DE; --border-input:#D0D7DE;
    --text:#24292F; --text-muted:#57606A;
    --accent:#0969DA; --accent2:#0550AE; --success:#1A7F37; --warn:#9A6700;
    --text-card-title:#57606A; --text-card-heading:#24292F; --text-input:#24292F;
    --pill-bg:#DDF4FF; --pill-border:#54AEFF; --pill-text:#0550AE;
    --ok-bg:#DAFBE1; --ok-border:#1A7F37;
    --wn-bg:#FFF8C5; --wn-border:#9A6700; --btn-sec-bg:#F6F8FA;
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg:#0d1117; --bg-subtle:#161b22; --bg-card:#161b22; --bg-input:#21262d;
        --border:#30363d; --border-input:#30363d;
        --text:#c9d1d9; --text-muted:#8b949e;
        --accent:#58a6ff; --accent2:#79c0ff; --success:#3fb950; --warn:#d29922;
        --text-card-title:#8b949e; --text-card-heading:#e6edf3; --text-input:#c9d1d9;
        --pill-border:#1f6feb; --pill-text:#79c0ff;
        --ok-bg:#23863622; --ok-border:#238636;
        --wn-bg:#9e680322; --wn-border:#9e6803; --btn-sec-bg:#21262d;
    }
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important; color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
}
[data-testid="stSidebar"] { background: var(--bg-subtle) !important; }
[data-testid="stHeader"]  { background: transparent !important; }
h1 { font-family: 'IBM Plex Mono', monospace; color: var(--accent); letter-spacing: -1px; }
h2, h3 { font-family: 'IBM Plex Mono', monospace; color: var(--accent2); }
.step-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-left: 4px solid var(--accent); border-radius: 8px;
    padding: 20px 24px; margin-bottom: 24px;
}
.step-card.done { border-left-color: var(--success); }
.step-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
    color: var(--text-card-title); text-transform: uppercase;
    letter-spacing: 2px; margin-bottom: 4px;
}
.step-heading {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem;
    color: var(--text-card-heading); margin-bottom: 16px;
}
.result-pill {
    display: inline-block; background: var(--pill-bg);
    border: 1px solid var(--pill-border); color: var(--pill-text);
    border-radius: 20px; padding: 2px 12px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; margin: 2px;
}
.success-pill {
    display: inline-block; background: var(--ok-bg);
    border: 1px solid var(--ok-border); color: var(--success);
    border-radius: 20px; padding: 4px 14px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
}
.warn-pill {
    display: inline-block; background: var(--wn-bg);
    border: 1px solid var(--wn-border); color: var(--warn);
    border-radius: 20px; padding: 4px 14px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
}
.log-box {
    background: var(--bg-subtle); border: 1px solid var(--border); border-radius: 6px;
    padding: 12px 16px; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; color: var(--text-muted);
    max-height: 220px; overflow-y: auto; white-space: pre-wrap;
}
.score-best {
    font-family: 'IBM Plex Mono', monospace; font-size: 2.4rem;
    color: var(--success); font-weight: 600;
}
.score-unit { font-size: 1rem; color: var(--text-muted); }
.stButton > button {
    background: var(--success); color: white; border: none; border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.88rem;
    padding: 8px 20px; transition: background 0.2s;
}
.stButton > button:hover { filter: brightness(1.15); }
.stButton > button[kind="secondary"] {
    background: var(--btn-sec-bg); border: 1px solid var(--border); color: var(--text);
}
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: var(--bg-input) !important; border: 1px solid var(--border-input) !important;
    color: var(--text-input) !important; border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-baseweb="slider"] { accent-color: var(--accent); }
.stDataFrame { border: 1px solid var(--border); border-radius: 6px; }
hr { border-color: var(--border); }
.step-divider { border: none; border-top: 1px dashed var(--border); margin: 32px 0; }
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--bg-subtle); border-bottom: 1px solid var(--border); gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem;
    color: var(--text-muted); background: transparent;
    border-radius: 6px 6px 0 0; padding: 10px 20px;
}
iframe { border: none !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = dict(
    workdir=None, ketcher_smi="",
    # Basic receptor
    pdb_token=None, receptor_fh=None, receptor_pdbqt=None,
    box_pdb=None, config_txt=None, cx=None, cy=None, cz=None,
    box_sx=16, box_sy=16, box_sz=16,
    ligand_pdb_path=None, receptor_done=False, receptor_log="",
    cocrystal_ligand_id="",
    # Basic ligand
    ligand_pdbqt=None, ligand_sdf=None, ligand_name="LIG",
    prot_smiles=None, ligand_done=False, ligand_log="",
    # Basic docking
    output_pdbqt=None, output_sdf=None, output_pv_sdf=None, dock_base=None,
    docking_done=False, docking_log="", score_df=None, pose_mols=None,
    # Basic redocking
    redock_done=False, redock_score=None, redock_result=None,
    confirmed_ref_score=None, confirmed_ref_pose=None, confirmed_ref_name=None,
    # Basic PoseView
    pv_image_png=None, pv_image_svg=None, pv_pose_key=None,
    pv_ref_png=None, pv_ref_svg=None,
    # Batch receptor
    b_pdb_token=None, b_receptor_fh=None, b_receptor_pdbqt=None,
    b_box_pdb=None, b_config_txt=None, b_cx=None, b_cy=None, b_cz=None,
    b_box_sx=16, b_box_sy=16, b_box_sz=16,
    b_ligand_pdb_path=None, b_receptor_done=False, b_receptor_log="",
    b_cocrystal_ligand_id="",
    # Batch results
    b_batch_done=False, b_batch_results=None, b_batch_log="",
    b_redock_score=None, b_redock_result=None,
    b_confirmed_ref_score=None, b_confirmed_ref_pose=None, b_confirmed_ref_name=None,
    # Batch PoseView
    b_pv2_image_png=None, b_pv2_image_svg=None, b_pv2_pose_key=None,
    b_pv2_ref_png=None, b_pv2_ref_svg=None,
    b_plot_png=None,
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.workdir is None:
    st.session_state.workdir = tempfile.mkdtemp(prefix="vina_")
WORKDIR       = Path(st.session_state.workdir)
BATCH_WORKDIR = WORKDIR / "batch"
BATCH_WORKDIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL CHECKS (cached — run once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⬇ Downloading AutoDock Vina 1.2.7…")
def _cached_vina():
    return get_vina_binary()

@st.cache_resource(show_spinner=False)
def _cached_obabel():
    return check_obabel()

VINA_PATH, _vina_err      = _cached_vina()
_OBABEL_OK, _OBABEL_VER   = _cached_obabel()


# ══════════════════════════════════════════════════════════════════════════════
#  3D VIEWER HELPER
# ══════════════════════════════════════════════════════════════════════════════

def show3d(view, height=480):
    try:
        from stmol import showmol
        showmol(view, height=height)
    except ImportError:
        raw  = view._make_html()
        resp = _re.sub(r'(width\s*[:=]\s*)["\']?\d+px?["\']?', r'\g<1>100%', raw)
        components.html(
            f'<div style="width:100%;overflow:hidden">{resp}</div>',
            height=height, scrolling=False,
        )

def _add_box_to_view(view, cx, cy, cz, sx, sy, sz):
    try:
        _c = {"x": float(cx), "y": float(cy), "z": float(cz)}
        _d = {"w": float(sx), "h": float(sy), "d": float(sz)}
        view.addBox({"center": _c, "dimensions": _d,
                     "color": "blue",  "opacity": 0.07, "wireframe": False})
        view.addBox({"center": _c, "dimensions": _d,
                     "color": "cyan",  "opacity": 0.90, "wireframe": True})
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  POSEVIEW LEGEND HTML
# ══════════════════════════════════════════════════════════════════════════════

_LEGEND_FULL = """
<div style="background:#fff;border:1px solid #D0D7DE;border-radius:6px;
     padding:12px 20px;font-family:'Helvetica Neue',Arial,sans-serif;font-size:13px;color:#333;margin-top:8px;">
  <div style="display:flex;align-items:center;gap:40px;margin-bottom:10px;">
    <div style="display:flex;align-items:center;gap:10px;white-space:nowrap;">
      <svg width="48" height="14"><line x1="0" y1="7" x2="48" y2="7" stroke="#5B9BD5" stroke-width="2" stroke-dasharray="5,3"/></svg>
      <span>hydrogen bond</span></div>
    <div style="display:flex;align-items:center;gap:10px;white-space:nowrap;">
      <svg width="48" height="14"><line x1="0" y1="7" x2="48" y2="7" stroke="#E85D8A" stroke-width="2" stroke-dasharray="5,3"/></svg>
      <span>ionic interaction</span></div>
    <div style="display:flex;align-items:center;gap:10px;white-space:nowrap;">
      <svg width="48" height="14"><line x1="0" y1="7" x2="48" y2="7" stroke="#F5C400" stroke-width="2" stroke-dasharray="5,3"/></svg>
      <span>metal interaction</span></div>
  </div>
  <div style="display:flex;align-items:center;gap:40px;">
    <div style="display:flex;align-items:center;gap:10px;white-space:nowrap;">
      <svg width="56" height="14"><circle cx="4" cy="7" r="4" fill="#44A44A"/>
        <line x1="8" y1="7" x2="48" y2="7" stroke="#AACC44" stroke-width="2" stroke-dasharray="5,3"/>
        <circle cx="52" cy="7" r="4" fill="#44A44A"/></svg>
      <span>cation-&#960;</span></div>
    <div style="display:flex;align-items:center;gap:10px;white-space:nowrap;">
      <svg width="56" height="14"><circle cx="5" cy="7" r="4" fill="#00BCD4"/>
        <line x1="9" y1="7" x2="47" y2="7" stroke="#00BCD4" stroke-width="2" stroke-dasharray="5,3"/>
        <circle cx="51" cy="7" r="4" fill="#00BCD4"/></svg>
      <span>&#960;-&#960;</span></div>
    <div style="display:flex;align-items:center;gap:10px;white-space:nowrap;">
      <svg width="48" height="14"><line x1="0" y1="7" x2="48" y2="7" stroke="#2E8B57" stroke-width="2.5"/></svg>
      <span>hydrophobic</span></div>
  </div>
</div>"""

_LEGEND_V1 = """
<div style="background:#fff;border:1px solid #D0D7DE;border-radius:6px;
     padding:12px 20px;font-family:'Helvetica Neue',Arial,sans-serif;font-size:13px;color:#333;margin-top:8px;">
  <div style="display:flex;align-items:center;gap:40px;">
    <div style="display:flex;align-items:center;gap:10px;white-space:nowrap;">
      <svg width="48" height="14"><line x1="0" y1="7" x2="48" y2="7" stroke="#000" stroke-width="2" stroke-dasharray="5,3"/></svg>
      <span>hydrogen bond</span></div>
    <div style="display:flex;align-items:center;gap:10px;white-space:nowrap;">
      <svg width="48" height="14"><line x1="0" y1="7" x2="48" y2="7" stroke="#2E8B57" stroke-width="2.5"/></svg>
      <span>hydrophobic</span></div>
  </div>
</div>"""


def _show_poseview_image(png_data, svg_data, caption, full_legend=False, stamp=""):
    legend = _LEGEND_FULL if full_legend else _LEGEND_V1
    if png_data:
        _display = stamp_png(png_data, stamp) if stamp else png_data
        _png_to_b64_img(_display)
        st.caption(caption)
        st.markdown(legend, unsafe_allow_html=True)
    elif svg_data:
        svg_str = svg_data.decode("utf-8") if isinstance(svg_data, bytes) else svg_data
        if stamp:
            _esc = (stamp
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))
            _svg_stamp = (
                '<g><foreignObject x="5%" y="92%" width="90%" height="40">'
                '<div xmlns="http://www.w3.org/1999/xhtml" '
                'style="display:flex;justify-content:center;align-items:center;height:100%;">'
                f'<span style="background:#E8E8E8;color:#1A1A1A;font-family:sans-serif;'
                f'font-size:15px;padding:6px 28px;border-radius:999px;white-space:nowrap;">'
                f'{_esc}</span>'
                '</div></foreignObject></g>'
            )
            svg_str = svg_str.replace("</svg>", f"{_svg_stamp}</svg>")
        svg_str = svg_str.replace(
            "<svg ", '<svg style="width:100%;height:auto;display:block;" ', 1
        )
        components.html(
            f'<div style="background:#fff;border-radius:8px;padding:12px;'
            f'border:1px solid #D0D7DE;">'
            f'{svg_str}'
            f'<p style="text-align:center;font-size:12px;color:#57606A;'
            f'margin:6px 0 0 0;">{caption}</p>'
            f'</div>',
            height=560, scrolling=True,
        )
        st.markdown(legend, unsafe_allow_html=True)
    else:
        st.warning("No image data available.")


# ══════════════════════════════════════════════════════════════════════════════
#  POSEVIEW UI BLOCK (reusable)
# ══════════════════════════════════════════════════════════════════════════════


def _poseview_ui(
    rec_key, pose_sdf_path,
    pdb_id="", cocrystal_ligand_id="",
    smiles_key="", pose_idx=0,
    img_png_key="", img_svg_key="", pose_key_key="",
    btn_key="", dl_png_key="", dl_svg_key="",
    ref_png_key="", ref_svg_key="",
    label_suffix="",
    lig_name="", lig_smiles="", binding_energy=None,
    ref_lig_name="", ref_lig_smiles="", ref_lig_energy=None,
    show_header=True,
):
    """
    3-tab 2D interaction diagram UI.
      Tab 1 - New local diagram (SVG style, no API)
      Tab 2 - Classic RDKit highlight-circle diagram (no API)
      Tab 3 - PoseView: download files only, no API calls
    """
    _pose_key = (
        f"{st.session_state.get(smiles_key, 'lig')}_pose{pose_idx+1}{label_suffix}"
    )
    _stale   = st.session_state.get(pose_key_key) != _pose_key
    _has_ref = bool(pdb_id and cocrystal_ligand_id)

    if show_header:
        st.markdown("---")
        st.markdown("**🧬 2D Interaction Diagrams**")

    _tab_new, _tab_rdkit, _tab_pv = st.tabs([
        "🧬 Anyone Can Dock 2D Diagram",
        "🔬 RDKit 2D Diagram",
        "⬇ PoseView",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — NEW LOCAL DIAGRAM
    # ══════════════════════════════════════════════════════════════════════════
    with _tab_new:
        _rec    = st.session_state.get(rec_key, "")
        _smiles = lig_smiles or st.session_state.get(smiles_key, "")

        if not _rec or not os.path.exists(_rec):
            st.warning("Complete receptor preparation first.")
        elif not os.path.exists(pose_sdf_path):
            st.warning("Pose SDF not ready.")
        else:
            _pfx          = rec_key.replace("receptor_fh", "")
            _lig_pdb_path = st.session_state.get(_pfx + "ligand_pdb_path", "")
            _has_ref_local = bool(_lig_pdb_path and os.path.exists(_lig_pdb_path))

            _cl, _cr = st.columns(2)
            with _cl:
                _cutoff = st.slider(
                    "Cutoff (Å)", 2.5, 5.5, 4.5, 0.1, key=btn_key + "_cut"
                )
            with _cr:
                _maxres = st.slider(
                    "Max residues", 4, 20, 14, 1, key=btn_key + "_max"
                )

            if st.button("🧬 Generate", key=btn_key + "_gen", type="primary"):
                with st.spinner("Generating docked pose diagram…"):
                    try:
                        _energy_part = ""
                        if binding_energy is not None:
                            _energy_part = f"  ·  {binding_energy:.2f} kcal/mol"
                        _title = f"Pose {pose_idx+1}  ·  {lig_name}{_energy_part}"
                        _svg = draw_interaction_diagram(
                            receptor_pdb=_rec,
                            pose_sdf=pose_sdf_path,
                            smiles=_smiles,
                            title=_title,
                            cutoff=_cutoff,
                            max_residues=_maxres,
                        )
                        _data = draw_interaction_diagram_data(
                            receptor_pdb=_rec,
                            pose_sdf=pose_sdf_path,
                            smiles=_smiles,
                            title=_title,
                            cutoff=_cutoff,
                            max_residues=_maxres,
                        )
                        _html = _render_interactive_diagram(_data) if _data else None
                        st.session_state[img_svg_key + "_new"]   = _svg
                        st.session_state[img_svg_key + "_ihtml"] = _html
                        st.session_state[pose_key_key + "_new"]  = _pose_key
                    except Exception as e:
                        st.error(f"Diagram error: {e}")

                if _has_ref_local:
                    with st.spinner("Generating co-crystal diagram…"):
                        try:
                            import subprocess as _sp
                            from rdkit import Chem
                            _ref_sdf = _lig_pdb_path.replace(".pdb", "_ref.sdf")
                            _sp.run(
                                f'obabel "{_lig_pdb_path}" -O "{_ref_sdf}" 2>/dev/null',
                                shell=True, capture_output=True,
                            )
                            _ref_smiles = ref_lig_smiles or ""
                            if not _ref_smiles and os.path.exists(_ref_sdf):
                                _sup = Chem.SDMolSupplier(_ref_sdf, sanitize=True, removeHs=True)
                                _rm = next((m for m in _sup if m is not None), None)
                                if _rm:
                                    try:
                                        _ref_smiles = Chem.MolToSmiles(Chem.RemoveHs(_rm))
                                    except Exception:
                                        pass
                            _ref_energy_part = ""
                            if ref_lig_energy is not None:
                                _ref_energy_part = f"  ·  {ref_lig_energy:.2f} kcal/mol"
                            _ref_lbl = ref_lig_name or cocrystal_ligand_id
                            _rtitle = f"{_ref_lbl}  ·  Co-crystal{_ref_energy_part}"
                            _ref_sdf_src = _ref_sdf if os.path.exists(_ref_sdf) else pose_sdf_path
                            _ref_svg = draw_interaction_diagram(
                                receptor_pdb=_rec,
                                pose_sdf=_ref_sdf_src,
                                smiles=_ref_smiles,
                                title=_rtitle,
                                cutoff=_cutoff,
                                max_residues=_maxres,
                            )
                            _ref_data = draw_interaction_diagram_data(
                                receptor_pdb=_rec,
                                pose_sdf=_ref_sdf_src,
                                smiles=_ref_smiles,
                                title=_rtitle,
                                cutoff=_cutoff,
                                max_residues=_maxres,
                            )
                            _ref_html = _render_interactive_diagram(_ref_data) if _ref_data else None
                            st.session_state[ref_svg_key + "_new"]   = _ref_svg
                            st.session_state[ref_svg_key + "_ihtml"] = _ref_html
                        except Exception as e:
                            st.warning(f"Co-crystal diagram: {e}")
                st.rerun()

            _new_svg     = st.session_state.get(img_svg_key + "_new")
            _new_ref_svg = st.session_state.get(ref_svg_key + "_new") if ref_svg_key else None
            _new_stale   = st.session_state.get(pose_key_key + "_new") != _pose_key

            if _new_stale and _new_svg:
                st.caption("Pose changed — click Generate to refresh.")

            def _show_svg_new(svg_data, fn_base):
                import base64
                svg_str = svg_data.decode() if isinstance(svg_data, bytes) else svg_data
                svg_str = svg_str.replace(
                    "<svg ", '<svg style="width:100%;height:auto;display:block;" ', 1
                )
                _pb   = svg_to_png(svg_data)
                _pb64 = base64.b64encode(_pb).decode() if _pb else ""
                _sb64 = base64.b64encode(
                    svg_data if isinstance(svg_data, bytes) else svg_data.encode()
                ).decode()
                _bs = (
                    "display:inline-block;padding:7px 18px;border-radius:6px;"
                    "font-size:13px;text-decoration:none;color:#24292F;"
                    "background:#F6F8FA;border:1px solid #D0D7DE;margin:3px;"
                )
                _dls = ""
                if _pb64:
                    _dls += (
                        f'<a href="data:image/png;base64,{_pb64}" '
                        f'download="{fn_base}.png" style="{_bs}">&#8595; PNG</a>'
                    )
                _dls += (
                    f'<a href="data:image/svg+xml;base64,{_sb64}" '
                    f'download="{fn_base}.svg" style="{_bs}">&#8595; SVG</a>'
                )
                components.html(
                    f'<div style="background:#fff;border-radius:8px;'
                    f'border:1px solid #D0D7DE;overflow:hidden;">'
                    f'{svg_str}'
                    f'<div style="padding:8px 12px;border-top:1px solid #eee;">'
                    f'{_dls}</div></div>',
                    height=800, scrolling=False,
                )

            _new_ihtml     = st.session_state.get(img_svg_key + "_ihtml")
            _new_ref_ihtml = st.session_state.get(ref_svg_key + "_ihtml") if ref_svg_key else None

            if _new_svg and not _new_stale:
                _view_mode = st.radio(
                    "View mode", ["🖱 Interactive (drag)", "🖼 Static SVG"],
                    horizontal=True, key=btn_key + "_viewmode",
                )
                _cl2, _cr2 = st.columns(2)
                with _cl2:
                    st.markdown("##### Docked Pose")
                    if _view_mode.startswith("🖱") and _new_ihtml:
                        components.html(_new_ihtml, height=860, scrolling=False)
                    else:
                        _show_svg_new(_new_svg, f"pose{pose_idx+1}_interaction")
                with _cr2:
                    st.markdown("##### Co-Crystal Reference")
                    if _new_ref_svg:
                        if _view_mode.startswith("🖱") and _new_ref_ihtml:
                            components.html(_new_ref_ihtml, height=860, scrolling=False)
                        else:
                            _show_svg_new(_new_ref_svg, "cocrystal_interaction")
                    elif _has_ref_local:
                        st.info("Click Generate to produce the co-crystal diagram.")
                    else:
                        st.caption("No co-crystal ligand — use Auto-detect in receptor prep.")


                st.markdown("---")
                st.markdown("### 🤖 Understand Your Results with AI")
                st.caption(
                    "Download your diagram (PNG button above), then paste the prompt "
                    "below + the image into **Claude**, **GPT-4o**, or **Gemini** "
                    "to get a plain-English explanation of your docking results."
                )

                _estr = (
                    f"{binding_energy:.2f} kcal/mol"
                    if binding_energy is not None else "[binding energy]"
                )
                _lig_display = lig_name or "[ligand]"
                _pdb_display = pdb_id.upper() or "[PDB ID]"
                _has_ref_prompt = bool(_new_ref_svg)
                _ref_display = ref_lig_name or cocrystal_ligand_id or "[co-crystal ligand]"
                # Was the co-crystal ligand re-docked (binding energy available)?
                # ref_lig_energy is only set when redocking was performed
                _ref_redocked = (ref_lig_energy is not None)
                _ref_estr = (
                    f"{ref_lig_energy:.2f} kcal/mol"
                    if _ref_redocked else None
                )

                if _has_ref_prompt:
                    _lines = [
                        "I have just run a molecular docking experiment and I need help",
                        "understanding what my results mean. I am attaching two 2D",
                        "interaction diagrams from the Anyone Can Dock app.",
                        "",
                        f"Docking software: AutoDock Vina v1.2.7",
                        f"Protein target (PDB): {_pdb_display}",
                        f"My docked ligand: {_lig_display}",
                        f"  Predicted binding energy: {_estr}",
                        f"  (more negative = stronger predicted binding)",
                        f"Reference: {_ref_display} co-crystallised in PDB {_pdb_display}" + (f"  |  binding energy from re-docking: {_ref_estr}" if _ref_redocked else "  (see 2D diagram — no re-docking performed)"),
                        "",
                        "How to read the diagrams:",
                        "  Green dashed line     = hydrogen bond (number on line = distance in Angstrom)",
                        "  Magenta dashed line   = pi-pi stacking (aromatic ring interaction)",
                        "  Blue circle (no line) = hydrophobic contact",
                        "  Labels on circles     = amino acid name + residue number + chain",
                        "",
                        "Please help me understand:",
                        "1. What are the most important interactions my ligand makes,",
                        "   and why do they matter for binding?",
                        "2. How does my docked ligand compare to the reference — are the",
                        "   key contacts conserved or different?",
                        "3. Based on the binding energy" + (" and the interaction pattern," if _ref_redocked else " (docked ligand only) and the interaction pattern,") + " does my",
                        "   ligand look like a promising binder, and what could be improved?",
                        "",
                        "Please explain in plain language that a non-expert can follow,",
                        "but include the specific residue names and distances from the diagram.",
                        "",
                        "Finally, write a short ready-to-use paragraph (3-4 sentences) that I",
                        "can copy directly into a report or presentation slide. It should",
                        "summarise the key interactions of my ligand versus the reference,",
                        ("mention both binding energies," if _ref_redocked else "mention the docked ligand's binding energy,"),
                        "and state what this suggests about the binding mode.",
                        "Label this section: 'Ready-to-use summary:'",
                    ]
                else:
                    _lines = [
                        "I have just run a molecular docking experiment and I need help",
                        "understanding what my results mean. I am attaching a 2D",
                        "interaction diagram from the Anyone Can Dock app.",
                        "",
                        f"Docking software: AutoDock Vina v1.2.7",
                        f"Protein target (PDB): {_pdb_display}",
                        f"My docked ligand: {_lig_display}",
                        f"  Predicted binding energy: {_estr}",
                        f"  (more negative = stronger predicted binding)",
                        "",
                        "How to read the diagram:",
                        "  Green dashed line     = hydrogen bond (number on line = distance in Angstrom)",
                        "  Magenta dashed line   = pi-pi stacking (aromatic ring interaction)",
                        "  Blue circle (no line) = hydrophobic contact",
                        "  Labels on circles     = amino acid name + residue number + chain",
                        "",
                        "Please help me understand:",
                        "1. What interactions is my ligand making with the protein,",
                        "   and which ones are most important for binding?",
                        "2. What does the binding energy value tell me — is this a",
                        "   strong or weak predicted binder?",
                        "3. Are there any obvious ways the binding could be improved",
                        "   (e.g. missing interactions, clashes, weak contacts)?",
                        "",
                        "Please explain in plain language that a non-expert can follow,",
                        "but include the specific residue names and distances from the diagram.",
                        "",
                        "Finally, write a short ready-to-use paragraph (3-4 sentences) that I",
                        "can copy directly into a report or presentation slide. It should",
                        "summarise the key protein-ligand interactions, mention the binding",
                        "energy, and state what this suggests about the binding mode.",
                        "Label this section: 'Ready-to-use summary:'",
                    ]

                _prompt = "\n".join(_lines)
                st.code(_prompt, language=None)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — RDKIT CLASSIC
    # ══════════════════════════════════════════════════════════════════════════
    with _tab_rdkit:
        _rec2    = st.session_state.get(rec_key, "")
        _smiles2 = lig_smiles or st.session_state.get(smiles_key, "")
        _pfx2    = rec_key.replace("receptor_fh", "")
        _lig_pdb2 = st.session_state.get(_pfx2 + "ligand_pdb_path", "")
        _has_ref_rdkit2 = bool(_lig_pdb2 and os.path.exists(_lig_pdb2))

        st.caption(
            "RDKit highlight-circle style — blue = H-bond/polar · "
            "green = hydrophobic · pink = other. "
            "Works locally with no server needed."
        )

        if not _rec2 or not os.path.exists(_rec2):
            st.warning("Complete receptor preparation first.")
        elif not os.path.exists(pose_sdf_path):
            st.warning("Pose SDF not ready.")
        else:
            with st.expander("🔍 SMILES used for 2D diagram", expanded=False):
                st.code(_smiles2 or "[no SMILES]", language=None)
                st.caption("Dimorphite-DL protonated SMILES (includes charges like [O-], [NH3+]).")

            _cl3, _cr3 = st.columns(2)
            with _cl3:
                _cut2 = st.slider(
                    "Interaction cutoff (Å)", 2.5, 5.0, 3.5, 0.1,
                    key=btn_key + "_rdk_cut",
                )
            with _cr3:
                _max2 = st.slider(
                    "Max residues shown", 4, 20, 10, 1,
                    key=btn_key + "_rdk_max",
                    help="Reduce to clean up busy diagrams.",
                )

            if st.button("🔬 Generate Both RDKit Diagrams", key=btn_key + "_rdk_gen", type="primary"):
                # — Docked pose —
                with st.spinner("⏳ Generating docked pose diagram…"):
                    try:
                        _mols2 = load_mols_from_sdf(pose_sdf_path)
                        _mol2  = _mols2[0] if _mols2 else None
                        if _mol2 is None:
                            st.error("Could not read pose SDF.")
                        else:
                            _etitle2 = (
                                f"Pose {pose_idx+1}  ·  {lig_name}"
                                + (f"  ·  {binding_energy:.2f} kcal/mol"
                                   if binding_energy is not None else "")
                            )
                            _rdk_svg = draw_interactions_rdkit_classic(
                                lig_mol=_mol2,
                                receptor_pdb=_rec2,
                                smiles=_smiles2,
                                title=_etitle2,
                                cutoff=_cut2,
                                size=(650, 620),
                                max_residues=_max2,
                            )
                            st.session_state[img_svg_key + "_rdk"]  = _rdk_svg
                            st.session_state[pose_key_key + "_rdk"] = _pose_key
                    except Exception as e:
                        st.error(f"❌ RDKit docked pose error: {e}")

                # — Co-crystal reference —
                if _has_ref_rdkit2:
                    with st.spinner("⏳ Generating co-crystal reference diagram…"):
                        try:
                            from rdkit import Chem as _Chem2
                            import subprocess as _sp2
                            _ref_sdf_tmp = _lig_pdb2.replace(".pdb", "_ref.sdf")
                            _sp2.run(
                                f'obabel "{_lig_pdb2}" -O "{_ref_sdf_tmp}" 2>/dev/null',
                                shell=True, capture_output=True,
                            )
                            _ref_mol2 = None
                            if os.path.exists(_ref_sdf_tmp):
                                _sup2 = _Chem2.SDMolSupplier(_ref_sdf_tmp, sanitize=True, removeHs=True)
                                _ref_mol2 = next((m for m in _sup2 if m is not None), None)
                            if _ref_mol2 is None:
                                _ref_mol2 = _Chem2.MolFromPDBFile(_lig_pdb2, sanitize=False, removeHs=True)
                                if _ref_mol2 is not None:
                                    try: _Chem2.SanitizeMol(_ref_mol2)
                                    except: pass
                            if _ref_mol2 is not None:
                                _ref_smi2 = ref_lig_smiles or ""
                                if not _ref_smi2:
                                    try: _ref_smi2 = _Chem2.MolToSmiles(_Chem2.RemoveHs(_ref_mol2))
                                    except: _ref_smi2 = ""
                                _ref_title2 = (
                                    f"{ref_lig_name or cocrystal_ligand_id}  ·  Co-crystal"
                                    + (f"  ·  {ref_lig_energy:.2f} kcal/mol"
                                       if ref_lig_energy is not None else "")
                                )
                                _ref_rdk_svg = draw_interactions_rdkit_classic(
                                    lig_mol=_ref_mol2,
                                    receptor_pdb=_rec2,
                                    smiles=_ref_smi2,
                                    title=_ref_title2,
                                    cutoff=_cut2,
                                    size=(650, 620),
                                    max_residues=_max2,
                                )
                                st.session_state[img_svg_key + "_rdk_ref"] = _ref_rdk_svg
                            else:
                                st.warning("⚠️ Could not read co-crystal ligand PDB.")
                        except Exception as e:
                            st.warning(f"⚠️ Co-crystal RDKit diagram error: {e}")
                st.rerun()

            _rdk_svg     = st.session_state.get(img_svg_key + "_rdk")
            _ref_rdk_svg = st.session_state.get(img_svg_key + "_rdk_ref")
            _rdk_stale   = st.session_state.get(pose_key_key + "_rdk") != _pose_key

            if _rdk_stale and _rdk_svg:
                st.caption("⚠️ Pose changed — click **Generate RDKit Diagrams** to update.")

            _btn_style = (
                "flex:1;display:inline-block;text-align:center;text-decoration:none;"
                "padding:8px 0;border-radius:6px;font-size:13px;font-weight:500;"
                "color:#24292F;background:#F6F8FA;border:1px solid #D0D7DE;"
                "cursor:pointer;"
            )

            def _show_rdkit_svg_tab(svg_data, dl_key_prefix, dl_filename):
                import base64 as _b64
                _sv  = svg_data.decode() if isinstance(svg_data, bytes) else svg_data
                _sv  = _sv.replace("<svg ", '<svg style="width:100%;height:auto;display:block;" ', 1)
                _pb  = svg_to_png(svg_data)
                _pb64 = _b64.b64encode(_pb).decode() if _pb else ""
                _sb64 = _b64.b64encode(
                    svg_data if isinstance(svg_data, bytes) else svg_data.encode()
                ).decode()
                _png_fn = dl_filename.replace(".svg", ".png")
                _png_lnk = (
                    f'<a href="data:image/png;base64,{_pb64}" download="{_png_fn}"'
                    f' style="{_btn_style}">&#8595; PNG</a>'
                ) if _pb64 else ""
                _svg_lnk = (
                    f'<a href="data:image/svg+xml;base64,{_sb64}" download="{dl_filename}"'
                    f' style="{_btn_style}">&#8595; SVG</a>'
                )
                components.html(
                    f"""<div style="background:#fff;border-radius:8px;
                        border:1px solid #D0D7DE;overflow:hidden;font-family:'Helvetica Neue',Arial,sans-serif;">
                      {_sv}
                      <div style="display:flex;align-items:center;gap:20px;padding:10px 16px;
                           border-top:1px solid #D0D7DE;font-size:13px;color:#333;">
                        <div style="display:flex;align-items:center;gap:7px;">
                          <div style="width:14px;height:14px;border-radius:50%;flex-shrink:0;
                               background:rgba(89,156,214,0.55);border:1px solid #5B9BD5;"></div>
                          <span>H-bond / polar</span></div>
                        <div style="display:flex;align-items:center;gap:7px;">
                          <div style="width:14px;height:14px;border-radius:50%;flex-shrink:0;
                               background:rgba(44,141,87,0.55);border:1px solid #2E8B57;"></div>
                          <span>Hydrophobic</span></div>
                        <div style="display:flex;align-items:center;gap:7px;">
                          <div style="width:14px;height:14px;border-radius:50%;flex-shrink:0;
                               background:rgba(204,95,138,0.55);border:1px solid #cc5f8a;"></div>
                          <span>Other</span></div>
                      </div>
                      <div style="display:flex;gap:8px;padding:10px 12px;border-top:1px solid #D0D7DE;">
                        {_png_lnk}{_svg_lnk}
                      </div>
                    </div>""",
                    height=740, scrolling=False,
                )

            if _rdk_svg and not _rdk_stale:
                _col_l2, _col_r2 = st.columns(2)
                with _col_l2:
                    st.markdown("##### 🧪 Docked Pose (RDKit)")
                    _show_rdkit_svg_tab(
                        _rdk_svg,
                        dl_key_prefix=btn_key + "_rdk_dl",
                        dl_filename=f"pose{pose_idx+1}_rdkit.svg",
                    )
                with _col_r2:
                    st.markdown("##### 🔮 Co-Crystal Reference (RDKit)")
                    if _ref_rdk_svg:
                        _show_rdkit_svg_tab(
                            _ref_rdk_svg,
                            dl_key_prefix=btn_key + "_rdk_ref_dl",
                            dl_filename="cocrystal_rdkit.svg",
                        )
                    elif _has_ref_rdkit2:
                        st.info("Click **Generate RDKit Diagrams** to generate co-crystal diagram.")
                    else:
                        st.caption("⚠️ No co-crystal ligand — use Auto-detect in receptor preparation.")

                st.markdown("---")
                # AI Prompt (RDKit)
                st.markdown("### 🤖 Understand Your Results with AI")
                st.caption(
                    "Screenshot this diagram, then paste the prompt below "
                    "+ the screenshot into **Claude**, **GPT-4o**, or **Gemini**."
                )
                _estr_r = (
                    f"{binding_energy:.2f} kcal/mol"
                    if binding_energy is not None else "[binding energy]"
                )
                _lig_r   = lig_name or "[ligand]"
                _pdb_r   = pdb_id.upper() if pdb_id else "[PDB ID]"
                _ref_redocked_r = (ref_lig_energy is not None)
                _ref_display_r  = ref_lig_name or cocrystal_ligand_id or "[co-crystal ligand]"
                _ref_estr_r     = (
                    f"{ref_lig_energy:.2f} kcal/mol"
                    if _ref_redocked_r else None
                )
                if _ref_rdk_svg:
                    _rdk_lines = [
                        "I have just run a molecular docking experiment and I need help",
                        "understanding what my results mean. I am attaching two 2D",
                        "interaction diagrams (RDKit style) from the Anyone Can Dock app.",
                        "",
                        f"Docking software: AutoDock Vina v1.2.7",
                        f"Protein target (PDB): {_pdb_r}",
                        f"My docked ligand: {_lig_r}",
                        f"  Predicted binding energy: {_estr_r}",
                        f"  (more negative = stronger predicted binding)",
                        f"Reference: {_ref_display_r} co-crystallised in PDB {_pdb_r}"
                        + (f"  |  binding energy from re-docking: {_ref_estr_r}"
                           if _ref_redocked_r else "  (see 2D diagram — no re-docking performed)"),
                        "",
                        "How to read the diagrams:",
                        "  Blue circle   = H-bond / polar contact",
                        "  Green circle  = hydrophobic contact",
                        "  Pink circle   = other interaction",
                        "  Residue labels sit next to each colored circle",
                        "",
                        "Please help me understand:",
                        "1. What are the most important interactions my ligand makes,",
                        "   and why do they matter for binding?",
                        "2. How does my docked ligand compare to the reference — are the",
                        "   key contacts conserved or different?",
                        "3. Based on the binding energy and interaction pattern, does my",
                        "   ligand look like a promising binder, and what could be improved?",
                        "",
                        "Please explain in plain language that a non-expert can follow,",
                        "but include the specific residue names from the diagram.",
                        "",
                        "Finally, write a short ready-to-use paragraph (3-4 sentences) that I",
                        "can copy directly into a report or presentation slide. It should",
                        "summarise the key interactions of my ligand versus the reference,",
                        ("mention both binding energies,"
                         if _ref_redocked_r else "mention the docked ligand's binding energy,"),
                        "and state what this suggests about the binding mode.",
                        "Label this section: 'Ready-to-use summary:'",
                    ]
                else:
                    _rdk_lines = [
                        "I have just run a molecular docking experiment and I need help",
                        "understanding what my results mean. I am attaching a 2D",
                        "interaction diagram (RDKit style) from the Anyone Can Dock app.",
                        "",
                        f"Docking software: AutoDock Vina v1.2.7",
                        f"Protein target (PDB): {_pdb_r}",
                        f"My docked ligand: {_lig_r}",
                        f"  Predicted binding energy: {_estr_r}",
                        f"  (more negative = stronger predicted binding)",
                        "",
                        "How to read the diagram:",
                        "  Blue circle   = H-bond / polar contact",
                        "  Green circle  = hydrophobic contact",
                        "  Pink circle   = other interaction",
                        "  Residue labels sit next to each colored circle",
                        "",
                        "Please help me understand:",
                        "1. What interactions is my ligand making with the protein,",
                        "   and which ones are most important for binding?",
                        "2. What does the binding energy value tell me — is this a",
                        "   strong or weak predicted binder?",
                        "3. Are there any obvious ways the binding could be improved?",
                        "",
                        "Please explain in plain language that a non-expert can follow,",
                        "but include the specific residue names from the diagram.",
                        "",
                        "Finally, write a short ready-to-use paragraph (3-4 sentences) that I",
                        "can copy directly into a report or presentation slide. It should",
                        "summarise the key protein-ligand interactions, mention the binding",
                        "energy, and state what this suggests about the binding mode.",
                        "Label this section: 'Ready-to-use summary:'",
                    ]
                st.code("\n".join(_rdk_lines), language=None)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — POSEVIEW DOWNLOAD ONLY
    # ══════════════════════════════════════════════════════════════════════════
    with _tab_pv:
        st.caption(
            "Download these files and upload manually at "
            "[proteins.plus/poseview](https://proteins.plus/help/poseview). "
            "No API calls are made from this app."
        )
        _rec_path = st.session_state.get(rec_key, "")
        _pfx3     = rec_key.replace("receptor_fh", "")
        _lig_pdb3 = st.session_state.get(_pfx3 + "ligand_pdb_path", "")
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            st.markdown("**Receptor**")
            if _rec_path and os.path.exists(_rec_path):
                st.download_button(
                    "⬇ receptor.pdb", open(_rec_path, "rb"),
                    file_name="receptor.pdb", mime="chemical/x-pdb",
                    key=btn_key + "_pv_rec", width="stretch",
                )
            else:
                st.caption("Not ready.")
        with _c2:
            st.markdown("**Docked pose**")
            if os.path.exists(pose_sdf_path):
                st.download_button(
                    "⬇ docked_pose.sdf", open(pose_sdf_path, "rb"),
                    file_name=f"pose_{pose_idx+1}_docked.sdf",
                    mime="chemical/x-mdl-sdfile",
                    key=btn_key + "_pv_sdf", width="stretch",
                )
            else:
                st.caption("Not ready.")
        with _c3:
            st.markdown("**Co-crystal ligand**")
            if _lig_pdb3 and os.path.exists(_lig_pdb3):
                _cc_sdf = _lig_pdb3.replace(".pdb", "_cocrystal.sdf")
                if not os.path.exists(_cc_sdf):
                    import subprocess as _sp3
                    _sp3.run(
                        f'obabel "{_lig_pdb3}" -O "{_cc_sdf}" 2>/dev/null',
                        shell=True, capture_output=True,
                    )
                _dl_p  = _cc_sdf if os.path.exists(_cc_sdf) else _lig_pdb3
                _dl_fn = "cocrystal.sdf" if _dl_p.endswith(".sdf") else "cocrystal.pdb"
                st.download_button(
                    f"⬇ {_dl_fn}", open(_dl_p, "rb"),
                    file_name=_dl_fn,
                    mime=(
                        "chemical/x-mdl-sdfile"
                        if _dl_fn.endswith(".sdf") else "chemical/x-pdb"
                    ),
                    key=btn_key + "_pv_cc", width="stretch",
                )
            else:
                st.caption("No co-crystal ligand detected.")
        st.markdown("---")
        st.markdown(
            "**Steps:**  \n"
            "1. Download `receptor.pdb` and `docked_pose.sdf`  \n"
            "2. Go to [proteins.plus](https://proteins.plus/help/poseview)  \n"
            "3. Upload the receptor, then the ligand SDF  \n"
            "4. PoseView renders the diagram in your browser"
        )


def _receptor_section(pfx: str, wdir: Path, step_label: str):
    import py3Dmol
    from core import run_cmd as _run_cmd

    done     = st.session_state.get(pfx + "receptor_done", False)
    card_cls = "step-card done" if done else "step-card"

    st.markdown(
        f'<div class="{card_cls}"><div class="step-title">{step_label}</div>'
        f'<div class="step-heading">📦 Receptor Preparation</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        src = st.radio(
            "Structure source", ["Download from RCSB", "Upload PDB / CIF file"],
            horizontal=True, key=pfx + "src_mode",
        )
        if src == "Download from RCSB":
            _id_col, _fmt_col = st.columns([1.5, 1])
            with _id_col:
                pdb_id = st.text_input("PDB ID", value="1M17", max_chars=4, key=pfx + "pdb_id")
            with _fmt_col:
                rcsb_fmt = st.radio(
                    "Format", ["PDB", "CIF"],
                    horizontal=True, key=pfx + "rcsb_fmt",
                    help="CIF (mmCIF) is recommended for large or newer entries that may lack PDB-format files.",
                )
            upload_file = None
        else:
            upload_file = st.file_uploader(
                "Upload .pdb or .cif", type=["pdb", "cif", "mmcif"],
                key=pfx + "pdb_upload",
            )
            pdb_id   = None
            rcsb_fmt = None

        center_mode = st.radio(
            "Grid center",
            [
                "Auto-detect co-crystal ligand",
                "Enter XYZ manually",
                "Select by atom selection (ProDy)",
            ],
            horizontal=True, key=pfx + "center_mode",
        )
        if center_mode == "Enter XYZ manually":
            c1, c2, c3 = st.columns(3)
            c1.number_input("X", value=0.0, key=pfx + "mx")
            c2.number_input("Y", value=0.0, key=pfx + "my")
            c3.number_input("Z", value=0.0, key=pfx + "mz")
        elif center_mode == "Select by atom selection (ProDy)":
            st.text_input(
                "ProDy selection string",
                value="resid 702 820 and chain A",
                key=pfx + "mda_sel",
                help=(
                    "e.g. resname ATP"
                    " · resid 84 86 134 and chain A"
                    " · resname LIG and chain A"
                ),
            )
            st.caption(
                "💡 `resname LIG and chain A`"
                " · `resid 701 and chain A`"
                " · `resid 84 to 100 and chain B`"
            )

    with col_b:
        st.markdown("**Search box size (Å)**")
        sx = st.slider("X size", 10, 40, 16, 2, key=pfx + "sx")
        sy = st.slider("Y size", 10, 40, 16, 2, key=pfx + "sy")
        sz = st.slider("Z size", 10, 40, 16, 2, key=pfx + "sz")
        st.markdown(f"Box volume: **{sx*sy*sz:,} Å³**")

    if st.button("▶ Prepare Receptor", key=pfx + "btn_receptor", type="primary"):

        if src == "Download from RCSB":
            token = pdb_id.strip().upper()
            _fmt  = st.session_state.get(pfx + "rcsb_fmt", "PDB")
            if _fmt == "CIF":
                raw_path = str(wdir / "raw.cif")
                _dl_url  = f"https://files.rcsb.org/download/{token}.cif"
            else:
                raw_path = str(wdir / "raw.pdb")
                _dl_url  = f"https://files.rcsb.org/download/{token}.pdb"
            rc, _ = _run_cmd([
                "curl", "-sf", _dl_url, "-o", raw_path,
            ])
            if rc != 0 or not os.path.exists(raw_path) or os.path.getsize(raw_path) < 200:
                if _fmt == "PDB":
                    raw_path_cif = str(wdir / "raw.cif")
                    rc2, _ = _run_cmd([
                        "curl", "-sf",
                        f"https://files.rcsb.org/download/{token}.cif",
                        "-o", raw_path_cif,
                    ])
                    if rc2 == 0 and os.path.exists(raw_path_cif) and os.path.getsize(raw_path_cif) > 200:
                        raw_path = raw_path_cif
                        st.info(f"ℹ️ PDB format unavailable for {token} — using CIF instead.")
                    else:
                        st.error(f"❌ Download failed for {token} (tried both PDB and CIF)")
                        st.stop()
                else:
                    st.error(f"❌ Download failed for {token}")
                    st.stop()
            st.session_state[pfx + "pdb_token"] = token
        else:
            if upload_file is None:
                st.error("Please upload a PDB or CIF file first.")
                st.stop()
            _up_ext  = Path(upload_file.name).suffix.lower()
            if _up_ext in (".cif", ".mmcif"):
                raw_path = str(wdir / "raw.cif")
            else:
                raw_path = str(wdir / "raw.pdb")
            with open(raw_path, "wb") as f:
                f.write(upload_file.read())
            st.session_state[pfx + "pdb_token"] = Path(upload_file.name).stem

        _mode_map = {
            "Auto-detect co-crystal ligand":      "auto",
            "Enter XYZ manually":                 "manual",
            "Select by atom selection (ProDy)":   "selection",
        }
        _core_mode  = _mode_map[center_mode]
        _manual_xyz = (
            st.session_state.get(pfx + "mx", 0.0),
            st.session_state.get(pfx + "my", 0.0),
            st.session_state.get(pfx + "mz", 0.0),
        )
        _prody_sel = st.session_state.get(pfx + "mda_sel", "")

        with st.spinner("⏳ Preparing receptor…"):
            result = prepare_receptor(
                raw_pdb     = raw_path,
                wdir        = wdir,
                center_mode = _core_mode,
                manual_xyz  = _manual_xyz,
                prody_sel   = _prody_sel,
                box_size    = (sx, sy, sz),
            )

        if result["success"]:
            st.session_state.update({
                pfx + "receptor_fh":         result["rec_fh"],
                pfx + "receptor_pdbqt":      result["rec_pdbqt"],
                pfx + "box_pdb":             result["box_pdb"],
                pfx + "config_txt":          result["config_txt"],
                pfx + "cx":                  result["cx"],
                pfx + "cy":                  result["cy"],
                pfx + "cz":                  result["cz"],
                pfx + "box_sx":              result["sx"],
                pfx + "box_sy":              result["sy"],
                pfx + "box_sz":              result["sz"],
                pfx + "ligand_pdb_path":     result["ligand_pdb_path"],
                pfx + "cocrystal_ligand_id": result["cocrystal_ligand_id"],
                pfx + "receptor_done":       True,
                pfx + "receptor_log":        "\n".join(result["log"]),
            })
            clear_poseview_cache()
        else:
            st.error(f"❌ Receptor preparation failed: {result['error']}")
            st.session_state[pfx + "receptor_done"] = False
            st.session_state[pfx + "receptor_log"]  = "\n".join(result["log"])

    if st.session_state.get(pfx + "receptor_done"):
        token   = st.session_state.get(pfx + "pdb_token", "")
        cx_v    = st.session_state.get(pfx + "cx", 0)
        cy_v    = st.session_state.get(pfx + "cy", 0)
        cz_v    = st.session_state.get(pfx + "cz", 0)
        _sx     = st.session_state.get(pfx + "box_sx", 16)
        _sy     = st.session_state.get(pfx + "box_sy", 16)
        _sz     = st.session_state.get(pfx + "box_sz", 16)
        _lig_id = st.session_state.get(pfx + "cocrystal_ligand_id", "")

        _center_pill = f"Center ({cx_v:.2f}, {cy_v:.2f}, {cz_v:.2f})"
        _box_pill    = f"Box {_sx}x{_sy}x{_sz} A"
        st.markdown(
            f"{_pill('Receptor ready', 'success')} {_pill(token)} "
            f"{_pill(_center_pill)} {_pill(_box_pill)}"
            + (f" {_pill('PoseView2: ' + _lig_id)}" if _lig_id else ""),
            unsafe_allow_html=True,
        )
        with st.expander("📋 Preparation log", expanded=False):
            st.markdown(
                f'<div class="log-box">'
                f'{st.session_state.get(pfx + "receptor_log", "")}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with st.expander("🔭 3D: Receptor + Docking Box", expanded=True):
            v3 = py3Dmol.view(width="100%", height=480)
            v3.setBackgroundColor(_viewer_bg())
            mi = 0
            for _path, _style in [
                (st.session_state.get(pfx + "receptor_fh"),
                 {"cartoon": {"color": "spectrum", "opacity": 0.65}}),
                (st.session_state.get(pfx + "box_pdb"),
                 {"stick": {"radius": 0.2, "color": "gray"}}),
            ]:
                if _path and os.path.exists(_path):
                    v3.addModel(open(_path).read(), "pdb")
                    v3.setStyle({"model": mi}, _style)
                    mi += 1
            lig_p = st.session_state.get(pfx + "ligand_pdb_path")
            if lig_p and os.path.exists(lig_p):
                v3.addModel(open(lig_p).read(), "pdb")
                v3.setStyle({"model": mi}, {
                    "stick": {"colorscheme": "magentaCarbon", "radius": 0.25}
                })
            _add_box_to_view(v3, cx_v, cy_v, cz_v, _sx, _sy, _sz)
            try:
                for _end, _col, _lbl in [
                    ({"x": cx_v+8, "y": cy_v,   "z": cz_v},   "red",   "X"),
                    ({"x": cx_v,   "y": cy_v+8,  "z": cz_v},   "green", "Y"),
                    ({"x": cx_v,   "y": cy_v,    "z": cz_v+8}, "blue",  "Z"),
                ]:
                    _st = {"x": cx_v, "y": cy_v, "z": cz_v}
                    v3.addArrow({
                        "start": _st, "end": _end,
                        "radius": 0.15, "color": _col, "radiusRatio": 3.0,
                    })
                    v3.addLabel(_lbl, {
                        "fontSize": 14, "fontColor": _col,
                        "backgroundColor": "black", "backgroundOpacity": 0.6,
                        "inFront": True, "showBackground": True,
                    }, _end)
            except Exception:
                pass
            v3.zoomTo()
            show3d(v3, height=480)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<hr class="step-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:flex-start;gap:12px;">
<img src="https://raw.githubusercontent.com/nyelidl/anyone-docking/main/any-L.svg" width="70">
<h1 style="background:linear-gradient(90deg,#ff4b4b,#ff4fa3,#7a6cff,#21a5e9);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
margin:0;font-weight:700;padding-top:29px;">nyone can dock, everyone can do!</h1>
</div>""", unsafe_allow_html=True)

st.markdown(
    "Molecular docking powered by **AutoDock Vina 1.2.7,** "
    "**pKaNET Cloud**, and **RDkit**."
)
st.markdown(
    "**Basic** — single ligand. **Batch** — multiple ligands.")
st.markdown(
    "**☁️ Cloud-ready | 📱 Mobile-compatible**"
)

if VINA_PATH is None:
    st.error(f"❌ Could not download Vina binary: {_vina_err}")
    st.stop()

if not _OBABEL_OK:
    st.error(
        "❌ OpenBabel not found. "
        "Add `openbabel` to your `packages.txt` file and redeploy."
    )
    st.stop()

st.markdown(
    f"{_pill('Vina 1.2.7 ready', 'success')} ",
    unsafe_allow_html=True,
)
st.markdown('<hr class="step-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_basic, tab_batch = st.tabs([
    "🧪  Basic — single ligand",
    "🔬  Batch — multiple ligands",
])


# ╔════════════════════════════════════════════════════════════════════════════╗
#  TAB 1 — BASIC DOCKING
# ╚════════════════════════════════════════════════════════════════════════════╝
with tab_basic:

    _receptor_section(pfx="", wdir=WORKDIR, step_label="Step 1 of 4")

    # ── Step 2: Ligand ────────────────────────────────────────────────────────
    card_cls = "step-card done" if st.session_state.ligand_done else "step-card"
    st.markdown(
        f'<div class="{card_cls}"><div class="step-title">Step 2 of 4</div>'
        f'<div class="step-heading">⚗️ Ligand Preparation</div>',
        unsafe_allow_html=True,
    )

    lig_input_mode = st.radio(
        "Input mode",
        ["SMILES string", "Upload structure (.pdb)", "Draw structure (Ketcher)"],
        horizontal=True, key="lig_input_mode",
    )

    smiles_in = ""
    if lig_input_mode == "SMILES string":
        smiles_in = st.text_input(
            "SMILES string",
            value="COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
            key="smiles_in",
        )
    elif lig_input_mode == "Upload structure (.pdb)":
        st.file_uploader(
            "Upload structure file", type=["sdf", "mol2", "pdb"],
            key="lig_struct_file",
        )
    else:
        try:
            from streamlit_ketcher import st_ketcher
            _k = st_ketcher(
                st.session_state.get("ketcher_smi", ""),
                height=400, key="ketcher_widget",
            )
            if _k:
                st.session_state["ketcher_smi"] = _k
                smiles_in = _k
            else:
                smiles_in = st.session_state.get("ketcher_smi", "")
        except ImportError:
            st.error("❌ `streamlit-ketcher` not installed — add it to requirements.txt")
            smiles_in = ""

    lig_name_in = st.text_input("Output name", value="ELR", key="lig_name_in")
    ph_in       = st.number_input("Target pH", 0.0, 14.0, 7.4, 0.1, key="ph_in")

    if not st.session_state.receptor_done:
        st.caption("⚠ Complete Step 1 first.")
    if st.button(
        "▶ Prepare Ligand", key="btn_ligand", type="primary",
        disabled=not st.session_state.receptor_done,
    ):
        lig_name = lig_name_in.strip() or "LIG"
        with st.spinner("Preparing ligand…"):
            _mode = st.session_state.get("lig_input_mode", "SMILES string")

            if "Upload" in _mode:
                _sfobj = st.session_state.get("lig_struct_file")
                if _sfobj is None:
                    st.error("No structure file uploaded")
                    st.stop()
                _ext = Path(_sfobj.name).suffix.lower()
                _tmp = str(WORKDIR / f"lig_upload{_ext}")
                with open(_tmp, "wb") as _f:
                    _f.write(_sfobj.read())
                try:
                    smiles_in = smiles_from_file(_tmp, WORKDIR)
                except Exception as e:
                    st.error(f"❌ Could not read structure: {e}")
                    st.stop()
            elif "Ketcher" in _mode:
                smiles_in = st.session_state.get("ketcher_smi", "").strip()
                if not smiles_in:
                    st.error("No molecule drawn in Ketcher.")
                    st.stop()

            result = prepare_ligand(smiles_in, lig_name, ph_in, WORKDIR)

        if result["success"]:
            st.session_state.update({
                "ligand_pdbqt": result["pdbqt"],
                "ligand_sdf":   result["sdf"],
                "ligand_name":  lig_name,
                "prot_smiles":  result["prot_smiles"],
                "ligand_done":  True,
                "ligand_log":   "\n".join(result["log"]),
            })
        else:
            st.error(f"❌ Ligand preparation failed: {result['error']}")
            st.session_state.ligand_done = False
            st.session_state.ligand_log  = "\n".join(result["log"])

    if st.session_state.ligand_done:
        import py3Dmol
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw

        st.markdown(
            f"{_pill('Ligand ready', 'success')} {_pill(st.session_state.ligand_name)}",
            unsafe_allow_html=True,
        )
        with st.expander("📋 Preparation log", expanded=False):
            st.markdown(
                f'<div class="log-box">{st.session_state.ligand_log}</div>',
                unsafe_allow_html=True,
            )

        c2d, c3d = st.columns(2)
        with c2d:
            st.markdown("**2D Structure**")
            try:
                m2 = Chem.MolFromSmiles(st.session_state.prot_smiles)
                AllChem.Compute2DCoords(m2)
                buf = io.BytesIO()
                Draw.MolToImage(m2, size=(320, 260)).save(buf, format="PNG")
                _png_to_b64_img(
                    buf.getvalue(),
                    style="width:100%;max-width:320px;height:auto;border-radius:6px;",
                )
            except Exception as e:
                st.info(f"2D unavailable: {e}")
        with c3d:
            st.markdown("**3D Conformer**")
            try:
                vl = py3Dmol.view(width="100%", height=280)
                vl.setBackgroundColor(_viewer_bg())
                vl.addModel(open(st.session_state.ligand_sdf).read(), "sdf")
                vl.setStyle({}, {"stick": {"colorscheme": "yellowCarbon", "radius": 0.2}})
                vl.zoomTo()
                show3d(vl, height=280)
            except Exception as e:
                st.info(f"3D viewer unavailable: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<hr class="step-divider">', unsafe_allow_html=True)

    # ── Step 3: Docking ───────────────────────────────────────────────────────
    card_cls = "step-card done" if st.session_state.docking_done else "step-card"
    st.markdown(
        f'<div class="{card_cls}"><div class="step-title">Step 3 of 4</div>'
        f'<div class="step-heading">🚀 Run Docking</div>',
        unsafe_allow_html=True,
    )

    cd1, cd2 = st.columns([1.5, 1])
    with cd1:
        exh = st.slider("Exhaustiveness", 4, 64, 16, 2, key="exh_slider")
        nm  = st.slider("Number of poses", 5, 20, 10, 1, key="n_modes")
        er  = st.slider("Energy range (kcal/mol)", 1, 5, 3, 1, key="e_range")
    with cd2:
        est = max(1, exh // 8)
        st.markdown(
            f'<div style="background:var(--bg-subtle);border:1px solid var(--border);'
            f'border-radius:8px;padding:16px;">'
            f'<div style="color:var(--text-muted);font-size:0.8rem">ESTIMATED TIME</div>'
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:2rem;'
            f'color:var(--warn)">~{est}&#8211;{est*3} min</div>'
            f'<div style="color:var(--text-muted);font-size:0.8rem">'
            f'exhaustiveness = {exh}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Redocking validation**")
        do_redock = st.checkbox(
            "Dock co-crystal ligand as reference", value=False, key="do_redock",
            help="Dock the co-crystal ligand first as a reference. "
                 "Its score appears as a dashed line in the plot, "
                 "and RMSD is calculated against the crystal pose.",
        )
        if do_redock:
            st.text_input(
                "Co-crystal SMILES [name]",
                value=(
                    "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC "
                    "Erlotinib"
                ),
                key="redock_smiles",
            )
            st.caption("Score shown as dashed reference line in plot.")

    if not st.session_state.ligand_done:
        st.caption("⚠ Complete Steps 1 & 2 first.")
    if st.button(
        "▶ Run Docking", key="btn_dock", type="primary",
        disabled=not st.session_state.ligand_done,
    ):
        base   = st.session_state.ligand_name
        pv_sdf = str(WORKDIR / f"{base}_pv_ready.sdf")

        # ── Redocking ─────────────────────────────────────────────────────
        redock_score  = None
        redock_result = None
        if st.session_state.get("do_redock"):
            raw_rd = st.session_state.get("redock_smiles", "").strip()
            pts    = raw_rd.split(None, 1)
            rd_smi = pts[0]
            rd_nm  = pts[1].replace(" ", "_") if len(pts) > 1 else "redock"
            ph_val = st.session_state.get("ph_in", 7.4)
            with st.spinner(f"Docking reference ligand ({rd_nm})…"):
                rd_prep = prepare_ligand(rd_smi, "redock_" + rd_nm, ph_val, WORKDIR)
                if rd_prep["success"]:
                    rd_dock = run_vina(
                        st.session_state.receptor_pdbqt,
                        rd_prep["pdbqt"],
                        st.session_state.config_txt,
                        VINA_PATH, exh, nm, er,
                        WORKDIR, "redock_" + rd_nm,
                    )
                    if rd_dock["success"] and rd_dock["top_score"] is not None:
                        redock_score = rd_dock["top_score"]
                        rd_pv_sdf    = str(WORKDIR / f"redock_{rd_nm}_pv_ready.sdf")
                        fix_sdf_bond_orders(rd_dock["out_sdf"], rd_smi, rd_pv_sdf)
                        if not os.path.exists(rd_pv_sdf) or os.path.getsize(rd_pv_sdf) < 10:
                            rd_pv_sdf = rd_dock["out_sdf"]
                        rd_n = (
                            len(load_mols_from_sdf(rd_dock["out_sdf"], sanitize=False))
                            if os.path.exists(rd_dock["out_sdf"]) else 0
                        )
                        redock_result = {
                            "Name":        f"⭐ {rd_nm} (co-crystal ref)",
                            "ref_name":    rd_nm,
                            "SMILES":      rd_smi,
                            "prot_smiles": rd_prep["prot_smiles"],
                            "Charge":      rd_prep["charge"],
                            "Top Score":   redock_score,
                            "pose_scores": [s["affinity"] for s in rd_dock["scores"]],
                            "Poses":       rd_n,
                            "out_pdbqt":   rd_dock["out_pdbqt"],
                            "out_sdf":     rd_dock["out_sdf"],
                            "pv_sdf":      rd_pv_sdf,
                            "Status":      "OK",
                            "is_redock":   True,
                        }
                        st.success(
                            f"✓ Reference score: **{redock_score:.2f} kcal/mol** ({rd_nm})"
                        )
                    else:
                        st.warning("⚠ Redocking failed — no score returned")
                else:
                    st.warning(f"⚠ Reference ligand prep failed: {rd_prep.get('error')}")

        # ── Main ligand docking ───────────────────────────────────────────
        with st.spinner(f"Running Vina (exhaustiveness={exh})… ⏳"):
            dock = run_vina(
                receptor_pdbqt = st.session_state.receptor_pdbqt,
                ligand_pdbqt   = st.session_state.ligand_pdbqt,
                config_txt     = st.session_state.config_txt,
                vina_path      = VINA_PATH,
                exhaustiveness = exh,
                n_modes        = nm,
                energy_range   = er,
                wdir           = WORKDIR,
                out_name       = base,
            )

        if not dock["success"]:
            st.error(f"❌ Vina failed: {dock['error']}\n{dock['log'][:400]}")
            st.session_state.docking_done = False
        else:
            pv_log = fix_sdf_bond_orders(
                dock["out_sdf"], st.session_state.prot_smiles, pv_sdf
            )
            if not os.path.exists(pv_sdf) or os.path.getsize(pv_sdf) < 10:
                pv_sdf = dock["out_sdf"]

            df = (
                pd.DataFrame([{
                    "Pose":                  s["pose"],
                    "Affinity (kcal/mol)":   s["affinity"],
                    "RMSD lb":               s["rmsd_lb"],
                    "RMSD ub":               s["rmsd_ub"],
                } for s in dock["scores"]])
                .sort_values("Affinity (kcal/mol)")
                .reset_index(drop=True)
            ) if dock["scores"] else None

            mols = (
                load_mols_from_sdf(dock["out_sdf"], sanitize=False)
                if os.path.exists(dock["out_sdf"]) else []
            )

            _full_log = (
                dock["log"] + "\n\n── Bond-order fix ──\n" + "\n".join(pv_log)
            )
            st.session_state.update({
                "output_pdbqt":  dock["out_pdbqt"],
                "output_sdf":    dock["out_sdf"],
                "output_pv_sdf": pv_sdf,
                "dock_base":     base,
                "docking_done":  True,
                "docking_log":   _full_log,
                "score_df":      df,
                "pose_mols":     mols,
                "pv_image_png":  None,
                "pv_image_svg":  None,
                "pv_pose_key":   None,
                "redock_done":          redock_result is not None,
                "redock_score":         redock_score,
                "redock_result":        redock_result,
                "confirmed_ref_score":  None,
                "confirmed_ref_pose":   None,
                "confirmed_ref_name":   None,
            })

    if st.session_state.docking_done:
        _redock_score = st.session_state.get("redock_score")
        st.markdown(
            _pill("Docking complete", "success")
            + (_pill(f"Ref: {_redock_score:.2f} kcal/mol", "warn")
               if _redock_score is not None else ""),
            unsafe_allow_html=True,
        )
        with st.expander("📋 Vina output log", expanded=False):
            st.markdown(
                f'<div class="log-box">{st.session_state.docking_log}</div>',
                unsafe_allow_html=True,
            )
        if st.session_state.score_df is not None:
            best = st.session_state.score_df["Affinity (kcal/mol)"].min()
            cls  = (
                "Very strong" if best < -11 else
                "Strong"      if best < -9  else
                "Moderate"    if best < -7  else "Weak"
            )
            st.markdown(
                f'<div class="score-best">{best:.2f} '
                f'<span class="score-unit">kcal/mol</span></div>'
                f'<div style="color:#8b949e;font-size:0.9rem;margin-bottom:12px">'
                f'Best pose — {cls} predicted binding</div>',
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<hr class="step-divider">', unsafe_allow_html=True)

    # ── Step 4: Results ───────────────────────────────────────────────────────
    card_cls = "step-card done" if st.session_state.docking_done else "step-card"
    st.markdown(
        f'<div class="{card_cls}"><div class="step-title">Step 4 of 4</div>'
        f'<div class="step-heading">📊 Results & Visualization</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.docking_done:
        st.info("Complete Step 3 to see results here.")
    else:
        import py3Dmol
        from rdkit import Chem

        df   = st.session_state.score_df
        mols = st.session_state.pose_mols or []

        ct, cc = st.columns([1, 1.4])
        with ct:
            st.markdown("**Score Table**")
            if df is not None:
                st.dataframe(
                    df.style.background_gradient(
                        cmap="RdYlGn",
                        subset=["Affinity (kcal/mol)"],
                        gmap=-df["Affinity (kcal/mol)"],
                    ),
                    hide_index=True, width='stretch',
                )
        with cc:
            st.markdown("**Affinity by Pose**")
            if df is not None:
                _cc = _chart_colors()
                fig, ax = plt.subplots(figsize=(6, 3.5))
                fig.patch.set_facecolor(_cc["bg"])
                ax.set_facecolor(_cc["bg_sub"])
                cols = [
                    "#3fb950" if v == df["Affinity (kcal/mol)"].min() else "#58a6ff"
                    for v in df["Affinity (kcal/mol)"]
                ]
                ax.bar(
                    df["Pose"].astype(str), df["Affinity (kcal/mol)"],
                    color=cols, edgecolor=_cc["border"], linewidth=0.6,
                )
                ax.invert_yaxis()
                _ref_score_plot = st.session_state.get("confirmed_ref_score")
                if _ref_score_plot is None:
                    _ref_score_plot = st.session_state.get("redock_score")
                if _ref_score_plot is not None:
                    _ref_nm  = st.session_state.get("confirmed_ref_name") or "co-crystal ref"
                    _ref_lbl = f"{_ref_nm}: {_ref_score_plot:.2f}"
                    ax.axhline(
                        _ref_score_plot, color="#f85149", linewidth=1.8,
                        linestyle="--", label=_ref_lbl,
                    )
                    ax.legend(
                        facecolor=_cc["legend_bg"], edgecolor=_cc["border"],
                        labelcolor=_cc["text"], fontsize=8,
                    )
                ax.set_xlabel("Pose",                color=_cc["muted"], fontsize=9)
                ax.set_ylabel("Affinity (kcal/mol)", color=_cc["muted"], fontsize=9)
                ax.tick_params(colors=_cc["muted"], labelsize=8)
                for sp in ax.spines.values():
                    sp.set_edgecolor(_cc["border"])
                fig.tight_layout()
                st.pyplot(fig, width='stretch')
                plt.close(fig)

        st.markdown("---")

        # ── Redocking Reference Browser ───────────────────────────────────
        _redock_result = st.session_state.get("redock_result")
        _redock_score  = st.session_state.get("redock_score")
        if _redock_result and _redock_result.get("out_sdf") and os.path.exists(_redock_result["out_sdf"]):
            st.markdown("**⭐ Redocking Reference**")
            _rd_mols = load_mols_from_sdf(_redock_result["out_sdf"], sanitize=False)
            if _rd_mols:
                _rd_pose_i    = st.slider("Reference pose", 1, len(_rd_mols), 1, key="rd_pose_sel") - 1
                _rd_scores    = _redock_result.get("pose_scores", [])
                _rd_this_score = (
                    _rd_scores[_rd_pose_i]
                    if _rd_pose_i < len(_rd_scores)
                    else _redock_result.get("Top Score")
                )
                _rd_pills = _pill(f"Pose {_rd_pose_i+1}/{len(_rd_mols)}")
                _rsk      = "success" if (_rd_this_score is not None and _rd_this_score < -8) else "warn"
                _rd_pills += f" {_pill(f'{_rd_this_score:.2f} kcal/mol', _rsk)}" if _rd_this_score is not None else ""

                _cryst_pdb_rd = st.session_state.get("ligand_pdb_path") or ""
                if _cryst_pdb_rd and os.path.exists(_cryst_pdb_rd):
                    _rmsd_rd = calc_rmsd_heavy(_rd_mols[_rd_pose_i], _cryst_pdb_rd)
                    if _rmsd_rd is not None:
                        _rk = (
                            "success" if _rmsd_rd <= 2.0 else
                            "warn"    if _rmsd_rd <= 3.0 else "info"
                        )
                        _rd_pills += f" {_pill(f'RMSD {_rmsd_rd:.2f} A vs crystal', _rk)}"

                st.markdown(
                    _pill("⭐ Co-crystal reference ligand", "warn") + " " + _rd_pills,
                    unsafe_allow_html=True,
                )

                _rd_v_col, _rd_a_col = st.columns([3, 1])
                with _rd_v_col:
                    try:
                        _vrd = py3Dmol.view(width="100%", height=400)
                        _vrd.setBackgroundColor(_viewer_bg())
                        _mrd = 0
                        if st.session_state.receptor_fh and os.path.exists(st.session_state.receptor_fh):
                            _vrd.addModel(open(st.session_state.receptor_fh).read(), "pdb")
                            _vrd.setStyle({"model": _mrd}, {
                                "cartoon": {"color": "spectrum", "opacity": 0.7},
                                "stick":   {"radius": 0.08, "opacity": 0.15},
                            })
                            _mrd += 1
                        _lig_p_rd = st.session_state.get("ligand_pdb_path")
                        if _lig_p_rd and os.path.exists(_lig_p_rd):
                            _vrd.addModel(open(_lig_p_rd).read(), "pdb")
                            _vrd.setStyle({"model": _mrd}, {
                                "stick": {"colorscheme": "magentaCarbon", "radius": 0.2}
                            })
                            _mrd += 1
                        _vrd.addModel(Chem.MolToMolBlock(_rd_mols[_rd_pose_i]), "mol")
                        _vrd.setStyle({"model": _mrd}, {
                            "stick": {"colorscheme": "cyanCarbon", "radius": 0.28}
                        })
                        _vrd.addSurface(
                            "SES", {"opacity": 0.2, "color": "lightblue"},
                            {"model": 0}, {"model": _mrd},
                        )
                        _vrd.zoomTo()
                        _vrd.center({"model": _mrd})
                        show3d(_vrd, height=400)
                    except Exception as _e:
                        st.info(f"Viewer error: {_e}")

                with _rd_a_col:
                    st.markdown("**Actions**")
                    _c_ref_score = st.session_state.get("confirmed_ref_score")
                    _c_ref_pose  = st.session_state.get("confirmed_ref_pose")
                    if _rd_this_score is not None:
                        _already = (_c_ref_score == _rd_this_score and _c_ref_pose == _rd_pose_i + 1)
                        if st.button(
                            f"✅ Confirmed (pose {_rd_pose_i+1})" if _already
                            else f"📌 Use pose {_rd_pose_i+1} as reference",
                            key="confirm_ref_btn",
                            type="secondary" if _already else "primary",
                            width='stretch',
                        ):
                            _rd_nm = _redock_result.get("ref_name", "co-crystal ref")
                            st.session_state.update({
                                "confirmed_ref_score": _rd_this_score,
                                "confirmed_ref_pose":  _rd_pose_i + 1,
                                "confirmed_ref_name":  _rd_nm,
                            })
                            st.rerun()
                        if _c_ref_score is not None and not _already:
                            if st.button(
                                "🔄 Reset reference",
                                key="reset_ref_btn",
                                width='stretch',
                            ):
                                st.session_state.update({
                                    "confirmed_ref_score": None,
                                    "confirmed_ref_pose":  None,
                                    "confirmed_ref_name":  None,
                                })
                                st.rerun()
                    st.markdown("**Download**")
                    _rd_safe = _redock_result.get("ref_name", "redock")
                    _sp_rd   = str(WORKDIR / f"redock_pose{_rd_pose_i+1}.sdf")
                    write_single_pose(_rd_mols[_rd_pose_i], _sp_rd)
                    st.download_button(
                        f"⬇ Ref pose {_rd_pose_i+1} (.sdf)",
                        open(_sp_rd, "rb"),
                        file_name=f"redock_{_rd_safe}_pose{_rd_pose_i+1}.sdf",
                        key="dl_rd_pose",
                        width='stretch',
                    )
                    if _redock_result.get("out_pdbqt") and os.path.exists(_redock_result["out_pdbqt"]):
                        st.download_button(
                            "⬇ All ref poses (.pdbqt)",
                            open(_redock_result["out_pdbqt"], "rb"),
                            file_name=f"redock_{_rd_safe}_out.pdbqt",
                            key="dl_rd_pdbqt",
                            width='stretch',
                        )

            st.markdown("---")

        st.markdown("**🎬 Animated Pose Viewer**")
        anim_spd = st.slider("Interval (ms)", 500, 3000, 1500, 250, key="anim_spd")
        if st.session_state.output_sdf and os.path.exists(st.session_state.output_sdf):
            sdf_txt = open(st.session_state.output_sdf).read()
            va = py3Dmol.view(width="100%", height=440)
            va.setBackgroundColor(_viewer_bg())
            mai = 0
            if st.session_state.receptor_fh and os.path.exists(st.session_state.receptor_fh):
                va.addModel(open(st.session_state.receptor_fh).read(), "pdb")
                va.setStyle({"model": mai}, {
                    "cartoon": {"color": "spectrum", "opacity": 0.7},
                    "stick":   {"radius": 0.1, "opacity": 0.2},
                })
                mai += 1
            if st.session_state.ligand_pdb_path and os.path.exists(st.session_state.ligand_pdb_path):
                va.addModel(open(st.session_state.ligand_pdb_path).read(), "pdb")
                va.setStyle({"model": mai}, {
                    "stick": {"colorscheme": "magentaCarbon", "radius": 0.22}
                })
                mai += 1
            va.addModelsAsFrames(sdf_txt)
            va.setStyle({"model": mai}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.25}})
            va.animate({"interval": anim_spd, "loop": "forward"})
            va.addSurface("SES", {"opacity": 0.18, "color": "lightblue"}, {"model": 0}, {"model": mai})
            va.zoomTo()
            va.center({"model": mai})
            va.rotate(30)
            show3d(va, height=440)

        st.markdown("---")

        st.markdown("**🔎 Interactive Pose Selector**")
        if mols:
            pose_idx = st.slider("Select pose", 1, len(mols), 1, key="pose_sel") - 1
            sel_mol  = mols[pose_idx]

            _cryst_pdb = st.session_state.get("ligand_pdb_path") or ""
            if df is not None:
                row = df[df["Pose"] == pose_idx + 1]
                if len(row):
                    aff = row.iloc[0]["Affinity (kcal/mol)"]
                    _score_kind = "success" if aff < -8 else "warn"
                    _pose_pill  = _pill(f"Pose {pose_idx+1}/{len(mols)}")
                    _aff_pill   = _pill(f"Affinity: {aff:.2f} kcal/mol", _score_kind)
                    _pills_str  = f"{_pose_pill} {_aff_pill}"

                    if _cryst_pdb and os.path.exists(_cryst_pdb):
                        _rmsd = calc_rmsd_heavy(sel_mol, _cryst_pdb)
                        if _rmsd is not None:
                            _rk = (
                                "success" if _rmsd <= 2.0 else
                                "warn"    if _rmsd <= 3.0 else "info"
                            )
                            _pills_str += f" {_pill(f'RMSD {_rmsd:.2f} A vs crystal', _rk)}"
                    st.markdown(_pills_str, unsafe_allow_html=True)

            cpv, cdl = st.columns([3, 1])
            with cpv:
                try:
                    v2 = py3Dmol.view(width="100%", height=400)
                    v2.setBackgroundColor(_viewer_bg())
                    mi2 = 0
                    if st.session_state.receptor_fh and os.path.exists(st.session_state.receptor_fh):
                        v2.addModel(open(st.session_state.receptor_fh).read(), "pdb")
                        v2.setStyle({"model": mi2}, {
                            "cartoon": {"color": "spectrum", "opacity": 0.5},
                            "stick":   {"radius": 0.08, "opacity": 0.15},
                        })
                        mi2 += 1
                    if _cryst_pdb and os.path.exists(_cryst_pdb):
                        v2.addModel(open(_cryst_pdb).read(), "pdb")
                        v2.setStyle({"model": mi2}, {
                            "stick": {"colorscheme": "magentaCarbon", "radius": 0.2}
                        })
                        mi2 += 1
                    v2.addModel(Chem.MolToMolBlock(sel_mol), "mol")
                    v2.setStyle({"model": mi2}, {
                        "stick": {"colorscheme": "cyanCarbon", "radius": 0.28}
                    })
                    v2.addSurface(
                        "SES", {"opacity": 0.2, "color": "lightblue"},
                        {"model": 0}, {"model": mi2},
                    )
                    v2.zoomTo({"model": mi2})
                    show3d(v2, height=400)
                except Exception as e:
                    st.info(f"Viewer error: {e}")

            with cdl:
                st.markdown("**Download**")
                sp_raw = str(WORKDIR / f"pose_{pose_idx+1}_raw.sdf")
                write_single_pose(sel_mol, sp_raw)
                st.download_button(
                    f"⬇ Pose {pose_idx+1} (.sdf)",
                    open(sp_raw, "rb"),
                    file_name=f"pose_{pose_idx+1}.sdf",
                    key=f"dl_p_{pose_idx}",
                    width='stretch',
                )
                st.download_button(
                    "⬇ All poses (.pdbqt)",
                    open(st.session_state.output_pdbqt, "rb"),
                    file_name=f"{st.session_state.dock_base}_out.pdbqt",
                    key="dl_pdbqt",
                    width='stretch',
                )
                if df is not None:
                    st.download_button(
                        "⬇ Scores (.csv)",
                        df.to_csv(index=False).encode(),
                        file_name=f"{st.session_state.dock_base}_scores.csv",
                        mime="text/csv",
                        key="dl_csv",
                        width='stretch',
                    )
                if st.session_state.receptor_fh and os.path.exists(st.session_state.receptor_fh):
                    st.download_button(
                        "⬇ Receptor (.pdb)",
                        open(st.session_state.receptor_fh, "rb"),
                        file_name="receptor.pdb",
                        key="dl_rec",
                        width='stretch',
                    )

            st.markdown("---")
            st.markdown("**🔬 Binding Pocket View**")
            _bpl, _bpr = st.columns([2, 1])
            with _bpl:
                _cutoff = st.slider(
                    "Distance cutoff (A)", 2.5, 5.0, 3.5, 0.1, key="bp_cutoff"
                )
            with _bpr:
                _show_labels = st.checkbox(
                    "Show residue labels", value=True, key="bp_show_labels"
                )
                _show_surface = st.checkbox(
                    "Show protein surface", value=False, key="bp_show_surface"
                )

            try:
                vbp = py3Dmol.view(width="100%", height=440)
                vbp.setBackgroundColor(_viewer_bg())
                mbp = 0
                if st.session_state.receptor_fh and os.path.exists(st.session_state.receptor_fh):
                    vbp.addModel(open(st.session_state.receptor_fh).read(), "pdb")
                    vbp.setStyle({"model": mbp}, {
                        "cartoon": {"color": "spectrum", "opacity": 0.45}
                    })
                    if _show_surface:
                        vbp.addSurface(
                            py3Dmol.SAS,
                            {
                                "opacity": 0.55,
                                "color": "white",
                            },
                            {"model": mbp},
                        )
                    mbp += 1
                vbp.addModel(Chem.MolToMolBlock(sel_mol), "mol")
                _lig_m = mbp
                vbp.setStyle({"model": _lig_m}, {
                    "stick": {"colorscheme": "cyanCarbon", "radius": 0.30}
                })
                if st.session_state.receptor_fh and os.path.exists(st.session_state.receptor_fh):
                    _ir = get_interacting_residues(
                        st.session_state.receptor_fh, sel_mol, cutoff=_cutoff
                    )
                    for _rb in _ir:
                        vbp.setStyle(
                            {"model": 0, "chain": _rb["chain"], "resi": _rb["resi"]},
                            {"stick": {"colorscheme": "orangeCarbon", "radius": 0.20}},
                        )
                        if _show_labels:
                            vbp.addLabel(
                                f"{_rb['resn']}{_rb['resi']}",
                                {
                                    "fontSize": 11, "fontColor": "yellow",
                                    "backgroundColor": "black",
                                    "backgroundOpacity": 0.65,
                                    "inFront": True, "showBackground": True,
                                },
                                {"model": 0, "chain": _rb["chain"], "resi": _rb["resi"]},
                            )
                    _n         = len(_ir)
                    _res_label = f"{_n} residue" + ("s" if _n != 1 else "")
                    _res_kind  = "success" if _n else "warn"
                    st.markdown(
                        f"{_pill(f'Pose {pose_idx+1}')}"
                        f" {_pill(f'{_cutoff:.1f} A cutoff')}"
                        f" {_pill(_res_label, _res_kind)}"
                        + (f" {_pill('surface on', 'info')}" if _show_surface else ""),
                        unsafe_allow_html=True,
                    )
                vbp.zoomTo({"model": _lig_m})
                show3d(vbp, height=440)
            except Exception as _e:
                st.info(f"Binding pocket viewer error: {_e}")

            # PoseView 2D
            pv_sdf_all = st.session_state.get("output_pv_sdf", "")
            sp_pv = str(WORKDIR / f"pose_{pose_idx+1}_pv_ready.sdf")
            if pv_sdf_all and os.path.exists(pv_sdf_all):
                pv_mols = load_mols_from_sdf(pv_sdf_all)
                write_single_pose(
                    pv_mols[pose_idx] if pose_idx < len(pv_mols) else sel_mol,
                    sp_pv,
                )
            else:
                write_single_pose(sel_mol, sp_pv)

            _poseview_ui(
                rec_key             = "receptor_fh",
                pose_sdf_path       = sp_pv,
                pdb_id              = st.session_state.get("pdb_token", ""),
                cocrystal_ligand_id = st.session_state.get("cocrystal_ligand_id", ""),
                smiles_key          = "ligand_name",
                pose_idx            = pose_idx,
                img_png_key         = "pv_image_png",
                img_svg_key         = "pv_image_svg",
                pose_key_key        = "pv_pose_key",
                btn_key             = "btn_pv_basic",
                dl_png_key          = "dl_pv_png_basic",
                dl_svg_key          = "dl_pv_svg_basic",
                ref_png_key         = "pv_ref_png",
                ref_svg_key         = "pv_ref_svg",
                label_suffix        = "_basic",
                lig_name            = st.session_state.get("ligand_name", ""),
                lig_smiles          = st.session_state.get("prot_smiles", ""),
                binding_energy      = (
                    float(df[df["Pose"] == pose_idx+1]["Affinity (kcal/mol)"].iloc[0])
                    if df is not None and len(df[df["Pose"] == pose_idx+1]) > 0
                    else None
                ),
                ref_lig_name   = (
                    st.session_state.get("redock_result", {}).get("ref_name", "")
                    if st.session_state.get("redock_result") else ""
                ),
                ref_lig_smiles = (
                    (st.session_state.get("redock_result", {}).get("prot_smiles")
                     or st.session_state.get("redock_result", {}).get("SMILES", ""))
                    if st.session_state.get("redock_result") else ""
                ),
                ref_lig_energy = (
                    st.session_state.get("redock_result", {}).get("Top Score")
                    if st.session_state.get("redock_result") else None
                ),
            )


            # ── 📊 Summary Dashboard ──────────────────────────────────────────
            with st.expander("📊 Summary Dashboard", expanded=False):
                st.caption(
                    "Click **📸 PNG** or **🎨 SVG** to capture both panels. "
                    "Then copy the AI prompt to generate a figure caption."
                )

                # ── 💾 Capture figure + 🤖 AI prompt ────────────────────

                import io as _io_cap2, base64 as _b64cap2, re as _re_cap3

                _sv_acd  = st.session_state.get("pv_image_svg_new")
                _sv_rdk  = st.session_state.get("pv_image_svg_rdk")
                _sv_use  = _sv_acd or _sv_rdk
                _sv_src  = "Anyone Can Dock" if _sv_acd else ("RDKit" if _sv_rdk else "")
                _sv_b    = (_sv_use if isinstance(_sv_use, bytes)
                            else _sv_use.encode() if _sv_use else None)
                _fn_stem = (
                    "Pose" + str(pose_idx+1) + "_"
                    + st.session_state.get("ligand_name", "Ligand")
                    + "_Anyone_Can_Dock"
                )

                # 3D pocket HTML
                _c3d = ""
                try:
                    from rdkit import Chem as _Ch3
                    _r3 = st.session_state.get("receptor_fh", "")
                    _v3 = py3Dmol.view(width=600, height=420)
                    _v3.setBackgroundColor("white")
                    _m3 = 0
                    if _r3 and os.path.exists(_r3):
                        _v3.addModel(open(_r3).read(), "pdb")
                        _v3.setStyle({"model": _m3},
                                     {"cartoon": {"color": "spectrum", "opacity": 0.25}})
                        _m3 += 1
                    _v3.addModel(_Ch3.MolToMolBlock(sel_mol), "mol")
                    _l3 = _m3
                    _v3.setStyle({"model": _l3},
                                 {"stick": {"colorscheme": "cyanCarbon", "radius": 0.25}})
                    if _r3 and os.path.exists(_r3):
                        for _rb in get_interacting_residues(_r3, sel_mol, cutoff=4.5):
                            _v3.setStyle(
                                {"model": 0, "chain": _rb["chain"], "resi": _rb["resi"]},
                                {"stick": {"colorscheme": "orangeCarbon", "radius": 0.16}})
                            _v3.addLabel(
                                _rb["resn"] + str(_rb["resi"]),
                                {"fontSize": 8, "fontColor": "yellow",
                                 "backgroundColor": "black", "backgroundOpacity": 0.6,
                                 "inFront": True, "showBackground": True},
                                {"model": 0, "chain": _rb["chain"], "resi": _rb["resi"]})
                    _v3.zoomTo({"model": _l3})
                    _raw3 = _v3._make_html()
                    _c3d  = _re_cap3.sub(
                        r'(width|height)\s*[:=]\s*["\']?\d+px?["\']?',
                        lambda m: m.group(1) + ":100%", _raw3)
                except Exception as _e3:
                    _c3d = "<p style='padding:20px;color:#aaa'>3D: " + str(_e3) + "</p>"

                # 2D diagram → base64 PNG
                _d64 = ""
                if _sv_b:
                    try:
                        import cairosvg as _csv3
                        _d64 = _b64cap2.b64encode(
                            _csv3.svg2png(bytestring=_sv_b, dpi=300)).decode()
                    except Exception:
                        pass

                # inline SVG fallback
                _dinl = ""
                if _sv_b and not _d64:
                    _dinl = _sv_b.decode("utf-8", errors="replace").replace(
                        "<svg ", '<svg style="width:100%;height:auto;" ', 1)

                _ctitle = _fn_stem.replace("_", " \u00b7 ")
                _blbl   = "(b) 2D Diagram" + (" \u00b7 " + _sv_src if _sv_src else "")
                _panb   = (
                    "<img src='data:image/png;base64," + _d64 + "' />"
                    if _d64 else
                    "<div class='dsvg'>" + _dinl + "</div>"
                    if _dinl else
                    "<div class='nodiag'>No 2D diagram yet.<br>Generate above first.</div>"
                )

                _JS_PNG = (
                    "function capPNG(){"
                    "var s=document.getElementById('st');"
                    "s.textContent='Rendering\u2026 please wait 2.5s';"
                    "setTimeout(function(){"
                    "html2canvas(document.getElementById('cr'),{"
                    "backgroundColor:'#ffffff',scale:3,"
                    "useCORS:true,logging:false,allowTaint:true"
                    "}).then(function(c){"
                    "var a=document.createElement('a');"
                    "a.download=FN+'.png';"
                    "a.href=c.toDataURL('image/png');"
                    "a.click();"
                    "s.textContent='\u2705 PNG downloaded!';"
                    "}).catch(function(e){"
                    "s.textContent='\u274c '+e.message;});"
                    "},1500);}"
                )
                _JS_SVG = (
                    "function capSVG(){"
                    "var s=document.getElementById('st');"
                    "s.textContent='Building SVG\u2026 please wait 2.5s';"
                    "setTimeout(function(){"
                    "html2canvas(document.getElementById('cr'),{"
                    "backgroundColor:'#ffffff',scale:3,"
                    "useCORS:true,logging:false,allowTaint:true"
                    "}).then(function(c){"
                    "var w=c.width,h=c.height;"
                    "var img=c.toDataURL('image/png');"
                    "var svgNS='http://www.w3.org/2000/svg';"
                    "var root=document.createElementNS(svgNS,'svg');"
                    "root.setAttribute('xmlns',svgNS);"
                    "root.setAttribute('width',w);"
                    "root.setAttribute('height',h);"
                    "var im=document.createElementNS(svgNS,'image');"
                    "im.setAttribute('href',img);"
                    "im.setAttribute('width',w);"
                    "im.setAttribute('height',h);"
                    "root.appendChild(im);"
                    "var blob=new Blob([new XMLSerializer().serializeToString(root)],"
                    "{type:'image/svg+xml'});"
                    "var a=document.createElement('a');"
                    "a.download=FN+'.svg';"
                    "a.href=URL.createObjectURL(blob);"
                    "a.click();"
                    "s.textContent='\u2705 SVG downloaded!';"
                    "}).catch(function(e){"
                    "s.textContent='\u274c '+e.message;});"
                    "},1500);}"
                )

                _capture_html = (
                    "<!DOCTYPE html><html><head>"
                    "<script src='https://cdnjs.cloudflare.com/ajax/libs/"
                    "html2canvas/1.4.1/html2canvas.min.js'></script>"
                    "<style>"
                    "*{margin:0;padding:0;box-sizing:border-box;"
                    "font-family:'Helvetica Neue',Arial,sans-serif;}"
                    "body{background:#fff;padding:10px;}"
                    ".title{text-align:center;font-size:14px;font-weight:700;"
                    "color:#111;margin-bottom:8px;}"
                    ".grid{display:grid;grid-template-columns:1fr 1fr;"
                    "gap:8px;align-items:stretch;}"
                    ".p3d{border:1px solid #e0e0e0;border-radius:8px;"
                    "overflow:hidden;background:#fff;position:relative;height:420px;}"
                    ".p3d>div,.p3d iframe{width:100%!important;height:100%!important;}"
                    ".pim{border:1px solid #e0e0e0;border-radius:8px;"
                    "overflow:hidden;background:#fff;position:relative;"
                    "height:420px;display:flex;align-items:center;justify-content:center;}"
                    ".pim img,.dsvg{width:100%;height:100%;object-fit:contain;display:block;}"
                    ".lbl{position:absolute;top:5px;left:7px;font-size:12px;"
                    "font-weight:700;color:#111;background:rgba(255,255,255,.88);"
                    "padding:1px 4px;border-radius:3px;z-index:10;}"
                    ".nodiag{display:flex;align-items:center;"
                    "justify-content:center;min-height:160px;color:#aaa;"
                    "font-size:12px;text-align:center;padding:16px;}"
                    ".row{display:flex;gap:8px;margin-top:10px;}"
                    ".btn{flex:1;padding:9px 0;border:none;border-radius:7px;"
                    "font-size:13px;font-weight:600;cursor:pointer;color:#fff;}"
                    ".bp{background:linear-gradient(90deg,#ff4b4b,#cc44cc);}"
                    ".bs{background:linear-gradient(90deg,#4b8bff,#cc44cc);}"
                    "#st{text-align:center;font-size:11px;color:#888;"
                    "margin-top:5px;min-height:16px;}"
                    "</style></head><body>"
                    "<div id='cr'>"
                    "  <div class='title'>" + _ctitle + "</div>"
                    "  <div class='grid'>"
                    "    <div class='p3d'>"
                    "      <span class='lbl'>(a) Binding Pocket</span>"
                    + _c3d +
                    "    </div>"
                    "    <div class='pim'>"
                    "      <span class='lbl'>" + _blbl + "</span>"
                    + _panb +
                    "    </div>"
                    "  </div>"
                    "</div>"
                    "<div class='row'>"
                    "  <button class='btn bp' onclick='capPNG()'>"
                    "    \U0001f4f8 Capture &amp; Download PNG"
                    "  </button>"
                    "  <button class='btn bs' onclick='capSVG()'>"
                    "    \U0001f3a8 Capture &amp; Download SVG"
                    "  </button>"
                    "</div>"
                    "<div id='st'></div>"
                    "<script>"
                    "var FN='" + _fn_stem + "_dashboard';"
                    + _JS_PNG + _JS_SVG +
                    "</script></body></html>"
                )
                components.html(_capture_html, height=720, scrolling=True)

                # ── 🤖 AI Prompt ──────────────────────────────────────────────
                st.markdown("---")
                st.markdown("#### \U0001f916 AI Prompt \u2014 Copy & Paste into Claude / ChatGPT")
                st.caption(
                    "Paste this prompt with your downloaded figure into any AI "
                    "to get a publication-ready figure caption and results paragraph."
                )

                _plig  = st.session_state.get("ligand_name", "the ligand")
                _ppdb  = st.session_state.get("pdb_token", "uploaded receptor")
                _pcid  = st.session_state.get("cocrystal_ligand_id", "")
                _prs   = (st.session_state.get("confirmed_ref_score")
                          or st.session_state.get("redock_score"))
                _prn   = (st.session_state.get("confirmed_ref_name")
                          or _pcid or "co-crystal reference")
                _prp   = st.session_state.get("confirmed_ref_pose", "")
                _psc, _pbest, _pnp = [], None, 0
                if df is not None:
                    _psc   = df["Affinity (kcal/mol)"].tolist()
                    _pbest = min(_psc)
                    _pnp   = len(_psc)

                _pres = []
                try:
                    _rpr = st.session_state.get("receptor_fh", "")
                    if _rpr and os.path.exists(_rpr):
                        _pres = [r["resn"] + str(r["resi"])
                                 for r in get_interacting_residues(
                                     _rpr, sel_mol, cutoff=4.5)]
                except Exception:
                    pass

                _pints = []
                try:
                    from core import _detect_all_interactions as _dai2
                    _rpr2 = st.session_state.get("receptor_fh", "")
                    if _rpr2 and os.path.exists(_rpr2):
                        for _it in _dai2(sel_mol, _rpr2):
                            _pints.append(
                                _it["itype"].replace("_", " ")
                                + " with " + _it["resname"] + str(_it["resid"])
                                + " (" + str(round(_it["distance"], 1)) + " \u00c5)"
                            )
                except Exception:
                    pass

                _pdiag  = ("Anyone Can Dock 2D diagram" if _sv_acd
                           else "RDKit 2D diagram" if _sv_rdk else "not generated")
                _scstr  = ", ".join(str(round(s, 2)) for s in _psc) if _psc else "N/A"
                _bstr   = str(round(_pbest, 2)) + " kcal/mol" if _pbest is not None else "N/A"
                _refstr = (
                    _prn
                    + (" (pose " + str(_prp) + ")" if _prp else "")
                    + (" = " + str(round(_prs, 2)) + " kcal/mol"
                       if _prs is not None else ": not available")
                )

                _ptxt = (
                    "I have a 2-panel molecular docking figure (attached):\n"
                    "(a) Binding pocket 3D view: orange sticks = interacting residues,"
                    " cyan sticks = docked ligand, protein shown as cartoon.\n"
                    "(b) 2D interaction diagram (" + _pdiag + "):"
                    " shows residues, interaction types, and binding energy label.\n\n"
                    "Docking data:\n"
                    "- Ligand: " + _plig + "\n"
                    "- Protein / PDB ID: " + _ppdb + "\n"
                    "- Docking software: AutoDock Vina 1.2.7\n"
                    "- Number of poses: " + str(_pnp) + "\n"
                    "- All pose scores (kcal/mol): " + _scstr + "\n"
                    "- Best pose score: " + _bstr + "\n"
                    "- Co-crystal reference: " + _refstr + "\n"
                    "- Residues within 4.5 \u00c5: "
                    + (", ".join(_pres) if _pres else "N/A") + "\n"
                    "- Detected interactions: "
                    + ("; ".join(_pints[:12]) if _pints else "N/A") + "\n\n"
                    "Please write:\n"
                    "1. A publication-ready FIGURE CAPTION for this 2-panel figure.\n"
                    "2. A RESULTS PARAGRAPH (3-5 sentences) covering: binding affinity,"
                    " comparison to reference, key residues, interaction types, and"
                    " whether this is a good or poor binder.\n\n"
                    "Use formal scientific English. Be specific with numbers."
                    " Do not invent data not listed above."
                )

                st.code(_ptxt, language="markdown")

                _esc = (
                    _ptxt
                    .replace("\\", "\\\\")
                    .replace("`",  "\\`")
                    .replace("$",  "\\$")
                    .replace("\n", "\\n")
                )
                _copy_html = (
                    "<html><body style='margin:0;padding:0;'>"
                    "<button onclick='cp()' style='padding:9px 20px;width:100%;"
                    "background:linear-gradient(90deg,#ff4b4b,#cc44cc);color:#fff;"
                    "border:none;border-radius:7px;font-size:13px;font-weight:600;"
                    "cursor:pointer;font-family:sans-serif;'>"
                    "\U0001f4cb Copy Prompt to Clipboard"
                    "</button>"
                    "<div id='m' style='text-align:center;font-size:11px;"
                    "color:#4caf50;margin-top:5px;height:15px;"
                    "font-family:sans-serif;'></div>"
                    "<script>"
                    "function cp(){const t=`" + _esc + "`;"
                    "navigator.clipboard.writeText(t)"
                    ".then(()=>{document.getElementById('m').textContent='\u2705 Copied!';"
                    "setTimeout(()=>document.getElementById('m').textContent='',2500);})"
                    ".catch(()=>{const a=document.createElement('textarea');a.value=t;"
                    "document.body.appendChild(a);a.select();document.execCommand('copy');"
                    "document.body.removeChild(a);"
                    "document.getElementById('m').textContent='\u2705 Copied!';"
                    "setTimeout(()=>document.getElementById('m').textContent='',2500);});}"
                    "</script></body></html>"
                )
                components.html(_copy_html, height=60)

    st.markdown('</div>', unsafe_allow_html=True)


# ╔════════════════════════════════════════════════════════════════════════════╗
#  TAB 2 — BATCH DOCKING
# ╚════════════════════════════════════════════════════════════════════════════╝
with tab_batch:

    _receptor_section(pfx="b_", wdir=BATCH_WORKDIR, step_label="Step B1 of B3")

    b_rec_done   = st.session_state.get("b_receptor_done", False)
    b_batch_done = st.session_state.get("b_batch_done", False)
    card_cls = "step-card done" if b_batch_done else "step-card"
    st.markdown(
        f'<div class="{card_cls}"><div class="step-title">Step B2 of B3</div>'
        f'<div class="step-heading">⚗️ Batch Ligand Input & Docking</div>',
        unsafe_allow_html=True,
    )

    col_b1, col_b2 = st.columns([1.6, 1])
    with col_b1:
        b_input_mode = st.radio(
            "Input mode", ["SMILES list (text)", "Upload .smi file"],
            key="b_input_mode",
        )
        if b_input_mode == "SMILES list (text)":
            st.text_area("One `SMILES [name]` per line",
                value=("C1=CC(=CC=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O Apigenin\n"
                       "C1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C(=C3O)O)O Baicalein\n"
                       "CC1=CC=C(C=C1)NC2=NC=NC3=C2C=C(C=C3)O Osimertinib\n"
                       "C1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C(=C3O)O)O Luteolin\n"
                       "CC(C)OC1=C(C=C2C(=C1)N=CN2)NC3=CC=CC(=C3)C#C Gefitinib\n"
                       "C1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C(=C3O)OC)O Kaempferol\n"
                       "CCOC1=CC=C(C=C1)NC2=NC=NC3=C2C=C(C=C3)F Lapatinib\n"
                       "CC1=CC=C(C=C1)NC2=NC=NC3=C2C=C(C=C3)Cl Afatinib\n"
                       "C1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C(=C3O)OC)O Galangin\n"
                       "CC1=C(C=C(C=C1)NC2=NC=NC3=C2C=CC=C3)OC Imatinib"),
                height=300, key="b_smiles_text")
        else:
            st.file_uploader("Upload .smi file", type=["smi", "txt"], key="b_smi_file")
        b_ph = st.number_input("Target pH", 0.0, 14.0, 7.4, 0.1, key="b_ph")

    with col_b2:
        st.markdown("**Redocking validation**")
        b_do_redock = st.checkbox(
            "Dock co-crystal ligand as reference", value=True, key="b_do_redock"
        )
        if b_do_redock:
            st.text_input(
                "Co-crystal SMILES [name]",
                value=(
                    "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC "
                    "Erlotinib"
                ),
                key="b_redock_smiles",
            )
            st.caption("Score shown as dashed reference line in plot.")
        st.markdown("**Docking parameters**")
        b_exh = st.slider("Exhaustiveness", 4, 32, 8, 2, key="b_exh")
        b_nm  = st.slider("Poses per ligand", 5, 20, 10, 1, key="b_nm")
        b_er  = st.slider("Energy range (kcal/mol)", 1, 5, 3, 1, key="b_er")

    if not b_rec_done:
        st.caption("⚠ Complete Step B1 first.")
    if st.button(
        "▶ Run Batch Docking", key="b_btn_dock", type="primary",
        disabled=not b_rec_done,
    ):
        rec_pdbqt = st.session_state.get("b_receptor_pdbqt")
        config    = st.session_state.get("b_config_txt")
        b_ph_val  = st.session_state.get("b_ph", 7.4)

        smiles_pairs = []
        try:
            if st.session_state.get("b_input_mode") == "SMILES list (text)":
                for line in st.session_state.get("b_smiles_text", "").strip().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    pts = line.split(None, 1)
                    _nm = pts[1].replace(" ", "_") if len(pts) > 1 else f"lig_{len(smiles_pairs)+1}"
                    smiles_pairs.append((pts[0], _nm))
            else:
                fobj = st.session_state.get("b_smi_file")
                if fobj is None:
                    raise ValueError("No .smi file uploaded")
                for line in fobj.read().decode().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    pts = line.split(None, 1)
                    _nm = pts[1].replace(" ", "_") if len(pts) > 1 else f"lig_{len(smiles_pairs)+1}"
                    smiles_pairs.append((pts[0], _nm))
            if not smiles_pairs:
                raise ValueError("No valid SMILES found")
        except Exception as e:
            st.error(f"❌ Input parsing failed: {e}")
            st.stop()

        redock_score  = None
        redock_result = None
        if st.session_state.get("b_do_redock"):
            raw_rd = st.session_state.get("b_redock_smiles", "").strip()
            pts    = raw_rd.split(None, 1)
            rd_smi = pts[0]
            rd_nm  = pts[1].replace(" ", "_") if len(pts) > 1 else "redock"
            with st.spinner(f"Docking reference ligand ({rd_nm})…"):
                rd_prep = prepare_ligand(rd_smi, "redock_" + rd_nm, b_ph_val, BATCH_WORKDIR)
                if rd_prep["success"]:
                    rd_dock = run_vina(
                        rec_pdbqt, rd_prep["pdbqt"], config,
                        VINA_PATH, b_exh, b_nm, b_er,
                        BATCH_WORKDIR, "redock_" + rd_nm,
                    )
                    if rd_dock["success"] and rd_dock["top_score"] is not None:
                        redock_score = rd_dock["top_score"]
                        rd_pv_sdf    = str(BATCH_WORKDIR / f"redock_{rd_nm}_pv_ready.sdf")
                        fix_sdf_bond_orders(rd_dock["out_sdf"], rd_smi, rd_pv_sdf)
                        if not os.path.exists(rd_pv_sdf) or os.path.getsize(rd_pv_sdf) < 10:
                            rd_pv_sdf = rd_dock["out_sdf"]
                        rd_n = (
                            len(load_mols_from_sdf(rd_dock["out_sdf"], sanitize=False))
                            if os.path.exists(rd_dock["out_sdf"]) else 0
                        )
                        redock_result = {
                            "Name":        f"⭐ {rd_nm} (co-crystal ref)",
                            "ref_name":    rd_nm,
                            "SMILES":      rd_smi,
                            "prot_smiles": rd_prep["prot_smiles"],
                            "Charge":      rd_prep["charge"],
                            "Top Score":   redock_score,
                            "pose_scores": [s["affinity"] for s in rd_dock["scores"]],
                            "Poses":       rd_n,
                            "out_pdbqt":   rd_dock["out_pdbqt"],
                            "out_sdf":     rd_dock["out_sdf"],
                            "pv_sdf":      rd_pv_sdf,
                            "Status":      "OK",
                            "is_redock":   True,
                        }
                        st.success(
                            f"✓ Reference score: **{redock_score:.2f} kcal/mol** ({rd_nm})"
                        )
                    else:
                        st.warning("⚠ Redocking failed — no score returned")
                else:
                    st.warning(f"⚠ Reference ligand prep failed: {rd_prep.get('error')}")

        results  = []
        n        = len(smiles_pairs)
        prog     = st.progress(0, text=f"Docking 0/{n}…")
        log_slot = st.empty()
        all_logs = []

        for i, (smi, name) in enumerate(smiles_pairs):
            prog.progress(i / n, text=f"Docking {name} ({i+1}/{n})…")
            prep = prepare_ligand(smi, name, b_ph_val, BATCH_WORKDIR)
            if not prep["success"]:
                results.append({
                    "Name": name, "SMILES": smi, "Charge": None,
                    "Top Score": None,
                    "Status": f"PREP FAILED: {prep['error']}",
                })
                all_logs.append(f"[{name}] PREP ERROR: {prep['error']}")
                continue

            dock = run_vina(
                rec_pdbqt, prep["pdbqt"], config,
                VINA_PATH, b_exh, b_nm, b_er,
                BATCH_WORKDIR, name,
            )
            all_logs.append(
                f"[{name}] score={dock.get('top_score')} | "
                f"{dock.get('log','')[:100]}"
            )
            log_slot.markdown(
                f'<div class="log-box">{"".join(all_logs[-5:])}</div>',
                unsafe_allow_html=True,
            )

            if not dock["success"] or dock["top_score"] is None:
                results.append({
                    "Name": name, "SMILES": smi,
                    "Charge": prep["charge"],
                    "Top Score": None, "Status": "DOCK FAILED",
                })
                continue

            pv_sdf = str(BATCH_WORKDIR / f"{name}_pv_ready.sdf")
            fix_sdf_bond_orders(dock["out_sdf"], smi, pv_sdf)
            if not os.path.exists(pv_sdf) or os.path.getsize(pv_sdf) < 10:
                pv_sdf = dock["out_sdf"]

            n_poses = (
                len(load_mols_from_sdf(dock["out_sdf"], sanitize=False))
                if os.path.exists(dock["out_sdf"]) else 0
            )
            results.append({
                "Name":        name,
                "SMILES":      smi,
                "prot_smiles": prep["prot_smiles"],
                "Charge":      prep["charge"],
                "Top Score":   dock["top_score"],
                "pose_scores": [s["affinity"] for s in dock["scores"]],
                "Poses":       n_poses,
                "out_pdbqt":   dock["out_pdbqt"],
                "out_sdf":     dock["out_sdf"],
                "pv_sdf":      pv_sdf,
                "Status":      "OK",
            })

        n_ok = sum(1 for r in results if r["Status"] == "OK")
        prog.progress(1.0, text=f"Done — {n_ok}/{n} ligands docked successfully")
        log_slot.empty()

        st.session_state.update({
            "b_batch_done":          True,
            "b_batch_results":       results,
            "b_batch_log":           "\n".join(all_logs),
            "b_redock_score":        redock_score,
            "b_redock_result":       redock_result,
            "b_confirmed_ref_score": None,
            "b_confirmed_ref_pose":  None,
            "b_confirmed_ref_name":  None,
            "b_pv2_image_png":       None,
            "b_pv2_image_svg":       None,
            "b_pv2_pose_key":        None,
            "b_pv2_ref_png":         None,
            "b_pv2_ref_svg":         None,
            "b_plot_png":            None,
        })

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<hr class="step-divider">', unsafe_allow_html=True)

    # ── Step B3: Batch Results ────────────────────────────────────────────────
    b_batch_done = st.session_state.get("b_batch_done", False)
    card_cls = "step-card done" if b_batch_done else "step-card"
    st.markdown(
        f'<div class="{card_cls}"><div class="step-title">Step B3 of B3</div>'
        f'<div class="step-heading">📊 Batch Results</div>',
        unsafe_allow_html=True,
    )

    if not b_batch_done:
        st.info("Complete Step B2 to see batch results here.")
    else:
        import py3Dmol
        from rdkit import Chem

        results       = st.session_state.get("b_batch_results", [])
        redock_score  = st.session_state.get("b_redock_score")
        redock_result = st.session_state.get("b_redock_result")
        c_ref_score   = st.session_state.get("b_confirmed_ref_score")
        c_ref_pose    = st.session_state.get("b_confirmed_ref_pose")
        active_ref    = c_ref_score if c_ref_score is not None else redock_score

        n_ok   = sum(1 for r in results if r["Status"] == "OK")
        n_fail = len(results) - n_ok
        st.markdown(
            f"{_pill(f'{n_ok} ligands docked', 'success')}"
            f" {_pill('AutoDock Vina 1.2.7')}"
            + (f" {_pill(f'{n_fail} failed', 'warn')}" if n_fail else ""),
            unsafe_allow_html=True,
        )

        ok_results = [
            r for r in results
            if r["Status"] == "OK"
            and r.get("out_sdf") and os.path.exists(r["out_sdf"])
        ]
        browsable = (
            [redock_result]
            if redock_result and os.path.exists(redock_result.get("out_sdf", ""))
            else []
        ) + ok_results

        if browsable:
            st.markdown("**🔎 Pose Browser**")
            sel_nm = st.selectbox(
                "Select ligand",
                [r["Name"] for r in browsable],
                index=0, key="b_lig_sel",
            )
            sel_res       = next(r for r in browsable if r["Name"] == sel_nm)
            is_redock_sel = sel_res.get("is_redock", False)
            pose_scores_l = sel_res.get("pose_scores", [])

            b_mols = load_mols_from_sdf(sel_res["out_sdf"], sanitize=False)
            if b_mols:
                b_pose_i    = st.slider("Pose", 1, len(b_mols), 1, key="b_pose_sel") - 1
                this_score  = (
                    pose_scores_l[b_pose_i]
                    if b_pose_i < len(pose_scores_l)
                    else sel_res["Top Score"]
                )
                _score_kind = "success" if (this_score is not None and this_score < -8) else "warn"
                _pose_pill  = _pill(f"Pose {b_pose_i+1} / {len(b_mols)}")
                _score_pill = (
                    _pill(f"Score: {this_score:.2f} kcal/mol", _score_kind)
                    if this_score is not None else ""
                )
                row_pills = f"{_pose_pill} {_score_pill}"

                if pose_scores_l and b_pose_i > 0:
                    _delta = this_score - pose_scores_l[0]
                    _sign  = "+" if _delta > 0 else ""
                    row_pills += f" {_pill(f'delta {_sign}{_delta:.2f} vs pose 1')}"

                if is_redock_sel:
                    _cryst = st.session_state.get("b_ligand_pdb_path") or ""
                    if _cryst and os.path.exists(_cryst):
                        _rmsd = calc_rmsd_heavy(b_mols[b_pose_i], _cryst)
                        if _rmsd is not None:
                            _rk = (
                                "success" if _rmsd <= 2.0 else
                                "warn"    if _rmsd <= 3.0 else "info"
                            )
                            row_pills += f" {_pill(f'RMSD {_rmsd:.2f} A vs crystal', _rk)}"
                    st.markdown(_pill("⭐ Co-crystal reference ligand", "warn"), unsafe_allow_html=True)

                st.markdown(row_pills, unsafe_allow_html=True)

                cbv, cbd = st.columns([3, 1])
                with cbv:
                    try:
                        vb = py3Dmol.view(width="100%", height=420)
                        vb.setBackgroundColor(_viewer_bg())
                        bmi = 0
                        _rec_fh = st.session_state.get("b_receptor_fh")
                        if _rec_fh and os.path.exists(_rec_fh):
                            vb.addModel(open(_rec_fh).read(), "pdb")
                            vb.setStyle({"model": bmi}, {
                                "cartoon": {"color": "spectrum", "opacity": 0.7},
                                "stick":   {"radius": 0.08, "opacity": 0.15},
                            })
                            bmi += 1
                        _lig_p = st.session_state.get("b_ligand_pdb_path")
                        if _lig_p and os.path.exists(_lig_p):
                            vb.addModel(open(_lig_p).read(), "pdb")
                            vb.setStyle({"model": bmi}, {
                                "stick": {"colorscheme": "magentaCarbon", "radius": 0.2}
                            })
                            bmi += 1
                        vb.addModel(Chem.MolToMolBlock(b_mols[b_pose_i]), "mol")
                        vb.setStyle({"model": bmi}, {
                            "stick": {"colorscheme": "cyanCarbon", "radius": 0.28}
                        })
                        vb.addSurface(
                            "SES", {"opacity": 0.2, "color": "lightblue"},
                            {"model": 0}, {"model": bmi},
                        )
                        vb.zoomTo()
                        vb.center({"model": bmi})
                        show3d(vb, height=420)
                    except Exception as e:
                        st.info(f"Viewer error: {e}")

                with cbd:
                    st.markdown("**Actions**")
                    if is_redock_sel and this_score is not None:
                        already = (c_ref_score == this_score and c_ref_pose == b_pose_i + 1)
                        if st.button(
                            f"✅ Confirmed (pose {b_pose_i+1})" if already
                            else f"📌 Use pose {b_pose_i+1} as reference",
                            key="b_confirm_ref_btn",
                            type="secondary" if already else "primary",
                            width='stretch',
                        ):
                            st.session_state.update({
                                "b_confirmed_ref_score": this_score,
                                "b_confirmed_ref_pose":  b_pose_i + 1,
                                "b_confirmed_ref_name":  sel_nm,
                            })
                            st.rerun()
                        if c_ref_score is not None and not already:
                            if st.button(
                                "🔄 Reset reference",
                                key="b_reset_ref_btn",
                                width='stretch',
                            ):
                                st.session_state.update({
                                    "b_confirmed_ref_score": None,
                                    "b_confirmed_ref_pose":  None,
                                    "b_confirmed_ref_name":  None,
                                })
                                st.rerun()
                    st.markdown("**Download**")
                    safe_nm = sel_nm.replace("⭐ ", "").replace(" (co-crystal ref)", "")
                    sp3 = str(BATCH_WORKDIR / f"{safe_nm}_pose{b_pose_i+1}.sdf")
                    write_single_pose(b_mols[b_pose_i], sp3)
                    st.download_button(
                        f"⬇ Pose {b_pose_i+1} (.sdf)",
                        open(sp3, "rb"),
                        file_name=f"{safe_nm}_pose{b_pose_i+1}.sdf",
                        key="b_dl_pose",
                        width='stretch',
                    )
                    if sel_res.get("out_pdbqt") and os.path.exists(sel_res["out_pdbqt"]):
                        st.download_button(
                            "⬇ All poses (.pdbqt)",
                            open(sel_res["out_pdbqt"], "rb"),
                            file_name=f"{safe_nm}_out.pdbqt",
                            key="b_dl_pdbqt",
                            width='stretch',
                        )

        st.markdown("---")
        with st.expander("📋 Full docking log", expanded=False):
            st.markdown(
                f'<div class="log-box">'
                f'{st.session_state.get("b_batch_log", "")}'
                f'</div>',
                unsafe_allow_html=True,
            )

        df_res = pd.DataFrame([{
            "Name":                  r["Name"],
            "Top Score (kcal/mol)":  r["Top Score"],
            "Charge":                f"{r['Charge']:+d}" if r.get("Charge") is not None else "—",
            "Status":                r["Status"],
        } for r in results])
        ok_df   = (
            df_res[df_res["Status"] == "OK"]
            .sort_values("Top Score (kcal/mol)")
            .reset_index(drop=True)
        )
        plot_df = df_res[df_res["Status"] == "OK"].reset_index(drop=True)

        if not plot_df.empty:
            _n    = len(plot_df)
            _best = ok_df["Top Score (kcal/mol)"].min()

            def _draw_plot(ax):
                _cc = _chart_colors()
                ax.get_figure().patch.set_facecolor(_cc["bg"])
                ax.set_facecolor(_cc["bg_sub"])
                scores = plot_df["Top Score (kcal/mol)"].values
                xs     = list(range(_n))
                colors = ["#3fb950" if s == _best else "#58a6ff" for s in scores]
                ax.scatter(
                    xs, scores, color=colors, s=90, zorder=3,
                    edgecolors=_cc["border"], linewidths=0.5,
                )
                ax.plot(xs, scores, color=_cc["border"], linewidth=0.8, zorder=2)
                ax.set_xticks(xs)
                ax.set_xticklabels(plot_df["Name"].values, rotation=40, ha="right")
                ax.set_xlim(-0.5, _n - 0.5)
                if active_ref is not None:
                    _ref_lbl = (
                        f"Confirmed ref (pose {c_ref_pose}): {active_ref:.2f} kcal/mol"
                        if c_ref_score is not None
                        else f"Co-crystal ref: {active_ref:.2f} kcal/mol"
                    )
                    ax.axhline(
                        active_ref, color="#f85149", linewidth=1.8,
                        linestyle="--", label=_ref_lbl,
                    )
                    ax.legend(
                        facecolor=_cc["legend_bg"], edgecolor=_cc["border"],
                        labelcolor=_cc["text"], fontsize=8,
                    )
                ax.set_ylabel("Vina score (kcal/mol)", color=_cc["muted"], fontsize=9)
                ax.set_xlabel("Ligand",                color=_cc["muted"], fontsize=9)
                ax.tick_params(colors=_cc["muted"], labelsize=7)
                for sp in ax.spines.values():
                    sp.set_edgecolor(_cc["border"])

            if _n <= 10:
                ct2, cp2 = st.columns([1, 1.6])
                with ct2:
                    st.markdown("**Score Table**")
                    st.dataframe(df_res, hide_index=True, width='stretch')
                with cp2:
                    st.markdown("**Top Score per Ligand**")
                    fig, ax = plt.subplots(figsize=(max(5, _n * 0.6 + 1.5), 3.5))
                    _draw_plot(ax)
                    fig.tight_layout()
                    _buf = io.BytesIO()
                    fig.savefig(
                        _buf, format="png", dpi=150,
                        bbox_inches="tight", facecolor=fig.get_facecolor(),
                    )
                    _buf.seek(0)
                    st.session_state["b_plot_png"] = _buf.getvalue()
                    st.pyplot(fig, width='stretch')
                    plt.close(fig)
            else:
                st.markdown("**Top Score per Ligand**")
                fig, ax = plt.subplots(figsize=(max(6, _n * 0.9 + 1.5), 4))
                _draw_plot(ax)
                fig.tight_layout()
                _buf = io.BytesIO()
                fig.savefig(
                    _buf, format="png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor(),
                )
                _buf.seek(0)
                st.session_state["b_plot_png"] = _buf.getvalue()
                st.pyplot(fig, width='stretch')
                plt.close(fig)
                st.markdown("**Score Table**")
                st.dataframe(df_res, hide_index=True, width='stretch')
        else:
            st.markdown("**Score Table**")
            st.dataframe(df_res, hide_index=True, width='stretch')


        # ── 📊 Summary Dashboard (Batch) ─────────────────────────────────────
        with st.expander("📊 Summary Dashboard", expanded=False):
            st.caption(
                "Score distribution across all ligands + 2D interaction diagram "
                "for the selected ligand. Export charts as PNG / SVG."
            )

            _bd_c1, _bd_c2 = st.columns(2)

            # ── Left: violin / box score distribution ─────────────────────────
            with _bd_c1:
                st.markdown("##### Score Distribution")
                if not plot_df.empty:
                    _bcc   = _chart_colors()
                    _scores = plot_df["Top Score (kcal/mol)"].values
                    _bfig, _bax = plt.subplots(figsize=(4, 3.5))
                    _bfig.patch.set_facecolor(_bcc["bg"])
                    _bax.set_facecolor(_bcc["bg_sub"])

                    if len(_scores) >= 4:
                        _vp = _bax.violinplot(
                            [_scores], positions=[0], showmedians=True,
                            showextrema=True,
                        )
                        for _pc in _vp["bodies"]:
                            _pc.set_facecolor("#58a6ff")
                            _pc.set_alpha(0.45)
                        for _part in ("cmedians","cmins","cmaxes","cbars"):
                            if _part in _vp:
                                _vp[_part].set_color(_bcc["muted"])
                                _vp[_part].set_linewidth(1.2)
                    import numpy as _np_bd
                    _jitter = _np_bd.random.default_rng(42).uniform(-0.08, 0.08, len(_scores))
                    _bax.scatter(
                        _jitter, _scores,
                        color=["#3fb950" if s == _scores.min() else "#58a6ff" for s in _scores],
                        s=40, zorder=3, edgecolors=_bcc["border"], linewidths=0.4,
                    )
                    if active_ref is not None:
                        _bax.axhline(
                            active_ref, color="#f85149", linewidth=1.6,
                            linestyle="--", label=f"Ref: {active_ref:.2f}",
                        )
                        _bax.legend(
                            facecolor=_bcc["legend_bg"], edgecolor=_bcc["border"],
                            labelcolor=_bcc["text"], fontsize=7,
                        )
                    _bax.invert_yaxis()
                    _bax.set_ylabel("Vina score (kcal/mol)", color=_bcc["muted"], fontsize=8)
                    _bax.set_xticks([])
                    _bax.tick_params(colors=_bcc["muted"], labelsize=7)
                    for _sp in _bax.spines.values():
                        _sp.set_edgecolor(_bcc["border"])
                    _bfig.tight_layout()
                    st.pyplot(_bfig, use_container_width=True)

                    import io as _iob
                    _bb_buf = _iob.BytesIO()
                    _bfig.savefig(
                        _bb_buf, format="png", dpi=200,
                        bbox_inches="tight", facecolor=_bfig.get_facecolor(),
                    )
                    _bb_buf.seek(0)
                    st.download_button(
                        "⬇ Distribution chart (PNG)", _bb_buf.getvalue(),
                        file_name="batch_score_distribution.png",
                        mime="image/png",
                        key="bdb_dl_dist_png",
                        use_container_width=True,
                    )
                    plt.close(_bfig)
                else:
                    st.info("No scores available.")

            # ── Right: 2D diagram for selected ligand ─────────────────────────
            with _bd_c2:
                st.markdown("##### 2D Interaction Diagram")
                st.caption("Select a ligand in the Pose Browser above, then generate its diagram.")
                _bdb_svg = st.session_state.get("b_pv_image_svg_new")
                if _bdb_svg:
                    import base64 as _b64b
                    _bdb_s = _bdb_svg.decode() if isinstance(_bdb_svg, bytes) else _bdb_svg
                    _bdb_s = _bdb_s.replace(
                        "<svg ", '<svg style="width:100%;height:auto;display:block;" ', 1
                    )
                    st.components.v1.html(
                        '<div style="background:#fff;border-radius:8px;border:1px solid #eee;">' +
                        _bdb_s + "</div>",
                        height=420, scrolling=False,
                    )
                    st.download_button(
                        "⬇ 2D Diagram (SVG)",
                        _bdb_svg if isinstance(_bdb_svg, bytes) else _bdb_svg.encode(),
                        file_name="batch_interaction_diagram.svg",
                        mime="image/svg+xml",
                        key="bdb_dl_diag_svg",
                        use_container_width=True,
                    )
                else:
                    st.info(
                        "Generate the 2D diagram for this ligand first "
                        "(🧬 Anyone Can Dock 2D Diagram tab → Generate)."
                    )

        st.markdown("---")
        st.markdown("**⬇ Download All Results**")
        c_csv, c_zip = st.columns(2)
        with c_csv:
            if not ok_df.empty:
                st.download_button(
                    "⬇ Top scores (.csv)",
                    ok_df.to_csv(index=False).encode(),
                    file_name="batch_scores.csv",
                    mime="text/csv",
                    key="b_dl_csv",
                    width='stretch',
                )
        with c_zip:
            zb = io.BytesIO()
            with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as zf:
                for r in ([redock_result] if redock_result else []) + ok_results:
                    sn = r["Name"].replace("⭐ ", "").replace(" (co-crystal ref)", "")
                    for src_path, arc in [
                        (r.get("out_sdf"),   f"poses/{sn}_out.sdf"),
                        (r.get("pv_sdf"),    f"poses_pv_ready/{sn}_pv_ready.sdf"),
                        (r.get("out_pdbqt"), f"pdbqt/{sn}_out.pdbqt"),
                    ]:
                        if src_path and os.path.exists(src_path):
                            zf.write(src_path, arc)
                if not ok_df.empty:
                    zf.writestr("batch_scores.csv", ok_df.to_csv(index=False))
                _rec_fh = st.session_state.get("b_receptor_fh")
                if _rec_fh and os.path.exists(_rec_fh):
                    zf.write(_rec_fh, "receptor.pdb")
                if st.session_state.get("b_plot_png"):
                    zf.writestr("plots/batch_score_plot.png", st.session_state["b_plot_png"])
                for _sfx, _pk, _sk in [("poseview2", "b_pv2_image_png", "b_pv2_image_svg")]:
                    if st.session_state.get(_pk):
                        zf.writestr(f"poseview2/{_sfx}.png", st.session_state[_pk])
                    if st.session_state.get(_sk):
                        zf.writestr(f"poseview2/{_sfx}.svg", st.session_state[_sk])
            zb.seek(0)
            st.download_button(
                "⬇ Download ALL (.zip)", zb,
                file_name="anyone_can_dock.zip",
                mime="application/zip",
                key="b_dl_zip",
                width='stretch',
            )

        # 2D Interaction diagram
        st.markdown("---")
        st.markdown("### 🧬 2D Interaction Diagram")
        pv_browsable = [
            r for r in browsable
            if r.get("out_sdf") and os.path.exists(r["out_sdf"])
        ]
        if pv_browsable:
            pv_sel_nm   = st.selectbox(
                "Associate docked ligand (for AI prompt)",
                [r["Name"] for r in pv_browsable],
                index=0, key="b_pv_lig_sel",
            )
            pv_sel_res  = next(r for r in pv_browsable if r["Name"] == pv_sel_nm)
            pv_safe_nm  = pv_sel_nm.replace("⭐ ", "").replace(" (co-crystal ref)", "")
            pv_all_mols = load_mols_from_sdf(pv_sel_res["out_sdf"], sanitize=False)

            if pv_all_mols:
                pv_pose_i = (
                    st.slider(
                        "Pose (AI prompt context)",
                        1, len(pv_all_mols), 1,
                        key="b_pv_pose_sel",
                    ) - 1
                )
                pv_scores = pv_sel_res.get("pose_scores", [])
                pv_score  = (
                    pv_scores[pv_pose_i]
                    if pv_pose_i < len(pv_scores)
                    else pv_sel_res.get("Top Score")
                )

                st.session_state["_b_pv2_smiles"] = pv_sel_res.get("prot_smiles") or pv_sel_res.get("SMILES", pv_sel_nm)

                sp_pv2      = str(BATCH_WORKDIR / f"{pv_safe_nm}_pose{pv_pose_i+1}_pv2_ready.sdf")
                pv_all_path = pv_sel_res.get("pv_sdf", "")
                if pv_all_path and os.path.exists(pv_all_path):
                    pv_fixed = load_mols_from_sdf(pv_all_path)
                    write_single_pose(
                        pv_fixed[pv_pose_i] if pv_pose_i < len(pv_fixed) else pv_all_mols[pv_pose_i],
                        sp_pv2,
                    )
                else:
                    write_single_pose(pv_all_mols[pv_pose_i], sp_pv2)

                _poseview_ui(
                    rec_key             = "b_receptor_fh",
                    pose_sdf_path       = sp_pv2,
                    pdb_id              = st.session_state.get("b_pdb_token", ""),
                    cocrystal_ligand_id = st.session_state.get("b_cocrystal_ligand_id", ""),
                    smiles_key          = "_b_pv2_smiles",
                    pose_idx            = pv_pose_i,
                    img_png_key         = "b_pv2_image_png",
                    img_svg_key         = "b_pv2_image_svg",
                    pose_key_key        = "b_pv2_pose_key",
                    btn_key             = "btn_pv2_batch",
                    dl_png_key          = "dl_pv2_png_batch",
                    dl_svg_key          = "dl_pv2_svg_batch",
                    ref_png_key         = "b_pv2_ref_png",
                    ref_svg_key         = "b_pv2_ref_svg",
                    label_suffix        = f"_pv2_{pv_safe_nm}",
                    lig_name            = pv_safe_nm,
                    lig_smiles          = pv_sel_res.get("prot_smiles") or pv_sel_res.get("SMILES", ""),
                    binding_energy      = pv_score,
                    ref_lig_name        = redock_result.get("ref_name", "") if redock_result else "",
                    ref_lig_smiles      = (redock_result.get("prot_smiles") or redock_result.get("SMILES", "")) if redock_result else "",
                    ref_lig_energy      = redock_result.get("Top Score")     if redock_result else None,
                    show_header         = False,
                )

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="step-divider">', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;color:#57606A;font-size:0.78rem;'
    'font-family:\'IBM Plex Mono\',monospace;">'
    'AutoDock Vina 1.2.7 · Meeko · RDKit · OpenBabel · py3Dmol<br>'
    'Eberhardt et al. J. Chem. Inf. Model. 2021, 61, 3891&#8211;3898 &nbsp;·&nbsp; '
    '<a href="https://pubs.acs.org/doi/10.1021/acs.jcim.5c02852" target="_blank" '
    'style="color:#58a6ff;text-decoration:none;">'
    'DFDD &#8212; Hengphasatporn et al. J. Chem. Inf. Model. 2026</a>'
    '</div>',
    unsafe_allow_html=True,
)
