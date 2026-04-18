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
    prepare_ligand_from_file,
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

try:
    from core import COFACTOR_NAMES as _COFACTOR_NAMES
except ImportError:
    _COFACTOR_NAMES = {
        "ATP", "ADP", "AMP", "GTP", "GDP", "GMP",
        "NAD", "NAP", "NDP", "FAD", "FMN",
        "GOL", "PEG", "EDO", "MPD", "PGE", "PG4",
        "SO4", "PO4", "SUL", "PHO",
        "IHP", "TTP", "CTP", "UTP",
        "COA", "SAM", "SAH",
        "EPE", "MES", "TRS", "ACT", "ACY",
    }

try:
    from core import HEME_RESNAMES as _HEME_RESNAMES
except ImportError:
    _HEME_RESNAMES = {"HEM", "HEC", "HEA", "HEB", "HDD", "HDM"}

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

    title_esc = title.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")

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
  <svg id="iac-svg" viewBox="0 0 {W} {H}"
       style="width:100%;display:block;cursor:default;user-select:none;">
    <rect width="{W}" height="{H}" fill="white"/>
    <rect x="{pill_x:.1f}" y="12" width="{tw:.0f}" height="44"
          rx="22" ry="22" fill="#f2f2f2" stroke="none"/>
    <text x="{W/2:.1f}" y="34" text-anchor="middle" dominant-baseline="central"
          font-family="Arial,sans-serif" font-size="20" font-weight="700"
          fill="#1a1a1a">{title}</text>
    <g id="iac-lines"></g>
    <g id="iac-ligand" transform="translate(0,0)"
       style="cursor:move;" title="Drag to reposition ligand">{lig_svg}</g>
    <g id="iac-residues"></g>
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

  const pos = {{}};
  PLACEMENTS.forEach(p => {{ pos[p.id] = {{ x: p.bx, y: p.by }}; }});

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
      updateElement(p.id);
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

  // ── Ligand structure drag ──────────────────────────────────────────────────
  // The entire ligand group (#iac-ligand) can be moved as a whole.
  // Residue interaction lines originate from absolute SVG coordinates (lx/ly)
  // that don't change when the ligand moves — so we also shift the line origins
  // by storing a global ligand offset and applying it in updateElement.
  (function() {{
    const ligG = document.getElementById("iac-ligand");
    if (!ligG) return;

    let ligOffset = {{ x: 0, y: 0 }};   // running translate of the ligand group
    let dragActive = false;
    let startMouse = null;
    let startOffset = null;

    function applyOffset(ox, oy) {{
      ligG.setAttribute("transform", `translate(${{ox}},${{oy}})`);
      // Shift all residue line start-points by the same delta so lines track the atoms
      PLACEMENTS.forEach(p => {{
        const cache = els[p.id];
        if (cache && cache.line) {{
          cache.line.setAttribute("x1", p.lx + ox);
          cache.line.setAttribute("y1", p.ly + oy);
        }}
        if (cache && cache.distRect && cache.distTxt) {{
          // Recompute distance label position with shifted lx/ly
          const {{ x, y }} = pos[p.id];
          const lx2 = p.lx + ox, ly2 = p.ly + oy;
          const t  = 0.4;
          const mx = lx2 + (x - lx2) * t, my = ly2 + (y - ly2) * t;
          const dx = x - lx2, dy = y - ly2;
          const len = Math.sqrt(dx*dx + dy*dy) + 0.001;
          const px  = -dy/len*14, py = dx/len*14;
          cache.distRect.setAttribute("x", mx + px - cache.tw2/2);
          cache.distRect.setAttribute("y", my + py - 8);
          cache.distTxt.setAttribute("x", mx + px);
          cache.distTxt.setAttribute("y", my + py);
        }}
      }});
    }}

    ligG.addEventListener("mousedown", function(e) {{
      dragActive = true;
      startMouse = toSVGCoords(e.clientX, e.clientY);
      startOffset = {{ ...ligOffset }};
      ligG.style.cursor = "grabbing";
      e.preventDefault();
      e.stopPropagation();
    }});
    ligG.addEventListener("touchstart", function(e) {{
      dragActive = true;
      startMouse = toSVGCoords(e.touches[0].clientX, e.touches[0].clientY);
      startOffset = {{ ...ligOffset }};
      e.preventDefault();
      e.stopPropagation();
    }}, {{passive: false}});

    window.addEventListener("mousemove", function(e) {{
      if (!dragActive) return;
      const cur = toSVGCoords(e.clientX, e.clientY);
      ligOffset.x = startOffset.x + cur.x - startMouse.x;
      ligOffset.y = startOffset.y + cur.y - startMouse.y;
      applyOffset(ligOffset.x, ligOffset.y);
    }});
    window.addEventListener("touchmove", function(e) {{
      if (!dragActive) return;
      const cur = toSVGCoords(e.touches[0].clientX, e.touches[0].clientY);
      ligOffset.x = startOffset.x + cur.x - startMouse.x;
      ligOffset.y = startOffset.y + cur.y - startMouse.y;
      applyOffset(ligOffset.x, ligOffset.y);
      e.preventDefault();
    }}, {{passive: false}});
    window.addEventListener("mouseup",  function() {{ dragActive = false; ligG.style.cursor = "move"; }});
    window.addEventListener("touchend", function() {{ dragActive = false; }});

    // Extend resetLayout to also reset ligand position
    const _origReset = window.resetLayout;
    window.resetLayout = function() {{
      ligOffset = {{ x: 0, y: 0 }};
      applyOffset(0, 0);
      if (_origReset) _origReset();
    }};
  }})();
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
.stDownloadButton > button {
    background: var(--success); color: white; border: none; border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.88rem;
    padding: 8px 20px; transition: background 0.2s;
}
.stDownloadButton > button:hover { filter: brightness(1.15); }
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
    dock_run_id=0,
    pdb_token=None, receptor_fh=None, receptor_pdbqt=None,
    box_pdb=None, config_txt=None, cx=None, cy=None, cz=None,
    box_sx=16, box_sy=16, box_sz=16,
    ligand_pdb_path=None, receptor_done=False, receptor_log="",
    cocrystal_ligand_id="",
    ligand_pdbqt=None, ligand_sdf=None, ligand_name="LIG",
    prot_smiles=None, ligand_done=False, ligand_log="",
    output_pdbqt=None, output_sdf=None, output_pv_sdf=None, dock_base=None,
    docking_done=False, docking_log="", score_df=None, pose_mols=None,
    redock_done=False, redock_score=None, redock_result=None,
    confirmed_ref_score=None, confirmed_ref_pose=None, confirmed_ref_name=None,
    pv_image_png=None, pv_image_svg=None, pv_pose_key=None,
    pv_ref_png=None, pv_ref_svg=None,
    b_pdb_token=None, b_receptor_fh=None, b_receptor_pdbqt=None,
    b_box_pdb=None, b_config_txt=None, b_cx=None, b_cy=None, b_cz=None,
    b_box_sx=16, b_box_sy=16, b_box_sz=16,
    b_ligand_pdb_path=None, b_receptor_done=False, b_receptor_log="",
    b_cocrystal_ligand_id="",
    b_batch_done=False, b_batch_results=None, b_batch_log="",
    b_redock_score=None, b_redock_result=None,
    b_confirmed_ref_score=None, b_confirmed_ref_pose=None, b_confirmed_ref_name=None,
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
#  TOOL CHECKS
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


def _add_heme_to_view(view, rec_fh, model_idx):
    """
    Add heme atoms from rec_fh as orange sticks to an existing py3Dmol view.
    Returns updated model_idx (incremented if heme was added).
    """
    if rec_fh and os.path.exists(rec_fh):
        _heme_lines = [
            l for l in open(rec_fh)
            if l[:6].strip() in ("ATOM", "HETATM")
            and l[17:20].strip().upper() in _HEME_RESNAMES
        ]
        if _heme_lines:
            view.addModel("".join(_heme_lines) + "END\n", "pdb")
            view.setStyle({"model": model_idx}, {
                "stick": {"colorscheme": "orangeCarbon", "radius": 0.25}
            })
            view.addLabel("HEM", {
                "fontSize": 12, "fontColor": "orange",
                "backgroundColor": "black", "backgroundOpacity": 0.5,
                "inFront": True, "showBackground": True,
            }, {"model": model_idx})
            model_idx += 1
    return model_idx


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


# ══════════════════════════════════════════════════════════════════════════════
#  READY-TO-USE FIGURE  —  2-panel (single) and 4-panel (batch) layouts
#  Inserted between 2D diagram section and 🤖 AI prompt.
# ══════════════════════════════════════════════════════════════════════════════

def _render_binding_pocket_panel(
    rec_fh: str,
    mol,                    # RDKit mol (selected pose)
    cutoff: float,
    show_labels: bool,
    show_surface: bool,
    heme_rec_fh: str = "",
    cryst_pdb: str = "",    # ignored inside figure (no co-crystal in figure view)
    height: int = 440,
    key_prefix: str = "fig",
    show_cryst: bool = False,   # False = figure mode, True = normal app mode
):
    """
    Render the Binding Pocket View.
    show_cryst=False (default for figure) suppresses the co-crystal ligand overlay.
    """
    import py3Dmol
    from rdkit import Chem
    try:
        v = py3Dmol.view(width="100%", height=height)
        v.setBackgroundColor(_viewer_bg())
        mi = 0
        if rec_fh and os.path.exists(rec_fh):
            v.addModel(open(rec_fh).read(), "pdb")
            v.setStyle({"model": mi}, {"cartoon": {"color": "spectrum", "opacity": 0.45}})
            if show_surface:
                v.addSurface(py3Dmol.SAS, {"opacity": 0.55, "color": "white"}, {"model": mi})
            mi += 1
        mi = _add_heme_to_view(v, heme_rec_fh or rec_fh, mi)
        # Co-crystal only in normal app mode, never in figure panel
        if show_cryst and cryst_pdb and os.path.exists(cryst_pdb):
            v.addModel(open(cryst_pdb).read(), "pdb")
            v.setStyle({"model": mi}, {"stick": {"colorscheme": "magentaCarbon", "radius": 0.18}})
            mi += 1
        v.addModel(Chem.MolToMolBlock(mol), "mol")
        _lig_m = mi
        v.setStyle({"model": _lig_m}, {"stick": {"colorscheme": "cyanCarbon", "radius": 0.30}})
        if rec_fh and os.path.exists(rec_fh):
            _ir = get_interacting_residues(rec_fh, mol, cutoff=cutoff)
            for _rb in _ir:
                _has_chain = bool(_rb["chain"] and _rb["chain"].strip())
                _sel = {"model": 0, "resi": _rb["resi"]}
                if _has_chain:
                    _sel["chain"] = _rb["chain"]
                v.setStyle(_sel, {"stick": {"colorscheme": "orangeCarbon", "radius": 0.20}})
                if show_labels:
                    _lbl_chain = _rb["chain"] if _has_chain else ""
                    v.addLabel(
                        f"{_rb['resn']}{_rb['resi']}{_lbl_chain}",
                        {"fontSize": 11, "fontColor": "yellow",
                         "backgroundColor": "black", "backgroundOpacity": 0.65,
                         "inFront": True, "showBackground": True},
                        _sel,
                    )
        v.zoomTo({"model": _lig_m})
        show3d(v, height=height)
    except Exception as _e:
        st.info(f"Binding pocket viewer error: {_e}")


def _strip_acd_toolbar(html_str: str) -> str:
    """
    Remove the toolbar div AND the top title pill from the ACD interactive HTML
    so only the clean diagram is shown inside the figure panel.

    What is stripped:
    1. The <div style="display:flex..."> toolbar block (Reset / Export / drag hint).
    2. The title pill <rect> + <text> at the top of the SVG (ligand-name capsule).
    """
    import re
    # 1. Remove the outer toolbar div (between <div style="display:flex... and </div>)
    cleaned = re.sub(
        r'<div\s+style="display:flex[^"]*"[^>]*>.*?</div>\s*(?=<svg)',
        '',
        html_str,
        count=1,
        flags=re.DOTALL,
    )
    # 2. Strip the border/background wrapper div styling
    cleaned = cleaned.replace(
        'style="font-family:Arial,sans-serif;background:white;border-radius:8px;\n'
        '            border:1px solid #e0e0e0;overflow:hidden;"',
        'style="font-family:Arial,sans-serif;background:white;"',
    )
    # 3. Remove the title pill: <rect … rx="22" … fill="#f2f2f2" …/>
    #    and the <text … fill="#1a1a1a"> title </text> that follows it.
    #    Both are inside the SVG; we target the two elements by their unique attributes.
    cleaned = re.sub(
        r'<rect\s[^>]*rx="22"[^>]*fill="#f2f2f2"[^>]*/>\s*'
        r'<text\s[^>]*fill="#1a1a1a"[^>]*>.*?</text>',
        '',
        cleaned,
        count=1,
        flags=re.DOTALL,
    )
    return cleaned


def _2d_svg_bytes(acd_svg, acd_ihtml, rdk_svg, pv_svg, pv_png,
                  diag_source: str):
    """
    Return (svg_bytes_or_None, png_bytes_or_None) for the currently selected
    2D diagram source, for use in the figure export.
    """
    import base64, re
    if diag_source == "acd":
        if acd_svg:
            raw = acd_svg if isinstance(acd_svg, bytes) else acd_svg.encode()
            return raw, None
        if acd_ihtml:
            # Extract SVG element from the interactive HTML
            m = re.search(r'(<svg\b.*?</svg>)', acd_ihtml, re.DOTALL)
            if m:
                return m.group(1).encode(), None
        return None, None
    elif diag_source == "rdkit":
        if rdk_svg:
            raw = rdk_svg if isinstance(rdk_svg, bytes) else rdk_svg.encode()
            return raw, None
        return None, None
    else:  # poseview
        if pv_png:
            return None, pv_png
        if pv_svg:
            raw = pv_svg if isinstance(pv_svg, bytes) else pv_svg.encode()
            return raw, None
        return None, None


def _build_figure_svg(
    diag_svg_bytes,  # bytes of 2D SVG (may be None if only PNG available)
    diag_png_bytes,  # bytes of 2D PNG (fallback)
    capsule_text: str,
    plot_fig=None,   # matplotlib figure for top-left panel (batch 4-panel only)
    layout: str = "2panel",  # "2panel" | "4panel"
) -> bytes:
    """
    Compose a publication-quality SVG figure:

    2-panel  [a | b]  — a = placeholder note (3D can't embed), b = 2D diagram
    4-panel  top: [a=plot | b=note]  bottom: [c=note | d=2D diagram]

    The 3D binding pocket view cannot be embedded in SVG (WebGL), so panel a/c
    shows a clean placeholder label. The exported SVG is intended for vector-
    quality 2D diagram + capsule + legend export. For a screenshot of the
    combined 3D+2D layout, instruct the user to use browser screenshot.
    """
    import base64, io

    W, H_panel = 900, 500  # per-panel width x height

    LEGEND_ITEMS = [
        ("#a0c8ff", "#2287ff", None,      "Hydrophobic"),
        ("#80dd80", "#1a7a1a", "5 3",     "H-bond"),
        ("#f0a0ff", "#e200e8", "5 3",     "π-π stacking"),
        ("#ffe090", "#cc8800", "3 2",     "Metal"),
        ("#ffb0d0", "#cc2277", "5 2",     "Halogen bond"),
        ("#c4a0ff", "#6633aa", "4 2 1 2", "H···Halogen"),
    ]

    def _capsule_svg(cx, cy, text, panel_w):
        tw = len(text) * 8.5 + 40
        px = cx - tw / 2
        return (
            f'<rect x="{px:.1f}" y="{cy-15:.1f}" width="{tw:.0f}" height="30" rx="15"'
            f' fill="#f0f0ec" stroke="#c4c4c0" stroke-width="1"/>'
            f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" dominant-baseline="central"'
            f' font-family="Arial,sans-serif" font-size="13" font-weight="700" fill="#1e1e1c">'
            f'{text}</text>'
        )

    def _legend_svg(cx, y, items):
        parts = []
        entry_w = 110
        total = len(items) * entry_w
        x0 = cx - total / 2
        parts.append(
            f'<rect x="{x0-8:.0f}" y="{y-6}" width="{total+16:.0f}" height="28"'
            f' fill="white" stroke="#e8e8e4" stroke-width="0.8" rx="5"/>'
        )
        for k, (fill, stroke, dash, label) in enumerate(items):
            ix = x0 + k * entry_w + 10
            parts.append(
                f'<circle cx="{ix+7:.0f}" cy="{y+8}" r="7"'
                f' fill="{fill}" opacity="0.6" stroke="{stroke}" stroke-width="1.2"/>'
            )
            if dash:
                parts.append(
                    f'<line x1="{ix+17:.0f}" y1="{y+8}" x2="{ix+30:.0f}" y2="{y+8}"'
                    f' stroke="{stroke}" stroke-width="1.5" stroke-dasharray="{dash}"/>'
                )
                tx = ix + 34
            else:
                tx = ix + 18
            parts.append(
                f'<text x="{tx:.0f}" y="{y+8}" dominant-baseline="central"'
                f' font-family="Arial,sans-serif" font-size="11" font-weight="700"'
                f' fill="#555">{label}</text>'
            )
        return "\n".join(parts)

    def _panel_label(x, y, label):
        return (
            f'<text x="{x}" y="{y}" font-family="Arial,sans-serif"'
            f' font-size="22" font-weight="700" fill="#1e1e1c">{label}</text>'
        )

    def _3d_placeholder(px, py, pw, ph, label):
        return (
            f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}"'
            f' fill="#f8f8f5" stroke="#d0d0cc" stroke-width="1" rx="6"/>'
            f'<text x="{px+pw/2:.0f}" y="{py+ph/2-10:.0f}" text-anchor="middle"'
            f' font-family="Arial,sans-serif" font-size="14" fill="#888882">'
            f'🔬 3D Binding Pocket View</text>'
            f'<text x="{px+pw/2:.0f}" y="{py+ph/2+12:.0f}" text-anchor="middle"'
            f' font-family="Arial,sans-serif" font-size="11" fill="#aaa">'
            f'(screenshot the interactive panel above)</text>'
        )

    def _embed_2d(svg_b, png_b, px, py, pw, ph):
        """Embed 2D diagram SVG (inline) or PNG (base64) into the figure SVG."""
        if svg_b:
            raw = svg_b.decode("utf-8", errors="replace") if isinstance(svg_b, bytes) else svg_b
            # Strip xml declaration and embed as foreignObject or image
            raw = raw.strip()
            if raw.startswith("<?xml"):
                raw = raw[raw.index("<svg"):]
            # Embed via <image> using base64 data URI
            b64 = base64.b64encode(raw.encode()).decode()
            return (
                f'<image x="{px}" y="{py}" width="{pw}" height="{ph}"'
                f' href="data:image/svg+xml;base64,{b64}"'
                f' preserveAspectRatio="xMidYMid meet"/>'
            )
        elif png_b:
            b64 = base64.b64encode(png_b).decode()
            return (
                f'<image x="{px}" y="{py}" width="{pw}" height="{ph}"'
                f' href="data:image/png;base64,{b64}"'
                f' preserveAspectRatio="xMidYMid meet"/>'
            )
        return (
            f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}"'
            f' fill="#f8f8f5" stroke="#d0d0cc" stroke-width="1" rx="6"/>'
            f'<text x="{px+pw/2:.0f}" y="{py+ph/2:.0f}" text-anchor="middle"'
            f' font-family="Arial,sans-serif" font-size="13" fill="#888">No 2D diagram yet</text>'
        )

    if layout == "2panel":
        SVG_W = W * 2
        SVG_H = H_panel + 80  # extra for capsule + legend
        pad = 20
        pw = W - pad * 2

        body = [
            f'<svg width="{SVG_W}" height="{SVG_H}" viewBox="0 0 {SVG_W} {SVG_H}"'
            f' xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">',
            '<rect width="100%" height="100%" fill="white"/>',
            # Panel a — 3D placeholder
            _panel_label(pad, 28, "a)"),
            _3d_placeholder(pad, 36, pw, H_panel - 36, "a"),
            # Panel b — 2D diagram
            _panel_label(W + pad, 28, "b)"),
            _embed_2d(diag_svg_bytes, diag_png_bytes, W + pad, 36, pw, H_panel - 80),
            _capsule_svg(W + W / 2, H_panel - 30, capsule_text, W),
            _legend_svg(W + W / 2, H_panel + 12, LEGEND_ITEMS),
            '</svg>',
        ]

    else:  # 4panel
        TOP_H    = 320
        BOT_H    = H_panel + 80
        SVG_W    = W * 2
        SVG_H    = TOP_H + BOT_H + 16
        pad      = 20
        pw       = W - pad * 2

        # Top-left: embed matplotlib plot PNG if available
        plot_png = None
        if plot_fig is not None:
            _buf = io.BytesIO()
            plot_fig.savefig(_buf, format="png", dpi=150,
                             bbox_inches="tight", facecolor=plot_fig.get_facecolor())
            _buf.seek(0)
            plot_png = _buf.getvalue()

        body = [
            f'<svg width="{SVG_W}" height="{SVG_H}" viewBox="0 0 {SVG_W} {SVG_H}"'
            f' xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">',
            '<rect width="100%" height="100%" fill="white"/>',
            # Top row
            _panel_label(pad, 24, "a)"),
            _embed_2d(None, plot_png, pad, 32, pw, TOP_H - 36) if plot_png else
                _3d_placeholder(pad, 32, pw, TOP_H - 36, "a"),
            _panel_label(W + pad, 24, "b)"),
            _3d_placeholder(W + pad, 32, pw, TOP_H - 36, "b"),
            # Bottom row
            _panel_label(pad, TOP_H + 16 + 8, "c)"),
            _3d_placeholder(pad, TOP_H + 16 + 16, pw, BOT_H - 80, "c"),
            _panel_label(W + pad, TOP_H + 16 + 8, "d)"),
            _embed_2d(diag_svg_bytes, diag_png_bytes,
                      W + pad, TOP_H + 16 + 16, pw, BOT_H - 100),
            _capsule_svg(W + W / 2, TOP_H + 16 + BOT_H - 60, capsule_text, W),
            _legend_svg(W + W / 2, TOP_H + 16 + BOT_H - 22, LEGEND_ITEMS),
            '</svg>',
        ]

    return "\n".join(body).encode()


def _render_2d_panel_b(
    acd_svg, acd_ihtml,
    rdk_svg,
    pv_svg, pv_png,
    diag_source: str,           # "acd" | "rdkit" | "poseview"
    capsule_text: str,
    height: int = 480,
    key_prefix: str = "fig",
):
    """
    Render the 2D diagram + capsule + legend for panel b / panel d.
    The ACD interactive toolbar is stripped so only the clean diagram shows.
    """
    import base64

    # ── Diagram ──────────────────────────────────────────────────────────────
    if diag_source == "acd":
        if acd_ihtml:
            # Strip toolbar/controls — show only the SVG diagram
            _clean = _strip_acd_toolbar(acd_ihtml)
            components.html(_clean, height=height, scrolling=False)
        elif acd_svg:
            svg_str = acd_svg.decode() if isinstance(acd_svg, bytes) else acd_svg
            b64 = base64.b64encode(svg_str.encode()).decode()
            st.markdown(
                f'<img src="data:image/svg+xml;base64,{b64}" '
                f'style="width:100%;height:auto;display:block;">',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:center;'
                'height:200px;background:#f8f8f5;border-radius:6px;color:#888;">'
                'Generate the ACD 2D diagram first (tab above).</div>',
                unsafe_allow_html=True,
            )
    elif diag_source == "rdkit":
        if rdk_svg:
            svg_str = rdk_svg.decode() if isinstance(rdk_svg, bytes) else rdk_svg
            b64 = base64.b64encode(svg_str.encode()).decode()
            st.markdown(
                f'<img src="data:image/svg+xml;base64,{b64}" '
                f'style="width:100%;height:auto;display:block;">',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:center;'
                'height:200px;background:#f8f8f5;border-radius:6px;color:#888;">'
                'Generate the RDKit 2D diagram first (tab above).</div>',
                unsafe_allow_html=True,
            )
    else:  # poseview
        if pv_png:
            b64 = base64.b64encode(pv_png).decode()
            st.markdown(
                f'<img src="data:image/png;base64,{b64}" '
                f'style="width:100%;height:auto;display:block;">',
                unsafe_allow_html=True,
            )
        elif pv_svg:
            svg_str = pv_svg.decode() if isinstance(pv_svg, bytes) else pv_svg
            b64 = base64.b64encode(svg_str.encode()).decode()
            st.markdown(
                f'<img src="data:image/svg+xml;base64,{b64}" '
                f'style="width:100%;height:auto;display:block;">',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:center;'
                'height:200px;background:#f8f8f5;border-radius:6px;color:#888;">'
                'Generate the PoseView 2D diagram first (tab above).</div>',
                unsafe_allow_html=True,
            )

    # ── Capsule label ─────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="text-align:center;margin:12px 0 6px;">'
        f'<span style="display:inline-block;padding:7px 26px;border-radius:999px;'
        f'background:#f0f0ec;border:1px solid #c4c4c0;font-size:14px;'
        f'font-weight:700;color:#1e1e1c;letter-spacing:0.01em;">'
        f'{capsule_text}</span></div>',
        unsafe_allow_html=True,
    )

    # ── Interaction legend ─────────────────────────────────────────────────────
    LEGEND_ITEMS = [
        ("#a0c8ff", "#2287ff", None,      "Hydrophobic"),
        ("#80dd80", "#1a7a1a", "5 3",     "H-bond"),
        ("#f0a0ff", "#e200e8", "5 3",     "π-π stacking"),
        ("#ffe090", "#cc8800", "3 2",     "Metal"),
        ("#ffb0d0", "#cc2277", "5 2",     "Halogen bond"),
        ("#c4a0ff", "#6633aa", "4 2 1 2", "H···Halogen"),
    ]
    parts = []
    for fill, stroke, dash, label in LEGEND_ITEMS:
        circle = (
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'border-radius:50%;background:{fill};border:1.5px solid {stroke};'
            f'vertical-align:middle;margin-right:3px;"></span>'
        )
        line = (
            f'<span style="display:inline-block;width:18px;height:2px;'
            f'border-top:2px dashed {stroke};vertical-align:middle;margin:0 3px;"></span>'
            if dash else ""
        )
        parts.append(
            f'<span style="margin:0 6px 4px 0;display:inline-flex;align-items:center;'
            f'font-size:11px;color:#555;">{circle}{line}{label}</span>'
        )
    st.markdown(
        '<div style="text-align:center;margin:4px 0 2px;line-height:1.8;">'
        + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


def _make_figure_html(
    rec_fh: str,
    mol,
    cutoff: float,
    show_labels: bool,
    show_surface: bool,
    acd_svg, acd_ihtml,
    rdk_svg,
    pv_svg, pv_png,
    diag_source: str,
    capsule_text: str,
    layout: str = "2panel",   # "2panel" | "4panel"
    plot_png_bytes=None,
    b_browsable=None,
    b_sel_nm: str = "",
    b_mols=None,
    b_pose_i: int = 0,
    b_rec_fh: str = "",
) -> str:
    """
    Build a self-contained HTML string that renders a TRUE two-panel (or
    four-panel) artwork canvas at fixed pixel dimensions.

    Key layout decisions:
    • Each panel sits inside a fixed-width .col wrapper with explicit `width:Npx`
      (NOT flex-grow), so the columns always sit side-by-side even inside a
      narrow parent iframe.
    • Interactive content (3D viewer, ACD dragable SVG) is isolated in nested
      <iframe src="data:text/html;base64,…"> so its scripts don't clash with
      sibling panels' scripts.
    • Static content (RDKit SVG, PoseView PNG) uses <img> with object-fit:
      contain so it fills and centres inside its panel.

    Layout 2-panel:  [ a: 3D pocket | b: 2D diagram + capsule + legend ]
    Layout 4-panel:  top  [ a: plot | b: pose viewer ]   (shorter)
                     bot  [ c: 3D pocket | d: 2D + caps + leg ]  (taller)
    """
    import base64, re
    from rdkit import Chem

    # ── Canvas geometry (FIXED, does not depend on viewport) ──────────────────
    CANVAS_W   = 1100
    PAD        = 16
    GAP        = 16

    PANEL_W    = (CANVAS_W - PAD * 2 - GAP) // 2     # 526 px each

    if layout == "2panel":
        PANEL_H        = 500
        PANEL_H_2D     = 500
        CANVAS_H       = PAD + 28 + PANEL_H + 90 + PAD   # label + panel + caps+legend
    else:
        TOP_PANEL_H    = 260
        BOT_PANEL_H    = 440
        PANEL_H        = BOT_PANEL_H
        PANEL_H_2D     = BOT_PANEL_H
        CANVAS_H       = PAD + 28 + TOP_PANEL_H + GAP + 28 + BOT_PANEL_H + 90 + PAD

    # ── Helper: wrap any HTML fragment in a full HTML page + base64 iframe ────
    def _iframe_wrap(inner_html: str, bg: str = "white") -> str:
        page = (
            f'<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<style>html,body{{margin:0;padding:0;background:{bg};'
            f'width:100%;height:100%;overflow:hidden;}}</style></head>'
            f'<body>{inner_html}</body></html>'
        )
        b64 = base64.b64encode(page.encode("utf-8")).decode()
        return (f'<iframe src="data:text/html;base64,{b64}" '
                f'style="width:100%;height:100%;border:0;display:block;" '
                f'frameborder="0" scrolling="no"></iframe>')

    # ── Build 3D panel (panel a / panel c) — real py3Dmol in isolated iframe ──
    import py3Dmol
    _3d_pocket_html = ""
    try:
        v3d = py3Dmol.view(width=PANEL_W, height=PANEL_H)
        v3d.setBackgroundColor("#ffffff")
        mi = 0
        if rec_fh and os.path.exists(rec_fh):
            v3d.addModel(open(rec_fh).read(), "pdb")
            v3d.setStyle({"model": mi}, {"cartoon": {"color": "spectrum", "opacity": 0.45}})
            if show_surface:
                v3d.addSurface(py3Dmol.SAS, {"opacity": 0.55, "color": "white"}, {"model": mi})
            mi += 1
        mi = _add_heme_to_view(v3d, rec_fh, mi)
        # NO co-crystal ligand in the figure panel
        v3d.addModel(Chem.MolToMolBlock(mol), "mol")
        _lig_m = mi
        v3d.setStyle({"model": _lig_m}, {"stick": {"colorscheme": "cyanCarbon", "radius": 0.30}})
        if rec_fh and os.path.exists(rec_fh):
            _ir = get_interacting_residues(rec_fh, mol, cutoff=cutoff)
            for _rb in _ir:
                _has_chain = bool(_rb["chain"] and _rb["chain"].strip())
                _sel = {"model": 0, "resi": _rb["resi"]}
                if _has_chain:
                    _sel["chain"] = _rb["chain"]
                v3d.setStyle(_sel, {"stick": {"colorscheme": "orangeCarbon", "radius": 0.20}})
                if show_labels:
                    _lbl_chain = _rb["chain"] if _has_chain else ""
                    v3d.addLabel(
                        f"{_rb['resn']}{_rb['resi']}{_lbl_chain}",
                        {"fontSize": 10, "fontColor": "yellow",
                         "backgroundColor": "black", "backgroundOpacity": 0.6,
                         "inFront": True, "showBackground": True},
                        _sel,
                    )
        v3d.zoomTo({"model": _lig_m})
        # _make_html() returns a FULL HTML page (with 3Dmol <script src>).
        # We iframe it AS-IS via base64, so the scripts run in an isolated doc.
        raw3d = v3d._make_html()
        b3d = base64.b64encode(raw3d.encode("utf-8")).decode()
        _3d_pocket_html = (
            f'<iframe src="data:text/html;base64,{b3d}" '
            f'style="width:100%;height:100%;border:0;display:block;" '
            f'frameborder="0" scrolling="no"></iframe>'
        )
    except Exception as _e3:
        _3d_pocket_html = (
            f'<div style="display:flex;align-items:center;justify-content:center;'
            f'height:100%;color:#888;font-family:Arial,sans-serif;font-size:13px;">'
            f'3D viewer: {_e3}</div>'
        )

    # ── Build 2D panel content (panel b / panel d) ────────────────────────────
    _2d_content = ""
    if diag_source == "acd" and acd_ihtml:
        # Strip toolbar + title pill, then iframe-isolate so drag script works
        _clean = _strip_acd_toolbar(acd_ihtml)
        # Make the SVG fill its iframe
        _clean = re.sub(
            r'<svg([^>]*?)style="[^"]*"',
            r'<svg\1style="width:100%;height:100%;display:block;"',
            _clean, count=1,
        )
        _clean = re.sub(
            r'<svg(?![^>]*style=)([^>]*?)>',
            r'<svg\1 style="width:100%;height:100%;display:block;">',
            _clean, count=1,
        )
        _2d_content = _iframe_wrap(_clean)
    elif diag_source == "acd" and acd_svg:
        s = acd_svg.decode() if isinstance(acd_svg, bytes) else acd_svg
        b64 = base64.b64encode(s.encode()).decode()
        _2d_content = (
            f'<img src="data:image/svg+xml;base64,{b64}" '
            f'style="width:100%;height:100%;object-fit:contain;display:block;">'
        )
    elif diag_source == "rdkit" and rdk_svg:
        s = rdk_svg.decode() if isinstance(rdk_svg, bytes) else rdk_svg
        b64 = base64.b64encode(s.encode()).decode()
        _2d_content = (
            f'<img src="data:image/svg+xml;base64,{b64}" '
            f'style="width:100%;height:100%;object-fit:contain;display:block;">'
        )
    elif diag_source == "poseview" and pv_png:
        b64 = base64.b64encode(pv_png).decode()
        _2d_content = (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;height:100%;object-fit:contain;display:block;">'
        )
    elif diag_source == "poseview" and pv_svg:
        s = pv_svg.decode() if isinstance(pv_svg, bytes) else pv_svg
        b64 = base64.b64encode(s.encode()).decode()
        _2d_content = (
            f'<img src="data:image/svg+xml;base64,{b64}" '
            f'style="width:100%;height:100%;object-fit:contain;display:block;">'
        )
    else:
        _2d_content = (
            '<div style="display:flex;align-items:center;justify-content:center;'
            'height:100%;color:#aaa;font-family:Arial,sans-serif;font-size:13px;">'
            'Generate a 2D diagram first (tab above)</div>'
        )

    # ── Capsule + legend (below 2D panel only) ────────────────────────────────
    _cap_html = (
        f'<div style="text-align:center;margin:10px 0 4px;">'
        f'<span style="display:inline-block;padding:7px 28px;border-radius:999px;'
        f'background:#f0f0ec;border:1px solid #c4c4c0;font-size:14px;'
        f'font-weight:700;color:#1e1e1c;letter-spacing:0.01em;'
        f'font-family:Arial,sans-serif;">'
        f'{capsule_text}</span></div>'
    )
    LEGEND_ITEMS = [
        ("#a0c8ff", "#2287ff", None,      "Hydrophobic"),
        ("#80dd80", "#1a7a1a", "5 3",     "H-bond"),
        ("#f0a0ff", "#e200e8", "5 3",     "π-π stacking"),
        ("#ffe090", "#cc8800", "3 2",     "Metal"),
        ("#ffb0d0", "#cc2277", "5 2",     "Halogen bond"),
        ("#c4a0ff", "#6633aa", "4 2 1 2", "H···Halogen"),
    ]
    _leg_parts = []
    for fill, stroke, dash, label in LEGEND_ITEMS:
        circ = (f'<span style="display:inline-block;width:13px;height:13px;'
                f'border-radius:50%;background:{fill};border:1.5px solid {stroke};'
                f'vertical-align:middle;margin-right:3px;"></span>')
        dash_el = (f'<span style="display:inline-block;width:16px;height:2px;'
                   f'border-top:2px dashed {stroke};vertical-align:middle;margin:0 3px;"></span>'
                   if dash else "")
        _leg_parts.append(
            f'<span style="margin:0 5px 2px 0;display:inline-flex;align-items:center;'
            f'font-size:11px;color:#555;font-family:Arial,sans-serif;">'
            f'{circ}{dash_el}{label}</span>'
        )
    _leg_html = ('<div style="text-align:center;line-height:1.7;">'
                 + "".join(_leg_parts) + '</div>')

    # ── 4-panel top row content (plot + pose browser) ─────────────────────────
    _plot_html = ""
    _browser_html = ""
    if layout == "4panel":
        if plot_png_bytes:
            b64 = base64.b64encode(plot_png_bytes).decode()
            _plot_html = (f'<img src="data:image/png;base64,{b64}" '
                          f'style="width:100%;height:100%;object-fit:contain;display:block;">')
        else:
            _plot_html = ('<div style="display:flex;align-items:center;justify-content:center;'
                          'height:100%;color:#aaa;font-family:Arial,sans-serif;font-size:12px;">'
                          'Score plot — run batch docking</div>')

        # Mini 3D pose viewer for panel b
        if b_browsable and b_mols:
            try:
                import py3Dmol as _p3b
                from rdkit import Chem as _Cb
                _sc = b_mols[b_pose_i] if b_pose_i < len(b_mols) else None
                _vb = _p3b.view(width=PANEL_W, height=TOP_PANEL_H)
                _vb.setBackgroundColor("#ffffff")
                _bi = 0
                if b_rec_fh and os.path.exists(b_rec_fh):
                    _vb.addModel(open(b_rec_fh).read(), "pdb")
                    _vb.setStyle({"model": _bi}, {"cartoon": {"color": "spectrum", "opacity": 0.6}})
                    _bi += 1
                _bi = _add_heme_to_view(_vb, b_rec_fh, _bi)
                if _sc is not None:
                    _vb.addModel(_Cb.MolToMolBlock(_sc), "mol")
                    _vb.setStyle({"model": _bi}, {"stick": {"colorscheme": "cyanCarbon", "radius": 0.28}})
                    _vb.zoomTo({"model": _bi})
                raw_b = _vb._make_html()
                b_b64 = base64.b64encode(raw_b.encode("utf-8")).decode()
                _browser_html = (
                    f'<iframe src="data:text/html;base64,{b_b64}" '
                    f'style="width:100%;height:100%;border:0;display:block;" '
                    f'frameborder="0" scrolling="no"></iframe>'
                )
            except Exception as _be:
                _browser_html = (f'<div style="display:flex;align-items:center;justify-content:center;'
                                 f'height:100%;color:#888;font-family:Arial,sans-serif;font-size:13px;">'
                                 f'Pose viewer: {_be}</div>')
        else:
            _browser_html = ('<div style="display:flex;align-items:center;justify-content:center;'
                             'height:100%;color:#aaa;font-family:Arial,sans-serif;font-size:12px;">'
                             'Pose Browser</div>')

    # ── CSS with EXPLICIT-WIDTH columns (not flex-grow) ───────────────────────
    #    This is the critical fix: .col has width:PANEL_W px so the two columns
    #    always sit side-by-side regardless of parent width.
    CSS = f"""
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{
        background: #ffffff;
        font-family: Arial, sans-serif;
        overflow-x: auto;
    }}
    body {{ padding: 8px 0; display: flex; justify-content: center; }}
    #canvas {{
        width: {CANVAS_W}px;
        min-width: {CANVAS_W}px;
        max-width: {CANVAS_W}px;
        background: white;
        border: 1.5px solid #c8c8c4;
        border-radius: 8px;
        padding: {PAD}px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    .row {{
        display: flex;
        gap: {GAP}px;
        align-items: flex-start;
        width: 100%;
    }}
    .col {{
        width: {PANEL_W}px;
        min-width: {PANEL_W}px;
        max-width: {PANEL_W}px;
        flex: 0 0 {PANEL_W}px;
    }}
    .plabel {{
        font-family: Arial, sans-serif;
        font-size: 18px;
        font-weight: 700;
        color: #1e1e1c;
        margin: 0 0 6px 2px;
        line-height: 1;
    }}
    .panel {{
        width: {PANEL_W}px;
        background: #fafaf8;
        border: 1px solid #ececea;
        border-radius: 4px;
        overflow: hidden;
        position: relative;
    }}
    """

    # ── Compose final HTML ────────────────────────────────────────────────────
    if layout == "2panel":
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<div id="canvas">
  <div class="row">
    <div class="col">
      <p class="plabel">a)</p>
      <div class="panel" style="height:{PANEL_H}px;">{_3d_pocket_html}</div>
    </div>
    <div class="col">
      <p class="plabel">b)</p>
      <div class="panel" style="height:{PANEL_H_2D}px;">{_2d_content}</div>
      {_cap_html}
      {_leg_html}
    </div>
  </div>
</div>
</body></html>"""
    else:  # 4panel
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<div id="canvas">
  <div class="row" style="margin-bottom:{GAP}px;">
    <div class="col">
      <p class="plabel">a)</p>
      <div class="panel" style="height:{TOP_PANEL_H}px;">{_plot_html}</div>
    </div>
    <div class="col">
      <p class="plabel">b)</p>
      <div class="panel" style="height:{TOP_PANEL_H}px;">{_browser_html}</div>
    </div>
  </div>
  <div class="row">
    <div class="col">
      <p class="plabel">c)</p>
      <div class="panel" style="height:{BOT_PANEL_H}px;">{_3d_pocket_html}</div>
    </div>
    <div class="col">
      <p class="plabel">d)</p>
      <div class="panel" style="height:{PANEL_H_2D}px;">{_2d_content}</div>
      {_cap_html}
      {_leg_html}
    </div>
  </div>
</div>
</body></html>"""

    return html


def _ready_figure_section(
    mode: str,           # "single" | "batch"
    # ── single dock state ──────────────────────────────────────────
    rec_fh: str = "",
    sel_mol=None,
    pose_idx: int = 0,
    lig_name: str = "",
    binding_energy=None,
    cryst_pdb: str = "",
    acd_svg=None,  acd_ihtml=None,
    rdk_svg=None,
    pv_svg=None,   pv_png=None,
    # ── batch-only state ───────────────────────────────────────────
    b_browsable=None,
    b_sel_res=None,
    b_mols=None,
    b_pose_i: int = 0,
    b_plot_draw_fn=None,   # callable(ax) — reuses existing _draw_plot
    b_plot_n: int = 0,
    b_rec_fh: str = "",
    b_cryst_pdb: str = "",
    b_acd_svg=None,  b_acd_ihtml=None,
    b_rdk_svg=None,
    b_pv_svg=None,   b_pv_png=None,
    b_this_score=None,
    b_sel_nm: str = "",
):
    """
    Ready-to-use Figure section.

    Renders a FIXED-SIZE artwork canvas (1100 px wide) via components.html()
    so the layout is completely independent of browser window width.

    Single mode: [a: 3D pocket | b: 2D diagram + capsule + legend]
    Batch mode:  [a: plot | b: pose browser] (shorter top row)
                 [c: 3D pocket | d: 2D diagram + capsule + legend] (taller bottom row)

    All controls are OUTSIDE the canvas. Export buttons produce a composed SVG
    (2D panel as vector, 3D panel as label) and PNG via cairosvg.
    """
    import base64, io
    st.markdown("---")
    st.markdown("### 📊 Ready-to-use Figure")

    # ═════════════════════════════════════════════════════════════════════════
    #  CONTROLS  — all outside the artwork canvas
    # ═════════════════════════════════════════════════════════════════════════
    with st.expander("⚙️ Figure settings", expanded=True):
        _src = st.radio(
            "2D diagram source",
            ["🧬 Anyone Can Dock", "🔬 RDKit", "🔬 PoseView"],
            horizontal=True,
            key=f"rtf_src_{mode}",
        )
        _src_key = "acd" if "Anyone" in _src else ("rdkit" if "RDKit" in _src else "poseview")

        if mode == "batch":
            _layout = st.radio(
                "Layout",
                ["[a | b]  Single-style", "[a b / c d]  4-panel"],
                horizontal=True,
                key="rtf_batch_layout",
            )
            _4panel = "4-panel" in _layout
        else:
            _4panel = False

        _ctl1, _ctl2, _ctl3 = st.columns(3)
        with _ctl1:
            _cutoff = st.slider("Pocket cutoff (Å)", 2.5, 5.0, 3.5, 0.1, key=f"rtf_cutoff_{mode}")
        with _ctl2:
            _show_labels = st.checkbox("Residue labels", value=True, key=f"rtf_lbl_{mode}")
        with _ctl3:
            _show_surf = st.checkbox("Protein surface", value=False, key=f"rtf_surf_{mode}")

    # ── Resolve data ──────────────────────────────────────────────────────────
    if mode == "single":
        _rec     = rec_fh;    _mol    = sel_mol
        _p_idx   = pose_idx;  _lname  = lig_name
        _score   = binding_energy
        _a_svg   = acd_svg;   _a_ihtml = acd_ihtml
        _r_svg   = rdk_svg
        _pv_svg_ = pv_svg;    _pv_png_ = pv_png
        _plot_fn = None;      _plot_n  = 0
        _browsable_rtf = None
    else:
        _b_mols  = b_mols or []
        _mol     = _b_mols[b_pose_i] if _b_mols else None
        _rec     = b_rec_fh
        _p_idx   = b_pose_i
        _lname   = b_sel_nm.replace("⭐ ", "").replace(" (co-crystal ref)", "")
        _score   = b_this_score
        _a_svg   = b_acd_svg;  _a_ihtml = b_acd_ihtml
        _r_svg   = b_rdk_svg
        _pv_svg_ = b_pv_svg;   _pv_png_ = b_pv_png
        _plot_fn = b_plot_draw_fn; _plot_n = b_plot_n
        _browsable_rtf = b_browsable

    _capsule = (
        f"Pose {_p_idx + 1}  ·  {_lname}"
        + (f"  ·  {_score:.2f} kcal/mol" if _score is not None else "")
    )

    if _mol is None:
        st.info("Select a pose to render the figure.")
        return

    # ── Score plot PNG for 4-panel export ─────────────────────────────────────
    _plot_png = None
    if _4panel and _plot_fn and _plot_n > 0:
        _pfig, _pax = plt.subplots(figsize=(max(4, _plot_n * 0.55 + 1.2), 2.8))
        _plot_fn(_pax)
        _pfig.tight_layout()
        _pbuf = io.BytesIO()
        _pfig.savefig(_pbuf, format="png", dpi=150, bbox_inches="tight",
                      facecolor=_pfig.get_facecolor())
        _pbuf.seek(0)
        _plot_png = _pbuf.getvalue()
        plt.close(_pfig)

    _layout_key = "4panel" if _4panel else "2panel"

    # ═════════════════════════════════════════════════════════════════════════
    #  FIXED ARTWORK CANVAS — rendered as a single components.html() block
    #  Width is locked at 1100 px; height adapts to layout.
    # ═════════════════════════════════════════════════════════════════════════
    _canvas_h = 680 if _layout_key == "2panel" else 860
    _fig_html = _make_figure_html(
        rec_fh=_rec, mol=_mol, cutoff=_cutoff,
        show_labels=_show_labels, show_surface=_show_surf,
        acd_svg=_a_svg, acd_ihtml=_a_ihtml,
        rdk_svg=_r_svg,
        pv_svg=_pv_svg_, pv_png=_pv_png_,
        diag_source=_src_key,
        capsule_text=_capsule,
        layout=_layout_key,
        plot_png_bytes=_plot_png,
        b_browsable=_browsable_rtf,
        b_sel_nm=b_sel_nm,
        b_mols=_b_mols if mode == "batch" else None,
        b_pose_i=b_pose_i,
        b_rec_fh=b_rec_fh,
    )
    components.html(_fig_html, height=_canvas_h + 40, scrolling=False)

    # ═════════════════════════════════════════════════════════════════════════
    #  EXPORT BUTTONS — outside artwork canvas
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<p style="font-size:12px;color:#888;margin:6px 0 4px;">'
        '⬇ Export full figure (2D diagram panel exported as vector SVG; '
        '3D panel requires browser screenshot)</p>',
        unsafe_allow_html=True,
    )

    # Build the composed SVG (vector 2D + frame)
    _diag_svg_b, _diag_png_b = _2d_svg_bytes(
        _a_svg, _a_ihtml, _r_svg, _pv_svg_, _pv_png_, _src_key
    )
    _export_svg = _build_figure_svg(
        diag_svg_bytes=_diag_svg_b,
        diag_png_bytes=_diag_png_b,
        capsule_text=_capsule,
        plot_fig=None,       # plot already embedded as PNG in _make_figure_html
        layout=_layout_key,
    )

    _ecol1, _ecol2 = st.columns(2)
    with _ecol1:
        st.download_button(
            "⬇ Save SVG",
            data=_export_svg,
            file_name="ready_to_use_figure.svg",
            mime="image/svg+xml",
            key=f"rtf_dl_svg_{mode}",
            type="primary",
            width="stretch",
        )
    with _ecol2:
        try:
            import cairosvg as _cs
            _export_png = _cs.svg2png(bytestring=_export_svg, scale=2, background_color="white")
        except Exception:
            _export_png = None
        if _export_png:
            st.download_button(
                "⬇ Save PNG",
                data=_export_png,
                file_name="ready_to_use_figure.png",
                mime="image/png",
                key=f"rtf_dl_png_{mode}",
                type="primary",
                width="stretch",
            )
        else:
            st.caption("PNG export requires cairosvg — SVG available above.")


def _ai_prompt_section(
    engine: str,          # "acd" | "rdkit" | "poseview"
    lig_name: str,
    pdb_id: str,
    binding_energy,       # float or None
    has_ref: bool,
    cocrystal_ligand_id: str = "",
    ref_lig_name: str = "",
    ref_lig_energy=None,  # float or None
    ref_redocked: bool = False,
    rmsd_crystal=None,    # float or None — RMSD of selected pose vs co-crystal
    key_suffix: str = "",
):
    """
    Render the 🤖 AI interpretation prompt block, adapted to the diagram engine.
    Call this at the bottom of every 2D diagram tab after images are shown.
    """
    st.markdown("---")
    st.markdown("### 🤖 Understand Your Results with AI")

    _engine_labels = {
        "acd":      "Anyone Can Dock 2D Diagram",
        "rdkit":    "RDKit 2D Diagram",
        "poseview": "PoseView (proteins.plus)",
    }
    _engine_label = _engine_labels.get(engine, "2D interaction diagram")

    _legend = {
        "acd": (
            "  Green dashed line     = hydrogen bond (number on line = distance in Å)\n"
            "  Magenta dashed line   = pi-pi stacking (aromatic ring interaction)\n"
            "  Blue circle (no line) = hydrophobic contact\n"
            "  Gold dashed line      = metal / heme coordination\n"
            "  Labels on circles     = amino acid name + residue number + chain"
        ),
        "rdkit": (
            "  Blue highlight circle   = hydrogen bond / polar interaction\n"
            "  Green highlight circle  = hydrophobic contact\n"
            "  Pink highlight circle   = other interaction (ionic, metal, etc.)\n"
            "  Labels on circles       = amino acid name + residue number + chain"
        ),
        "poseview": (
            "  Green dashed line     = hydrogen bond (number on line = distance in Å)\n"
            "  Yellow filled circle  = hydrophobic contact\n"
            "  Blue dashed line      = pi-pi / aromatic stacking\n"
            "  Labels on circles     = amino acid name + residue number + chain"
        ),
    }.get(engine, "  See legend in diagram.")

    _n_diagrams = "two 2D" if has_ref else "a 2D"
    _plural     = "diagrams" if has_ref else "diagram"

    if engine == "acd":
        st.caption(
            f"Download your diagram (PNG button above), then paste the prompt "
            f"below + the image into **Claude**, **GPT-4o**, or **Gemini** "
            f"to get a plain-English explanation of your docking results."
        )
    else:
        st.caption(
            f"Screenshot or download the {_engine_label} above, then paste the prompt "
            f"below + the image into **Claude**, **GPT-4o**, or **Gemini** "
            f"to get a plain-English explanation of your docking results."
        )

    _estr       = f"{binding_energy:.2f} kcal/mol" if binding_energy is not None else "[binding energy]"
    _lig_disp   = lig_name or "[ligand]"
    _pdb_disp   = pdb_id.upper() if pdb_id else "[PDB ID]"
    _ref_disp   = ref_lig_name or cocrystal_ligand_id or "[co-crystal ligand]"
    _ref_estr   = f"{ref_lig_energy:.2f} kcal/mol" if ref_lig_energy is not None else None
    # RMSD line — shown when available regardless of redocking
    _rmsd_line  = (
        f"  RMSD vs co-crystal pose: {rmsd_crystal:.2f} Å"
        + (" ✓ (≤2.0 Å — good reproduction)" if rmsd_crystal <= 2.0
           else " ⚠ (2–3 Å — moderate)" if rmsd_crystal <= 3.0
           else " ✗ (>3 Å — pose differs significantly from crystal)")
        if rmsd_crystal is not None else ""
    )

    if has_ref:
        _ref_line = (
            f"Reference: {_ref_disp} co-crystallised in PDB {_pdb_disp}"
            + (f"  |  binding energy from re-docking: {_ref_estr}" if ref_redocked
               else "  (see 2D diagram — no re-docking performed)")
        )
        _lines = [
            "I have just run a molecular docking experiment and I need help",
            f"understanding what my results mean. I am attaching {_n_diagrams}",
            f"interaction {_plural} generated with the {_engine_label} in the Anyone Can Dock app.",
            "",
            "Docking software: AutoDock Vina v1.2.7",
            f"Protein target (PDB): {_pdb_disp}",
            f"My docked ligand: {_lig_disp}",
            f"  Predicted binding energy: {_estr}",
            "  (more negative = stronger predicted binding)",
        ] + ([_rmsd_line] if _rmsd_line else []) + [
            _ref_line,
            "",
            "How to read the diagram:",
            _legend,
            "",
            "Please help me understand:",
            "1. What are the most important interactions my ligand makes,",
            "   and why do they matter for binding?",
            "2. How does my docked ligand compare to the reference — are the",
            "   key contacts conserved or different?",
            "3. Based on the binding energy"
            + (" and the interaction pattern," if ref_redocked else " (docked ligand only) and the interaction pattern,")
            + " does my",
            "   ligand look like a promising binder, and what could be improved?",
        ] + (["4. My pose RMSD vs the co-crystal is " + _rmsd_line.strip() + ". What does this tell me",
              "   about how well the docking reproduced the experimental binding mode?"] if _rmsd_line else []) + [
            "",
            "Please explain in plain language that a non-expert can follow,",
            "but include the specific residue names and distances from the diagram.",
            "",
            "Finally, write a short ready-to-use paragraph (3-4 sentences) that I",
            "can copy directly into a report or presentation slide.",
            "Label this section: 'Ready-to-use summary:'",
        ]
    else:
        _lines = [
            "I have just run a molecular docking experiment and I need help",
            f"understanding what my results mean. I am attaching {_n_diagrams}",
            f"interaction {_plural} generated with the {_engine_label} in the Anyone Can Dock app.",
            "",
            "Docking software: AutoDock Vina v1.2.7",
            f"Protein target (PDB): {_pdb_disp}",
            f"My docked ligand: {_lig_disp}",
            f"  Predicted binding energy: {_estr}",
            "  (more negative = stronger predicted binding)",
        ] + ([_rmsd_line] if _rmsd_line else []) + [
            "",
            "How to read the diagram:",
            _legend,
            "",
            "Please help me understand:",
            "1. What interactions is my ligand making with the protein,",
            "   and which ones are most important for binding?",
            "2. What does the binding energy value tell me?",
            "3. Are there any obvious ways the binding could be improved?",
        ] + (["4. My pose RMSD vs the co-crystal is " + _rmsd_line.strip() + ". What does this tell me",
              "   about how well the docking reproduced the experimental binding mode?"] if _rmsd_line else []) + [
            "",
            "Please explain in plain language that a non-expert can follow,",
            "but include the specific residue names and distances from the diagram.",
            "",
            "Finally, write a short ready-to-use paragraph (3-4 sentences) that I",
            "can copy directly into a report or presentation slide.",
            "Label this section: 'Ready-to-use summary:'",
        ]

    st.code("\n".join(_lines), language=None)


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
    rmsd_crystal=None,   # float or None — RMSD of selected pose vs co-crystal PDB
):
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
        "🔬 PoseView (proteins.plus)",
    ])

    # ── TAB 1: Anyone Can Dock diagram ────────────────────────────────────────
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
                if _has_ref_local:
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
                        else:
                            st.info("Click Generate to produce the co-crystal diagram.")
                else:
                    st.markdown("##### Docked Pose")
                    if _view_mode.startswith("🖱") and _new_ihtml:
                        components.html(_new_ihtml, height=860, scrolling=False)
                    else:
                        _show_svg_new(_new_svg, f"pose{pose_idx+1}_interaction")
                    st.caption("ℹ️ No co-crystal ligand detected — co-crystal reference diagram is not available.")

                _ai_prompt_section(
                    engine="acd",
                    lig_name=lig_name,
                    pdb_id=pdb_id,
                    binding_energy=binding_energy,
                    has_ref=bool(_new_ref_svg),
                    cocrystal_ligand_id=cocrystal_ligand_id,
                    ref_lig_name=ref_lig_name,
                    ref_lig_energy=ref_lig_energy,
                    ref_redocked=(ref_lig_energy is not None),
                    rmsd_crystal=rmsd_crystal,
                    key_suffix=btn_key + "_acd",
                )

    # ── TAB 2: RDKit diagram ──────────────────────────────────────────────────
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
                )

            if st.button("🔬 Generate Both RDKit Diagrams", key=btn_key + "_rdk_gen", type="primary"):
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
            )

            def _show_rdkit_svg_tab(svg_data, dl_filename):
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
                        border:1px solid #D0D7DE;overflow:hidden;">
                      {_sv}
                      <div style="display:flex;align-items:center;gap:20px;padding:10px 16px;
                           border-top:1px solid #D0D7DE;font-size:13px;color:#333;">
                        <div style="display:flex;align-items:center;gap:7px;">
                          <div style="width:14px;height:14px;border-radius:50%;
                               background:rgba(89,156,214,0.55);border:1px solid #5B9BD5;"></div>
                          <span>H-bond / polar</span></div>
                        <div style="display:flex;align-items:center;gap:7px;">
                          <div style="width:14px;height:14px;border-radius:50%;
                               background:rgba(44,141,87,0.55);border:1px solid #2E8B57;"></div>
                          <span>Hydrophobic</span></div>
                        <div style="display:flex;align-items:center;gap:7px;">
                          <div style="width:14px;height:14px;border-radius:50%;
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
                if _has_ref_rdkit2:
                    _col_l2, _col_r2 = st.columns(2)
                    with _col_l2:
                        st.markdown("##### 🧪 Docked Pose (RDKit)")
                        _show_rdkit_svg_tab(_rdk_svg, f"pose{pose_idx+1}_rdkit.svg")
                    with _col_r2:
                        st.markdown("##### 🔮 Co-Crystal Reference (RDKit)")
                        if _ref_rdk_svg:
                            _show_rdkit_svg_tab(_ref_rdk_svg, "cocrystal_rdkit.svg")
                        else:
                            st.info("Click **Generate RDKit Diagrams** to generate co-crystal diagram.")
                else:
                    st.markdown("##### 🧪 Docked Pose (RDKit)")
                    _show_rdkit_svg_tab(_rdk_svg, f"pose{pose_idx+1}_rdkit.svg")
                    st.caption("ℹ️ No co-crystal ligand detected.")

                _ai_prompt_section(
                    engine="rdkit",
                    lig_name=lig_name,
                    pdb_id=pdb_id,
                    binding_energy=binding_energy,
                    has_ref=bool(_has_ref_rdkit2 and _ref_rdk_svg),
                    cocrystal_ligand_id=cocrystal_ligand_id,
                    ref_lig_name=ref_lig_name,
                    ref_lig_energy=ref_lig_energy,
                    ref_redocked=(ref_lig_energy is not None),
                    rmsd_crystal=rmsd_crystal,
                    key_suffix=btn_key + "_rdkit",
                )

    # ── TAB 3: PoseView ───────────────────────────────────────────────────────
    with _tab_pv:
        _ci, _cb = st.columns([3, 1])
        with _ci:
            if _stale and st.session_state.get(img_svg_key):
                st.caption("⚠️ Pose changed — click **Generate** to update.")
            else:
                _ref_note = (
                    f" · PoseView2 reference: **{pdb_id.upper()}** `{cocrystal_ligand_id}`"
                    if _has_ref else
                    " · No co-crystal ID — only docked pose diagram will be generated."
                )
                st.caption(
                    "**Left:** PoseView v1 — docked pose"
                    " · **Right:** PoseView2 — co-crystal" + _ref_note
                )
        with _cb:
            _run = st.button("🔬 Generate 2D Diagrams", key=btn_key + "_pv_run", type="primary")

        with st.expander("🔍 Test PoseView API", expanded=False):
            st.caption("Sends a known-good test structure (PDB 4AGN) to PoseView.")
            if st.button("▶ Run API Test", key=btn_key + "_pv_diag"):
                with st.spinner("Testing proteins.plus PoseView API…"):
                    try:
                        from core import diagnose_poseview as _diagnose_poseview
                        _diag = _diagnose_poseview()
                        st.session_state[btn_key + "_pv_diag_result"] = _diag
                    except ImportError:
                        st.error("❌ diagnose_poseview not found — please update core.py")
            _diag = st.session_state.get(btn_key + "_pv_diag_result")
            if _diag:
                for _line in _diag["log"]:
                    if _line.startswith("✓"):
                        st.success(_line)
                    else:
                        st.error(_line)

        with st.expander("⬇ Download files for manual PoseView upload", expanded=False):
            _rec_path = st.session_state.get(rec_key, "")
            _pfx3     = rec_key.replace("receptor_fh", "")
            _dl_c1, _dl_c2 = st.columns(2)
            with _dl_c1:
                if _rec_path and os.path.exists(_rec_path):
                    st.download_button(
                        "⬇ receptor.pdb", data=open(_rec_path, "rb"),
                        file_name="receptor.pdb", mime="chemical/x-pdb",
                        key=btn_key + "_pv_dl_rec", width='stretch',
                    )
            with _dl_c2:
                if os.path.exists(pose_sdf_path):
                    st.download_button(
                        "⬇ docked_pose.sdf", data=open(pose_sdf_path, "rb"),
                        file_name=f"pose_{pose_idx+1}_docked.sdf",
                        mime="chemical/x-mdl-sdfile",
                        key=btn_key + "_pv_dl_sdf", width='stretch',
                    )

        if _run:
            _rec = st.session_state.get(rec_key, "")
            if not _rec or not os.path.exists(_rec):
                st.error("Receptor PDB not found.")
            elif not os.path.exists(pose_sdf_path):
                st.error("Pose SDF not found.")
            else:
                with st.spinner("⏳ PoseView v1 — generating 2D diagram… (30–60 s)"):
                    _svg, _err = call_poseview_v1(_rec, pose_sdf_path)
                if _err:
                    st.error(f"❌ PoseView v1 error:\n\n```\n{_err}\n```")
                else:
                    _png = svg_to_png(_svg)
                    st.session_state[img_png_key]  = _png
                    st.session_state[img_svg_key]  = _svg
                    st.session_state[pose_key_key] = _pose_key

                if _has_ref and ref_png_key and ref_svg_key:
                    with st.spinner(f"⏳ PoseView2 — {pdb_id.upper()} / {cocrystal_ligand_id}…"):
                        _ref_svg, _ref_err = call_poseview2_ref(pdb_id, cocrystal_ligand_id)
                    if _ref_err:
                        st.warning(f"⚠️ PoseView2 error:\n\n```\n{_ref_err}\n```")
                    else:
                        st.session_state[ref_png_key] = svg_to_png(_ref_svg)
                        st.session_state[ref_svg_key] = _ref_svg
                st.rerun()

        _pose_svg = st.session_state.get(img_svg_key)
        _ref_svg2 = st.session_state.get(ref_svg_key) if ref_svg_key else None

        if _pose_svg and not _stale:
            _lbl = st.session_state.get(smiles_key, "ligand")[:20]
            if _has_ref:
                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown("##### 🧪 Docked Pose (PoseView v1)")
                    _png_data = st.session_state.get(img_png_key)
                    _show_poseview_image(
                        _png_data, _pose_svg,
                        f"Docked pose {pose_idx+1} — {_lbl}",
                        full_legend=False,
                        stamp=f"Pose {pose_idx+1}  ·  {_lbl}",
                    )
                    _d1, _d2 = st.columns(2)
                    with _d1:
                        if _png_data:
                            st.download_button(
                                "⬇ PNG", data=_png_data,
                                file_name=f"pose{pose_idx+1}_poseview.png",
                                mime="image/png", key=dl_png_key + "_pv", width='stretch',
                            )
                    with _d2:
                        st.download_button(
                            "⬇ SVG", data=_pose_svg,
                            file_name=f"pose{pose_idx+1}_poseview.svg",
                            mime="image/svg+xml", key=dl_svg_key + "_pv", width='stretch',
                        )
                with col_r:
                    st.markdown("##### 🔬 Co-Crystal Reference (PoseView2)")
                    if _ref_svg2:
                        _ref_png2 = st.session_state.get(ref_png_key) if ref_png_key else None
                        _show_poseview_image(
                            _ref_png2, _ref_svg2,
                            f"Co-crystal: {pdb_id.upper()} · {cocrystal_ligand_id}",
                            full_legend=True,
                            stamp=f"{pdb_id.upper()}  ·  {cocrystal_ligand_id}",
                        )
                        _r1, _r2 = st.columns(2)
                        with _r1:
                            if _ref_png2:
                                st.download_button(
                                    "⬇ PNG", data=_ref_png2,
                                    file_name=f"cocrystal_{pdb_id}_{cocrystal_ligand_id}.png",
                                    mime="image/png", key=dl_png_key + "_pv_ref", width='stretch',
                                )
                        with _r2:
                            st.download_button(
                                "⬇ SVG", data=_ref_svg2,
                                file_name=f"cocrystal_{pdb_id}_{cocrystal_ligand_id}.svg",
                                mime="image/svg+xml", key=dl_svg_key + "_pv_ref", width='stretch',
                            )
                    else:
                        st.info("Click **Generate 2D Diagrams** to load the co-crystal reference.")
            else:
                st.markdown("##### 🧪 Docked Pose (PoseView v1)")
                _png_data = st.session_state.get(img_png_key)
                _show_poseview_image(
                    _png_data, _pose_svg,
                    f"Docked pose {pose_idx+1} — {_lbl}",
                    full_legend=False,
                    stamp=f"Pose {pose_idx+1}  ·  {_lbl}",
                )
                _d1, _d2 = st.columns(2)
                with _d1:
                    if _png_data:
                        st.download_button(
                            "⬇ PNG", data=_png_data,
                            file_name=f"pose{pose_idx+1}_poseview.png",
                            mime="image/png", key=dl_png_key + "_pv", width='stretch',
                        )
                with _d2:
                    st.download_button(
                        "⬇ SVG", data=_pose_svg,
                        file_name=f"pose{pose_idx+1}_poseview.svg",
                        mime="image/svg+xml", key=dl_svg_key + "_pv", width='stretch',
                    )
                st.caption("ℹ️ No co-crystal ligand detected.")

            # ── AI prompt for PoseView ──────────────────────────────────────
            if _pose_svg and not _stale:
                _ai_prompt_section(
                    engine="poseview",
                    lig_name=lig_name,
                    pdb_id=pdb_id,
                    binding_energy=binding_energy,
                    has_ref=bool(_has_ref and _ref_svg2),
                    cocrystal_ligand_id=cocrystal_ligand_id,
                    ref_lig_name=ref_lig_name,
                    ref_lig_energy=ref_lig_energy,
                    ref_redocked=(ref_lig_energy is not None),
                    rmsd_crystal=rmsd_crystal,
                    key_suffix=btn_key + "_pv",
                )


# ══════════════════════════════════════════════════════════════════════════════
#  RECEPTOR SECTION (shared between Basic and Batch)
# ══════════════════════════════════════════════════════════════════════════════

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
                    help="CIF recommended for large/newer entries.",
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
        if center_mode == "Auto-detect co-crystal ligand":
            # ── Multi-ligand selector ─────────────────────────────────────
            # Pre-scan the file for qualifying ligands so the user can pick
            # which one to center the grid on before clicking Prepare.
            # We only scan when a file is already on disk (download) or
            # uploaded — skip if neither is ready yet.
            _scan_path = None
            if src == "Download from RCSB" and st.session_state.get(pfx + "pdb_token"):
                _maybe = str(wdir / "raw_prefiltered.pdb")
                if not os.path.exists(_maybe):
                    _maybe = str(wdir / "raw.pdb")
                if os.path.exists(_maybe):
                    _scan_path = _maybe
            elif src != "Download from RCSB" and upload_file is not None:
                _scan_path = str(wdir / "raw_upload_prescan.pdb")
                if not os.path.exists(_scan_path):
                    with open(_scan_path, "wb") as _sf:
                        _sf.write(upload_file.getvalue() if hasattr(upload_file, "getvalue") else upload_file.read())

            if _scan_path and os.path.exists(_scan_path):
                try:
                    from core import scan_ligands as _scan_ligands
                    _found_ligs = _scan_ligands(_scan_path)
                    if len(_found_ligs) > 1:
                        _lig_labels = [
                            f"{d['resname']} (chain {d['chain'] or '—'}, "
                            f"resid {d['resid']}, {d['n_atoms']} atoms)"
                            for d in _found_ligs
                        ]
                        _sel_idx = st.selectbox(
                            f"🔍 {len(_found_ligs)} ligands detected — select grid center:",
                            options=range(len(_lig_labels)),
                            format_func=lambda i: _lig_labels[i],
                            key=pfx + "preferred_lig_idx",
                        )
                        st.session_state[pfx + "preferred_ligand"] = _found_ligs[_sel_idx]["resname"]
                        st.caption(
                            f"All {len(_found_ligs)} ligands will be removed from the receptor. "
                            f"Grid will be centered on the selected one."
                        )
                    elif len(_found_ligs) == 1:
                        st.session_state[pfx + "preferred_ligand"] = _found_ligs[0]["resname"]
                    else:
                        st.session_state[pfx + "preferred_ligand"] = ""
                except Exception:
                    st.session_state.setdefault(pfx + "preferred_ligand", "")
            # ─────────────────────────────────────────────────────────────
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
            )
            st.caption("💡 `resname LIG and chain A` · `resid 701 and chain A`")

    with col_b:
        st.markdown("**Search box size (Å)**")
        sx = st.slider("X size", 10, 40, 16, 2, key=pfx + "sx")
        sy = st.slider("Y size", 10, 40, 16, 2, key=pfx + "sy")
        sz = st.slider("Z size", 10, 40, 16, 2, key=pfx + "sz")
        st.markdown(f"Box volume: **{sx*sy*sz:,} Å³**")

    blind = st.checkbox(
        "🔍 Blind docking (cover whole protein)",
        value=False, key=pfx + "blind_docking",
    )
    if blind:
        st.caption("⚠️ Blind docking — box will cover entire protein extent.")

    with st.expander("⚗️ Cofactor options", expanded=False):
        keep_cofactors = st.checkbox(
            "Keep cofactors in receptor", value=True, key=pfx + "keep_cofactors",
            help="ATP, FAD, NAD, CoA, SAM, etc. Uncheck to dock into a cofactor-free pocket.",
        )
        keep_metals = st.checkbox(
            "Keep metal ions in receptor", value=True, key=pfx + "keep_metals",
            help="ZN, MG, CA, MN, FE, CU, etc. Uncheck to remove metals before docking.",
        )
        _strip_set = _COFACTOR_NAMES | _HEME_RESNAMES
        from core import METAL_RESNAMES as _METAL_RESNAMES
        _strip_sorted_cof   = ", ".join(sorted(_strip_set))
        _strip_sorted_metal = ", ".join(sorted(_METAL_RESNAMES))
        if keep_cofactors:
            st.caption(f"✅ Cofactors **kept**: {_strip_sorted_cof}")
        else:
            st.caption(f"⚠️ Cofactors **stripped**: {_strip_sorted_cof}")
        if keep_metals:
            st.caption(f"✅ Metal ions **kept**: {_strip_sorted_metal}")
        else:
            st.caption(f"⚠️ Metal ions **stripped**: {_strip_sorted_metal}")

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
            rc, _ = _run_cmd(["curl", "-sf", _dl_url, "-o", raw_path])
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
                        st.error(f"❌ Download failed for {token}")
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
            raw_path = str(wdir / ("raw.cif" if _up_ext in (".cif", ".mmcif") else "raw.pdb"))
            with open(raw_path, "wb") as f:
                f.write(upload_file.read())
            st.session_state[pfx + "pdb_token"] = Path(upload_file.name).stem

        # ── Deduplicate identical protein chains ──────────────────────────
        try:
            from prody import parsePDB as _pPDB_ch, writePDB as _wPDB_ch
            _atoms_ch = _pPDB_ch(raw_path)
            if _atoms_ch is not None:
                _hv_ch = _atoms_ch.getHierView()
                _prot_chains = [
                    _c for _c in _hv_ch.iterChains()
                    if _atoms_ch.select(f"chain {_c.getChid()} and protein") is not None
                ]
                if len(_prot_chains) > 1:
                    _prot_chains.sort(key=lambda c: (c.getChid() != "A", c.getChid()))
                    _seen_seqs = {}
                    _keep_chids = []
                    for _c in _prot_chains:
                        _seq = "".join(
                            r.getResname() for r in _c.iterResidues()
                            if r.getResname() not in ("HOH", "WAT", "DOD")
                        )
                        if _seq not in _seen_seqs:
                            _seen_seqs[_seq] = _c.getChid()
                            _keep_chids.append(_c.getChid())
                    _dup_chids = [c.getChid() for c in _prot_chains if c.getChid() not in _keep_chids]
                    if _dup_chids:
                        _ch_sel_str = " ".join(f"chain {c}" for c in _keep_chids)
                        _atoms_filt = _atoms_ch.select(_ch_sel_str)
                        _raw_chain_path = str(wdir / "raw_chain_filtered.pdb")
                        _wPDB_ch(_raw_chain_path, _atoms_filt)
                        raw_path = _raw_chain_path
                        st.info(
                            f"Multiple identical chain(s) detected — "
                            f"kept: **{', '.join(_keep_chids)}** · "
                            f"removed duplicate(s): {', '.join(_dup_chids)}"
                        )
        except Exception:
            pass

        # ── Pre-filter cofactors / heme / metals ──────────────────────────
        _keep        = st.session_state.get(pfx + "keep_cofactors", True)
        _keep_metals = st.session_state.get(pfx + "keep_metals", True)
        _cofactor_strip = _COFACTOR_NAMES if not _keep else set()

        # Import metal resnames for optional stripping
        try:
            from core import METAL_RESNAMES as _MRNS
        except ImportError:
            _MRNS = {"MG","ZN","CA","MN","FE","CU","CO","NI","CD","HG","NA","K"}
        _metal_strip = _MRNS if not _keep_metals else set()

        _filtered_path = str(wdir / "raw_prefiltered.pdb")
        _heme_lines    = []
        _n_cofactor    = 0
        _n_metal       = 0

        with open(raw_path) as _fin, open(_filtered_path, "w") as _fout:
            for _line in _fin:
                _field = _line[:6].strip()
                if _field in ("ATOM", "HETATM"):
                    _rn = _line[17:20].strip().upper()
                    if _rn in _HEME_RESNAMES:
                        _heme_lines.append(_line)
                        continue
                    if _rn in _cofactor_strip:
                        _n_cofactor += 1
                        continue
                    # Strip metals if user unchecked "Keep metal ions"
                    # (they won't be re-injected → not scored by Vina)
                    if _rn in _metal_strip:
                        _n_metal += 1
                        continue
                _fout.write(_line)

        raw_path = _filtered_path

        # ── Compute heme center for auto-detect fallback ───────────────────
        # If heme is the only notable feature (no drug-like ligand present),
        # use the Fe atom (or heme centroid) as the auto-detected grid center.
        _heme_center = None
        if _heme_lines:
            try:
                _fe_lines = [l for l in _heme_lines
                             if l[12:16].strip().upper() == "FE"]
                _ref_lines = _fe_lines if _fe_lines else _heme_lines
                _hxs = [float(l[30:38]) for l in _ref_lines]
                _hys = [float(l[38:46]) for l in _ref_lines]
                _hzs = [float(l[46:54]) for l in _ref_lines]
                _heme_center = (
                    sum(_hxs) / len(_hxs),
                    sum(_hys) / len(_hys),
                    sum(_hzs) / len(_hzs),
                )
            except Exception:
                pass

        if _heme_lines:
            _heme_names = ", ".join(sorted({l[17:20].strip() for l in _heme_lines}))
            st.info(
                f"Heme ({_heme_names}, {len(_heme_lines)} atoms) stripped before "
                f"OpenBabel — will be re-injected with AD4 atom types."
            )
        if _n_cofactor:
            st.info(f"Stripped {_n_cofactor} cofactor atom(s) from receptor.")
        if _n_metal:
            st.info(f"Stripped {_n_metal} metal ion atom(s) from receptor (will not be re-injected).")

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

        # ── Blind docking ─────────────────────────────────────────────────
        _blind = st.session_state.get(pfx + "blind_docking", False)
        if _blind:
            try:
                import numpy as _np
                from prody import parsePDB as _parsePDB
                _atoms = _parsePDB(raw_path)
                _prot  = _atoms.select("protein") or _atoms
                _coords = _prot.getCoords()
                _mn = _coords.min(axis=0); _mx = _coords.max(axis=0)
                _ctr = ((_mn + _mx) / 2).tolist()
                _pad = 4.0
                _sz  = ((_mx - _mn) + 2 * _pad).tolist()
                _core_mode  = "manual"
                _manual_xyz = tuple(_ctr)
                sx = int(round(_sz[0])); sy = int(round(_sz[1])); sz = int(round(_sz[2]))
                st.info(
                    f"🔍 Blind docking box — "
                    f"center ({_ctr[0]:.1f}, {_ctr[1]:.1f}, {_ctr[2]:.1f}) · "
                    f"size {sx} × {sy} × {sz} Å"
                )
            except Exception as _be:
                st.warning(f"⚠️ Could not compute blind docking box: {_be}")

        with st.spinner("⏳ Preparing receptor…"):
            result = prepare_receptor(
                raw_pdb          = raw_path,
                wdir             = wdir,
                center_mode      = _core_mode,
                manual_xyz       = _manual_xyz,
                prody_sel        = _prody_sel,
                box_size         = (sx, sy, sz),
                preferred_ligand = st.session_state.get(pfx + "preferred_ligand", ""),
            )

        if result["success"]:
            # ── Heme center fallback ────────────────────────────────────────
            # If auto-detect found no drug-like ligand but heme was present,
            # re-center the grid on the Fe atom (substrate binding site).
            if (_core_mode == "auto"
                    and not result.get("cocrystal_ligand_id")
                    and _heme_center is not None):
                from core import write_vina_config as _wvc, write_box_pdb as _wbp
                _hcx, _hcy, _hcz = _heme_center
                _wbp(result["box_pdb"],    _hcx, _hcy, _hcz, result["sx"], result["sy"], result["sz"])
                _wvc(result["config_txt"], _hcx, _hcy, _hcz, result["sx"], result["sy"], result["sz"])
                result["cx"] = _hcx; result["cy"] = _hcy; result["cz"] = _hcz
                _fe_found = any(l[12:16].strip().upper() == "FE" for l in _heme_lines)
                st.info(
                    f"🧲 No co-crystal ligand found — grid auto-centered at "
                    f"{'Fe' if _fe_found else 'heme centroid'} "
                    f"({_hcx:.2f}, {_hcy:.2f}, {_hcz:.2f})"
                )
            # ── Re-inject heme ─────────────────────────────────────────────
            _heme_log = []
            if _heme_lines:
                _AD4_TYPE = {"FE": "Fe", "N": "NA", "O": "OA", "C": "A", "S": "SA"}
                _AD4_CHG  = {"FE": 2.0, "N": -0.4, "C": 0.1, "O": -0.4, "S": 0.0}
                try:
                    _pdbqt_path  = result["rec_pdbqt"]
                    _pdbqt_lines = [
                        l for l in open(_pdbqt_path).readlines()
                        if l.strip() != "END"
                    ]
                    _injected = 0
                    for _hl in _heme_lines:
                        try:
                            _serial  = int(_hl[6:11])
                            _aname   = _hl[12:16].strip()
                            _resname = _hl[17:20].strip().upper()
                            _chain   = _hl[21] if len(_hl) > 21 else "A"
                            _resid   = int(_hl[22:26])
                            _x       = float(_hl[30:38])
                            _y       = float(_hl[38:46])
                            _z       = float(_hl[46:54])
                            _el_raw  = (
                                _hl[76:78].strip().upper()
                                if len(_hl) > 76 and _hl[76:78].strip()
                                else _aname[:2].strip().upper()
                            )
                            _el      = _el_raw.upper()
                            _atype   = _AD4_TYPE.get(_el, "C")
                            _charge  = _AD4_CHG.get(_el, 0.0)
                            # Right-justify atom type in 2 chars for valid PDBQT
                            _vina_type = f"{_atype:>2s}"
                            _pdbqt_lines.append(
                                f"HETATM{_serial:5d} {_aname:<4s} {_resname:<3s} "
                                f"{_chain}{_resid:4d}    "
                                f"{_x:8.3f}{_y:8.3f}{_z:8.3f}  1.00  0.00"
                                f"    {_charge:+.3f} {_vina_type}\n"
                            )
                            _injected += 1
                        except Exception as _he:
                            _heme_log.append(f"  Could not re-inject heme line: {_he}")
                    _pdbqt_lines.append("END\n")
                    with open(_pdbqt_path, "w") as _pf:
                        _pf.writelines(_pdbqt_lines)

                    with open(result["rec_fh"], "a") as _rf:
                        _rf.writelines(_heme_lines)

                    _heme_log.append(
                        f"Re-injected {_injected} heme atom(s) into PDBQT and rec.pdb"
                    )
                except Exception as _he2:
                    _heme_log.append(f"Heme re-injection failed: {_he2}")

            _full_log = result["log"] + _heme_log
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
                pfx + "receptor_log":        "\n".join(_full_log),
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

        st.markdown(
            f"{_pill('Receptor ready', 'success')} {_pill(token)} "
            f"{_pill(f'Center ({cx_v:.2f}, {cy_v:.2f}, {cz_v:.2f})')} "
            f"{_pill(f'Box {_sx}x{_sy}x{_sz} A')}"
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

            _rec_path = st.session_state.get(pfx + "receptor_fh")
            if _rec_path and os.path.exists(_rec_path):
                v3.addModel(open(_rec_path).read(), "pdb")
                v3.setStyle({"model": mi}, {"cartoon": {"color": "spectrum", "opacity": 0.4}})
                try:
                    from prody import parsePDB as _pPDB
                    _ra = _pPDB(_rec_path)
                    _hx, _hy, _hz = _sx / 2.0, _sy / 2.0, _sz / 2.0
                    _pocket = _ra.select(
                        f"protein and "
                        f"x > {cx_v - _hx:.2f} and x < {cx_v + _hx:.2f} and "
                        f"y > {cy_v - _hy:.2f} and y < {cy_v + _hy:.2f} and "
                        f"z > {cz_v - _hz:.2f} and z < {cz_v + _hz:.2f}"
                    )
                    if _pocket is not None and _pocket.numAtoms() > 0:
                        _resi_list = sorted(set(int(r) for r in _pocket.getResnums()))
                        v3.setStyle(
                            {"model": mi, "resi": _resi_list},
                            {"stick": {"colorscheme": "whiteCarbon", "radius": 0.18},
                             "cartoon": {"color": "spectrum", "opacity": 0.75}},
                        )
                except Exception:
                    pass
                mi += 1

            _box_mi   = None
            _box_path = st.session_state.get(pfx + "box_pdb")
            if _box_path and os.path.exists(_box_path):
                v3.addModel(open(_box_path).read(), "pdb")
                v3.setStyle({"model": mi}, {"stick": {"radius": 0.2, "color": "gray"}})
                _box_mi = mi
                mi += 1

            lig_p = st.session_state.get(pfx + "ligand_pdb_path")
            if lig_p and os.path.exists(lig_p):
                v3.addModel(open(lig_p).read(), "pdb")
                v3.setStyle({"model": mi}, {
                    "stick": {"colorscheme": "magentaCarbon", "radius": 0.25}
                })
                mi += 1

            # ── Heme cofactor ─────────────────────────────────────────────
            mi = _add_heme_to_view(v3, st.session_state.get(pfx + "receptor_fh"), mi)

            _add_box_to_view(v3, cx_v, cy_v, cz_v, _sx, _sy, _sz)
            try:
                for _end, _col, _lbl in [
                    ({"x": cx_v+8, "y": cy_v,   "z": cz_v},   "red",   "X"),
                    ({"x": cx_v,   "y": cy_v+8,  "z": cz_v},   "green", "Y"),
                    ({"x": cx_v,   "y": cy_v,    "z": cz_v+8}, "blue",  "Z"),
                ]:
                    _st = {"x": cx_v, "y": cy_v, "z": cz_v}
                    v3.addArrow({"start": _st, "end": _end, "radius": 0.15, "color": _col, "radiusRatio": 3.0})
                    v3.addLabel(_lbl, {
                        "fontSize": 14, "fontColor": _col,
                        "backgroundColor": "black", "backgroundOpacity": 0.6,
                        "inFront": True, "showBackground": True,
                    }, _end)
            except Exception:
                pass

            if _box_mi is not None:
                v3.zoomTo({"model": _box_mi})
            else:
                v3.zoomTo()
                v3.center({"x": float(cx_v), "y": float(cy_v), "z": float(cz_v)})
                v3.zoom(1.5)
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
st.markdown("**Basic** — single ligand. **Batch** — multiple ligands.")
st.markdown("**☁️ Cloud-ready | 📱 Mobile-compatible**")

if VINA_PATH is None:
    st.error(f"❌ Could not download Vina binary: {_vina_err}")
    st.stop()

if not _OBABEL_OK:
    st.error("❌ OpenBabel not found. Add `openbabel` to packages.txt and redeploy.")
    st.stop()

st.markdown(f"{_pill('Vina 1.2.7 ready', 'success')} ", unsafe_allow_html=True)
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
        ["SMILES string", "Upload structure (.pdb/.mol2)", "Draw structure (Ketcher)"],
        horizontal=True, key="lig_input_mode",
    )

    smiles_in = ""
    if lig_input_mode == "SMILES string":
        smiles_in = st.text_input(
            "SMILES string",
            value="COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
            key="smiles_in",
        )
    elif lig_input_mode == "Upload structure (.pdb/.mol2)":
        st.file_uploader(
            "Upload structure file", type=["sdf", "mol2", "pdb"],
            key="lig_struct_file",
        )
        lig_upload_prot = st.radio(
            "Ligand preparation mode",
            ["Use the uploaded form", "Protonation at target pH"],
            horizontal=True, key="lig_upload_prot",
        )
    else:
        try:
            from streamlit_ketcher import st_ketcher
            _k = st_ketcher(st.session_state.get("ketcher_smi", ""), height=400, key="ketcher_widget")
            if _k:
                st.session_state["ketcher_smi"] = _k
                smiles_in = _k
            else:
                smiles_in = st.session_state.get("ketcher_smi", "")
        except ImportError:
            st.error("❌ `streamlit-ketcher` not installed")
            smiles_in = ""

    lig_name_in = st.text_input("Output name", value="ELR", key="lig_name_in")
    ph_in       = st.number_input("Target pH", 0.0, 14.0, 7.4, 0.1, key="ph_in")

    # ── Protonation mode ──────────────────────────────────────────────────────
    prot_mode = st.radio(
        "Protonation mode",
        ["⚡ Fast (Dimorphite-DL)", "🧪 pKaNET Cloud (recommended)", "🔬 Neutral (add H only)"],
        horizontal=True, key="prot_mode",
        help=(
            "**Fast**: Dimorphite-DL SMARTS rules, ~1–3 s. "
            "**pKaNET Cloud**: Dimorphite + PubChem pKa lookup + "
            "Henderson–Hasselbalch microstate ranking (Hengphasatporn et al. 2026), "
            "~2–5 s (cached after first run). "
            "**Neutral**: no ionization — adds all H in neutral form, ~0.5 s."
        ),
    )
    _prot_mode_map = {
        "⚡ Fast (Dimorphite-DL)":          "dimorphite",
        "🧪 pKaNET Cloud (recommended)":    "pkanet",
        "🔬 Neutral (add H only)":          "neutral",
    }
    _prot_mode_key = _prot_mode_map[prot_mode]

    _use_pubchem = True
    if _prot_mode_key == "pkanet":
        _use_pubchem = st.checkbox(
            "Query PubChem for experimental pKa (cached, ~1s per new compound)",
            value=True, key="use_pubchem",
            help="Looks up experimental dissociation constants on PubChem by InChIKey. "
                 "Results are cached in memory so repeated runs are instant. "
                 "Uncheck for faster batch docking of novel/proprietary compounds."
        )
        if _prot_mode_key == "pkanet" and _use_pubchem:
            st.caption(
                "💡 PubChem is queried once per unique compound (InChIKey cache). "
                "Novel or proprietary structures not in PubChem fall back gracefully "
                "to heuristic SMARTS pKa table."
            )
    # ─────────────────────────────────────────────────────────────────────────

    if not st.session_state.receptor_done:
        st.caption("⚠ Complete Step 1 first.")
    if st.button(
        "▶ Prepare Ligand", key="btn_ligand", type="primary",
        disabled=not st.session_state.receptor_done,
    ):
        lig_name = lig_name_in.strip() or "LIG"
        with st.spinner("Preparing ligand…"):
            _mode = st.session_state.get("lig_input_mode", "SMILES string")
            _prot_mode_key  = st.session_state.get("prot_mode", "⚡ Fast (Dimorphite-DL)")
            _prot_mode_key  = {"⚡ Fast (Dimorphite-DL)": "dimorphite",
                                "🧪 pKaNET Cloud (recommended)": "pkanet",
                                "🔬 Neutral (add H only)": "neutral"}.get(_prot_mode_key, "dimorphite")
            _use_pubchem    = st.session_state.get("use_pubchem", True)

            if "Upload" in _mode:
                _sfobj = st.session_state.get("lig_struct_file")
                if _sfobj is None:
                    st.error("No structure file uploaded"); st.stop()
                _ext = Path(_sfobj.name).suffix.lower()
                _tmp = str(WORKDIR / f"lig_upload{_ext}")
                with open(_tmp, "wb") as _f:
                    _f.write(_sfobj.read())
                _upload_prot = st.session_state.get("lig_upload_prot", "Use the uploaded form")
                if _upload_prot == "Use the uploaded form":
                    result = prepare_ligand_from_file(_tmp, lig_name, WORKDIR)
                else:
                    try:
                        smiles_in = smiles_from_file(_tmp, WORKDIR)
                    except Exception as e:
                        st.error(f"❌ Could not read structure: {e}"); st.stop()
                    result = prepare_ligand(smiles_in, lig_name, ph_in, WORKDIR,
                                            mode=_prot_mode_key, use_pubchem=_use_pubchem)
            elif "Ketcher" in _mode:
                smiles_in = st.session_state.get("ketcher_smi", "").strip()
                if not smiles_in:
                    st.error("No molecule drawn in Ketcher."); st.stop()
                result = prepare_ligand(smiles_in, lig_name, ph_in, WORKDIR,
                                        mode=_prot_mode_key, use_pubchem=_use_pubchem)
            else:
                result = prepare_ligand(smiles_in, lig_name, ph_in, WORKDIR,
                                        mode=_prot_mode_key, use_pubchem=_use_pubchem)

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
                _png_to_b64_img(buf.getvalue(), style="width:100%;max-width:320px;height:auto;border-radius:6px;")
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
        )
        if do_redock:
            st.text_input(
                "Co-crystal SMILES [name]",
                value="COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC Erlotinib",
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

        redock_score  = None
        redock_result = None
        if st.session_state.get("do_redock"):
            raw_rd = st.session_state.get("redock_smiles", "").strip()
            pts    = raw_rd.split(None, 1)
            rd_smi = pts[0]
            rd_nm  = pts[1].replace(" ", "_") if len(pts) > 1 else "redock"
            ph_val = st.session_state.get("ph_in", 7.4)
            _rd_prot_mode = st.session_state.get("prot_mode", "⚡ Fast (Dimorphite-DL)")
            _rd_prot_mode = {"⚡ Fast (Dimorphite-DL)": "dimorphite",
                              "🧪 pKaNET Cloud (recommended)": "pkanet",
                              "🔬 Neutral (add H only)": "neutral"}.get(_rd_prot_mode, "dimorphite")
            _rd_use_pubchem = st.session_state.get("use_pubchem", True)
            with st.spinner(f"Docking reference ligand ({rd_nm})…"):
                rd_prep = prepare_ligand(rd_smi, "redock_" + rd_nm, ph_val, WORKDIR,
                                         mode=_rd_prot_mode, use_pubchem=_rd_use_pubchem)
                if rd_prep["success"]:
                    rd_dock = run_vina(
                        st.session_state.receptor_pdbqt, rd_prep["pdbqt"],
                        st.session_state.config_txt,
                        VINA_PATH, exh, nm, er, WORKDIR, "redock_" + rd_nm,
                    )
                    if rd_dock["success"] and rd_dock["top_score"] is not None:
                        redock_score = rd_dock["top_score"]
                        rd_pv_sdf    = str(WORKDIR / f"redock_{rd_nm}_pv_ready.sdf")
                        fix_sdf_bond_orders(rd_dock["out_sdf"], rd_smi, rd_pv_sdf)
                        if not os.path.exists(rd_pv_sdf) or os.path.getsize(rd_pv_sdf) < 10:
                            rd_pv_sdf = rd_dock["out_sdf"]
                        rd_n = len(load_mols_from_sdf(rd_dock["out_sdf"], sanitize=False)) if os.path.exists(rd_dock["out_sdf"]) else 0
                        redock_result = {
                            "Name": f"⭐ {rd_nm} (co-crystal ref)", "ref_name": rd_nm,
                            "SMILES": rd_smi, "prot_smiles": rd_prep["prot_smiles"],
                            "Charge": rd_prep["charge"], "Top Score": redock_score,
                            "pose_scores": [s["affinity"] for s in rd_dock["scores"]],
                            "Poses": rd_n, "out_pdbqt": rd_dock["out_pdbqt"],
                            "out_sdf": rd_dock["out_sdf"], "pv_sdf": rd_pv_sdf,
                            "Status": "OK", "is_redock": True,
                        }
                        st.success(f"✓ Reference score: **{redock_score:.2f} kcal/mol** ({rd_nm})")
                    else:
                        st.warning("⚠ Redocking failed — no score returned")
                else:
                    st.warning(f"⚠ Reference ligand prep failed: {rd_prep.get('error')}")

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
            pv_log = fix_sdf_bond_orders(dock["out_sdf"], st.session_state.prot_smiles, pv_sdf)
            if not os.path.exists(pv_sdf) or os.path.getsize(pv_sdf) < 10:
                pv_sdf = dock["out_sdf"]
            if dock["scores"]:
                _cryst_pdb_df = st.session_state.get("ligand_pdb_path") or ""
                _pose_mols_df = load_mols_from_sdf(dock["out_sdf"], sanitize=False) if os.path.exists(dock["out_sdf"]) else []
                _rows = []
                for s in dock["scores"]:
                    _pose_num = s["pose"]
                    _rmsd_crystal = None
                    if _cryst_pdb_df and os.path.exists(_cryst_pdb_df):
                        _mol_idx = (_pose_num - 1) if _pose_num else 0
                        if _mol_idx < len(_pose_mols_df):
                            _rmsd_crystal = calc_rmsd_heavy(_pose_mols_df[_mol_idx], _cryst_pdb_df)
                    _rows.append({
                        "Pose": _pose_num,
                        "Affinity (kcal/mol)": s["affinity"],
                        "RMSD vs crystal (Å)": round(_rmsd_crystal, 2) if _rmsd_crystal is not None else "—",
                    })
                df = pd.DataFrame(_rows).sort_values("Affinity (kcal/mol)").reset_index(drop=True)
            else:
                df = None

            mols = load_mols_from_sdf(dock["out_sdf"], sanitize=False) if os.path.exists(dock["out_sdf"]) else []
            _full_log = dock["log"] + "\n\n── Bond-order fix ──\n" + "\n".join(pv_log)
            st.session_state.update({
                "output_pdbqt": dock["out_pdbqt"], "output_sdf": dock["out_sdf"],
                "output_pv_sdf": pv_sdf, "dock_base": base,
                "docking_done": True, "docking_log": _full_log,
                "score_df": df, "pose_mols": mols,
                "pv_image_png": None, "pv_image_svg": None, "pv_pose_key": None,
                "dock_run_id": st.session_state.get("dock_run_id", 0) + 1,
                "redock_done": redock_result is not None,
                "redock_score": redock_score, "redock_result": redock_result,
                "confirmed_ref_score": None, "confirmed_ref_pose": None, "confirmed_ref_name": None,
            })

    if st.session_state.docking_done:
        _redock_score = st.session_state.get("redock_score")
        st.markdown(
            _pill("Docking complete", "success")
            + (_pill(f"Ref: {_redock_score:.2f} kcal/mol", "warn") if _redock_score is not None else ""),
            unsafe_allow_html=True,
        )
        with st.expander("📋 Vina output log", expanded=False):
            st.markdown(f'<div class="log-box">{st.session_state.docking_log}</div>', unsafe_allow_html=True)
        if st.session_state.score_df is not None:
            best = st.session_state.score_df["Affinity (kcal/mol)"].min()
            cls  = "Very strong" if best < -11 else "Strong" if best < -9 else "Moderate" if best < -7 else "Weak"
            st.markdown(
                f'<div class="score-best">{best:.2f} <span class="score-unit">kcal/mol</span></div>'
                f'<div style="color:#8b949e;font-size:0.9rem;margin-bottom:12px">Best pose — {cls} predicted binding</div>',
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
                _run_id = st.session_state.get("dock_run_id", 0)
                _fmt = {"Affinity (kcal/mol)": "{:.2f}"}
                _styled = (
                    df.style
                    .background_gradient(cmap="RdYlGn", subset=["Affinity (kcal/mol)"], gmap=-df["Affinity (kcal/mol)"])
                    .format(_fmt)
                )
                st.dataframe(_styled, hide_index=True, width='stretch', key=f"score_table_{_run_id}")
        with cc:
            st.markdown("**Affinity by Pose**")
            if df is not None:
                _cc = _chart_colors()
                fig, ax = plt.subplots(figsize=(6, 3.5))
                fig.patch.set_facecolor(_cc["bg"]); ax.set_facecolor(_cc["bg_sub"])
                cols = ["#3fb950" if v == df["Affinity (kcal/mol)"].min() else "#58a6ff" for v in df["Affinity (kcal/mol)"]]
                ax.bar(df["Pose"].astype(str), df["Affinity (kcal/mol)"], color=cols, edgecolor=_cc["border"], linewidth=0.6)
                ax.invert_yaxis()
                _ref_score_plot = st.session_state.get("confirmed_ref_score") or st.session_state.get("redock_score")
                if _ref_score_plot is not None:
                    _ref_nm = st.session_state.get("confirmed_ref_name") or "co-crystal ref"
                    ax.axhline(_ref_score_plot, color="#f85149", linewidth=1.8, linestyle="--", label=f"{_ref_nm}: {_ref_score_plot:.2f}")
                    ax.legend(facecolor=_cc["legend_bg"], edgecolor=_cc["border"], labelcolor=_cc["text"], fontsize=8)
                ax.set_xlabel("Pose", color=_cc["muted"], fontsize=9)
                ax.set_ylabel("Affinity (kcal/mol)", color=_cc["muted"], fontsize=9)
                ax.tick_params(colors=_cc["muted"], labelsize=8)
                for sp in ax.spines.values(): sp.set_edgecolor(_cc["border"])
                fig.tight_layout()
                st.pyplot(fig, width='stretch')
                plt.close(fig)

        st.markdown("---")

        # ── Redocking Reference Browser ───────────────────────────────────
        _redock_result = st.session_state.get("redock_result")
        if _redock_result and _redock_result.get("out_sdf") and os.path.exists(_redock_result["out_sdf"]):
            st.markdown("**⭐ Redocking Reference**")
            _rd_mols = load_mols_from_sdf(_redock_result["out_sdf"], sanitize=False)
            if _rd_mols:
                _rd_pose_i = st.slider("Reference pose", 1, len(_rd_mols), 1, key="rd_pose_sel") - 1
                _rd_scores = _redock_result.get("pose_scores", [])
                _rd_this_score = _rd_scores[_rd_pose_i] if _rd_pose_i < len(_rd_scores) else _redock_result.get("Top Score")
                _rsk = "success" if (_rd_this_score is not None and _rd_this_score < -8) else "warn"
                _rd_pills = _pill(f"Pose {_rd_pose_i+1}/{len(_rd_mols)}")
                if _rd_this_score is not None:
                    _rd_pills += f" {_pill(f'{_rd_this_score:.2f} kcal/mol', _rsk)}"

                _cryst_pdb_rd = st.session_state.get("ligand_pdb_path") or ""
                if _cryst_pdb_rd and os.path.exists(_cryst_pdb_rd):
                    _rmsd_rd = calc_rmsd_heavy(_rd_mols[_rd_pose_i], _cryst_pdb_rd)
                    if _rmsd_rd is not None:
                        _rk = "success" if _rmsd_rd <= 2.0 else "warn" if _rmsd_rd <= 3.0 else "info"
                        _rd_pills += f" {_pill(f'RMSD {_rmsd_rd:.2f} A vs crystal', _rk)}"

                st.markdown(_pill("⭐ Co-crystal reference ligand", "warn") + " " + _rd_pills, unsafe_allow_html=True)

                _rd_v_col, _rd_a_col = st.columns([3, 1])
                with _rd_v_col:
                    try:
                        _vrd = py3Dmol.view(width="100%", height=400)
                        _vrd.setBackgroundColor(_viewer_bg())
                        _mrd = 0
                        if st.session_state.receptor_fh and os.path.exists(st.session_state.receptor_fh):
                            _vrd.addModel(open(st.session_state.receptor_fh).read(), "pdb")
                            _vrd.setStyle({"model": _mrd}, {"cartoon": {"color": "spectrum", "opacity": 0.7}, "stick": {"radius": 0.08, "opacity": 0.15}})
                            _mrd += 1
                        _lig_p_rd = st.session_state.get("ligand_pdb_path")
                        if _lig_p_rd and os.path.exists(_lig_p_rd):
                            _vrd.addModel(open(_lig_p_rd).read(), "pdb")
                            _vrd.setStyle({"model": _mrd}, {"stick": {"colorscheme": "magentaCarbon", "radius": 0.2}})
                            _mrd += 1
                        # Heme
                        _mrd = _add_heme_to_view(_vrd, st.session_state.get("receptor_fh"), _mrd)
                        _vrd.addModel(Chem.MolToMolBlock(_rd_mols[_rd_pose_i]), "mol")
                        _vrd.setStyle({"model": _mrd}, {"stick": {"colorscheme": "cyanCarbon", "radius": 0.28}})
                        _vrd.addSurface("SES", {"opacity": 0.2, "color": "lightblue"}, {"model": 0}, {"model": _mrd})
                        _vrd.zoomTo({"model": _mrd})
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
                            f"✅ Confirmed (pose {_rd_pose_i+1})" if _already else f"📌 Use pose {_rd_pose_i+1} as reference",
                            key="confirm_ref_btn",
                            type="secondary" if _already else "primary", width='stretch',
                        ):
                            _rd_nm = _redock_result.get("ref_name", "co-crystal ref")
                            st.session_state.update({
                                "confirmed_ref_score": _rd_this_score,
                                "confirmed_ref_pose": _rd_pose_i + 1,
                                "confirmed_ref_name": _rd_nm,
                            })
                            st.rerun()
                        if _c_ref_score is not None and not _already:
                            if st.button("🔄 Reset reference", key="reset_ref_btn", width='stretch'):
                                st.session_state.update({
                                    "confirmed_ref_score": None, "confirmed_ref_pose": None, "confirmed_ref_name": None,
                                })
                                st.rerun()
                    st.markdown("**Download**")
                    _rd_safe = _redock_result.get("ref_name", "redock")
                    _sp_rd   = str(WORKDIR / f"redock_pose{_rd_pose_i+1}.sdf")
                    write_single_pose(_rd_mols[_rd_pose_i], _sp_rd)
                    st.download_button(f"⬇ Ref pose {_rd_pose_i+1} (.sdf)", open(_sp_rd, "rb"),
                        file_name=f"redock_{_rd_safe}_pose{_rd_pose_i+1}.sdf", key="dl_rd_pose", width='stretch')
                    if _redock_result.get("out_pdbqt") and os.path.exists(_redock_result["out_pdbqt"]):
                        st.download_button("⬇ All ref poses (.pdbqt)", open(_redock_result["out_pdbqt"], "rb"),
                            file_name=f"redock_{_rd_safe}_out.pdbqt", key="dl_rd_pdbqt", width='stretch')
            st.markdown("---")

        # ── Animated Pose Viewer ──────────────────────────────────────────
        st.markdown("**🎬 Animated Pose Viewer**")
        anim_spd = st.slider("Interval (ms)", 500, 3000, 1500, 250, key="anim_spd")
        if st.session_state.output_sdf and os.path.exists(st.session_state.output_sdf):
            sdf_txt = open(st.session_state.output_sdf).read()
            va = py3Dmol.view(width="100%", height=440)
            va.setBackgroundColor(_viewer_bg())
            mai = 0
            if st.session_state.receptor_fh and os.path.exists(st.session_state.receptor_fh):
                va.addModel(open(st.session_state.receptor_fh).read(), "pdb")
                va.setStyle({"model": mai}, {"cartoon": {"color": "spectrum", "opacity": 0.7}, "stick": {"radius": 0.1, "opacity": 0.2}})
                mai += 1
            if st.session_state.ligand_pdb_path and os.path.exists(st.session_state.ligand_pdb_path):
                va.addModel(open(st.session_state.ligand_pdb_path).read(), "pdb")
                va.setStyle({"model": mai}, {"stick": {"colorscheme": "magentaCarbon", "radius": 0.22}})
                mai += 1
            # ── Heme ──────────────────────────────────────────────────────
            mai = _add_heme_to_view(va, st.session_state.get("receptor_fh"), mai)
            # ─────────────────────────────────────────────────────────────
            va.addModelsAsFrames(sdf_txt)
            va.setStyle({"model": mai}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.25}})
            va.animate({"interval": anim_spd, "loop": "forward"})
            va.addSurface("SES", {"opacity": 0.18, "color": "lightblue"}, {"model": 0}, {"model": mai})
            va.zoomTo(); va.center({"model": mai}); va.rotate(30)
            show3d(va, height=440)

        st.markdown("---")

        # ── Interactive Pose Selector ─────────────────────────────────────
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
                    _pills_str  = f"{_pill(f'Pose {pose_idx+1}/{len(mols)}')} {_pill(f'Affinity: {aff:.2f} kcal/mol', _score_kind)}"
                    if _cryst_pdb and os.path.exists(_cryst_pdb):
                        _rmsd = calc_rmsd_heavy(sel_mol, _cryst_pdb)
                        if _rmsd is not None:
                            _rk = "success" if _rmsd <= 2.0 else "warn" if _rmsd <= 3.0 else "info"
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
                        v2.setStyle({"model": mi2}, {"cartoon": {"color": "spectrum", "opacity": 0.5}, "stick": {"radius": 0.08, "opacity": 0.15}})
                        mi2 += 1
                    if _cryst_pdb and os.path.exists(_cryst_pdb):
                        v2.addModel(open(_cryst_pdb).read(), "pdb")
                        v2.setStyle({"model": mi2}, {"stick": {"colorscheme": "magentaCarbon", "radius": 0.2}})
                        mi2 += 1
                    # ── Heme ──────────────────────────────────────────────
                    mi2 = _add_heme_to_view(v2, st.session_state.get("receptor_fh"), mi2)
                    # ─────────────────────────────────────────────────────
                    v2.addModel(Chem.MolToMolBlock(sel_mol), "mol")
                    v2.setStyle({"model": mi2}, {"stick": {"colorscheme": "cyanCarbon", "radius": 0.28}})
                    v2.addSurface("SES", {"opacity": 0.2, "color": "lightblue"}, {"model": 0}, {"model": mi2})
                    v2.zoomTo({"model": mi2})
                    show3d(v2, height=400)
                except Exception as e:
                    st.info(f"Viewer error: {e}")

            with cdl:
                st.markdown("**Download**")
                sp_raw = str(WORKDIR / f"pose_{pose_idx+1}_raw.sdf")
                write_single_pose(sel_mol, sp_raw)
                st.download_button(f"⬇ Pose {pose_idx+1} (.sdf)", open(sp_raw, "rb"),
                    file_name=f"pose_{pose_idx+1}.sdf", key=f"dl_p_{pose_idx}", width='stretch')
                st.download_button("⬇ All poses (.pdbqt)", open(st.session_state.output_pdbqt, "rb"),
                    file_name=f"{st.session_state.dock_base}_out.pdbqt", key="dl_pdbqt", width='stretch')
                if df is not None:
                    st.download_button("⬇ Scores (.csv)", df.to_csv(index=False).encode(),
                        file_name=f"{st.session_state.dock_base}_scores.csv", mime="text/csv",
                        key="dl_csv", width='stretch')
                if st.session_state.receptor_fh and os.path.exists(st.session_state.receptor_fh):
                    st.download_button("⬇ Receptor (.pdb)", open(st.session_state.receptor_fh, "rb"),
                        file_name="receptor.pdb", key="dl_rec", width='stretch')

            st.markdown("---")

            # ── Binding Pocket View ───────────────────────────────────────
            st.markdown("**🔬 Binding Pocket View**")
            _bpl, _bpr = st.columns([2, 1])
            with _bpl:
                _cutoff = st.slider("Distance cutoff (A)", 2.5, 5.0, 3.5, 0.1, key="bp_cutoff")
            with _bpr:
                _show_labels  = st.checkbox("Show residue labels",  value=True,  key="bp_show_labels")
                _show_surface = st.checkbox("Show protein surface",  value=False, key="bp_show_surface")

            try:
                vbp = py3Dmol.view(width="100%", height=440)
                vbp.setBackgroundColor(_viewer_bg())
                mbp = 0
                if st.session_state.receptor_fh and os.path.exists(st.session_state.receptor_fh):
                    vbp.addModel(open(st.session_state.receptor_fh).read(), "pdb")
                    vbp.setStyle({"model": mbp}, {"cartoon": {"color": "spectrum", "opacity": 0.45}})
                    if _show_surface:
                        vbp.addSurface(py3Dmol.SAS, {"opacity": 0.55, "color": "white"}, {"model": mbp})
                    mbp += 1
                # ── Heme ──────────────────────────────────────────────────
                mbp = _add_heme_to_view(vbp, st.session_state.get("receptor_fh"), mbp)
                # ─────────────────────────────────────────────────────────
                vbp.addModel(Chem.MolToMolBlock(sel_mol), "mol")
                _lig_m = mbp
                vbp.setStyle({"model": _lig_m}, {"stick": {"colorscheme": "cyanCarbon", "radius": 0.30}})

                if st.session_state.receptor_fh and os.path.exists(st.session_state.receptor_fh):
                    _ir = get_interacting_residues(st.session_state.receptor_fh, sel_mol, cutoff=_cutoff)
                    for _rb in _ir:
                        # ── Empty chain ID fix ────────────────────────────
                        _has_chain = bool(_rb["chain"] and _rb["chain"].strip())
                        _sel = {"model": 0, "resi": _rb["resi"]}
                        if _has_chain:
                            _sel["chain"] = _rb["chain"]
                        # ─────────────────────────────────────────────────
                        vbp.setStyle(_sel, {"stick": {"colorscheme": "orangeCarbon", "radius": 0.20}})
                        if _show_labels:
                            _lbl_chain = _rb["chain"] if _has_chain else ""
                            vbp.addLabel(
                                f"{_rb['resn']}{_rb['resi']}{_lbl_chain}",
                                {"fontSize": 11, "fontColor": "yellow",
                                 "backgroundColor": "black", "backgroundOpacity": 0.65,
                                 "inFront": True, "showBackground": True},
                                _sel,
                            )
                    _n = len(_ir)
                    _res_kind = "success" if _n else "warn"
                    st.markdown(
                        f"{_pill(f'Pose {pose_idx+1}')}"
                        f" {_pill(f'{_cutoff:.1f} A cutoff')}"
                        f" {_pill(f'{_n} residue' + ('s' if _n != 1 else ''), _res_kind)}"
                        + (f" {_pill('surface on', 'info')}" if _show_surface else ""),
                        unsafe_allow_html=True,
                    )
                vbp.zoomTo({"model": _lig_m})
                show3d(vbp, height=440)
            except Exception as _e:
                st.info(f"Binding pocket viewer error: {_e}")

            # ── PoseView 2D ───────────────────────────────────────────────
            pv_sdf_all = st.session_state.get("output_pv_sdf", "")
            sp_pv = str(WORKDIR / f"pose_{pose_idx+1}_pv_ready.sdf")
            if pv_sdf_all and os.path.exists(pv_sdf_all):
                pv_mols = load_mols_from_sdf(pv_sdf_all)
                write_single_pose(pv_mols[pose_idx] if pose_idx < len(pv_mols) else sel_mol, sp_pv)
            else:
                write_single_pose(sel_mol, sp_pv)

            # RMSD for selected pose vs co-crystal — pass to AI prompt
            _cryst_pdb_pv = st.session_state.get("ligand_pdb_path") or ""
            _rmsd_pv = None
            if _cryst_pdb_pv and os.path.exists(_cryst_pdb_pv):
                try:
                    _rmsd_pv = calc_rmsd_heavy(sel_mol, _cryst_pdb_pv)
                except Exception:
                    pass

            # ── Ready-to-use Figure ───────────────────────────────────────
            _ready_figure_section(
                mode            = "single",
                rec_fh          = st.session_state.get("receptor_fh", ""),
                sel_mol         = sel_mol,
                pose_idx        = pose_idx,
                lig_name        = st.session_state.get("ligand_name", ""),
                binding_energy  = (
                    float(df[df["Pose"] == pose_idx+1]["Affinity (kcal/mol)"].iloc[0])
                    if df is not None and len(df[df["Pose"] == pose_idx+1]) > 0 else None
                ),
                cryst_pdb       = _cryst_pdb_pv,
                acd_svg         = st.session_state.get("pv_image_svg_new"),
                acd_ihtml       = st.session_state.get("pv_image_svg_ihtml"),
                rdk_svg         = st.session_state.get("pv_image_svg_rdk"),
                pv_svg          = st.session_state.get("pv_image_svg"),
                pv_png          = st.session_state.get("pv_image_png"),
            )

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
                    if df is not None and len(df[df["Pose"] == pose_idx+1]) > 0 else None
                ),
                ref_lig_name   = (st.session_state.get("redock_result", {}).get("ref_name", "") if st.session_state.get("redock_result") else ""),
                ref_lig_smiles = ((st.session_state.get("redock_result", {}).get("prot_smiles") or st.session_state.get("redock_result", {}).get("SMILES", "")) if st.session_state.get("redock_result") else ""),
                ref_lig_energy = (st.session_state.get("redock_result", {}).get("Top Score") if st.session_state.get("redock_result") else None),
                rmsd_crystal   = _rmsd_pv,
            )

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
            "Input mode", ["SMILES list (text)", "Upload .smi file", "Upload structure files (.pdb/.mol2)"],
            key="b_input_mode",
        )
        if b_input_mode == "SMILES list (text)":
            st.text_area("One `SMILES [name]` per line",
                value=("C1=CC(=CC=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O Apigenin\n"
                       "C1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C(=C3O)O)O Baicalein\n"
                       "CC1=CC=C(C=C1)NC2=NC=NC3=C2C=C(C=C3)O Osimertinib\n"
                       "C1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C(=C3O)O)O Luteolin\n"
                       "CC(C)OC1=C(C=C2C(=C1)N=CN2)NC3=CC=CC(=C3)C#C Gefitinib"),
                height=200, key="b_smiles_text")
        elif b_input_mode == "Upload .smi file":
            st.file_uploader("Upload .smi file", type=["smi", "txt"], key="b_smi_file")
        else:
            st.file_uploader("Upload structure files", type=["sdf", "mol2", "pdb"],
                accept_multiple_files=True, key="b_struct_files")
            b_struct_prot = st.radio(
                "Ligand preparation mode",
                ["Use the uploaded form", "Protonation at target pH"],
                horizontal=True, key="b_struct_prot",
            )
        b_ph = st.number_input("Target pH", 0.0, 14.0, 7.4, 0.1, key="b_ph")

        b_prot_mode = st.radio(
            "Protonation mode",
            ["⚡ Fast (Dimorphite-DL)", "🧪 pKaNET Cloud (recommended)", "🔬 Neutral (add H only)"],
            horizontal=True, key="b_prot_mode",
        )
        _b_prot_mode_map = {
            "⚡ Fast (Dimorphite-DL)":       "dimorphite",
            "🧪 pKaNET Cloud (recommended)": "pkanet",
            "🔬 Neutral (add H only)":       "neutral",
        }
        _b_use_pubchem = True
        if b_prot_mode == "🧪 pKaNET Cloud (recommended)":
            _b_use_pubchem = st.checkbox(
                "Query PubChem for experimental pKa (cached per compound)",
                value=True, key="b_use_pubchem",
            )

    with col_b2:
        st.markdown("**Redocking validation**")
        b_do_redock = st.checkbox("Dock co-crystal ligand as reference", value=True, key="b_do_redock")
        if b_do_redock:
            st.text_input(
                "Co-crystal SMILES [name]",
                value="COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC Erlotinib",
                key="b_redock_smiles",
            )
        st.markdown("**Docking parameters**")
        b_exh = st.slider("Exhaustiveness", 4, 32, 8, 2, key="b_exh")
        b_nm  = st.slider("Poses per ligand", 5, 20, 10, 1, key="b_nm")
        b_er  = st.slider("Energy range (kcal/mol)", 1, 5, 3, 1, key="b_er")

    if not b_rec_done:
        st.caption("⚠ Complete Step B1 first.")
    if st.button("▶ Run Batch Docking", key="b_btn_dock", type="primary", disabled=not b_rec_done):
        rec_pdbqt = st.session_state.get("b_receptor_pdbqt")
        config    = st.session_state.get("b_config_txt")
        b_ph_val      = st.session_state.get("b_ph", 7.4)
        _b_prot_mode  = _b_prot_mode_map.get(
            st.session_state.get("b_prot_mode", "⚡ Fast (Dimorphite-DL)"), "dimorphite"
        )
        _b_use_pubchem = st.session_state.get("b_use_pubchem", True)

        smiles_pairs = []
        struct_file_pairs = []
        _b_use_struct_files = False
        try:
            if st.session_state.get("b_input_mode") == "SMILES list (text)":
                for line in st.session_state.get("b_smiles_text", "").strip().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    pts = line.split(None, 1)
                    _nm = pts[1].replace(" ", "_") if len(pts) > 1 else f"lig_{len(smiles_pairs)+1}"
                    smiles_pairs.append((pts[0], _nm))
            elif st.session_state.get("b_input_mode") == "Upload .smi file":
                fobj = st.session_state.get("b_smi_file")
                if fobj is None: raise ValueError("No .smi file uploaded")
                for line in fobj.read().decode().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    pts = line.split(None, 1)
                    _nm = pts[1].replace(" ", "_") if len(pts) > 1 else f"lig_{len(smiles_pairs)+1}"
                    smiles_pairs.append((pts[0], _nm))
            else:
                _b_use_struct_files = True
                _uploaded = st.session_state.get("b_struct_files", [])
                if not _uploaded: raise ValueError("No structure files uploaded")
                for _uf in _uploaded:
                    _ext = Path(_uf.name).suffix.lower()
                    _nm  = Path(_uf.name).stem.replace(" ", "_")
                    _tmp = str(BATCH_WORKDIR / f"{_nm}{_ext}")
                    with open(_tmp, "wb") as _wf: _wf.write(_uf.read())
                    struct_file_pairs.append((_tmp, _nm))
            if not smiles_pairs and not struct_file_pairs:
                raise ValueError("No valid input found")
        except Exception as e:
            st.error(f"❌ Input parsing failed: {e}"); st.stop()

        redock_score  = None
        redock_result = None
        if st.session_state.get("b_do_redock"):
            raw_rd = st.session_state.get("b_redock_smiles", "").strip()
            pts    = raw_rd.split(None, 1)
            rd_smi = pts[0]
            rd_nm  = pts[1].replace(" ", "_") if len(pts) > 1 else "redock"
            with st.spinner(f"Docking reference ligand ({rd_nm})…"):
                rd_prep = prepare_ligand(rd_smi, "redock_" + rd_nm, b_ph_val, BATCH_WORKDIR,
                                         mode=_b_prot_mode, use_pubchem=_b_use_pubchem)
                if rd_prep["success"]:
                    rd_dock = run_vina(rec_pdbqt, rd_prep["pdbqt"], config,
                        VINA_PATH, b_exh, b_nm, b_er, BATCH_WORKDIR, "redock_" + rd_nm)
                    if rd_dock["success"] and rd_dock["top_score"] is not None:
                        redock_score = rd_dock["top_score"]
                        rd_pv_sdf = str(BATCH_WORKDIR / f"redock_{rd_nm}_pv_ready.sdf")
                        fix_sdf_bond_orders(rd_dock["out_sdf"], rd_smi, rd_pv_sdf)
                        if not os.path.exists(rd_pv_sdf) or os.path.getsize(rd_pv_sdf) < 10:
                            rd_pv_sdf = rd_dock["out_sdf"]
                        rd_n = len(load_mols_from_sdf(rd_dock["out_sdf"], sanitize=False)) if os.path.exists(rd_dock["out_sdf"]) else 0
                        redock_result = {
                            "Name": f"⭐ {rd_nm} (co-crystal ref)", "ref_name": rd_nm,
                            "SMILES": rd_smi, "prot_smiles": rd_prep["prot_smiles"],
                            "Charge": rd_prep["charge"], "Top Score": redock_score,
                            "pose_scores": [s["affinity"] for s in rd_dock["scores"]],
                            "Poses": rd_n, "out_pdbqt": rd_dock["out_pdbqt"],
                            "out_sdf": rd_dock["out_sdf"], "pv_sdf": rd_pv_sdf,
                            "Status": "OK", "is_redock": True,
                        }
                        st.success(f"✓ Reference score: **{redock_score:.2f} kcal/mol** ({rd_nm})")
                    else:
                        st.warning("⚠ Redocking failed")
                else:
                    st.warning(f"⚠ Reference ligand prep failed: {rd_prep.get('error')}")

        results = []
        _items  = struct_file_pairs if _b_use_struct_files else smiles_pairs
        n       = len(_items)
        prog    = st.progress(0, text=f"Docking 0/{n}…")
        log_slot = st.empty()
        all_logs = []

        for i, item in enumerate(_items):
            if _b_use_struct_files:
                fpath, name = item; smi = ""
            else:
                smi, name = item; fpath = None

            prog.progress(i / n, text=f"Docking {name} ({i+1}/{n})…")

            if _b_use_struct_files:
                _struct_prot = st.session_state.get("b_struct_prot", "Use the uploaded form")
                if _struct_prot == "Use the uploaded form":
                    prep = prepare_ligand_from_file(fpath, name, BATCH_WORKDIR)
                else:
                    try: smi = smiles_from_file(fpath, BATCH_WORKDIR)
                    except Exception as e:
                        results.append({"Name": name, "SMILES": "", "Charge": None, "Top Score": None, "Status": f"PREP FAILED: {e}"})
                        all_logs.append(f"[{name}] PREP ERROR: {e}"); continue
                    prep = prepare_ligand(smi, name, b_ph_val, BATCH_WORKDIR,
                                          mode=_b_prot_mode, use_pubchem=_b_use_pubchem)
            else:
                prep = prepare_ligand(smi, name, b_ph_val, BATCH_WORKDIR,
                                      mode=_b_prot_mode, use_pubchem=_b_use_pubchem)

            if not prep["success"]:
                results.append({"Name": name, "SMILES": smi, "Charge": None, "Top Score": None, "Status": f"PREP FAILED: {prep['error']}"})
                all_logs.append(f"[{name}] PREP ERROR: {prep['error']}"); continue

            smi = smi or prep.get("prot_smiles", "")
            dock = run_vina(rec_pdbqt, prep["pdbqt"], config, VINA_PATH, b_exh, b_nm, b_er, BATCH_WORKDIR, name)
            all_logs.append(f"[{name}] score={dock.get('top_score')} | {dock.get('log','')[:100]}")
            log_slot.markdown(f'<div class="log-box">{"".join(all_logs[-5:])}</div>', unsafe_allow_html=True)

            if not dock["success"] or dock["top_score"] is None:
                results.append({"Name": name, "SMILES": smi, "Charge": prep["charge"], "Top Score": None, "Status": "DOCK FAILED"})
                continue

            pv_sdf = str(BATCH_WORKDIR / f"{name}_pv_ready.sdf")
            fix_sdf_bond_orders(dock["out_sdf"], smi, pv_sdf)
            if not os.path.exists(pv_sdf) or os.path.getsize(pv_sdf) < 10:
                pv_sdf = dock["out_sdf"]
            n_poses = len(load_mols_from_sdf(dock["out_sdf"], sanitize=False)) if os.path.exists(dock["out_sdf"]) else 0
            results.append({
                "Name": name, "SMILES": smi, "prot_smiles": prep["prot_smiles"],
                "Charge": prep["charge"], "Top Score": dock["top_score"],
                "pose_scores": [s["affinity"] for s in dock["scores"]],
                "Poses": n_poses, "out_pdbqt": dock["out_pdbqt"],
                "out_sdf": dock["out_sdf"], "pv_sdf": pv_sdf, "Status": "OK",
            })

        n_ok = sum(1 for r in results if r["Status"] == "OK")
        prog.progress(1.0, text=f"Done — {n_ok}/{n} ligands docked successfully")
        log_slot.empty()

        st.session_state.update({
            "b_batch_done": True, "b_batch_results": results,
            "b_batch_log": "\n".join(all_logs),
            "b_redock_score": redock_score, "b_redock_result": redock_result,
            "b_confirmed_ref_score": None, "b_confirmed_ref_pose": None, "b_confirmed_ref_name": None,
            "b_pv2_image_png": None, "b_pv2_image_svg": None, "b_pv2_pose_key": None,
            "b_pv2_ref_png": None, "b_pv2_ref_svg": None, "b_plot_png": None,
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
            f"{_pill(f'{n_ok} ligands docked', 'success')} {_pill('AutoDock Vina 1.2.7')}"
            + (f" {_pill(f'{n_fail} failed', 'warn')}" if n_fail else ""),
            unsafe_allow_html=True,
        )

        ok_results = [r for r in results if r["Status"] == "OK" and r.get("out_sdf") and os.path.exists(r["out_sdf"])]
        browsable = (
            [redock_result] if redock_result and os.path.exists(redock_result.get("out_sdf", "")) else []
        ) + ok_results

        if browsable:
            st.markdown("**🔎 Pose Browser**")
            sel_nm = st.selectbox("Select ligand", [r["Name"] for r in browsable], index=0, key="b_lig_sel")
            sel_res       = next(r for r in browsable if r["Name"] == sel_nm)
            is_redock_sel = sel_res.get("is_redock", False)
            pose_scores_l = sel_res.get("pose_scores", [])

            b_mols = load_mols_from_sdf(sel_res["out_sdf"], sanitize=False)
            if b_mols:
                b_pose_i = st.slider("Pose", 1, len(b_mols), 1, key="b_pose_sel") - 1
                this_score = pose_scores_l[b_pose_i] if b_pose_i < len(pose_scores_l) else sel_res["Top Score"]
                _score_kind = "success" if (this_score is not None and this_score < -8) else "warn"
                row_pills = f"{_pill(f'Pose {b_pose_i+1} / {len(b_mols)}')} {_pill(f'Score: {this_score:.2f} kcal/mol', _score_kind) if this_score is not None else ''}"

                if is_redock_sel:
                    _cryst = st.session_state.get("b_ligand_pdb_path") or ""
                    if _cryst and os.path.exists(_cryst):
                        _rmsd = calc_rmsd_heavy(b_mols[b_pose_i], _cryst)
                        if _rmsd is not None:
                            _rk = "success" if _rmsd <= 2.0 else "warn" if _rmsd <= 3.0 else "info"
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
                            vb.setStyle({"model": bmi}, {"cartoon": {"color": "spectrum", "opacity": 0.7}, "stick": {"radius": 0.08, "opacity": 0.15}})
                            bmi += 1
                        _lig_p = st.session_state.get("b_ligand_pdb_path")
                        if _lig_p and os.path.exists(_lig_p):
                            vb.addModel(open(_lig_p).read(), "pdb")
                            vb.setStyle({"model": bmi}, {"stick": {"colorscheme": "magentaCarbon", "radius": 0.2}})
                            bmi += 1
                        # Heme
                        bmi = _add_heme_to_view(vb, st.session_state.get("b_receptor_fh"), bmi)
                        vb.addModel(Chem.MolToMolBlock(b_mols[b_pose_i]), "mol")
                        vb.setStyle({"model": bmi}, {"stick": {"colorscheme": "cyanCarbon", "radius": 0.28}})
                        vb.addSurface("SES", {"opacity": 0.2, "color": "lightblue"}, {"model": 0}, {"model": bmi})
                        vb.zoomTo({"model": bmi}); vb.center({"model": bmi})
                        show3d(vb, height=420)
                    except Exception as e:
                        st.info(f"Viewer error: {e}")

                with cbd:
                    st.markdown("**Actions**")
                    if is_redock_sel and this_score is not None:
                        already = (c_ref_score == this_score and c_ref_pose == b_pose_i + 1)
                        if st.button(
                            f"✅ Confirmed (pose {b_pose_i+1})" if already else f"📌 Use pose {b_pose_i+1} as reference",
                            key="b_confirm_ref_btn", type="secondary" if already else "primary", width='stretch',
                        ):
                            st.session_state.update({
                                "b_confirmed_ref_score": this_score,
                                "b_confirmed_ref_pose": b_pose_i + 1,
                                "b_confirmed_ref_name": sel_nm,
                            })
                            st.rerun()
                        if c_ref_score is not None and not already:
                            if st.button("🔄 Reset reference", key="b_reset_ref_btn", width='stretch'):
                                st.session_state.update({
                                    "b_confirmed_ref_score": None, "b_confirmed_ref_pose": None, "b_confirmed_ref_name": None,
                                })
                                st.rerun()
                    st.markdown("**Download**")
                    safe_nm = sel_nm.replace("⭐ ", "").replace(" (co-crystal ref)", "")
                    sp3 = str(BATCH_WORKDIR / f"{safe_nm}_pose{b_pose_i+1}.sdf")
                    write_single_pose(b_mols[b_pose_i], sp3)
                    st.download_button(f"⬇ Pose {b_pose_i+1} (.sdf)", open(sp3, "rb"),
                        file_name=f"{safe_nm}_pose{b_pose_i+1}.sdf", key="b_dl_pose", width='stretch')
                    if sel_res.get("out_pdbqt") and os.path.exists(sel_res["out_pdbqt"]):
                        st.download_button("⬇ All poses (.pdbqt)", open(sel_res["out_pdbqt"], "rb"),
                            file_name=f"{safe_nm}_out.pdbqt", key="b_dl_pdbqt", width='stretch')

        st.markdown("---")
        with st.expander("📋 Full docking log", expanded=False):
            st.markdown(f'<div class="log-box">{st.session_state.get("b_batch_log", "")}</div>', unsafe_allow_html=True)

        df_res = pd.DataFrame([{
            "Name": r["Name"], "Top Score (kcal/mol)": r["Top Score"],
            "Charge": f"{r['Charge']:+d}" if r.get("Charge") is not None else "—",
            "Status": r["Status"],
        } for r in results])
        ok_df   = df_res[df_res["Status"] == "OK"].sort_values("Top Score (kcal/mol)").reset_index(drop=True)
        plot_df = df_res[df_res["Status"] == "OK"].reset_index(drop=True)

        if not plot_df.empty:
            _n    = len(plot_df)
            _best = ok_df["Top Score (kcal/mol)"].min()

            def _draw_plot(ax):
                _cc = _chart_colors()
                ax.get_figure().patch.set_facecolor(_cc["bg"]); ax.set_facecolor(_cc["bg_sub"])
                scores = plot_df["Top Score (kcal/mol)"].values
                xs = list(range(_n))
                colors = ["#3fb950" if s == _best else "#58a6ff" for s in scores]
                ax.scatter(xs, scores, color=colors, s=90, zorder=3, edgecolors=_cc["border"], linewidths=0.5)
                ax.plot(xs, scores, color=_cc["border"], linewidth=0.8, zorder=2)
                ax.set_xticks(xs); ax.set_xticklabels(plot_df["Name"].values, rotation=40, ha="right")
                ax.set_xlim(-0.5, _n - 0.5)
                if active_ref is not None:
                    _ref_lbl = (f"Confirmed ref (pose {c_ref_pose}): {active_ref:.2f} kcal/mol"
                                if c_ref_score is not None else f"Co-crystal ref: {active_ref:.2f} kcal/mol")
                    ax.axhline(active_ref, color="#f85149", linewidth=1.8, linestyle="--", label=_ref_lbl)
                    ax.legend(facecolor=_cc["legend_bg"], edgecolor=_cc["border"], labelcolor=_cc["text"], fontsize=8)
                ax.set_ylabel("Vina score (kcal/mol)", color=_cc["muted"], fontsize=9)
                ax.set_xlabel("Ligand", color=_cc["muted"], fontsize=9)
                ax.tick_params(colors=_cc["muted"], labelsize=7)
                for sp in ax.spines.values(): sp.set_edgecolor(_cc["border"])

            if _n <= 10:
                ct2, cp2 = st.columns([1, 1.6])
                with ct2:
                    st.markdown("**Score Table**")
                    st.dataframe(df_res, hide_index=True, width='stretch')
                with cp2:
                    st.markdown("**Top Score per Ligand**")
                    fig, ax = plt.subplots(figsize=(max(5, _n * 0.6 + 1.5), 3.5))
                    _draw_plot(ax); fig.tight_layout()
                    _buf = io.BytesIO(); fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
                    _buf.seek(0); st.session_state["b_plot_png"] = _buf.getvalue()
                    st.pyplot(fig, width='stretch'); plt.close(fig)
            else:
                st.markdown("**Top Score per Ligand**")
                fig, ax = plt.subplots(figsize=(max(6, _n * 0.9 + 1.5), 4))
                _draw_plot(ax); fig.tight_layout()
                _buf = io.BytesIO(); fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
                _buf.seek(0); st.session_state["b_plot_png"] = _buf.getvalue()
                st.pyplot(fig, width='stretch'); plt.close(fig)
                st.markdown("**Score Table**"); st.dataframe(df_res, hide_index=True, width='stretch')
        else:
            st.markdown("**Score Table**"); st.dataframe(df_res, hide_index=True, width='stretch')

        st.markdown("---")
        st.markdown("**⬇ Download All Results**")
        c_csv, c_zip = st.columns(2)
        with c_csv:
            if not ok_df.empty:
                st.download_button("⬇ Top scores (.csv)", ok_df.to_csv(index=False).encode(),
                    file_name="batch_scores.csv", mime="text/csv", key="b_dl_csv", width='stretch')
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
            zb.seek(0)
            st.download_button("⬇ Download ALL (.zip)", zb, file_name="anyone_can_dock.zip",
                mime="application/zip", key="b_dl_zip", width='stretch')

        # ── 2D Interaction diagram ────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🧬 2D Interaction Diagram")
        pv_browsable = [r for r in browsable if r.get("out_sdf") and os.path.exists(r["out_sdf"])]
        if pv_browsable:
            pv_sel_nm   = st.selectbox("Associate docked ligand (for AI prompt)",
                [r["Name"] for r in pv_browsable], index=0, key="b_pv_lig_sel")
            pv_sel_res  = next(r for r in pv_browsable if r["Name"] == pv_sel_nm)
            pv_safe_nm  = pv_sel_nm.replace("⭐ ", "").replace(" (co-crystal ref)", "")
            pv_all_mols = load_mols_from_sdf(pv_sel_res["out_sdf"], sanitize=False)

            if pv_all_mols:
                pv_pose_i = st.slider("Pose (AI prompt context)", 1, len(pv_all_mols), 1, key="b_pv_pose_sel") - 1
                pv_scores = pv_sel_res.get("pose_scores", [])
                pv_score  = pv_scores[pv_pose_i] if pv_pose_i < len(pv_scores) else pv_sel_res.get("Top Score")

                st.session_state["_b_pv2_smiles"] = pv_sel_res.get("prot_smiles") or pv_sel_res.get("SMILES", pv_sel_nm)

                sp_pv2      = str(BATCH_WORKDIR / f"{pv_safe_nm}_pose{pv_pose_i+1}_pv2_ready.sdf")
                pv_all_path = pv_sel_res.get("pv_sdf", "")
                if pv_all_path and os.path.exists(pv_all_path):
                    pv_fixed = load_mols_from_sdf(pv_all_path)
                    write_single_pose(pv_fixed[pv_pose_i] if pv_pose_i < len(pv_fixed) else pv_all_mols[pv_pose_i], sp_pv2)
                else:
                    write_single_pose(pv_all_mols[pv_pose_i], sp_pv2)

                # RMSD for selected batch pose vs co-crystal
                _b_cryst_pv = st.session_state.get("b_ligand_pdb_path") or ""
                _rmsd_pv2   = None
                if _b_cryst_pv and os.path.exists(_b_cryst_pv) and pv_all_mols:
                    try:
                        _rmsd_pv2 = calc_rmsd_heavy(
                            pv_all_mols[pv_pose_i] if pv_pose_i < len(pv_all_mols) else pv_all_mols[0],
                            _b_cryst_pv,
                        )
                    except Exception:
                        pass

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
                    ref_lig_energy      = redock_result.get("Top Score") if redock_result else None,
                    show_header         = False,
                    rmsd_crystal        = _rmsd_pv2,
                )

                # ── Ready-to-use Figure ───────────────────────────────────
                def _b_draw_plot_fn(ax):
                    if not plot_df.empty:
                        _draw_plot(ax)

                _ready_figure_section(
                    mode            = "batch",
                    rec_fh          = st.session_state.get("b_receptor_fh", ""),
                    sel_mol         = pv_all_mols[pv_pose_i] if pv_pose_i < len(pv_all_mols) else None,
                    pose_idx        = pv_pose_i,
                    lig_name        = pv_safe_nm,
                    binding_energy  = pv_score,
                    cryst_pdb       = _b_cryst_pv,
                    acd_svg         = st.session_state.get("b_pv2_image_svg_new"),
                    acd_ihtml       = st.session_state.get("b_pv2_image_svg_ihtml"),
                    rdk_svg         = st.session_state.get("b_pv2_image_svg_rdk"),
                    pv_svg          = st.session_state.get("b_pv2_image_svg"),
                    pv_png          = st.session_state.get("b_pv2_image_png"),
                    b_browsable     = browsable,
                    b_sel_res       = pv_sel_res,
                    b_mols          = pv_all_mols,
                    b_pose_i        = pv_pose_i,
                    b_plot_draw_fn  = _b_draw_plot_fn,
                    b_plot_n        = len(plot_df),
                    b_rec_fh        = st.session_state.get("b_receptor_fh", ""),
                    b_cryst_pdb     = _b_cryst_pv,
                    b_acd_svg       = st.session_state.get("b_pv2_image_svg_new"),
                    b_acd_ihtml     = st.session_state.get("b_pv2_image_svg_ihtml"),
                    b_rdk_svg       = st.session_state.get("b_pv2_image_svg_rdk"),
                    b_pv_svg        = st.session_state.get("b_pv2_image_svg"),
                    b_pv_png        = st.session_state.get("b_pv2_image_png"),
                    b_this_score    = pv_score,
                    b_sel_nm        = pv_sel_nm,
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
