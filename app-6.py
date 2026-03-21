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
    from core import draw_interactions_rdkit
except ImportError:
    draw_interactions_rdkit = None

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
    _pose_key = (
        f"{st.session_state.get(smiles_key, 'lig')}_pose{pose_idx+1}{label_suffix}"
    )
    _stale   = st.session_state.get(pose_key_key) != _pose_key
    _has_ref = bool(pdb_id and cocrystal_ligand_id)
    _lbl     = st.session_state.get(smiles_key, "ligand")[:20]

    if show_header:
        st.markdown("---")
        st.markdown("**🧬 2D Interaction Diagrams**")

    # ── Engine selector ───────────────────────────────────────────────────────
    _engine = st.radio(
        "2D diagram engine",
        ["🔬 PoseView (proteins.plus)", "🐍 RDKit (local, always works)"],
        horizontal=True,
        key=btn_key + "_engine",
        help=(
            "PoseView: high-quality but requires proteins.plus server. "
            "RDKit: local, no server needed, works on Streamlit Cloud."
        ),
    )
    _use_rdkit = "RDKit" in _engine

    # ── RDKit branch — instant, no server ────────────────────────────────────
    if _use_rdkit:
        if draw_interactions_rdkit is None:
            st.warning("⚠️ RDKit diagram requires the latest **core.py** — please update it on GitHub.")
            return

        _rec    = st.session_state.get(rec_key, "")
        _smiles = lig_smiles or st.session_state.get(smiles_key, "")

        if not _rec or not os.path.exists(_rec):
            st.warning("Complete receptor preparation first.")
            return
        if not _smiles:
            st.warning("No ligand SMILES available.")
            return

        # Co-crystal ligand PDB — saved during receptor prep
        # Use same session state key pattern as rec_key e.g. "" → "ligand_pdb_path"
        #                                                     "b_" → "b_ligand_pdb_path"
        _pfx = rec_key.replace("receptor_fh", "")   # extract prefix: "" or "b_"
        _lig_pdb_path = st.session_state.get(_pfx + "ligand_pdb_path", "")
        _has_ref_rdkit = bool(
            _lig_pdb_path and os.path.exists(_lig_pdb_path)
        )

        # Controls
        _ctrl_l, _ctrl_r = st.columns(2)
        with _ctrl_l:
            _cutoff_rdkit = st.slider(
                "Interaction cutoff (Å)", 2.5, 5.0, 3.5, 0.1,
                key=btn_key + "_rdkit_cutoff",
            )
        with _ctrl_r:
            _max_res = st.slider(
                "Max residues shown", 4, 20, 10, 1,
                key=btn_key + "_rdkit_maxres",
                help="Reduce to clean up busy diagrams.",
            )

        if st.button("🐍 Generate RDKit Diagrams", key=btn_key + "_rdkit", type="primary"):
            # Left: docked pose
            with st.spinner("⏳ Generating docked pose diagram…"):
                try:
                    _mols = load_mols_from_sdf(pose_sdf_path)
                    _mol  = _mols[0] if _mols else None
                    if _mol is None:
                        st.error("Could not read pose SDF.")
                    else:
                        _title = (
                            f"{lig_name} · Pose {pose_idx+1}"
                            + (f" · {binding_energy:.2f} kcal/mol"
                               if binding_energy is not None else "")
                        )
                        _rdkit_svg = draw_interactions_rdkit(
                            lig_mol      = _mol,
                            receptor_pdb = _rec,
                            smiles       = _smiles,
                            title        = _title,
                            cutoff       = _cutoff_rdkit,
                            size         = (500, 500),
                            max_residues = _max_res,
                        )
                        st.session_state[img_svg_key + "_rdkit"]  = _rdkit_svg
                        st.session_state[pose_key_key + "_rdkit"] = _pose_key
                except Exception as e:
                    st.error(f"❌ RDKit docked pose error: {e}")

            # Right: co-crystal reference
            if _has_ref_rdkit:
                with st.spinner("⏳ Generating co-crystal reference diagram…"):
                    try:
                        from rdkit import Chem
                        _ref_mol = Chem.MolFromPDBFile(
                            _lig_pdb_path, sanitize=True, removeHs=True
                        )
                        if _ref_mol is None:
                            _ref_mol = Chem.MolFromPDBFile(
                                _lig_pdb_path, sanitize=False, removeHs=True
                            )
                        if _ref_mol is not None:
                            # Get SMILES for clean 2D — use ref_lig_smiles if provided
                            _ref_smiles = (
                                ref_lig_smiles
                                or Chem.MolToSmiles(Chem.RemoveHs(_ref_mol))
                            )
                            _ref_title = (
                                f"{ref_lig_name or cocrystal_ligand_id} · Co-crystal"
                                + (f" · {ref_lig_energy:.2f} kcal/mol"
                                   if ref_lig_energy is not None else "")
                            )
                            _ref_rdkit_svg = draw_interactions_rdkit(
                                lig_mol      = _ref_mol,
                                receptor_pdb = _rec,
                                smiles       = _ref_smiles,
                                title        = _ref_title,
                                cutoff       = _cutoff_rdkit,
                                size         = (500, 500),
                                max_residues = _max_res,
                            )
                            st.session_state[ref_svg_key + "_rdkit"] = _ref_rdkit_svg
                        else:
                            st.warning("⚠️ Could not read co-crystal ligand PDB.")
                    except Exception as e:
                        st.warning(f"⚠️ Co-crystal RDKit diagram error: {e}")
            st.rerun()

        # ── Display: left=docked, right=co-crystal ────────────────────────────
        _rdkit_svg     = st.session_state.get(img_svg_key + "_rdkit")
        _ref_rdkit_svg = st.session_state.get(ref_svg_key + "_rdkit") if ref_svg_key else None

        _LEGEND_RDKIT = """
<div style="background:#fff;border:1px solid #D0D7DE;border-radius:6px;
     padding:10px 18px;font-family:'Helvetica Neue',Arial,sans-serif;
     font-size:13px;color:#333;margin-top:6px;">
  <div style="display:flex;align-items:center;gap:28px;">
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
</div>"""

        def _show_rdkit_svg(svg_data, dl_key, dl_filename):
            svg_str = svg_data.decode() if isinstance(svg_data, bytes) else svg_data
            svg_str = svg_str.replace(
                "<svg ", '<svg style="width:100%;height:auto;display:block;" ', 1
            )
            components.html(
                f'<div style="background:#fff;border-radius:8px;padding:10px;'
                f'border:1px solid #D0D7DE;">{svg_str}</div>',
                height=530, scrolling=False,
            )
            st.markdown(_LEGEND_RDKIT, unsafe_allow_html=True)
            st.download_button(
                "⬇ SVG", data=svg_data,
                file_name=dl_filename,
                mime="image/svg+xml",
                key=dl_key,
                width='stretch',
            )

        if _rdkit_svg:
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("##### 🧪 Docked Pose (RDKit)")
                _show_rdkit_svg(
                    _rdkit_svg,
                    dl_key      = dl_svg_key + "_rdkit",
                    dl_filename = f"pose{pose_idx+1}_rdkit.svg",
                )
            with col_r:
                st.markdown("##### 🔬 Co-Crystal Reference (RDKit)")
                if _ref_rdkit_svg:
                    _show_rdkit_svg(
                        _ref_rdkit_svg,
                        dl_key      = dl_svg_key + "_rdkit_ref",
                        dl_filename = f"cocrystal_rdkit.svg",
                    )
                elif _has_ref_rdkit:
                    st.info("Click **Generate RDKit Diagrams** to generate co-crystal diagram.")
                else:
                    st.caption("⚠️ No co-crystal ligand — use Auto-detect in receptor preparation.")

            # ── AI prompt — same logic as PoseView ───────────────────────────
            st.markdown("---")
            _energy_str = (
                f"{binding_energy:.2f} kcal/mol"
                if binding_energy is not None else "[binding energy]"
            )
            _lig_str = (
                f"{lig_name} (SMILES: {_smiles})"
                if lig_name else _smiles or "[ligand]"
            )
            _has_ref_b = bool(_ref_rdkit_svg)
            _ref_clause = ""
            if _has_ref_b and (ref_lig_name or ref_lig_smiles):
                _rf  = (
                    f"{ref_lig_name} (SMILES: {ref_lig_smiles})"
                    if ref_lig_name and ref_lig_smiles
                    else ref_lig_name or ref_lig_smiles
                )
                _re_ = (
                    f", binding energy {ref_lig_energy:.2f} kcal/mol"
                    if ref_lig_energy is not None else ""
                )
                _ref_clause = f", and compare with co-crystallized reference {_rf}{_re_}"

            _prompt = (
                f"Analyze the RDKit interaction diagram for PDB ID "
                f"{pdb_id.upper() or '[PDB ID]'}, "
                f"docked ligand {_lig_str}, AutoDock Vina v1.2.7, "
                f"binding energy {_energy_str}{_ref_clause}.\n\n"
                "Legend: Blue circle = H-bond/polar · Green circle = hydrophobic"
                " · Pink circle = other interaction\n\n"
                "1. Identify key ligand-protein interactions.\n"
                "2. List main interacting residues and their roles.\n"
                + (
                    "3. Compare docked pose with the co-crystal reference.\n"
                    "4. Highlight similarities/differences in binding mode.\n"
                    "5. Evaluate whether interactions support the predicted binding energy.\n\n"
                    if _has_ref_b else
                    "3. Evaluate whether interactions support the predicted binding energy.\n\n"
                )
                + "Provide a concise structural interpretation of the binding mode."
            )
            st.markdown("### 🤖 AI Prompt")
            st.caption("Copy into any AI tool (GPT, Claude, Gemini, …) with the diagram above.")
            st.code(_prompt, language=None)

        return   # ← skip PoseView UI entirely when RDKit is selected

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
        _run = st.button("🔬 Generate 2D Diagrams", key=btn_key, type="primary")

    with st.expander("🔍 Test PoseView API", expanded=False):
        st.caption(
            "Sends a known-good test structure (PDB 4AGN) to PoseView "
            "to check if the server is working — independent of your files."
        )
        if st.button("▶ Run API Test", key=btn_key + "_diag"):
            with st.spinner("Testing proteins.plus PoseView API…"):
                try:
                    from core import diagnose_poseview as _diagnose_poseview
                    _diag = _diagnose_poseview()
                except ImportError:
                    st.error("❌ diagnose_poseview not found — please deploy the latest core.py")
                    _diag = None
            if _diag:
                for _line in _diag["log"]:
                    if _line.startswith("✓"):
                        st.success(_line)
                    else:
                        st.error(_line)
                if _diag["poseview_ok"]:
                    st.success(
                        "✅ API is working fine — if your docking diagram fails, "
                        "the issue is with your specific receptor/ligand files."
                    )
                    if _diag["image_url"]:
                        st.markdown(f"[View test SVG]({_diag['image_url']})")
                elif _diag["server_reachable"]:
                    st.warning(
                        f"⚠️ Server reachable but PoseView failed: {_diag['error']}"
                    )
                else:
                    st.error(
                        f"❌ Server unreachable: {_diag['error']}"
                    )

    if _run:
        _rec = st.session_state.get(rec_key, "")
        if not _rec or not os.path.exists(_rec):
            st.error("Receptor PDB not found — complete receptor preparation first.")
        elif not os.path.exists(pose_sdf_path):
            st.error("Pose SDF not found.")
        else:
            with st.spinner("⏳ PoseView v1 — generating 2D diagram… (30–60 s)"):
                _svg, _err = call_poseview_v1(_rec, pose_sdf_path)
            if _err:
                st.error(f"❌ PoseView v1 error:\n\n```\n{_err}\n```")
                st.caption(
                    "💡 If the error says 'Server rejected job', try clicking Generate again. "
                    "If it persists, the proteins.plus server may be temporarily unavailable."
                )
            else:
                _png = svg_to_png(_svg)
                st.session_state[img_png_key]  = _png
                st.session_state[img_svg_key]  = _svg
                st.session_state[pose_key_key] = _pose_key

            if _has_ref and ref_png_key and ref_svg_key:
                with st.spinner(
                    f"⏳ PoseView2 — {pdb_id.upper()} / {cocrystal_ligand_id}… (may retry up to 3×)"
                ):
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
                        file_name=f"pose{pose_idx+1}_docked.png",
                        mime="image/png", key=dl_png_key, width='stretch',
                    )
            with _d2:
                st.download_button(
                    "⬇ SVG", data=_pose_svg,
                    file_name=f"pose{pose_idx+1}_docked.svg",
                    mime="image/svg+xml", key=dl_svg_key, width='stretch',
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
                            mime="image/png",
                            key=dl_png_key + "_ref",
                            width='stretch',
                        )
                with _r2:
                    st.download_button(
                        "⬇ SVG", data=_ref_svg2,
                        file_name=f"cocrystal_{pdb_id}_{cocrystal_ligand_id}.svg",
                        mime="image/svg+xml",
                        key=dl_svg_key + "_ref",
                        width='stretch',
                    )
            elif _has_ref:
                st.info("Click **Generate 2D Diagrams** to load the co-crystal reference.")
            else:
                st.caption(
                    "⚠️ No co-crystal ligand ID — use Auto-detect in receptor preparation."
                )

        # AI analysis prompt
        st.markdown("---")
        _energy_str = (
            f"{binding_energy:.2f} kcal/mol"
            if binding_energy is not None else "[binding energy]"
        )
        _lig_str = (
            f"{lig_name} (SMILES: {lig_smiles})"
            if lig_name and lig_smiles else lig_name or "[ligand]"
        )
        _has_ref_b = bool(ref_lig_name or ref_lig_smiles)
        _both      = bool(_pose_svg and _ref_svg2)

        _ref_clause = ""
        if _has_ref_b:
            _rf  = (
                f"{ref_lig_name} (SMILES: {ref_lig_smiles})"
                if ref_lig_name and ref_lig_smiles
                else ref_lig_name or ref_lig_smiles
            )
            _re_ = (
                f", binding energy {ref_lig_energy:.2f} kcal/mol"
                if ref_lig_energy is not None else ""
            )
            _ref_clause = f", and compare with co-crystallized reference {_rf}{_re_}"

        _prompt = (
            f"Analyze the PoseView2 interaction diagram for PDB ID "
            f"{pdb_id.upper() or '[PDB ID]'}, "
            f"docked ligand {_lig_str}, AutoDock Vina v1.2.7, "
            f"binding energy {_energy_str}{_ref_clause}.\n\n"
            + (
                "Legend (docked pose): Black dashed = H-bond"
                " · Dark green solid = hydrophobic\n"
                "Legend (co-crystal):  Blue dashed = H-bond"
                " · Pink dashed = ionic · Yellow dashed = metal\n"
                "  Green dot-dash = cation-pi · Cyan dot-dash = pi-pi"
                " · Dark green solid = hydrophobic\n\n"
                if _both else
                "Legend: Black dashed = H-bond · Dark green solid = hydrophobic\n\n"
            )
            + "1. Identify key ligand-protein interactions.\n"
            + "2. List main interacting residues and their roles.\n"
            + (
                "3. Compare docked pose with the reference ligand.\n"
                "4. Highlight similarities/differences in binding orientation.\n"
                "5. Evaluate whether interactions support the predicted binding energy.\n\n"
                if _has_ref_b else
                "3. Evaluate whether interactions support the predicted binding energy.\n\n"
            )
            + "Provide a concise structural interpretation of the binding mode."
        )
        st.markdown("### 🤖 AI Prompt")
        st.caption("Copy into any AI tool (GPT, Claude, Gemini, …) with the diagram above.")
        st.code(_prompt, language=None)


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED: RECEPTOR PREPARATION SECTION
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
            "PDB source", ["Download from RCSB", "Upload PDB file"],
            horizontal=True, key=pfx + "src_mode",
        )
        if src == "Download from RCSB":
            pdb_id     = st.text_input("PDB ID", value="1M17", max_chars=4, key=pfx + "pdb_id")
            upload_pdb = None
        else:
            upload_pdb = st.file_uploader("Upload .pdb", type=["pdb"], key=pfx + "pdb_upload")
            pdb_id     = None

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
        raw_path = str(wdir / "raw.pdb")

        if src == "Download from RCSB":
            token = pdb_id.strip().upper()
            rc, _ = _run_cmd([
                "curl", "-sf",
                f"https://files.rcsb.org/download/{token}.pdb",
                "-o", raw_path,
            ])
            if rc != 0 or not os.path.exists(raw_path) or os.path.getsize(raw_path) < 200:
                st.error(f"❌ Download failed for {token}")
                st.stop()
            st.session_state[pfx + "pdb_token"] = token
        else:
            if upload_pdb is None:
                st.error("Please upload a PDB file first.")
                st.stop()
            with open(raw_path, "wb") as f:
                f.write(upload_pdb.read())
            st.session_state[pfx + "pdb_token"] = Path(upload_pdb.name).stem

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
            # Pre-upload receptor to MoleculeHandler/Protoss in background
            # so PoseView calls later are fast (no 30-60 s upload wait)
            clear_poseview_cache()
            with st.spinner("⏳ Pre-processing receptor for PoseView (Protoss)…"):
                _wok, _wmsg = warm_poseview_cache(result["rec_fh"])
            if _wok:
                st.toast(f"✓ PoseView receptor ready: {_wmsg}", icon="🧬")
            else:
                st.toast(f"⚠️ PoseView pre-processing skipped: {_wmsg}", icon="⚠️")
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
    "Molecular docking powered by **AutoDock Vina 1.2.7**, "
    "**Dimorphite-DL**, and **PoseView2**."
)
st.markdown(
    "**Basic** — single ligand. **Batch** — multiple ligands. "
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
        ["SMILES string", "Upload structure (.sdf/.mol2/.pdb)", "Draw structure (Ketcher)"],
        horizontal=True, key="lig_input_mode",
    )

    smiles_in = ""
    if lig_input_mode == "SMILES string":
        smiles_in = st.text_input(
            "SMILES string",
            value="COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
            key="smiles_in",
        )
    elif lig_input_mode == "Upload structure (.sdf/.mol2/.pdb)":
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

    if not st.session_state.ligand_done:
        st.caption("⚠ Complete Steps 1 & 2 first.")
    if st.button(
        "▶ Run Docking", key="btn_dock", type="primary",
        disabled=not st.session_state.ligand_done,
    ):
        base   = st.session_state.ligand_name
        pv_sdf = str(WORKDIR / f"{base}_pv_ready.sdf")

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
            })

    if st.session_state.docking_done:
        st.markdown(_pill("Docking complete", "success"), unsafe_allow_html=True)
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
                ax.set_xlabel("Pose",                color=_cc["muted"], fontsize=9)
                ax.set_ylabel("Affinity (kcal/mol)", color=_cc["muted"], fontsize=9)
                ax.tick_params(colors=_cc["muted"], labelsize=8)
                for sp in ax.spines.values():
                    sp.set_edgecolor(_cc["border"])
                fig.tight_layout()
                st.pyplot(fig, width='stretch')
                plt.close(fig)

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

            try:
                vbp = py3Dmol.view(width="100%", height=440)
                vbp.setBackgroundColor(_viewer_bg())
                mbp = 0
                if st.session_state.receptor_fh and os.path.exists(st.session_state.receptor_fh):
                    vbp.addModel(open(st.session_state.receptor_fh).read(), "pdb")
                    vbp.setStyle({"model": mbp}, {
                        "cartoon": {"color": "spectrum", "opacity": 0.45}
                    })
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
                        f" {_pill(_res_label, _res_kind)}",
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
        st.markdown("### 🧬 2D Interaction Diagram — PoseView2")
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

                st.session_state["_b_pv2_smiles"] = pv_sel_res.get("SMILES", pv_sel_nm)

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
                    lig_smiles          = pv_sel_res.get("SMILES", ""),
                    binding_energy      = pv_score,
                    ref_lig_name        = redock_result.get("ref_name", "") if redock_result else "",
                    ref_lig_smiles      = redock_result.get("SMILES", "")   if redock_result else "",
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
