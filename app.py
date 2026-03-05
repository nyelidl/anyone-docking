#!/usr/bin/env python3
"""
AutoDock Vina 1.2.7 — Streamlit Docking Interface
Tabs: Basic (single ligand) | Batch (multiple ligands)
Bond-order correction applied automatically before PoseView2 submission.
PoseView2 REST API: pdbCode + ligand identifier (resname_chain_resnum).
"""

import streamlit as st
import os, sys, subprocess, tempfile, io, zipfile, re as _re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit.components.v1 as components

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="anyone can dock",
    page_icon="https://raw.githubusercontent.com/nyelidl/anyone-docking/main/any-L.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Theme Helper ─────────────────────────────────────────────────────────────
import streamlit.components.v1 as _comps

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

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

:root {
    --bg:          #FFFFFF;
    --bg-subtle:   #F6F8FA;
    --bg-card:     #F0F4F8;
    --bg-input:    #FFFFFF;
    --border:      #D0D7DE;
    --border-input:#D0D7DE;
    --text:        #24292F;
    --text-muted:  #57606A;
    --accent:      #0969DA;
    --accent2:     #0550AE;
    --success:     #1A7F37;
    --warn:        #9A6700;
    --text-card-title:   #57606A;
    --text-card-heading: #24292F;
    --text-input:        #24292F;
    --pill-bg:     #DDF4FF;
    --pill-border: #54AEFF;
    --pill-text:   #0550AE;
    --ok-bg:       #DAFBE1;
    --ok-border:   #1A7F37;
    --wn-bg:       #FFF8C5;
    --wn-border:   #9A6700;
    --btn-sec-bg:  #F6F8FA;
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg:          #0d1117;
        --bg-subtle:   #161b22;
        --bg-card:     #161b22;
        --bg-input:    #21262d;
        --border:      #30363d;
        --text:        #c9d1d9;
        --text-muted:  #8b949e;
        --accent:      #58a6ff;
        --accent2:     #79c0ff;
        --success:     #3fb950;
        --warn:        #d29922;
        --text-card-title:   #8b949e;
        --text-card-heading: #e6edf3;
        --text-input:        #c9d1d9;
        --border-input:      #30363d;
        --pill-border: #1f6feb;
        --pill-text:   #79c0ff;
        --ok-bg:       #23863622;
        --ok-border:   #238636;
        --wn-bg:       #9e680322;
        --wn-border:   #9e6803;
        --btn-sec-bg:  #21262d;
    }
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text);
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
.step-card.done    { border-left-color: var(--success); }
.step-card.running { border-left-color: var(--warn); }
.step-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem; color: var(--text-card-title);
    text-transform: uppercase; letter-spacing: 2px; margin-bottom: 4px;
}
.step-heading {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem; color: var(--text-card-heading); margin-bottom: 16px;
}
.result-pill {
    display: inline-block;
    background: var(--pill-bg); border: 1px solid var(--pill-border); color: var(--pill-text);
    border-radius: 20px; padding: 2px 12px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; margin: 2px;
}
.success-pill {
    display: inline-block;
    background: var(--ok-bg); border: 1px solid var(--ok-border); color: var(--success);
    border-radius: 20px; padding: 4px 14px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
}
.warn-pill {
    display: inline-block;
    background: var(--wn-bg); border: 1px solid var(--wn-border); color: var(--warn);
    border-radius: 20px; padding: 4px 14px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
}
.log-box {
    background: var(--bg-subtle); border: 1px solid var(--border); border-radius: 6px;
    padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: var(--text-muted);
    max-height: 220px; overflow-y: auto; white-space: pre-wrap;
}
.score-best { font-family: 'IBM Plex Mono', monospace; font-size: 2.4rem; color: var(--success); font-weight: 600; }
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
.stButton > button[kind="secondary"]:hover { filter: brightness(0.95); }
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: var(--bg-input) !important; border: 1px solid var(--border-input) !important;
    color: var(--text-input) !important; border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.stSlider > div { color: var(--text); }
[data-baseweb="slider"] { accent-color: var(--accent); }
.stDataFrame { border: 1px solid var(--border); border-radius: 6px; }
hr { border-color: var(--border); }
.step-divider { border: none; border-top: 1px dashed var(--border); margin: 32px 0; }
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--bg-subtle); border-bottom: 1px solid var(--border); gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem;
    color: var(--text-muted); background: transparent; border-radius: 6px 6px 0 0;
    padding: 10px 20px;
}
/* ── Hide Streamlit iframe close/fullscreen toolbar (py3Dmol viewers) ── */
iframe { border: none !important; }
[data-testid="stIFrame"] > div > div:first-child,
div[class*="toolbar"],
div[class*="FrameToolbar"] { display: none !important; }

/* ── Ketcher Apply button — match primary green ── */
.ketcher-apply-button,
button[data-testid="ketcher-apply"],
.ketcher-container button[type="submit"],
.ketcher-container button.apply,
#ketcher-apply-btn { background: var(--success) !important; color: white !important; }

""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = dict(
    workdir=None,
    ketcher_smi="",          # Ketcher sketcher last SMILES output
    # Basic — receptor
    pdb_token=None, raw_pdb=None, receptor_fh=None, receptor_pdbqt=None,
    box_pdb=None, config_txt=None, cx=None, cy=None, cz=None,
    ligand_pdb_path=None, receptor_done=False, receptor_log="",
    cocrystal_ligand_id="",  # NEW: e.g. "ELR_A_701"
    # Basic — ligand
    ligand_pdbqt=None, ligand_sdf=None, ligand_name="ELR",
    prot_smiles=None, ligand_done=False, ligand_log="",
    # Basic — docking
    output_pdbqt=None, output_sdf=None, output_pv_sdf=None, dock_base=None,
    docking_done=False, docking_log="", score_df=None, pose_mols=None,
    # Basic — PoseView (docked pose) + PoseView2 (co-crystal reference)
    pv_image_url=None, pv_image_png=None, pv_image_svg=None, pv_pose_key=None,
    pv_ref_png=None, pv_ref_svg=None,
    # Batch — receptor
    b_pdb_token=None, b_raw_pdb=None, b_receptor_fh=None, b_receptor_pdbqt=None,
    b_box_pdb=None, b_config_txt=None, b_cx=None, b_cy=None, b_cz=None,
    b_ligand_pdb_path=None, b_receptor_done=False, b_receptor_log="",
    b_cocrystal_ligand_id="",  # NEW
    # Batch — results
    b_batch_done=False, b_batch_results=None, b_batch_log="",
    b_redock_score=None, b_redock_result=None,
    b_confirmed_ref_score=None, b_confirmed_ref_pose=None, b_confirmed_ref_name=None,
    # Batch — PoseView2
    b_pv_image_url=None, b_pv_image_png=None, b_pv_image_svg=None, b_pv_pose_key=None,
    b_pv2_image_url=None, b_pv2_image_png=None, b_pv2_image_svg=None, b_pv2_pose_key=None,
    b_pv2_ref_png=None, b_pv2_ref_svg=None,
    b_plot_png=None,
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Working Directories ──────────────────────────────────────────────────────
if st.session_state.workdir is None:
    st.session_state.workdir = tempfile.mkdtemp(prefix="vina_")
WORKDIR       = Path(st.session_state.workdir)
BATCH_WORKDIR = WORKDIR / "batch"
BATCH_WORKDIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  GENERAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def show3d(view, height=480):
    try:
        from stmol import showmol
        showmol(view, height=height)
    except ImportError:
        raw  = view._make_html()
        resp = _re.sub(r'(width\s*[:=]\s*)["\']?\d+px?["\']?', r'\g<1>100%', raw)
        # Inject CSS to remove any close/X toolbar Streamlit adds around iframes
        _no_x_css = (
            "<style>"
            "button[title='Close'],button[aria-label='Close'],"
            "div[data-testid='stToolbar'],div[class*='toolbar'],"
            ".stComponentContainer > div > div:first-child{"
            "display:none!important}"
            "</style>"
        )
        components.html(
            f'<div style="width:100%;overflow:hidden">{_no_x_css}{resp}</div>',
            height=height, scrolling=False)

def _pill(text, kind="info"):
    cls = {"info": "result-pill", "success": "success-pill", "warn": "warn-pill"}.get(kind, "result-pill")
    return f'<span class="{cls}">{text}</span>'

def run_cmd(cmd, cwd=None):
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True, cwd=cwd)
    return r.returncode, (r.stdout + r.stderr).strip()

def _rdkit_six_patch():
    try:
        from rdkit import six  # noqa
    except ImportError:
        from io import StringIO as _SIO
        from types import ModuleType as _MT
        import rdkit as _rdkit
        _m = _MT("six"); _m.StringIO = _SIO; _m.PY3 = True
        _rdkit.six = _m; sys.modules["rdkit.six"] = _m

def _meeko_to_pdbqt(mol, out_path):
    from meeko import MoleculePreparation
    prep = MoleculePreparation()
    try:
        from meeko import PDBQTWriterLegacy
        setups = prep.prepare(mol)
        pdbqt_str, _, _ = PDBQTWriterLegacy.write_string(setups[0])
    except (ImportError, AttributeError):
        prep.prepare(mol)
        pdbqt_str = prep.write_pdbqt_string()
    with open(out_path, "w") as f:
        f.write(pdbqt_str)


# ══════════════════════════════════════════════════════════════════════════════
#  BOND ORDER CORRECTION
# ══════════════════════════════════════════════════════════════════════════════
def _bo_template(smiles: str):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles!r}")
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol


def _bo_fix_mol(probe, template):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    probe_noH = Chem.RemoveHs(probe, sanitize=False)
    try:
        fixed = AllChem.AssignBondOrdersFromTemplate(template, probe_noH)
    except ValueError as exc:
        raise RuntimeError(
            f"AssignBondOrdersFromTemplate failed (atom/connectivity mismatch): {exc}"
        ) from exc

    match = fixed.GetSubstructMatch(template)
    if match:
        em = Chem.RWMol(fixed)
        for tmpl_idx, fix_idx in enumerate(match):
            fc = template.GetAtomWithIdx(tmpl_idx).GetFormalCharge()
            em.GetAtomWithIdx(fix_idx).SetFormalCharge(fc)
        fixed = em.GetMol()

    Chem.SanitizeMol(fixed)
    for prop in probe.GetPropsAsDict():
        fixed.SetProp(prop, probe.GetProp(prop))
    return fixed


def _fix_sdf_bond_orders(raw_sdf: str, smiles: str, fixed_sdf: str) -> list[str]:
    from rdkit import Chem
    log = []
    try:
        template = _bo_template(smiles)
    except Exception as e:
        log.append(f"⚠ Could not build template: {e} — skipping fix")
        import shutil
        shutil.copy(raw_sdf, fixed_sdf)
        return log

    supplier = Chem.SDMolSupplier(raw_sdf, sanitize=False, removeHs=False)
    writer  = Chem.SDWriter(fixed_sdf)
    writer.SetKekulize(False)

    ok = err = 0
    for i, mol in enumerate(supplier):
        if mol is None:
            log.append(f"⚠ Pose {i+1}: could not read mol — skipped")
            err += 1
            continue
        try:
            fixed = _bo_fix_mol(mol, template)
            fixed_h = Chem.AddHs(fixed, addCoords=False)
            writer.write(fixed_h)
            ok += 1
        except Exception as e:
            log.append(f"⚠ Pose {i+1}: bond-order fix failed ({e}) — writing raw")
            writer.write(mol)
            err += 1

    writer.close()
    log.append(f"Bond-order + charge fix: {ok} OK, {err} fallback")
    return log


def _load_pv_mols(pv_sdf: str):
    from rdkit import Chem
    return [m for m in Chem.SDMolSupplier(pv_sdf, sanitize=True, removeHs=False) if m]


def _write_single_pose(mol, path: str) -> None:
    from rdkit import Chem
    with Chem.SDWriter(path) as w:
        w.write(mol)


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTION HELPERS — distance-based residue highlight + docking grid box
# ══════════════════════════════════════════════════════════════════════════════
def _get_interacting_residues(receptor_pdb: str, lig_mol, cutoff: float = 3.5):
    """
    Return protein residues within `cutoff` Å of any ligand heavy atom.
    Each entry: {'chain': str, 'resi': int, 'resn': str}.
    Uses ProDy for fast coordinate parsing of the receptor.
    """
    try:
        import numpy as np
        from prody import parsePDB
        conf    = lig_mol.GetConformer()
        lig_xyz = np.array([
            [conf.GetAtomPosition(i).x,
             conf.GetAtomPosition(i).y,
             conf.GetAtomPosition(i).z]
            for i in range(lig_mol.GetNumAtoms())
        ])
        rec      = parsePDB(receptor_pdb)
        r_xyz    = rec.getCoords()
        chains   = rec.getChids()
        resids   = rec.getResnums()
        resnames = rec.getResnames()
        seen: dict = {}
        for j in range(len(r_xyz)):
            if np.linalg.norm(lig_xyz - r_xyz[j], axis=1).min() <= cutoff:
                key = (str(chains[j]), int(resids[j]))
                if key not in seen:
                    seen[key] = str(resnames[j])
        return [{"chain": k[0], "resi": k[1], "resn": v} for k, v in seen.items()]
    except Exception:
        return []


def _add_box_to_view(view, cx, cy, cz, sx, sy, sz):
    """
    Add a docking search-space grid box to a py3Dmol view.
    Renders a translucent filled volume plus a solid cyan wireframe outline.
    """
    try:
        _c = {"x": float(cx), "y": float(cy), "z": float(cz)}
        _d = {"w": float(sx), "h": float(sy), "d": float(sz)}
        # Faint filled interior
        view.addBox({"center": _c, "dimensions": _d,
                     "color": "blue", "opacity": 0.07, "wireframe": False})
        # Crisp wireframe outline
        view.addBox({"center": _c, "dimensions": _d,
                     "color": "cyan", "opacity": 0.90, "wireframe": True})
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  POSEVIEW REST API v1  (proteins.plus)
#  Upload receptor PDB + docked pose SDF → 2D interaction SVG of the docked pose
#  NOTE: PoseView2 (/api/poseview2_rest) only accepts pdbCode+ligandID and always
#  shows the co-crystal ligand — it does NOT support file upload.
#  To visualise a docked pose we must use PoseView v1 (/api/v2/poseview/).
# ══════════════════════════════════════════════════════════════════════════════
def _call_poseview2(receptor_pdb: str, pose_sdf: str):
    """
    Submit receptor PDB + docked pose SDF to PoseView (v1) REST API.
    Returns (svg_bytes, error_string).

    Endpoint: POST https://proteins.plus/api/v2/poseview/
    Docs:     https://proteins.plus/help/poseview_rest
    """
    import requests, time

    _SUBMIT = "https://proteins.plus/api/v2/poseview/"
    _JOBS   = "https://proteins.plus/api/v2/poseview/jobs/"

    # ── Submit job ────────────────────────────────────────────────────────────
    try:
        with open(receptor_pdb) as rf, open(pose_sdf) as lf:
            r = requests.post(
                _SUBMIT,
                files={"protein_file": rf, "ligand_file": lf},
                timeout=30,
            )
        r.raise_for_status()
        data   = r.json()
        job_id = data.get("job_id") or data.get("id")
        if not job_id:
            return None, f"No job_id in submission response: {data}"
    except Exception as e:
        return None, f"Submission failed: {e}"

    # ── Poll for result ───────────────────────────────────────────────────────
    for attempt in range(30):
        time.sleep(2)
        try:
            job    = requests.get(_JOBS + job_id + "/", timeout=10).json()
            status = job.get("status", "")
            if status in ("done", "success"):
                # Try known result keys; result_image is SVG URL in v1
                img = (job.get("result_image") or job.get("image")
                       or job.get("result")    or job.get("image_url"))
                if not img:
                    return None, f"Job finished but no image key found. Keys: {list(job.keys())}"
                if isinstance(img, str) and img.startswith("http"):
                    resp = requests.get(img, timeout=20)
                    resp.raise_for_status()
                    return resp.content, None
                return (img.encode() if isinstance(img, str) else img), None
            if status == "failed":
                return None, f"PoseView job failed: {job.get('message', '')}"
            if status not in ("pending", "running", "processing", ""):
                return None, f"Unexpected job status: '{status}'"
        except Exception as e:
            return None, f"Polling error (attempt {attempt+1}): {e}"

    return None, "Timed out waiting for PoseView result (60 s)."


def _call_poseview2_ref(pdb_code: str, ligand_id: str):
    """
    Submit a job to the PoseView2 REST API using pdbCode + ligandID.
    Returns (svg_bytes, error_string).
    Always shows the co-crystal ligand from the PDB entry.

    Endpoint: POST https://proteins.plus/api/poseview2_rest
    """
    import requests, time

    _SUBMIT = "https://proteins.plus/api/poseview2_rest"

    try:
        r = requests.post(
            _SUBMIT,
            json={"poseview2": {"pdbCode": pdb_code.strip().lower(),
                                "ligand":  ligand_id.strip()}},
            headers={"Accept": "application/json",
                     "Content-Type": "application/json"},
            timeout=30,
        )
        data = r.json()
        if r.status_code not in (200, 202):
            return None, f"Submission failed ({r.status_code}): {data.get('message', '')}"
        location = data.get("location", "")
        if not location:
            return None, "API returned no job location."
    except Exception as e:
        return None, f"Submission error: {e}"

    for attempt in range(30):
        time.sleep(2)
        try:
            poll   = requests.get(location,
                                  headers={"Accept": "application/json"},
                                  timeout=15).json()
            status = poll.get("status_code")
            if status == 200:
                svg_url = poll.get("result_svg", "")
                if not svg_url:
                    return None, "Job finished but result_svg is empty."
                resp = requests.get(svg_url, timeout=20)
                resp.raise_for_status()
                return resp.content, None
            elif status == 202:
                continue
            else:
                return None, f"Unexpected poll status: {status}"
        except Exception as e:
            return None, f"Polling error (attempt {attempt+1}): {e}"

    return None, "Timed out waiting for PoseView2 result (60 s)."


def _svg_to_png(svg_bytes: bytes):
    try:
        import cairosvg
        return cairosvg.svg2png(bytestring=svg_bytes, scale=2, background_color="white")
    except ImportError:
        st.caption("ℹ️ `cairosvg` not installed — showing SVG directly (install it for PNG export).")
        return None
    except Exception as e:
        st.caption(f"ℹ️ PNG conversion failed ({e}) — showing SVG directly.")
        return None


# Full legend — used for PoseView2 (co-crystal reference, all interaction types)
_POSEVIEW_LEGEND_HTML = """
<div style="
    background:#ffffff;
    border:1px solid #D0D7DE;
    border-radius:6px;
    padding:12px 20px;
    font-family:'Helvetica Neue',Arial,sans-serif;
    font-size:13px;
    color:#333;
    margin-top:8px;
">
  <!-- Row 1 -->
  <div style="display:flex;align-items:center;gap:40px;margin-bottom:10px;">
    <!-- hydrogen bond -->
    <div style="display:flex;align-items:center;gap:10px;white-space:nowrap;">
      <svg width="48" height="14"><line x1="0" y1="7" x2="48" y2="7"
        stroke="#5B9BD5" stroke-width="2" stroke-dasharray="5,3"/></svg>
      <span>hydrogen bond</span>
    </div>
    <!-- ionic interaction -->
    <div style="display:flex;align-items:center;gap:10px;white-space:nowrap;">
      <svg width="48" height="14"><line x1="0" y1="7" x2="48" y2="7"
        stroke="#E85D8A" stroke-width="2" stroke-dasharray="5,3"/></svg>
      <span>ionic interaction</span>
    </div>
    <!-- metal interaction -->
    <div style="display:flex;align-items:center;gap:10px;white-space:nowrap;">
      <svg width="48" height="14"><line x1="0" y1="7" x2="48" y2="7"
        stroke="#F5C400" stroke-width="2" stroke-dasharray="5,3"/></svg>
      <span>metal interaction</span>
    </div>
  </div>
  <!-- Row 2 -->
  <div style="display:flex;align-items:center;gap:40px;">
    <!-- cation-pi interaction -->
    <div style="display:flex;align-items:center;gap:10px;white-space:nowrap;">
      <svg width="56" height="14">
        <circle cx="4"  cy="7" r="4" fill="#44A44A"/>
        <line x1="8" y1="7" x2="48" y2="7"
          stroke="#AACC44" stroke-width="2" stroke-dasharray="5,3"/>
        <circle cx="52" cy="7" r="4" fill="#44A44A"/>
      </svg>
      <span>cation-pi interaction</span>
    </div>
    <!-- pi-pi interaction -->
    <div style="display:flex;align-items:center;gap:10px;white-space:nowrap;">
      <svg width="56" height="14">
        <circle cx="5"  cy="7" r="4" fill="#00BCD4"/>
        <line x1="9" y1="7" x2="47" y2="7"
          stroke="#00BCD4" stroke-width="2" stroke-dasharray="5,3"/>
        <circle cx="51" cy="7" r="4" fill="#00BCD4"/>
      </svg>
      <span>pi-pi interaction</span>
    </div>
    <!-- hydrophobic contact -->
    <div style="display:flex;align-items:center;gap:10px;white-space:nowrap;">
      <svg width="48" height="14"><line x1="0" y1="7" x2="48" y2="7"
        stroke="#2E8B57" stroke-width="2.5"/></svg>
      <span>hydrophobic contact</span>
    </div>
  </div>
</div>
"""

# Reduced legend — used for PoseView v1 (docked pose: hydrogen bond + hydrophobic only)
_POSEVIEW_V1_LEGEND_HTML = """
<div style="
    background:#ffffff;
    border:1px solid #D0D7DE;
    border-radius:6px;
    padding:12px 20px;
    font-family:'Helvetica Neue',Arial,sans-serif;
    font-size:13px;
    color:#333;
    margin-top:8px;
">
  <div style="display:flex;align-items:center;gap:40px;">
    <!-- hydrogen bond -->
    <div style="display:flex;align-items:center;gap:10px;white-space:nowrap;">
      <svg width="48" height="14"><line x1="0" y1="7" x2="48" y2="7"
        stroke="#000000" stroke-width="2" stroke-dasharray="5,3"/></svg>
      <span>hydrogen bond</span>
    </div>
    <!-- hydrophobic contact -->
    <div style="display:flex;align-items:center;gap:10px;white-space:nowrap;">
      <svg width="48" height="14"><line x1="0" y1="7" x2="48" y2="7"
        stroke="#2E8B57" stroke-width="2.5"/></svg>
      <span>hydrophobic contact</span>
    </div>
  </div>
</div>
"""


def _show_poseview_image(png_data, svg_data, caption, is_poseview2: bool = False):
    """
    Render a PoseView diagram.
    is_poseview2=True  → full 6-interaction legend (co-crystal reference)
    is_poseview2=False → reduced legend: hydrogen bond + hydrophobic contact only (docked pose)
    """
    _legend = _POSEVIEW_LEGEND_HTML if is_poseview2 else _POSEVIEW_V1_LEGEND_HTML
    if png_data:
        st.image(png_data, use_container_width=True)
        st.caption(caption)
        st.markdown(_legend, unsafe_allow_html=True)
    elif svg_data:
        svg_str = svg_data.decode("utf-8") if isinstance(svg_data, bytes) else svg_data
        svg_str = svg_str.replace("<svg ", '<svg style="width:100%;height:auto;display:block;" ', 1)
        components.html(
            f'''<div style="background:#ffffff;border-radius:8px;padding:12px;
                           border:1px solid #D0D7DE;font-family:sans-serif;">
                {svg_str}
                <p style="text-align:center;font-size:12px;color:#57606A;margin:6px 0 0 0;">
                    {caption}
                </p>
            </div>''',
            height=560,
            scrolling=True,
        )
        st.markdown(_legend, unsafe_allow_html=True)
    else:
        st.warning("No image data available.")


# ══════════════════════════════════════════════════════════════════════════════
#  POSEVIEW2 UI BLOCK  (reusable)
# ══════════════════════════════════════════════════════════════════════════════
def _poseview_ui(
    # ── PoseView v1 inputs (docked pose) ────────────────────────────────────
    rec_key: str,         # session-state key holding receptor PDB path
    pose_sdf_path: str,   # path to the single-pose SDF file to submit
    # ── PoseView2 inputs (co-crystal reference) ──────────────────────────────
    pdb_id: str = "",          # e.g. "1M17"
    cocrystal_ligand_id: str = "",  # e.g. "AQ4_A_999"
    # ── Session-state keys for docked pose image cache ───────────────────────
    smiles_key: str = "",
    pose_idx: int = 0,
    img_url_key: str = "",
    img_png_key: str = "",
    img_svg_key: str = "",
    pose_key_key: str = "",
    btn_key: str = "",
    dl_png_key: str = "",
    dl_svg_key: str = "",
    # ── Session-state keys for co-crystal reference image cache ──────────────
    ref_png_key: str = "",
    ref_svg_key: str = "",
    label_suffix: str = "",
    # ── Context for auto-filled AI prompt ────────────────────────────────────
    lig_name: str = "",
    lig_smiles: str = "",
    binding_energy: float | None = None,
    ref_lig_name: str = "",
    ref_lig_smiles: str = "",
    ref_lig_energy: float | None = None,
    show_header: bool = True,
):
    """
    2D interaction diagram block — shows both:
      LEFT : PoseView v1  — docked pose (file upload)
      RIGHT: PoseView2    — co-crystal reference (pdbCode + ligandID)
    """
    _pose_key  = f"{st.session_state.get(smiles_key, 'lig')}_pose{pose_idx+1}{label_suffix}"
    _pv_stale  = st.session_state.get(pose_key_key) != _pose_key
    _has_ref   = bool(pdb_id and cocrystal_ligand_id)
    _lig_label = st.session_state.get(smiles_key, "ligand")[:20]

    if show_header:
        st.markdown("---")
        st.markdown("**🧬 2D Interaction Diagrams**")

    _ci, _cb = st.columns([3, 1])
    with _ci:
        if _pv_stale and st.session_state.get(img_svg_key):
            st.caption("⚠️ Pose changed — click **Generate** to update.")
        else:
            _ref_note = (f" · PoseView2 co-crystal reference: **{pdb_id.upper()}** `{cocrystal_ligand_id}`"
                         if _has_ref else " · No co-crystal ID detected — only docked pose diagram will be generated.")
            st.caption(
                "**Left:** PoseView v1 — docked pose (file upload) · "
                "**Right:** PoseView2 — co-crystal reference (PDB)" + _ref_note
            )
    with _cb:
        _run_pv = st.button("🔬 Generate 2D Diagrams", key=btn_key, type="primary")

    if _run_pv:
        _rec = st.session_state.get(rec_key, "")
        if not _rec or not os.path.exists(_rec):
            st.error("Receptor PDB not found — complete receptor preparation first.")
        elif not os.path.exists(pose_sdf_path):
            st.error("Pose SDF not found.")
        else:
            # ── Left: PoseView v1 — docked pose ──────────────────────────────
            with st.spinner("⏳ PoseView v1 — docked pose… (10–60 s)"):
                _svg, _err = _call_poseview2(_rec, pose_sdf_path)
            if _err:
                st.error(f"❌ PoseView (docked pose) error: {_err}")
            else:
                _png = _svg_to_png(_svg)
                st.session_state[img_url_key]  = None
                st.session_state[img_png_key]  = _png
                st.session_state[img_svg_key]  = _svg
                st.session_state[pose_key_key] = _pose_key

            # ── Right: PoseView2 — co-crystal reference ───────────────────────
            if _has_ref and ref_png_key and ref_svg_key:
                with st.spinner(f"⏳ PoseView2 — co-crystal reference {pdb_id.upper()} / {cocrystal_ligand_id}… (10–60 s)"):
                    _ref_svg, _ref_err = _call_poseview2_ref(pdb_id, cocrystal_ligand_id)
                if _ref_err:
                    st.warning(f"⚠️ PoseView2 (co-crystal) error: {_ref_err}")
                else:
                    st.session_state[ref_png_key] = _svg_to_png(_ref_svg)
                    st.session_state[ref_svg_key] = _ref_svg

            st.rerun()

    # ── Display side-by-side ──────────────────────────────────────────────────
    _pose_svg = st.session_state.get(img_svg_key)
    _ref_svg2 = st.session_state.get(ref_svg_key) if ref_svg_key else None

    if _pose_svg and not _pv_stale:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("##### 🧪 Docked Pose (PoseView v1)")
            _png_data = st.session_state.get(img_png_key)
            _show_poseview_image(_png_data, _pose_svg,
                                 f"Docked pose {pose_idx+1} — {_lig_label}",
                                 is_poseview2=False)
            _dl1, _dl2 = st.columns(2)
            with _dl1:
                if _png_data:
                    st.download_button("⬇ PNG", data=_png_data,
                                       file_name=f"pose{pose_idx+1}_docked.png",
                                       mime="image/png", key=dl_png_key,
                                       use_container_width=True)
            with _dl2:
                st.download_button("⬇ SVG", data=_pose_svg,
                                   file_name=f"pose{pose_idx+1}_docked.svg",
                                   mime="image/svg+xml", key=dl_svg_key,
                                   use_container_width=True)

        with col_r:
            st.markdown("##### 🔬 Co-Crystal Reference (PoseView2)")
            if _ref_svg2:
                _ref_png2 = st.session_state.get(ref_png_key) if ref_png_key else None
                _show_poseview_image(_ref_png2, _ref_svg2,
                                     f"Co-crystal: {pdb_id.upper()} · {cocrystal_ligand_id}",
                                     is_poseview2=True)
                _dr1, _dr2 = st.columns(2)
                with _dr1:
                    if _ref_png2:
                        st.download_button("⬇ PNG", data=_ref_png2,
                                           file_name=f"cocrystal_{pdb_id}_{cocrystal_ligand_id}.png",
                                           mime="image/png",
                                           key=dl_png_key + "_ref",
                                           use_container_width=True)
                with _dr2:
                    st.download_button("⬇ SVG", data=_ref_svg2,
                                       file_name=f"cocrystal_{pdb_id}_{cocrystal_ligand_id}.svg",
                                       mime="image/svg+xml",
                                       key=dl_svg_key + "_ref",
                                       use_container_width=True)
            elif _has_ref:
                st.info("Click **Generate 2D Diagrams** to load the co-crystal reference.")
            else:
                st.caption("⚠️ No co-crystal ligand ID — use Auto-detect in receptor preparation.")

        if lig_smiles and ("[O-]" in lig_smiles or "+" in lig_smiles):
            st.info(
                "⚠️ **Note:** PoseView v1 uses its own protonation and may not reflect "
                f"the exact state used for docking. Docked SMILES: `{lig_smiles}`",
                icon="🧪",
            )

        # ── AI Prompt ─────────────────────────────────────────────────────────
        st.markdown("---")

        _pdb_str    = pdb_id.upper() if pdb_id else "[PDB ID]"
        _lig_str    = (f"{lig_name} (SMILES: {lig_smiles})"
                       if lig_name and lig_smiles
                       else lig_name if lig_name else "[ligand name]")
        _ref_str    = ref_lig_name.upper() if ref_lig_name else "[reference ligand]"
        _energy_str = (f"{binding_energy:.2f} kcal/mol"
                       if binding_energy is not None else "[binding energy]")

        _has_ref = bool(ref_lig_name or ref_lig_smiles)
        # Show co-crystal reference legend only when BOTH diagrams were actually generated
        _both_diagrams = bool(
            st.session_state.get(img_svg_key) and ref_svg_key
            and st.session_state.get(ref_svg_key)
        )
        if _has_ref:
            if ref_lig_name and ref_lig_smiles:
                _ref_full = f"{ref_lig_name} (SMILES: {ref_lig_smiles})"
            elif ref_lig_name:
                _ref_full = ref_lig_name
            else:
                _ref_full = ref_lig_smiles
            _ref_energy_str = (f", binding energy {ref_lig_energy:.2f} kcal/mol"
                               if ref_lig_energy is not None else "")
            _ref_clause = (
                f", and compare with the co-crystallized reference ligand "
                f"{_ref_full}{_ref_energy_str} in the same binding pocket"
            )
        else:
            _ref_clause = ""

        _prompt_text = (
            f"Analyze the attached Proteins.Plus PoseView2 interaction diagram "
            f"for PDB ID {_pdb_str}, docked ligand {_lig_str}, "
            f"generated using AutoDock Vina v1.2.7 with predicted binding energy "
            f"{_energy_str}{_ref_clause}.\n\n"
            + (f"Note: PoseView2 may re-protonate the ligand using its own internal algorithm "
               f"and may not reflect the actual protonation state used for docking. "
               f"The SMILES submitted was: `{lig_smiles}` — refer to the AI prompt below "
               f"for the correct charge state.\n\n"
               if lig_smiles and ("[O-]" in lig_smiles or "+" in lig_smiles) else "")
            + (f"Note: The docked ligand contains charged groups. "
               f"Please interpret interactions accordingly.\n\n"
               if lig_smiles and ("[O-]" in lig_smiles or "[NH2+]" in lig_smiles
                                   or "[NH+]" in lig_smiles or "[N+]" in lig_smiles) else "")
            + (
                # Both diagrams generated — show legend for each separately
                "Diagram legend (interaction types shown in the docked pose figure):\n"
                "  - Black dashed line      : hydrogen bond\n"
                "  - Dark green solid line  : hydrophobic contact\n\n"
                "Diagram legend (interaction types shown in the co-crystal reference figure):\n"
                "  - Blue dashed line         : hydrogen bond\n"
                "  - Pink dashed line         : ionic interaction\n"
                "  - Yellow dashed line       : metal interaction\n"
                "  - Green dot-dashed line    : cation-pi interaction\n"
                "  - Cyan dot-dashed line     : pi-pi interaction\n"
                "  - Dark green solid line    : hydrophobic contact\n\n"
                if _both_diagrams else
                # Only docked pose diagram available
                "Diagram legend (interaction types shown in the docked pose figure):\n"
                "  - Black dashed line      : hydrogen bond\n"
                "  - Dark green solid line  : hydrophobic contact\n\n"
            )
            + f"1. Identify key ligand–protein interactions (hydrogen bonds, hydrophobic contacts, "
            f"π–π interactions, salt bridges, etc.).\n"
            f"2. List the main interacting residues and describe their roles in stabilizing the ligand.\n"
            + (
                f"3. Compare the docking pose with the reference ligand in the same pocket.\n"
                f"4. Highlight similarities or differences in binding orientation and interaction patterns.\n"
                f"5. Evaluate whether the interaction profile supports the predicted binding energy.\n\n"
                if _has_ref else
                f"3. Evaluate whether the interaction profile supports the predicted binding energy.\n\n"
            )
            + f"Provide a concise structural interpretation of the binding mode."
        )

        st.markdown("### 🤖 AI Prompt for PoseView2 Interpretation")
        st.caption(
            "Copy and paste into any AI tool (GPT, Claude, Gemini, DeepSeek, …) "
            "together with the PoseView2 figure above. Fields are auto-filled from your session."
        )
        st.code(_prompt_text, language=None)


# ══════════════════════════════════════════════════════════════════════════════
#  VINA BINARY + pKa MODEL
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⬇ Downloading AutoDock Vina 1.2.7…")
def _get_vina():
    path = "/tmp/vina_1.2.7"
    if not os.path.exists(path) or os.path.getsize(path) < 100_000:
        rc, out = run_cmd(["wget", "-q",
            "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/"
            "v1.2.7/vina_1.2.7_linux_x86_64", "-O", path])
        if rc != 0:
            return None, out
    os.chmod(path, 0o755)
    return path, "ok"

@st.cache_resource(show_spinner="Loading pKa model…")
def _get_pka_model():
    try:
        from pkapredict import load_model
        return load_model()
    except Exception:
        return None

VINA_PATH, _vina_err = _get_vina()
PKA_MODEL             = _get_pka_model()

# ─── Ligand exclusion lists ───────────────────────────────────────────────────
_EXCLUDE_IONS   = set("HOH,WAT,DOD,SOL,NA,CL,K,CA,MG,ZN,MN,FE,CU,CO,NI,CD,HG".split(","))
_GLYCAN_NAMES   = {"NAG","BMA","MAN","FUC","GAL","GLC","SIA","NGA","FUL","GLA","BGC"}
_COFACTOR_NAMES = {"ATP","ADP","AMP","GTP","GDP","FAD","FMN","HEM","GOL","PEG","EDO","SO4","PO4"}


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED: RECEPTOR PREPARATION
# ══════════════════════════════════════════════════════════════════════════════
def _receptor_section(pfx: str, wdir: Path, step_label: str):
    import py3Dmol
    done     = st.session_state.get(pfx + "receptor_done", False)
    card_cls = "step-card done" if done else "step-card"

    st.markdown(
        f'<div class="{card_cls}"><div class="step-title">{step_label}</div>'
        f'<div class="step-heading">📦 Receptor Preparation</div>',
        unsafe_allow_html=True)

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        src = st.radio("PDB source", ["Download from RCSB", "Upload PDB file"],
                       horizontal=True, key=pfx+"src_mode")
        if src == "Download from RCSB":
            pdb_id     = st.text_input("PDB ID", value="1M17", max_chars=4, key=pfx+"pdb_id")
            upload_pdb = None
        else:
            upload_pdb = st.file_uploader("Upload .pdb", type=["pdb"], key=pfx+"pdb_upload")
            pdb_id     = None

        center_mode = st.radio("Grid center",
            ["Auto-detect co-crystal ligand", "Enter XYZ manually", "Select by atom selection (ProDy)"],
            horizontal=True, key=pfx+"center_mode")
        if center_mode == "Enter XYZ manually":
            c1, c2, c3 = st.columns(3)
            mx = c1.number_input("X", value=0.0, key=pfx+"mx")
            my = c2.number_input("Y", value=0.0, key=pfx+"my")
            mz = c3.number_input("Z", value=0.0, key=pfx+"mz")
        elif center_mode == "Select by atom selection (ProDy)":
            st.text_input(
                "ProDy selection string",
                value="resid 702 820 and chain A",
                key=pfx+"mda_sel",
                help=(
                    "Examples:\n"
                    "  resid 701 and segid A\n"
                    "  resname ELR and segid A\n"
                    "  resid 900:905 and segid B\n"
                    "  resname ATP\n"
                    "Uses ProDy selection syntax (same as used throughout this app).\n\n"
                    "Examples:\n"
                    "  resname ATP                  - by ligand residue name\n"
                    "  resname ATP and chain A       - ligand in specific chain\n"
                    "  resid 84 86 134 and chain A   - specific protein residues\n"
                    "  resid 84 to 100 and chain B   - residue range\n\n"
                    "Grid center = geometric center of all matched atoms."
                ),
            )
            st.caption(
                "💡 **ProDy examples:** "
                "`resname LIG and chain A` · "
                "`resid 701 and chain A` · "
                "`resid 84 to 100 and chain B` · "
                "`resname ATP`"
            )
    
    with col_b:
        st.markdown("**Search box size (Å)**")
        sx = st.slider("X size", 10, 40, 16, 2, key=pfx+"sx")
        sy = st.slider("Y size", 10, 40, 16, 2, key=pfx+"sy")
        sz = st.slider("Z size", 10, 40, 16, 2, key=pfx+"sz")
        st.markdown(f"Box volume: **{sx*sy*sz:,} Å³**")

    if st.button("▶ Prepare Receptor", key=pfx+"btn_receptor", type="primary"):
        from prody import parsePDB, calcCenter, writePDB
        log = []
        try:
            raw_path = str(wdir / "raw.pdb")

            if src == "Download from RCSB":
                token = pdb_id.strip().upper()
                rc, _ = run_cmd(["curl", "-sf",
                    f"https://files.rcsb.org/download/{token}.pdb", "-o", raw_path])
                if rc != 0 or not os.path.exists(raw_path) or os.path.getsize(raw_path) < 200:
                    raise ValueError(f"Download failed for {token}")
                st.session_state[pfx+"pdb_token"] = token
                log.append(f"⬇ Downloaded {token}")
            else:
                if upload_pdb is None:
                    st.error("Please upload a PDB file first."); st.stop()
                with open(raw_path, "wb") as f:
                    f.write(upload_pdb.read())
                st.session_state[pfx+"pdb_token"] = Path(upload_pdb.name).stem
                log.append(f"📂 Loaded: {upload_pdb.name}")

            atoms = parsePDB(raw_path)
            log.append(f"✓ Parsed {atoms.numAtoms()} atoms")

            ligand_pdb_path = None
            cx = cy = cz   = 0.0
            ligand_sel_str  = None
            rn = ch = ""
            ri = 0

            if center_mode == "Auto-detect co-crystal ligand":
                het = atoms.select("hetero and not water")
                if het is not None:
                    excl  = _EXCLUDE_IONS | _GLYCAN_NAMES | _COFACTOR_NAMES
                    cands = [r for r in het.getHierView().iterResidues()
                             if (r.getResname() or "").strip() not in excl]
                    if cands:
                        cands.sort(key=lambda r: (-r.numAtoms(), r.getChid() != "A"))
                        chosen = cands[0]
                        rn, ch, ri = chosen.getResname(), chosen.getChid(), chosen.getResnum()
                        ligand_sel_str  = f"resname {rn} and resid {ri} and chain {ch}"
                        lig_atoms       = atoms.select(ligand_sel_str)
                        ligand_pdb_path = str(wdir / "LIG.pdb")
                        writePDB(ligand_pdb_path, lig_atoms)
                        cx, cy, cz = (float(v) for v in calcCenter(lig_atoms))
                        log.append(f"✓ Ligand: {rn} chain {ch} resnum {ri} ({lig_atoms.numAtoms()} atoms)")
                        log.append(f"📍 Center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
                        log.append(f"🔑 PoseView2 ligand ID: {rn}_{ch}_{ri}")
                    else:
                        log.append("⚠ No co-crystal ligand found after filtering")
            elif center_mode == "Enter XYZ manually":
                cx, cy, cz = mx, my, mz
                log.append(f"🛠 Manual XYZ center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
            else:  # ProDy atom selection
                _prody_sel_str = st.session_state.get(pfx+"mda_sel", "").strip()
                if not _prody_sel_str:
                    raise ValueError("ProDy selection string is empty.")
                _ref_atoms = atoms.select(_prody_sel_str)
                if _ref_atoms is None or _ref_atoms.numAtoms() == 0:
                    raise ValueError(
                        f"ProDy selection '{_prody_sel_str}' matched 0 atoms. "
                        "Check resname / resid / chain and try again."
                    )
                cx, cy, cz = (float(v) for v in calcCenter(_ref_atoms))
                log.append(f"🔬 ProDy selection: '{_prody_sel_str}' → {_ref_atoms.numAtoms()} atoms")
                log.append(f"📍 Center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
                # Extract resname/chain/resid for PoseView2 ligand ID (same logic as Colab 1.1)
                _resnames = list(dict.fromkeys(_ref_atoms.getResnames()))
                _resids   = list(dict.fromkeys(_ref_atoms.getResnums()))
                _chains   = list(dict.fromkeys(_ref_atoms.getChids()))
                if len(_resnames) == 1 and len(_resids) == 1:
                    rn = _resnames[0]
                    ri = int(_resids[0])
                    ch = _chains[0] if _chains else "A"
                    ligand_sel_str  = f"resname {rn} and resid {ri} and chain {ch}"
                    ligand_pdb_path = str(wdir / "LIG.pdb")
                    writePDB(ligand_pdb_path, _ref_atoms)
                    log.append(f"✓ Ligand: {rn} chain {ch} resnum {ri} ({_ref_atoms.numAtoms()} atoms)")
                    log.append(f"🔑 PoseView2 ligand ID: {rn}_{ch}_{ri}")
                else:
                    # Multi-residue selection (e.g. binding site) — centers correctly, no PoseView2 ID
                    ligand_pdb_path = str(wdir / "LIG_ref.pdb")
                    writePDB(ligand_pdb_path, _ref_atoms)
                    log.append(f"⚠ Multi-residue selection ({len(_resnames)} resnames, {len(_resids)} resids) — PoseView2 ligand ID not set")
                    log.append(f"✓ Reference atoms written to LIG_ref.pdb for 3D viewer")

            sel_str = (f"not ({ligand_sel_str}) and not water"
                       if ligand_sel_str else "not water")
            rec_sel  = atoms.select(sel_str)
            rec_raw  = str(wdir / "receptor_atoms.pdb")
            writePDB(rec_raw, rec_sel)
            log.append(f"✓ Receptor: {rec_sel.numAtoms()} atoms")

            rec_fh    = str(wdir / "rec.pdb")
            rec_pdbqt = str(wdir / "rec.pdbqt")
            run_cmd(f'obabel "{rec_raw}" -O "{rec_fh}" -h 2>/dev/null')
            if os.path.getsize(rec_fh) < 100:
                raise ValueError("OpenBabel H-addition produced empty file")
            run_cmd(f'obabel "{rec_fh}" -O "{rec_pdbqt}" -xr --partialcharge gasteiger 2>/dev/null')
            if os.path.getsize(rec_pdbqt) < 100:
                raise ValueError("PDBQT conversion produced empty file")
            log.append("✓ Receptor PDBQT ready")

            box_pdb  = str(wdir / "rec.box.pdb")
            cfg_path = str(wdir / "rec.box.txt")
            hx, hy, hz = sx/2, sy/2, sz/2
            corners = [(cx+dx, cy+dy, cz+dz)
                       for dx in (-hx, hx) for dy in (-hy, hy) for dz in (-hz, hz)]
            with open(box_pdb, "w") as f:
                for i, (x, y, z) in enumerate(corners, 1):
                    f.write(f"HETATM{i:5d}  C   BOX A   1    {x:8.3f}{y:8.3f}{z:8.3f}"
                            f"  1.00  0.00           C\n")
                f.write("CONECT    1    2    3    5\nCONECT    2    1    4    6\n"
                        "CONECT    3    1    4    7\nCONECT    4    2    3    8\n"
                        "CONECT    5    1    6    7\nCONECT    6    2    5    8\n"
                        "CONECT    7    3    5    8\nCONECT    8    4    6    7\n")
            with open(cfg_path, "w") as f:
                f.write(f"center_x = {cx:.4f}\ncenter_y = {cy:.4f}\ncenter_z = {cz:.4f}\n"
                        f"size_x = {sx}\nsize_y = {sy}\nsize_z = {sz}\n")
            log.append("✓ Box + config written")

            # ── Build PoseView2 ligand identifier: resname_chain_resnum ──────
            cocrystal_ligand_id = f"{rn}_{ch}_{ri}" if ligand_sel_str else ""

            st.session_state.update({
                pfx+"raw_pdb": raw_path,         pfx+"receptor_fh": rec_fh,
                pfx+"receptor_pdbqt": rec_pdbqt, pfx+"box_pdb": box_pdb,
                pfx+"config_txt": cfg_path,      pfx+"cx": cx,
                pfx+"cy": cy,                    pfx+"cz": cz,
                pfx+"ligand_pdb_path": ligand_pdb_path,
                pfx+"cocrystal_rn": rn if ligand_sel_str else "N/A",
                pfx+"cocrystal_ligand_id": cocrystal_ligand_id,  # NEW
                pfx+"receptor_done": True,        pfx+"receptor_log": "\n".join(log),
            })
        except Exception as e:
            st.error(f"❌ Receptor preparation failed: {e}")
            st.session_state[pfx+"receptor_done"] = False
            st.session_state[pfx+"receptor_log"]  = "\n".join(log) + f"\nERROR: {e}"

    if st.session_state.get(pfx+"receptor_done"):
        token     = st.session_state.get(pfx+"pdb_token", "")
        cx_v      = st.session_state.get(pfx+"cx", 0)
        cy_v      = st.session_state.get(pfx+"cy", 0)
        cz_v      = st.session_state.get(pfx+"cz", 0)
        _sx       = st.session_state.get(pfx+"sx", 16)
        _sy       = st.session_state.get(pfx+"sy", 16)
        _sz       = st.session_state.get(pfx+"sz", 16)
        _lig_id   = st.session_state.get(pfx+"cocrystal_ligand_id", "")
        st.markdown(
            f"{_pill('Receptor ready ✓', 'success')} {_pill(token)} "
            f"{_pill(f'Center ({cx_v:.2f}, {cy_v:.2f}, {cz_v:.2f})')} "
            f"{_pill(f'Box {_sx}×{_sy}×{_sz} Å')}"
            + (f" {_pill(f'PoseView2 ligand: {_lig_id}')}" if _lig_id else ""),
            unsafe_allow_html=True)
        with st.expander("📋 Preparation log", expanded=False):
            st.markdown(
                f'<div class="log-box">{st.session_state.get(pfx+"receptor_log","")}</div>',
                unsafe_allow_html=True)
        with st.expander("🔭 3D: Receptor + Docking Box", expanded=True):
            v3 = py3Dmol.view(width="100%", height=480)
            v3.setBackgroundColor(_viewer_bg())
            mi = 0
            for path, style in [
                (st.session_state.get(pfx+"receptor_fh"),
                 {"cartoon": {"color": "spectrum", "opacity": 0.65}}),
                (st.session_state.get(pfx+"box_pdb"),
                 {"stick": {"radius": 0.2, "color": "gray"}}),
            ]:
                if path and os.path.exists(path):
                    v3.addModel(open(path).read(), "pdb")
                    v3.setStyle({"model": mi}, style); mi += 1
            lig_p = st.session_state.get(pfx+"ligand_pdb_path")
            if lig_p and os.path.exists(lig_p):
                v3.addModel(open(lig_p).read(), "pdb")
                v3.setStyle({"model": mi},
                             {"stick": {"colorscheme": "magentaCarbon", "radius": 0.25}})
            v3.zoomTo()
            if lig_p and os.path.exists(lig_p):
                v3.center({"model": mi})
            # ── Docking grid box overlay ──────────────────────────────────
            _add_box_to_view(v3,
                st.session_state.get(pfx+"cx", 0),
                st.session_state.get(pfx+"cy", 0),
                st.session_state.get(pfx+"cz", 0),
                st.session_state.get(pfx+"sx", 16),
                st.session_state.get(pfx+"sy", 16),
                st.session_state.get(pfx+"sz", 16))
            # ── Simple XYZ axis arrows from box center ─────────────────────
            try:
                _ocx = float(st.session_state.get(pfx+"cx", 0))
                _ocy = float(st.session_state.get(pfx+"cy", 0))
                _ocz = float(st.session_state.get(pfx+"cz", 0))
                _ax_len = 8.0
                for _ax_end, _ax_col, _ax_lbl in [
                    ({"x": _ocx+_ax_len, "y": _ocy,         "z": _ocz        }, "red",   "X"),
                    ({"x": _ocx,         "y": _ocy+_ax_len, "z": _ocz        }, "green", "Y"),
                    ({"x": _ocx,         "y": _ocy,         "z": _ocz+_ax_len}, "blue",  "Z"),
                ]:
                    v3.addArrow({
                        "start":  {"x": _ocx, "y": _ocy, "z": _ocz},
                        "end":    _ax_end,
                        "radius": 0.15,
                        "color":  _ax_col,
                        "radiusRatio": 3.0,
                    })
                    v3.addLabel(
                        _ax_lbl,
                        {"fontSize": 14, "fontColor": _ax_col,
                         "backgroundColor": "black", "backgroundOpacity": 0.6,
                         "inFront": True, "showBackground": True},
                        _ax_end,
                    )
            except Exception:
                pass
            show3d(v3, height=480)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<hr class="step-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
"""
<div style="display:flex; align-items:flex-start; gap:12px;">
<img src="https://raw.githubusercontent.com/nyelidl/anyone-docking/main/any-L.svg" width="70">

<h1 style="
background: linear-gradient(90deg,#ff4b4b,#ff4fa3,#7a6cff,#21a5e9);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
margin:0;
font-weight:700;
">
nyone can dock, everyone can do!
</h1>

</div>
""",
unsafe_allow_html=True
)
st.markdown("Molecular docking powered by **AutoDock Vina 1.2.7**, **pKaNET Cloud**, and **PoseView2 2D interaction**.")
st.markdown("**Basic** — single ligand.  **Batch** — multiple ligands.")
st.markdown("**☁️ Cloud-ready | 📱 iPad and smartphone-compatible**")
if VINA_PATH is None:
    st.error(f"❌ Could not download Vina binary: {_vina_err}")
    st.stop()

st.markdown(_pill("Vina 1.2.7 ready ✓", "success"), unsafe_allow_html=True)

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

    # ── Step 1: Receptor ──────────────────────────────────────────────────────
    _receptor_section(pfx="", wdir=WORKDIR, step_label="Step 1 of 4")

    # ── Step 2: Ligand ────────────────────────────────────────────────────────
    card_cls = "step-card done" if st.session_state.ligand_done else "step-card"
    st.markdown(
        f'<div class="{card_cls}"><div class="step-title">Step 2 of 4</div>'
        f'<div class="step-heading">⚗️ Ligand Preparation</div>',
        unsafe_allow_html=True)

    cl1, cl2 = st.columns([2, 1])
    with cl1:
        lig_input_mode = st.radio("Input mode",
            ["SMILES string", "Upload structure (.pdb)", "Draw structure (Ketcher)"],
            horizontal=True, key="lig_input_mode")

        smiles_in = ""
        if lig_input_mode == "SMILES string":
            smiles_in = st.text_input("SMILES string",
                value="COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
                key="smiles_in")
        elif lig_input_mode == "Upload structure (.pdb)":
            st.file_uploader("Upload structure file (.pdb)",
                             type=["sdf", "mol2", "pdb"], key="lig_struct_file")
        else:  # Draw structure (Ketcher)
            try:
                from streamlit_ketcher import st_ketcher
                _ketch_smi = st_ketcher(
                    st.session_state.get("ketcher_smi", ""),
                    height=400,
                    key="ketcher_widget",
                )
                if _ketch_smi:
                    st.session_state["ketcher_smi"] = _ketch_smi
                    smiles_in = _ketch_smi
                    st.markdown(
                        f'<div style="background:var(--bg-subtle);border:1px solid var(--border);'
                        f'border-radius:6px;padding:8px 14px;margin-top:6px;">'
                        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.8rem;'
                        f'color:var(--text-muted)">SMILES: </span>'
                        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.8rem;'
                        f'color:var(--text)">{_ketch_smi}</span></div>',
                        unsafe_allow_html=True)
                else:
                    smiles_in = st.session_state.get("ketcher_smi", "")
            except ImportError:
                st.error(
                    "❌ `streamlit-ketcher` is not installed. "
                    "Add `streamlit-ketcher==0.0.1` to your `requirements.txt` and restart the app.")
                smiles_in = ""

    with cl2:
        lig_name_in = st.text_input("Output name", value="ELR", key="lig_name_in")
        ph_in       = st.number_input("Target pH", 0.0, 14.0, 7.4, 0.1, key="ph_in")

    if not st.session_state.receptor_done:
        st.caption("⚠ Complete Step 1 first.")
    if st.button("▶ Prepare Ligand", key="btn_ligand", type="primary",
                 disabled=not st.session_state.receptor_done):
        _rdkit_six_patch()
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw
        log      = []
        lig_name = lig_name_in.strip() or "LIG"
        out_pdbqt = str(WORKDIR / f"{lig_name}.pdbqt")
        out_sdf   = str(WORKDIR / f"{lig_name}_3d.sdf")
        with st.spinner("Preparing ligand…"):
            try:
                _lig_mode = st.session_state.get("lig_input_mode", "SMILES string")
                if _lig_mode == "Upload structure (.pdb)":
                    _sfobj = st.session_state.get("lig_struct_file")
                    if _sfobj is None: raise ValueError("No structure file uploaded")
                    _ext = Path(_sfobj.name).suffix.lower()
                    _tmp = str(WORKDIR / f"lig_upload{_ext}")
                    with open(_tmp, "wb") as _f: _f.write(_sfobj.read())
                    if _ext == ".sdf":
                        _umols = [m for m in Chem.SDMolSupplier(_tmp, sanitize=True) if m]
                        if not _umols: raise ValueError("No valid molecule in SDF")
                        prot = Chem.MolToSmiles(_umols[0])
                    else:
                        _smi_tmp = _tmp + ".smi"
                        run_cmd(f'obabel "{_tmp}" -O "{_smi_tmp}" --canonical 2>/dev/null')
                        prot = ""
                        for _ln in open(_smi_tmp):
                            _pts = _ln.strip().split(None, 1)
                            if _pts: prot = _pts[0]; break
                        if not prot: raise ValueError("Could not convert structure to SMILES")
                    log.append(f"✓ Structure loaded: {_sfobj.name}")
                elif _lig_mode == "Draw structure (Ketcher)":
                    prot = st.session_state.get("ketcher_smi", "").strip()
                    if not prot: raise ValueError("No molecule drawn in Ketcher — draw a structure first")
                    log.append("✓ Structure from Ketcher sketcher")
                else:
                    prot = smiles_in.strip()
                try:
                    from dimorphite_dl import protonate_smiles
                    vs = protonate_smiles(prot, ph_min=ph_in, ph_max=ph_in, max_variants=1)
                    if vs: prot = vs[0]; log.append(f"✓ Dimorphite-DL pH {ph_in}")
                except Exception as e:
                    log.append(f"⚠ Dimorphite-DL skipped: {e}")

                mol = Chem.MolFromSmiles(prot)
                if mol is None: raise ValueError("RDKit could not parse SMILES")
                log.append(f"✓ Formal charge: {Chem.GetFormalCharge(mol):+d}")
                mol = Chem.AddHs(mol)
                try:    params = AllChem.ETKDGv3()
                except: params = AllChem.ETKDG()
                params.randomSeed = 42
                if AllChem.EmbedMolecule(mol, params) == -1:
                    AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
                if AllChem.MMFFHasAllMoleculeParams(mol):
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
                else:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=500)
                log.append("✓ 3D conformer generated")
                with Chem.SDWriter(out_sdf) as w: w.write(mol)
                _meeko_to_pdbqt(mol, out_pdbqt)
                log.append("✓ PDBQT written")
                st.session_state.update(dict(
                    ligand_pdbqt=out_pdbqt, ligand_sdf=out_sdf,
                    ligand_name=lig_name, prot_smiles=prot,
                    ligand_done=True, ligand_log="\n".join(log)))
            except Exception as e:
                st.error(f"❌ Ligand preparation failed: {e}")
                st.session_state.ligand_done = False
                st.session_state.ligand_log  = "\n".join(log) + f"\nERROR: {e}"

    if st.session_state.ligand_done:
        import py3Dmol
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw
        st.markdown(
            f"{_pill('Ligand ready ✓', 'success')} {_pill(st.session_state.ligand_name)}",
            unsafe_allow_html=True)
        with st.expander("📋 Preparation log", expanded=False):
            st.markdown(f'<div class="log-box">{st.session_state.ligand_log}</div>',
                        unsafe_allow_html=True)
        c2d, c3d = st.columns(2)
        with c2d:
            st.markdown("**2D Structure**")
            try:
                m2 = Chem.MolFromSmiles(st.session_state.prot_smiles)
                AllChem.Compute2DCoords(m2)
                buf = io.BytesIO()
                Draw.MolToImage(m2, size=(320, 260)).save(buf, format="PNG")
                st.image(buf.getvalue(), width=320)
            except Exception as e:
                st.info(f"2D unavailable: {e}")
        with c3d:
            st.markdown("**3D Conformer**")
            try:
                vl = py3Dmol.view(width="100%", height=280)
                vl.setBackgroundColor(_viewer_bg())
                vl.addModel(open(st.session_state.ligand_sdf).read(), "sdf")
                vl.setStyle({}, {"stick": {"colorscheme": "yellowCarbon", "radius": 0.2}})
                vl.zoomTo(); show3d(vl, height=280)
            except Exception as e:
                st.info(f"3D viewer unavailable: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<hr class="step-divider">', unsafe_allow_html=True)

    # ── Step 3: Docking ───────────────────────────────────────────────────────
    card_cls = "step-card done" if st.session_state.docking_done else "step-card"
    st.markdown(
        f'<div class="{card_cls}"><div class="step-title">Step 3 of 4</div>'
        f'<div class="step-heading">🚀 Run Docking</div>',
        unsafe_allow_html=True)

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
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:2rem;color:var(--warn)">'
            f'~{est}–{est*3} min</div>'
            f'<div style="color:var(--text-muted);font-size:0.8rem">exhaustiveness = {exh}</div>'
            f'</div>', unsafe_allow_html=True)

    if not st.session_state.ligand_done:
        st.caption("⚠ Complete Steps 1 & 2 first.")
    if st.button("▶ Run Docking", key="btn_dock", type="primary",
                 disabled=not st.session_state.ligand_done):
        base      = st.session_state.ligand_name
        out_pdbqt = str(WORKDIR / f"{base}_out.pdbqt")
        out_sdf   = str(WORKDIR / f"{base}_out.sdf")
        pv_sdf    = str(WORKDIR / f"{base}_pv_ready.sdf")
        with st.spinner(f"Running Vina (exhaustiveness={exh})… ⏳"):
            rc, vlog = run_cmd(
                f'"{VINA_PATH}" '
                f'--receptor "{st.session_state.receptor_pdbqt}" '
                f'--ligand "{st.session_state.ligand_pdbqt}" '
                f'--config "{st.session_state.config_txt}" '
                f'--exhaustiveness {exh} --num_modes {nm} '
                f'--energy_range {er} --out "{out_pdbqt}"',
                cwd=str(WORKDIR))
            if rc != 0 or not os.path.exists(out_pdbqt):
                st.error(f"❌ Vina failed (exit {rc})\n{vlog[:500]}")
                st.session_state.docking_done = False
            else:
                run_cmd(f'obabel "{out_pdbqt}" -O "{out_sdf}" 2>/dev/null')

                pv_log = _fix_sdf_bond_orders(
                    out_sdf, st.session_state.prot_smiles, pv_sdf)
                vlog += "\n\n── Bond-order fix ──\n" + "\n".join(pv_log)
                if not os.path.exists(pv_sdf) or os.path.getsize(pv_sdf) < 10:
                    pv_sdf = out_sdf

                data = []; cur = None
                for line in open(out_pdbqt):
                    ln = line.strip()
                    if ln.startswith("MODEL"):
                        try: cur = int(ln.split()[1])
                        except: pass
                    elif ln.startswith("REMARK VINA RESULT:"):
                        try:
                            p = ln.split()
                            data.append({"Pose": cur,
                                         "Affinity (kcal/mol)": float(p[3]),
                                         "RMSD lb": float(p[4]),
                                         "RMSD ub": float(p[5])})
                        except: pass
                df = (pd.DataFrame(data)
                      .sort_values("Affinity (kcal/mol)")
                      .reset_index(drop=True)) if data else None

                from rdkit import Chem
                mols = ([m for m in Chem.SDMolSupplier(out_sdf, sanitize=False) if m]
                        if os.path.exists(out_sdf) else [])

                st.session_state.update(dict(
                    output_pdbqt=out_pdbqt, output_sdf=out_sdf,
                    output_pv_sdf=pv_sdf,   dock_base=base,
                    docking_done=True,       docking_log=vlog,
                    score_df=df,             pose_mols=mols,
                    pv_image_url=None, pv_image_png=None,
                    pv_image_svg=None, pv_pose_key=None,
                ))

    if st.session_state.docking_done:
        st.markdown(_pill("Docking complete ✓", "success"), unsafe_allow_html=True)
        with st.expander("📋 Vina output log", expanded=False):
            st.markdown(f'<div class="log-box">{st.session_state.docking_log}</div>',
                        unsafe_allow_html=True)
        if st.session_state.score_df is not None:
            best = st.session_state.score_df["Affinity (kcal/mol)"].min()
            cls  = ("Very strong" if best < -11 else "Strong" if best < -9
                    else "Moderate" if best < -7 else "Weak")
            st.markdown(
                f'<div class="score-best">{best:.2f} '
                f'<span class="score-unit">kcal/mol</span></div>'
                f'<div style="color:#8b949e;font-size:0.9rem;margin-bottom:12px">'
                f'Best pose — {cls} predicted binding</div>',
                unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<hr class="step-divider">', unsafe_allow_html=True)

    # ── Step 4: Results ───────────────────────────────────────────────────────
    card_cls = "step-card done" if st.session_state.docking_done else "step-card"
    st.markdown(
        f'<div class="{card_cls}"><div class="step-title">Step 4 of 4</div>'
        f'<div class="step-heading">📊 Results & Visualization</div>',
        unsafe_allow_html=True)

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
                        cmap="RdYlGn", subset=["Affinity (kcal/mol)"],
                        gmap=-df["Affinity (kcal/mol)"]),
                    hide_index=True, use_container_width=True)
        with cc:
            st.markdown("**Affinity by Pose**")
            if df is not None:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                _cc = _chart_colors()
                fig.patch.set_facecolor(_cc["bg"]); ax.set_facecolor(_cc["bg_sub"])
                cols = ["#3fb950" if v == df["Affinity (kcal/mol)"].min() else "#58a6ff"
                        for v in df["Affinity (kcal/mol)"]]
                ax.bar(df["Pose"].astype(str), df["Affinity (kcal/mol)"],
                       color=cols, edgecolor=_cc["border"], linewidth=0.6)
                ax.invert_yaxis()
                ax.set_xlabel("Pose", color=_cc["muted"], fontsize=9)
                ax.set_ylabel("Affinity (kcal/mol)", color=_cc["muted"], fontsize=9)
                ax.tick_params(colors=_cc["muted"], labelsize=8)
                for sp in ax.spines.values(): sp.set_edgecolor(_cc["border"])
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close(fig)

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
                va.setStyle({"model": mai},
                             {"cartoon": {"color": "spectrum", "opacity": 0.7},
                              "stick":   {"radius": 0.1, "opacity": 0.2}}); mai += 1
            if st.session_state.ligand_pdb_path and os.path.exists(st.session_state.ligand_pdb_path):
                va.addModel(open(st.session_state.ligand_pdb_path).read(), "pdb")
                va.setStyle({"model": mai},
                             {"stick": {"colorscheme": "magentaCarbon", "radius": 0.22}}); mai += 1
            va.addModelsAsFrames(sdf_txt)
            va.setStyle({"model": mai}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.25}})
            va.animate({"interval": anim_spd, "loop": "forward"})
            va.addSurface("SES",
                {"opacity": 0.18, "color": "lightblue"},
                {"model": 0},
                {"model": mai},
            )
            va.zoomTo()
            va.center({"model": mai})
            va.rotate(30)
            show3d(va, height=440)

        st.markdown("---")

        st.markdown("**🔎 Interactive Pose Selector**")
        if mols:
            pose_idx = st.slider("Select pose", 1, len(mols), 1, key="pose_sel") - 1
            sel_mol  = mols[pose_idx]
            if df is not None:
                row = df[df["Pose"] == pose_idx + 1]
                if len(row):
                    aff = row.iloc[0]["Affinity (kcal/mol)"]
                    st.markdown(
                        f'{_pill(f"Pose {pose_idx+1}/{len(mols)}")} '
                        f'{_pill(f"Affinity: {aff:.2f} kcal/mol", "success" if aff < -8 else "warn")}',
                        unsafe_allow_html=True)

            cpv, cdl = st.columns([3, 1])
            with cpv:
                try:
                    v2 = py3Dmol.view(width="100%", height=400)
                    v2.setBackgroundColor(_viewer_bg())
                    mi2 = 0
                    if st.session_state.receptor_fh and os.path.exists(st.session_state.receptor_fh):
                        v2.addModel(open(st.session_state.receptor_fh).read(), "pdb")
                        v2.setStyle({"model": mi2},
                                     {"cartoon": {"color": "spectrum", "opacity": 0.5},
                                      "stick":   {"radius": 0.08, "opacity": 0.15}}); mi2 += 1
                    if st.session_state.ligand_pdb_path and os.path.exists(st.session_state.ligand_pdb_path):
                        v2.addModel(open(st.session_state.ligand_pdb_path).read(), "pdb")
                        v2.setStyle({"model": mi2},
                                     {"stick": {"colorscheme": "magentaCarbon", "radius": 0.2}}); mi2 += 1
                    v2.addModel(Chem.MolToMolBlock(sel_mol), "mol")
                    v2.setStyle({"model": mi2},
                                 {"stick": {"colorscheme": "cyanCarbon", "radius": 0.28}})
                    v2.addSurface("SES",
                        {"opacity": 0.2, "color": "lightblue"},
                        {"model": 0},
                        {"model": mi2},
                    )
                    v2.zoomTo()
                    v2.center({"model": mi2})
                    show3d(v2, height=400)
                except Exception as e:
                    st.info(f"Viewer error: {e}")

            with cdl:
                st.markdown("**Download**")
                sp_raw = str(WORKDIR / f"pose_{pose_idx+1}_raw.sdf")
                _write_single_pose(sel_mol, sp_raw)
                st.download_button(f"⬇ Pose {pose_idx+1} (.sdf)", open(sp_raw, "rb"),
                    file_name=f"pose_{pose_idx+1}.sdf", key=f"dl_p_{pose_idx}")
                st.download_button("⬇ All poses (.pdbqt)",
                    open(st.session_state.output_pdbqt, "rb"),
                    file_name=f"{st.session_state.dock_base}_out.pdbqt", key="dl_pdbqt")
                if df is not None:
                    st.download_button("⬇ Scores (.csv)",
                        df.to_csv(index=False).encode(),
                        file_name=f"{st.session_state.dock_base}_scores.csv",
                        mime="text/csv", key="dl_csv")
                if st.session_state.receptor_fh and os.path.exists(st.session_state.receptor_fh):
                    st.download_button("⬇ Receptor (.pdb)",
                        open(st.session_state.receptor_fh, "rb"),
                        file_name="receptor.pdb", key="dl_rec")

            # ── Binding Pocket View — residues within user-defined Å of selected pose ──
            st.markdown("---")
            st.markdown("**🔬 Binding Pocket View**")

            _bp_ctrl_l, _bp_ctrl_r = st.columns([2, 1])
            with _bp_ctrl_l:
                _bp_cutoff = st.slider(
                    "Residue distance cutoff (Å)", 2.5, 5.0, 3.5, 0.1,
                    key="bp_cutoff",
                    help="Show protein residues within this distance of any ligand atom. "
                         "Updates instantly when you change pose or cutoff.")
            with _bp_ctrl_r:
                _bp_show_labels = st.checkbox(
                    "Show residue labels", value=True, key="bp_show_labels",
                    help="Toggle yellow residue-name + resid labels on interacting residues.")

            st.caption(
                f"Protein residues within **{_bp_cutoff:.1f} Å** of "
                f"**pose {pose_idx + 1}** (orange sticks). Co-crystal ligand excluded.")

            try:
                vbp = py3Dmol.view(width="100%", height=440)
                vbp.setBackgroundColor(_viewer_bg())
                mbp = 0
                if st.session_state.receptor_fh and os.path.exists(st.session_state.receptor_fh):
                    vbp.addModel(open(st.session_state.receptor_fh).read(), "pdb")
                    vbp.setStyle({"model": mbp},
                                  {"cartoon": {"color": "spectrum", "opacity": 0.45}}); mbp += 1

                # Docked pose — cyan sticks
                vbp.addModel(Chem.MolToMolBlock(sel_mol), "mol")
                _lig_bp_model = mbp
                vbp.setStyle({"model": _lig_bp_model},
                              {"stick": {"colorscheme": "cyanCarbon", "radius": 0.30}})

                # Interacting residues — reactive to pose_idx + _bp_cutoff
                if st.session_state.receptor_fh and os.path.exists(st.session_state.receptor_fh):
                    _ir_bp = _get_interacting_residues(
                        st.session_state.receptor_fh, sel_mol, cutoff=_bp_cutoff)
                    for _rb in _ir_bp:
                        vbp.setStyle(
                            {"model": 0, "chain": _rb["chain"], "resi": _rb["resi"]},
                            {"stick": {"colorscheme": "orangeCarbon", "radius": 0.20}},
                        )
                        if _bp_show_labels:
                            vbp.addLabel(
                                f"{_rb['resn']}{_rb['resi']}",
                                {"fontSize": 11, "fontColor": "yellow",
                                 "backgroundColor": "black", "backgroundOpacity": 0.65,
                                 "inFront": True, "showBackground": True},
                                {"model": 0, "chain": _rb["chain"], "resi": _rb["resi"]},
                            )
                    _n_res = len(_ir_bp)
                    _res_label = f"{_n_res} residue{'s' if _n_res != 1 else ''}"
                    _res_kind  = "success" if _n_res else "warn"
                    st.markdown(
                        f"{_pill(f'Pose {pose_idx+1}')} "
                        f"{_pill(f'{_bp_cutoff:.1f} Å cutoff')} "
                        f"{_pill(_res_label, _res_kind)}",
                        unsafe_allow_html=True)

                vbp.zoomTo({"model": _lig_bp_model})
                show3d(vbp, height=440)
            except Exception as _e_bp:
                st.info(f"Binding pocket viewer error: {_e_bp}")

            # ── PoseView2 2D Interaction ──────────────────────────────────────
            # Write bond-order-fixed single pose SDF for PoseView2 submission
            pv_sdf_all = st.session_state.get("output_pv_sdf", "")
            sp_pv = str(WORKDIR / f"pose_{pose_idx+1}_pv_ready.sdf")
            if pv_sdf_all and os.path.exists(pv_sdf_all):
                pv_mols_all = _load_pv_mols(pv_sdf_all)
                if pv_mols_all and pose_idx < len(pv_mols_all):
                    _write_single_pose(pv_mols_all[pose_idx], sp_pv)
                else:
                    _write_single_pose(sel_mol, sp_pv)
            else:
                _write_single_pose(sel_mol, sp_pv)

            _poseview_ui(
                rec_key              = "receptor_fh",
                pose_sdf_path        = sp_pv,
                pdb_id               = st.session_state.get("pdb_token", ""),
                cocrystal_ligand_id  = st.session_state.get("cocrystal_ligand_id", ""),
                smiles_key           = "ligand_name",
                pose_idx             = pose_idx,
                img_url_key          = "pv_image_url",
                img_png_key          = "pv_image_png",
                img_svg_key          = "pv_image_svg",
                pose_key_key         = "pv_pose_key",
                btn_key              = "btn_pv_basic",
                dl_png_key           = "dl_pv_png_basic",
                dl_svg_key           = "dl_pv_svg_basic",
                ref_png_key          = "pv_ref_png",
                ref_svg_key          = "pv_ref_svg",
                label_suffix         = "_basic",
                lig_name             = st.session_state.get("ligand_name", ""),
                lig_smiles           = st.session_state.get("prot_smiles", ""),
                binding_energy       = (
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

    # ── Step B2 ───────────────────────────────────────────────────────────────
    b_rec_done   = st.session_state.get("b_receptor_done", False)
    b_batch_done = st.session_state.get("b_batch_done", False)
    card_cls = "step-card done" if b_batch_done else "step-card"
    st.markdown(
        f'<div class="{card_cls}"><div class="step-title">Step B2 of B3</div>'
        f'<div class="step-heading">⚗️ Batch Ligand Input & Docking</div>',
        unsafe_allow_html=True)

    col_b1, col_b2 = st.columns([1.6, 1])
    with col_b1:
        b_input_mode = st.radio("Input mode",
            ["SMILES list (text)", "Upload .smi file"],
            key="b_input_mode")
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
        elif b_input_mode == "Upload .smi file":
            st.file_uploader("Upload .smi file", type=["smi", "txt"], key="b_smi_file")
        b_ph = st.number_input("Target pH", 0.0, 14.0, 7.4, 0.1, key="b_ph")

    with col_b2:
        st.markdown("**Redocking validation**")
        b_do_redock = st.checkbox("Dock co-crystal ligand first as reference",
                                  value=True, key="b_do_redock")
        if b_do_redock:
            st.text_input("Co-crystal SMILES [name]",
                value="COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC Erlotinib",
                key="b_redock_smiles")
            st.caption("Score shown as dashed reference line in plot.")

        st.markdown("**Docking parameters**")
        b_exh = st.slider("Exhaustiveness", 4, 32, 8, 2, key="b_exh")
        st.caption(
            "🔍 Controls search thoroughness. Higher = more accurate but slower. "
            "**8** is fast for screening; **16–32** for final validation."
        )
        b_nm  = st.slider("Poses per ligand", 5, 20, 10, 1, key="b_nm")
        st.caption(
            "📐 Number of binding poses saved per ligand. "
            "**10** captures diverse conformations for analysis."
        )
        b_er  = st.slider("Energy range (kcal/mol)", 1, 5, 3, 1, key="b_er")
        st.caption(
            "⚡ Only poses within this range of the best score are kept. "
            "**3 kcal/mol** balances diversity vs. relevance."
        )

    if not b_rec_done:
        st.caption("⚠ Complete Step B1 first.")
    if st.button("▶ Run Batch Docking", key="b_btn_dock", type="primary",
                 disabled=not b_rec_done):
        _rdkit_six_patch()
        from rdkit import Chem
        from rdkit.Chem import AllChem

        rec_pdbqt = st.session_state.get("b_receptor_pdbqt")
        config    = st.session_state.get("b_config_txt")
        b_ph_val  = st.session_state.get("b_ph", 7.4)

        smiles_pairs = []
        try:
            mode = st.session_state.get("b_input_mode", "SMILES list (text)")
            if mode == "SMILES list (text)":
                for line in st.session_state.get("b_smiles_text", "").strip().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    pts = line.split(None, 1)
                    smiles_pairs.append((
                        pts[0],
                        pts[1].replace(" ", "_") if len(pts) > 1 else f"lig_{len(smiles_pairs)+1}"
                    ))
            elif mode == "Upload .smi file":
                fobj = st.session_state.get("b_smi_file")
                if fobj is None: raise ValueError("No .smi file uploaded")
                for line in fobj.read().decode().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    pts = line.split(None, 1)
                    smiles_pairs.append((
                        pts[0],
                        pts[1].replace(" ", "_") if len(pts) > 1 else f"lig_{len(smiles_pairs)+1}"
                    ))
            if not smiles_pairs: raise ValueError("No valid SMILES found")
        except Exception as e:
            st.error(f"❌ Input parsing failed: {e}"); st.stop()

        def _prep_one(smi, name, ph, wdir):
            pdbqt_path = str(wdir / f"{name}.pdbqt")
            try:
                prot = smi
                try:
                    from dimorphite_dl import protonate_smiles
                    vs = protonate_smiles(prot, ph_min=ph, ph_max=ph, max_variants=1)
                    if vs: prot = vs[0]
                except Exception: pass
                mol = Chem.MolFromSmiles(prot)
                if mol is None: raise ValueError(f"Cannot parse SMILES: {smi[:50]}")
                charge = Chem.GetFormalCharge(mol)
                mol = Chem.AddHs(mol)
                try:    params = AllChem.ETKDGv3()
                except: params = AllChem.ETKDG()
                params.randomSeed = 42
                if AllChem.EmbedMolecule(mol, params) == -1:
                    AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
                if AllChem.MMFFHasAllMoleculeParams(mol):
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
                else:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=500)
                _meeko_to_pdbqt(mol, pdbqt_path)
                return pdbqt_path, None, charge
            except Exception as e:
                return None, str(e), None

        def _dock_one(pdbqt_in, name, exh, nm, er):
            out_pdbqt = str(BATCH_WORKDIR / f"{name}_out.pdbqt")
            out_sdf   = str(BATCH_WORKDIR / f"{name}_out.sdf")
            rc, log   = run_cmd(
                f'"{VINA_PATH}" --receptor "{rec_pdbqt}" --ligand "{pdbqt_in}" '
                f'--config "{config}" --exhaustiveness {exh} --num_modes {nm} '
                f'--energy_range {er} --out "{out_pdbqt}"',
                cwd=str(BATCH_WORKDIR))
            if rc != 0 or not os.path.exists(out_pdbqt):
                return None, None, log, None, []
            run_cmd(f'obabel "{out_pdbqt}" -O "{out_sdf}" 2>/dev/null')
            pose_scores = []
            for line in open(out_pdbqt):
                if line.strip().startswith("REMARK VINA RESULT:"):
                    try: pose_scores.append(float(line.split()[3]))
                    except: pass
            top = pose_scores[0] if pose_scores else None
            return out_pdbqt, out_sdf, log, top, pose_scores

        redock_score  = None
        redock_result = None
        if st.session_state.get("b_do_redock", False):
            raw_rd = st.session_state.get("b_redock_smiles", "").strip()
            pts    = raw_rd.split(None, 1)
            rd_smi = pts[0]
            rd_nm  = (pts[1].replace(" ", "_") if len(pts) > 1 else "redock")
            with st.spinner(f"Docking reference ligand ({rd_nm})…"):
                rd_pdbqt, rd_err, rd_charge = _prep_one(rd_smi, "redock_" + rd_nm, b_ph_val, BATCH_WORKDIR)
                if rd_pdbqt:
                    rd_out_pdbqt, rd_out_sdf, _, rd_top, rd_pose_scores = _dock_one(
                        rd_pdbqt, "redock_" + rd_nm, b_exh, b_nm, b_er)
                    if rd_top is not None:
                        redock_score = rd_top
                        rd_pv_sdf = str(BATCH_WORKDIR / f"redock_{rd_nm}_pv_ready.sdf")
                        _fix_sdf_bond_orders(rd_out_sdf, rd_smi, rd_pv_sdf)
                        if not os.path.exists(rd_pv_sdf) or os.path.getsize(rd_pv_sdf) < 10:
                            rd_pv_sdf = rd_out_sdf

                        rd_n_poses = 0
                        if rd_out_sdf and os.path.exists(rd_out_sdf):
                            rd_n_poses = sum(
                                1 for m in Chem.SDMolSupplier(rd_out_sdf, sanitize=False) if m)
                        redock_result = {
                            "Name":        f"⭐ {rd_nm} (co-crystal ref)",
                            "ref_name":    rd_nm if rd_nm != "redock" else "",
                            "SMILES":      rd_smi,
                            "Charge":      rd_charge,
                            "Top Score":   rd_top,
                            "pose_scores": rd_pose_scores,
                            "Poses":       rd_n_poses,
                            "out_pdbqt":   rd_out_pdbqt,
                            "out_sdf":     rd_out_sdf,
                            "pv_sdf":      rd_pv_sdf,
                            "Status":      "OK",
                            "is_redock":   True,
                        }
                        st.success(f"✓ Reference score: **{redock_score:.2f} kcal/mol** ({rd_nm})")
                    else:
                        st.warning("⚠ Redocking failed — no score returned")
                else:
                    st.warning(f"⚠ Reference ligand prep failed: {rd_err}")

        results  = []
        n        = len(smiles_pairs)
        prog     = st.progress(0, text=f"Docking 0/{n}…")
        log_slot = st.empty()
        all_logs = []

        for i, (smi, name) in enumerate(smiles_pairs):
            prog.progress(i / n, text=f"Docking {name} ({i+1}/{n})…")
            pdbqt_in, prep_err, charge = _prep_one(smi, name, b_ph_val, BATCH_WORKDIR)
            if pdbqt_in is None:
                results.append({"Name": name, "SMILES": smi, "Charge": None,
                                 "Top Score": None, "Poses": 0,
                                 "Status": f"PREP FAILED: {prep_err}"})
                all_logs.append(f"[{name}] PREP ERROR: {prep_err}"); continue

            out_pdbqt, out_sdf, dock_log, top, pose_scores = _dock_one(
                pdbqt_in, name, b_exh, b_nm, b_er)
            all_logs.append(f"[{name}] score={top} | {dock_log[:120]}")
            log_slot.markdown(
                f'<div class="log-box">{"".join(all_logs[-5:])}</div>',
                unsafe_allow_html=True)

            if top is None:
                results.append({"Name": name, "SMILES": smi, "Charge": charge,
                                 "Top Score": None, "Poses": 0,
                                 "Status": "DOCK FAILED"}); continue

            pv_sdf = str(BATCH_WORKDIR / f"{name}_pv_ready.sdf")
            _fix_sdf_bond_orders(out_sdf, smi, pv_sdf)
            if not os.path.exists(pv_sdf) or os.path.getsize(pv_sdf) < 10:
                pv_sdf = out_sdf

            n_poses = 0
            if out_sdf and os.path.exists(out_sdf):
                n_poses = sum(1 for m in Chem.SDMolSupplier(out_sdf, sanitize=False) if m)
            results.append({
                "Name": name, "SMILES": smi, "Charge": charge, "Top Score": top,
                "pose_scores": pose_scores, "Poses": n_poses,
                "out_pdbqt": out_pdbqt, "out_sdf": out_sdf,
                "pv_sdf": pv_sdf, "Status": "OK",
            })

        n_ok_final = sum(1 for r in results if r["Status"] == "OK")
        prog.progress(1.0, text=f"✓ Done — {n_ok_final}/{n} ligands docked successfully")
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
            "b_pv_image_url": None, "b_pv_image_png": None,
            "b_pv_image_svg": None, "b_pv_pose_key":  None,
            "b_pv2_image_url": None, "b_pv2_image_png": None,
            "b_pv2_image_svg": None, "b_pv2_pose_key":  None,
            "b_plot_png": None,
        })

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<hr class="step-divider">', unsafe_allow_html=True)

    # ── Step B3: Results ──────────────────────────────────────────────────────
    b_batch_done = st.session_state.get("b_batch_done", False)
    card_cls = "step-card done" if b_batch_done else "step-card"
    st.markdown(
        f'<div class="{card_cls}"><div class="step-title">Step B3 of B3</div>'
        f'<div class="step-heading">📊 Batch Results</div>',
        unsafe_allow_html=True)

    if not b_batch_done:
        st.info("Complete Step B2 to see batch results here.")
    else:
        import py3Dmol
        from rdkit import Chem
        results             = st.session_state.get("b_batch_results", [])
        redock_score        = st.session_state.get("b_redock_score")
        redock_result       = st.session_state.get("b_redock_result")
        confirmed_ref_score = st.session_state.get("b_confirmed_ref_score")
        confirmed_ref_pose  = st.session_state.get("b_confirmed_ref_pose")
        confirmed_ref_name  = st.session_state.get("b_confirmed_ref_name")
        active_ref_score    = (confirmed_ref_score if confirmed_ref_score is not None
                               else redock_score)

        n_ok   = sum(1 for r in results if r["Status"] == "OK")
        n_fail = len(results) - n_ok
        st.markdown(
            f"{_pill(f'{n_ok} ligands docked ✓', 'success')} "
            f"{_pill('AutoDock Vina 1.2.7')}"
            + (f" {_pill(f'{n_fail} failed', 'warn')}" if n_fail else ""),
            unsafe_allow_html=True)

        st.markdown("**🔎 Pose Browser**")
        ok_results = [r for r in results
                      if r["Status"] == "OK"
                      and r.get("out_sdf") and os.path.exists(r["out_sdf"])]
        if redock_result and redock_result.get("out_sdf") and os.path.exists(redock_result["out_sdf"]):
            browsable = [redock_result] + ok_results
        else:
            browsable = ok_results

        if browsable:
            sel_nm = st.selectbox(
                "Select ligand", [r["Name"] for r in browsable], index=0, key="b_lig_sel")
            sel_res       = next(r for r in browsable if r["Name"] == sel_nm)
            is_redock_sel = sel_res.get("is_redock", False)
            pose_scores_list = sel_res.get("pose_scores", [])

            b_mols = [m for m in Chem.SDMolSupplier(sel_res["out_sdf"], sanitize=False) if m]
            if b_mols:
                b_pose_i = st.slider("Pose", 1, len(b_mols), 1, key="b_pose_sel") - 1
                this_pose_score = (pose_scores_list[b_pose_i]
                                   if pose_scores_list and b_pose_i < len(pose_scores_list)
                                   else sel_res["Top Score"])
                score_kind = ("success" if (this_pose_score is not None
                                            and this_pose_score < -8) else "warn")

                row_pills = (
                    f'{_pill(f"Pose {b_pose_i+1} / {len(b_mols)}")}'
                    f'{_pill(f"Score: {this_pose_score:.2f} kcal/mol", score_kind) if this_pose_score is not None else ""}'
                )
                if pose_scores_list and b_pose_i > 0 and len(pose_scores_list) > 1:
                    delta = this_pose_score - pose_scores_list[0]
                    row_pills += f' {_pill(f"Δ {delta:+.2f} vs pose 1")}'

                if is_redock_sel:
                    st.markdown(
                        f'<div style="margin-bottom:6px">'
                        f'{_pill("⭐ Co-crystal reference ligand", "warn")}</div>',
                        unsafe_allow_html=True)
                    if confirmed_ref_score is not None:
                        st.markdown(
                            f'<div style="background:#23863622;border:1px solid #238636;'
                            f'border-radius:8px;padding:10px 16px;margin-bottom:10px;'
                            f'font-family:\'IBM Plex Mono\',monospace;">'
                            f'<span style="color:#3fb950;font-size:0.85rem;">✅ Reference locked:</span> '
                            f'<b style="color:#3fb950">{confirmed_ref_score:.2f} kcal/mol</b>'
                            f'<span style="color:#8b949e;font-size:0.8rem;"> — pose '
                            f'{confirmed_ref_pose} of {confirmed_ref_name}</span></div>',
                            unsafe_allow_html=True)

                st.markdown(row_pills, unsafe_allow_html=True)

                cbv, cbd = st.columns([3, 1])
                with cbv:
                    try:
                        vb = py3Dmol.view(width="100%", height=420)
                        vb.setBackgroundColor(_viewer_bg()); bmi = 0
                        rec_fh = st.session_state.get("b_receptor_fh")
                        if rec_fh and os.path.exists(rec_fh):
                            vb.addModel(open(rec_fh).read(), "pdb")
                            vb.setStyle({"model": bmi},
                                         {"cartoon": {"color": "spectrum", "opacity": 0.7},
                                          "stick":   {"radius": 0.08, "opacity": 0.15}}); bmi += 1
                        lig_p = st.session_state.get("b_ligand_pdb_path")
                        if lig_p and os.path.exists(lig_p):
                            vb.addModel(open(lig_p).read(), "pdb")
                            vb.setStyle({"model": bmi},
                                         {"stick": {"colorscheme": "magentaCarbon",
                                                    "radius": 0.2}}); bmi += 1
                        vb.addModel(Chem.MolToMolBlock(b_mols[b_pose_i]), "mol")
                        vb.setStyle({"model": bmi},
                                     {"stick": {"colorscheme": "cyanCarbon", "radius": 0.28}})
                        vb.addSurface("SES",
                            {"opacity": 0.2, "color": "lightblue"},
                            {"model": 0},
                            {"model": bmi},
                        )
                        vb.zoomTo()
                        vb.center({"model": bmi})
                        show3d(vb, height=420)
                    except Exception as e:
                        st.info(f"Viewer error: {e}")

                with cbd:
                    st.markdown("**Actions**")

                    if is_redock_sel and this_pose_score is not None:
                        already_confirmed = (
                            confirmed_ref_score == this_pose_score
                            and confirmed_ref_pose == b_pose_i + 1
                        )
                        btn_label = (f"✅ Confirmed (pose {b_pose_i+1})"
                                     if already_confirmed
                                     else f"📌 Use pose {b_pose_i+1} as reference")
                        if st.button(btn_label, key="b_confirm_ref_btn",
                                     type="primary" if not already_confirmed else "secondary",
                                     use_container_width=True):
                            st.session_state["b_confirmed_ref_score"] = this_pose_score
                            st.session_state["b_confirmed_ref_pose"]  = b_pose_i + 1
                            st.session_state["b_confirmed_ref_name"]  = sel_nm
                            st.rerun()
                        if confirmed_ref_score is not None and not already_confirmed:
                            if st.button("🔄 Reset reference", key="b_reset_ref_btn",
                                         use_container_width=True):
                                st.session_state["b_confirmed_ref_score"] = None
                                st.session_state["b_confirmed_ref_pose"]  = None
                                st.session_state["b_confirmed_ref_name"]  = None
                                st.rerun()

                    st.markdown("**Download**")
                    safe_sel_nm = sel_nm.replace("⭐ ", "").replace(" (co-crystal ref)", "")
                    sp3 = str(BATCH_WORKDIR / f"{safe_sel_nm}_pose{b_pose_i+1}.sdf")
                    _write_single_pose(b_mols[b_pose_i], sp3)
                    st.download_button(f"⬇ Pose {b_pose_i+1} (.sdf)", open(sp3, "rb"),
                        file_name=f"{safe_sel_nm}_pose{b_pose_i+1}.sdf", key="b_dl_pose")
                    if sel_res.get("out_pdbqt") and os.path.exists(sel_res["out_pdbqt"]):
                        st.download_button("⬇ All poses (.pdbqt)",
                            open(sel_res["out_pdbqt"], "rb"),
                            file_name=f"{safe_sel_nm}_out.pdbqt", key="b_dl_pdbqt")

        st.markdown("---")

        with st.expander("📋 Full docking log", expanded=False):
            st.markdown(
                f'<div class="log-box">{st.session_state.get("b_batch_log","")}</div>',
                unsafe_allow_html=True)

        df_res = pd.DataFrame([
            {"Name": r["Name"],
             "Top Score (kcal/mol)": r["Top Score"],
             "Charge": (f"{r['Charge']:+d}" if r.get("Charge") is not None else "—"),
             "Status": r["Status"]}
            for r in results
        ])
        ok_df = (df_res[df_res["Status"] == "OK"]
                 .sort_values("Top Score (kcal/mol)")
                 .reset_index(drop=True))

        # plot_df: OK ligands in original input order (same as Score Table)
        plot_df = df_res[df_res["Status"] == "OK"].reset_index(drop=True)

        # Score Table — shown as-is (Poses column already removed from df_res)
        _display_df = df_res.copy()

        if not plot_df.empty:
            _n_ligs = len(plot_df)
            _best_score = ok_df["Top Score (kcal/mol)"].min()

            def _draw_plot(ax, fw_hint):
                _cc = _chart_colors()
                ax.get_figure().patch.set_facecolor(_cc["bg"])
                ax.set_facecolor(_cc["bg_sub"])
                scores = plot_df["Top Score (kcal/mol)"].values
                names  = plot_df["Name"].values
                colors = ["#3fb950" if s == _best_score else "#58a6ff" for s in scores]
                # Integer x-positions → each ligand gets its own slot, duplicates stay separate
                xs = list(range(_n_ligs))
                ax.scatter(xs, scores, color=colors, s=90, zorder=3,
                           edgecolors=_cc["border"], linewidths=0.5)
                ax.plot(xs, scores, color=_cc["border"], linewidth=0.8, zorder=2)
                ax.set_xticks(xs)
                ax.set_xticklabels(names, rotation=40, ha="right")
                ax.set_xlim(-0.5, _n_ligs - 0.5)
                if active_ref_score is not None:
                    ref_label = (
                        f"✓ Confirmed ref (pose {confirmed_ref_pose}): {active_ref_score:.2f} kcal/mol"
                        if confirmed_ref_score is not None
                        else f"Co-crystal ref (top pose): {active_ref_score:.2f} kcal/mol"
                    )
                    ax.axhline(active_ref_score, color="#f85149", linewidth=1.8,
                               linestyle="--", label=ref_label)
                    ax.legend(facecolor=_cc["legend_bg"], edgecolor=_cc["border"],
                              labelcolor=_cc["text"], fontsize=8)
                ax.set_ylabel("Vina score (kcal/mol)", color=_cc["muted"], fontsize=9)
                ax.set_xlabel("Ligand",               color=_cc["muted"], fontsize=9)
                ax.tick_params(colors=_cc["muted"], labelsize=7)
                for sp in ax.spines.values(): sp.set_edgecolor(_cc["border"])
                ax.grid(axis="y", color=_cc["bg_sub"], linewidth=0.5)

            if _n_ligs <= 10:
                # ≤10 ligands: original side-by-side layout
                ct2, cp2 = st.columns([1, 1.6])
                with ct2:
                    st.markdown("**Score Table**")
                    st.dataframe(_display_df, hide_index=True, use_container_width=True)
                with cp2:
                    st.markdown("**Top Score per Ligand**")
                    fig, ax = plt.subplots(figsize=(max(5, _n_ligs * 0.6 + 1.5), 3.5))
                    _draw_plot(ax, max(5, _n_ligs * 0.6 + 1.5))
                    fig.tight_layout()
                    _buf = __import__("io").BytesIO()
                    fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight",
                                facecolor=fig.get_facecolor())
                    _buf.seek(0)
                    st.session_state["b_plot_png"] = _buf.getvalue()
                    st.pyplot(fig, use_container_width=True); plt.close(fig)
            else:
                # >10 ligands: plot full-width first, table below
                st.markdown("**Top Score per Ligand**")
                _fw = max(6, _n_ligs * 0.9 + 1.5)
                fig, ax = plt.subplots(figsize=(_fw, 4))
                _draw_plot(ax, _fw)
                fig.tight_layout()
                _buf = __import__("io").BytesIO()
                fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                _buf.seek(0)
                st.session_state["b_plot_png"] = _buf.getvalue()
                st.pyplot(fig, use_container_width=True); plt.close(fig)
                st.markdown("**Score Table**")
                st.dataframe(_display_df, hide_index=True, use_container_width=True)
        else:
            st.markdown("**Score Table**")
            st.dataframe(_display_df, hide_index=True, use_container_width=True)

        st.markdown("---")

        st.markdown("**⬇ Download All Results**")
        c_csv, c_zip = st.columns(2)
        with c_csv:
            if not ok_df.empty:
                st.download_button("⬇ Top scores (.csv)",
                    ok_df.to_csv(index=False).encode(),
                    file_name="batch_scores.csv", mime="text/csv", key="b_dl_csv")
        with c_zip:
            zb = io.BytesIO()
            zip_results = ([redock_result] if redock_result else []) + ok_results
            with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as zf:
                for r in zip_results:
                    sn = r["Name"].replace("⭐ ", "").replace(" (co-crystal ref)", "")
                    if r.get("out_sdf") and os.path.exists(r["out_sdf"]):
                        zf.write(r["out_sdf"], f"poses/{sn}_out.sdf")
                    if r.get("pv_sdf") and os.path.exists(r["pv_sdf"]):
                        zf.write(r["pv_sdf"], f"poses_pv_ready/{sn}_pv_ready.sdf")
                    if r.get("out_pdbqt") and os.path.exists(r["out_pdbqt"]):
                        zf.write(r["out_pdbqt"], f"pdbqt/{sn}_out.pdbqt")
                if not ok_df.empty:
                    zf.writestr("batch_scores.csv", ok_df.to_csv(index=False))
                rec_fh = st.session_state.get("b_receptor_fh")
                if rec_fh and os.path.exists(rec_fh):
                    zf.write(rec_fh, "receptor.pdb")
                _plot_bytes = st.session_state.get("b_plot_png")
                if _plot_bytes:
                    zf.writestr("plots/batch_score_plot.png", _plot_bytes)
                for _sfx, _pk, _sk in [
                    ("poseview2_browser",  "b_pv_image_png",  "b_pv_image_svg"),
                    ("poseview2_selector", "b_pv2_image_png", "b_pv2_image_svg"),
                ]:
                    _png = st.session_state.get(_pk)
                    _svg = st.session_state.get(_sk)
                    if _png: zf.writestr(f"poseview2/{_sfx}.png", _png)
                    if _svg: zf.writestr(f"poseview2/{_sfx}.svg", _svg)
            zb.seek(0)
            st.download_button("⬇ Download ALL results (.zip) — structures + plot + 2D diagrams", zb,
                file_name="anyone_can_dock.zip",
                mime="application/zip", key="b_dl_zip")

        # ── PoseView2 2D Interaction — independent selector ───────────────────
        st.markdown("---")
        st.markdown("### 🧬 2D Interaction Diagram — PoseView2")
        st.caption(
            "Generates a 2D interaction map for the co-crystal ligand from the PDB entry "
            "using the PoseView2 REST API (pdbCode + ligand identifier). "
            "Select any docked ligand below to associate context for the AI prompt."
        )

        pv_browsable = [r for r in browsable
                        if r.get("out_sdf") and os.path.exists(r["out_sdf"])]
        if pv_browsable:
            pv_sel_nm  = st.selectbox(
                "Associate docked ligand (for AI prompt context)",
                [r["Name"] for r in pv_browsable],
                index=0,
                key="b_pv_lig_sel",
            )
            pv_sel_res  = next(r for r in pv_browsable if r["Name"] == pv_sel_nm)
            pv_safe_nm  = pv_sel_nm.replace("⭐ ", "").replace(" (co-crystal ref)", "")
            pv_all_mols = [m for m in Chem.SDMolSupplier(
                               pv_sel_res["out_sdf"], sanitize=False) if m]

            if pv_all_mols:
                pv_pose_i = st.slider(
                    "Pose (for AI prompt context)", 1, len(pv_all_mols), 1,
                    key="b_pv_pose_sel") - 1

                pv_pose_scores = pv_sel_res.get("pose_scores", [])
                pv_score = (pv_pose_scores[pv_pose_i]
                            if pv_pose_scores and pv_pose_i < len(pv_pose_scores)
                            else pv_sel_res.get("Top Score"))

                st.session_state["_b_pv2_smiles"] = pv_sel_res.get("SMILES", pv_sel_nm)

                # Write bond-order-fixed single pose SDF for PoseView2 submission
                pv_sdf_all_path = pv_sel_res.get("pv_sdf", "")
                sp_pv2 = str(BATCH_WORKDIR / f"{pv_safe_nm}_pose{pv_pose_i+1}_pv2_ready.sdf")
                if pv_sdf_all_path and os.path.exists(pv_sdf_all_path):
                    pv_mols2 = _load_pv_mols(pv_sdf_all_path)
                    if pv_mols2 and pv_pose_i < len(pv_mols2):
                        _write_single_pose(pv_mols2[pv_pose_i], sp_pv2)
                    else:
                        _write_single_pose(pv_all_mols[pv_pose_i], sp_pv2)
                else:
                    _write_single_pose(pv_all_mols[pv_pose_i], sp_pv2)

                _poseview_ui(
                    rec_key             = "b_receptor_fh",
                    pose_sdf_path       = sp_pv2,
                    pdb_id              = st.session_state.get("b_pdb_token", ""),
                    cocrystal_ligand_id = st.session_state.get("b_cocrystal_ligand_id", ""),
                    smiles_key          = "_b_pv2_smiles",
                    pose_idx            = pv_pose_i,
                    img_url_key         = "b_pv2_image_url",
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
                    ref_lig_name        = (redock_result.get("ref_name", "")
                                           if redock_result else ""),
                    ref_lig_smiles      = (redock_result.get("SMILES", "")
                                           if redock_result else ""),
                    ref_lig_energy      = (redock_result.get("Top Score")
                                           if redock_result else None),
                    show_header         = False,
                )

    st.markdown('</div>', unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown('<hr class="step-divider">', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;color:#57606A;font-size:0.78rem;'
    'font-family:\'IBM Plex Mono\',monospace;">'
    'AutoDock Vina 1.2.7 · Meeko · RDKit · OpenBabel · py3Dmol<br>'
    'Eberhardt et al. J. Chem. Inf. Model. 2021, 61, 3891–3898 &nbsp;·&nbsp; '
    '<a href="https://pubs.acs.org/doi/10.1021/acs.jcim.5c02852" target="_blank" '
    'style="color:#58a6ff;text-decoration:none;">'
    'DFDD — Hengphasatporn et al. J. Chem. Inf. Model. 2026</a>'
    '</div>',
    unsafe_allow_html=True,
)
