#!/usr/bin/env python3
"""
app.py — Streamlit UI layer for Anyone Can Dock.

Responsibilities:
  - All st.* widgets, layout, CSS, HTML, JavaScript
  - Session state management
  - Imports all computation from core.py
  - No business logic here (no bond-order math, no receptor prep, etc.)

Architecture rule: if a function has no st.* call, it belongs in core.py.
"""

import os
import io
import sys
import json
import time
import base64
import tempfile
import zipfile
import textwrap
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Core imports (computation layer — no Streamlit inside)
# ---------------------------------------------------------------------------
from core import (
    # Vina
    get_vina_binary, run_vina,
    # Receptor
    prepare_receptor, scan_ligands, detect_cocrystal_ligand,
    write_box_pdb, write_vina_config, convert_cif_to_pdb, is_cif_file,
    strip_and_convert_receptor,
    # Ligand
    prepare_ligand, prepare_ligand_from_file, smiles_from_file,
    # Bond orders / SDF
    fix_sdf_bond_orders, load_mols_from_sdf, write_single_pose,
    write_single_pose_pdb, convert_sdf_to_v2000,
    # Analysis
    get_interacting_residues, calc_rmsd_heavy,
    # PoseView
    call_poseview_v1, call_poseview2_ref, warm_poseview_cache,
    # Image
    svg_to_png, stamp_png,
    # 2D diagram
    draw_interaction_diagram, draw_interaction_diagram_data,
    draw_interactions_rdkit, draw_interactions_rdkit_classic,
    draw_interaction_diagram_interactive,
    # Search (moved from old app.py)
    search_pubchem, search_rcsb,
    # ADMET (moved + upgraded)
    calc_adme_properties, _load_admet_model,
    # Utilities
    run_cmd, check_obabel,
    HEME_RESNAMES, METAL_RESNAMES,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Anyone Can Dock",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* General */
.main .block-container { padding-top: 1.2rem; }
h1 { font-size: 1.65rem !important; font-weight: 700; }
h2 { font-size: 1.25rem !important; }
h3 { font-size: 1.05rem !important; }

/* Score badge */
.score-badge {
    display: inline-block;
    background: #0e4d92;
    color: white;
    font-weight: 700;
    font-size: 1.25rem;
    padding: 4px 18px;
    border-radius: 20px;
    letter-spacing: 0.03em;
}

/* Metric card */
.metric-card {
    background: #f7f9fc;
    border: 1.5px solid #dde3ef;
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
    margin-bottom: 6px;
}
.metric-card .label {
    font-size: 11px;
    color: #666;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.metric-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1a1a2e;
}

/* ADME ML badge */
.ml-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #EAF3E5;
    border: 1.5px solid #4CAF50;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 600;
    color: #2e7d32;
    margin-bottom: 8px;
}
.ml-badge-fallback {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #FFF8E1;
    border: 1.5px solid #FFA000;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 600;
    color: #e65100;
    margin-bottom: 8px;
}

/* CYP bar */
.cyp-bar-wrap {
    background: #eee;
    border-radius: 6px;
    height: 8px;
    width: 100%;
    margin-top: 3px;
}
.cyp-bar-fill {
    height: 8px;
    border-radius: 6px;
    background: linear-gradient(90deg, #43a047, #e53935);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Locating AutoDock-Vina binary…")
def _cached_vina():
    return get_vina_binary()


@st.cache_resource(show_spinner="Loading ADMET-AI ML models (first run may take ~30 s)…")
def _cached_admet_model():
    """Cache the ADMET-AI model at the Streamlit resource level."""
    return _load_admet_model()


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_SS_DEFAULTS = {
    # Receptor
    "src_mode":              "Download PDB",
    "pdb_id":                "",
    "pdb_token":             "",
    "rcsb_fmt":              "CIF",
    "rec_search_query":      "",
    "rec_search_results":    [],
    "rec_prepared":          False,
    "rec_info":              {},
    "center_mode":           "Auto-detect co-crystal ligand",
    "mda_sel":               "",
    "cx": 0.0, "cy": 0.0, "cz": 0.0,
    "sx": 16,  "sy": 16,  "sz": 16,
    "blind_docking":         False,
    "keep_cofactors":        True,
    "keep_metals":           True,
    "rcsb_prefer_complete":  True,
    # Ligand
    "lig_input_mode":        "PubChem / SMILES",
    "smiles_in":             "",
    "lig_name_in":           "ligand",
    "ph_in":                 7.4,
    "lig_upload_prot":       "Use the uploaded form",
    "lig_prepared":          False,
    "lig_info":              {},
    "pub_search_query":      "",
    "pub_search_result":     {},
    # Docking
    "docking_done":          False,
    "docking_result":        {},
    "exh_slider":            16,
    "n_modes":               10,
    "e_range":               3,
    "do_redock":             True,
    "redock_smiles":         "",
    # Results
    "pose_sel":              1,
    "anim_spd":              1500,
    "bp_cutoff":             3.5,
    "bp_show_labels":        True,
    "bp_show_surface":       False,
    # Batch
    "b_input_mode":          "SMILES list (text area)",
    "b_smiles_text":         "",
    "b_exh":                 8,
    "b_nm":                  9,
    "b_do_redock":           False,
    # ADME
    "adme_results":          {},
}

for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Working directory helpers
# ---------------------------------------------------------------------------

def _wdir() -> Path:
    if "wdir" not in st.session_state:
        st.session_state["wdir"] = Path(tempfile.mkdtemp(prefix="acd_"))
    return st.session_state["wdir"]


# ---------------------------------------------------------------------------
# 50 widget help strings  (English only)
# ---------------------------------------------------------------------------

_H = {
    # ---- Receptor (10) -----------------------------------------------
    "src_mode": (
        "Source of the protein 3D structure.\n\n"
        "📖 Download = fetch from RCSB by 4-letter PDB ID.\n"
        "   Upload  = use your own .pdb or .cif file.\n"
        "⚙️ Use Upload for homology models or pre-prepared structures.\n"
        "⚠️ CIF recommended for large entries (>100 kDa) — PDB truncates at 99,999 atoms."
    ),
    "pdb_id": (
        "4-letter Protein Data Bank identifier (e.g. 1M17, 4AGN).\n\n"
        "📖 Each ID is a unique experimental structure.\n"
        "   Same protein can have many IDs with different resolutions or ligands.\n"
        "⚙️ Prefer resolution < 2.5 Å, no missing residues.\n"
        "⚠️ Different entries of the same protein can give different docking results.\n"
        "   Search at rcsb.org to compare options."
    ),
    "rcsb_fmt": (
        "Download format from RCSB.\n\n"
        "📖 PDB = classic format, widely compatible.\n"
        "   CIF = modern mmCIF, required for large structures.\n"
        "⚙️ Use CIF if the protein has > 62 chains or > 99,999 atoms.\n"
        "⚠️ If PDB download fails, CIF is tried automatically."
    ),
    "center_mode": (
        "Method to set the center of the docking search box.\n\n"
        "📖 The box MUST cover the binding site.\n"
        "   Wrong center = Vina searches the wrong region.\n"
        "⚙️ Auto-detect: best when co-crystal ligand is in the PDB.\n"
        "   Manual XYZ : enter coordinates from a known binding site.\n"
        "   ProDy sel  : select by residue/chain string.\n"
        "⚠️ If no ligand is found, grid defaults to protein centroid."
    ),
    "mda_sel": (
        "ProDy atom selection string to define the grid center.\n"
        "The centroid of selected atoms becomes the box center.\n\n"
        "📖 Examples:\n"
        "   resid 702 and chain A\n"
        "   resname ATP and chain B\n"
        "   resid 100 to 120 and chain A\n"
        "⚙️ Use residue numbers from literature or binding site databases.\n"
        "⚠️ Chain IDs must match exactly — check your PDB file header."
    ),
    "box_size": (
        "Docking search box size along this axis (Angstroms).\n\n"
        "📖 Vina only searches for poses within this box.\n"
        "⚙️ 16–20 Å : most drug-like molecules (default).\n"
        "   24–30 Å : large ligands or flexible loops.\n"
        "   12 Å    : tight rigid binding pockets.\n"
        "⚠️ Box > 30 Å significantly increases calculation time."
    ),
    "blind_docking": (
        "Expand the search box to cover the entire protein.\n\n"
        "📖 Use when the binding site is completely unknown.\n"
        "   Vina will explore all surface pockets simultaneously.\n"
        "⚙️ Enable for novel targets with no structural data.\n"
        "⚠️ 5–20× slower than focused docking.\n"
        "   Results are less reliable — use to identify candidate sites,\n"
        "   then re-dock with a focused box."
    ),
    "keep_cofactors": (
        "Retain cofactors (ATP, FAD, NAD, CoA, heme…) in the receptor.\n\n"
        "📖 Cofactors shape the binding pocket.\n"
        "   Removing them creates a cavity that may not exist in vivo.\n"
        "⚙️ Keep ON  : cofactor is always present (most cases).\n"
        "   Turn OFF : designing a competitive cofactor-site inhibitor.\n"
        "⚠️ ATP in kinases and FAD in oxidoreductases are critical —\n"
        "   removing them usually gives unrealistic poses."
    ),
    "keep_metals": (
        "Retain metal ions (Zn, Mg, Ca, Fe, Cu…) in the receptor.\n\n"
        "📖 Metal ions in active sites act as coordination centers.\n"
        "   Chelating ligands must be placed near the ion.\n"
        "⚙️ Keep ON  : metalloenzymes (proteases, carbonic anhydrase).\n"
        "   Turn OFF : testing non-chelating pose hypotheses.\n"
        "⚠️ Vina does not model metal coordination explicitly.\n"
        "   Validate metal-binding poses with QM or specialized tools."
    ),
    "rcsb_prefer_complete": (
        "Prefer RCSB entries with no missing residues in search results.\n\n"
        "📖 Missing residues = structural gaps.\n"
        "   Gaps near the binding site reduce docking reliability.\n"
        "⚙️ Keep ON for most targets (default).\n"
        "   Disable if the best-resolution structure has missing residues.\n"
        "⚠️ Even 'complete' structures may have flexible loops.\n"
        "   Always inspect the structure before docking."
    ),

    # ---- Ligand (5) --------------------------------------------------
    "lig_input_mode": (
        "How to provide the ligand structure.\n\n"
        "📖 PubChem/SMILES : text-based molecule representation.\n"
        "   Upload          : use an existing 3D file (.sdf/.mol2/.pdb).\n"
        "   Ketcher         : draw the structure interactively.\n"
        "⚙️ SMILES for known drugs — search PubChem by name.\n"
        "   Upload if you already have a prepared 3D file.\n"
        "⚠️ Uploaded files skip protonation — ensure correct H atoms."
    ),
    "smiles_in": (
        "SMILES string for the ligand (e.g. CCO = ethanol).\n\n"
        "📖 Text representation of molecular structure.\n"
        "   Each letter = atom, numbers = ring closures.\n"
        "⚙️ Copy from PubChem, ChEMBL, or DrugBank.\n"
        "   Use the search box above to auto-fill from compound name.\n"
        "⚠️ Include correct stereochemistry (@/@@) if chiral centers exist.\n"
        "   Wrong stereochemistry = unreliable docking results."
    ),
    "lig_name_in": (
        "Short identifier for output files from this ligand.\n\n"
        "📖 Used to name: name.pdbqt, name_out.sdf, name_scores.csv.\n"
        "⚙️ Keep it short and descriptive (e.g. Erlotinib, Cpd_01).\n"
        "⚠️ No spaces or special characters — use underscores instead."
    ),
    "ph_in": (
        "pH used to calculate the ligand protonation state.\n\n"
        "📖 pH determines which atoms are charged.\n"
        "   Amines (pKa ~10) are +1 at pH 7.4.\n"
        "   Carboxylic acids (pKa ~4.5) are -1 at pH 7.4.\n"
        "⚙️ 7.4 = plasma/cytosol (default).\n"
        "   6.5 = tumor microenvironment.\n"
        "   5.0 = lysosome / endosome.\n"
        "⚠️ Wrong pH can shift net charge ±1 — large effect on affinity."
    ),
    "lig_upload_prot": (
        "How to handle hydrogens for an uploaded structure file.\n\n"
        "📖 Use uploaded form : keep all atoms exactly as in the file.\n"
        "   Protonate at pH   : re-run Dimorphite-DL on extracted SMILES.\n"
        "⚙️ 'As uploaded' if file is already correctly prepared.\n"
        "   'At pH' if file is only a 3D template for coordinates.\n"
        "⚠️ 'As uploaded' skips all protonation checks.\n"
        "   Incorrect H atoms in the file = poor docking quality."
    ),

    # ---- Docking (5) -------------------------------------------------
    "exh_slider": (
        "Search thoroughness — higher = more reliable, slower.\n\n"
        "📖 Controls number of Monte Carlo steps Vina performs.\n"
        "   Higher values reduce chance of missing the global minimum.\n"
        "⚙️ 8  = quick test / screening.\n"
        "   16 = standard work (default).\n"
        "   32 = publication quality.\n"
        "   64 = maximum (30–60 min per ligand).\n"
        "⚠️ Run time ∝ exhaustiveness — plan accordingly."
    ),
    "n_modes": (
        "Maximum number of binding poses to output.\n\n"
        "📖 Poses ranked best (1) to worst by binding affinity.\n"
        "   Pose 1 = predicted best binding mode.\n"
        "⚙️ 9–10 : standard (default).\n"
        "   15–20 : when diverse alternative poses are needed.\n"
        "⚠️ More poses does not improve the best result.\n"
        "   Poses below rank 5 are often unreliable."
    ),
    "e_range": (
        "Max allowed energy gap from the best pose (kcal/mol).\n\n"
        "📖 Poses worse than (best + range) are discarded.\n"
        "   Example: best=-9.0, range=3 → discard above -6.0.\n"
        "⚙️ 3 kcal/mol = standard (default).\n"
        "   1–2 = only tightly clustered top poses.\n"
        "   4–5 = explore diverse binding modes.\n"
        "⚠️ Large range + many poses can produce low-quality output."
    ),
    "do_redock": (
        "Re-dock the known co-crystal ligand as a validation control.\n\n"
        "📖 If re-docking reproduces the crystal pose (RMSD ≤ 2 Å),\n"
        "   the protocol is validated. Score = reference baseline.\n"
        "⚙️ Enable when a co-crystal ligand exists in the PDB entry.\n"
        "   Disable for fast screening (saves one docking run).\n"
        "⚠️ RMSD > 2 Å = protocol may have issues.\n"
        "   Check: box placement, receptor prep, ligand SMILES."
    ),
    "redock_smiles": (
        "SMILES and name for the redocking reference ligand.\n"
        "Format: SMILES LigandName  (space-separated).\n\n"
        "📖 Example: CCO Ethanol\n"
        "   Name labels output files and score plots.\n"
        "⚙️ Copy SMILES from PubChem or ChEMBL.\n"
        "⚠️ SMILES must exactly match the PDB co-crystal ligand\n"
        "   chemistry — not just a similar compound."
    ),

    # ---- Results (7) -------------------------------------------------
    "pose_sel": (
        "Select which docking pose to inspect.\n\n"
        "📖 Pose 1 = best predicted binding affinity.\n"
        "   Higher poses = progressively less favorable.\n"
        "⚙️ Start with pose 1. Check pose 2–3 for alternative modes.\n"
        "⚠️ Pose rank ≠ biological relevance.\n"
        "   A slightly worse pose with better pharmacophore match\n"
        "   may be more meaningful."
    ),
    "anim_spd": (
        "Interval between frames in the pose animation (milliseconds).\n\n"
        "📖 Lower = faster cycling. Higher = longer per pose.\n"
        "⚙️ 500 ms  = quick overview of all poses.\n"
        "   1500 ms = default, comfortable viewing speed.\n"
        "   3000 ms = slow, careful inspection per pose."
    ),
    "bp_cutoff": (
        "Max distance (Å) to count a residue as interacting with the ligand.\n\n"
        "📖 Residues within cutoff → shown as orange sticks + labeled.\n"
        "   < 3.5 Å : direct interactions (H-bond, ionic).\n"
        "   < 5.0 Å : includes hydrophobic contacts.\n"
        "⚙️ Start at 3.5 Å. Increase if too few residues shown.\n"
        "⚠️ > 5.0 Å includes irrelevant residues — clutters the view."
    ),
    "bp_show_labels": (
        "Show residue name + number + chain as 3D labels.\n\n"
        "📖 Yellow labels on each interacting residue in the viewer.\n"
        "⚙️ Enable for analysis, disable for clean screenshots.\n"
        "⚠️ Dense pockets → overlapping labels.\n"
        "   Reduce distance cutoff to show fewer residues."
    ),
    "bp_show_surface": (
        "Render a solvent-excluded surface (SES) around the protein.\n\n"
        "📖 Shows the 3D shape of the binding pocket.\n"
        "   Good fit = ligand sits snugly inside the cavity.\n"
        "⚙️ Enable to assess shape complementarity.\n"
        "   Disable for clearer residue contact view.\n"
        "⚠️ Increases GPU load — may be slow on mobile or large proteins."
    ),
    "diag_cutoff": (
        "Interaction distance cutoff for the 2D diagram (Å).\n\n"
        "📖 Residues within this distance → drawn as labeled circles.\n"
        "   Circle color = interaction type (see legend below diagram).\n"
        "⚙️ 4.0–4.5 Å = captures most meaningful interactions.\n"
        "   Can differ from the 3D pocket viewer cutoff.\n"
        "⚠️ < 3.0 Å may miss hydrophobic contacts.\n"
        "   > 5.0 Å clutters the diagram with weak contacts."
    ),
    "diag_max_res": (
        "Max residues shown in the 2D interaction diagram.\n\n"
        "📖 Top N by priority (metal > ionic > H-bond > hydrophobic).\n"
        "   Remaining residues are omitted for readability.\n"
        "⚙️ 10–14 : standard (default).\n"
        "    6–8  : simple ligands with few contacts.\n"
        "   18–20 : complex multi-residue interactions.\n"
        "⚠️ Too many residues → unreadable diagram."
    ),

    # ---- ADME (13) ---------------------------------------------------
    "adme_mw": (
        "Total molecular mass (Daltons).\n\n"
        "📖 ≤ 500 Da : passes Lipinski RO5.\n"
        "   > 600 Da : likely poor oral absorption.\n"
        "   Most oral drugs: 200–500 Da.\n"
        "⚙️ Reduce: remove heavy groups, use bioisosteres.\n"
        "⚠️ MW alone is insufficient — consider all descriptors."
    ),
    "adme_logp": (
        "Lipophilicity (octanol-water log partition coefficient).\n\n"
        "📖 < 0 : too hydrophilic — may not cross cell membranes.\n"
        "   0–3 : ideal range.\n"
        "   3–5 : acceptable (Lipinski ≤ 5).\n"
        "   > 5 : poor solubility, toxicity risk.\n"
        "⚙️ Reduce: add polar groups (OH, NH, COOH).\n"
        "   Increase: add aromatic rings, halogens.\n"
        "⚠️ LogP > 5 correlates strongly with hERG inhibition."
    ),
    "adme_tpsa": (
        "Sum of polar atom surface areas — predicts membrane permeability.\n\n"
        "📖 < 90 Å²  : good GI absorption + BBB penetration.\n"
        "   90–140 Å²: moderate GI absorption.\n"
        "   > 140 Å² : poor oral bioavailability.\n"
        "   > 200 Å² : essentially impermeable.\n"
        "⚙️ CNS drugs: target TPSA < 90 Å².\n"
        "   Reduce: N-methylation, prodrug strategies.\n"
        "⚠️ Calculated from 2D topology — an approximation, not true 3D."
    ),
    "adme_hbd": (
        "Number of NH and OH groups (hydrogen bond donors).\n\n"
        "📖 ≤ 3 : good for membrane permeability.\n"
        "   ≤ 5 : Lipinski RO5 threshold.\n"
        "   > 5 : Lipinski violation, poor membrane crossing.\n"
        "⚙️ Reduce: N-methylation, prodrug strategies.\n"
        "⚠️ Each donor has a desolvation cost at the membrane.\n"
        "   Too many donors = high penalty even if LogP is acceptable."
    ),
    "adme_hba": (
        "Number of N and O atoms (hydrogen bond acceptors).\n\n"
        "📖 ≤ 7  : good.\n"
        "   ≤ 10 : Lipinski RO5 threshold.\n"
        "   > 10 : Lipinski violation.\n"
        "⚙️ Reduce: remove ether oxygens, use bioisosteres.\n"
        "⚠️ HBA > 12 strongly predicts poor oral absorption."
    ),
    "adme_qed": (
        "Composite drug-likeness score (0 = worst, 1 = best).\n\n"
        "📖 0.67–1.0 : high drug-likeness.\n"
        "   0.35–0.67: moderate.\n"
        "   0–0.35   : poor drug-likeness.\n"
        "   Most oral drugs: 0.5–0.9.\n"
        "⚙️ Improve by optimizing the 8 underlying properties.\n"
        "⚠️ Derived from oral small molecules — not valid\n"
        "   for biologics, PROTACs, or other modalities."
    ),
    "adme_fsp3": (
        "Fraction of carbon atoms that are sp³ (non-aromatic).\n\n"
        "📖 > 0.25: better solubility, selectivity, clinical success.\n"
        "   < 0.25: flat molecule — tends to aggregate, less selective.\n"
        "⚙️ Increase: add piperidine, cyclohexane, chiral centers.\n"
        "   'Escape from flatland' — a key medicinal chemistry strategy.\n"
        "⚠️ A guideline, not a rule. Some flat drugs work very well."
    ),
    "adme_gi": (
        "Predicted oral GI absorption.\n\n"
        "📖 High   : good for oral dosing.\n"
        "   Medium : may need formulation help.\n"
        "   Low    : consider IV or alternative route.\n"
        "⚙️ Improve: reduce MW, HBD, TPSA.\n"
        "⚠️ Estimated value only.\n"
        "   Confirm with Caco-2 assay or in vivo data."
    ),
    "adme_bbb": (
        "Predicted blood-brain barrier penetration.\n\n"
        "📖 Penetrant     : expected CNS exposure.\n"
        "   Possible      : partial penetration.\n"
        "   Non-penetrant : mainly peripheral.\n"
        "⚙️ Increase: TPSA < 90, MW < 450, HBD ≤ 3, LogP 1–3.\n"
        "   Decrease: add polar groups, increase MW.\n"
        "⚠️ For peripheral targets, Non-penetrant is preferred\n"
        "   to avoid CNS side effects."
    ),
    "adme_pgp": (
        "Predicted P-glycoprotein efflux transporter substrate.\n\n"
        "📖 Likely   : P-gp may actively efflux the drug.\n"
        "             Reduces CNS penetration + oral bioavailability.\n"
        "   Unlikely : lower efflux liability.\n"
        "⚙️ Reduce liability: lower MW, HBA/HBD, polar surface.\n"
        "⚠️ Critical for CNS drugs — P-gp is highly expressed at BBB."
    ),
    "adme_cyp": (
        "Predicted CYP enzyme inhibition (drug metabolism).\n\n"
        "📖 Possible  : may inhibit this isoform → DDI risk.\n"
        "   Unlikely  : lower inhibition risk.\n"
        "   CYP3A4 metabolizes ~50% of all drugs.\n"
        "⚙️ Reduce risk: avoid basic N + aromatic ring combos.\n"
        "⚠️ SMARTS flags have ~40% false positive rate.\n"
        "   Use as a signal to investigate, not a definitive result."
    ),
    "adme_pains": (
        "Pan-Assay INterference compound alerts.\n\n"
        "📖 0 alerts : clean scaffold.\n"
        "   1+ alerts: may give false positives in biochemical screens.\n"
        "   Common issues: aggregation, fluorescence, redox cycling.\n"
        "⚙️ Avoid PAINS substructures in hit-to-lead optimization.\n"
        "⚠️ PAINS ≠ inactive. It means confirmation assays are needed.\n"
        "   Some approved drugs contain PAINS-flagged substructures."
    ),
    "adme_brenk": (
        "Structural alerts for reactive or unstable substructures.\n\n"
        "📖 0 alerts : no reactive groups.\n"
        "   1+ alerts: stability issues or metabolic liabilities.\n"
        "⚙️ Review each alert individually.\n"
        "   Michael acceptors may be intentional for covalent inhibitors.\n"
        "⚠️ BRENK alerts vary in severity — not all are disqualifying.\n"
        "   Prioritize by relevance to your assay conditions."
    ),

    # ---- Batch (5) ---------------------------------------------------
    "b_input_mode": (
        "How to provide multiple ligands for batch docking.\n\n"
        "📖 SMILES list    : type directly, one per line.\n"
        "   Upload .smi    : file with one SMILES [name] per line.\n"
        "   Structure files: .sdf/.mol2/.pdb files.\n"
        "⚙️ SMILES fastest for known compounds.\n"
        "   .smi upload for large libraries (100+).\n"
        "⚠️ All ligands share the same receptor, box, and parameters."
    ),
    "b_smiles_text": (
        "One ligand per line: SMILES [optional_name].\n\n"
        "📖 Format example:\n"
        "   CCO Ethanol\n"
        "   c1ccccc1 Benzene\n"
        "   # this line is a comment\n"
        "⚙️ Names: short, no spaces (use underscores).\n"
        "   No name = auto-numbered (lig_1, lig_2…).\n"
        "⚠️ Very large molecules (> 100 heavy atoms) may fail.\n"
        "   Test problematic compounds individually."
    ),
    "b_exh": (
        "Search thoroughness per ligand (same as single-ligand mode).\n\n"
        "📖 Lower = faster but potentially less accurate.\n"
        "⚙️ 4–8  : initial screening of large libraries.\n"
        "   16   : focused follow-up on promising hits.\n"
        "   32   : final validation.\n"
        "⚠️ Total time = N ligands × time per ligand.\n"
        "   20 ligands × exh 16 ≈ 20–60 min total."
    ),
    "b_nm": (
        "Max binding poses per ligand in batch mode.\n\n"
        "📖 Pose 1 affinity = used for ranking across the batch.\n"
        "   Extra poses available for inspecting selected hits.\n"
        "⚙️ 5–9 : standard for batch screening.\n"
        "⚠️ More poses → larger files, more memory.\n"
        "   For 50+ ligands, keep poses ≤ 9."
    ),
    "b_do_redock": (
        "Dock the reference co-crystal ligand as a validation control.\n\n"
        "📖 Reference score = dashed red line in the batch score plot.\n"
        "   Ligands better than reference → prioritize for follow-up.\n"
        "⚙️ Enable for any serious batch campaign.\n"
        "⚠️ Adds one full docking run to total time.\n"
        "   Uses the same exhaustiveness as the batch."
    ),

    # ---- Figure (5) --------------------------------------------------
    "rtf_src": (
        "Source for the 2D ligand-receptor interaction diagram.\n\n"
        "📖 RDKit       : fast, local, no network required.\n"
        "   PoseView v1 : proteins.plus REST API (server-side render).\n"
        "   PoseView2   : reference mode using PDB co-crystal ligand ID.\n"
        "⚙️ Use RDKit for offline work or rapid iteration.\n"
        "   Use PoseView for publication-quality diagrams.\n"
        "⚠️ PoseView requires internet access and may be slow."
    ),
    "rtf_layout": (
        "Layout of the multi-panel figure.\n\n"
        "📖 2-panel : 3D view + 2D diagram side by side.\n"
        "   4-panel : 3D view + 2D + ADME radar + score table.\n"
        "⚙️ 2-panel for presentations. 4-panel for full analysis.\n"
        "⚠️ 4-panel requires ADME to be calculated first."
    ),
    "rtf_cutoff": (
        "Distance cutoff for including residues in the figure diagram.\n\n"
        "📖 Same meaning as the 2D diagram cutoff in Results section.\n"
        "⚙️ 4.0–4.5 Å for most targets.\n"
        "⚠️ Larger cutoff = more residues = slower rendering."
    ),
    "rtf_labels": (
        "Show residue labels in the figure's 3D panel.\n\n"
        "📖 Labels appear as overlaid text on the protein in the PNG.\n"
        "⚙️ Keep ON for analysis figures, OFF for presentation slides.\n"
        "⚠️ Dense pockets with many residues produce overlapping labels."
    ),
    "rtf_surf": (
        "Render protein surface in the figure's 3D panel.\n\n"
        "📖 SES surface shows the cavity shape around the ligand.\n"
        "⚙️ Enable for shape complementarity visualization.\n"
        "⚠️ Surface rendering adds GPU load — may be slow for large proteins."
    ),
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _h(key: str) -> str:
    """Return help string for a widget key, or empty string if not defined."""
    return _H.get(key, "")


def _smiles_img_html(img_url: str, width: int = 160) -> str:
    return (
        f'<img src="{img_url}" width="{width}" '
        f'style="border-radius:8px;border:1px solid #dde;background:white;" />'
    )


def _affinity_color(aff: float) -> str:
    if aff <= -9.0:
        return "#1a7f37"
    if aff <= -7.0:
        return "#9a6700"
    return "#cf222e"


def _lipinski_badge(pass_: bool) -> str:
    color = "#1a7f37" if pass_ else "#cf222e"
    label = "Pass" if pass_ else "Fail"
    return (
        f'<span style="background:{color};color:white;'
        f'font-size:11px;font-weight:700;padding:2px 9px;'
        f'border-radius:10px;">{label}</span>'
    )


def _rule_badge(pass_: bool, label_pass: str = "Pass", label_fail: str = "Fail") -> str:
    color = "#1a7f37" if pass_ else "#cf222e"
    return (
        f'<span style="background:{color};color:white;'
        f'font-size:11px;font-weight:700;padding:2px 9px;'
        f'border-radius:10px;">{"✓ " + label_pass if pass_ else "✗ " + label_fail}</span>'
    )


# ---------------------------------------------------------------------------
#  PubChem compound search (UI wrapper)
# ---------------------------------------------------------------------------

def _pubchem_search_widget(prefix: str = ""):
    """Inline PubChem search bar that auto-fills SMILES and name."""
    key_q  = prefix + "pub_search_query"
    key_res = prefix + "pub_search_result"
    if key_q not in st.session_state:
        st.session_state[key_q] = ""
    if key_res not in st.session_state:
        st.session_state[key_res] = {}

    with st.expander("🔍 Search compound by name (PubChem)", expanded=False):
        col_a, col_b = st.columns([4, 1])
        with col_a:
            q = st.text_input(
                "Compound name", value=st.session_state[key_q],
                placeholder="e.g. erlotinib, ibuprofen",
                key=key_q + "_inp",
                label_visibility="collapsed",
            )
        with col_b:
            search_clicked = st.button("Search", key=prefix + "btn_pubchem")

        if search_clicked and q.strip():
            with st.spinner("Searching PubChem…"):
                result = search_pubchem(q.strip())
            st.session_state[key_res] = result
            st.session_state[key_q]   = q.strip()

        res = st.session_state.get(key_res, {})
        if res.get("found"):
            c1, c2 = st.columns([1, 3])
            with c1:
                if res.get("img_url"):
                    st.markdown(_smiles_img_html(res["img_url"], 120), unsafe_allow_html=True)
            with c2:
                st.markdown(f"**{res.get('iupac', '')}**")
                st.caption(
                    f"Formula: {res.get('formula','')}  "
                    f"MW: {res.get('mw',0):.1f} Da  "
                    f"CID: {res.get('cid','')}"
                )
                if st.button("✓ Use this compound", key=prefix + "btn_use_pubchem"):
                    st.session_state[prefix + "smiles_in"]   = res["smiles"]
                    st.session_state[prefix + "lig_name_in"] = (
                        res.get("iupac", "ligand").replace(" ", "_")[:20]
                    )
                    st.rerun()
        elif res and not res.get("found"):
            st.warning(res.get("error", "Not found."))


# ---------------------------------------------------------------------------
#  RCSB protein search (UI wrapper)
# ---------------------------------------------------------------------------

def _rcsb_search_widget(prefix: str = ""):
    key_q   = prefix + "rec_search_query"
    key_res = prefix + "rec_search_results"
    key_sel = prefix + "rec_search_sel"
    if key_q   not in st.session_state: st.session_state[key_q]   = ""
    if key_res not in st.session_state: st.session_state[key_res] = []
    if key_sel not in st.session_state: st.session_state[key_sel] = 0

    with st.expander("🔍 Search RCSB by protein name", expanded=False):
        col_a, col_b = st.columns([4, 1])
        with col_a:
            q = st.text_input(
                "Protein name / keyword",
                value=st.session_state[key_q],
                placeholder="e.g. EGFR kinase, CDK2",
                key=key_q + "_inp",
                label_visibility="collapsed",
            )
        with col_b:
            searched = st.button("Search", key=prefix + "btn_rcsb")

        if searched and q.strip():
            with st.spinner("Searching RCSB…"):
                results = search_rcsb(q.strip(), top_n=25)
            st.session_state[key_res] = results
            st.session_state[key_q]   = q.strip()
            st.session_state[key_sel] = 0

        results = st.session_state.get(key_res, [])
        if not results:
            return

        st.caption(f"{len(results)} results — sorted by resolution")
        _labels = []
        for r in results:
            res_s   = f"{r['resolution']:.2f} Å" if r.get("resolution") else "—"
            miss    = "" if r.get("no_missing_residues") else " ⚠"
            method  = f" [{r.get('method','')[:4]}]" if r.get("method") else ""
            _labels.append(
                f"{r['pdb_id']}  {res_s}{method}{miss}  {r.get('title','')[:50]}"
            )

        sel_idx = st.radio(
            "Select a structure",
            options=range(len(_labels)),
            format_func=lambda i: _labels[i],
            key=key_sel + "_radio",
            label_visibility="collapsed",
        )
        _picked = results[sel_idx]

        # Row: Use PDB button + View on RCSB link
        _btn_col, _link_col = st.columns([2, 1.2])
        with _btn_col:
            if st.button(
                "✓ Use selected PDB ID",
                key=prefix + "use_selected_pdb",
                type="primary",
                help=_h("pdb_id"),
            ):
                st.session_state[prefix + "pdb_id"]    = _picked["pdb_id"]
                st.session_state[prefix + "pdb_token"] = _picked["pdb_id"]
                st.rerun()
        with _link_col:
            _rcsb_url = f"https://www.rcsb.org/structure/{_picked['pdb_id']}"
            st.markdown(
                f'<a href="{_rcsb_url}" target="_blank" style="'
                f'display:inline-flex;align-items:center;gap:6px;'
                f'padding:7px 16px;border-radius:6px;'
                f'background:#F6F8FA;border:1px solid #D0D7DE;'
                f'color:#24292F;text-decoration:none;'
                f'font-size:0.85rem;white-space:nowrap;">'
                f'🔗 View on RCSB</a>',
                unsafe_allow_html=True,
            )
        # Info
        st.caption(
            f"Resolution: {_picked.get('resolution', '—')} Å  |  "
            f"Method: {_picked.get('method', '—')}  |  "
            f"Protein: {_picked.get('protein_name', '—')[:60]}"
        )


# ---------------------------------------------------------------------------
#  RECEPTOR SECTION
# ---------------------------------------------------------------------------

def _receptor_section(prefix: str = ""):
    st.subheader("Step 1 — Receptor")

    # Structure source
    src = st.radio(
        "Structure source",
        ["Download PDB", "Upload file"],
        key=prefix + "src_mode",
        horizontal=True,
        help=_h("src_mode"),
    )

    if src == "Download PDB":
        _rcsb_search_widget(prefix)
        c1, c2 = st.columns([2, 1])
        with c1:
            pdb_id = st.text_input(
                "PDB ID",
                value=st.session_state.get(prefix + "pdb_id", ""),
                placeholder="e.g. 1M17",
                key=prefix + "pdb_id",
                help=_h("pdb_id"),
            ).strip().upper()
        with c2:
            fmt = st.radio(
                "Format",
                ["CIF", "PDB"],
                key=prefix + "rcsb_fmt",
                horizontal=True,
                help=_h("rcsb_fmt"),
            )
    else:
        uploaded_rec = st.file_uploader(
            "Upload receptor (.pdb or .cif)",
            type=["pdb", "cif"],
            key=prefix + "rec_upload",
        )

    # Grid center
    center_mode = st.radio(
        "Grid center",
        ["Auto-detect co-crystal ligand", "Manual XYZ", "ProDy selection"],
        key=prefix + "center_mode",
        horizontal=True,
        help=_h("center_mode"),
    )

    if center_mode == "Manual XYZ":
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("X", value=st.session_state.get(prefix+"cx", 0.0),
                            key=prefix+"cx", step=0.5, format="%.2f")
        with c2:
            st.number_input("Y", value=st.session_state.get(prefix+"cy", 0.0),
                            key=prefix+"cy", step=0.5, format="%.2f")
        with c3:
            st.number_input("Z", value=st.session_state.get(prefix+"cz", 0.0),
                            key=prefix+"cz", step=0.5, format="%.2f")

    elif center_mode == "ProDy selection":
        st.text_input(
            "ProDy selection string",
            placeholder="e.g. resid 702 and chain A",
            key=prefix + "mda_sel",
            help=_h("mda_sel"),
        )

    # Box size
    with st.expander("🔲 Box size (Angstroms)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.slider("X", 10, 40, st.session_state.get(prefix+"sx", 16), 2,
                      key=prefix+"sx", help=_h("box_size"))
        with c2:
            st.slider("Y", 10, 40, st.session_state.get(prefix+"sy", 16), 2,
                      key=prefix+"sy", help=_h("box_size"))
        with c3:
            st.slider("Z", 10, 40, st.session_state.get(prefix+"sz", 16), 2,
                      key=prefix+"sz", help=_h("box_size"))
        sx = st.session_state.get(prefix+"sx", 16)
        sy = st.session_state.get(prefix+"sy", 16)
        sz = st.session_state.get(prefix+"sz", 16)
        st.caption(f"Volume ≈ {sx*sy*sz:,} ų")

    # Advanced options
    with st.expander("⚙️ Advanced receptor options", expanded=False):
        st.checkbox("Blind docking (expand box to whole protein)",
                    key=prefix+"blind_docking", help=_h("blind_docking"))
        st.checkbox("Keep cofactors (ATP, FAD, heme…)",
                    key=prefix+"keep_cofactors", help=_h("keep_cofactors"))
        st.checkbox("Keep metal ions (Zn, Mg, Ca…)",
                    key=prefix+"keep_metals", help=_h("keep_metals"))
        st.checkbox("Prefer structures with no missing residues",
                    key=prefix+"rcsb_prefer_complete", help=_h("rcsb_prefer_complete"))

    # Prepare button
    if st.button("▶ Prepare Receptor", key=prefix+"btn_prep_rec", type="primary"):
        with st.spinner("Preparing receptor…"):
            _do_prepare_receptor(prefix)

    # Status
    if st.session_state.get(prefix+"rec_prepared"):
        info = st.session_state.get(prefix+"rec_info", {})
        st.success(
            f"✓ Receptor ready  |  {info.get('n_atoms', '?')} atoms  |  "
            f"Center: ({info.get('cx',0):.2f}, {info.get('cy',0):.2f}, {info.get('cz',0):.2f})  |  "
            f"Box: {info.get('sx',16)}×{info.get('sy',16)}×{info.get('sz',16)} Å"
        )
        if info.get("cocrystal_ligand_id"):
            st.info(f"Co-crystal ligand detected: **{info['cocrystal_ligand_id']}**")


def _do_prepare_receptor(prefix: str = ""):
    """Run receptor preparation and store result in session state."""
    wdir = _wdir()
    info = {}
    try:
        src_mode = st.session_state.get(prefix+"src_mode", "Download PDB")
        if src_mode == "Download PDB":
            pdb_id = st.session_state.get(prefix+"pdb_id", "").strip().upper()
            if not pdb_id:
                st.error("Enter a PDB ID first.")
                return
            fmt    = st.session_state.get(prefix+"rcsb_fmt", "CIF")
            ext    = "cif" if fmt == "CIF" else "pdb"
            url    = (
                f"https://files.rcsb.org/download/{pdb_id}.{ext}"
                if fmt == "PDB"
                else f"https://files.rcsb.org/download/{pdb_id}.cif"
            )
            import requests as _req
            r = _req.get(url, timeout=30)
            if r.status_code != 200:
                st.error(f"Download failed (HTTP {r.status_code}). Try the other format.")
                return
            raw_path = str(wdir / f"{pdb_id}.{ext}")
            with open(raw_path, "wb") as f:
                f.write(r.content)
        else:
            uploaded = st.session_state.get(prefix+"rec_upload")
            if uploaded is None:
                st.error("Upload a PDB/CIF file first.")
                return
            raw_path = str(wdir / uploaded.name)
            with open(raw_path, "wb") as f:
                f.write(uploaded.read())

        center_mode = st.session_state.get(prefix+"center_mode", "Auto-detect co-crystal ligand")
        manual_xyz  = (
            st.session_state.get(prefix+"cx", 0.0),
            st.session_state.get(prefix+"cy", 0.0),
            st.session_state.get(prefix+"cz", 0.0),
        )
        prody_sel = st.session_state.get(prefix+"mda_sel", "")
        box_size  = (
            st.session_state.get(prefix+"sx", 16),
            st.session_state.get(prefix+"sy", 16),
            st.session_state.get(prefix+"sz", 16),
        )
        if st.session_state.get(prefix+"blind_docking", False):
            box_size = (60, 60, 60)

        _cm_map = {
            "Auto-detect co-crystal ligand": "auto",
            "Manual XYZ":                    "manual",
            "ProDy selection":               "selection",
        }
        result = prepare_receptor(
            raw_pdb    = raw_path,
            wdir       = wdir,
            center_mode = _cm_map.get(center_mode, "auto"),
            manual_xyz  = manual_xyz,
            prody_sel   = prody_sel,
            box_size    = box_size,
        )
        if not result["success"]:
            st.error(f"Receptor prep failed: {result.get('error','')}")
            for line in result.get("log", []):
                st.caption(line)
            return

        st.session_state[prefix+"rec_prepared"] = True
        st.session_state[prefix+"rec_info"]     = result
        st.session_state[prefix+"rec_pdbqt"]    = result["rec_pdbqt"]
        st.session_state[prefix+"rec_fh"]       = result["rec_fh"]
        st.session_state[prefix+"box_pdb"]      = result["box_pdb"]
        st.session_state[prefix+"config_txt"]   = result["config_txt"]
        st.session_state[prefix+"cx"]           = result["cx"]
        st.session_state[prefix+"cy"]           = result["cy"]
        st.session_state[prefix+"cz"]           = result["cz"]
        with st.expander("Receptor preparation log", expanded=False):
            for line in result.get("log", []):
                st.caption(line)
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
#  LIGAND SECTION
# ---------------------------------------------------------------------------

def _ligand_section(prefix: str = ""):
    st.subheader("Step 2 — Ligand")

    input_mode = st.radio(
        "Ligand input mode",
        ["PubChem / SMILES", "Upload structure file", "Draw (Ketcher)"],
        key=prefix+"lig_input_mode",
        horizontal=True,
        help=_h("lig_input_mode"),
    )

    if input_mode == "PubChem / SMILES":
        _pubchem_search_widget(prefix)
        st.text_input(
            "SMILES string",
            value=st.session_state.get(prefix+"smiles_in", ""),
            placeholder="e.g. CCO",
            key=prefix+"smiles_in",
            help=_h("smiles_in"),
        )

    elif input_mode == "Upload structure file":
        uploaded_lig = st.file_uploader(
            "Upload ligand (.sdf, .mol2, .pdb)",
            type=["sdf", "mol2", "pdb"],
            key=prefix+"lig_upload",
        )
        st.radio(
            "Protonation mode",
            ["Use the uploaded form", "Protonate at target pH"],
            key=prefix+"lig_upload_prot",
            horizontal=True,
            help=_h("lig_upload_prot"),
        )

    else:  # Draw
        st.info("Ketcher sketcher integration — paste SMILES below after drawing.")
        st.text_input(
            "SMILES from Ketcher",
            key=prefix+"smiles_in",
            help=_h("smiles_in"),
        )

    col_n, col_ph = st.columns(2)
    with col_n:
        st.text_input(
            "Output name",
            value=st.session_state.get(prefix+"lig_name_in", "ligand"),
            key=prefix+"lig_name_in",
            help=_h("lig_name_in"),
        )
    with col_ph:
        st.number_input(
            "Target pH",
            min_value=0.0, max_value=14.0,
            value=float(st.session_state.get(prefix+"ph_in", 7.4)),
            step=0.1,
            key=prefix+"ph_in",
            help=_h("ph_in"),
        )

    if st.button("▶ Prepare Ligand", key=prefix+"btn_prep_lig", type="primary"):
        with st.spinner("Preparing ligand…"):
            _do_prepare_ligand(prefix)

    if st.session_state.get(prefix+"lig_prepared"):
        info = st.session_state.get(prefix+"lig_info", {})
        st.success(
            f"✓ Ligand ready  |  Charge: {info.get('charge', '?'):+d}  |  "
            f"SMILES: {str(info.get('prot_smiles',''))[:50]}"
        )


def _do_prepare_ligand(prefix: str = ""):
    wdir       = _wdir()
    input_mode = st.session_state.get(prefix+"lig_input_mode", "PubChem / SMILES")
    name       = st.session_state.get(prefix+"lig_name_in", "ligand").strip() or "ligand"
    ph         = float(st.session_state.get(prefix+"ph_in", 7.4))

    try:
        if input_mode == "Upload structure file":
            uploaded = st.session_state.get(prefix+"lig_upload")
            if uploaded is None:
                st.error("Upload a structure file first.")
                return
            file_path = str(wdir / uploaded.name)
            with open(file_path, "wb") as f:
                f.write(uploaded.read())
            prot_mode = st.session_state.get(prefix+"lig_upload_prot", "Use the uploaded form")
            if "Protonate" in prot_mode:
                smiles = smiles_from_file(file_path, wdir)
                result = prepare_ligand(smiles, name, ph, wdir, mode="dimorphite")
            else:
                result = prepare_ligand_from_file(file_path, name, wdir)
        else:
            smiles = st.session_state.get(prefix+"smiles_in", "").strip()
            if not smiles:
                st.error("Enter a SMILES string first.")
                return
            result = prepare_ligand(smiles, name, ph, wdir, mode="dimorphite")

        if not result["success"]:
            st.error(f"Ligand prep failed: {result.get('error','')}")
            return

        st.session_state[prefix+"lig_prepared"] = True
        st.session_state[prefix+"lig_info"]     = result
        st.session_state[prefix+"lig_pdbqt"]    = result["pdbqt"]
        st.session_state[prefix+"lig_sdf"]      = result["sdf"]

        with st.expander("Ligand preparation log", expanded=False):
            for line in result.get("log", []):
                st.caption(line)
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
#  DOCKING SECTION
# ---------------------------------------------------------------------------

def _docking_section(prefix: str = ""):
    st.subheader("Step 3 — Docking")

    ready_rec = st.session_state.get(prefix+"rec_prepared", False)
    ready_lig = st.session_state.get(prefix+"lig_prepared", False)
    if not ready_rec:
        st.warning("Complete Step 1 (receptor preparation) first.")
        return
    if not ready_lig:
        st.warning("Complete Step 2 (ligand preparation) first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.slider(
            "Exhaustiveness", 4, 64,
            int(st.session_state.get(prefix+"exh_slider", 16)), 2,
            key=prefix+"exh_slider",
            help=_h("exh_slider"),
        )
    with col2:
        st.slider(
            "Number of poses", 5, 20,
            int(st.session_state.get(prefix+"n_modes", 10)), 1,
            key=prefix+"n_modes",
            help=_h("n_modes"),
        )
    with col3:
        st.slider(
            "Energy range (kcal/mol)", 1, 5,
            int(st.session_state.get(prefix+"e_range", 3)), 1,
            key=prefix+"e_range",
            help=_h("e_range"),
        )

    rec_info = st.session_state.get(prefix+"rec_info", {})
    lig_info = st.session_state.get(prefix+"lig_info", {})
    if rec_info.get("cocrystal_ligand_id"):
        st.checkbox(
            f"Also dock co-crystal ligand ({rec_info['cocrystal_ligand_id']}) for validation",
            key=prefix+"do_redock",
            help=_h("do_redock"),
        )
    else:
        co_smiles_raw = st.text_input(
            "Co-crystal ligand SMILES [name] (optional validation)",
            key=prefix+"redock_smiles",
            placeholder="e.g. c1ccc(cc1)NC(=O)c2ccccc2 Reference",
            help=_h("redock_smiles"),
        )

    if st.button("🚀 Run Docking", key=prefix+"btn_dock", type="primary"):
        with st.spinner("Running AutoDock Vina…"):
            _do_docking(prefix)


def _do_docking(prefix: str = ""):
    wdir = _wdir()
    vina_path, vina_msg = _cached_vina()
    if vina_path is None:
        st.error(f"Vina binary not available: {vina_msg}")
        return

    rec_pdbqt  = st.session_state.get(prefix+"rec_pdbqt", "")
    lig_pdbqt  = st.session_state.get(prefix+"lig_pdbqt", "")
    config_txt = st.session_state.get(prefix+"config_txt", "")
    name       = st.session_state.get(prefix+"lig_name_in", "ligand")

    if not all([rec_pdbqt, lig_pdbqt, config_txt]):
        st.error("Missing receptor or ligand files.")
        return

    exh    = int(st.session_state.get(prefix+"exh_slider", 16))
    n_modes = int(st.session_state.get(prefix+"n_modes", 10))
    e_range = int(st.session_state.get(prefix+"e_range", 3))

    result = run_vina(
        receptor_pdbqt = rec_pdbqt,
        ligand_pdbqt   = lig_pdbqt,
        config_txt     = config_txt,
        vina_path      = vina_path,
        exhaustiveness = exh,
        n_modes        = n_modes,
        energy_range   = e_range,
        wdir           = wdir,
        out_name       = name,
    )

    if not result["success"]:
        st.error(f"Docking failed: {result.get('error','')}")
        st.code(result.get("log", "")[:500])
        return

    # Bond order correction
    lig_info = st.session_state.get(prefix+"lig_info", {})
    prot_smi = lig_info.get("prot_smiles", "")
    out_sdf  = result["out_sdf"]
    if prot_smi and os.path.exists(out_sdf):
        fixed_sdf = out_sdf.replace(".sdf", "_fixed.sdf")
        fix_sdf_bond_orders(out_sdf, prot_smi, fixed_sdf)
        if os.path.exists(fixed_sdf) and os.path.getsize(fixed_sdf) > 10:
            result["out_sdf_fixed"] = fixed_sdf
        else:
            result["out_sdf_fixed"] = out_sdf
    else:
        result["out_sdf_fixed"] = out_sdf

    st.session_state[prefix+"docking_done"]   = True
    st.session_state[prefix+"docking_result"] = result
    st.session_state["adme_results"]          = {}  # clear stale ADME

    scores = result.get("scores", [])
    if scores:
        st.success(
            f"✓ Docking complete  |  "
            f"Best affinity: {scores[0]['affinity']:.2f} kcal/mol  |  "
            f"{len(scores)} poses"
        )


# ---------------------------------------------------------------------------
#  RESULTS SECTION
# ---------------------------------------------------------------------------

def _results_section(prefix: str = ""):
    st.subheader("Step 4 — Results")

    if not st.session_state.get(prefix+"docking_done", False):
        st.info("Run docking first.")
        return

    result = st.session_state.get(prefix+"docking_result", {})
    scores  = result.get("scores", [])
    out_sdf = result.get("out_sdf_fixed", result.get("out_sdf", ""))

    if not scores:
        st.warning("No docking poses found. Try increasing the box size or exhaustiveness.")
        return

    # Score table
    with st.expander("📊 Affinity scores", expanded=True):
        _rows = []
        for s in scores:
            _rows.append({
                "Pose": s["pose"],
                "Affinity (kcal/mol)": f"{s['affinity']:.2f}",
                "RMSD LB": f"{s['rmsd_lb']:.2f}",
                "RMSD UB": f"{s['rmsd_ub']:.2f}",
            })
        st.table(_rows)

    # Pose selector
    n_poses = len(scores)
    pose_idx = st.slider(
        "Select pose",
        min_value=1, max_value=n_poses,
        value=int(st.session_state.get(prefix+"pose_sel", 1)),
        key=prefix+"pose_sel",
        help=_h("pose_sel"),
    ) - 1  # 0-based index

    mols = load_mols_from_sdf(out_sdf)
    if not mols or pose_idx >= len(mols):
        st.warning("Could not load docked poses from SDF.")
        return

    sel_mol  = mols[pose_idx]
    sel_score = scores[min(pose_idx, len(scores)-1)]

    # Score badge
    aff   = sel_score["affinity"]
    aff_c = _affinity_color(aff)
    st.markdown(
        f'<div style="margin:8px 0;">'
        f'Pose {pose_idx+1} — Affinity: '
        f'<span class="score-badge" style="background:{aff_c};">{aff:.2f} kcal/mol</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 3D viewer + download columns
    col_3d, col_dl = st.columns([3, 1])

    with col_3d:
        _show_3d_viewer(sel_mol, prefix, pose_idx)

    with col_dl:
        st.markdown("**Downloads**")
        # Single pose SDF
        pose_sdf_path = str(_wdir() / f"pose_{pose_idx+1}.sdf")
        write_single_pose(sel_mol, pose_sdf_path)
        with open(pose_sdf_path, "rb") as f:
            st.download_button(
                "⬇ Pose SDF",
                data=f.read(),
                file_name=f"pose_{pose_idx+1}.sdf",
                mime="chemical/x-mdl-sdfile",
                key=f"dl_sdf_{pose_idx}",
            )
        # All poses SDF
        with open(out_sdf, "rb") as f:
            st.download_button(
                "⬇ All poses SDF",
                data=f.read(),
                file_name=f"all_poses.sdf",
                mime="chemical/x-mdl-sdfile",
                key=f"dl_all_sdf",
            )

        # Save PNG (Capture from py3Dmol)
        st.markdown("---")
        st.markdown("**Snapshot**")
        _png_key = f"pose_png_{pose_idx}"
        if st.button(
            "📷 Capture view",
            key=f"btn_capture_{pose_idx}",
            help="Capture current 3D viewport as PNG",
        ):
            _png = _capture_3dmol_png(sel_mol, prefix)
            if _png and len(_png) > 100:
                st.session_state[_png_key] = _png
            else:
                st.warning("Could not capture PNG from viewer.")

        _saved_png = st.session_state.get(_png_key)
        if _saved_png:
            st.download_button(
                "⬇ PNG",
                data=_saved_png,
                file_name=f"pose_{pose_idx+1}.png",
                mime="image/png",
                key=f"dl_png_{pose_idx}",
            )

    # 2D interaction diagram
    _results_2d_section(sel_mol, prefix, pose_idx)


def _capture_3dmol_png(mol, prefix: str = "") -> bytes:
    """Render a static PNG of the pose using py3Dmol."""
    try:
        import py3Dmol
        from rdkit import Chem
        view = py3Dmol.view(width=600, height=400)
        sdf_str = Chem.MolToMolBlock(mol)
        view.addModel(sdf_str, "sdf")
        view.setStyle({}, {"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}})
        # Receptor
        rec_fh = st.session_state.get(prefix+"rec_fh", "")
        if rec_fh and os.path.exists(rec_fh):
            with open(rec_fh) as f:
                view.addModel(f.read(), "pdb")
            view.setStyle({"model": -1}, {"cartoon": {"color": "spectrum", "opacity": 0.85}})
        view.zoomTo()
        return view.png()
    except Exception:
        return b""


def _show_3d_viewer(mol, prefix: str = "", pose_idx: int = 0):
    """Render the docked pose in the py3Dmol viewer."""
    try:
        import py3Dmol
        from rdkit import Chem
        from stmol import showmol

        view = py3Dmol.view(width=560, height=420)
        sdf_str = Chem.MolToMolBlock(mol)
        view.addModel(sdf_str, "sdf")
        view.setStyle({}, {"stick": {"radius": 0.18, "colorscheme": "cyanCarbon"},
                           "sphere": {"scale": 0.22}})

        rec_fh    = st.session_state.get(prefix+"rec_fh", "")
        bp_cutoff = float(st.session_state.get(prefix+"bp_cutoff", 3.5))
        show_lbl  = st.session_state.get(prefix+"bp_show_labels", True)
        show_surf = st.session_state.get(prefix+"bp_show_surface", False)

        if rec_fh and os.path.exists(rec_fh):
            with open(rec_fh) as f:
                pdb_str = f.read()
            view.addModel(pdb_str, "pdb")
            view.setStyle({"model": -1}, {"cartoon": {"color": "spectrum", "opacity": 0.8}})

            # Interacting residues
            interacting = get_interacting_residues(rec_fh, mol, cutoff=bp_cutoff)
            for res in interacting:
                sel = {"resi": res["resi"], "chain": res["chain"]}
                view.addStyle(sel, {"stick": {"colorscheme": "orangeCarbon", "radius": 0.15}})
                if show_lbl:
                    view.addLabel(
                        f"{res['resn']}{res['resi']}{res['chain']}",
                        {"position": sel, "fontSize": 9,
                         "backgroundColor": "rgba(255,165,0,0.6)",
                         "fontColor": "black", "borderRadius": 4},
                    )
            if show_surf:
                view.addSurface("SES", {"opacity": 0.5, "colorscheme": "whiteCarbon"})

        view.zoomTo()

        # Viewer controls in a compact row
        c1, c2, c3 = st.columns(3)
        with c1:
            st.slider(
                "Distance cutoff (Å)", 2.5, 6.0,
                float(st.session_state.get(prefix+"bp_cutoff", 3.5)), 0.5,
                key=prefix+"bp_cutoff", help=_h("bp_cutoff"),
            )
        with c2:
            st.checkbox("Labels", key=prefix+"bp_show_labels", help=_h("bp_show_labels"))
        with c3:
            st.checkbox("Surface", key=prefix+"bp_show_surface", help=_h("bp_show_surface"))

        showmol(view, height=420, width=560)

    except ImportError:
        st.info("Install stmol and py3Dmol for 3D visualization: pip install stmol py3Dmol")
    except Exception as exc:
        st.warning(f"3D viewer error: {exc}")


def _results_2d_section(mol, prefix: str = "", pose_idx: int = 0):
    """Draw the 2D interaction diagram for the selected pose."""
    st.markdown("#### 2D Interaction Diagram")
    c1, c2 = st.columns(2)
    with c1:
        cutoff = st.slider(
            "Diagram cutoff (Å)", 3.0, 6.0,
            float(st.session_state.get(prefix+"diag_cutoff", 4.5)), 0.5,
            key=prefix+"diag_cutoff", help=_h("diag_cutoff"),
        )
    with c2:
        max_res = st.slider(
            "Max residues", 4, 24,
            int(st.session_state.get(prefix+"diag_max_res", 14)), 1,
            key=prefix+"diag_max_res", help=_h("diag_max_res"),
        )

    rec_fh   = st.session_state.get(prefix+"rec_fh", "")
    lig_info = st.session_state.get(prefix+"lig_info", {})
    smiles   = lig_info.get("prot_smiles", "")
    name     = st.session_state.get(prefix+"lig_name_in", "Ligand")

    if not rec_fh or not os.path.exists(rec_fh):
        st.info("Receptor file not found — diagram unavailable.")
        return

    if st.button(
        "Draw 2D diagram",
        key=f"btn_diag_{pose_idx}",
        help="Generate 2D ligand–protein interaction diagram",
    ):
        with st.spinner("Generating 2D diagram…"):
            from rdkit import Chem
            pose_sdf = str(_wdir() / f"pose_{pose_idx+1}_diag.sdf")
            write_single_pose(mol, pose_sdf)
            svg_bytes = draw_interaction_diagram(
                receptor_pdb = rec_fh,
                pose_sdf     = pose_sdf,
                smiles       = smiles,
                title        = f"{name} — pose {pose_idx+1}",
                cutoff       = cutoff,
                max_residues = max_res,
            )
        st.image(svg_bytes, use_column_width=True)
        st.download_button(
            "⬇ Download SVG",
            data=svg_bytes,
            file_name=f"diagram_pose_{pose_idx+1}.svg",
            mime="image/svg+xml",
            key=f"dl_svg_{pose_idx}",
        )


# ---------------------------------------------------------------------------
#  ADME SECTION  (ML + rule-based fallback)
# ---------------------------------------------------------------------------

def _adme_section(prefix: str = ""):
    st.subheader("Step 5 — ADMET Properties")

    lig_info = st.session_state.get(prefix+"lig_info", {})
    smiles   = lig_info.get("prot_smiles", "") or st.session_state.get(prefix+"smiles_in", "")

    if not smiles:
        st.info("Prepare a ligand first to calculate ADMET properties.")
        return

    if st.button("⚗️ Calculate ADMET", key=prefix+"btn_adme", type="primary"):
        with st.spinner("Computing ADMET properties…"):
            # Prime the ADMET-AI model cache via the Streamlit resource cache
            _cached_admet_model()
            props = calc_adme_properties(smiles)
        st.session_state["adme_results"] = props

    props = st.session_state.get("adme_results", {})
    if not props or props.get("error"):
        if props.get("error"):
            st.error(f"ADMET error: {props['error']}")
        return

    # ------------------------------------------------------------------ #
    #  ML source banner
    # ------------------------------------------------------------------ #
    ml_avail = props.get("ml_available", False)
    ml_src   = props.get("ml_source", "")
    ml_err   = props.get("ml_error")

    if ml_avail:
        st.markdown(
            f'<div class="ml-badge">🤖 {ml_src}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="ml-badge-fallback">⚠️ ADMET-AI unavailable — showing rule-based estimates'
            + (f'  |  {ml_err}' if ml_err else "  |  pip install admet-ai")
            + "</div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------ #
    #  Overview metrics (4 cards)
    # ------------------------------------------------------------------ #
    st.markdown("#### Overview")
    ov1, ov2, ov3, ov4 = st.columns(4)
    with ov1:
        st.metric("QED", f"{props.get('qed',0):.3f}", help=_h("adme_qed"))
    with ov2:
        if ml_avail and props.get("bioavailability_ml") is not None:
            val_f = props["ml_results"].get("bioavailability_ml", props.get("bioavailability_ml", None))
            if val_f is not None:
                st.metric("Bioavailability (ML)", f"{float(val_f):.2f}", help=_h("adme_gi"))
            else:
                st.metric("GI Absorption", props.get("gi_absorption", "—"), help=_h("adme_gi"))
        else:
            st.metric("GI Absorption", props.get("gi_absorption", "—"), help=_h("adme_gi"))
    with ov3:
        if ml_avail and props.get("ml_results", {}).get("lipophilicity_ml") is not None:
            st.metric("Lipophilicity (ML)", f"{props['ml_results']['lipophilicity_ml']:.2f}",
                      help=_h("adme_logp"))
        else:
            st.metric("LogP", f"{props.get('logp',0):.2f}", help=_h("adme_logp"))
    with ov4:
        if ml_avail and props.get("ml_results", {}).get("solubility") is not None:
            st.metric("Solubility (ML)", f"{props['ml_results']['solubility']:.2f} log mol/L",
                      help="ML-predicted aqueous solubility (log mol/L)")
        else:
            st.metric("TPSA", f"{props.get('tpsa',0):.1f} Å²", help=_h("adme_tpsa"))

    # ------------------------------------------------------------------ #
    #  Physicochemical descriptors
    # ------------------------------------------------------------------ #
    with st.expander("🧪 Physicochemical descriptors", expanded=True):
        d1, d2, d3, d4, d5, d6 = st.columns(6)
        d1.metric("MW (Da)",   f"{props.get('mw',0):.1f}",      help=_h("adme_mw"))
        d2.metric("LogP",      f"{props.get('logp',0):.2f}",    help=_h("adme_logp"))
        d3.metric("TPSA (Å²)", f"{props.get('tpsa',0):.1f}",   help=_h("adme_tpsa"))
        d4.metric("HBD",       str(props.get("hbd", 0)),        help=_h("adme_hbd"))
        d5.metric("HBA",       str(props.get("hba", 0)),        help=_h("adme_hba"))
        d6.metric("Fsp³",      f"{props.get('fsp3',0):.3f}",   help=_h("adme_fsp3"))

        st.markdown("---")
        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(
            f"Lipinski RO5 {_rule_badge(props.get('lipinski_pass', False))}  "
            f"({props.get('lipinski_violations', 0)} violation(s))",
            unsafe_allow_html=True,
        )
        r2.markdown(
            f"Veber  {_rule_badge(props.get('veber_pass', False))}",
            unsafe_allow_html=True,
        )
        r3.markdown(
            f"Egan   {_rule_badge(props.get('egan_pass', False))}",
            unsafe_allow_html=True,
        )
        r4.markdown(
            f"Muegge {_rule_badge(props.get('muegge_pass', False))}",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------ #
    #  Absorption & Distribution
    # ------------------------------------------------------------------ #
    with st.expander("💊 Absorption & Distribution", expanded=True):
        ab1, ab2, ab3, ab4, ab5, ab6 = st.columns(6)
        # Caco-2
        if ml_avail and props.get("caco2") is not None:
            ab1.metric("Caco-2 (log cm/s)", f"{props['caco2']:.3f}",
                       help="ML-predicted Caco-2 permeability (Wang dataset)")
        else:
            ab1.metric("GI Absorption", props.get("gi_absorption", "—"), help=_h("adme_gi"))

        # HIA
        if ml_avail and props.get("hia") is not None:
            ab2.metric("HIA", f"{props['hia']:.2f}",
                       help="ML-predicted Human Intestinal Absorption (Hou dataset)")
        else:
            ab2.metric("TPSA (Å²)", f"{props.get('tpsa',0):.1f}", help=_h("adme_tpsa"))

        # P-gp
        ab3.markdown(
            f"**P-gp substrate** {_rule_badge(props.get('pgp_substrate','Unlikely')=='Likely', 'Likely', 'Unlikely')}",
            unsafe_allow_html=True,
        )

        # BBB
        bbb_val = props.get("bbb", "—")
        ab4.metric("BBB", bbb_val, help=_h("adme_bbb"))
        if ml_avail and props.get("bbb_prob") is not None:
            ab4.caption(f"p = {props['bbb_prob']:.3f}")

        # PPB
        if ml_avail and props.get("ppb") is not None:
            ppb_v = props["ppb"]
            ppb_c = "#CF222E" if ppb_v > 90 else "#9A6700" if ppb_v > 70 else "#1A7F37"
            ab5.markdown(
                f'<div style="border:1.5px solid {ppb_c};border-radius:8px;padding:8px;text-align:center;">'
                f'<div style="font-size:11px;color:#666;font-weight:600;">PPB (%)</div>'
                f'<div style="font-size:1.3rem;font-weight:700;color:{ppb_c};">{ppb_v:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            ab5.metric("PPB", "—", help="Plasma Protein Binding (requires ADMET-AI)")

        # VDd
        if ml_avail and props.get("vdd") is not None:
            ab6.metric("VDd (L/kg)", f"{props['vdd']:.2f}",
                       help="ML-predicted Volume of Distribution (Lombardo dataset)")
        else:
            ab6.metric("VDd", "—", help="Volume of Distribution (requires ADMET-AI)")

    # ------------------------------------------------------------------ #
    #  CYP inhibition (probability bars)
    # ------------------------------------------------------------------ #
    with st.expander("⚗️ CYP450 Inhibition", expanded=True):
        cyp_flags = props.get("cyp_flags", {})
        cyp_probs = props.get("cyp_probs", {})
        cyp_cols  = st.columns(5)
        for i, cyp in enumerate(["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]):
            flag = cyp_flags.get(cyp, False)
            prob = cyp_probs.get(cyp)
            bg   = "#FFEBE9" if flag else "#DAFBE1"
            clr  = "#CF222E" if flag else "#1A7F37"
            ico  = "⚠ Inhibitor"  if flag else "✓ Safe"
            with cyp_cols[i]:
                st.markdown(
                    f'<div style="background:{bg};border:1.5px solid {clr};'
                    f'border-radius:8px;padding:8px;text-align:center;">'
                    f'<div style="font-size:11px;font-weight:600;color:{clr};">{cyp}</div>'
                    f'<div style="font-size:14px;font-weight:700;color:{clr};">{ico}</div>'
                    + (
                        f'<div class="cyp-bar-wrap">'
                        f'<div class="cyp-bar-fill" style="width:{int(prob*100)}%;"></div>'
                        f'</div><div style="font-size:10px;color:#888;">p={prob:.2f}</div>'
                        if prob is not None else ""
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )

        # CYP substrate
        cyp_sub   = props.get("cyp_substrate", {})
        cyp_sub_p = props.get("cyp_substrate_probs", {})
        if cyp_sub and ml_avail:
            st.markdown("**CYP Substrate predictions (ML)**")
            sub_cols = st.columns(3)
            for j, cyp_s in enumerate(["CYP2C9", "CYP2D6", "CYP3A4"]):
                v    = cyp_sub.get(cyp_s)
                prob = cyp_sub_p.get(cyp_s)
                bg   = "#FFF8C5" if v else "#DAFBE1"
                clr  = "#9A6700" if v else "#1A7F37"
                lbl  = "⚠ Substrate" if v else "✓ Non-substrate"
                with sub_cols[j]:
                    st.markdown(
                        f'<div style="background:{bg};border:1.5px solid {clr};'
                        f'border-radius:8px;padding:8px;text-align:center;">'
                        f'<div style="font-size:11px;font-weight:600;color:{clr};">{cyp_s}</div>'
                        f'<div style="font-size:13px;font-weight:700;color:{clr};">{lbl}</div>'
                        + (f'<div style="font-size:10px;color:#888;">p={prob:.2f}</div>'
                           if prob is not None else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )

    # ------------------------------------------------------------------ #
    #  Excretion
    # ------------------------------------------------------------------ #
    if ml_avail and (props.get("half_life") is not None or props.get("clearance") is not None):
        with st.expander("🔄 Excretion (ML)", expanded=False):
            ex1, ex2 = st.columns(2)
            if props.get("half_life") is not None:
                ex1.metric("Half-life (hr)", f"{props['half_life']:.2f}",
                           help="ML-predicted half-life (Obach dataset)")
            if props.get("clearance") is not None:
                ex2.metric("Hepatic Clearance (mL/min/g)", f"{props['clearance']:.2f}",
                           help="ML-predicted hepatic clearance (AZ dataset)")

    # ------------------------------------------------------------------ #
    #  Toxicity (ML only)
    # ------------------------------------------------------------------ #
    if ml_avail and props.get("herg") is not None:
        with st.expander("☠️ Toxicity Predictions (ML)", expanded=True):
            tox_cols = st.columns(4)
            for col, (lbl, flag_key, prob_key, tip) in zip(tox_cols, [
                ("hERG inhibition", "herg", "herg_prob",
                 "Cardiac toxicity risk — QT prolongation → arrhythmia"),
                ("AMES mutagenicity", "ames", "ames_prob",
                 "Genotoxicity risk — bacterial reverse mutation assay"),
                ("DILI (liver)", "dili", "dili_prob",
                 "Drug-induced liver injury"),
                ("Skin reaction", "skin_reaction", "skin_prob",
                 "Skin sensitization"),
            ]):
                flag = props.get(flag_key, False)
                prob = props.get(prob_key)
                bg   = "#FFEBE9" if flag else "#DAFBE1"
                clr  = "#CF222E" if flag else "#1A7F37"
                ico  = "⚠ Positive" if flag else "✓ Negative"
                with col:
                    st.markdown(
                        f'<div style="background:{bg};border:1.5px solid {clr};'
                        f'border-radius:8px;padding:10px;text-align:center;" title="{tip}">'
                        f'<div style="font-size:11px;font-weight:600;color:{clr};">{lbl}</div>'
                        f'<div style="font-size:15px;font-weight:700;color:{clr};">{ico}</div>'
                        + (f'<div style="font-size:10px;color:#888;">p = {prob:.3f}</div>'
                           if prob is not None else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )

            # LD50
            if props.get("ld50") is not None:
                st.metric(
                    "LD50 (oral, mg/kg)", f"{props['ld50']:.1f}",
                    help="ML-predicted oral lethal dose in 50% of rats (Zhu dataset)",
                )

    # ------------------------------------------------------------------ #
    #  Structural alerts
    # ------------------------------------------------------------------ #
    with st.expander("🚨 Structural Alerts", expanded=False):
        pains = props.get("pains_alerts", [])
        brenk = props.get("brenk_alerts", [])
        c_pa, c_br = st.columns(2)
        with c_pa:
            st.markdown(f"**PAINS** — {len(pains)} alert(s)", help=_h("adme_pains"))
            if pains:
                for a in pains:
                    st.warning(a)
            else:
                st.success("No PAINS alerts")
        with c_br:
            st.markdown(f"**BRENK** — {len(brenk)} alert(s)", help=_h("adme_brenk"))
            if brenk:
                for a in brenk[:8]:
                    st.warning(a)
                if len(brenk) > 8:
                    st.caption(f"… and {len(brenk)-8} more")
            else:
                st.success("No BRENK alerts")

    # ------------------------------------------------------------------ #
    #  Summary table + copy/export
    # ------------------------------------------------------------------ #
    with st.expander("📋 Summary table & export", expanded=False):
        _rows = [
            ("MW (Da)",        f"{props.get('mw',0):.2f}"),
            ("LogP",           f"{props.get('logp',0):.2f}"),
            ("TPSA (Å²)",     f"{props.get('tpsa',0):.1f}"),
            ("HBD",            str(props.get("hbd",0))),
            ("HBA",            str(props.get("hba",0))),
            ("Fsp³",           f"{props.get('fsp3',0):.3f}"),
            ("QED",            f"{props.get('qed',0):.3f}"),
            ("Lipinski",       "Pass" if props.get("lipinski_pass") else "Fail"),
            ("GI absorption",  props.get("gi_absorption","—")),
            ("BBB",            props.get("bbb","—")),
            ("P-gp substrate", props.get("pgp_substrate","—")),
            ("PAINS alerts",   str(len(props.get("pains_alerts",[])))),
            ("BRENK alerts",   str(len(props.get("brenk_alerts",[])))),
        ]
        if ml_avail:
            for k, label in [
                ("herg",         "hERG"),
                ("ames",         "AMES"),
                ("dili",         "DILI"),
                ("skin_reaction","Skin reaction"),
                ("half_life",    "Half-life (hr)"),
                ("clearance",    "Clearance (mL/min/g)"),
                ("ld50",         "LD50 (mg/kg)"),
                ("ppb",          "PPB (%)"),
            ]:
                v = props.get(k)
                if v is not None:
                    _rows.append((label, str(v) if not isinstance(v, bool) else ("Positive" if v else "Negative")))

        import pandas as pd
        df = pd.DataFrame(_rows, columns=["Property", "Value"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Download CSV
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", data=csv_bytes,
                           file_name="admet_summary.csv", mime="text/csv",
                           key="dl_adme_csv")

        # Report paragraph
        if st.button("📝 Write report paragraph ↗", key="btn_adme_report"):
            _lines = [
                f"The compound has a molecular weight of {props.get('mw',0):.1f} Da, "
                f"a calculated LogP of {props.get('logp',0):.2f}, "
                f"TPSA of {props.get('tpsa',0):.1f} Å², "
                f"with {props.get('hbd',0)} hydrogen bond donors and {props.get('hba',0)} acceptors. "
                f"QED = {props.get('qed',0):.3f}. "
                f"Lipinski RO5 {'passed' if props.get('lipinski_pass') else 'violated'}. "
                f"GI absorption: {props.get('gi_absorption','—')}. "
                f"BBB: {props.get('bbb','—')}. "
                f"P-gp substrate: {props.get('pgp_substrate','—')}."
            ]
            if ml_avail:
                if props.get("herg") is not None:
                    _lines.append(
                        f"hERG inhibition predicted {'positive' if props['herg'] else 'negative'} "
                        f"(p={props.get('herg_prob',0):.3f}). "
                        f"AMES {'positive' if props.get('ames') else 'negative'}. "
                        f"DILI risk {'positive' if props.get('dili') else 'negative'}."
                    )
            report_text = " ".join(_lines)
            st.text_area("Report paragraph", value=report_text, height=100,
                         key="adme_report_text")

        # Disclaimer
        st.caption(
            "⚠️ ADMET predictions are computational estimates for early-stage "
            "drug discovery guidance only. Always validate with experimental assays "
            "before making lead selection decisions."
        )


# ---------------------------------------------------------------------------
#  BATCH DOCKING SECTION
# ---------------------------------------------------------------------------

def _batch_section(prefix: str = ""):
    st.subheader("Batch Docking")

    if not st.session_state.get(prefix+"rec_prepared", False):
        st.warning("Prepare a receptor (Step 1) before running batch docking.")
        return

    st.radio(
        "Ligand input mode (batch)",
        ["SMILES list (text area)", "Upload .smi file"],
        key=prefix+"b_input_mode",
        horizontal=True,
        help=_h("b_input_mode"),
    )

    if st.session_state.get(prefix+"b_input_mode") == "SMILES list (text area)":
        st.text_area(
            "SMILES list (one per line: SMILES [name])",
            height=160,
            key=prefix+"b_smiles_text",
            placeholder="CCO Ethanol\nc1ccccc1 Benzene",
            help=_h("b_smiles_text"),
        )
    else:
        st.file_uploader("Upload .smi file", type=["smi", "txt"],
                         key=prefix+"b_smi_upload")

    bc1, bc2 = st.columns(2)
    with bc1:
        st.slider(
            "Exhaustiveness (batch)", 4, 32,
            int(st.session_state.get(prefix+"b_exh", 8)), 2,
            key=prefix+"b_exh", help=_h("b_exh"),
        )
    with bc2:
        st.slider(
            "Poses per ligand", 3, 15,
            int(st.session_state.get(prefix+"b_nm", 9)), 1,
            key=prefix+"b_nm", help=_h("b_nm"),
        )

    st.checkbox(
        "Dock co-crystal ligand for validation",
        key=prefix+"b_do_redock",
        help=_h("b_do_redock"),
    )

    if st.button("🚀 Run Batch Docking", key=prefix+"btn_batch", type="primary"):
        st.info("Batch docking — processing queue started. Results will appear below.")
        _do_batch_docking(prefix)


def _do_batch_docking(prefix: str = ""):
    """Run batch docking for all ligands in the list."""
    wdir       = _wdir()
    vina_path, vina_msg = _cached_vina()
    if vina_path is None:
        st.error(f"Vina unavailable: {vina_msg}")
        return

    input_mode = st.session_state.get(prefix+"b_input_mode", "SMILES list (text area)")
    exh        = int(st.session_state.get(prefix+"b_exh", 8))
    n_modes    = int(st.session_state.get(prefix+"b_nm", 9))
    ph         = float(st.session_state.get(prefix+"ph_in", 7.4))

    # Parse ligand list
    pairs = []
    if input_mode == "SMILES list (text area)":
        text = st.session_state.get(prefix+"b_smiles_text", "")
        for i, line in enumerate(text.strip().splitlines()):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            smi   = parts[0]
            name  = parts[1].strip() if len(parts) > 1 else f"lig_{i+1}"
            pairs.append((smi, name))
    else:
        uploaded = st.session_state.get(prefix+"b_smi_upload")
        if uploaded is None:
            st.error("Upload a .smi file first.")
            return
        for i, line in enumerate(uploaded.getvalue().decode().splitlines()):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            smi   = parts[0]
            name  = parts[1].strip() if len(parts) > 1 else f"lig_{i+1}"
            pairs.append((smi, name))

    if not pairs:
        st.warning("No valid SMILES found in the input.")
        return

    rec_pdbqt  = st.session_state.get(prefix+"rec_pdbqt", "")
    config_txt = st.session_state.get(prefix+"config_txt", "")
    results    = []
    progress   = st.progress(0.0, text=f"Docking 0/{len(pairs)}")

    for idx, (smi, name) in enumerate(pairs):
        lig_res = prepare_ligand(smi, name, ph, wdir, mode="dimorphite")
        if not lig_res["success"]:
            results.append({"name": name, "smiles": smi, "top_score": None,
                            "error": lig_res.get("error","prep failed")})
            continue
        dock_res = run_vina(
            receptor_pdbqt = rec_pdbqt,
            ligand_pdbqt   = lig_res["pdbqt"],
            config_txt     = config_txt,
            vina_path      = vina_path,
            exhaustiveness = exh,
            n_modes        = n_modes,
            energy_range   = 3,
            wdir           = wdir,
            out_name       = name,
        )
        top = dock_res.get("scores", [{}])[0].get("affinity") if dock_res["success"] else None
        results.append({"name": name, "smiles": smi, "top_score": top,
                        "error": dock_res.get("error")})
        progress.progress((idx+1)/len(pairs), text=f"Docking {idx+1}/{len(pairs)}: {name}")

    progress.empty()
    st.session_state[prefix+"batch_results"] = results

    import pandas as pd
    df = pd.DataFrame(results).sort_values("top_score")
    st.dataframe(df, use_container_width=True)
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("⬇ Download batch CSV", data=csv_bytes,
                       file_name="batch_results.csv", mime="text/csv",
                       key="dl_batch_csv")


# ---------------------------------------------------------------------------
#  FIGURE EXPORT SECTION
# ---------------------------------------------------------------------------

def _figure_section(prefix: str = ""):
    st.subheader("Publication Figure")

    if not st.session_state.get(prefix+"docking_done", False):
        st.info("Run docking first to generate a figure.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.radio(
            "2D diagram source",
            ["RDKit (local)", "PoseView v1 (API)", "PoseView2 reference"],
            key=prefix+"rtf_src",
            help=_h("rtf_src"),
        )
        st.radio(
            "Figure layout",
            ["2-panel (3D + 2D)", "4-panel (3D + 2D + ADME + scores)"],
            key=prefix+"rtf_layout",
            help=_h("rtf_layout"),
        )
    with c2:
        st.slider(
            "Pocket cutoff (Å)", 3.0, 6.0,
            float(st.session_state.get(prefix+"rtf_cutoff", 4.0)), 0.5,
            key=prefix+"rtf_cutoff",
            help=_h("rtf_cutoff"),
        )
        st.checkbox("Residue labels", key=prefix+"rtf_labels", help=_h("rtf_labels"))
        st.checkbox("Protein surface", key=prefix+"rtf_surf",  help=_h("rtf_surf"))

    if st.button("🖼 Generate Figure", key=prefix+"btn_figure", type="primary"):
        with st.spinner("Generating figure…"):
            _generate_figure(prefix)


def _generate_figure(prefix: str = ""):
    """Generate a single-pose publication figure with 2D diagram."""
    result   = st.session_state.get(prefix+"docking_result", {})
    out_sdf  = result.get("out_sdf_fixed", result.get("out_sdf", ""))
    rec_fh   = st.session_state.get(prefix+"rec_fh", "")
    lig_info = st.session_state.get(prefix+"lig_info", {})
    smiles   = lig_info.get("prot_smiles", "")
    name     = st.session_state.get(prefix+"lig_name_in", "Ligand")
    src      = st.session_state.get(prefix+"rtf_src", "RDKit (local)")
    cutoff   = float(st.session_state.get(prefix+"rtf_cutoff", 4.0))

    if not out_sdf or not os.path.exists(out_sdf):
        st.error("SDF file not found. Re-run docking.")
        return

    mols = load_mols_from_sdf(out_sdf)
    if not mols:
        st.error("No poses in SDF.")
        return

    sel_mol  = mols[0]
    pose_sdf = str(_wdir() / "fig_pose.sdf")
    write_single_pose(sel_mol, pose_sdf)

    # 2D diagram
    if "PoseView v1" in src:
        svg_bytes, err = call_poseview_v1(rec_fh, pose_sdf)
        if err:
            st.warning(f"PoseView v1 failed: {err}. Falling back to RDKit.")
            svg_bytes = None
    elif "PoseView2" in src:
        pdb_id  = st.session_state.get(prefix+"pdb_id", "")
        coc_id  = st.session_state.get(prefix+"rec_info", {}).get("cocrystal_ligand_id", "")
        if pdb_id and coc_id:
            svg_bytes, err = call_poseview2_ref(pdb_id, coc_id)
            if err:
                st.warning(f"PoseView2 failed: {err}. Falling back to RDKit.")
                svg_bytes = None
        else:
            st.warning("PoseView2 needs PDB ID + co-crystal ligand ID.")
            svg_bytes = None
    else:
        svg_bytes = None

    if svg_bytes is None:
        svg_bytes = draw_interaction_diagram(
            receptor_pdb = rec_fh,
            pose_sdf     = pose_sdf,
            smiles       = smiles,
            title        = f"{name} — pose 1",
            cutoff       = cutoff,
            max_residues = 14,
        )

    st.image(svg_bytes, use_column_width=True)

    # Downloads
    dc1, dc2 = st.columns(2)
    with dc1:
        st.download_button(
            "⬇ Download SVG",
            data=svg_bytes,
            file_name=f"{name}_figure.svg",
            mime="image/svg+xml",
            key="dl_fig_svg",
        )
    with dc2:
        png_bytes = svg_to_png(svg_bytes)
        if png_bytes:
            st.download_button(
                "⬇ Download PNG",
                data=png_bytes,
                file_name=f"{name}_figure.png",
                mime="image/png",
                key="dl_fig_png",
            )


# ---------------------------------------------------------------------------
#  MAIN APP LAYOUT
# ---------------------------------------------------------------------------

def main():
    st.title("🔬 Anyone Can Dock")
    st.caption("AutoDock Vina · ADMET-AI · PoseView · ProDy")

    # Check dependencies
    ob_ok, ob_msg = check_obabel()
    if not ob_ok:
        st.error(f"OpenBabel not found: {ob_msg}")
        st.stop()

    tabs = st.tabs([
        "🧬 Receptor",
        "💊 Ligand",
        "⚡ Docking",
        "📊 Results",
        "🧪 ADMET",
        "📦 Batch",
        "🖼 Figure",
    ])

    with tabs[0]:
        _receptor_section()

    with tabs[1]:
        _ligand_section()

    with tabs[2]:
        _docking_section()

    with tabs[3]:
        _results_section()

    with tabs[4]:
        _adme_section()

    with tabs[5]:
        _batch_section()

    with tabs[6]:
        _figure_section()


if __name__ == "__main__":
    main()
