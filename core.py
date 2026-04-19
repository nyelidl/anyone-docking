#!/usr/bin/env python3
"""
core.py — Pure computation layer for Anyone Can Dock.
No Streamlit imports. All functions return plain dicts / tuples.
Safe to import in Colab notebooks, pytest, or any UI framework.

Fixes vs previous version:
  - load_mols_from_sdf: suppress RDKit kekulize noise, fallback sanitize loop
  - fix_sdf_bond_orders: AddHs with addCoords=True preserves docked pose coords
  - convert_sdf_to_v2000: removed --gen3d flag (was destroying docked pose)
  - call_poseview_v1: posts receptor PDB *directly* to /api/v2/poseview/ —
    old MoleculeHandler/Protoss round-trip returned re-protonated coords that
    mismatched the ligand, causing "failure". Direct POST guarantees consistency.
    Retries up to 3x, strips explicit Hs from receptor, polls up to 120 s.
  - warm_poseview_cache: now a no-op (MoleculeHandler pre-upload removed)
  - call_poseview2_ref: retry + full-response logging
  - Heme: _AROM_ATOMS / _AROM_ATOM_NAMES extended for porphyrin π-detection
  - Heme: hydrophobic detection extended to heme residue names
  - Minor: type annotations, cleaner log messages
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

METAL_RESNAMES = {
    "MG", "ZN", "CA", "MN", "FE", "CU", "CO", "NI", "CD", "HG", "NA", "K",
}
METAL_CHARGES = {
    "MG": 2.0, "ZN": 2.0, "CA": 2.0, "MN": 2.0, "FE": 3.0,
    "CU": 2.0, "CO": 2.0, "NI": 2.0, "CD": 2.0, "HG": 2.0,
    "NA": 1.0, "K":  1.0,
}

EXCLUDE_IONS = set(
    "HOH,WAT,DOD,SOL,NA,CL,K,CA,MG,ZN,MN,FE,CU,CO,NI,CD,HG".split(",")
)
GLYCAN_NAMES = {
    "NAG", "BMA", "MAN", "FUC", "GAL", "GLC", "SIA", "NGA",
    "FUL", "GLA", "BGC", "A2G", "LAT", "MAL", "CEL", "SUC",
    "TRE", "GCS", "NDG", "NGC",
}
COFACTOR_NAMES = {
    "ATP", "ADP", "AMP", "GTP", "GDP", "GMP",
    "NAD", "NAP", "NDP", "FAD", "FMN",
    "HEM", "HEC", "HEA",
    "GOL", "PEG", "EDO", "MPD", "PGE", "PG4",
    "SO4", "PO4", "SUL", "PHO",
    "IHP", "TTP", "CTP", "UTP",
    "COA", "SAM", "SAH",
    "EPE", "MES", "TRS", "ACT", "ACY",
}

HEME_RESNAMES = {"HEM", "HEC", "HEA", "HEB", "HDD", "HDM"}

# PoseView: max retries on "failure" before giving up
_PV_MAX_RETRIES = 3
# PoseView: seconds to wait between retries
_PV_RETRY_DELAY = 10
# PoseView: polling attempts per try (2 s each → 120 s max)
_PV_POLL_ATTEMPTS = 60


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def run_cmd(cmd, cwd=None):
    """Run a shell command. Returns (returncode, combined_stdout_stderr)."""
    r = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return r.returncode, (r.stdout + r.stderr).strip()


def _rdkit_six_patch():
    """Compatibility shim for older RDKit builds that import rdkit.six."""
    try:
        from rdkit import six  # noqa
    except ImportError:
        from io import StringIO as _SIO
        from types import ModuleType as _MT
        import rdkit as _rdkit
        _m = _MT("six")
        _m.StringIO = _SIO
        _m.PY3 = True
        _rdkit.six = _m
        sys.modules["rdkit.six"] = _m


def _strip_h_from_pdb(pdb_path: str, out_path: str) -> bool:
    """
    Remove explicit hydrogen lines from a PDB file.
    Returns True on success. Falls back to copying unchanged on error.
    """
    import shutil
    try:
        lines = []
        with open(pdb_path) as f:
            for line in f:
                rec = line[:6].strip()
                if rec in ("ATOM", "HETATM"):
                    atom_name = line[12:16].strip()
                    element   = line[76:78].strip() if len(line) > 76 else ""
                    if element.upper() == "H" or atom_name.startswith("H"):
                        continue
                lines.append(line)
        with open(out_path, "w") as f:
            f.writelines(lines)
        return True
    except Exception:
        shutil.copy(pdb_path, out_path)
        return False


def convert_cif_to_pdb(cif_path: str, pdb_out_path: str) -> dict:
    """
    Convert an mmCIF (.cif) file to PDB format.
    Tries gemmi first (preserves metadata better), then OpenBabel as fallback.
    Returns dict: success, pdb_path, log, error
    """
    log = []

    # ── Strategy A: gemmi ─────────────────────────────────────────────────────
    try:
        import gemmi
        doc    = gemmi.cif.read(cif_path)
        block  = doc.sole_block()
        st_obj = gemmi.make_structure_from_block(block)
        st_obj.setup_entities()
        st_obj.assign_label_seq_id()
        pdb_str = st_obj.make_pdb_headers() + st_obj.make_pdb_string()
        with open(pdb_out_path, "w") as f:
            f.write(pdb_str)
        if os.path.exists(pdb_out_path) and os.path.getsize(pdb_out_path) > 100:
            log.append("✓ CIF → PDB conversion via gemmi")
            return {"success": True, "pdb_path": pdb_out_path, "log": log}
        else:
            log.append("⚠ gemmi produced empty PDB — trying OpenBabel")
    except ImportError:
        log.append("⚠ gemmi not installed — trying OpenBabel")
    except Exception as e:
        log.append(f"⚠ gemmi failed ({e}) — trying OpenBabel")

    # ── Strategy B: OpenBabel ─────────────────────────────────────────────────
    try:
        rc, out = run_cmd(f'obabel "{cif_path}" -O "{pdb_out_path}" -ipdb')
        if not os.path.exists(pdb_out_path) or os.path.getsize(pdb_out_path) < 100:
            rc, out = run_cmd(f'obabel "{cif_path}" -O "{pdb_out_path}"')
        if os.path.exists(pdb_out_path) and os.path.getsize(pdb_out_path) > 100:
            log.append("✓ CIF → PDB conversion via OpenBabel")
            return {"success": True, "pdb_path": pdb_out_path, "log": log}
        else:
            log.append(f"⚠ OpenBabel CIF→PDB produced empty file (exit {rc}): {out[:300]}")
    except Exception as e:
        log.append(f"⚠ OpenBabel CIF→PDB failed: {e}")

    # ── Strategy C: ProDy parseMMCIF ──────────────────────────────────────────
    try:
        from prody import parseMMCIF, writePDB as _writePDB
        atoms = parseMMCIF(cif_path)
        if atoms is not None and atoms.numAtoms() > 0:
            _writePDB(pdb_out_path, atoms)
            if os.path.exists(pdb_out_path) and os.path.getsize(pdb_out_path) > 100:
                log.append("✓ CIF → PDB conversion via ProDy parseMMCIF")
                return {"success": True, "pdb_path": pdb_out_path, "log": log}
            else:
                log.append("⚠ ProDy parseMMCIF produced empty PDB")
        else:
            log.append("⚠ ProDy parseMMCIF returned None or 0 atoms")
    except ImportError:
        log.append("⚠ ProDy parseMMCIF not available")
    except Exception as e:
        log.append(f"⚠ ProDy parseMMCIF failed: {e}")

    return {
        "success": False,
        "pdb_path": pdb_out_path,
        "log": log,
        "error": "All CIF→PDB conversion methods failed. "
                 "Install gemmi (`pip install gemmi`) for best results.",
    }


def is_cif_file(filepath: str) -> bool:
    """Check if a file is in mmCIF format by extension or content sniffing."""
    ext = Path(filepath).suffix.lower()
    if ext in (".cif", ".mmcif"):
        return True
    try:
        with open(filepath) as f:
            first_lines = f.read(512)
        if first_lines.strip().startswith("data_"):
            return True
    except Exception:
        pass
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL AVAILABILITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def check_obabel():
    """Return (available: bool, version_or_error: str)."""
    import shutil
    if shutil.which("obabel") is None:
        return False, "obabel not found — add 'openbabel' to packages.txt"
    _, out = run_cmd("obabel --version")
    return True, (out.splitlines()[0] if out else "ok")


# ══════════════════════════════════════════════════════════════════════════════
#  VINA BINARY
# ══════════════════════════════════════════════════════════════════════════════

def get_vina_binary(path: str = ""):
    """
    Download AutoDock Vina 1.2.7 for the current platform if not present.
    Supports Linux (x86_64), macOS (x86_64, arm64), and Windows (x86_64).
    Returns (binary_path, status_message).
    """
    import platform

    system  = platform.system().lower()
    machine = platform.machine().lower()

    _BASE = (
        "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/"
    )

    if system == "linux":
        _FNAME = "vina_1.2.7_linux_x86_64"
    elif system == "darwin":
        # GitHub releases use "aarch64" for Apple Silicon (not "arm64")
        _FNAME = ("vina_1.2.7_mac_aarch64"
                  if machine in ("arm64", "aarch64")
                  else "vina_1.2.7_mac_x86_64")
    elif system == "windows":
        _FNAME = "vina_1.2.7_windows_x86_64.exe"
    else:
        return None, f"Unsupported platform: {system}/{machine}"

    _URL = _BASE + _FNAME

    if not path:
        path = os.path.join(tempfile.gettempdir(), _FNAME)

    def _download(url, dest):
        """Download with requests (follows GitHub → S3 redirects reliably)."""
        import requests as _rq
        r = _rq.get(url, stream=True, timeout=120, allow_redirects=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    if not os.path.exists(path) or os.path.getsize(path) < 100_000:
        try:
            _download(_URL, path)
        except Exception as e1:
            # Apple Silicon fallback: x86_64 runs fine under Rosetta 2
            if system == "darwin" and machine in ("arm64", "aarch64"):
                _FNAME2 = "vina_1.2.7_mac_x86_64"
                path2   = os.path.join(tempfile.gettempdir(), _FNAME2)
                try:
                    _download(_BASE + _FNAME2, path2)
                    path = path2
                except Exception as e2:
                    return None, f"Download failed: {e1} / x86_64 fallback: {e2}"
            else:
                return None, f"Download failed: {e1}"

    if system != "windows":
        os.chmod(path, 0o755)
    return path, f"ok ({system}/{machine})"


# ══════════════════════════════════════════════════════════════════════════════
#  RECEPTOR PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

# Minimum heavy atoms for a HETATM group to be treated as a real ligand.
# Groups with ≤ this count are treated as ions/solvent and ignored.
_MIN_LIG_ATOMS = 4


def _collect_removable_ligands(atoms) -> list:
    """
    Return a list of dicts, one per HETATM residue that qualifies as a
    "drug-like" molecule to be removed from the receptor before docking.

    Qualification rules (all must pass):
      • Not in EXCLUDE_IONS, GLYCAN_NAMES, COFACTOR_NAMES,
        HEME_RESNAMES, or METAL_RESNAMES  (those are handled separately)
      • Has > _MIN_LIG_ATOMS heavy atoms  (filters out ions / small solvent)

    Each dict has keys: resname, chain, resid, sel_str, n_atoms, atoms
    Sorted descending by atom count (largest = primary co-crystal ligand).
    """
    from prody import calcCenter

    excl = EXCLUDE_IONS | GLYCAN_NAMES | COFACTOR_NAMES | HEME_RESNAMES | METAL_RESNAMES
    # Use "hetatm" not "hetero": ProDy's "hetero" = "hetatm AND NOT protein".
    # Non-standard resnames (MOL, LIG, UNL, etc.) are sometimes incorrectly
    # flagged as protein=True by ProDy, so "hetero" silently drops them.
    # "hetatm" selects purely on the HETATM record type — always correct.
    het  = atoms.select("hetatm and not water")

    # Backbone atom names that identify a residue as a modified amino acid.
    # If a HETATM residue contains ALL four backbone atoms it is treated as
    # part of the protein (e.g. CYP = proximal cysteine in CYP450 enzymes,
    # MSE = selenomethionine, etc.) and must NOT be removed as a ligand.
    _BACKBONE = {"N", "CA", "C", "O"}
    if het is None:
        return []

    results = []
    for r in het.getHierView().iterResidues():
        rn = (r.getResname() or "").strip().upper()
        if rn in excl:
            continue
        if r.numAtoms() <= _MIN_LIG_ATOMS:
            continue   # too small — ion / solvent fragment
        # Skip modified amino acids: residues that contain all 4 backbone
        # heavy atoms (N, CA, C, O) are protein residues stored as HETATM
        # (e.g. CYP = proximal Cys in P450, MSE = selenomethionine, etc.)
        _atom_names = set(r.getNames())
        if _BACKBONE.issubset(_atom_names):
            continue
        ch = r.getChid()
        ri = r.getResnum()
        sel = (f"resname {rn} and resid {ri} and chain {ch}"
               if ch and ch.strip()
               else f"resname {rn} and resid {ri}")
        lig_atoms = atoms.select(sel)
        if lig_atoms is None or lig_atoms.numAtoms() == 0:
            continue
        cx_, cy_, cz_ = (float(v) for v in calcCenter(lig_atoms))
        results.append({
            "resname":   rn,
            "chain":     ch,
            "resid":     ri,
            "sel_str":   sel,
            "ligand_id": f"{rn}_{ch}_{ri}",
            "n_atoms":   lig_atoms.numAtoms(),
            "atoms":     lig_atoms,
            "cx": cx_, "cy": cy_, "cz": cz_,
        })

    # Sort: largest first, chain A preferred on tie
    results.sort(key=lambda d: (-d["n_atoms"], d["chain"] != "A"))
    return results


def detect_cocrystal_ligand(raw_pdb: str) -> dict:
    """
    Parse PDB and return the best co-crystal ligand candidate
    (largest HETATM residue with > _MIN_LIG_ATOMS atoms that is not
    a known cofactor, heme, metal, ion, or glycan).

    Returns dict with keys:
        found, resname, chain, resid, sel_str, ligand_id,
        cx, cy, cz, n_atoms, atoms
    """
    from prody import parsePDB

    atoms = parsePDB(raw_pdb)
    if atoms is None:
        return {"found": False}

    cands = _collect_removable_ligands(atoms)
    if not cands:
        return {"found": False}

    chosen = cands[0]   # largest qualifying residue
    return {
        "found":     True,
        "resname":   chosen["resname"],
        "chain":     chosen["chain"],
        "resid":     chosen["resid"],
        "sel_str":   chosen["sel_str"],
        "ligand_id": chosen["ligand_id"],
        "cx": chosen["cx"], "cy": chosen["cy"], "cz": chosen["cz"],
        "n_atoms":   chosen["n_atoms"],
        "atoms":     chosen["atoms"],
    }


def scan_ligands(raw_pdb: str) -> list:
    """
    Quick scan of a PDB file to list all removable ligands without
    performing any receptor preparation.  Used by app.py to populate
    the ligand-selector UI before the user clicks Prepare Receptor.

    Returns a list of dicts (may be empty):
        [{"resname": str, "chain": str, "resid": int, "n_atoms": int}, ...]
    Sorted by atom count descending (largest = most likely co-crystal ligand).
    """
    try:
        from prody import parsePDB, confProDy
        confProDy(verbosity="none")

        # CIF auto-convert
        if is_cif_file(raw_pdb):
            import tempfile as _tf
            _tmp = _tf.mktemp(suffix=".pdb")
            res  = convert_cif_to_pdb(raw_pdb, _tmp)
            if res["success"]:
                raw_pdb = _tmp

        atoms = parsePDB(raw_pdb)
        if atoms is None:
            return []
        ligs = _collect_removable_ligands(atoms)
        return [
            {"resname": d["resname"], "chain": d["chain"],
             "resid": d["resid"], "n_atoms": d["n_atoms"]}
            for d in ligs
        ]
    except Exception:
        return []


def strip_and_convert_receptor(rec_raw: str, wdir) -> dict:
    """
    Full receptor PDBQT preparation with metal-ion safety:
      (a) Strip metal HETATM lines before OpenBabel
      (b) Add hydrogens + convert to PDBQT (metal-free file only)
      (c) Re-inject metals into PDBQT with correct formal charges

    Returns dict: success, rec_fh, rec_pdbqt, log, error
    """
    wdir = Path(wdir)
    log  = []

    rec_fh    = str(wdir / "rec.pdb")
    rec_pdbqt = str(wdir / "rec.pdbqt")

    try:
        # (a) Strip metal ions
        metal_lines = []
        clean_lines = []
        with open(rec_raw) as f:
            for line in f:
                field = line[:6].strip()
                if (field in ("ATOM", "HETATM")
                        and line[17:20].strip().upper() in METAL_RESNAMES):
                    metal_lines.append(line)
                else:
                    clean_lines.append(line)

        rec_nometal = str(wdir / "receptor_atoms_nometal.pdb")
        with open(rec_nometal, "w") as f:
            f.writelines(clean_lines)

        if metal_lines:
            names = ", ".join(sorted({l[17:20].strip() for l in metal_lines}))
            log.append(
                f"⚠ Stripped {len(metal_lines)} metal atom(s) "
                f"before OpenBabel: {names}"
            )

        # (b) Add H + convert to PDBQT
        rc1, out1 = run_cmd(f'obabel "{rec_nometal}" -O "{rec_fh}" -h')
        if not os.path.exists(rec_fh) or os.path.getsize(rec_fh) < 100:
            raise ValueError(
                f"OpenBabel H-addition produced empty file "
                f"(exit {rc1}). Output: {out1[:400]}"
            )
        log.append("✓ Hydrogens added")

        rc2, out2 = run_cmd(
            f'obabel "{rec_fh}" -O "{rec_pdbqt}" -xr --partialcharge gasteiger'
        )
        if not os.path.exists(rec_pdbqt) or os.path.getsize(rec_pdbqt) < 100:
            raise ValueError(
                f"PDBQT conversion produced empty file "
                f"(exit {rc2}). Output: {out2[:400]}"
            )
        log.append("✓ PDBQT conversion complete")

        # (c) Re-inject metal ions
        if metal_lines:
            pdbqt_lines = open(rec_pdbqt).readlines()
            pdbqt_lines = [l for l in pdbqt_lines if l.strip() != "END"]
            injected = 0
            for ml in metal_lines:
                try:
                    resname = ml[17:20].strip().upper()
                    serial  = int(ml[6:11])
                    name    = ml[12:16].strip()
                    chain   = ml[21] if len(ml) > 21 else "A"
                    resid   = int(ml[22:26])
                    x       = float(ml[30:38])
                    y       = float(ml[38:46])
                    z       = float(ml[46:54])
                    charge  = METAL_CHARGES.get(resname, 0.0)
                    atype   = resname.capitalize()
                    pdbqt_lines.append(
                        f"HETATM{serial:5d} {name:<4s} {resname:<3s} "
                        f"{chain}{resid:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                        f"    {charge:+.3f} {atype}\n"
                    )
                    injected += 1
                except Exception as e:
                    log.append(f"⚠ Could not re-inject metal line: {e}")
            pdbqt_lines.append("END\n")
            with open(rec_pdbqt, "w") as f:
                f.writelines(pdbqt_lines)
            log.append(f"✅ Re-injected {injected} metal atom(s) into PDBQT")

        log.append("✓ Receptor PDBQT ready")
        return {
            "success":   True,
            "rec_fh":    rec_fh,
            "rec_pdbqt": rec_pdbqt,
            "log":       log,
        }

    except Exception as e:
        log.append(f"ERROR: {e}")
        return {"success": False, "error": str(e), "log": log}


def write_box_pdb(filename: str, cx, cy, cz, sx, sy, sz):
    """Write 8-corner wireframe docking box as PDB HETATM records."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    corners = [
        (cx + dx, cy + dy, cz + dz)
        for dx in (-hx, hx)
        for dy in (-hy, hy)
        for dz in (-hz, hz)
    ]
    with open(filename, "w") as f:
        for i, (x, y, z) in enumerate(corners, 1):
            f.write(
                f"HETATM{i:5d}  C   BOX A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        f.write(
            "CONECT    1    2    3    5\nCONECT    2    1    4    6\n"
            "CONECT    3    1    4    7\nCONECT    4    2    3    8\n"
            "CONECT    5    1    6    7\nCONECT    6    2    5    8\n"
            "CONECT    7    3    5    8\nCONECT    8    4    6    7\n"
        )


def write_vina_config(filename: str, cx, cy, cz, sx, sy, sz):
    """Write Vina config text file."""
    with open(filename, "w") as f:
        f.write(
            f"center_x = {cx:.4f}\n"
            f"center_y = {cy:.4f}\n"
            f"center_z = {cz:.4f}\n"
            f"size_x = {sx}\n"
            f"size_y = {sy}\n"
            f"size_z = {sz}\n"
        )


def prepare_receptor(
    raw_pdb: str,
    wdir,
    center_mode: str = "auto",
    manual_xyz: tuple = (0.0, 0.0, 0.0),
    prody_sel: str = "",
    box_size: tuple = (16, 16, 16),
    preferred_ligand: str = "",
) -> dict:
    """
    Full receptor preparation pipeline.
    Accepts PDB or mmCIF (.cif) files — CIF is auto-converted to PDB first.
    center_mode: 'auto' | 'manual' | 'selection'
    preferred_ligand: resname of the ligand to center on when multiple are
                      detected (e.g. "ELR"). Empty = use largest (default).
    Returns dict: success, rec_fh, rec_pdbqt, box_pdb, config_txt,
                  cx, cy, cz, sx, sy, sz, ligand_pdb_path,
                  cocrystal_ligand_id, n_atoms, log, error,
                  all_ligands (list of detected ligand dicts)
    """
    from prody import parsePDB, calcCenter, writePDB

    wdir = Path(wdir)
    log  = []
    sx, sy, sz = box_size

    try:
        # ── CIF auto-detection and conversion ─────────────────────────────
        if is_cif_file(raw_pdb):
            log.append("📄 Detected mmCIF format — converting to PDB…")
            converted_pdb = str(wdir / "converted_from_cif.pdb")
            cif_result = convert_cif_to_pdb(raw_pdb, converted_pdb)
            log.extend(cif_result["log"])
            if not cif_result["success"]:
                raise ValueError(
                    f"CIF → PDB conversion failed: {cif_result.get('error', 'unknown')}"
                )
            raw_pdb = converted_pdb

        atoms = parsePDB(raw_pdb)
        if atoms is None:
            raise ValueError("ProDy parsePDB returned None")
        log.append(f"✓ Parsed {atoms.numAtoms()} atoms")

        ligand_pdb_path     = None
        ligand_sel_str      = None
        cocrystal_ligand_id = ""
        rn = ch = ""
        ri = 0
        cx = cy = cz = 0.0

        # ── Detect ALL removable ligands (>_MIN_LIG_ATOMS atoms, non-cofactor) ─
        # This runs for every center mode so the receptor is always clean.
        _all_ligs = _collect_removable_ligands(atoms)

        # Honour caller's preferred ligand (by resname) when multiple detected.
        # Falls back to largest (index 0) if preferred not found or not specified.
        _primary = None
        if _all_ligs:
            if preferred_ligand:
                _pref = preferred_ligand.strip().upper()
                _primary = next(
                    (d for d in _all_ligs if d["resname"].upper() == _pref),
                    None,
                )
                if _primary is None:
                    log.append(
                        f"⚠ Preferred ligand '{preferred_ligand}' not found — "
                        f"using largest ({_all_ligs[0]['resname']})"
                    )
            if _primary is None:
                _primary = _all_ligs[0]

        # Save primary ligand PDB for RMSD, 3D viewer, PoseView2
        if _primary is not None:
            rn  = _primary["resname"]
            ch  = _primary["chain"]
            ri  = _primary["resid"]
            ligand_sel_str      = _primary["sel_str"]
            cocrystal_ligand_id = _primary["ligand_id"]
            ligand_pdb_path     = str(wdir / "LIG.pdb")
            writePDB(ligand_pdb_path, _primary["atoms"])
            _n_extra = len(_all_ligs) - 1
            log.append(
                f"✓ Co-crystal ligand: {rn} chain '{ch}' resnum {ri} "
                f"({_primary['n_atoms']} atoms)"
                + (f"  +{_n_extra} additional ligand(s) will also be removed" if _n_extra else "")
            )

        if center_mode == "auto":
            if _primary is not None:
                cx, cy, cz = _primary["cx"], _primary["cy"], _primary["cz"]
                log.append(f"📍 Auto center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
                log.append(f"🔑 PoseView2 ligand ID: {cocrystal_ligand_id}")
            else:
                log.append("⚠ No co-crystal ligand found (all HETATM ≤ 4 atoms or excluded)")

        elif center_mode == "manual":
            cx, cy, cz = (float(v) for v in manual_xyz)
            log.append(f"🛠 Manual center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
            if _primary is not None:
                log.append(f"🔑 PoseView2 ligand ID: {cocrystal_ligand_id}")

        elif center_mode == "selection":
            if not prody_sel.strip():
                raise ValueError("ProDy selection string is empty.")
            ref_atoms = atoms.select(prody_sel.strip())
            if ref_atoms is None or ref_atoms.numAtoms() == 0:
                raise ValueError(
                    f"ProDy selection '{prody_sel}' matched 0 atoms."
                )
            cx, cy, cz = (float(v) for v in calcCenter(ref_atoms))
            log.append(
                f"🔬 ProDy selection: '{prody_sel}' "
                f"→ {ref_atoms.numAtoms()} atoms"
            )
            log.append(f"📍 Center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
            if _primary is not None:
                log.append(f"🔑 PoseView2 ligand ID: {cocrystal_ligand_id}")
            else:
                log.append("⚠ No co-crystal ligand found — RMSD and co-crystal diagrams skipped")

        # ── Build receptor selection: exclude ALL removable ligands + water ───
        # Each ligand's sel_str is negated so multi-ligand structures are
        # fully cleaned regardless of which center mode was chosen.
        if _all_ligs:
            _excl_expr = " or ".join(f"({d['sel_str']})" for d in _all_ligs)
            sel_str = f"not ({_excl_expr}) and not water"
            if len(_all_ligs) > 1:
                log.append(
                    f"🧹 Removing {len(_all_ligs)} ligand(s) from receptor: "
                    + ", ".join(f"{d['resname']}({d['n_atoms']}at)" for d in _all_ligs)
                )
        else:
            sel_str = "not water"
        rec_sel = atoms.select(sel_str)
        if rec_sel is None or rec_sel.numAtoms() == 0:
            raise ValueError("Receptor selection returned no atoms")

        rec_raw_path = str(wdir / "receptor_atoms.pdb")
        writePDB(rec_raw_path, rec_sel)
        log.append(f"✓ Receptor: {rec_sel.numAtoms()} atoms")

        # ── Fix blank chain IDs ───────────────────────────────────────────
        # PDB files from MD software (GROMACS, AMBER, etc.) often have no
        # chain ID (space). OpenBabel and Vina both require a real chain
        # letter — a fully-blank-chain receptor causes Vina exit code 1.
        # If ALL atom/HETATM records have blank chain, assign chain A.
        try:
            _rec_lines  = open(rec_raw_path).readlines()
            _coord_lines = [l for l in _rec_lines if l[:6].strip() in ("ATOM", "HETATM")]
            _all_blank  = all(
                (l[21] == " " if len(l) > 21 else True)
                for l in _coord_lines
            )
            if _all_blank and _coord_lines:
                _fixed = []
                for l in _rec_lines:
                    if l[:6].strip() in ("ATOM", "HETATM") and len(l) > 21:
                        l = l[:21] + "A" + l[22:]
                    _fixed.append(l)
                with open(rec_raw_path, "w") as _f:
                    _f.writelines(_fixed)
                log.append("✓ Assigned chain A to all blank-chain atoms")
        except Exception as _ce:
            log.append(f"⚠ Chain-fix skipped: {_ce}")
        # ─────────────────────────────────────────────────────────────────

        # PDBQT conversion (metal-safe)
        conv = strip_and_convert_receptor(rec_raw_path, wdir)
        log.extend(conv["log"])
        if not conv["success"]:
            raise ValueError(conv["error"])

        # Box PDB + Vina config
        box_pdb  = str(wdir / "rec.box.pdb")
        cfg_path = str(wdir / "rec.box.txt")
        write_box_pdb(box_pdb, cx, cy, cz, sx, sy, sz)
        write_vina_config(cfg_path, cx, cy, cz, sx, sy, sz)
        log.append("✓ Box + config written")

        return {
            "success":             True,
            "rec_fh":              conv["rec_fh"],
            "rec_pdbqt":           conv["rec_pdbqt"],
            "box_pdb":             box_pdb,
            "config_txt":          cfg_path,
            "cx": cx, "cy": cy, "cz": cz,
            "sx": sx, "sy": sy, "sz": sz,
            "ligand_pdb_path":     ligand_pdb_path,
            "cocrystal_ligand_id": cocrystal_ligand_id,
            "n_atoms":             rec_sel.numAtoms(),
            "all_ligands":         [
                {"resname": d["resname"], "chain": d["chain"],
                 "resid": d["resid"], "n_atoms": d["n_atoms"]}
                for d in _all_ligs
            ],
            "log":                 log,
        }

    except Exception as e:
        log.append(f"ERROR: {e}")
        return {"success": False, "error": str(e), "log": log}


# ══════════════════════════════════════════════════════════════════════════════
#  LIGAND PREPARATION
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  PKANET CLOUD — ionizable site table + HH scoring helpers
#  Ported from pKaNET_Cloud notebook (Hengphasatporn et al. 2026)
# ══════════════════════════════════════════════════════════════════════════════

# Ionizable site SMARTS table: (label, SMARTS, heuristic_pKa, site_type)
_IONIZABLE_SITE_DEF = [
    ("sulfonic_acid",     "[SX4](=O)(=O)[OX2H1]",                               1.0,  "acid"),
    ("phosphoric_mono",   "[PX4](=O)([OX2H1])([OX2H1])[OX2H1]",                2.1,  "acid"),
    ("carboxylic_acid",   "[CX3](=O)[OX2H1]",                                   4.5,  "acid"),
    ("tetrazole",         "c1nn[nH]n1",                                          4.9,  "acid"),
    ("imidazole_acid",    "c1cn[nH]c1",                                          6.0,  "acid"),
    ("benzimidazole",     "c1ccc2[nH]cnc2c1",                                   5.5,  "acid"),
    ("phosphonate",       "[PX4](=O)([OX2H1])[OX2H1,OX1-]",                    6.5,  "acid"),
    ("sulfonamide_NH",    "[SX4](=O)(=O)[NX3;H1]",                             10.1,  "acid"),
    ("imide_NH",          "[CX3](=O)[NX3;H1][CX3]=O",                           9.6,  "acid"),
    ("phenol",            "c[OX2H1]",                                           10.0,  "acid"),
    ("thiol_arom",        "c[SX2H1]",                                            6.5,  "acid"),
    ("thiol_aliph",       "[CX4][SX2H1]",                                       10.5,  "acid"),
    ("aniline",           "c[NX3;H1,H2;!$(N~[!#6])]",                           4.6,  "base"),
    ("pyridine_like",     "[$([nX2]1:[c,n]:c:[c,n]:c1),$([nX2]:c:n)]",          5.2,  "base"),
    ("aliphatic_amine",   "[NX3;H1,H2;!$(NC=O);!$(N~[!#6;!H]);!$([nH])]",      9.5,  "base"),
    ("aliphatic_amine_t", "[NX3;H0;!$(NC=O);!$(Nc);!$([nH]);!$([N]~[!#6])]",   9.0,  "base"),
    ("amidine",           "[CX3](=[NX2;H0,H1])[NX3;H1,H2]",                   12.4,  "base"),
    ("guanidine",         "[NX3][CX3](=[NX2])[NX3]",                           13.0,  "base"),
]

# Chemistry bonus/penalty rules for tautomer plausibility scoring
_CHEM_BONUS_DEF = [
    ("amide",            +2.5, "[CX3](=O)[NX3;H1,H2]"),
    ("lactam",           +2.5, "[C;R](=O)[N;R]"),
    ("urea_NH",          +1.5, "[NX3;H1][CX3](=O)[NX3;H1,H2]"),
    ("aromatic_ring",    +0.3, "c1ccccc1"),
]
_CHEM_PENALTY_DEF = [
    ("lactim_ring",      -4.0, "[C;R](=[NX2])[OX2H1]"),
    ("iminol_general",   -3.5, "[NX2]=[CX3][OX2H1]"),
    ("amide_N_deproton", -5.0, "[$([NX3-]C=O),$([NX3-]c=O)]"),
    ("enol_simple",      -1.2, "[CX3](=[CX3])[OX2H1]"),
]

# In-memory pKaNET cache: InChIKey → {pka_values, confidence, source}
_PKANET_CACHE: dict = {}
# PubChem pKa regex patterns
import re as _re
_PKA_PATTERNS = [
    _re.compile(r"pK[aA][\w\s\(\)]*?=\s*([+-]?\d+(?:\.\d+)?)", _re.IGNORECASE),
    _re.compile(r"([+-]?\d+(?:\.\d+)?)\s*\((?:pK[aA]|acid dissociation)[^)]*\)", _re.IGNORECASE),
]


def _compile_smarts_table(defs):
    from rdkit import Chem
    compiled = []
    for row in defs:
        lbl, *rest = row
        sma = rest[0] if isinstance(rest[0], str) else rest[-1]
        extra = rest[1:] if isinstance(rest[0], str) else []
        pat = Chem.MolFromSmarts(sma)
        if pat is not None:
            compiled.append((lbl, pat) + tuple(extra))
    return compiled


def _pubchem_pka_lookup(smiles: str) -> dict:
    """
    Fetch experimental pKa values from PubChem for a SMILES.
    Returns dict: {available, pka_values, confidence, source}
    Cached by InChIKey — free, but ~0.5–2 s on first call.
    """
    from rdkit import Chem
    import time as _time
    result = {"available": False, "pka_values": [], "confidence": "low", "source": "pubchem"}
    try:
        import requests as _req
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return result
        ik = Chem.MolToInchiKey(mol)
        if not ik:
            return result

        # Check cache
        if ik in _PKANET_CACHE:
            return _PKANET_CACHE[ik]

        _time.sleep(0.25)  # PubChem rate limit
        r = _req.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{ik}/cids/JSON",
            timeout=8,
        )
        if r.status_code != 200:
            _PKANET_CACHE[ik] = result
            return result
        cid = r.json()["IdentifierList"]["CID"][0]

        _time.sleep(0.25)
        r2 = _req.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
            f"?heading=Dissociation+Constants",
            timeout=10,
        )
        if r2.status_code != 200:
            _PKANET_CACHE[ik] = result
            return result

        # Parse pKa values from response text
        texts = []
        def _collect(sec):
            if "dissociation" in sec.get("TOCHeading", "").lower():
                for info in sec.get("Information", []):
                    for swm in info.get("Value", {}).get("StringWithMarkup", []):
                        s = swm.get("String", "").strip()
                        if s:
                            texts.append(s)
            for sub in sec.get("Section", []):
                _collect(sub)
        for sec in r2.json().get("Record", {}).get("Section", []):
            _collect(sec)

        vals = []
        for text in texts:
            for pat in _PKA_PATTERNS:
                for m in pat.finditer(text):
                    try:
                        v = float(m.group(1))
                        if -5.0 <= v <= 20.0 and not any(abs(v - e) < 0.05 for e in vals):
                            vals.append(v)
                    except ValueError:
                        pass

        confidence = "high" if len(vals) == 1 else ("medium" if vals else "low")
        result = {"available": bool(vals), "pka_values": vals,
                  "confidence": confidence, "source": "pubchem"}
        _PKANET_CACHE[ik] = result
        return result

    except Exception:
        return result


def _find_ionizable_sites(mol) -> list:
    """Return ionizable sites found in mol using heuristic SMARTS table."""
    from rdkit import Chem
    _compiled = _compile_smarts_table(
        [(lbl, sma, pka, stype) for lbl, sma, pka, stype in _IONIZABLE_SITE_DEF]
    )
    sites = []
    seen = set()
    for lbl, pat, pka, stype in _compiled:
        for match in mol.GetSubstructMatches(pat):
            k = frozenset(match)
            if k not in seen:
                seen.add(k)
                sites.append({"label": lbl, "atom_indices": list(match),
                               "heuristic_pka": pka, "site_type": stype})
    return sites


def _hh_fraction_charged(pka: float, ph: float, site_type: str) -> float:
    """Henderson–Hasselbalch fraction in charged form."""
    if site_type == "acid":
        return 1.0 / (1.0 + 10.0 ** (pka - ph))
    return 1.0 / (1.0 + 10.0 ** (ph - pka))


def _hh_match_score(pka: float, ph: float, site_type: str, actual_charge: int) -> float:
    """Score how well an observed site charge matches HH prediction."""
    f_charged = _hh_fraction_charged(pka, ph, site_type)
    dpH = abs(ph - pka)
    if site_type == "acid":
        expected_neg = f_charged > 0.5
        if expected_neg and actual_charge < 0:
            return min(1.2, dpH * 0.45)
        elif expected_neg:
            return -min(1.0, dpH * 0.35)
        elif actual_charge >= 0:
            return 0.1
        else:
            return -min(1.2, dpH * 0.40)
    else:
        expected_pos = f_charged > 0.5
        if expected_pos and actual_charge > 0:
            return min(1.2, dpH * 0.45)
        elif expected_pos:
            return -min(1.0, dpH * 0.35)
        elif actual_charge <= 0:
            return 0.1
        else:
            return -min(1.2, dpH * 0.40)


def _score_tautomer(smiles: str) -> float:
    """Simple tautomer plausibility score from SMARTS bonuses/penalties."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -999.0
    total = 0.0
    for lbl, wt, sma in _CHEM_BONUS_DEF + _CHEM_PENALTY_DEF:
        pat = Chem.MolFromSmarts(sma)
        if pat and mol.HasSubstructMatch(pat):
            total += wt
    return total


def _score_microstate(smiles: str, ph: float,
                       taut_score: float, pubchem: dict) -> float:
    """
    Layered microstate score (simplified from pKaNET Cloud Stage G).
    Layer 1: amide N-deprotonation safety penalty
    Layer 2: tautomer plausibility
    Layer 3: Henderson–Hasselbalch pH consistency
    Layer 4: PubChem evidence bonus
    Layer 5: charge-structure reasonableness
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1e9

    fc_map = {a.GetIdx(): int(a.GetFormalCharge()) for a in mol.GetAtoms()}
    net = sum(fc_map.values())
    n_pos = sum(1 for v in fc_map.values() if v > 0)
    n_neg = sum(1 for v in fc_map.values() if v < 0)

    # Layer 1 — amide N-deprotonation hard penalty
    pat_bad = Chem.MolFromSmarts("[$([NX3-]C=O),$([NX3-]c=O)]")
    s1 = -5.0 * len(mol.GetSubstructMatches(pat_bad)) if pat_bad else 0.0

    # Layer 2 — tautomer plausibility
    s2 = 0.65 * taut_score

    # Layer 3 — HH pH consistency
    sites = _find_ionizable_sites(mol)
    s3 = 0.0
    for site in sites:
        pka = site["heuristic_pka"]
        # Upgrade pKa if PubChem has high-confidence data
        if pubchem.get("available") and pubchem.get("confidence") in ("high", "medium"):
            pc_vals = pubchem.get("pka_values", [])
            if pc_vals:
                pka = min(pc_vals, key=lambda v: abs(v - pka))
        site_charge = sum(fc_map.get(i, 0) for i in site["atom_indices"])
        s3 += _hh_match_score(pka, ph, site["site_type"], site_charge)

    # Layer 4 — PubChem bonus (charge-sign only)
    s4 = 0.0
    if pubchem.get("available"):
        w = {"high": 1.0, "medium": 0.6, "low": 0.2}.get(pubchem.get("confidence", "low"), 0.2)
        for pv in pubchem.get("pka_values", []):
            exp = -1 if _hh_fraction_charged(pv, ph, "acid") > 0.5 else 0
            s4 += 0.25 * w if net == exp else -0.15 * w
        s4 = max(-0.4, min(0.5, s4))

    # Layer 5 — zwitterion reasonableness
    has_acid = any(s["site_type"] == "acid" and (ph - s["heuristic_pka"]) > 1.0 for s in sites)
    has_base = any(s["site_type"] == "base" and (s["heuristic_pka"] - ph) > 1.0 for s in sites)
    is_zw = (n_pos > 0 and n_neg > 0 and net == 0)
    if is_zw:
        s5 = 0.8 if (has_acid and has_base) else -0.6
    else:
        s5 = 0.0

    return s1 + s2 + s3 + s4 + s5


def protonate_pkanet(
    smiles: str,
    ph: float,
    use_pubchem: bool = True,
    max_tautomers: int = 8,   # match standalone pKaNET Cloud+ default
    ph_window: float = 1.0,   # Dimorphite enumeration window ±ph_window
) -> tuple:
    """
    pKaNET Cloud protonation pipeline (Hengphasatporn et al. 2026).

    Steps:
      A. RDKit standardize (largest fragment, normalize, canonicalize)
      B. PubChem experimental pKa lookup (optional, cached by InChIKey)
      C. Dimorphite-DL protonation enumeration  (ph ± ph_window)
      D. Tautomer enumeration (up to max_tautomers per candidate)
      E. HH microstate ranking over all tautomer×protonation states
      F. Return best-scoring protonation state SMILES

    Returns (best_smiles, charge, log_list)
    Falls back to plain Dimorphite result on any error.
    """
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    log = []
    canonical = smiles.strip()

    # ── A. Standardize ────────────────────────────────────────────────────────
    try:
        mol = Chem.MolFromSmiles(canonical)
        if mol:
            mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
            try:
                mol = rdMolStandardize.Normalizer().normalize(mol)
            except Exception:
                pass
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            log.append("✓ RDKit standardized")
    except Exception as e:
        log.append(f"⚠ Standardization skipped: {e}")

    # ── B. PubChem lookup (optional, cached) ──────────────────────────────────
    pubchem = {"available": False, "pka_values": [], "confidence": "low"}
    if use_pubchem:
        try:
            pubchem = _pubchem_pka_lookup(canonical)
            if pubchem["available"]:
                log.append(
                    f"✓ PubChem pKa: {pubchem['pka_values']} "
                    f"(confidence: {pubchem['confidence']})"
                )
            else:
                log.append("ℹ PubChem: no experimental pKa found — using heuristic SMARTS")
        except Exception as e:
            log.append(f"⚠ PubChem lookup failed: {e}")

    # ── C. Dimorphite-DL protonation enumeration ──────────────────────────────
    candidates = [canonical]
    try:
        from dimorphite_dl import protonate_smiles as _dim
        raw = None
        for kwargs in [
            {"ph_min": ph - ph_window, "ph_max": ph + ph_window, "max_variants": 16},
            {"min_ph": ph - ph_window, "max_ph": ph + ph_window, "max_variants": 16},
        ]:
            try:
                raw = _dim(canonical, **kwargs)
                break
            except TypeError:
                continue
        if raw:
            seen = {canonical}
            for s in (raw if isinstance(raw, list) else [raw]):
                c = Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True) if Chem.MolFromSmiles(s) else None
                if c and c not in seen:
                    seen.add(c)
                    candidates.append(c)
        log.append(f"✓ Dimorphite-DL: {len(candidates)} protonation state(s)")
    except Exception as e:
        log.append(f"⚠ Dimorphite-DL skipped: {e}")

    # ── D. Tautomer expansion (up to max_tautomers per protonation state) ─────
    all_states = []
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize as _rms
        te = _rms.TautomerEnumerator()
        te.SetMaxTautomers(max_tautomers)
        seen_all = set()
        for base_smi in candidates:
            base_mol = Chem.MolFromSmiles(base_smi)
            if base_mol is None:
                all_states.append(base_smi)
                continue
            try:
                tauts = te.Enumerate(base_mol)
                for t in tauts:
                    ts = Chem.MolToSmiles(t, canonical=True)
                    if ts not in seen_all:
                        seen_all.add(ts)
                        all_states.append(ts)
            except Exception:
                if base_smi not in seen_all:
                    seen_all.add(base_smi)
                    all_states.append(base_smi)
        log.append(f"✓ Tautomer expansion: {len(all_states)} total state(s) to rank")
    except Exception as e:
        log.append(f"⚠ Tautomer expansion skipped: {e}")
        all_states = candidates

    # ── E. Microstate ranking (HH scoring over all states) ────────────────────
    try:
        scored = []
        for smi in all_states:
            ts = _score_tautomer(smi)
            ms = _score_microstate(smi, ph, ts, pubchem)
            scored.append((ms, smi))
        scored.sort(reverse=True)
        best_smi = scored[0][1]
        log.append(
            f"✓ pKaNET ranked {len(scored)} state(s) — "
            f"best score: {scored[0][0]:.2f}"
        )
    except Exception as e:
        log.append(f"⚠ Ranking failed ({e}) — using first Dimorphite state")
        best_smi = candidates[0]

    mol_best = Chem.MolFromSmiles(best_smi)
    charge = int(Chem.GetFormalCharge(mol_best)) if mol_best else 0
    return best_smi, charge, log


# ══════════════════════════════════════════════════════════════════════════════
#  LIGAND PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def _meeko_to_pdbqt(mol, out_path: str):
    """Convert an RDKit mol to PDBQT via Meeko."""
    from meeko import MoleculePreparation
    prep = MoleculePreparation()
    try:
        from meeko import PDBQTWriterLegacy
        setups    = prep.prepare(mol)
        pdbqt_str, _, _ = PDBQTWriterLegacy.write_string(setups[0])
    except (ImportError, AttributeError):
        prep.prepare(mol)
        pdbqt_str = prep.write_pdbqt_string()
    with open(out_path, "w") as f:
        f.write(pdbqt_str)


def prepare_ligand(
    smiles: str,
    name: str,
    ph: float,
    wdir,
    mode: str = "dimorphite",
    use_pubchem: bool = True,
    max_tautomers: int = 8,
    ph_window: float = 1.0,
) -> dict:
    """
    Protonate ligand → 3D conformer (ETKDGv3) → MMFF/UFF → PDBQT + SDF.

    Parameters
    ----------
    smiles        : input SMILES string
    name          : output file base name
    ph            : target pH
    wdir          : working directory
    mode          : protonation mode —
                   "neutral"     : add all H, no ionization (fastest)
                   "dimorphite"  : Dimorphite-DL only (default, fast)
                   "pkanet"      : pKaNET Cloud pipeline — Dimorphite +
                                   PubChem pKa + HH microstate ranking
    use_pubchem   : if mode=="pkanet", query PubChem (cached; ~0.5–2s first call)
    max_tautomers : pKaNET only — tautomers per protonation state (default 8,
                   matching standalone pKaNET Cloud+ default)
    ph_window     : pKaNET/Dimorphite — Dimorphite enumeration window ±ph_window

    Returns dict: success, pdbqt, sdf, prot_smiles, charge, log, error
    """
    _rdkit_six_patch()
    from rdkit import Chem
    from rdkit.Chem import AllChem

    wdir      = Path(wdir)
    log       = []
    out_pdbqt = str(wdir / f"{name}.pdbqt")
    out_sdf   = str(wdir / f"{name}_3d.sdf")

    try:
        raw = smiles.strip()

        # ── Protonation ───────────────────────────────────────────────────────
        if mode == "neutral":
            # Just add all hydrogens — no ionization, no pKa
            mol_check = Chem.MolFromSmiles(raw)
            if mol_check is None:
                raise ValueError(f"RDKit could not parse SMILES: {raw[:60]}")
            # Strip any existing charges and add H neutrally
            from rdkit.Chem.MolStandardize import rdMolStandardize
            try:
                mol_check = rdMolStandardize.Uncharger().uncharge(mol_check)
                raw = Chem.MolToSmiles(mol_check, isomericSmiles=True, canonical=True)
            except Exception:
                pass
            prot = raw
            log.append("✓ Neutral form (all H added, no ionization)")

        elif mode == "pkanet":
            # Full pKaNET Cloud pipeline
            prot, _, pkanet_log = protonate_pkanet(
                raw, ph,
                use_pubchem=use_pubchem,
                max_tautomers=max_tautomers,
                ph_window=ph_window,
            )
            log.extend(pkanet_log)

        else:
            # Default: Dimorphite-DL (mode == "dimorphite")
            prot = raw
            try:
                from dimorphite_dl import protonate_smiles
                vs = protonate_smiles(prot, ph_min=ph, ph_max=ph, max_variants=1)
                if vs:
                    prot = vs[0]
                    log.append(f"✓ Dimorphite-DL pH {ph}")
            except Exception as e:
                log.append(f"⚠ Dimorphite-DL skipped: {e}")

        mol = Chem.MolFromSmiles(prot)
        if mol is None:
            raise ValueError(f"RDKit could not parse protonated SMILES: {prot[:60]}")

        charge = Chem.GetFormalCharge(mol)
        log.append(f"✓ Formal charge: {charge:+d}")

        # ── 3D conformer + minimization ───────────────────────────────────────
        mol = Chem.AddHs(mol)
        try:
            params = AllChem.ETKDGv3()
        except AttributeError:
            params = AllChem.ETKDG()
        params.randomSeed = 42

        if AllChem.EmbedMolecule(mol, params) == -1:
            AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)

        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        log.append("✓ 3D conformer generated + minimised")

        with Chem.SDWriter(out_sdf) as w:
            w.write(mol)

        _meeko_to_pdbqt(mol, out_pdbqt)
        log.append("✓ PDBQT written (Meeko)")

        return {
            "success":     True,
            "pdbqt":       out_pdbqt,
            "sdf":         out_sdf,
            "prot_smiles": prot,
            "charge":      charge,
            "log":         log,
        }

    except Exception as e:
        log.append(f"ERROR: {e}")
        return {"success": False, "error": str(e), "log": log}




def prepare_ligand_from_file(file_path: str, name: str, wdir) -> dict:
    """
    Prepare a ligand directly from an uploaded structure file (PDB/SDF/MOL2)
    WITHOUT protonation — use the molecule exactly as provided.
    Returns dict: success, pdbqt, sdf, prot_smiles, charge, log, error
    """
    _rdkit_six_patch()
    from rdkit import Chem
    from rdkit.Chem import AllChem

    wdir      = Path(wdir)
    log       = []
    out_pdbqt = str(wdir / f"{name}.pdbqt")
    out_sdf   = str(wdir / f"{name}_3d.sdf")
    ext       = Path(file_path).suffix.lower()

    try:
        mol = None
        if ext == ".sdf":
            supp = Chem.SDMolSupplier(file_path, removeHs=False, sanitize=True)
            mols = [m for m in supp if m]
            if not mols:
                supp = Chem.SDMolSupplier(file_path, removeHs=False, sanitize=False)
                mols = [m for m in supp if m]
            if mols:
                mol = mols[0]
        elif ext in (".mol2", ".pdb"):
            _ob_sdf = str(wdir / f"{name}_ob.sdf")
            import subprocess
            subprocess.run(
                f'obabel "{file_path}" -O "{_ob_sdf}" 2>/dev/null',
                shell=True, timeout=30,
            )
            if Path(_ob_sdf).exists() and Path(_ob_sdf).stat().st_size > 10:
                supp = Chem.SDMolSupplier(_ob_sdf, removeHs=False, sanitize=True)
                mols = [m for m in supp if m]
                if not mols:
                    supp = Chem.SDMolSupplier(_ob_sdf, removeHs=False, sanitize=False)
                    mols = [m for m in supp if m]
                if mols:
                    mol = mols[0]
                    log.append("✓ Converted via OpenBabel")
            if mol is None:
                if ext == ".mol2":
                    mol = Chem.MolFromMol2File(file_path, removeHs=False, sanitize=True)
                    if mol is None:
                        mol = Chem.MolFromMol2File(file_path, removeHs=False, sanitize=False)
                else:
                    mol = Chem.MolFromPDBFile(file_path, removeHs=False, sanitize=True)
                    if mol is None:
                        mol = Chem.MolFromPDBFile(file_path, removeHs=False, sanitize=False)

        if mol is None:
            raise ValueError(f"Could not read molecule from {Path(file_path).name}")

        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            frags = sorted(frags, key=lambda m: m.GetNumAtoms(), reverse=True)
            mol = frags[0]
            log.append(f"⚠ {len(frags)} fragments — kept largest ({mol.GetNumAtoms()} atoms)")

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass

        log.append("✓ Loaded molecule from file (no protonation)")

        try:
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        except Exception:
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception:
                smi = name
        try:
            charge = Chem.GetFormalCharge(mol)
        except Exception:
            charge = 0
        log.append(f"✓ Formal charge: {charge:+d}")

        mol = Chem.AddHs(mol, addCoords=True)
        log.append("✓ All hydrogens made explicit")

        conf = mol.GetConformer(0) if mol.GetNumConformers() > 0 else None
        if conf is None or conf.Is3D() is False:
            try:
                params = AllChem.ETKDGv3()
            except AttributeError:
                params = AllChem.ETKDG()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol, params) == -1:
                AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            log.append("✓ 3D conformer generated (no coords in file)")
        else:
            log.append("✓ Using 3D coordinates from uploaded file")

        with Chem.SDWriter(out_sdf) as w:
            w.write(mol)

        try:
            _meeko_to_pdbqt(mol, out_pdbqt)
            log.append("✓ PDBQT written (Meeko)")
        except Exception as e_meeko:
            log.append(f"⚠ Meeko failed ({e_meeko}), trying OpenBabel…")
            import subprocess
            subprocess.run(
                f'obabel "{out_sdf}" -O "{out_pdbqt}" -xh 2>/dev/null',
                shell=True, timeout=30,
            )
            if not Path(out_pdbqt).exists() or Path(out_pdbqt).stat().st_size < 10:
                raise ValueError(f"Both Meeko and OpenBabel failed: {e_meeko}")
            log.append("✓ PDBQT written (OpenBabel fallback)")

        return {
            "success":     True,
            "pdbqt":       out_pdbqt,
            "sdf":         out_sdf,
            "prot_smiles": smi,
            "charge":      charge,
            "log":         log,
        }
    except Exception as e:
        log.append(f"ERROR: {e}")
        return {"success": False, "error": str(e), "log": log}


def smiles_from_file(file_path: str, wdir) -> str:
    """Extract SMILES from SDF / MOL2 / PDB. Raises ValueError on failure."""
    wdir = Path(wdir)
    ext  = Path(file_path).suffix.lower()

    if ext == ".sdf":
        from rdkit import Chem
        mols = [m for m in Chem.SDMolSupplier(file_path, sanitize=True) if m]
        if not mols:
            raise ValueError("No valid molecule in SDF file")
        return Chem.MolToSmiles(mols[0])
    else:
        smi_tmp = str(wdir / "lig_upload.smi")
        run_cmd(f'obabel "{file_path}" -O "{smi_tmp}" --canonical 2>/dev/null')
        for line in open(smi_tmp):
            pts = line.strip().split(None, 1)
            if pts:
                return pts[0]
        raise ValueError("Could not convert structure file to SMILES")


# ══════════════════════════════════════════════════════════════════════════════
#  DOCKING
# ══════════════════════════════════════════════════════════════════════════════

def run_vina(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    config_txt: str,
    vina_path: str,
    exhaustiveness: int = 16,
    n_modes: int = 10,
    energy_range: int = 3,
    wdir = ".",
    out_name: str = "out",
) -> dict:
    """
    Run AutoDock Vina and convert output to SDF.
    Returns dict: success, out_pdbqt, out_sdf, scores, top_score, log, error
    """
    wdir      = Path(wdir)
    out_pdbqt = str(wdir / f"{out_name}_out.pdbqt")
    out_sdf   = str(wdir / f"{out_name}_out.sdf")

    rc, vlog = run_cmd(
        f'"{vina_path}" '
        f'--receptor "{receptor_pdbqt}" '
        f'--ligand "{ligand_pdbqt}" '
        f'--config "{config_txt}" '
        f'--exhaustiveness {exhaustiveness} '
        f'--num_modes {n_modes} '
        f'--energy_range {energy_range} '
        f'--out "{out_pdbqt}"',
        cwd=str(wdir),
    )

    if rc != 0 or not os.path.exists(out_pdbqt):
        return {"success": False, "error": f"Vina exit code {rc}", "log": vlog}

    run_cmd(f'obabel "{out_pdbqt}" -O "{out_sdf}" 2>/dev/null')

    scores    = []
    cur_model = None
    for line in open(out_pdbqt):
        ln = line.strip()
        if ln.startswith("MODEL"):
            try:
                cur_model = int(ln.split()[1])
            except Exception:
                pass
        elif ln.startswith("REMARK VINA RESULT:"):
            try:
                p = ln.split()
                scores.append({
                    "pose":     cur_model,
                    "affinity": float(p[3]),
                    "rmsd_lb":  float(p[4]),
                    "rmsd_ub":  float(p[5]),
                })
            except Exception:
                pass

    return {
        "success":   True,
        "out_pdbqt": out_pdbqt,
        "out_sdf":   out_sdf,
        "scores":    scores,
        "top_score": scores[0]["affinity"] if scores else None,
        "log":       vlog,
    }


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
        raise RuntimeError(f"AssignBondOrdersFromTemplate failed: {exc}") from exc

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


def fix_sdf_bond_orders(raw_sdf: str, smiles: str, fixed_sdf: str) -> list:
    """
    Apply bond-order + formal-charge correction to all poses in an SDF.
    Returns list of log messages.
    """
    import shutil
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")

    log = []

    try:
        template = _bo_template(smiles)
    except Exception as e:
        log.append(f"⚠ Could not build template: {e} — skipping fix")
        RDLogger.EnableLog("rdApp.error")
        shutil.copy(raw_sdf, fixed_sdf)
        return log

    supplier = Chem.SDMolSupplier(raw_sdf, sanitize=False, removeHs=False)
    writer   = Chem.SDWriter(fixed_sdf)
    writer.SetKekulize(False)

    ok = err = 0
    for i, mol in enumerate(supplier):
        if mol is None:
            log.append(f"⚠ Pose {i+1}: could not read — skipped")
            err += 1
            continue
        try:
            fixed = _bo_fix_mol(mol, template)

            try:
                fixed_h = Chem.AddHs(fixed, addCoords=True)
                conf    = fixed_h.GetConformer()
                bad = any(
                    abs(conf.GetAtomPosition(j).x)
                    + abs(conf.GetAtomPosition(j).y)
                    + abs(conf.GetAtomPosition(j).z) < 0.01
                    for j in range(fixed_h.GetNumAtoms())
                    if fixed_h.GetAtomWithIdx(j).GetAtomicNum() == 1
                )
                writer.write(fixed if bad else fixed_h)
            except Exception:
                writer.write(fixed)

            ok += 1
        except Exception as e:
            log.append(f"⚠ Pose {i+1}: fix failed ({e}) — writing raw")
            mol_noH = Chem.RemoveHs(mol, sanitize=False)
            writer.write(mol_noH)
            err += 1

    writer.close()
    RDLogger.EnableLog("rdApp.error")
    log.append(f"Bond-order fix: {ok} OK, {err} fallback")
    return log


# ══════════════════════════════════════════════════════════════════════════════
#  SDF UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_mols_from_sdf(sdf_path: str, sanitize: bool = True) -> list:
    """Load all valid mols from an SDF file."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    mols = []
    try:
        sup = Chem.SDMolSupplier(sdf_path, sanitize=sanitize, removeHs=False)
        for m in sup:
            if m is not None:
                mols.append(m)
    except Exception:
        pass

    if sanitize:
        try:
            sup2 = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
            raw  = [m for m in sup2 if m is not None]
            if len(raw) > len(mols):
                result = []
                for m in raw:
                    try:
                        Chem.SanitizeMol(m)
                    except Exception:
                        pass
                    result.append(m)
                RDLogger.EnableLog("rdApp.error")
                return result
        except Exception:
            pass

    RDLogger.EnableLog("rdApp.error")
    return mols


def write_single_pose(mol, path: str) -> None:
    """Write a single RDKit mol to SDF."""
    from rdkit import Chem
    with Chem.SDWriter(path) as w:
        w.write(mol)


def convert_sdf_to_v2000(sdf_path: str) -> str:
    """Convert an SDF to Kekulized V2000 format suitable for PoseView."""
    from rdkit import Chem, RDLogger

    out = sdf_path.replace(".sdf", "_v2000.sdf")
    if out == sdf_path:
        out = sdf_path + "_v2000.sdf"

    RDLogger.DisableLog("rdApp.*")
    try:
        mol = None
        for sanitize in (True, False):
            sup = Chem.SDMolSupplier(sdf_path, sanitize=sanitize, removeHs=True)
            mol = next((m for m in sup if m is not None), None)
            if mol is not None:
                if not sanitize:
                    try:
                        Chem.SanitizeMol(mol)
                    except Exception:
                        pass
                break

        if mol is not None and mol.GetNumConformers() > 0:
            mol_noH = Chem.RemoveHs(mol, sanitize=False)
            try:
                Chem.SanitizeMol(mol_noH)
            except Exception:
                pass
            try:
                Chem.Kekulize(mol_noH, clearAromaticFlags=True)
            except Exception:
                pass

            w = Chem.SDWriter(out)
            w.SetKekulize(True)
            w.write(mol_noH)
            w.close()

            if os.path.exists(out) and os.path.getsize(out) > 10:
                RDLogger.EnableLog("rdApp.error")
                return out
    except Exception:
        pass
    RDLogger.EnableLog("rdApp.error")

    rc, _ = run_cmd(f'obabel "{sdf_path}" -O "{out}" 2>/dev/null')
    if rc == 0 and os.path.exists(out) and os.path.getsize(out) > 10:
        return out

    return sdf_path


# ══════════════════════════════════════════════════════════════════════════════
#  STRUCTURAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def get_interacting_residues(receptor_pdb: str, lig_mol, cutoff: float = 3.5) -> list:
    """
    Return protein residues within cutoff Å of any ligand heavy atom.
    Each entry: {"chain": str, "resi": int, "resn": str}
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

        seen = {}
        for j in range(len(r_xyz)):
            if np.linalg.norm(lig_xyz - r_xyz[j], axis=1).min() <= cutoff:
                key = (str(chains[j]), int(resids[j]))
                if key not in seen:
                    seen[key] = str(resnames[j])

        return [{"chain": k[0], "resi": k[1], "resn": v} for k, v in seen.items()]

    except Exception:
        return []


def calc_rmsd_heavy(pose_mol, crystal_pdb_path: str):
    """
    Heavy-atom RMSD (Å) between a docked pose and a co-crystal PDB.
    Uses MCS matching; requires ≥60% MCS coverage.
    Returns float or None on any failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFMCS
        import numpy as np

        if not os.path.exists(crystal_pdb_path):
            return None

        cryst = None
        for sanitize, removeHs, proxBonding in [
            (True,  True, True),
            (False, True, True),
            (True,  True, False),
            (False, True, False),
        ]:
            try:
                cryst = Chem.MolFromPDBFile(
                    crystal_pdb_path,
                    sanitize=sanitize,
                    removeHs=removeHs,
                    proximityBonding=proxBonding,
                )
                if cryst is not None and cryst.GetNumConformers() > 0:
                    if not sanitize:
                        try:
                            Chem.SanitizeMol(cryst)
                        except Exception:
                            pass
                    break
                cryst = None
            except Exception:
                cryst = None

        if cryst is None or cryst.GetNumConformers() == 0:
            return None

        pose = Chem.RemoveHs(pose_mol, sanitize=False)
        try:
            Chem.SanitizeMol(pose)
        except Exception:
            pass
        if pose.GetNumConformers() == 0:
            return None

        n_smaller = min(pose.GetNumAtoms(), cryst.GetNumAtoms())
        mcs = rdFMCS.FindMCS(
            [pose, cryst],
            timeout=10,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            completeRingsOnly=False,
            matchValences=False,
        )

        if mcs.numAtoms < 3 or mcs.numAtoms < 0.6 * n_smaller:
            return None

        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        if mcs_mol is None:
            return None

        pose_matches  = pose.GetSubstructMatches(mcs_mol,  uniquify=False)
        cryst_matches = cryst.GetSubstructMatches(mcs_mol, uniquify=False)
        if not pose_matches or not cryst_matches:
            return None

        pc, cc = pose.GetConformer(), cryst.GetConformer()

        def _rmsd(pm, cm):
            sq = sum(
                (pc.GetAtomPosition(pi).x - cc.GetAtomPosition(ci).x) ** 2 +
                (pc.GetAtomPosition(pi).y - cc.GetAtomPosition(ci).y) ** 2 +
                (pc.GetAtomPosition(pi).z - cc.GetAtomPosition(ci).z) ** 2
                for pi, ci in zip(pm, cm)
            )
            return float(np.sqrt(sq / len(pm)))

        return min(_rmsd(pm, cm) for pm in pose_matches for cm in cryst_matches)

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  POSEVIEW REST API
# ══════════════════════════════════════════════════════════════════════════════

_PP_BASE          = "https://proteins.plus/api/v2/"
_PP_UPLOAD        = _PP_BASE + "molecule_handler/upload/"
_PP_UPLOAD_JOBS   = _PP_BASE + "molecule_handler/upload/jobs/"
_PP_POSEVIEW      = _PP_BASE + "poseview/"
_PP_POSEVIEW_JOBS = _PP_BASE + "poseview/jobs/"

_PP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://proteins.plus/",
    "Origin":  "https://proteins.plus",
}

_PP_PROTEIN_CACHE: dict = {}


def _pp_poll(job_id: str, poll_url: str, poll_interval: int = 2,
             max_polls: int = 60) -> dict:
    """Poll a ProteinsPlus job until it leaves pending/running state."""
    import requests
    job    = requests.get(poll_url + job_id + "/", headers=_PP_HEADERS, timeout=15).json()
    status = str(job.get("status", "")).lower()
    polls  = 0
    while status in ("pending", "running", "processing", "queued", ""):
        if polls >= max_polls:
            raise RuntimeError(
                f"Job {job_id} still '{status}' after "
                f"{max_polls * poll_interval} s"
            )
        time.sleep(poll_interval)
        polls += 1
        job    = requests.get(poll_url + job_id + "/", headers=_PP_HEADERS, timeout=15).json()
        status = str(job.get("status", "")).lower()
    return job


def _prepare_pdb_for_poseview(receptor_pdb: str) -> str:
    """Write a minimal, clean PDB suitable for PoseView's CDK-based parser."""
    rec_dir = os.path.dirname(os.path.abspath(receptor_pdb))
    candidates = [
        os.path.join(rec_dir, "receptor_atoms.pdb"),
        receptor_pdb,
    ]
    source = next((p for p in candidates if os.path.exists(p) and os.path.getsize(p) > 100), receptor_pdb)

    out = os.path.join(rec_dir, "receptor_pv_clean.pdb")
    try:
        kept   = []
        serial = 0
        with open(source) as f:
            for line in f:
                rec = line[:6].strip()
                if rec not in ("ATOM", "HETATM", "TER", "END"):
                    continue
                if rec in ("ATOM", "HETATM"):
                    atom_name = line[12:16].strip()
                    element   = line[76:78].strip() if len(line) > 76 else ""
                    if element.upper() == "H" or (not element and atom_name.startswith("H")):
                        continue
                    serial += 1
                    line = f"{line[:6]}{serial:5d}{line[11:]}"
                kept.append(line if line.endswith("\n") else line + "\n")

        if not kept:
            return receptor_pdb

        if not any(l.startswith("END") for l in kept):
            kept.append("END\n")

        with open(out, "w") as f:
            f.writelines(kept)

        if os.path.exists(out) and os.path.getsize(out) > 100:
            return out
    except Exception:
        pass

    return receptor_pdb


def call_poseview_v1(receptor_pdb: str, pose_sdf: str) -> tuple:
    """
    Submit receptor PDB + docked pose SDF to PoseView v1 REST API.
    Returns (svg_bytes, error_string) — one will be None.
    """
    import requests

    last_error = "Unknown error"
    rec_to_send = _prepare_pdb_for_poseview(receptor_pdb)

    for attempt in range(1, _PV_MAX_RETRIES + 1):
        if attempt > 1:
            time.sleep(_PV_RETRY_DELAY)

        try:
            with open(rec_to_send) as rf, open(pose_sdf) as lf:
                r = requests.post(
                    _PP_POSEVIEW,
                    files={
                        "protein_file": ("receptor.pdb", rf, "chemical/x-pdb"),
                        "ligand_file":  ("ligand.sdf",   lf, "chemical/x-mdl-sdfile"),
                    },
                    headers=_PP_HEADERS,
                    timeout=30,
                )
            r.raise_for_status()
            pv_data   = r.json()
            pv_job_id = pv_data.get("job_id") or pv_data.get("id")
            if not pv_job_id:
                last_error = f"PoseView submission: no job_id in {pv_data}"
                continue
        except Exception as e:
            last_error = f"PoseView submission failed (attempt {attempt}): {e}"
            continue

        try:
            pv_job = _pp_poll(pv_job_id, _PP_POSEVIEW_JOBS)
            status = str(pv_job.get("status", "")).lower()
        except RuntimeError as e:
            last_error = str(e)
            continue
        except Exception as e:
            last_error = f"Polling error (attempt {attempt}): {e}"
            continue

        if status in ("failed", "failure", "error"):
            last_error = (
                f"PoseView rejected job (attempt {attempt}). "
                f"Full response: {pv_job}"
            )
            continue
        if status != "success":
            last_error = (
                f"Unexpected status '{status}' (attempt {attempt}). "
                f"Full response: {pv_job}"
            )
            continue

        img_url = pv_job.get("image")
        if not img_url:
            last_error = (
                f"Job succeeded but 'image' key missing. "
                f"Keys: {list(pv_job.keys())}"
            )
            continue
        try:
            resp = requests.get(img_url, headers=_PP_HEADERS, timeout=20)
            resp.raise_for_status()
            return resp.content, None
        except Exception as e:
            last_error = f"SVG download failed (attempt {attempt}): {e}"
            continue

    return None, last_error


def warm_poseview_cache(receptor_pdb: str) -> tuple:
    """No-op — kept for API compatibility with app.py."""
    return True, "Direct POST mode — no pre-upload needed"


def clear_poseview_cache():
    """Clear receptor upload cache when a new receptor is prepared."""
    _PP_PROTEIN_CACHE.clear()


def call_poseview2_ref(pdb_code: str, ligand_id: str) -> tuple:
    """
    Submit co-crystal reference job to PoseView2 REST API.
    Returns (svg_bytes, error_string) — one will be None.
    """
    import requests

    _SUBMIT = "https://proteins.plus/api/poseview2_rest"
    last_error = "Unknown error"

    for attempt in range(1, _PV_MAX_RETRIES + 1):
        if attempt > 1:
            time.sleep(_PV_RETRY_DELAY)

        try:
            r = requests.post(
                _SUBMIT,
                json={"poseview2": {
                    "pdbCode": pdb_code.strip().lower(),
                    "ligand":  ligand_id.strip(),
                }},
                headers={
                    "Accept":       "application/json",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            data = r.json()
            if r.status_code not in (200, 202):
                last_error = (
                    f"Submission failed ({r.status_code}), attempt {attempt}: "
                    f"{data}"
                )
                continue
            location = data.get("location", "")
            if not location:
                last_error = f"API returned no job location. Response: {data}"
                continue
        except Exception as e:
            last_error = f"Submission error (attempt {attempt}): {e}"
            continue

        job_failed = False
        for poll_i in range(_PV_POLL_ATTEMPTS):
            time.sleep(2)
            try:
                poll        = requests.get(
                    location,
                    headers={"Accept": "application/json"},
                    timeout=15,
                ).json()
                status_code = poll.get("status_code")

                if status_code == 200:
                    svg_url = poll.get("result_svg", "")
                    if not svg_url:
                        last_error = (
                            f"Job finished but result_svg is empty. "
                            f"Full response: {poll}"
                        )
                        job_failed = True
                        break
                    resp = requests.get(svg_url, timeout=20)
                    resp.raise_for_status()
                    return resp.content, None

                elif status_code == 202:
                    continue

                else:
                    last_error = (
                        f"Unexpected poll status {status_code} "
                        f"(attempt {attempt}). Full response: {poll}"
                    )
                    job_failed = True
                    break

            except Exception as e:
                last_error = (
                    f"Polling error (attempt {attempt}, poll {poll_i+1}): {e}"
                )
                continue

        if not job_failed:
            last_error = (
                f"Timed out after {_PV_POLL_ATTEMPTS * 2} s "
                f"(attempt {attempt})"
            )

    return None, last_error


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def svg_to_png(svg_bytes: bytes):
    """Convert SVG bytes → PNG bytes via cairosvg. Returns None if unavailable."""
    try:
        import cairosvg
        return cairosvg.svg2png(bytestring=svg_bytes, scale=2, background_color="white")
    except Exception:
        return None


def stamp_png(png_bytes: bytes, text: str) -> bytes:
    """Burn a centred label pill into the bottom of a PNG."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io as _io

        img  = Image.open(_io.BytesIO(png_bytes)).convert("RGBA")
        draw = ImageDraw.Draw(img)

        font = None
        for fp, sz in [
            ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 28),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",                 28),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",        26),
        ]:
            try:
                font = ImageFont.truetype(fp, sz)
                break
            except Exception:
                pass
        if font is None:
            font = ImageFont.load_default()

        bbox   = draw.textbbox((0, 0), text, font=font, anchor="lt")
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad_x, pad_y = 36, 16
        pill_w = tw + pad_x * 2
        pill_h = th + pad_y * 2
        pill_r = pill_h // 2
        px     = (img.width  - pill_w) // 2
        py_    =  img.height - pill_h  - 28

        draw.rounded_rectangle(
            [px, py_, px + pill_w, py_ + pill_h],
            radius=pill_r,
            fill=(232, 232, 232, 230),
        )
        draw.text(
            (px + pill_w // 2, py_ + pill_h // 2),
            text,
            font=font,
            fill=(26, 26, 26, 255),
            anchor="mm",
        )
        buf = _io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return png_bytes


# ══════════════════════════════════════════════════════════════════════════════
#  POSEVIEW DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_poseview() -> dict:
    """Test the PoseView API with a known-good minimal PDB + SDF from RCSB."""
    import requests

    result = {
        "server_reachable": False,
        "upload_ok":        False,
        "poseview_ok":      False,
        "status":           "",
        "job_response":     {},
        "image_url":        "",
        "error":            "",
        "log":              [],
    }
    log = result["log"]

    try:
        r = requests.get("https://proteins.plus/api/v2/", timeout=10)
        r.raise_for_status()
        result["server_reachable"] = True
        log.append(f"✓ Server reachable (HTTP {r.status_code})")
    except Exception as e:
        result["error"] = f"Server unreachable: {e}"
        log.append(f"✗ Server unreachable: {e}")
        return result

    try:
        r = requests.post(
            _PP_UPLOAD, data={"pdb_code": "4agn"}, timeout=30
        )
        r.raise_for_status()
        job_id = r.json().get("job_id")
        log.append(f"✓ Upload job submitted: {job_id}")
        job = _pp_poll(job_id, _PP_UPLOAD_JOBS, poll_interval=1, max_polls=30)
        log.append(f"✓ Upload job: {job.get('status')}")

        protein_id   = job["output_protein"]
        protein_json = requests.get(
            _PP_BASE + "molecule_handler/proteins/" + protein_id + "/",
            timeout=15,
        ).json()
        pdb_text = protein_json["file_string"]

        ligand_id   = protein_json["ligand_set"][0]
        ligand_json = requests.get(
            _PP_BASE + "molecule_handler/ligands/" + ligand_id + "/",
            timeout=15,
        ).json()
        sdf_text = ligand_json["file_string"]
        log.append(
            f"✓ Got protein ({len(pdb_text)} chars) "
            f"+ ligand {ligand_json.get('name')} ({len(sdf_text)} chars)"
        )
        result["upload_ok"] = True
    except Exception as e:
        result["error"] = f"MoleculeHandler step failed: {e}"
        log.append(f"✗ MoleculeHandler failed: {e}")
        return result

    try:
        import io as _io
        r = requests.post(
            _PP_POSEVIEW,
            files={
                "protein_file": ("test.pdb", _io.StringIO(pdb_text), "chemical/x-pdb"),
                "ligand_file":  ("test.sdf", _io.StringIO(sdf_text), "chemical/x-mdl-sdfile"),
            },
            timeout=30,
        )
        r.raise_for_status()
        pv_job_id = r.json().get("job_id")
        log.append(f"✓ PoseView job submitted: {pv_job_id}")

        pv_job = _pp_poll(pv_job_id, _PP_POSEVIEW_JOBS, poll_interval=2, max_polls=30)
        result["job_response"] = pv_job
        result["status"]       = pv_job.get("status", "")
        log.append(f"✓ PoseView job status: {result['status']}")

        if result["status"] == "success":
            result["image_url"]   = pv_job.get("image", "")
            result["poseview_ok"] = True
            log.append(f"✓ Image URL: {result['image_url']}")
        else:
            result["error"] = (
                f"PoseView returned '{result['status']}'. "
                f"Full response: {pv_job}"
            )
            log.append(f"✗ {result['error']}")
    except Exception as e:
        result["error"] = f"PoseView step failed: {e}"
        log.append(f"✗ PoseView failed: {e}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM 2D INTERACTION DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

import math as _math

_ITYPE_PRIORITY = [
    "metal", "ionic", "halogen", "hbond_to_halogen",
    "hbond", "pi_pi", "cation_pi", "hydrophobic",
]

_CLR_HBOND   = "#1a7a1a"
_CLR_PIPI    = "#e200e8"
_CLR_HYDRO   = "#2287ff"
_CLR_IONIC   = "#aa0077"
_CLR_METAL   = "#cc8800"
_CLR_HAL     = "#cc2277"
_CLR_HBXHAL  = "#6633aa"

_RES_CIRCLE = {
    "hbond":            dict(fill="#80dd80", opacity=0.2),
    "hbond_to_halogen": dict(fill="#80dd80", opacity=0.2),
    "pi_pi":            dict(fill="#e200e8", opacity=0.2),
    "cation_pi":        dict(fill="#e200e8", opacity=0.2),
    "hydrophobic":      dict(fill="#2287ff", opacity=0.2),
    "ionic":            dict(fill="#ffaae0", opacity=0.2),
    "metal":            dict(fill="#ffe080", opacity=0.2),
    "halogen":          dict(fill="#ffb0d0", opacity=0.2),
}

_LBL_CLR = {
    "hbond":            _CLR_HBOND,
    "hbond_to_halogen": _CLR_HBOND,
    "pi_pi":            _CLR_PIPI,
    "cation_pi":        _CLR_PIPI,
    "hydrophobic":      _CLR_HYDRO,
    "ionic":            _CLR_IONIC,
    "metal":            _CLR_METAL,
    "halogen":          _CLR_HAL,
}

_LINE_CLR = {
    "hbond":            _CLR_HBOND,
    "hbond_to_halogen": _CLR_HBXHAL,
    "pi_pi":            _CLR_PIPI,
    "cation_pi":        _CLR_PIPI,
    "ionic":            _CLR_IONIC,
    "metal":            _CLR_METAL,
    "halogen":          _CLR_HAL,
    # hydrophobic: NO LINE
}

_ATOM_CLR = {
    "C":"#1a1a1a", "N":"#1a5fa8", "O":"#cc2222",
    "S":"#c8a800", "P":"#e07000", "F":"#1a7a1a",
    "CL":"#1a7a1a", "BR":"#8b2500", "I":"#5c2d8a", "H":"#555555",
}
_AROM_DOT_CLR = "#1a7a1a"

_METALS_SET = {"MG","ZN","CA","MN","FE","CU","CO","NI","CD","HG","NA","K"}

# ── Aromatic residues: amino acids + heme porphyrin ──────────────────────────
_AROM_ATOMS = {"PHE","TYR","TRP","HIS","HEM","HEC","HEA","HEB","HDD","HDM"}

# ── Aromatic atom names: amino acid rings + heme porphyrin macrocycle ─────────
_AROM_ATOM_NAMES = {
    # Amino acid aromatic ring atoms
    "CG","CD1","CD2","CE1","CE2","CZ",
    "ND1","NE2","CE3","CZ2","CZ3","CH2",
    # Heme porphyrin macrocycle — meso and pyrrole carbons
    "C1","C2","C3","C4","C5","C6","C7","C8",
    "C10","C11","C12","C13","C14","C15","C16","C17",
    "C19","C20",
    "CA","CB",        # pyrrole alpha/beta (some naming conventions)
    "CAA","CAB","CAC","CAD",   # alpha carbons (alt convention)
    "CBA","CBB","CBC","CBD",   # beta carbons  (alt convention)
    "C2A","C3A","C4A",         # pyrrole A ring
    "C2B","C3B","C4B",         # pyrrole B ring
    "C2C","C3C","C4C",         # pyrrole C ring
    "C2D","C3D","C4D",         # pyrrole D ring
    "CHA","CHB","CHC","CHD",   # meso carbons
}

# ── Hydrophobic residues: amino acids + heme ─────────────────────────────────
_HYDR_BASE = {"ALA","VAL","ILE","LEU","MET","PHE","TRP","PRO","GLY","TYR","HIS"}
_HYDR_EXTENDED = _HYDR_BASE | HEME_RESNAMES


# ──────────────────────────────────────────────────────────────────────────────
#  INTERACTION DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def _get_aromatic_ring_data(mol, conf):
    import numpy as np
    results = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) not in (5, 6): continue
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring): continue
        coords   = np.array([[conf.GetAtomPosition(i).x,
                               conf.GetAtomPosition(i).y,
                               conf.GetAtomPosition(i).z] for i in ring])
        centroid = coords.mean(axis=0)
        v1 = coords[1]-coords[0]; v2 = coords[2]-coords[0]
        n  = np.cross(v1,v2); nl = np.linalg.norm(n)
        if nl > 0: n /= nl
        results.append((centroid, n, list(ring)))
    return results


def _detect_all_interactions(lig_mol_3d, receptor_pdb: str,
                              cutoff: float = 4.5) -> list:
    """
    Detect protein-ligand interactions with proper geometry criteria.
    Includes heme Fe coordination (metal), H-bonds to heme N/O,
    π-stacking with porphyrin, and hydrophobic contacts with heme carbons.
    """
    import numpy as np
    from prody import parsePDB

    rec = parsePDB(receptor_pdb)
    if rec is None:
        return []

    rc  = np.array(rec.getCoords(),  dtype=float)
    rrn = rec.getResnames()
    rch = rec.getChids()
    rri = rec.getResnums()
    ran = rec.getNames()
    rel = rec.getElements()

    conf = lig_mol_3d.GetConformer()
    nl   = lig_mol_3d.GetNumAtoms()
    lxyz = np.array([[conf.GetAtomPosition(i).x,
                      conf.GetAtomPosition(i).y,
                      conf.GetAtomPosition(i).z] for i in range(nl)], dtype=float)
    latom = [lig_mol_3d.GetAtomWithIdx(i) for i in range(nl)]
    lel   = [a.GetSymbol().upper() for a in latom]
    lchg  = [a.GetFormalCharge() for a in latom]

    _VDW = {"H":1.20,"C":1.70,"N":1.55,"O":1.52,"S":1.80,"P":1.80,
            "F":1.47,"CL":1.75,"BR":1.85,"I":1.98,"SE":1.90}

    nr = len(rc)
    r_el  = [rel[j].strip().upper() if rel[j] and rel[j].strip()
             else ran[j][:1].upper() for j in range(nr)]
    r_el_arr = np.array(r_el)
    r_rn  = np.array([rrn[j].strip() for j in range(nr)])
    r_an  = np.array([ran[j].strip() for j in range(nr)])
    r_ch  = np.array([rch[j].strip() for j in range(nr)])
    r_ri  = np.array([int(rri[j])    for j in range(nr)])

    HYDL = {"C","S","CL","BR","I","F"}
    LIG_ACCEPTOR_EL = {"N","O","F","S"}

    lig_is_acceptor    = np.array([lel[i] in LIG_ACCEPTOR_EL for i in range(nl)])
    lig_is_hydrophobic = np.array([lel[i] in HYDL for i in range(nl)])

    def _is_lig_donor(i):
        a = latom[i]
        if lel[i] not in ("N","O","S","F"): return False
        for nb in a.GetNeighbors():
            if nb.GetAtomicNum() == 1: return True
        return a.GetTotalNumHs() > 0

    lig_is_donor = np.array([_is_lig_donor(i) for i in range(nl)])

    def _angles_at_b(a_pts, b_pt, c_pt):
        va = a_pts - b_pt
        vc = c_pt  - b_pt
        na = np.linalg.norm(va, axis=1)
        nc = float(np.linalg.norm(vc))
        if nc < 1e-8: return np.zeros(len(a_pts))
        cos_t = (va @ vc) / (na * nc + 1e-12)
        cos_t = np.clip(cos_t, -1.0, 1.0)
        return np.degrees(np.arccos(cos_t))

    def _angle3(a, b, c):
        va = a - b; vc = c - b
        na = np.linalg.norm(va); nc = np.linalg.norm(vc)
        if na < 1e-8 or nc < 1e-8: return 0.0
        return float(np.degrees(np.arccos(
            np.clip(np.dot(va, vc) / (na * nc), -1.0, 1.0))))

    h_mask    = r_el_arr == "H"
    h_idx     = np.where(h_mask)[0]
    heavy_idx = np.where(~h_mask)[0]

    h_to_heavy = {}
    if len(h_idx) and len(heavy_idx):
        h_coords  = rc[h_idx]
        hv_coords = rc[heavy_idx]
        diff      = h_coords[:, None, :] - hv_coords[None, :, :]
        dists_hh  = np.linalg.norm(diff, axis=2)
        closest   = np.argmin(dists_hh, axis=1)
        min_d     = dists_hh[np.arange(len(h_idx)), closest]
        for k, (hi, ci, md) in enumerate(zip(h_idx, closest, min_d)):
            if md < 1.15:
                h_to_heavy[int(hi)] = int(heavy_idx[ci])

    heavy_to_h = {}
    for hj, hk in h_to_heavy.items():
        heavy_to_h.setdefault(hk, []).append(hj)

    PROT_DONOR_ATOMS = {
        "N","OG","OG1","OH","SG","NZ","NH1","NH2","NE",
        "ND1","NE2","NE1","ND2",
    }
    PROT_ACCEPTOR_ATOMS = {
        "O","OD1","OD2","OE1","OE2","OG","OG1","OH",
        "ND1","NE2","SD",
    }

    HBOND_DA_MAX  = 3.5
    HBOND_ANG_MIN = 120.0
    HBOND_PROXY   = 90.0

    results        = []
    hbond_residues = set()

    # ── H-BOND ────────────────────────────────────────────────────────────────
    polar_mask = np.array([
        r_el_arr[j] in ("N","O","S") and r_el_arr[j] != "H"
        and r_rn[j] not in ("HOH","WAT","DOD")
        for j in range(nr)
    ])
    polar_idx = np.where(polar_mask)[0]

    for j in polar_idx:
        an = r_an[j]; ch = r_ch[j]; ri = int(r_ri[j]); el = r_el[j]
        rp = rc[j]
        key = (ch, ri)
        if key in hbond_residues:
            continue

        dists_j = np.linalg.norm(lxyz - rp, axis=1)

        if an in PROT_DONOR_ATOMS:
            cand = np.where(lig_is_acceptor & (dists_j <= HBOND_DA_MAX))[0]
            if len(cand):
                hs = heavy_to_h.get(j, [])
                for i in cand:
                    d_DA = float(dists_j[i])
                    if hs:
                        best = max(
                            _angle3(rp, rc[hj], lxyz[i]) for hj in hs
                        )
                        if best < HBOND_ANG_MIN: continue
                    else:
                        nbs_i = [nb.GetIdx() for nb in latom[i].GetNeighbors()
                                 if nb.GetAtomicNum() != 1 and nb.GetIdx() < nl]
                        if nbs_i:
                            if _angle3(rp, lxyz[i], lxyz[nbs_i[0]]) < HBOND_PROXY:
                                continue
                    hbond_residues.add(key)
                    results.append(dict(resname=r_rn[j],chain=ch,resid=ri,
                        itype="hbond",distance=round(d_DA,1),lig_atom_idx=int(i),
                        prot_el=el,is_donor=True,ring_atom_indices=None))
                    break

        if key in hbond_residues: continue

        if an in PROT_ACCEPTOR_ATOMS:
            cand = np.where(lig_is_donor & (dists_j <= HBOND_DA_MAX))[0]
            if len(cand):
                for i in cand:
                    d_DA = float(dists_j[i])
                    lig_hs = [nb.GetIdx() for nb in latom[i].GetNeighbors()
                              if nb.GetAtomicNum() == 1 and nb.GetIdx() < nl]
                    if lig_hs:
                        best = max(_angle3(lxyz[i], lxyz[hi], rp)
                                   for hi in lig_hs)
                        if best < HBOND_ANG_MIN: continue
                    else:
                        nbs_i = [nb.GetIdx() for nb in latom[i].GetNeighbors()
                                 if nb.GetAtomicNum() != 1 and nb.GetIdx() < nl]
                        if nbs_i:
                            if _angle3(rp, lxyz[i], lxyz[nbs_i[0]]) < HBOND_PROXY:
                                continue
                    hbond_residues.add(key)
                    results.append(dict(resname=r_rn[j],chain=ch,resid=ri,
                        itype="hbond",distance=round(d_DA,1),lig_atom_idx=int(i),
                        prot_el=el,is_donor=False,ring_atom_indices=None))
                    break

    # ── HYDROPHOBIC, IONIC, METAL ─────────────────────────────────────────────
    for j in range(nr):
        rn = r_rn[j]; ch = r_ch[j]; ri = int(r_ri[j]); el = r_el[j]; rp = rc[j]
        dists_j = np.linalg.norm(lxyz - rp, axis=1)
        md = float(dists_j.min()); mi = int(dists_j.argmin())
        if md > max(cutoff + 1.0, 5.6): continue

        # Hydrophobic: extended to include heme residue names
        if el in {"C","S","CL","BR","I"} and rn in _HYDR_EXTENDED:
            cand = np.where(lig_is_hydrophobic & (dists_j < cutoff))[0]
            if len(cand):
                i = int(cand[0])
                results.append(dict(resname=rn,chain=ch,resid=ri,
                    itype="hydrophobic",distance=round(float(dists_j[i]),1),
                    lig_atom_idx=i,prot_el=el,is_donor=False,ring_atom_indices=None))

        if rn in {"ASP","GLU"} and el == "O":
            for i in range(nl):
                if lchg[i] > 0 and float(dists_j[i]) < 4.0:
                    results.append(dict(resname=rn,chain=ch,resid=ri,
                        itype="ionic",distance=round(float(dists_j[i]),1),
                        lig_atom_idx=i,prot_el=el,is_donor=False,ring_atom_indices=None)); break
        if rn in {"LYS","ARG"} and el == "N":
            for i in range(nl):
                if lchg[i] < 0 and float(dists_j[i]) < 4.0:
                    results.append(dict(resname=rn,chain=ch,resid=ri,
                        itype="ionic",distance=round(float(dists_j[i]),1),
                        lig_atom_idx=i,prot_el=el,is_donor=True,ring_atom_indices=None)); break

        _is_heme_fe = (r_rn[j] in HEME_RESNAMES
                       and r_an[j].strip().upper() == "FE")
        if rn.strip().upper() in _METALS_SET or el in _METALS_SET or _is_heme_fe:
            if md < 2.8:
                results.append(dict(resname=rn,chain=ch,resid=ri,
                    itype="metal",distance=round(md,1),lig_atom_idx=mi,
                    prot_el=el,is_donor=False,ring_atom_indices=None))

    # ── π-π AND CATION-π (amino acids + heme porphyrin) ──────────────────────
    lr = _get_aromatic_ring_data(lig_mol_3d, conf)
    if lr:
        arom_mask = np.array([
            r_rn[j] in _AROM_ATOMS and r_an[j] in _AROM_ATOM_NAMES
            and r_el[j] == "C" for j in range(nr)
        ])
        arom_idx = np.where(arom_mask)[0]
        for j in arom_idx:
            rp = rc[j]
            for lc, _, ring_idxs in lr:
                d = float(np.linalg.norm(lc - rp))
                if d < 5.5:
                    results.append(dict(resname=r_rn[j],chain=r_ch[j],resid=int(r_ri[j]),
                        itype="pi_pi",distance=round(d,1),
                        lig_atom_idx=ring_idxs[0],prot_el="C",is_donor=False,
                        ring_atom_indices=ring_idxs)); break

        cat_mask = np.array([r_rn[j] in {"LYS","ARG"} and r_el[j] == "N"
                             for j in range(nr)])
        for j in np.where(cat_mask)[0]:
            rp = rc[j]
            for lc, _, ring_idxs in lr:
                d = float(np.linalg.norm(lc - rp))
                if d < 5.0:
                    results.append(dict(resname=r_rn[j],chain=r_ch[j],resid=int(r_ri[j]),
                        itype="cation_pi",distance=round(d,1),
                        lig_atom_idx=ring_idxs[0],prot_el="N",is_donor=True,
                        ring_atom_indices=ring_idxs)); break

    # ── HALOGEN BOND ──────────────────────────────────────────────────────────
    _XD  = {17:"CL", 35:"BR", 53:"I"}
    _XA_el = {"O","N","S","F"}

    arom_C_mask = np.array([
        r_rn[j] in _AROM_ATOMS and r_an[j] in _AROM_ATOM_NAMES and r_el[j] == "C"
        for j in range(nr)
    ])
    xb_acc_mask = np.array([r_el[j] in _XA_el for j in range(nr)]) | arom_C_mask

    for i in range(nl):
        ano = latom[i].GetAtomicNum()
        if ano not in _XD: continue
        xel = _XD[ano]; xp = lxyz[i]; vdw_x = _VDW.get(xel, 1.80)
        c_nb = next((nb.GetIdx() for nb in latom[i].GetNeighbors()
                     if nb.GetAtomicNum() == 6 and nb.GetIdx() < nl), None)
        if c_nb is None: continue
        c_pos = lxyz[c_nb]

        max_d = vdw_x + 1.98 + 0.5
        cand_mask = xb_acc_mask & (np.linalg.norm(rc - xp, axis=1) <= max_d)
        cand_idx  = np.where(cand_mask)[0]
        if not len(cand_idx): continue

        for j in cand_idx:
            ael = r_el[j]; ap = rc[j]
            vdw_sum = vdw_x + _VDW.get(ael, 1.70) + 0.5
            d = float(np.linalg.norm(xp - ap))
            if d > vdw_sum: continue
            ang1 = _angle3(c_pos, xp, ap)
            if ang1 < 140.0: continue
            r_nbs = [k for k in range(nr)
                     if k != j and r_el[k] != "H"
                     and float(np.linalg.norm(rc[k] - ap)) < 1.85]
            if r_nbs:
                if _angle3(xp, ap, rc[r_nbs[0]]) < 90.0: continue
            results.append(dict(resname=r_rn[j],chain=r_ch[j],resid=int(r_ri[j]),
                itype="halogen",distance=round(d,1),lig_atom_idx=i,
                prot_el=ael,is_donor=False,ring_atom_indices=None))

    # ── H-BOND TO HALOGEN ─────────────────────────────────────────────────────
    _HA = {9:"F", 17:"CL", 35:"BR", 53:"I"}
    _HD = {"O","N","S"}

    if len(h_idx):
        h_coords = rc[h_idx]
        for i in range(nl):
            ano = latom[i].GetAtomicNum()
            if ano not in _HA: continue
            xel = _HA[ano]; xp = lxyz[i]; vdw_x = _VDW.get(xel, 1.80)
            c_nb2 = next((nb.GetIdx() for nb in latom[i].GetNeighbors()
                          if nb.GetIdx() < nl), None)
            if c_nb2 is None: continue
            c_pos2 = lxyz[c_nb2]
            dhx_all = np.linalg.norm(h_coords - xp, axis=1)
            close_h = np.where(dhx_all <= _VDW["H"] + vdw_x)[0]
            for kk in close_h:
                hj = int(h_idx[kk]); hp = rc[hj]; dhx = float(dhx_all[kk])
                pk = h_to_heavy.get(hj)
                if pk is None or r_el[pk] not in _HD: continue
                dp = rc[pk]
                if _angle3(dp, hp, xp) < 120.0: continue
                if not (70.0 <= _angle3(c_pos2, xp, hp) <= 120.0): continue
                results.append(dict(resname=r_rn[hj],chain=r_ch[hj],resid=int(r_ri[hj]),
                    itype="hbond_to_halogen",distance=round(dhx,1),
                    lig_atom_idx=i,prot_el="N",is_donor=True,ring_atom_indices=None))

    return results


def _deduplicate_interactions(interactions: list) -> list:
    priority={t:i for i,t in enumerate(_ITYPE_PRIORITY)}
    best:dict={}
    for ix in interactions:
        key=(ix["chain"],ix["resid"])
        if key not in best: best[key]=ix
        else:
            pn=priority.get(ix["itype"],99); po=priority.get(best[key]["itype"],99)
            if pn<po or (pn==po and ix["distance"]<best[key]["distance"]): best[key]=ix
    return list(best.values())


def _enrich_with_res_xyz(interactions, mol3d, receptor_pdb):
    import numpy as np
    from prody import parsePDB
    try:
        rec = parsePDB(receptor_pdb)
        if rec is None: return
        rc  = rec.getCoords()
        rch = rec.getChids()
        rri = rec.getResnums()
        conf  = mol3d.GetConformer()
        n_lig = mol3d.GetNumAtoms()
        lxyz  = np.array([[conf.GetAtomPosition(i).x,
                            conf.GetAtomPosition(i).y,
                            conf.GetAtomPosition(i).z] for i in range(n_lig)])
    except Exception:
        return

    for ix in interactions:
        ch  = ix.get("chain", "")
        ri  = ix.get("resid", 0)
        lai = ix.get("lig_atom_idx", 0)
        lai = min(lai, n_lig - 1)

        ring_idxs = ix.get("ring_atom_indices") or []
        if ring_idxs:
            valid = [i for i in ring_idxs if 0 <= i < n_lig]
            if valid:
                lig_anchor = lxyz[valid].mean(axis=0)
            else:
                lig_anchor = lxyz[lai]
        else:
            lig_anchor = lxyz[lai]
        ix["lig_anchor_xyz"] = lig_anchor.tolist()

        res_idx = [j for j in range(len(rc))
                   if rch[j].strip() == ch and int(rri[j]) == ri]
        if not res_idx:
            ix["res_xyz"] = lig_anchor.tolist()
            continue

        res_coords = rc[res_idx]
        dists = np.linalg.norm(res_coords - lig_anchor, axis=1)
        closest_local = int(dists.argmin())
        ix["res_xyz"] = rc[res_idx[closest_local]].tolist()


def _compute_svg_coords(mol2d, cx, cy, target_size=280):
    from rdkit.Chem import rdDepictor
    if mol2d.GetNumConformers()==0: rdDepictor.Compute2DCoords(mol2d)
    conf=mol2d.GetConformer(); n=mol2d.GetNumAtoms()
    if n==0: return {}
    xs=[conf.GetAtomPosition(i).x for i in range(n)]
    ys=[conf.GetAtomPosition(i).y for i in range(n)]
    span=max(max(xs)-min(xs),max(ys)-min(ys),0.01)
    sc=target_size/span; mx=(min(xs)+max(xs))/2; my=(min(ys)+max(ys))/2
    return {i:(cx+(xs[i]-mx)*sc, cy-(ys[i]-my)*sc) for i in range(n)}


def _ring_centroid_2d(ring_atom_indices, svg_coords):
    xs=[svg_coords[i][0] for i in ring_atom_indices if i in svg_coords]
    ys=[svg_coords[i][1] for i in ring_atom_indices if i in svg_coords]
    if not xs: return None, None
    return sum(xs)/len(xs), sum(ys)/len(ys)


def _ring_centroid_from_atom(mol2d, atom_idx_2d, svg_coords):
    ring_info = mol2d.GetRingInfo()
    best_ring = None
    best_size = 999
    for ring in ring_info.AtomRings():
        if atom_idx_2d in ring:
            if mol2d.GetAtomWithIdx(ring[0]).GetIsAromatic():
                if len(ring) < best_size:
                    best_ring = ring
                    best_size = len(ring)
    if best_ring is None:
        return None, None
    xs = [svg_coords[i][0] for i in best_ring if i in svg_coords]
    ys = [svg_coords[i][1] for i in best_ring if i in svg_coords]
    if not xs:
        return None, None
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _place_residues_no_cross(interactions, svg_coords, cx, cy, R=210):
    if not interactions: return []

    items = []
    for ix in interactions:
        if ix.get("ring_atom_indices"):
            ax, ay = _ring_centroid_2d(ix["ring_atom_indices"], svg_coords)
            if ax is None:
                ax, ay = svg_coords.get(ix.get("lig_atom_idx", 0), (cx, cy))
        else:
            ai = ix.get("lig_atom_idx", 0)
            ax, ay = svg_coords.get(ai, (cx, cy))
        angle = _math.atan2(ay - cy, ax - cx)
        items.append({**ix, "angle": angle, "anchor_angle": angle})

    items.sort(key=lambda x: x["angle"])
    n = len(items)
    if n == 1:
        a = items[0]["angle"]
        return [{**items[0],
                 "bx": cx + R * _math.cos(a),
                 "by": cy + R * _math.sin(a),
                 "slot_angle": a}]

    for i in range(n):
        for j in range(i + 1, n):
            if abs(items[j]["angle"] - items[i]["angle"]) < 0.001:
                items[j]["angle"] += 0.05 * (j - i)

    min_dist_px = 78.0

    for _ in range(500):
        delta = [0.0] * n
        any_overlap = False

        for i in range(n):
            for j in range(i + 1, n):
                xi = cx + R * _math.cos(items[i]["angle"])
                yi = cy + R * _math.sin(items[i]["angle"])
                xj = cx + R * _math.cos(items[j]["angle"])
                yj = cy + R * _math.sin(items[j]["angle"])
                dx, dy = xj - xi, yj - yi
                dist = _math.sqrt(dx * dx + dy * dy)

                if dist < min_dist_px:
                    overlap_px = min_dist_px - dist
                    push_angle = overlap_px / max(R, 1.0)
                    delta[i] -= push_angle * 0.5
                    delta[j] += push_angle * 0.5
                    any_overlap = True

        for i in range(n):
            items[i]["angle"] += delta[i]

        if not any_overlap:
            break

    result = []
    for item in items:
        a = item["angle"]
        bx = cx + R * _math.cos(a)
        by = cy + R * _math.sin(a)
        result.append({
            **item,
            "bx": bx,
            "by": by,
            "slot_angle": a,
        })
    return result


# ── Radial-collapse layout helpers ───────────────────────────────────────────

def _rl_ligand_center(svg_coords):
    pts = list(svg_coords.values())
    if not pts:
        return 400.0, 380.0
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return cx, cy


def _rl_anchor_angle(ix, svg_coords, cx, cy):
    ai = ix.get("lig_atom_idx", 0)
    ax, ay = svg_coords.get(ai, (cx, cy))
    ring_idxs = ix.get("ring_atom_indices")
    if ring_idxs:
        rx, ry = _ring_centroid_2d(ring_idxs, svg_coords)
        if rx is not None:
            ax, ay = rx, ry
    return _math.atan2(ay - cy, ax - cx)


def _rl_ray_boundary(cx, cy, theta, atom_xy, atom_r=20.0):
    dx = _math.cos(theta)
    dy = _math.sin(theta)
    r2 = atom_r * atom_r
    t_max = 0.0
    for (ax, ay) in atom_xy:
        vx = ax - cx
        vy = ay - cy
        t_proj = vx * dx + vy * dy
        if t_proj < -atom_r:
            continue
        perp2 = vx * vx + vy * vy - t_proj * t_proj
        if perp2 >= r2:
            continue
        t_exit = t_proj + _math.sqrt(max(r2 - perp2, 0.0))
        if t_exit > t_max:
            t_max = t_exit
    return t_max if t_max > 1.0 else 20.0


def _rl_place_radially(sorted_interactions, svg_coords, cx, cy,
                        atom_xy, atom_r, gap, node_r, rng):
    JITTER_T = 7.0
    JITTER_R = 4.0

    n = len(sorted_interactions)
    jit_t = rng.uniform(-JITTER_T, JITTER_T, n)
    jit_r = rng.uniform(-JITTER_R, JITTER_R, n)

    positions = []
    for k, ix in enumerate(sorted_interactions):
        theta  = _rl_anchor_angle(ix, svg_coords, cx, cy)
        bdist  = _rl_ray_boundary(cx, cy, theta, atom_xy, atom_r)
        r_place = bdist + gap + node_r + jit_r[k]

        cos_t = _math.cos(theta)
        sin_t = _math.sin(theta)
        tx, ty = -sin_t, cos_t

        bx = cx + r_place * cos_t + jit_t[k] * tx
        by = cy + r_place * sin_t + jit_t[k] * ty
        positions.append((bx, by))

    return positions


def _rl_resolve_overlaps(positions, atom_xy, cx, cy,
                          node_r=24.55, excl_r=46.0, max_iters=400):
    n = len(positions)
    if n <= 1:
        return list(positions)

    min_sep = node_r * 2.0 + 8.0

    bx = [p[0] for p in positions]
    by = [p[1] for p in positions]

    for i in range(n):
        for j in range(i + 1, n):
            ddx = bx[j] - bx[i]; ddy = by[j] - by[i]
            if _math.sqrt(ddx * ddx + ddy * ddy) < 1.0:
                ang = (i * 137.508 + j * 73.2) * _math.pi / 180.0
                kick = min_sep * 0.5
                bx[i] -= _math.cos(ang) * kick; by[i] -= _math.sin(ang) * kick
                bx[j] += _math.cos(ang) * kick; by[j] += _math.sin(ang) * kick

    for _it in range(max_iters):
        dx_acc = [0.0] * n
        dy_acc = [0.0] * n
        any_overlap = False

        for i in range(n):
            for j in range(i + 1, n):
                ddx = bx[j] - bx[i]
                ddy = by[j] - by[i]
                d   = _math.sqrt(ddx * ddx + ddy * ddy) + 1e-8
                if d < min_sep:
                    frac = (min_sep - d) * 0.55 / d
                    dx_acc[i] -= frac * ddx;  dy_acc[i] -= frac * ddy
                    dx_acc[j] += frac * ddx;  dy_acc[j] += frac * ddy
                    any_overlap = True

        for i in range(n):
            bx[i] += dx_acc[i]
            by[i] += dy_acc[i]

        for i in range(n):
            for (ax, ay) in atom_xy:
                ddx = bx[i] - ax
                ddy = by[i] - ay
                d   = _math.sqrt(ddx * ddx + ddy * ddy) + 1e-8
                if d < excl_r:
                    push = (excl_r - d) * 0.5 / d
                    if d < 0.5:
                        ddx = bx[i] - cx
                        ddy = by[i] - cy
                        d   = _math.sqrt(ddx * ddx + ddy * ddy) + 1e-8
                        push = excl_r * 0.5 / d
                    bx[i] += push * ddx
                    by[i] += push * ddy

        if not any_overlap:
            break

    return list(zip(bx, by))


def _rl_reduce_crossings(ix_list, bx_by, svg_coords, cx, cy):
    n      = len(ix_list)
    bx_by  = list(bx_by)

    def _anchor(ix):
        ai = ix.get("lig_atom_idx", 0)
        ax, ay = svg_coords.get(ai, (cx, cy))
        ri = ix.get("ring_atom_indices")
        if ri:
            rx, ry = _ring_centroid_2d(ri, svg_coords)
            if rx is not None:
                return (rx, ry)
        return (ax, ay)

    def _cross(a1, b1, a2, b2):
        def _side(O, A, B):
            return (B[0]-O[0])*(A[1]-O[1]) - (B[1]-O[1])*(A[0]-O[0])
        d1 = _side(a2, b2, a1);  d2 = _side(a2, b2, b1)
        d3 = _side(a1, b1, a2);  d4 = _side(a1, b1, b2)
        return (d1 * d2 < 0) and (d3 * d4 < 0)

    for _pass in range(n * 2):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if _cross(_anchor(ix_list[i]), bx_by[i],
                          _anchor(ix_list[j]), bx_by[j]):
                    bx_by[i], bx_by[j] = bx_by[j], bx_by[i]
                    improved = True
        if not improved:
            break

    return bx_by


def _place_residues_pca(interactions, svg_coords, mol3d, cx, cy, R=210):
    if not interactions:
        return []

    try:
        import numpy as _np_rc

        NODE_R  = 24.55
        GAP     = 45.0
        ATOM_R  = 20.0

        lx, ly  = _rl_ligand_center(svg_coords)
        atom_xy = list(svg_coords.values())

        anchor_a = [_rl_anchor_angle(ix, svg_coords, lx, ly)
                    for ix in interactions]
        order    = sorted(range(len(interactions)),
                          key=lambda k: anchor_a[k])
        sorted_ix = [interactions[k] for k in order]

        rng       = _np_rc.random.default_rng(42)
        positions = _rl_place_radially(
            sorted_ix, svg_coords, lx, ly,
            atom_xy, ATOM_R, GAP, NODE_R, rng,
        )

        positions = _rl_resolve_overlaps(
            positions, atom_xy, lx, ly,
            node_r=NODE_R,
            excl_r=NODE_R + ATOM_R,
        )

        positions = _rl_reduce_crossings(
            sorted_ix, positions, svg_coords, lx, ly,
        )

        result = []
        for k, (bx, by) in enumerate(positions):
            ang = _math.atan2(by - ly, bx - lx)
            result.append({
                **sorted_ix[k],
                "angle":      ang,
                "bx":         float(bx),
                "by":         float(by),
                "slot_angle": ang,
            })
        return result

    except Exception:
        return _place_residues_no_cross(interactions, svg_coords, cx, cy, R)


def _render_ligand_svg(mol2d, svg_coords):
    from rdkit import Chem
    parts=[]
    ri=mol2d.GetRingInfo()
    arom_bonds=set(); arom_rings=[]
    for ring in ri.AtomRings():
        if all(mol2d.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            for k in range(len(ring)):
                arom_bonds.add(frozenset([ring[k],ring[(k+1)%len(ring)]]))
            arom_rings.append(ring)

    def _sh(fx,fy,tx,ty,sym):
        if sym not in ("C",""):
            dx,dy=tx-fx,ty-fy; L=_math.sqrt(dx*dx+dy*dy)+1e-9
            r={"H":8,"N":9,"O":9,"S":11,"P":11,"F":8,"CL":13,"BR":13,"I":11}.get(sym,9)
            return fx+dx/L*r, fy+dy/L*r
        return fx,fy

    for bond in mol2d.GetBonds():
        i1,i2=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        x1,y1=svg_coords.get(i1,(0,0)); x2,y2=svg_coords.get(i2,(0,0))
        s1=mol2d.GetAtomWithIdx(i1).GetSymbol().upper()
        s2=mol2d.GetAtomWithIdx(i2).GetSymbol().upper()
        x1s,y1s=_sh(x1,y1,x2,y2,s1); x2s,y2s=_sh(x2,y2,x1,y1,s2)
        bt=bond.GetBondType()
        if frozenset([i1,i2]) in arom_bonds:
            parts.append(f'<line x1="{x1s:.2f}" y1="{y1s:.2f}" x2="{x2s:.2f}" y2="{y2s:.2f}"'
                         f' stroke="#1a1a1a" stroke-width="1.77" opacity="0.9"/>')
        elif bt==Chem.BondType.DOUBLE:
            dx,dy=x2s-x1s,y2s-y1s; L=_math.sqrt(dx*dx+dy*dy)+1e-9
            px,py=-dy/L*2.5,dx/L*2.5
            for sg in (1,-1):
                parts.append(f'<line x1="{x1s+px*sg:.2f}" y1="{y1s+py*sg:.2f}"'
                             f' x2="{x2s+px*sg:.2f}" y2="{y2s+py*sg:.2f}"'
                             f' stroke="#1a1a1a" stroke-width="2.44" opacity="0.9"/>')
        elif bt==Chem.BondType.TRIPLE:
            dx,dy=x2s-x1s,y2s-y1s; L=_math.sqrt(dx*dx+dy*dy)+1e-9
            px,py=-dy/L*3.8,dx/L*3.8
            for m in (-1,0,1):
                parts.append(f'<line x1="{x1s+px*m:.2f}" y1="{y1s+py*m:.2f}"'
                             f' x2="{x2s+px*m:.2f}" y2="{y2s+py*m:.2f}"'
                             f' stroke="#1a1a1a" stroke-width="2.0" opacity="0.9"/>')
        else:
            bd=bond.GetBondDir()
            if bd==Chem.BondDir.BEGINWEDGE:
                dx,dy=x2s-x1s,y2s-y1s; L=_math.sqrt(dx*dx+dy*dy)+1e-9
                px,py=-dy/L*3.0,dx/L*3.0
                parts.append(f'<polygon points="{x1s:.2f},{y1s:.2f}'
                             f' {x2s+px:.2f},{y2s+py:.2f} {x2s-px:.2f},{y2s-py:.2f}"'
                             f' fill="#1a1a1a" stroke="none"/>')
            elif bd==Chem.BondDir.BEGINDASH:
                dx,dy=x2s-x1s,y2s-y1s; L=_math.sqrt(dx*dx+dy*dy)+1e-9
                px,py=-dy/L,dx/L
                for step in range(1,6):
                    t=step/7; mx2=x1s+dx*t; my2=y1s+dy*t; w=t*5.0
                    parts.append(f'<line x1="{mx2-px*w:.2f}" y1="{my2-py*w:.2f}"'
                                 f' x2="{mx2+px*w:.2f}" y2="{my2+py*w:.2f}"'
                                 f' stroke="#1a1a1a" stroke-width="1.6"/>')
            else:
                parts.append(f'<line x1="{x1s:.2f}" y1="{y1s:.2f}" x2="{x2s:.2f}" y2="{y2s:.2f}"'
                             f' stroke="#1a1a1a" stroke-width="1.77" opacity="0.9"/>')

    for ring in arom_rings:
        rcoords=[svg_coords.get(i,(0,0)) for i in ring]
        rcx=sum(x for x,y in rcoords)/len(rcoords)
        rcy=sum(y for x,y in rcoords)/len(rcoords)
        avg=sum(_math.sqrt((x-rcx)**2+(y-rcy)**2) for x,y in rcoords)/len(rcoords)
        cr=avg*0.58
        parts.append(f'<circle cx="{rcx:.2f}" cy="{rcy:.2f}" r="{cr:.2f}"'
                     f' fill="none" stroke="#1a1a1a" stroke-width="1.77"'
                     f' stroke-dasharray="5.43 2.72" opacity="0.7"/>')
        parts.append(f'<circle cx="{rcx:.2f}" cy="{rcy:.2f}" r="5.43"'
                     f' fill="{_AROM_DOT_CLR}"/>')

    for i in range(mol2d.GetNumAtoms()):
        atom=mol2d.GetAtomWithIdx(i); sym=atom.GetSymbol()
        if sym=="C": continue
        ax,ay=svg_coords.get(i,(0,0))
        clr=_ATOM_CLR.get(sym.upper(),"#555")
        fs={"H":16}.get(sym,17.65)
        hw={"H":7,"N":9,"O":9,"S":11,"P":11,"F":8,"CL":16,"BR":16,"I":11}.get(sym.upper(),9)
        parts.append(f'<rect x="{ax-hw:.1f}" y="{ay-11:.1f}"'
                     f' width="{hw*2:.0f}" height="22" fill="white" stroke="none"/>')
        parts.append(f'<text x="{ax:.2f}" y="{ay:.2f}" text-anchor="middle"'
                     f' dominant-baseline="central"'
                     f' font-family="Arial,sans-serif" font-size="{fs}"'
                     f' font-weight="700" fill="{clr}">{sym}</text>')
        fc=atom.GetFormalCharge()
        if fc!=0:
            fcs="+" if fc==1 else "−" if fc==-1 else f"{fc:+d}"
            parts.append(f'<text x="{ax+hw:.1f}" y="{ay-hw+2:.1f}"'
                         f' font-family="Arial,sans-serif" font-size="10"'
                         f' fill="{clr}">{fcs}</text>')
    return "".join(parts)


def _render_diagram_svg(mol2d, svg_coords, placements, title, W, H):
    parts=[]
    parts.append(f'<svg width="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">')
    parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')

    if title:
        esc=title.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        tw=len(esc)*14.5+48; tw=max(tw,240); tw=min(tw,W-40)
        px=(W-tw)/2; ph=46; pr=ph/2
        parts.append(f'<rect x="{px:.1f}" y="14" width="{tw:.0f}" height="{ph}"'
                     f' rx="{pr:.1f}" ry="{pr:.1f}" fill="#f2f2f2" stroke="none"/>')
        parts.append(f'<text x="{W/2:.1f}" y="37" text-anchor="middle"'
                     f' dominant-baseline="central"'
                     f' font-family="Arial,sans-serif" font-size="24.93"'
                     f' font-weight="700" fill="#1a1a1a">{esc}</text>')

    for p in placements:
        itype=p["itype"]
        bx,by=p["bx"],p["by"]
        cbx=max(50,min(bx,W-50)); cby=max(70,min(by,H-65))
        bg=_RES_CIRCLE.get(itype,dict(fill="#cccccc",opacity=0.2))
        parts.append(f'<circle cx="{cbx:.1f}" cy="{cby:.1f}" r="24.55"'
                     f' fill="{bg["fill"]}" opacity="{bg["opacity"]}"/>')

    for p in placements:
        itype=p["itype"]
        if itype=="hydrophobic": continue

        bx,by=p["bx"],p["by"]
        cbx=max(50,min(bx,W-50)); cby=max(70,min(by,H-65))

        if itype in ("pi_pi","cation_pi") and p.get("ring_atom_indices"):
            lx,ly=_ring_centroid_2d(p["ring_atom_indices"],svg_coords)
            if lx is None:
                ai=p.get("lig_atom_idx",0); lx,ly=svg_coords.get(ai,(W//2,H//2))
        else:
            ai=p.get("lig_atom_idx",0); lx,ly=svg_coords.get(ai,(W//2,H//2))

        clr=_LINE_CLR.get(itype,"#888")
        dash_attr=' stroke-dasharray="5 3"'
        dash_attr_hbx=' stroke-dasharray="4 2 1 2"'

        if itype in ("pi_pi","cation_pi"):
            parts.append(f'<line x1="{lx:.2f}" y1="{ly:.2f}"'
                         f' x2="{cbx:.2f}" y2="{cby:.2f}"'
                         f' stroke="{clr}" stroke-width="1.6"'
                         f'{dash_attr} opacity="0.85"/>')
        elif itype in ("hbond","hbond_to_halogen"):
            da=dash_attr if itype=="hbond" else dash_attr_hbx
            parts.append(f'<line x1="{lx:.2f}" y1="{ly:.2f}"'
                         f' x2="{cbx:.2f}" y2="{cby:.2f}"'
                         f' stroke="{clr}" stroke-width="1.6"'
                         f'{da} opacity="0.85"/>')
            if p.get("distance") is not None:
                t_along = 0.40
                bx_l = lx + (cbx - lx) * t_along
                by_l = ly + (cby - ly) * t_along
                dx_l = cbx - lx; dy_l = cby - ly
                _len = _math.sqrt(dx_l*dx_l + dy_l*dy_l) + 1e-9
                px_l = -dy_l / _len; py_l = dx_l / _len
                mx2 = bx_l + px_l * 14
                my2 = by_l + py_l * 14
                ds=f"{p['distance']}\u00c5"
                tw2=len(ds)*7+8
                parts.append(f'<rect x="{mx2-tw2/2:.1f}" y="{my2-9:.1f}"'
                             f' width="{tw2:.0f}" height="17" rx="4"'
                             f' fill="white" stroke="{clr}" stroke-width="0.5"/>')
                parts.append(f'<text x="{mx2:.1f}" y="{my2:.1f}" text-anchor="middle"'
                             f' dominant-baseline="central"'
                             f' font-family="Arial,sans-serif" font-size="14"'
                             f' font-weight="700" fill="{clr}">{ds}</text>')
        else:
            da={"ionic":"6 2 2 2","metal":"3 2","halogen":"5 2"}.get(itype,"5 3")
            parts.append(f'<line x1="{lx:.2f}" y1="{ly:.2f}"'
                         f' x2="{cbx:.2f}" y2="{cby:.2f}"'
                         f' stroke="{clr}" stroke-width="1.8"'
                         f' stroke-dasharray="{da}" opacity="0.85"/>')
            if p.get("distance") is not None:
                t_along = 0.40
                bx_l = lx + (cbx - lx) * t_along
                by_l = ly + (cby - ly) * t_along
                dx_l = cbx - lx; dy_l = cby - ly
                _len = _math.sqrt(dx_l*dx_l + dy_l*dy_l) + 1e-9
                px_l = -dy_l / _len; py_l = dx_l / _len
                mx2 = bx_l + px_l * 14
                my2 = by_l + py_l * 14
                ds=f"{p['distance']}\u00c5"; tw2=len(ds)*7+8
                parts.append(f'<rect x="{mx2-tw2/2:.1f}" y="{my2-9:.1f}"'
                             f' width="{tw2:.0f}" height="17" rx="4"'
                             f' fill="white" stroke="{clr}" stroke-width="0.5"/>')
                parts.append(f'<text x="{mx2:.1f}" y="{my2:.1f}" text-anchor="middle"'
                             f' dominant-baseline="central"'
                             f' font-family="Arial,sans-serif" font-size="14"'
                             f' font-weight="700" fill="{clr}">{ds}</text>')

    parts.append(_render_ligand_svg(mol2d, svg_coords))

    for p in placements:
        itype=p["itype"]
        bx,by=p["bx"],p["by"]
        cbx=max(50,min(bx,W-50)); cby=max(70,min(by,H-65))
        rn=p["resname"]; ri=p["resid"]; ch=p.get("chain","")
        _NO_NUM = HEME_RESNAMES | METAL_RESNAMES
        lbl = rn.upper() if rn.upper() in _NO_NUM else f"{rn.upper()} {ri}{ch}"
        lbl_clr=_LBL_CLR.get(itype,"#333")
        parts.append(f'<text x="{cbx:.1f}" y="{cby:.1f}" text-anchor="middle"'
                     f' dominant-baseline="central"'
                     f' font-family="Arial,sans-serif" font-size="14.29"'
                     f' font-weight="700" fill="{lbl_clr}">{lbl}</text>')

    _LEG_ORDER=["hydrophobic","hbond","pi_pi","cation_pi",
                "hbond_to_halogen","ionic","metal","halogen"]
    _LEG_LABEL={"hydrophobic":"Hydrophobic","hbond":"Hydrogen bond",
                "hbond_to_halogen":"H···Halogen",
                "pi_pi":"π-π stacking","cation_pi":"Cation-π",
                "ionic":"Ionic","metal":"Metal","halogen":"Halogen bond"}
    active=[t for t in _LEG_ORDER if any(p["itype"]==t for p in placements)]
    if active:
        ly0 = H - 52
        def _entry_w(t):
            txt = _LEG_LABEL.get(t, t)
            return 9.54*2 + (4+28 if t != "hydrophobic" else 0) + 6 + len(txt)*9.5 + 20
        total_w = sum(_entry_w(t) for t in active)
        total_w = min(total_w, W - 40)
        lx0 = (W - total_w) / 2
        parts.append(f'<rect x="{lx0-8:.0f}" y="{ly0-5}" width="{total_w+16:.0f}"'
                     f' height="44" fill="white" stroke="#e0e0e0"'
                     f' stroke-width="0.8" rx="6"/>')
        for k,it in enumerate(active):
            bg   = _RES_CIRCLE.get(it, dict(fill="#ccc", opacity=0.2))
            clr  = _LINE_CLR.get(it, bg["fill"])
            lbl  = _LEG_LABEL.get(it, it)
            text_w  = len(lbl) * 9.5 + 6
            glyph_w = 9.54*2 + (4 + 28 if it != "hydrophobic" else 0)
            entry_w = glyph_w + 6 + text_w
            circ_cx = lx0 + sum(
                (9.54*2 + (4+28 if active[kk]!="hydrophobic" else 0) + 6 + len(_LEG_LABEL.get(active[kk],active[kk]))*9.5 + 6 + 20)
                for kk in range(k)
            ) + 9.54
            parts.append(f'<circle cx="{circ_cx:.1f}" cy="{ly0+10}" r="9.54"'
                         f' fill="{bg["fill"]}" opacity="{bg["opacity"]}"/>')
            line_x1 = circ_cx + 9.54 + 4
            line_x2 = line_x1 + 28
            if it != "hydrophobic":
                parts.append(f'<line x1="{line_x1:.1f}" y1="{ly0+10}"'
                             f' x2="{line_x2:.1f}" y2="{ly0+10}"'
                             f' stroke="{clr}" stroke-width="2"'
                             f' stroke-dasharray="5 3" opacity="0.85"/>')
            text_x = (line_x2 if it != "hydrophobic" else circ_cx + 9.54) + 6
            parts.append(f'<text x="{text_x:.1f}" y="{ly0+10}" text-anchor="start"'
                         f' dominant-baseline="central"'
                         f' font-family="Arial,sans-serif" font-size="16"'
                         f' font-weight="700" fill="#555">{lbl}</text>')

    parts.append('</svg>')
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def draw_interaction_diagram(
    receptor_pdb: str,
    pose_sdf: str,
    smiles: str,
    title: str = "",
    cutoff: float = 4.5,
    size: tuple = (800, 759),
    max_residues: int = 14,
) -> bytes:
    """2D interaction diagram — SVG bytes."""
    try:
        mol2d, sc, pl, W, H = _build_diagram_data(
            receptor_pdb, pose_sdf, smiles, cutoff, max_residues, size
        )
    except Exception as e:
        W, H = size
        return (f'<svg viewBox="0 0 {W} 80" xmlns="http://www.w3.org/2000/svg">'
                f'<rect width="{W}" height="80" fill="white"/>'
                f'<text x="{W//2}" y="44" text-anchor="middle"'
                f' font-family="Arial,sans-serif" font-size="13" fill="#cc2222">'
                f'Error: {e}</text></svg>').encode()
    svg = _render_diagram_svg(mol2d, sc, pl, title, W, H)
    return svg.encode()


def draw_interaction_diagram_data(
    receptor_pdb: str,
    pose_sdf: str,
    smiles: str,
    title: str = "",
    cutoff: float = 4.5,
    size: tuple = (800, 759),
    max_residues: int = 14,
) -> dict:
    """Same pipeline as draw_interaction_diagram but returns a dict for interactive renderer."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDepictor
    RDLogger.DisableLog("rdApp.*")
    W, H = size
    try:
        mol3d = None
        for san in (True, False):
            sup = Chem.SDMolSupplier(pose_sdf, sanitize=san, removeHs=False)
            mol3d = next((m for m in sup if m is not None), None)
            if mol3d is not None:
                if not san:
                    try: Chem.SanitizeMol(mol3d)
                    except: pass
                break
        if mol3d is None or mol3d.GetNumConformers() == 0:
            return None
    except Exception:
        return None

    mol2d = None
    if smiles and smiles.strip():
        mol2d = Chem.MolFromSmiles(smiles.strip())
    if mol2d is None:
        mol2d = Chem.RemoveHs(mol3d, sanitize=False)
        try: Chem.SanitizeMol(mol2d)
        except: pass
    mol2d = Chem.RemoveHs(mol2d)
    rdDepictor.Compute2DCoords(mol2d)

    m3 = Chem.RemoveHs(mol3d, sanitize=False)
    try: Chem.SanitizeMol(m3)
    except: pass
    m3to2d = {}
    try:
        mt = m3.GetSubstructMatch(mol2d)
        if len(mt) == mol2d.GetNumAtoms():
            for i2, i3 in enumerate(mt): m3to2d[i3] = i2
    except: pass

    try: raw = _detect_all_interactions(mol3d, receptor_pdb, cutoff=cutoff)
    except: raw = []

    try: _enrich_with_res_xyz(raw, mol3d, receptor_pdb)
    except: pass

    for ix in raw:
        ix["lig_atom_idx"] = m3to2d.get(ix.get("lig_atom_idx", 0), 0)
        if ix.get("ring_atom_indices"):
            ix["ring_atom_indices"] = [m3to2d.get(i, i) for i in ix["ring_atom_indices"]]

    pm = {t: i for i, t in enumerate(_ITYPE_PRIORITY)}
    ded = _deduplicate_interactions(raw)
    ded.sort(key=lambda x: (pm.get(x["itype"], 99), x["distance"]))
    ded = ded[:max_residues]

    cx, cy = W // 2, H // 2
    sc = _compute_svg_coords(mol2d, cx, cy, target_size=280)
    pl = _place_residues_pca(ded, sc, mol3d, cx, cy, R=210)

    lig_svg = _render_ligand_svg(mol2d, sc)
    sc_serial = {str(k): [round(v[0], 2), round(v[1], 2)] for k, v in sc.items()}

    import json as _json
    pl_serial = []
    for p in pl:
        lx, ly = sc.get(p.get("lig_atom_idx", 0), (cx, cy))
        if p.get("ring_atom_indices"):
            rx, ry = _ring_centroid_2d(p["ring_atom_indices"], sc)
            if rx is not None:
                lx, ly = rx, ry
        pl_serial.append({
            "id":       f"r{len(pl_serial)}",
            "label":    (p['resname'].upper() if p['resname'].upper() in (HEME_RESNAMES | METAL_RESNAMES)
                         else f"{p['resname']} {p['resid']}{p.get('chain','')}"),
            "itype":    p["itype"],
            "distance": p.get("distance"),
            "lx":       round(lx, 2),
            "ly":       round(ly, 2),
            "bx":       round(p["bx"], 2),
            "by":       round(p["by"], 2),
        })

    RDLogger.EnableLog("rdApp.error")
    return {
        "W": W, "H": H, "title": title,
        "ligand_svg": lig_svg,
        "placements": pl_serial,
        "svg_coords": sc_serial,
    }


def _build_diagram_data(receptor_pdb, pose_sdf, smiles, cutoff, max_residues,
                         size=(800, 759)):
    """Shared setup for both static SVG and interactive HTML renderers."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDepictor
    RDLogger.DisableLog("rdApp.*")
    W, H = size
    mol3d = None
    for san in (True, False):
        sup = Chem.SDMolSupplier(pose_sdf, sanitize=san, removeHs=False)
        mol3d = next((m for m in sup if m is not None), None)
        if mol3d is not None:
            if not san:
                try: Chem.SanitizeMol(mol3d)
                except: pass
            break
    if mol3d is None or mol3d.GetNumConformers() == 0:
        raise ValueError("No valid 3D pose in SDF")
    mol2d = None
    if smiles and smiles.strip():
        mol2d = Chem.MolFromSmiles(smiles.strip())
    if mol2d is None:
        mol2d = Chem.RemoveHs(mol3d, sanitize=False)
        try: Chem.SanitizeMol(mol2d)
        except: pass
    mol2d = Chem.RemoveHs(mol2d)
    rdDepictor.Compute2DCoords(mol2d)
    m3 = Chem.RemoveHs(mol3d, sanitize=False)
    try: Chem.SanitizeMol(m3)
    except: pass
    m3to2d = {}
    try:
        mt = m3.GetSubstructMatch(mol2d)
        if len(mt) == mol2d.GetNumAtoms():
            for i2, i3 in enumerate(mt): m3to2d[i3] = i2
    except: pass
    try: raw = _detect_all_interactions(mol3d, receptor_pdb, cutoff=cutoff)
    except: raw = []

    try: _enrich_with_res_xyz(raw, mol3d, receptor_pdb)
    except: pass

    for ix in raw:
        ix["lig_atom_idx"] = m3to2d.get(ix.get("lig_atom_idx", 0), 0)
        if ix.get("ring_atom_indices"):
            ix["ring_atom_indices"] = [m3to2d.get(i, i) for i in ix["ring_atom_indices"]]
    pm = {t: i for i, t in enumerate(_ITYPE_PRIORITY)}
    ded = _deduplicate_interactions(raw)
    ded.sort(key=lambda x: (pm.get(x["itype"], 99), x["distance"]))
    ded = ded[:max_residues]
    cx, cy = W // 2, H // 2
    sc = _compute_svg_coords(mol2d, cx, cy, target_size=280)
    pl = _place_residues_pca(ded, sc, mol3d, cx, cy, R=210)
    RDLogger.EnableLog("rdApp.error")
    return mol2d, sc, pl, W, H


def draw_interaction_diagram_interactive(
    receptor_pdb: str,
    pose_sdf: str,
    smiles: str,
    title: str = "",
    cutoff: float = 4.5,
    size: tuple = (800, 759),
    max_residues: int = 14,
) -> str:
    """Interactive HTML version of draw_interaction_diagram."""
    import json

    try:
        mol2d, sc, pl, W, H = _build_diagram_data(
            receptor_pdb, pose_sdf, smiles, cutoff, max_residues, size
        )
    except Exception as e:
        return f'<p style="color:red">Error: {e}</p>'

    lig_svg = _render_ligand_svg(mol2d, sc)

    residues_js = []
    for p in pl:
        itype = p["itype"]
        ai    = p.get("lig_atom_idx", 0)
        lx, ly = sc.get(ai, (W // 2, H // 2))
        if itype in ("pi_pi", "cation_pi") and p.get("ring_atom_indices"):
            rx, ry = _ring_centroid_2d(p["ring_atom_indices"], sc)
            if rx is not None: lx, ly = rx, ry
        residues_js.append({
            "id":      p["resname"] + str(p["resid"]) + p.get("chain",""),
            "label":   (p['resname'].upper() if p['resname'].upper() in (HEME_RESNAMES | METAL_RESNAMES)
                        else f"{p['resname']} {p['resid']}{p.get('chain','')}"),
            "itype":   itype,
            "dist":    str(p["distance"]) + "Å" if p.get("distance") else "",
            "lx":      round(lx, 2),
            "ly":      round(ly, 2),
            "bx":      round(p["bx"], 2),
            "by":      round(p["by"], 2),
        })

    legend_types = list(dict.fromkeys(p["itype"] for p in pl))
    _LEG_LABEL = {
        "hbond":"H-bond", "hbond_to_halogen":"H···Halogen",
        "pi_pi":"π-π stacking","cation_pi":"Cation-π",
        "hydrophobic":"Hydrophobic","ionic":"Ionic",
        "metal":"Metal","halogen":"Halogen bond",
    }

    esc_title = (title.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
    tw = min(len(esc_title) * 14 + 48, W - 40)
    pill_x = (W - tw) / 2

    data_json = json.dumps(residues_js)

    html = f"""
<style>
  #diag-wrap {{ position:relative; width:100%; background:white; border:1px solid #e0e0e0; border-radius:8px; overflow:hidden; }}
  #diag-toolbar {{ display:flex; gap:8px; padding:8px 12px; border-bottom:1px solid #eee; align-items:center; flex-wrap:wrap; background:#fafafa; }}
  #diag-toolbar button {{ font-size:12px; padding:5px 10px; border:1px solid #ccc; border-radius:5px; background:#fff; cursor:pointer; font-weight:600; }}
  #diag-toolbar button:hover {{ background:#f0f0f0; }}
  #diag-hint {{ font-size:11px; color:#888; margin-left:auto; }}
  #diag-svg {{ width:100%; display:block; cursor:default; user-select:none; -webkit-user-select:none; }}
  .r-circle {{ cursor:grab; }}
  .r-circle:active {{ cursor:grabbing; }}
</style>
<div id="diag-wrap">
  <div id="diag-toolbar">
    <button onclick="resetLayout()">&#8635; Reset</button>
    <button onclick="exportSVG()">&#8595; SVG</button>
    <button onclick="exportPNG()">&#8595; PNG</button>
    <select id="dpi-sel" title="PNG resolution" style="font-size:12px;padding:4px 6px;border:1px solid #ccc;border-radius:5px;background:#fff;cursor:pointer;">
      <option value="1">Screen (1×)</option>
      <option value="2" selected>Print 150dpi (2×)</option>
      <option value="3">Print 300dpi (3×)</option>
      <option value="4">High-res 600dpi (4×)</option>
    </select>
    <span id="diag-hint">Drag any residue label to reposition it</span>
  </div>
  <svg id="diag-svg" viewBox="0 0 {W} {H}">
    <rect width="{W}" height="{H}" fill="white"/>
    <rect x="{pill_x:.1f}" y="12" width="{tw:.0f}" height="44" rx="22" fill="#f2f2f2"/>
    <text x="{W/2:.1f}" y="34" text-anchor="middle" dominant-baseline="central"
          font-family="Arial,sans-serif" font-size="18" font-weight="700" fill="#1a1a1a">{esc_title}</text>
    <g id="g-lines"></g>
    <g id="g-ligand">{lig_svg}</g>
    <g id="g-residues"></g>
    <g id="g-legend"></g>
  </svg>
</div>
<script>
(function(){{
const W={W}, H={H};
const RESIDUES={data_json};
const LEG_TYPES={json.dumps(legend_types)};
const LEG_LABEL={json.dumps(_LEG_LABEL)};

const TYPE_CFG = {{
  hbond:           {{fill:"#80dd80",stroke:"#1a7a1a",line:"#1a7a1a",dash:"5 3",lw:1.6}},
  hbond_to_halogen:{{fill:"#c8b0ff",stroke:"#6633aa",line:"#6633aa",dash:"4 2 1 2",lw:1.6}},
  pi_pi:           {{fill:"#e200e8",stroke:"#e200e8",line:"#e200e8",dash:"5 3",lw:1.6}},
  cation_pi:       {{fill:"#e200e8",stroke:"#e200e8",line:"#e200e8",dash:"5 3",lw:1.6}},
  hydrophobic:     {{fill:"#2287ff",stroke:"#2287ff",line:null,     dash:"",  lw:0}},
  ionic:           {{fill:"#ffb0d0",stroke:"#cc2277",line:"#cc2277",dash:"6 2 2 2",lw:1.8}},
  metal:           {{fill:"#ffe080",stroke:"#cc8800",line:"#cc8800",dash:"3 2",lw:1.8}},
  halogen:         {{fill:"#ffb0d0",stroke:"#cc2277",line:"#cc2277",dash:"5 2",lw:1.6}},
}};

const svg   = document.getElementById("diag-svg");
const gLines= document.getElementById("g-lines");
const gRes  = document.getElementById("g-residues");
const gLeg  = document.getElementById("g-legend");

let pos = {{}};
let els = {{}};

function ns(tag){{ return document.createElementNS("http://www.w3.org/2000/svg",tag); }}
function attr(el,obj){{ for(const[k,v] of Object.entries(obj)) el.setAttribute(k,v); return el; }}

function svgPt(cx,cy){{
  const r=svg.getBoundingClientRect();
  const vb=svg.viewBox.baseVal;
  return {{x:(cx-r.left)*vb.width/r.width+vb.x, y:(cy-r.top)*vb.height/r.height+vb.y}};
}}

function initPos(){{
  RESIDUES.forEach(r=>{{ pos[r.id]={{x:r.bx,y:r.by}}; }});
}}

function drawAll(){{
  gLines.innerHTML=""; gRes.innerHTML=""; gLeg.innerHTML="";
  els={{}};
  RESIDUES.forEach(r=>drawResidue(r));
  drawLegend();
}}

function distLabel(r){{
  const cfg=TYPE_CFG[r.itype]||TYPE_CFG.hbond;
  if(!r.dist||!cfg.line) return null;
  const p=pos[r.id];
  const t=0.40;
  const mx=r.lx+(p.x-r.lx)*t, my=r.ly+(p.y-r.ly)*t;
  const dx=p.x-r.lx, dy=p.y-r.ly;
  const ln=Math.sqrt(dx*dx+dy*dy)+0.001;
  const ox=-dy/ln*14, oy=dx/ln*14;
  return {{x:mx+ox, y:my+oy}};
}}

function drawResidue(r){{
  const cfg=TYPE_CFG[r.itype]||TYPE_CFG.hbond;
  const p=pos[r.id];
  const lclr={{hbond:"#1a7a1a",hbond_to_halogen:"#6633aa",pi_pi:"#9900aa",
               cation_pi:"#9900aa",hydrophobic:"#1a5fa8",ionic:"#880055",
               metal:"#885500",halogen:"#880044"}}[r.itype]||"#333";

  const grp=ns("g"); grp.setAttribute("class","r-circle"); grp.setAttribute("data-id",r.id);

  let lineEl=null, drectEl=null, dtxtEl=null;
  if(cfg.line){{
    lineEl=attr(ns("line"),{{x1:r.lx,y1:r.ly,x2:p.x,y2:p.y,
      stroke:cfg.line,"stroke-width":cfg.lw,"stroke-dasharray":cfg.dash,opacity:0.85}});
    gLines.appendChild(lineEl);

    if(r.dist){{
      const dl=distLabel(r);
      const tw=r.dist.length*7+8;
      drectEl=attr(ns("rect"),{{x:dl.x-tw/2,y:dl.y-8,width:tw,height:15,rx:4,
        fill:"white",stroke:cfg.line,"stroke-width":0.5}});
      dtxtEl=attr(ns("text"),{{x:dl.x,y:dl.y,"text-anchor":"middle",
        "dominant-baseline":"central","font-family":"Arial,sans-serif",
        "font-size":12,"font-weight":700,fill:cfg.line}});
      dtxtEl.textContent=r.dist;
      gLines.appendChild(drectEl);
      gLines.appendChild(dtxtEl);
    }}
  }}

  const circ=attr(ns("circle"),{{cx:p.x,cy:p.y,r:24.55,fill:cfg.fill,opacity:0.2,
    stroke:cfg.stroke,"stroke-width":1.2}});
  grp.appendChild(circ);

  const txt=attr(ns("text"),{{x:p.x,y:p.y,"text-anchor":"middle",
    "dominant-baseline":"central","font-family":"Arial,sans-serif",
    "font-size":11,"font-weight":700,fill:lclr}});
  txt.textContent=r.label;
  grp.appendChild(txt);

  gRes.appendChild(grp);
  els[r.id]={{lineEl,drectEl,dtxtEl,circ,txt}};
  makeDraggable(grp,r);
}}

function updateResidue(r){{
  const p=pos[r.id];
  const e=els[r.id]; if(!e) return;
  e.circ.setAttribute("cx",p.x); e.circ.setAttribute("cy",p.y);
  e.txt.setAttribute("x",p.x);   e.txt.setAttribute("y",p.y);
  if(e.lineEl){{ e.lineEl.setAttribute("x2",p.x); e.lineEl.setAttribute("y2",p.y); }}
  if(e.drectEl&&e.dtxtEl&&r.dist){{
    const dl=distLabel(r);
    const tw=r.dist.length*7+8;
    e.drectEl.setAttribute("x",dl.x-tw/2); e.drectEl.setAttribute("y",dl.y-8);
    e.dtxtEl.setAttribute("x",dl.x);       e.dtxtEl.setAttribute("y",dl.y);
  }}
}}

function makeDraggable(grp,r){{
  let dragging=false, s0x,s0y,p0x,p0y;
  grp.addEventListener("mousedown",e=>{{
    dragging=true;
    const pt=svgPt(e.clientX,e.clientY);
    s0x=pt.x; s0y=pt.y; p0x=pos[r.id].x; p0y=pos[r.id].y;
    e.preventDefault();
  }});
  grp.addEventListener("touchstart",e=>{{
    dragging=true;
    const pt=svgPt(e.touches[0].clientX,e.touches[0].clientY);
    s0x=pt.x; s0y=pt.y; p0x=pos[r.id].x; p0y=pos[r.id].y;
    e.preventDefault();
  }},{{passive:false}});
  window.addEventListener("mousemove",e=>{{
    if(!dragging) return;
    const pt=svgPt(e.clientX,e.clientY);
    pos[r.id]={{x:p0x+pt.x-s0x, y:p0y+pt.y-s0y}};
    updateResidue(r);
  }});
  window.addEventListener("touchmove",e=>{{
    if(!dragging) return;
    const pt=svgPt(e.touches[0].clientX,e.touches[0].clientY);
    pos[r.id]={{x:p0x+pt.x-s0x, y:p0y+pt.y-s0y}};
    updateResidue(r);
    e.preventDefault();
  }},{{passive:false}});
  window.addEventListener("mouseup",()=>{{dragging=false;}});
  window.addEventListener("touchend",()=>{{dragging=false;}});
}}

function drawLegend(){{
  const ly0=H-48;
  const entries=LEG_TYPES.map(t=>{{
    const cfg=TYPE_CFG[t]||TYPE_CFG.hbond;
    const lbl=LEG_LABEL[t]||t;
    const hasLine=!!cfg.line;
    const w=9.54*2+(hasLine?4+28:0)+6+lbl.length*8+16;
    return {{t,cfg,lbl,hasLine,w}};
  }});
  const total=entries.reduce((s,e)=>s+e.w,0);
  let cx2=(W-total)/2;
  const box=attr(ns("rect"),{{x:cx2-8,y:ly0-6,width:total+16,height:42,
    fill:"white",stroke:"#e0e0e0","stroke-width":0.8,rx:6}});
  gLeg.appendChild(box);
  entries.forEach(e=>{{
    const circX=cx2+9.54;
    const circ=attr(ns("circle"),{{cx:circX,cy:ly0+10,r:9.54,
      fill:e.cfg.fill,opacity:0.2}});
    gLeg.appendChild(circ);
    let tx=circX+9.54+6;
    if(e.hasLine){{
      const ln=attr(ns("line"),{{x1:circX+9.54+4,y1:ly0+10,
        x2:circX+9.54+4+28,y2:ly0+10,
        stroke:e.cfg.line,"stroke-width":2,"stroke-dasharray":e.cfg.dash,opacity:0.85}});
      gLeg.appendChild(ln);
      tx=circX+9.54+4+28+6;
    }}
    const tt=attr(ns("text"),{{x:tx,y:ly0+10,"dominant-baseline":"central",
      "font-family":"Arial,sans-serif","font-size":13,"font-weight":700,fill:"#555"}});
    tt.textContent=e.lbl;
    gLeg.appendChild(tt);
    cx2+=e.w;
  }});
}}

window.resetLayout=function(){{ initPos(); drawAll(); }};

window.exportSVG=function(){{
  const clone=svg.cloneNode(true);
  clone.setAttribute("xmlns","http://www.w3.org/2000/svg");
  const blob=new Blob([clone.outerHTML],{{type:"image/svg+xml"}});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob); a.download="interaction_diagram.svg"; a.click();
}};

window.exportPNG=function(){{
  const scale=parseInt(document.getElementById("dpi-sel").value)||2;
  const clone=svg.cloneNode(true);
  clone.setAttribute("xmlns","http://www.w3.org/2000/svg");
  clone.setAttribute("width", W);
  clone.setAttribute("height", H);
  clone.removeAttribute("style");
  const svgStr=new XMLSerializer().serializeToString(clone);
  const svgBlob=new Blob([svgStr],{{type:"image/svg+xml;charset=utf-8"}});
  const url=URL.createObjectURL(svgBlob);
  const img=new Image();
  img.onload=function(){{
    const canvas=document.createElement("canvas");
    canvas.width=W*scale; canvas.height=H*scale;
    const ctx=canvas.getContext("2d");
    ctx.fillStyle="#ffffff";
    ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.scale(scale,scale);
    ctx.drawImage(img,0,0,W,H);
    URL.revokeObjectURL(url);
    const a=document.createElement("a");
    a.download="interaction_diagram.png";
    a.href=canvas.toDataURL("image/png");
    a.click();
  }};
  img.onerror=function(){{
    URL.revokeObjectURL(url);
    alert("PNG export failed — try SVG export instead.");
  }};
  img.src=url;
}};

initPos(); drawAll();
}})();
</script>
"""
    return html


def draw_interactions_rdkit_classic(
    lig_mol,
    receptor_pdb: str,
    smiles: str,
    title: str = "",
    cutoff: float = 3.5,
    size: tuple = (650, 620),
    max_residues: int = 10,
) -> bytes:
    """RDKit highlight-circle style 2D interaction diagram. Returns SVG bytes."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw, rdDepictor, AllChem

    RDLogger.DisableLog("rdApp.*")

    _C_HB = (0.35, 0.61, 0.84, 0.45)
    _C_HP = (0.17, 0.55, 0.34, 0.45)
    _C_OT = (0.80, 0.37, 0.54, 0.45)

    def _parse_robust(smi):
        if not smi: return None
        m = Chem.MolFromSmiles(smi)
        if m: return m
        try:
            m = Chem.MolFromSmiles(smi, sanitize=False)
            if m is None: return None
            m.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(m)
            Chem.SetAromaticity(m)
            m2 = Chem.MolFromSmiles(Chem.MolToSmiles(m))
            return m2 if m2 else m
        except Exception:
            return None

    W, H = size
    mol2d = _parse_robust(smiles)
    if mol2d is None:
        mol2d = Chem.RemoveHs(lig_mol, sanitize=False)
        try: Chem.SanitizeMol(mol2d)
        except: pass

    rdDepictor.Compute2DCoords(mol2d)
    n2d = mol2d.GetNumAtoms()

    mol3d_noH = Chem.RemoveHs(lig_mol, sanitize=False)
    try: Chem.SanitizeMol(mol3d_noH)
    except: pass

    idx3d_to_2d = {}
    try:
        match = mol3d_noH.GetSubstructMatch(mol2d)
        if len(match) == mol2d.GetNumAtoms():
            for i2d, i3d in enumerate(match):
                idx3d_to_2d[i3d] = i2d
    except: pass

    try:
        raw = _detect_all_interactions(lig_mol, receptor_pdb, cutoff=cutoff)
    except:
        raw = []

    for ix in raw:
        ix["lig_atom_idx"] = idx3d_to_2d.get(ix.get("lig_atom_idx", 0), 0)

    pm = {t: i for i, t in enumerate(_ITYPE_PRIORITY)}
    deduped = _deduplicate_interactions(raw)
    deduped.sort(key=lambda x: (pm.get(x["itype"], 99), x["distance"]))
    deduped = deduped[:max_residues]

    if not deduped:
        d2d = Draw.MolDraw2DSVG(W, H)
        d2d.DrawMolecule(mol2d, legend=title or "No interactions found")
        d2d.FinishDrawing()
        RDLogger.EnableLog("rdApp.error")
        return d2d.GetDrawingText().encode()

    lig_ext = Chem.RWMol(mol2d)
    pts, clrs = [], {}

    for ix in deduped:
        itype = ix["itype"]
        ai = ix.get("lig_atom_idx", 0)
        if ai >= n2d: ai = 0

        if itype in ("hbond", "hbond_to_halogen"):
            color = _C_HB
        elif itype == "hydrophobic":
            color = _C_HP
        else:
            color = _C_OT

        rn = ix["resname"]; ri = ix["resid"]; ch = ix.get("chain","")
        _NO_NUM = HEME_RESNAMES | METAL_RESNAMES
        lbl = rn.upper() if rn.upper() in _NO_NUM else f"{rn}{ri}{ch}"

        res_atom = Chem.Atom(0)
        res_atom.SetProp("atomLabel", lbl)
        aid = lig_ext.AddAtom(res_atom)
        pts.append(aid)
        clrs[aid] = color
        lig_ext.AddBond(aid, ai, Chem.BondType.ZERO)

    rdDepictor.Compute2DCoords(lig_ext)

    d2d = Draw.MolDraw2DSVG(W, H)
    opts = d2d.drawOptions()
    opts.circleAtoms = True
    opts.fillHighlights = True
    opts.continuousHighlight = False
    opts.highlightRadius = 0.5
    opts.addAtomIndices = False
    opts.padding = 0.15
    try:
        d2d.SetDrawBounds(0, 0, W, H - 40)
    except AttributeError:
        pass

    d2d.DrawMolecule(lig_ext, highlightAtoms=pts, highlightAtomColors=clrs)
    d2d.FinishDrawing()
    svg_text = d2d.GetDrawingText()

    if title:
        svg_text = _svg_stamp(svg_text, title, W, H)

    RDLogger.EnableLog("rdApp.error")
    return svg_text.encode()


def draw_interactions_rdkit(lig_mol, receptor_pdb: str, smiles: str,
                            title: str="", cutoff: float=3.5,
                            size: tuple=(500,500), max_residues: int=10) -> bytes:
    """Backward-compatible alias → draw_interaction_diagram."""
    import tempfile
    from rdkit import Chem
    tmp=tempfile.NamedTemporaryFile(suffix=".sdf",delete=False)
    with Chem.SDWriter(tmp.name) as w: w.write(lig_mol)
    return draw_interaction_diagram(receptor_pdb=receptor_pdb,pose_sdf=tmp.name,
        smiles=smiles,title=title,cutoff=cutoff,size=(800,759),max_residues=max_residues)


def _svg_stamp(svg_text:str,title:str,w:int,h:int)->str:
    esc=title.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    pad=int(w*0.05); pw=w-2*pad; ph=28; py=h-ph-8; ty=py+ph//2; r=ph//2
    st=(f'<g><rect x="{pad}" y="{py}" width="{pw}" height="{ph}" rx="{r}" ry="{r}"'
        f' fill="#E8E8E8" fill-opacity="0.93" stroke="#C8C8C8" stroke-width="0.5"/>'
        f'<text x="{w//2}" y="{ty}" text-anchor="middle" dominant-baseline="middle"'
        f' font-family="Arial,sans-serif" font-size="13" font-weight="500" fill="#1A1A1A">'
        f'{esc}</text></g>')
    return svg_text.replace("</svg>",f"{st}</svg>")
