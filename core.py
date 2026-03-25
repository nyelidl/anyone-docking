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
        if machine in ("arm64", "aarch64"):
            _FNAME = "vina_1.2.7_mac_arm64"
        else:
            _FNAME = "vina_1.2.7_mac_x86_64"
    elif system == "windows":
        _FNAME = "vina_1.2.7_windows_x86_64.exe"
    else:
        return None, f"Unsupported platform: {system}/{machine}"

    _URL = _BASE + _FNAME

    if not path:
        path = os.path.join(tempfile.gettempdir(), _FNAME)

    if not os.path.exists(path) or os.path.getsize(path) < 100_000:
        try:
            import urllib.request
            urllib.request.urlretrieve(_URL, path)
        except Exception as e1:
            try:
                import requests
                r = requests.get(_URL, stream=True, timeout=120)
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            except Exception as e2:
                return None, f"Download failed: {e1} / {e2}"
    if system != "windows":
        os.chmod(path, 0o755)
    return path, f"ok ({system}/{machine})"


# ══════════════════════════════════════════════════════════════════════════════
#  RECEPTOR PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def detect_cocrystal_ligand(raw_pdb: str) -> dict:
    """
    Parse PDB and return the best co-crystal ligand candidate.
    Returns dict with keys:
        found, resname, chain, resid, sel_str, ligand_id,
        cx, cy, cz, n_atoms, atoms
    """
    from prody import parsePDB, calcCenter

    atoms = parsePDB(raw_pdb)
    if atoms is None:
        return {"found": False}

    excl = EXCLUDE_IONS | GLYCAN_NAMES | COFACTOR_NAMES
    het  = atoms.select("hetero and not water")
    if het is None:
        return {"found": False}

    cands = [
        r for r in het.getHierView().iterResidues()
        if (r.getResname() or "").strip() not in excl
    ]
    if not cands:
        return {"found": False}

    cands.sort(key=lambda r: (-r.numAtoms(), r.getChid() != "A"))
    chosen     = cands[0]
    rn         = chosen.getResname()
    ch         = chosen.getChid()
    ri         = chosen.getResnum()
    sel_str    = f"resname {rn} and resid {ri} and chain {ch}"
    lig_atoms  = atoms.select(sel_str)
    cx, cy, cz = (float(v) for v in calcCenter(lig_atoms))

    return {
        "found":     True,
        "resname":   rn,
        "chain":     ch,
        "resid":     ri,
        "sel_str":   sel_str,
        "ligand_id": f"{rn}_{ch}_{ri}",
        "cx": cx, "cy": cy, "cz": cz,
        "n_atoms":   lig_atoms.numAtoms(),
        "atoms":     lig_atoms,
    }


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
) -> dict:
    """
    Full receptor preparation pipeline.
    Accepts PDB or mmCIF (.cif) files — CIF is auto-converted to PDB first.
    center_mode: 'auto' | 'manual' | 'selection'
    Returns dict: success, rec_fh, rec_pdbqt, box_pdb, config_txt,
                  cx, cy, cz, sx, sy, sz, ligand_pdb_path,
                  cocrystal_ligand_id, n_atoms, log, error
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

        if center_mode == "auto":
            info = detect_cocrystal_ligand(raw_pdb)
            if info["found"]:
                rn, ch, ri          = info["resname"], info["chain"], info["resid"]
                ligand_sel_str      = info["sel_str"]
                cocrystal_ligand_id = info["ligand_id"]
                cx, cy, cz          = info["cx"], info["cy"], info["cz"]
                ligand_pdb_path     = str(wdir / "LIG.pdb")
                writePDB(ligand_pdb_path, info["atoms"])
                log.append(
                    f"✓ Co-crystal ligand: {rn} chain {ch} resnum {ri} "
                    f"({info['n_atoms']} atoms)"
                )
                log.append(f"📍 Center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
                log.append(f"🔑 PoseView2 ligand ID: {cocrystal_ligand_id}")
            else:
                log.append("⚠ No co-crystal ligand found after filtering")

        elif center_mode == "manual":
            cx, cy, cz = (float(v) for v in manual_xyz)
            log.append(f"🛠 Manual center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")

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

            _resnames = list(dict.fromkeys(ref_atoms.getResnames()))
            _resids   = list(dict.fromkeys(ref_atoms.getResnums()))
            _chains   = list(dict.fromkeys(ref_atoms.getChids()))

            if len(_resnames) == 1 and len(_resids) == 1:
                rn = _resnames[0]
                ri = int(_resids[0])
                ch = _chains[0] if _chains else "A"
                ligand_sel_str      = f"resname {rn} and resid {ri} and chain {ch}"
                cocrystal_ligand_id = f"{rn}_{ch}_{ri}"
                ligand_pdb_path     = str(wdir / "LIG.pdb")
                writePDB(ligand_pdb_path, ref_atoms)
                log.append(f"✓ Ligand: {rn} chain {ch} resnum {ri}")
                log.append(f"🔑 PoseView2 ligand ID: {cocrystal_ligand_id}")
            else:
                ligand_pdb_path = str(wdir / "LIG_ref.pdb")
                writePDB(ligand_pdb_path, ref_atoms)
                log.append("⚠ Multi-residue selection — PoseView2 ligand ID not set")

        # Write receptor PDB without co-crystal ligand
        sel_str = (
            f"not ({ligand_sel_str}) and not water"
            if ligand_sel_str else "not water"
        )
        rec_sel = atoms.select(sel_str)
        if rec_sel is None or rec_sel.numAtoms() == 0:
            raise ValueError("Receptor selection returned no atoms")

        rec_raw_path = str(wdir / "receptor_atoms.pdb")
        writePDB(rec_raw_path, rec_sel)
        log.append(f"✓ Receptor: {rec_sel.numAtoms()} atoms")

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
            "log":                 log,
        }

    except Exception as e:
        log.append(f"ERROR: {e}")
        return {"success": False, "error": str(e), "log": log}


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


def prepare_ligand(smiles: str, name: str, ph: float, wdir) -> dict:
    """
    Protonate at target pH → 3D conformer (ETKDGv3) → MMFF/UFF minimise
    → PDBQT (Meeko) + SDF.
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
        prot = smiles.strip()

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
            raise ValueError(f"RDKit could not parse SMILES: {prot[:60]}")

        charge = Chem.GetFormalCharge(mol)
        log.append(f"✓ Formal charge: {charge:+d}")

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

    FIX: Adds Hs with addCoords=True to preserve docked pose coordinates.
    Validates that no H atom landed at origin before writing with Hs.
    Falls back to heavy-atom-only mol if addCoords fails.

    Returns list of log messages.
    """
    import shutil
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    # Suppress kekulize noise from raw Vina SDF
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

            # Add Hs with real 3D coordinates (preserves docked pose)
            try:
                fixed_h = Chem.AddHs(fixed, addCoords=True)
                conf    = fixed_h.GetConformer()
                # Reject if any H landed at origin (addCoords=True failed silently)
                bad = any(
                    abs(conf.GetAtomPosition(j).x)
                    + abs(conf.GetAtomPosition(j).y)
                    + abs(conf.GetAtomPosition(j).z) < 0.01
                    for j in range(fixed_h.GetNumAtoms())
                    if fixed_h.GetAtomWithIdx(j).GetAtomicNum() == 1
                )
                writer.write(fixed if bad else fixed_h)
            except Exception:
                # Fallback: heavy atoms only — PoseView can add its own Hs
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
    """
    Load all valid mols from an SDF file.

    FIX: Suppresses RDKit kekulize / sanitize noise from raw Vina output SDFs.
    Falls back to per-mol sanitize on failure so poses are never silently lost.
    """
    from rdkit import Chem, RDLogger

    # Suppress "Can't kekulize" / sanitize errors —
    # Vina SDF aromatic bonds cause these on every pose when sanitize=True.
    RDLogger.DisableLog("rdApp.*")

    mols = []
    try:
        sup = Chem.SDMolSupplier(sdf_path, sanitize=sanitize, removeHs=False)
        for m in sup:
            if m is not None:
                mols.append(m)
    except Exception:
        pass

    # If sanitize=True produced 0 (or fewer than raw), fall back per-mol
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
                        pass   # keep mol even if partially unsanitized
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
    """
    Convert an SDF to Kekulized V2000 format suitable for PoseView.

    PoseView's bundled CDK mol parser requires:
      - V2000 format (not V3000)
      - Fully Kekulized bonds: explicit single/double (type 1/2), NOT aromatic
        bond type 4.  obabel preserves aromatic type 4 which PoseView rejects.
      - No explicit Hs (PoseView adds its own for H-bond detection)

    Strategy:
      1. RDKit: read → RemoveHs → Kekulize → SDWriter(SetKekulize=True)
         This is the canonical fix — guarantees type-1/2 bond encoding.
      2. Fallback: obabel plain format conversion (no -h, no --gen3d).

    Returns path to converted file, or original sdf_path on failure.
    """
    from rdkit import Chem, RDLogger

    out = sdf_path.replace(".sdf", "_v2000.sdf")
    if out == sdf_path:
        out = sdf_path + "_v2000.sdf"

    # ── Strategy 1: RDKit Kekulized V2000 ─────────────────────────────────────
    RDLogger.DisableLog("rdApp.*")
    try:
        # Try sanitized read first; fall back to unsanitized
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
            # Kekulize: convert aromatic bonds → alternating single/double
            # PoseView rejects MDL bond type 4 (aromatic)
            try:
                Chem.Kekulize(mol_noH, clearAromaticFlags=True)
            except Exception:
                pass  # SDWriter will attempt its own Kekulize

            w = Chem.SDWriter(out)
            w.SetKekulize(True)   # write type-1/2 bonds only, never type-4
            w.write(mol_noH)
            w.close()

            if os.path.exists(out) and os.path.getsize(out) > 10:
                RDLogger.EnableLog("rdApp.error")
                return out
    except Exception:
        pass
    RDLogger.EnableLog("rdApp.error")

    # ── Strategy 2: obabel plain conversion (no -h, no --gen3d) ──────────────
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
    """
    Poll a ProteinsPlus job until it leaves pending/running state.
    Returns the final job dict. Raises RuntimeError on timeout.
    """
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
    """
    Write a minimal, clean PDB suitable for PoseView's CDK-based parser.

    PoseView's Java engine is strict:
      - Accepts only ATOM / HETATM / TER / END records
      - Chokes on ANISOU, SIGUIJ, CONECT, non-standard REMARK variants
      - Must have no explicit Hs (PoseView adds its own for H-bond detection)
      - Serial numbers must be contiguous integers

    Source priority:
      1. receptor_atoms.pdb in the same directory  — ProDy writePDB, cleanest
      2. The passed receptor_pdb (obabel-processed rec.pdb)

    Returns path to the clean PDB, or original path on any failure.
    """
    # Prefer ProDy's clean output over obabel's processed file
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

                # Only standard coordinate records
                if rec not in ("ATOM", "HETATM", "TER", "END"):
                    continue

                if rec in ("ATOM", "HETATM"):
                    # Strip hydrogen atoms
                    atom_name = line[12:16].strip()
                    element   = line[76:78].strip() if len(line) > 76 else ""
                    if element.upper() == "H" or (not element and atom_name.startswith("H")):
                        continue

                    # Renumber serials to stay contiguous (PDBs from obabel
                    # can have gaps after H removal that confuse CDK)
                    serial += 1
                    line = f"{line[:6]}{serial:5d}{line[11:]}"

                kept.append(line if line.endswith("\n") else line + "\n")

        if not kept:
            return receptor_pdb

        # Ensure file ends with END
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

    Submits a clean receptor PDB + raw pose SDF directly to PoseView v1.

    KEY FINDING: the raw pose SDF (as downloaded by the user) works when
    uploaded manually to proteins.plus. All previous SDF transforms
    (fix_sdf_bond_orders → convert_sdf_to_v2000) were corrupting the file.
    Solution: send pose_sdf as-is — zero transformation.

    Protein: _prepare_pdb_for_poseview strips non-coordinate records,
    removes Hs, renumbers serials — using receptor_atoms.pdb (ProDy) as
    source where available (cleaner than obabel-processed rec.pdb).

    Flow:
      1 — _prepare_pdb_for_poseview → clean PDB
      2 — POST receptor + raw pose_sdf directly to /api/v2/poseview/
      3 — Poll → GET job['image'] SVG

    Returns (svg_bytes, error_string) — one will be None.
    """
    import requests

    last_error = "Unknown error"

    # Clean the receptor once — CDK-safe: only ATOM/HETATM/TER/END,
    # no Hs, contiguous serials, sourced from ProDy receptor_atoms.pdb.
    rec_to_send = _prepare_pdb_for_poseview(receptor_pdb)

    for attempt in range(1, _PV_MAX_RETRIES + 1):
        if attempt > 1:
            time.sleep(_PV_RETRY_DELAY)

        # ── POST receptor + raw SDF directly to PoseView ──────────────────────
        # IMPORTANT: send pose_sdf as-is — no bond-order fixing, no V2000
        # conversion. The raw Vina→obabel SDF is exactly what the user
        # downloads and what works when uploaded manually. Any transform
        # (fix_sdf_bond_orders, convert_sdf_to_v2000) was corrupting the file.
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

        # ── Step 3: Poll + fetch SVG ──────────────────────────────────────────
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
    """
    No-op — call_poseview_v1 now posts the receptor directly to PoseView
    without a MoleculeHandler/Protoss pre-upload step, so there is nothing
    to pre-cache.  Kept for API compatibility with app.py.
    """
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
    """
    Burn a centred label pill into the bottom of a PNG.
    Returns original bytes unchanged on any error.
    """
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
    """
    Test the PoseView API with a known-good minimal PDB + SDF from RCSB,
    completely independent of the user's receptor/ligand files.
    Uses PDB 4AGN + its native ligand NXG fetched live from proteins.plus.
    """
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
#  CUSTOM 2D INTERACTION DIAGRAM  —  Clean circle layout, no crossing lines
#
#  Design:
#    • Residues drawn as colored circles  (triplet-code + ID inside)
#    • Background color  =  interaction type
#    • H-bond lines show distance label on the dashed line
#    • Layout: angular-sort  →  lines never cross  (no spider-web)
#    • Ligand: full PoseView-style atomic structure
# ══════════════════════════════════════════════════════════════════════════════

import math as _math

# ── Interaction type priority (higher = displayed over lower) ─────────────────
_ITYPE_PRIORITY = [
    "metal", "ionic", "halogen", "hbond_to_halogen",
    "hbond", "pi_pi", "cation_pi", "hydrophobic",
]

# ── Per-interaction-type visual spec ─────────────────────────────────────────
#   circle_fill / circle_stroke  =  residue circle
#   line_stroke                  =  interaction line color
#   label_color                  =  text inside circle
#   dash                         =  SVG stroke-dasharray  (empty = solid)
#   show_dist                    =  show distance on line
_ITYPE_VIS = {
    "hbond":            dict(
        circle_fill="#dbeeff", circle_stroke="#1a5fa8",
        label_color="#0a3a70", line_stroke="#1a5fa8",
        dash="6,3", lw=1.8, show_dist=True,
        legend="H-bond",
    ),
    "hydrophobic":      dict(
        circle_fill="#d6f0d6", circle_stroke="#1a7a1a",
        label_color="#0d4a0d", line_stroke="#1a7a1a",
        dash="", lw=1.6, show_dist=False,
        legend="Hydrophobic",
    ),
    "pi_pi":            dict(
        circle_fill="#e8e4ff", circle_stroke="#5533bb",
        label_color="#2d1a88", line_stroke="#5533bb",
        dash="5,3", lw=1.6, show_dist=False,
        legend="π-π stack",
    ),
    "cation_pi":        dict(
        circle_fill="#fff2d6", circle_stroke="#a06000",
        label_color="#5c3800", line_stroke="#a06000",
        dash="5,3", lw=1.6, show_dist=False,
        legend="Cation-π",
    ),
    "ionic":            dict(
        circle_fill="#ffe0f0", circle_stroke="#aa0077",
        label_color="#660044", line_stroke="#aa0077",
        dash="6,2,2,2", lw=1.8, show_dist=True,
        legend="Ionic",
    ),
    "metal":            dict(
        circle_fill="#fff5d0", circle_stroke="#996600",
        label_color="#4d3300", line_stroke="#cc8800",
        dash="3,2", lw=1.8, show_dist=True,
        legend="Metal",
    ),
    "halogen":          dict(
        circle_fill="#ffe0ee", circle_stroke="#aa2266",
        label_color="#660033", line_stroke="#cc2277",
        dash="5,2", lw=1.6, show_dist=True,
        legend="Halogen bond",
    ),
    "hbond_to_halogen": dict(
        circle_fill="#ede0ff", circle_stroke="#6633aa",
        label_color="#3d1a66", line_stroke="#6633aa",
        dash="4,2,1,2", lw=1.6, show_dist=True,
        legend="H···Halogen",
    ),
}

# ── Ligand atom colors (PoseView palette) ─────────────────────────────────────
_ATOM_CLR = {
    "C": "#1a1a1a", "N": "#1a5fa8", "O": "#cc2222",
    "S": "#c8a800", "P": "#e07000", "F": "#1a7a1a",
    "CL": "#1a7a1a", "BR": "#8b2500", "I": "#5c2d8a",
    "H": "#666666",
}
_METALS_SET = {
    "MG","ZN","CA","MN","FE","CU","CO","NI","CD","HG","NA","K",
}
_AROM_ATOMS = {"PHE","TYR","TRP","HIS"}
_AROM_ATOM_NAMES = {
    "CG","CD1","CD2","CE1","CE2","CZ",
    "ND1","NE2","CE3","CZ2","CZ3","CH2",
}


# ──────────────────────────────────────────────────────────────────────────────
#  GEOMETRY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _get_aromatic_ring_data(mol, conf):
    import numpy as np
    results = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) not in (5, 6):
            continue
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        coords   = np.array([[conf.GetAtomPosition(i).x,
                               conf.GetAtomPosition(i).y,
                               conf.GetAtomPosition(i).z] for i in ring])
        centroid = coords.mean(axis=0)
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        normal = np.cross(v1, v2)
        n = np.linalg.norm(normal)
        if n > 0:
            normal /= n
        results.append((centroid, normal))
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  INTERACTION DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def _detect_all_interactions(lig_mol_3d, receptor_pdb: str,
                              cutoff: float = 4.5) -> list:
    import numpy as np
    from prody import parsePDB

    rec = parsePDB(receptor_pdb)
    if rec is None:
        return []

    rec_coords   = rec.getCoords()
    rec_resnames = rec.getResnames()
    rec_chains   = rec.getChids()
    rec_resids   = rec.getResnums()
    rec_names    = rec.getNames()
    rec_elements = rec.getElements()

    conf     = lig_mol_3d.GetConformer()
    n_lig    = lig_mol_3d.GetNumAtoms()
    lig_xyz  = np.array([[conf.GetAtomPosition(i).x,
                          conf.GetAtomPosition(i).y,
                          conf.GetAtomPosition(i).z] for i in range(n_lig)])
    lig_atoms    = [lig_mol_3d.GetAtomWithIdx(i) for i in range(n_lig)]
    lig_el       = [a.GetSymbol().upper() for a in lig_atoms]
    lig_aromatic = [a.GetIsAromatic()         for a in lig_atoms]
    lig_charge   = [a.GetFormalCharge()        for a in lig_atoms]

    POLAR   = {"N", "O", "S", "F"}
    HYDRO_L = {"C", "S", "CL", "BR", "I", "F"}
    HYDRO_R = {
        "ALA","VAL","ILE","LEU","MET","PHE","TRP",
        "PRO","GLY","TYR","HIS",
    }
    results = []

    for j in range(len(rec_coords)):
        rn  = rec_resnames[j].strip()
        ch  = rec_chains[j].strip()
        ri  = int(rec_resids[j])
        el  = (rec_elements[j].strip().upper()
               if rec_elements[j] and rec_elements[j].strip()
               else rec_names[j][:1].upper())
        rp  = rec_coords[j]

        dists    = np.linalg.norm(lig_xyz - rp, axis=1)
        min_dist = float(dists.min())
        min_idx  = int(dists.argmin())

        if min_dist > max(cutoff + 1.0, 5.6):
            continue

        # H-bond
        if el in POLAR:
            for i in range(n_lig):
                if lig_el[i] not in POLAR:
                    continue
                d = float(dists[i])
                if d < 3.5:
                    results.append(dict(resname=rn, chain=ch, resid=ri,
                                        itype="hbond", distance=round(d,2),
                                        lig_atom_idx=i,
                                        prot_el=el, is_donor=(el=="N")))
                    break

        # Hydrophobic
        if el in {"C","S","CL","BR","I"} and rn in HYDRO_R:
            for i in range(n_lig):
                if lig_el[i] not in HYDRO_L:
                    continue
                d = float(dists[i])
                if d < cutoff:
                    results.append(dict(resname=rn, chain=ch, resid=ri,
                                        itype="hydrophobic", distance=round(d,2),
                                        lig_atom_idx=i,
                                        prot_el=el, is_donor=False))
                    break

        # Ionic
        if rn in {"ASP","GLU"} and el == "O":
            for i in range(n_lig):
                if lig_charge[i] > 0 and float(dists[i]) < 4.0:
                    results.append(dict(resname=rn, chain=ch, resid=ri,
                                        itype="ionic", distance=round(float(dists[i]),2),
                                        lig_atom_idx=i, prot_el=el, is_donor=False))
                    break
        if rn in {"LYS","ARG"} and el == "N":
            for i in range(n_lig):
                if lig_charge[i] < 0 and float(dists[i]) < 4.0:
                    results.append(dict(resname=rn, chain=ch, resid=ri,
                                        itype="ionic", distance=round(float(dists[i]),2),
                                        lig_atom_idx=i, prot_el=el, is_donor=True))
                    break

        # Metal
        if rn.strip().upper() in _METALS_SET or el in _METALS_SET:
            if min_dist < 2.8:
                results.append(dict(
                    resname=rn if rn.strip().upper() in _METALS_SET else el.capitalize(),
                    chain=ch, resid=ri,
                    itype="metal", distance=round(min_dist,2),
                    lig_atom_idx=min_idx, prot_el=el, is_donor=False))

    # π-π
    lig_rings = _get_aromatic_ring_data(lig_mol_3d, conf)
    if lig_rings:
        for j in range(len(rec_coords)):
            rn = rec_resnames[j].strip()
            ch = rec_chains[j].strip()
            ri = int(rec_resids[j])
            an = rec_names[j].strip()
            if rn not in _AROM_ATOMS or an not in _AROM_ATOM_NAMES:
                continue
            rp = rec_coords[j]
            for lig_centroid, _ in lig_rings:
                dist = float(np.linalg.norm(lig_centroid - rp))
                if dist < 5.5:
                    best_i = min(
                        (i for i in range(n_lig) if lig_aromatic[i]),
                        key=lambda i: float(np.linalg.norm(lig_xyz[i] - rp)),
                        default=0,
                    )
                    results.append(dict(resname=rn, chain=ch, resid=ri,
                                        itype="pi_pi", distance=round(dist,2),
                                        lig_atom_idx=best_i,
                                        prot_el="C", is_donor=False))
                    break

    # Cation-π
    if lig_rings:
        for j in range(len(rec_coords)):
            rn = rec_resnames[j].strip()
            ch = rec_chains[j].strip()
            ri = int(rec_resids[j])
            el = (rec_elements[j].strip().upper()
                  if rec_elements[j] and rec_elements[j].strip()
                  else rec_names[j][:1].upper())
            if rn not in {"LYS","ARG"} or el != "N":
                continue
            rp = rec_coords[j]
            for lig_centroid, _ in lig_rings:
                dist = float(np.linalg.norm(lig_centroid - rp))
                if dist < 5.0:
                    best_i = min(
                        (i for i in range(n_lig) if lig_aromatic[i]),
                        key=lambda i: float(np.linalg.norm(lig_xyz[i] - rp)),
                        default=0,
                    )
                    results.append(dict(resname=rn, chain=ch, resid=ri,
                                        itype="cation_pi", distance=round(dist,2),
                                        lig_atom_idx=best_i,
                                        prot_el="N", is_donor=True))
                    break

    # VDW radii
    _VDW = {
        "H":1.20,"C":1.70,"N":1.55,"O":1.52,"S":1.80,
        "P":1.80,"F":1.47,"CL":1.75,"BR":1.85,"I":1.98,
    }

    # Halogen bond  C-X···A  (X = Cl/Br/I;  angle R-X···A >= 140°)
    _XB_DONORS    = {17:"CL", 35:"BR", 53:"I"}
    _XB_ACCEPTORS = {"O","N","S","P","F","CL","BR","I"}

    for i in range(n_lig):
        ano = lig_atoms[i].GetAtomicNum()
        if ano not in _XB_DONORS:
            continue
        x_el  = _XB_DONORS[ano]
        x_pos = lig_xyz[i]
        vdw_x = _VDW.get(x_el, 1.80)
        r_idx = next(
            (nb.GetIdx() for nb in lig_atoms[i].GetNeighbors()
             if nb.GetAtomicNum() == 6), None)
        if r_idx is None:
            continue
        r_pos = lig_xyz[r_idx]
        for j in range(len(rec_coords)):
            a_el = (rec_elements[j].strip().upper()
                    if rec_elements[j] and rec_elements[j].strip()
                    else rec_names[j][:1].upper())
            is_pi = (rec_resnames[j].strip() in _AROM_ATOMS
                     and rec_names[j].strip() in _AROM_ATOM_NAMES
                     and a_el == "C")
            if a_el not in _XB_ACCEPTORS and not is_pi:
                continue
            a_pos = rec_coords[j]
            dist  = float(np.linalg.norm(x_pos - a_pos))
            if dist > vdw_x + _VDW.get(a_el, 1.70):
                continue
            vRX   = r_pos - x_pos
            vXA   = a_pos - x_pos
            cos_a = np.dot(vRX, vXA) / (
                np.linalg.norm(vRX)*np.linalg.norm(vXA) + 1e-9)
            if _math.degrees(_math.acos(max(-1.0, min(1.0, cos_a)))) >= 140:
                rn = rec_resnames[j].strip()
                ch = rec_chains[j].strip()
                ri = int(rec_resids[j])
                results.append(dict(resname=rn, chain=ch, resid=ri,
                                    itype="halogen", distance=round(dist,2),
                                    lig_atom_idx=i, prot_el=a_el, is_donor=False))

    # H-bond to halogen  D-H···X
    _HBX_DONORS_REC = {"O","N","S"}
    _HBX_ACC_ANO    = {9:"F", 17:"CL", 35:"BR", 53:"I"}

    for i in range(n_lig):
        ano = lig_atoms[i].GetAtomicNum()
        if ano not in _HBX_ACC_ANO:
            continue
        x_el  = _HBX_ACC_ANO[ano]
        x_pos = lig_xyz[i]
        vdw_x = _VDW.get(x_el, 1.80)
        r_idx = next((nb.GetIdx() for nb in lig_atoms[i].GetNeighbors()), None)
        if r_idx is None:
            continue
        r_lig_pos = lig_xyz[r_idx]
        for j in range(len(rec_coords)):
            h_el = (rec_elements[j].strip().upper()
                    if rec_elements[j] and rec_elements[j].strip()
                    else rec_names[j][:1].upper())
            if h_el != "H":
                continue
            h_pos   = rec_coords[j]
            dist_hx = float(np.linalg.norm(h_pos - x_pos))
            if dist_hx > _VDW["H"] + vdw_x:
                continue
            d_pos = None
            for k in range(len(rec_coords)):
                if k == j:
                    continue
                d_el_k = (rec_elements[k].strip().upper()
                          if rec_elements[k] and rec_elements[k].strip()
                          else rec_names[k][:1].upper())
                if d_el_k not in _HBX_DONORS_REC:
                    continue
                if float(np.linalg.norm(rec_coords[k] - h_pos)) < 1.15:
                    d_pos = rec_coords[k]
                    break
            if d_pos is None:
                continue
            vHD = d_pos  - h_pos
            vHX = x_pos  - h_pos
            cos_d = np.dot(vHD, vHX) / (
                np.linalg.norm(vHD)*np.linalg.norm(vHX) + 1e-9)
            ang_dhx = _math.degrees(_math.acos(max(-1.0, min(1.0, cos_d))))
            if ang_dhx < 120:
                continue
            vXR = r_lig_pos - x_pos
            vXH = h_pos     - x_pos
            cos_r = np.dot(vXR, vXH) / (
                np.linalg.norm(vXR)*np.linalg.norm(vXH) + 1e-9)
            ang_rxh = _math.degrees(_math.acos(max(-1.0, min(1.0, cos_r))))
            if not (70 <= ang_rxh <= 120):
                continue
            rn = rec_resnames[j].strip()
            ch = rec_chains[j].strip()
            ri = int(rec_resids[j])
            results.append(dict(resname=rn, chain=ch, resid=ri,
                                itype="hbond_to_halogen", distance=round(dist_hx,2),
                                lig_atom_idx=i, prot_el="N", is_donor=True))

    return results


def _deduplicate_interactions(interactions: list) -> list:
    priority = {t: i for i, t in enumerate(_ITYPE_PRIORITY)}
    best: dict = {}
    for ix in interactions:
        key = (ix["chain"], ix["resid"])
        if key not in best:
            best[key] = ix
        else:
            p_new = priority.get(ix["itype"],   99)
            p_old = priority.get(best[key]["itype"], 99)
            if p_new < p_old or (p_new == p_old and ix["distance"] < best[key]["distance"]):
                best[key] = ix
    return list(best.values())


# ──────────────────────────────────────────────────────────────────────────────
#  2D COORDINATE MAPPING
# ──────────────────────────────────────────────────────────────────────────────

def _compute_svg_coords(mol2d, cx: float, cy: float,
                         target_size: float = 195) -> dict:
    from rdkit.Chem import rdDepictor
    if mol2d.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(mol2d)
    conf = mol2d.GetConformer()
    n    = mol2d.GetNumAtoms()
    if n == 0:
        return {}
    xs = [conf.GetAtomPosition(i).x for i in range(n)]
    ys = [conf.GetAtomPosition(i).y for i in range(n)]
    span = max(max(xs)-min(xs), max(ys)-min(ys), 0.01)
    sc   = target_size / span
    mx   = (min(xs)+max(xs)) / 2
    my   = (min(ys)+max(ys)) / 2
    return {i: (cx + (xs[i]-mx)*sc, cy - (ys[i]-my)*sc) for i in range(n)}


# ──────────────────────────────────────────────────────────────────────────────
#  NO-CROSSING PLACEMENT
#
#  Key insight:  if we sort residues by the angle of their associated ligand
#  atom (from ligand centroid), and assign radial positions in the SAME angular
#  order, the straight lines from ligand atoms to residue circles can never
#  cross each other.
# ──────────────────────────────────────────────────────────────────────────────

def _place_residues_no_cross(interactions: list,
                              svg_coords: dict,
                              cx: float, cy: float,
                              R: float = 235,
                              circ_r: float = 32) -> list:
    """
    Sort residues by the angle of their associated ligand atom,
    then distribute them evenly on a circle of radius R in that order.
    Result: interaction lines never cross.
    """
    import numpy as np

    if not interactions:
        return []

    # 1. Assign each residue an anchor angle from its closest ligand atom
    anchored = []
    for ix in interactions:
        ai     = ix.get("lig_atom_idx", 0)
        ax, ay = svg_coords.get(ai, (cx, cy))
        angle  = _math.atan2(ay - cy, ax - cx)
        anchored.append({**ix, "anchor_angle": angle})

    # 2. Sort by anchor angle — this is the key anti-crossing step
    anchored.sort(key=lambda x: x["anchor_angle"])
    n = len(anchored)

    # 3. Find the largest angular gap between consecutive anchor angles
    #    and start the ring placement there (keeps labels in the gap)
    angles = [a["anchor_angle"] for a in anchored]
    gaps   = []
    for k in range(n):
        a_cur  = angles[k]
        a_next = angles[(k+1) % n]
        gap    = (a_next - a_cur) % (2*_math.pi)
        gaps.append((gap, k))
    _, start_k = max(gaps) if gaps else (0, 0)

    # 4. Distribute residue circles evenly, starting after the largest gap
    placements = []
    for offset in range(n):
        idx_in_sorted = (start_k + 1 + offset) % n
        place_angle   = -_math.pi/2 + (2*_math.pi * offset / n)
        ix  = anchored[idx_in_sorted]
        bx  = cx + R * _math.cos(place_angle)
        by  = cy + R * _math.sin(place_angle)
        placements.append({
            **ix,
            "bx": bx,
            "by": by,
            "circ_r": circ_r,
            "place_angle": place_angle,
        })

    # 5. Re-sort placements back to match the sorted order so lines don't cross
    #    Assign each placement slot to the residue whose anchor angle is closest
    #    to the slot's place_angle — Hungarian-style greedy match.
    slot_angles = [-_math.pi/2 + (2*_math.pi*k/n) for k in range(n)]
    assigned    = [None] * n
    used        = [False] * n
    for res in anchored:
        aa = res["anchor_angle"]
        # find closest unused slot
        best_slot, best_d = 0, float("inf")
        for s in range(n):
            if used[s]:
                continue
            d = abs((_math.atan2(_math.sin(slot_angles[s]-aa),
                                  _math.cos(slot_angles[s]-aa))))
            if d < best_d:
                best_d, best_slot = d, s
        used[best_slot] = True
        bx = cx + R * _math.cos(slot_angles[best_slot])
        by = cy + R * _math.sin(slot_angles[best_slot])
        assigned[best_slot] = {
            **res,
            "bx": bx, "by": by,
            "circ_r": circ_r,
            "place_angle": slot_angles[best_slot],
        }

    return [a for a in assigned if a is not None]


# ──────────────────────────────────────────────────────────────────────────────
#  LIGAND SVG — PoseView atomic style
# ──────────────────────────────────────────────────────────────────────────────

def _render_ligand_svg(mol2d, svg_coords: dict) -> str:
    from rdkit import Chem
    parts = []
    ring_info  = mol2d.GetRingInfo()
    arom_bonds = set()
    for ring in ring_info.AtomRings():
        if all(mol2d.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            for k in range(len(ring)):
                arom_bonds.add(frozenset([ring[k], ring[(k+1)%len(ring)]]))

    def _shorten(fx, fy, tx, ty, sym):
        """Pull start point away from a heteroatom circle edge."""
        if sym not in ("C","H",""):
            dx, dy = tx-fx, ty-fy
            L  = _math.sqrt(dx*dx+dy*dy) + 1e-9
            r  = 11 if len(sym)<=1 else 14
            return fx + dx/L*r, fy + dy/L*r
        return fx, fy

    # Bonds
    for bond in mol2d.GetBonds():
        i1, i2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        x1, y1 = svg_coords.get(i1, (0,0))
        x2, y2 = svg_coords.get(i2, (0,0))
        s1     = mol2d.GetAtomWithIdx(i1).GetSymbol().upper()
        s2     = mol2d.GetAtomWithIdx(i2).GetSymbol().upper()
        x1s,y1s = _shorten(x1,y1,x2,y2,s1)
        x2s,y2s = _shorten(x2,y2,x1,y1,s2)
        bt = bond.GetBondType()

        if frozenset([i1,i2]) in arom_bonds:
            parts.append(
                f'<line x1="{x1s:.1f}" y1="{y1s:.1f}" x2="{x2s:.1f}" y2="{y2s:.1f}"'
                f' stroke="#1a1a1a" stroke-width="1.9" opacity="0.9"/>')
        elif bt == Chem.BondType.DOUBLE:
            dx,dy = x2s-x1s, y2s-y1s
            L  = _math.sqrt(dx*dx+dy*dy)+1e-9
            px,py = -dy/L*2.5, dx/L*2.5
            for sgn in (1,-1):
                parts.append(
                    f'<line x1="{x1s+px*sgn:.1f}" y1="{y1s+py*sgn:.1f}"'
                    f' x2="{x2s+px*sgn:.1f}" y2="{y2s+py*sgn:.1f}"'
                    f' stroke="#1a1a1a" stroke-width="1.5" opacity="0.9"/>')
        elif bt == Chem.BondType.TRIPLE:
            dx,dy = x2s-x1s, y2s-y1s
            L = _math.sqrt(dx*dx+dy*dy)+1e-9
            px,py = -dy/L*3.0, dx/L*3.0
            for m in (-1,0,1):
                parts.append(
                    f'<line x1="{x1s+px*m:.1f}" y1="{y1s+py*m:.1f}"'
                    f' x2="{x2s+px*m:.1f}" y2="{y2s+py*m:.1f}"'
                    f' stroke="#1a1a1a" stroke-width="1.3" opacity="0.9"/>')
        else:
            bdir = bond.GetBondDir()
            if bdir == Chem.BondDir.BEGINWEDGE:
                dx,dy = x2s-x1s, y2s-y1s
                L = _math.sqrt(dx*dx+dy*dy)+1e-9
                px,py = -dy/L*3.5, dx/L*3.5
                parts.append(
                    f'<polygon points="{x1s:.1f},{y1s:.1f} '
                    f'{x2s+px:.1f},{y2s+py:.1f} {x2s-px:.1f},{y2s-py:.1f}"'
                    f' fill="#1a1a1a" stroke="none"/>')
            elif bdir == Chem.BondDir.BEGINDASH:
                dx,dy = x2s-x1s, y2s-y1s
                L = _math.sqrt(dx*dx+dy*dy)+1e-9
                px,py = -dy/L, dx/L
                for step in range(1,6):
                    t  = step/7
                    mx = x1s+dx*t; my = y1s+dy*t
                    w  = t*3.5
                    parts.append(
                        f'<line x1="{mx-px*w:.1f}" y1="{my-py*w:.1f}"'
                        f' x2="{mx+px*w:.1f}" y2="{my+py*w:.1f}"'
                        f' stroke="#1a1a1a" stroke-width="1.2"/>')
            else:
                parts.append(
                    f'<line x1="{x1s:.1f}" y1="{y1s:.1f}" x2="{x2s:.1f}" y2="{y2s:.1f}"'
                    f' stroke="#1a1a1a" stroke-width="1.9" opacity="0.9"/>')

    # Aromatic circles
    for ring in ring_info.AtomRings():
        if not all(mol2d.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        rcoords = [svg_coords.get(i,(0,0)) for i in ring]
        rcx = sum(x for x,y in rcoords)/len(rcoords)
        rcy = sum(y for x,y in rcoords)/len(rcoords)
        avg = sum(_math.sqrt((x-rcx)**2+(y-rcy)**2) for x,y in rcoords)/len(rcoords)
        parts.append(
            f'<circle cx="{rcx:.1f}" cy="{rcy:.1f}" r="{avg*0.55:.1f}"'
            f' fill="none" stroke="#1a1a1a" stroke-width="1.2"'
            f' stroke-dasharray="3,2" opacity="0.6"/>')

    # Heteroatom labels (white bg circle + colored symbol)
    for i in range(mol2d.GetNumAtoms()):
        atom = mol2d.GetAtomWithIdx(i)
        sym  = atom.GetSymbol()
        if sym == "C":
            continue
        ax, ay = svg_coords.get(i,(0,0))
        clr    = _ATOM_CLR.get(sym.upper(), "#555")
        r_bg   = 11 if len(sym)==1 else 14
        parts.append(
            f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="{r_bg+1}"'
            f' fill="white" stroke="none"/>')
        parts.append(
            f'<text x="{ax:.1f}" y="{ay:.1f}" text-anchor="middle"'
            f' dominant-baseline="central"'
            f' font-family="Arial,sans-serif" font-size="13"'
            f' font-weight="600" fill="{clr}">{sym}</text>')
        fc = atom.GetFormalCharge()
        if fc != 0:
            fc_s = "⁺" if fc==1 else "⁻" if fc==-1 else f"{fc:+d}"
            parts.append(
                f'<text x="{ax+r_bg:.1f}" y="{ay-r_bg+2:.1f}"'
                f' font-family="Arial,sans-serif" font-size="9" fill="{clr}">{fc_s}</text>')

    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN SVG RENDERER
# ──────────────────────────────────────────────────────────────────────────────

def _render_clean_svg(mol2d, svg_coords: dict, placements: list,
                       title: str, W: int, H: int) -> str:
    parts = []
    parts.append(
        f'<svg width="100%" viewBox="0 0 {W} {H}" '
        f'xmlns="http://www.w3.org/2000/svg">')
    parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')

    if title:
        esc = title.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        parts.append(
            f'<text x="{W//2}" y="18" text-anchor="middle"'
            f' font-family="Arial,sans-serif" font-size="11" fill="#777">{esc}</text>')

    # ── STEP 1: Draw interaction lines (behind everything) ────────────────────
    for p in placements:
        itype  = p["itype"]
        vis    = _ITYPE_VIS.get(itype, _ITYPE_VIS["hbond"])
        ai     = p.get("lig_atom_idx", 0)
        lx, ly = svg_coords.get(ai, (W//2, H//2))
        bx, by = p["bx"], p["by"]
        cr     = p["circ_r"]

        # Shorten line to circle edge on residue side
        dx, dy = lx - bx, ly - by
        L  = _math.sqrt(dx*dx+dy*dy) + 1e-9
        ex = bx + dx/L * cr        # line end at circle perimeter
        ey = by + dy/L * cr

        dash_attr = f' stroke-dasharray="{vis["dash"]}"' if vis["dash"] else ""
        parts.append(
            f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{ex:.1f}" y2="{ey:.1f}"'
            f' stroke="{vis["line_stroke"]}" stroke-width="{vis["lw"]}"{dash_attr}'
            f' opacity="0.9"/>')

        # Distance label on line (H-bond and other show_dist types)
        if vis["show_dist"] and p.get("distance") is not None:
            # Place label at 60% along the line, in a small white pill
            mx  = lx + (ex-lx)*0.60
            my  = ly + (ey-ly)*0.60
            dist_str = f"{p['distance']:.1f}\u00c5"
            tw  = len(dist_str)*6 + 8
            # White pill background so it reads over the line
            parts.append(
                f'<rect x="{mx-tw/2:.1f}" y="{my-8:.1f}"'
                f' width="{tw:.0f}" height="14" rx="4"'
                f' fill="white" stroke="{vis["line_stroke"]}" stroke-width="0.6" opacity="0.92"/>')
            parts.append(
                f'<text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle"'
                f' dominant-baseline="central"'
                f' font-family="Arial,sans-serif" font-size="9"'
                f' fill="{vis["line_stroke"]}">{dist_str}</text>')

    # ── STEP 2: Ligand structure ──────────────────────────────────────────────
    parts.append(_render_ligand_svg(mol2d, svg_coords))

    # ── STEP 3: Residue circles ───────────────────────────────────────────────
    for p in placements:
        itype  = p["itype"]
        vis    = _ITYPE_VIS.get(itype, _ITYPE_VIS["hbond"])
        bx, by = p["bx"], p["by"]
        cr     = p["circ_r"]
        rn     = p["resname"]
        ri     = p["resid"]
        ch     = p.get("chain","")

        # Clamp to canvas
        bx = max(cr+4, min(bx, W-cr-4))
        by = max(cr+4, min(by, H-cr-60))

        # Circle background
        parts.append(
            f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="{cr}"'
            f' fill="{vis["circle_fill"]}" stroke="{vis["circle_stroke"]}"'
            f' stroke-width="2"/>')

        # Residue name (3-letter, italic) — top line
        # Residue number — bottom line
        parts.append(
            f'<text x="{bx:.1f}" y="{by-7:.1f}" text-anchor="middle"'
            f' dominant-baseline="central"'
            f' font-family="Arial,sans-serif" font-size="10"'
            f' font-style="italic" font-weight="600"'
            f' fill="{vis["label_color"]}">{rn}</text>')
        parts.append(
            f'<text x="{bx:.1f}" y="{by+8:.1f}" text-anchor="middle"'
            f' dominant-baseline="central"'
            f' font-family="Arial,sans-serif" font-size="10"'
            f' font-weight="500" fill="{vis["label_color"]}">{ri}{ch}</text>')

    # ── STEP 4: Legend ────────────────────────────────────────────────────────
    active_types  = list(dict.fromkeys(p["itype"] for p in placements))
    if not active_types:
        active_types = list(_ITYPE_VIS.keys())[:4]

    ly0 = H - 48
    n_leg = len(active_types)
    iw    = min(110, (W-40)/max(n_leg,1))
    lx_start = (W - iw*n_leg) / 2

    parts.append(
        f'<rect x="{lx_start-6:.0f}" y="{ly0-4}" '
        f'width="{iw*n_leg+12:.0f}" height="44" rx="7"'
        f' fill="#f8f8f8" stroke="#ddd" stroke-width="0.6"/>')

    for k, itype in enumerate(active_types):
        vis  = _ITYPE_VIS.get(itype, _ITYPE_VIS["hbond"])
        ix   = lx_start + iw*k + iw/2
        # Small colored circle
        parts.append(
            f'<circle cx="{ix:.0f}" cy="{ly0+10}" r="7"'
            f' fill="{vis["circle_fill"]}" stroke="{vis["circle_stroke"]}"'
            f' stroke-width="1.5"/>')
        # Line sample
        dash_attr = f' stroke-dasharray="{vis["dash"]}"' if vis["dash"] else ""
        parts.append(
            f'<line x1="{ix-18:.0f}" y1="{ly0+10}" x2="{ix-9:.0f}" y2="{ly0+10}"'
            f' stroke="{vis["line_stroke"]}" stroke-width="1.8"{dash_attr}/>')
        # Label
        parts.append(
            f'<text x="{ix:.0f}" y="{ly0+30}" text-anchor="middle"'
            f' font-family="Arial,sans-serif" font-size="9" fill="#444">'
            f'{vis["legend"]}</text>')

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
    size: tuple = (720, 700),
    max_residues: int = 14,
) -> bytes:
    """
    Generate a clean 2D protein-ligand interaction diagram.

    Visual style:
      • Protein residues  →  colored circles (3-letter code + number)
                             color = interaction type
      • H-bond            →  blue dashed line, distance on line
      • Hydrophobic       →  green solid line
      • π-π, cation-π     →  purple/amber dashed
      • Ionic             →  magenta dash-dot, distance shown
      • Halogen bond      →  pink dashed, distance shown
      • Metal             →  gold dashed, distance shown
      • H···Halogen       →  violet dash-dot, distance shown

    Layout: angular-sort placement — lines NEVER cross each other.

    Returns SVG bytes. Embed in Streamlit with components.html().
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDepictor

    RDLogger.DisableLog("rdApp.*")
    W, H = size

    # Load 3D pose
    try:
        mol3d = None
        for sanitize in (True, False):
            sup   = Chem.SDMolSupplier(pose_sdf, sanitize=sanitize, removeHs=False)
            mol3d = next((m for m in sup if m is not None), None)
            if mol3d is not None:
                if not sanitize:
                    try:
                        Chem.SanitizeMol(mol3d)
                    except Exception:
                        pass
                break
        if mol3d is None or mol3d.GetNumConformers() == 0:
            raise ValueError("No valid 3D pose in SDF")
    except Exception as e:
        RDLogger.EnableLog("rdApp.error")
        return (
            f'<svg viewBox="0 0 720 80" xmlns="http://www.w3.org/2000/svg">'
            f'<rect width="720" height="80" fill="white"/>'
            f'<text x="360" y="44" text-anchor="middle"'
            f' font-family="Arial,sans-serif" font-size="13" fill="#cc2222">'
            f'Error: {e}</text></svg>'
        ).encode()

    # 2D mol
    mol2d = None
    if smiles and smiles.strip():
        mol2d = Chem.MolFromSmiles(smiles.strip())
    if mol2d is None:
        mol2d = Chem.RemoveHs(mol3d, sanitize=False)
        try:
            Chem.SanitizeMol(mol2d)
        except Exception:
            pass
    mol2d = Chem.RemoveHs(mol2d)
    rdDepictor.Compute2DCoords(mol2d)

    # 3D → 2D atom index map
    mol3d_noH = Chem.RemoveHs(mol3d, sanitize=False)
    try:
        Chem.SanitizeMol(mol3d_noH)
    except Exception:
        pass
    match_3d_to_2d = {}
    try:
        match = mol3d_noH.GetSubstructMatch(mol2d)
        if len(match) == mol2d.GetNumAtoms():
            for idx2d, idx3d in enumerate(match):
                match_3d_to_2d[idx3d] = idx2d
    except Exception:
        pass

    # Detect
    try:
        raw = _detect_all_interactions(mol3d, receptor_pdb, cutoff=cutoff)
    except Exception:
        raw = []

    for ix in raw:
        ix["lig_atom_idx"] = match_3d_to_2d.get(ix.get("lig_atom_idx",0), 0)

    pri_map = {t: i for i, t in enumerate(_ITYPE_PRIORITY)}
    deduped = _deduplicate_interactions(raw)
    deduped.sort(key=lambda x: (pri_map.get(x["itype"],99), x["distance"]))
    deduped = deduped[:max_residues]

    cx, cy     = W // 2, int(H * 0.44)
    svg_coords = _compute_svg_coords(mol2d, cx, cy, target_size=200)
    placements = _place_residues_no_cross(deduped, svg_coords, cx, cy,
                                          R=240, circ_r=32)

    svg = _render_clean_svg(mol2d, svg_coords, placements, title, W, H)
    RDLogger.EnableLog("rdApp.error")
    return svg.encode()


def draw_interactions_rdkit(lig_mol, receptor_pdb: str, smiles: str,
                            title: str = "", cutoff: float = 3.5,
                            size: tuple = (500, 500),
                            max_residues: int = 10) -> bytes:
    """Backward-compatible alias → draw_interaction_diagram."""
    import tempfile
    from rdkit import Chem
    tmp = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
    with Chem.SDWriter(tmp.name) as w:
        w.write(lig_mol)
    return draw_interaction_diagram(
        receptor_pdb=receptor_pdb,
        pose_sdf=tmp.name,
        smiles=smiles,
        title=title,
        cutoff=cutoff,
        size=(720, 700),
        max_residues=max_residues,
    )


def _svg_stamp(svg_text: str, title: str, w: int, h: int) -> str:
    esc    = title.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    pad    = int(w*0.05)
    pill_w = w - 2*pad
    pill_h = 28
    pill_y = h - pill_h - 8
    text_y = pill_y + pill_h//2
    radius = pill_h//2
    stamp  = (
        f'<g><rect x="{pad}" y="{pill_y}" width="{pill_w}" height="{pill_h}"'
        f' rx="{radius}" ry="{radius}" fill="#E8E8E8" fill-opacity="0.93"'
        f' stroke="#C8C8C8" stroke-width="0.5"/>'
        f'<text x="{w//2}" y="{text_y}" text-anchor="middle"'
        f' dominant-baseline="middle"'
        f' font-family="Helvetica Neue,Arial,sans-serif"'
        f' font-size="13" font-weight="500" fill="#1A1A1A">{esc}</text></g>'
    )
    return svg_text.replace("</svg>", f"{stamp}</svg>")
