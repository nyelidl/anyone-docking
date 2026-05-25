#!/usr/bin/env python3
"""
core.py — Pure computation layer for Anyone Can Dock.
No Streamlit imports. All functions return plain dicts / tuples.
Safe to import in Colab notebooks, pytest, or any UI framework.

Patch 2025-05: three PDB-parsing fixes for atoms like Br in 4S9 / partial-occupancy entries:
  Fix A  _strip_anisou(text)         — remove ANISOU/SIGUIJ lines before ProDy / line-readers
  Fix B  _keep_best_altloc(text)     — keep highest-occupancy alt-conf, blank altLoc column
  Fix C  _atom_name_to_element()     — resolve "BR1" → "BR", "FE2" → "FE" etc.
  Fix D  _clean_pdb_text(text)       — convenience wrapper calling A+B
  Fix E  _clean_pdb_file(path)       — read/clean/write cleaned copy; returns cleaned path

These are applied in:
  scan_hetatm_residues()      — before parsePDB (Fix A+B) and in line-reader (Fix C)
  prepare_receptor()          — before parsePDB (Fix A+B)
  strip_and_convert_receptor()— in the HETATM line-reader (Fix C) and before obabel (Fix A+B)
"""

import os
import subprocess
import sys
import tempfile
import time
import re as _re
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

METAL_RESNAMES = {
    "MG", "ZN", "CA", "MN", "FE", "CU", "CO", "NI", "CD", "HG", "NA", "K", "HO",
    "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD", "TB", "DY", "ER", "TM", "YB", "LU",
}
METAL_CHARGES = {
    "MG": 2.0, "ZN": 2.0, "CA": 2.0, "MN": 2.0, "FE": 3.0,
    "CU": 2.0, "CO": 2.0, "NI": 2.0, "CD": 2.0, "HG": 2.0, "HO": 3.0,
    "LA": 3.0, "CE": 3.0, "PR": 3.0, "ND": 3.0, "PM": 3.0, "SM": 3.0,
    "EU": 3.0, "GD": 3.0, "TB": 3.0, "DY": 3.0, "ER": 3.0, "TM": 3.0,
    "YB": 3.0, "LU": 3.0,
    "NA": 1.0, "K":  1.0,
}

EXCLUDE_IONS = set(
    "HOH,WAT,DOD,SOL,NA,CL,K,CA,MG,ZN,MN,FE,CU,CO,NI,CD,HG,HO,LA,CE,PR,ND,PM,SM,EU,GD,TB,DY,ER,TM,YB,LU".split(",")
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
    "HO", "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD", "TB", "DY", "ER", "TM", "YB", "LU",
}

HEME_RESNAMES = {"HEM", "HEC", "HEA", "HEB", "HDD", "HDM"}

_PV_MAX_RETRIES = 3
_PV_RETRY_DELAY = 10
_PV_POLL_ATTEMPTS = 60


# ══════════════════════════════════════════════════════════════════════════════
#  PDB CLEANING HELPERS  (Patch 2025-05 — Fixes A–E)
# ══════════════════════════════════════════════════════════════════════════════

def _strip_anisou(pdb_text: str) -> str:
    """
    Fix A — Remove ANISOU and SIGUIJ records.

    These anisotropic B-factor lines immediately follow the ATOM/HETATM they
    belong to.  ProDy (some builds) and any line-by-line reader that advances
    one record at a time can land on the ANISOU line and either miscount
    atoms or silently drop the preceding HETATM.

    Example entry that triggered the bug:
        HETATM 7319 BR1  4S9 A1101  86.832 121.044  17.730  0.80 30.83  BR
        ANISOU 7319 BR1  4S9 A1101   3962   3965   3786   344    90   -56  BR

    Stripping ANISOU/SIGUIJ is safe — they carry no coordinate information.
    """
    out = []
    for line in pdb_text.splitlines(keepends=True):
        rec = line[:6]   # deliberately NOT .strip() — exact column check
        if rec in ("ANISOU", "SIGUIJ"):
            continue
        out.append(line)
    return "".join(out)


def _keep_best_altloc(pdb_text: str) -> str:
    """
    Fix B — Resolve alternate conformations.

    Keep the highest-occupancy alt-loc for each atom; blank the altLoc column
    (column 17, 0-indexed 16) so downstream tools see a normal single-conformer
    file.

    Example: BR1 at altLoc A / occ 0.80  +  altLoc B / occ 0.20
    Result:  BR1 with blank altLoc and occ 0.80 — no longer dropped.

    Without this fix, tools that skip non-blank or non-'A' altLoc atoms (e.g.
    Meeko, some ProDy selectors, obabel) silently lose the atom.
    """
    from collections import defaultdict

    lines = pdb_text.splitlines(keepends=True)

    # First pass: for each unique (chain, resseq, icode, atom-name) key that
    # has alt-loc variants, find the line index of the best conformer.
    # Key layout (all from fixed PDB columns):
    #   chain  = col 22   (0-indexed 21)
    #   resseq = cols 23-26 stripped
    #   icode  = col 27   (0-indexed 26)
    #   name   = cols 13-16
    best: dict = defaultdict(lambda: (" ", -1.0, -1))
    for i, line in enumerate(lines):
        rec = line[:6].strip()
        if rec not in ("ATOM", "HETATM") or len(line) < 27:
            continue
        altloc = line[16]
        if altloc == " ":
            continue  # no alt-conf — always kept; no tracking needed
        try:
            occ = float(line[54:60])
        except (ValueError, IndexError):
            occ = 0.0
        key = (
            line[21] if len(line) > 21 else " ",   # chain
            line[22:26].strip(),                     # resseq
            line[26] if len(line) > 26 else " ",    # icode
            line[12:16],                             # atom name (4 chars)
        )
        if occ > best[key][1]:
            best[key] = (altloc, occ, i)

    if not best:
        return pdb_text  # nothing to resolve — return unchanged

    keep_indices = {v[2] for v in best.values()}

    out = []
    for i, line in enumerate(lines):
        rec = line[:6].strip()
        if rec in ("ATOM", "HETATM") and len(line) > 16 and line[16] != " ":
            key = (
                line[21] if len(line) > 21 else " ",
                line[22:26].strip(),
                line[26] if len(line) > 26 else " ",
                line[12:16],
            )
            if i not in keep_indices:
                continue  # drop lower-occupancy conformer
            # Blank altLoc column so downstream tools treat it as normal
            line = line[:16] + " " + line[17:]
        out.append(line)
    return "".join(out)


def _atom_name_to_element(atom_name: str, element_col: str = "") -> str:
    """
    Fix C — Derive a 1–2 char element symbol from a PDB atom name field
    (cols 13–16) and the optional element column (cols 77–78).

    Handles non-standard names that include serial digits, e.g.:
      "BR1 " → "BR"      " FE2" → "FE"      " C  " → "C"
      "ZN1 " → "ZN"      "CL1 " → "CL"      "NA1 " → "NA"

    Priority: explicit element column > two-letter lookup > first letter.
    """
    # Prefer the explicit element column when present and purely alphabetic
    el = element_col.strip().upper()
    if el and el.isalpha():
        return el

    # Strip digits and whitespace, then take leading letters
    name = atom_name.strip().upper()
    letters = _re.sub(r"[^A-Z]", "", name)

    # Known two-letter elements that commonly appear as HETATM / metal ions
    TWO_LETTER = {
        "BR", "CL", "FE", "ZN", "MG", "CA", "MN", "CU", "CO",
        "NI", "CD", "HG", "SE", "NA", "AL", "SI", "AU", "AG",
        "RU", "RH", "PD", "PT", "IR", "OS", "RE", "MO", "TC",
        "SN", "PB", "BI", "SB", "TE", "AS", "GE", "GA", "IN",
        "TL", "BA", "SR", "CS", "RB", "LI", "BE", "SC", "TI",
        "CR", "YB", "LU", "TM", "ER", "DY", "TB", "GD",
        "EU", "SM", "PM", "ND", "PR", "CE", "LA",
    }
    if len(letters) >= 2 and letters[:2] in TWO_LETTER:
        return letters[:2]
    return letters[:1] if letters else "X"


def _clean_pdb_text(pdb_text: str) -> str:
    """
    Fix D — Apply Fix A + Fix B in sequence.
    Safe to call on any PDB string (no-op on CIF).
    """
    pdb_text = _strip_anisou(pdb_text)
    pdb_text = _keep_best_altloc(pdb_text)
    return pdb_text


def _clean_pdb_file(path: str) -> str:
    """
    Fix E — Read a PDB file, apply _clean_pdb_text, write a '_cleaned.pdb'
    copy alongside the original, and return the cleaned path.

    Returns the original path unchanged on any error or if the file is CIF.
    The cleaned file is idempotent — calling twice produces the same result.
    """
    try:
        # Skip CIF files — _clean_pdb_text is PDB-specific
        if is_cif_file(path):
            return path
        raw = open(path).read()
        cleaned = _clean_pdb_text(raw)
        if cleaned == raw:
            return path  # nothing changed; avoid extra write
        out_path = _re.sub(r"\.(pdb|ent)$", "_cleaned.pdb", path, flags=_re.IGNORECASE)
        if out_path == path:
            out_path = path + "_cleaned.pdb"
        with open(out_path, "w") as fh:
            fh.write(cleaned)
        return out_path
    except Exception:
        return path  # graceful degradation


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def run_cmd(cmd, cwd=None):
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str),
        capture_output=True, text=True, cwd=cwd,
    )
    return r.returncode, (r.stdout + r.stderr).strip()


def _rdkit_six_patch():
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
    log = []
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
            log.append("✓ CIF -> PDB conversion via gemmi")
            return {"success": True, "pdb_path": pdb_out_path, "log": log}
        else:
            log.append("⚠ gemmi produced empty PDB — trying OpenBabel")
    except ImportError:
        log.append("⚠ gemmi not installed — trying OpenBabel")
    except Exception as e:
        log.append(f"⚠ gemmi failed ({e}) — trying OpenBabel")

    try:
        rc, out = run_cmd(f'obabel "{cif_path}" -O "{pdb_out_path}" -ipdb')
        if not os.path.exists(pdb_out_path) or os.path.getsize(pdb_out_path) < 100:
            rc, out = run_cmd(f'obabel "{cif_path}" -O "{pdb_out_path}"')
        if os.path.exists(pdb_out_path) and os.path.getsize(pdb_out_path) > 100:
            log.append("✓ CIF -> PDB conversion via OpenBabel")
            return {"success": True, "pdb_path": pdb_out_path, "log": log}
        else:
            log.append(f"⚠ OpenBabel CIF->PDB produced empty file (exit {rc}): {out[:300]}")
    except Exception as e:
        log.append(f"⚠ OpenBabel CIF->PDB failed: {e}")

    try:
        from prody import parseMMCIF, writePDB as _writePDB
        atoms = parseMMCIF(cif_path)
        if atoms is not None and atoms.numAtoms() > 0:
            _writePDB(pdb_out_path, atoms)
            if os.path.exists(pdb_out_path) and os.path.getsize(pdb_out_path) > 100:
                log.append("✓ CIF -> PDB conversion via ProDy parseMMCIF")
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
        "error": "All CIF->PDB conversion methods failed.",
    }


def is_cif_file(filepath: str) -> bool:
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
    import shutil
    if shutil.which("obabel") is None:
        return False, "obabel not found — install OpenBabel (https://openbabel.org) and ensure it is on PATH"
    _, out = run_cmd("obabel --version")
    return True, (out.splitlines()[0] if out else "ok")


# ══════════════════════════════════════════════════════════════════════════════
#  VINA BINARY
# ══════════════════════════════════════════════════════════════════════════════

def get_vina_binary(path: str = ""):
    import shutil as _shutil
    # ── Check if vina is already on PATH (common for local installs) ──────
    _vina_in_path = _shutil.which("vina") or _shutil.which("vina_1.2.7") or _shutil.which("AutoDock-Vina")
    if _vina_in_path and os.path.getsize(_vina_in_path) > 100_000:
        return _vina_in_path, f"ok (found on PATH: {_vina_in_path})"

    import platform
    system  = platform.system().lower()
    machine = platform.machine().lower()
    _BASE = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/"
    if system == "linux":
        _FNAME = "vina_1.2.7_linux_x86_64"
    elif system == "darwin":
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

_MIN_LIG_ATOMS = 4

BUFFER_RESNAMES = {
    "GOL", "EDO", "PEG", "PGE", "PG4", "MPD", "DMS", "DMSO", "ACT",
    "ACY", "ACE", "TRS", "MES", "EPE", "BME", "SO4", "PO4", "NO3",
    "SCN", "FMT", "IPA", "EOH", "MOH", "CL", "BR", "IOD"
}
WATER_RESNAMES = {"HOH", "WAT", "DOD", "SOL"}


def _hetatm_key(resname, chain, resid):
    return f"{str(resname).strip().upper()}|{str(chain or '').strip() or '_'}|{int(resid)}"


def _guess_hetatm_type(resname, n_atoms):
    rn = (resname or "").strip().upper()
    if rn in WATER_RESNAMES:
        return "water"
    if rn in METAL_RESNAMES:
        return "metal"
    if rn in HEME_RESNAMES:
        return "heme/cofactor"
    if rn in COFACTOR_NAMES:
        return "cofactor"
    if rn in BUFFER_RESNAMES or n_atoms <= 3:
        return "buffer/ion"
    if n_atoms >= _MIN_LIG_ATOMS:
        return "ligand"
    return "other"


def _default_hetatm_action(type_guess, n_atoms):
    if type_guess == "water":
        return "remove"
    if type_guess == "metal":
        return "keep"
    if "cofactor" in type_guess:
        return "keep"
    if type_guess == "ligand":
        return "reference" if n_atoms >= _MIN_LIG_ATOMS else "remove"
    return "remove"


def _collect_hetatm_residues(atoms) -> list:
    from prody import calcCenter
    het = atoms.select("hetatm")
    if het is None:
        return []
    rows = []
    for r in het.getHierView().iterResidues():
        rn = (r.getResname() or "").strip().upper()
        ch = (r.getChid() or "").strip()
        ri = int(r.getResnum())
        n_atoms = int(r.numAtoms())
        tg = _guess_hetatm_type(rn, n_atoms)
        sel = (f"resname {rn} and resid {ri} and chain {ch}" if ch else f"resname {rn} and resid {ri}")
        res_atoms = atoms.select(sel)
        if res_atoms is None:
            continue
        cx_, cy_, cz_ = (float(v) for v in calcCenter(res_atoms))
        rows.append({
            "key": _hetatm_key(rn, ch, ri),
            "resname": rn, "chain": ch, "resid": ri, "n_atoms": n_atoms,
            "type_guess": tg, "default_action": _default_hetatm_action(tg, n_atoms),
            "action": _default_hetatm_action(tg, n_atoms),
            "sel_str": sel, "atoms": res_atoms, "cx": cx_, "cy": cy_, "cz": cz_,
        })
    rank = {"ligand": 0, "heme/cofactor": 1, "cofactor": 2, "metal": 3, "buffer/ion": 4, "other": 5, "water": 6}
    rows.sort(key=lambda d: (rank.get(d["type_guess"], 9), -d["n_atoms"], d["resname"], d["chain"], d["resid"]))
    seen_reference = False
    for d in rows:
        if d["type_guess"] == "ligand":
            if not seen_reference:
                d["default_action"] = d["action"] = "reference"
                seen_reference = True
            else:
                d["default_action"] = d["action"] = "remove"
    return rows


def scan_hetatm_residues(raw_pdb: str) -> list:
    """
    Return HETATM residues for receptor setup UI.

    FIX A+B applied: PDB text is cleaned (ANISOU stripped, best altloc kept)
    before ProDy parses it, so partial-occupancy atoms like BR1 at occ=0.80
    are correctly detected rather than silently dropped.

    action options:
      reference = use as co-crystal reference/grid center and remove from receptor
      keep      = keep in receptor
      remove    = remove from receptor
    """
    try:
        from prody import parsePDB, confProDy
        confProDy(verbosity="none")

        # ── Fix A+B: clean before ProDy sees the file ──────────────────────
        _working_path = raw_pdb
        if not is_cif_file(raw_pdb):
            _working_path = _clean_pdb_file(raw_pdb)

        if is_cif_file(_working_path):
            import tempfile as _tf
            _tmp = _tf.mktemp(suffix=".pdb")
            res = convert_cif_to_pdb(_working_path, _tmp)
            if res["success"]:
                _working_path = _tmp
                # Clean the converted PDB too
                _working_path = _clean_pdb_file(_working_path)

        atoms = parsePDB(_working_path)
        if atoms is None:
            return []
        rows = _collect_hetatm_residues(atoms)
        return [
            {k: d[k] for k in ("key", "resname", "chain", "resid", "n_atoms", "type_guess", "default_action", "action")}
            for d in rows
        ]
    except Exception:
        return []


def _collect_removable_ligands(atoms) -> list:
    from prody import calcCenter
    excl = EXCLUDE_IONS | GLYCAN_NAMES | COFACTOR_NAMES | HEME_RESNAMES | METAL_RESNAMES
    het  = atoms.select("hetatm and not water")
    _BACKBONE = {"N", "CA", "C", "O"}
    if het is None:
        return []
    results = []
    for r in het.getHierView().iterResidues():
        rn = (r.getResname() or "").strip().upper()
        if rn in excl:
            continue
        if r.numAtoms() <= _MIN_LIG_ATOMS:
            continue
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
    results.sort(key=lambda d: (-d["n_atoms"], d["chain"] != "A"))
    return results


def detect_cocrystal_ligand(raw_pdb: str) -> dict:
    from prody import parsePDB
    atoms = parsePDB(raw_pdb)
    if atoms is None:
        return {"found": False}
    cands = _collect_removable_ligands(atoms)
    if not cands:
        return {"found": False}
    chosen = cands[0]
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
    try:
        from prody import parsePDB, confProDy
        confProDy(verbosity="none")
        # Fix A+B
        if not is_cif_file(raw_pdb):
            raw_pdb = _clean_pdb_file(raw_pdb)
        if is_cif_file(raw_pdb):
            import tempfile as _tf
            _tmp = _tf.mktemp(suffix=".pdb")
            res  = convert_cif_to_pdb(raw_pdb, _tmp)
            if res["success"]:
                raw_pdb = _clean_pdb_file(_tmp)
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
    Convert a cleaned receptor PDB to rec.pdb + rec.pdbqt.

    Fix A+B: the input file is cleaned before obabel so partial-occupancy
             atoms (e.g. BR1 at occ=0.80) are not silently dropped.
    Fix C:   _atom_name_to_element() is used when identifying metal/heme atoms
             in the line-by-line reader, so "BR1", "FE2", "ZN1" etc. are all
             correctly categorised regardless of how RCSB names them.
    """
    wdir = Path(wdir)
    log  = []
    rec_fh    = str(wdir / "rec.pdb")
    rec_pdbqt = str(wdir / "rec.pdbqt")
    try:
        # ── Fix A+B: clean before any line-reader or obabel call ───────────
        try:
            raw_text = open(rec_raw).read()
            cleaned_text = _clean_pdb_text(raw_text)
            if cleaned_text != raw_text:
                rec_raw_clean = rec_raw.replace(".pdb", "_cleaned.pdb")
                if rec_raw_clean == rec_raw:
                    rec_raw_clean = rec_raw + "_cleaned.pdb"
                with open(rec_raw_clean, "w") as _fh:
                    _fh.write(cleaned_text)
                rec_raw = rec_raw_clean
                log.append("✓ PDB cleaned (ANISOU stripped, best alt-loc kept)")
        except Exception as _ce:
            log.append(f"⚠ PDB pre-clean skipped: {_ce}")

        metal_lines = []
        heme_lines  = []
        clean_lines = []

        with open(rec_raw) as f:
            for line in f:
                field = line[:6].strip()
                if field not in ("ATOM", "HETATM"):
                    clean_lines.append(line)
                    continue

                # Fix C: use robust element resolution instead of raw slice
                atom_name_raw = line[12:16] if len(line) > 15 else "    "
                element_col   = line[76:78] if len(line) > 78 else ""
                el = _atom_name_to_element(atom_name_raw, element_col)

                # Resname is still in cols 17-20 (reliable for classification)
                rn = line[17:20].strip().upper()

                if rn in METAL_RESNAMES or el in METAL_RESNAMES:
                    metal_lines.append(line)
                elif rn in HEME_RESNAMES:
                    heme_lines.append(line)
                else:
                    clean_lines.append(line)

        rec_nometal = str(wdir / "receptor_atoms_nometal.pdb")
        with open(rec_nometal, "w") as f:
            f.writelines(clean_lines)

        if metal_lines:
            names = ", ".join(sorted({l[17:20].strip() for l in metal_lines}))
            log.append(f"⚠ Stripped {len(metal_lines)} metal atom(s) before OpenBabel: {names}")
        if heme_lines:
            hnames = ", ".join(sorted({l[17:20].strip() for l in heme_lines}))
            log.append(f"⚠ Stripped {len(heme_lines)} heme atom(s) before OpenBabel: {hnames}")

        rc1, out1 = run_cmd(f'obabel "{rec_nometal}" -O "{rec_fh}" -h')
        if not os.path.exists(rec_fh) or os.path.getsize(rec_fh) < 100:
            raise ValueError(f"OpenBabel H-addition produced empty file (exit {rc1}). Output: {out1[:400]}")
        log.append("✓ Hydrogens added")

        rc2, out2 = run_cmd(f'obabel "{rec_fh}" -O "{rec_pdbqt}" -xr --partialcharge gasteiger')
        if not os.path.exists(rec_pdbqt) or os.path.getsize(rec_pdbqt) < 100:
            raise ValueError(f"PDBQT conversion produced empty file (exit {rc2}). Output: {out2[:400]}")
        log.append("✓ PDBQT conversion complete")

        # ── Re-add metals + heme to rec.pdb for display / PoseView ──────────
        if metal_lines or heme_lines:
            try:
                rec_lines = open(rec_fh).readlines()
                rec_lines = [l for l in rec_lines if l.strip() != "END"]
                rec_lines.extend(metal_lines)
                rec_lines.extend(heme_lines)
                rec_lines.append("END\n")
                with open(rec_fh, "w") as f:
                    f.writelines(rec_lines)
                log.append(
                    f"✓ Re-added {len(metal_lines)} metal + {len(heme_lines)} heme "
                    f"atom(s) to rec.pdb for display"
                )
            except Exception as e:
                log.append(f"⚠ Could not re-add metals/heme to rec.pdb: {e}")

        # ── Re-inject metals + heme into PDBQT with AD4 atom types ──────────
        if metal_lines or heme_lines:
            pdbqt_lines = open(rec_pdbqt).readlines()
            pdbqt_lines = [l for l in pdbqt_lines if l.strip() != "END"]

            injected        = 0
            skipped_exotic  = 0
            heme_injected   = 0

            _no_reinject = {
                "HO", "LA", "CE", "PR", "ND", "PM", "SM", "EU",
                "GD", "TB", "DY", "ER", "TM", "YB", "LU",
            }

            for ml in metal_lines:
                try:
                    resname = ml[17:20].strip().upper()
                    if resname in _no_reinject:
                        skipped_exotic += 1
                        continue
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

            for hl in heme_lines:
                try:
                    serial  = int(hl[6:11])
                    name    = hl[12:16].strip()
                    resname = hl[17:20].strip().upper()
                    chain   = hl[21] if len(hl) > 21 else "A"
                    resid   = int(hl[22:26])
                    x       = float(hl[30:38])
                    y       = float(hl[38:46])
                    z       = float(hl[46:54])
                    if name.upper() in ("FE", "FE2", "FE3"):
                        atype  = "Fe"
                        charge = +3.0
                    elif name.upper().startswith("N"):
                        atype  = "NA"
                        charge = -0.3
                    elif name.upper().startswith("C"):
                        atype  = "A"
                        charge = 0.0
                    else:
                        atype  = "C"
                        charge = 0.0
                    pdbqt_lines.append(
                        f"HETATM{serial:5d} {name:<4s} {resname:<3s} "
                        f"{chain}{resid:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                        f"    {charge:+.3f} {atype}\n"
                    )
                    heme_injected += 1
                except Exception as e:
                    log.append(f"⚠ Could not re-inject heme line: {e}")

            pdbqt_lines.append("END\n")
            with open(rec_pdbqt, "w") as f:
                f.writelines(pdbqt_lines)

            if injected:
                log.append(f"✅ Re-injected {injected} metal atom(s) into PDBQT")
            if heme_injected:
                log.append(f"✅ Re-injected {heme_injected} heme atom(s) into PDBQT with AD4 atom types")
            if skipped_exotic:
                log.append(
                    f"ℹ Skipped re-injection of {skipped_exotic} Ho/lanthanide ion(s) into "
                    f"docking PDBQT; kept only in rec.pdb for display"
                )

        log.append("✓ Receptor PDBQT ready")
        return {"success": True, "rec_fh": rec_fh, "rec_pdbqt": rec_pdbqt, "log": log}

    except Exception as e:
        log.append(f"ERROR: {e}")
        return {"success": False, "error": str(e), "log": log}


def write_box_pdb(filename: str, cx, cy, cz, sx, sy, sz):
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
    hetatm_policy: dict | None = None,
    reference_hetatm_key: str = "",
) -> dict:
    from prody import parsePDB, calcCenter, writePDB
    wdir = Path(wdir)
    log  = []
    sx, sy, sz = box_size
    try:
        if is_cif_file(raw_pdb):
            log.append("📄 Detected mmCIF format — converting to PDB…")
            converted_pdb = str(wdir / "converted_from_cif.pdb")
            cif_result = convert_cif_to_pdb(raw_pdb, converted_pdb)
            log.extend(cif_result["log"])
            if not cif_result["success"]:
                raise ValueError(f"CIF -> PDB conversion failed: {cif_result.get('error', 'unknown')}")
            raw_pdb = converted_pdb

        # ── Fix A+B: clean before ProDy ───────────────────────────────────
        raw_pdb = _clean_pdb_file(raw_pdb)
        log.append("✓ PDB pre-cleaned (ANISOU stripped, best alt-loc resolved)")

        atoms = parsePDB(raw_pdb)
        if atoms is None:
            raise ValueError("ProDy parsePDB returned None")
        log.append(f"✓ Parsed {atoms.numAtoms()} atoms")

        ligand_pdb_path     = None
        cocrystal_ligand_id = ""
        cx = cy = cz = 0.0

        _hetatm_rows = _collect_hetatm_residues(atoms)
        _policy = {str(k): str(v).lower() for k, v in (hetatm_policy or {}).items()}
        for _r in _hetatm_rows:
            _r["action"] = _policy.get(_r["key"], _r.get("default_action", "remove")).lower()

        _reference = None
        if reference_hetatm_key:
            _reference = next((d for d in _hetatm_rows if d["key"] == reference_hetatm_key), None)
        if _reference is None:
            _reference = next((d for d in _hetatm_rows if d.get("action") == "reference"), None)

        _all_ligs = _collect_removable_ligands(atoms)
        _primary = None
        if _reference is not None:
            _primary = {
                "resname": _reference["resname"], "chain": _reference["chain"],
                "resid": _reference["resid"], "sel_str": _reference["sel_str"],
                "ligand_id": f"{_reference['resname']}_{_reference['chain']}_{_reference['resid']}",
                "n_atoms": _reference["n_atoms"], "atoms": _reference["atoms"],
                "cx": _reference["cx"], "cy": _reference["cy"], "cz": _reference["cz"],
            }
        elif not hetatm_policy and _all_ligs:
            if preferred_ligand:
                _pref = preferred_ligand.strip().upper()
                _primary = next((d for d in _all_ligs if d["resname"].upper() == _pref), None)
                if _primary is None:
                    log.append(f"⚠ Preferred ligand '{preferred_ligand}' not found — using largest ({_all_ligs[0]['resname']})")
            if _primary is None:
                _primary = _all_ligs[0]

        if _primary is not None:
            rn  = _primary["resname"]
            ch  = _primary["chain"]
            ri  = _primary["resid"]
            cocrystal_ligand_id = _primary["ligand_id"]
            ligand_pdb_path     = str(wdir / "LIG.pdb")
            writePDB(ligand_pdb_path, _primary["atoms"])
            log.append(f"✓ Reference HETATM for grid: {rn} chain '{ch}' resnum {ri} ({_primary['n_atoms']} atoms)")

        if center_mode == "auto":
            if _primary is not None:
                cx, cy, cz = _primary["cx"], _primary["cy"], _primary["cz"]
                log.append(f"📍 Auto center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
                log.append(f"🔑 PoseView2 ligand ID: {cocrystal_ligand_id}")
            else:
                log.append("⚠ No co-crystal ligand found")
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
                raise ValueError(f"ProDy selection '{prody_sel}' matched 0 atoms.")
            cx, cy, cz = (float(v) for v in calcCenter(ref_atoms))
            log.append(f"🔬 ProDy selection: '{prody_sel}' -> {ref_atoms.numAtoms()} atoms")
            log.append(f"📍 Center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
            if _primary is not None:
                log.append(f"🔑 PoseView2 ligand ID: {cocrystal_ligand_id}")

        if hetatm_policy:
            _remove_rows = [d for d in _hetatm_rows if d.get("action") in {"remove", "reference"}]
            if _remove_rows:
                _excl_expr = " or ".join(f"({d['sel_str']})" for d in _remove_rows)
                sel_str = f"not ({_excl_expr})"
            else:
                sel_str = "all"
            _kept = [d for d in _hetatm_rows if d.get("action") == "keep"]
            log.append(
                f"🧾 HETATM policy: keep {len(_kept)}, remove/reference {len(_remove_rows)}"
            )
            if _remove_rows:
                log.append(
                    "🧹 Removed/reference HETATM: "
                    + ", ".join(f"{d['resname']}:{d['chain'] or '-'}:{d['resid']}[{d.get('action')}]" for d in _remove_rows[:25])
                    + (" …" if len(_remove_rows) > 25 else "")
                )
        elif _all_ligs:
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

        try:
            _rec_lines   = open(rec_raw_path).readlines()
            _coord_lines = [l for l in _rec_lines if l[:6].strip() in ("ATOM", "HETATM")]
            _all_blank   = all(
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

        conv = strip_and_convert_receptor(rec_raw_path, wdir)
        log.extend(conv["log"])
        if not conv["success"]:
            raise ValueError(conv["error"])

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
            "hetatm_table":        [
                {k: d.get(k) for k in ("key", "resname", "chain", "resid", "n_atoms", "type_guess", "action")}
                for d in _hetatm_rows
            ],
            "log": log,
        }
    except Exception as e:
        log.append(f"ERROR: {e}")
        return {"success": False, "error": str(e), "log": log}


# ══════════════════════════════════════════════════════════════════════════════
#  PKANET CLOUD  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

_PKANET_CACHE: dict = {}
_PKA_PATTERNS = [
    _re.compile(r"pK[aA][\w\s\(\)]*?=\s*([+-]?\d+(?:\.\d+)?)", _re.IGNORECASE),
    _re.compile(r"([+-]?\d+(?:\.\d+)?)\s*\((?:pK[aA]|acid dissociation)[^)]*\)", _re.IGNORECASE),
]

_W_AROM_RING_LOST         = 8.0
_W_PHENOL_TO_KETO_FLIP    = 6.0
_W_PYROGALLOL_TRIKETO     = 6.0
_W_CATECHOL_DIKETO        = 4.0
_W_PHENOL_PRESERVED_BONUS = 0.5
_TAUTOMER_PLAUSIBILITY_CUTOFF = 3.0
_AMBIGUITY_SCORE_GAP          = 0.5


def _detect_chromone_system(mol):
    from rdkit import Chem
    ring_info = mol.GetRingInfo()
    rings = [set(r) for r in ring_info.AtomRings() if len(r) == 6]
    if not rings:
        return set()

    def _has_exo_carbonyl(atom_idx):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() != "C":
            return False
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetSymbol() != "O" or other.IsInRing():
                continue
            bo = bond.GetBondTypeAsDouble()
            if bo == 2.0:
                return True
            if bo == 1.5 and other.GetTotalNumHs() == 0 and other.GetDegree() == 1:
                return True
        return False

    pyrone_rings = [
        ring for ring in rings
        if sum(1 for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == "O") == 1
        and any(_has_exo_carbonyl(i) for i in ring)
    ]
    if not pyrone_rings:
        return set()

    system: set = set()
    for py in pyrone_rings:
        system.update(py)
        for other in rings:
            if other is not py and len(py & other) >= 2:
                system.update(other)
    return system


def _find_flavone_A_ring_phenols(mol):
    chromone_atoms = _detect_chromone_system(mol)
    if not chromone_atoms:
        return []

    ring_carbonyl_idx = None
    ring_oxygen_idx   = None

    for idx in chromone_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "C":
            for bond in atom.GetBonds():
                other = bond.GetOtherAtom(atom)
                if (other.GetSymbol() == "O"
                        and not other.IsInRing()
                        and bond.GetBondTypeAsDouble() in (2.0, 1.5)
                        and other.GetTotalNumHs() == 0
                        and other.GetDegree() == 1):
                    ring_carbonyl_idx = idx
                    break
        elif atom.GetSymbol() == "O" and atom.IsInRing():
            ring_oxygen_idx = idx

    def _chromone_nbrs(idx):
        return [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                if n.GetIdx() in chromone_atoms]

    def _has_phenolic_OH(c_idx):
        for bond in mol.GetAtomWithIdx(c_idx).GetBonds():
            other = bond.GetOtherAtom(mol.GetAtomWithIdx(c_idx))
            if (other.GetSymbol() == "O"
                    and other.GetTotalNumHs() >= 1
                    and other.GetDegree() == 1
                    and bond.GetBondTypeAsDouble() == 1.0
                    and not other.IsInRing()):
                return True
        return False

    candidates = []
    for atom in mol.GetAtoms():
        c_idx = atom.GetIdx()
        if c_idx not in chromone_atoms:
            continue
        if atom.GetSymbol() != "C" or not atom.GetIsAromatic():
            continue
        if c_idx == ring_carbonyl_idx:
            continue
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if (other.GetSymbol() == "O"
                    and other.GetTotalNumHs() >= 1
                    and other.GetDegree() == 1
                    and bond.GetBondTypeAsDouble() == 1.0
                    and not other.IsInRing()):
                candidates.append((c_idx, other.GetIdx()))
                break

    sites = []
    for c_idx, o_idx in candidates:
        chromone_nbrs = _chromone_nbrs(c_idx)
        ortho_carbons = [n for n in chromone_nbrs
                         if mol.GetAtomWithIdx(n).GetSymbol() == "C"]

        ortho_to_carbonyl = False
        carbonyl_direct   = False
        if ring_carbonyl_idx is not None:
            if ring_carbonyl_idx in chromone_nbrs:
                ortho_to_carbonyl = True
                carbonyl_direct   = True
            else:
                for nb in chromone_nbrs:
                    if any(n.GetIdx() == ring_carbonyl_idx
                           for n in mol.GetAtomWithIdx(nb).GetNeighbors()):
                        ortho_to_carbonyl = True
                        carbonyl_direct   = False
                        break

        ortho_to_ring_O = (ring_oxygen_idx is not None
                           and ring_oxygen_idx in chromone_nbrs)
        n_ortho_phenols = sum(1 for n in ortho_carbons if _has_phenolic_OH(n))

        if ortho_to_carbonyl:
            if carbonyl_direct:
                label, pka = "flavone_3OH_flavonol", 7.0
            else:
                label, pka = "flavone_5OH_chelated", 11.0
        elif ortho_to_ring_O:
            label, pka = "flavone_8OH_ortho_pyranO", 8.5
        elif n_ortho_phenols >= 2:
            label, pka = "flavone_6OH_pyrogallol_center", 8.5
        elif n_ortho_phenols == 1:
            label, pka = "flavone_phenol_catechol_pair", 7.0
        else:
            label, pka = "flavone_phenol_isolated", 8.5

        sites.append({
            "label":         label,
            "atom_indices":  [o_idx, c_idx],
            "ionizable_idx": o_idx,
            "heuristic_pka": pka,
            "site_type":     "acid",
        })

    if sites:
        detail = ", ".join(
            f"{s['label'].replace('flavone_','')}(pKa={s['heuristic_pka']})"
            for s in sites
        )
        print(f"    🌸  Detected {len(sites)} flavonoid A-ring phenol(s): {detail}")

    return sites


_IONIZABLE_SITE_DEF = [
    ("sulfonic_acid",      "[SX4](=O)(=O)[OX2H1]",                             1.0,  "acid"),
    ("phosphate_diester",  "[PX4](=O)([OX2H1])([OX2,OX1-])[OX2,OX1-]",       2.1,  "acid"),
    ("phosphonate",       "[PX4](=O)([OX2H1])[OX2H1]",                         2.1,  "acid"),
    ("carboxylic_acid",    "[CX3](=O)[OX2H1]",                                 4.5,  "acid"),
    ("tetrazole",          "c1nn[nH]n1",                                        4.9,  "acid"),
    ("benzimidazole",      "c1ccc2[nH]cnc2c1",                                 5.5,  "base"),
    ("sulfonyl_imide_NH_cyclic",   "[CX3;R](=O)[NX3;H1;R][SX4;R](=O)(=O)",       2.0,  "acid"),
    ("sulfonylurea_NH",            "[NX3;H0,H1][CX3;!R](=O)[NX3;H1;!R][SX4;!R](=O)(=O)", 5.5,  "acid"),
    ("sulfonyl_imide_NH",          "[CX3](=O)[NX3;H1][SX4](=O)(=O)",             2.0,  "acid"),
    ("sulfonamide_5het_NH",        "[SX4](=O)(=O)[NX3;H1][c;$([c]1[c,n][o,n,s][c,n][c,n]1),$([c]1[c,n][c,n][o,n,s][c,n]1)]",  5.7, "acid"),
    ("sulfonamide_thiazole_NH",    "[SX4](=O)(=O)[NX3;H1]c1nccs1",               7.0,  "acid"),
    ("sulfonamide_oxazole_NH",     "[SX4](=O)(=O)[NX3;H1]c1ncco1",               6.5,  "acid"),
    ("sulfonamide_pyrim2_NH",      "[SX4](=O)(=O)[NX3;H1]c1ncccn1",              7.0,  "acid"),
    ("sulfonamide_pyrim4_NH",      "[SX4](=O)(=O)[NX3;H1]c1ccncn1",              6.5,  "acid"),
    ("sulfonamide_pyrim5_NH",      "[SX4](=O)(=O)[NX3;H1]c1cncnc1",              6.5,  "acid"),
    ("sulfonamide_pyrazin_NH",     "[SX4](=O)(=O)[NX3;H1]c1cnccn1",              7.0,  "acid"),
    ("sulfonamide_pyridazin_NH",   "[SX4](=O)(=O)[NX3;H1]c1ccnnc1",              7.0,  "acid"),
    ("sulfonamide_thiadiazole_NH", "[SX4](=O)(=O)[NX3;H1]c1nncs1",               5.5,  "acid"),
    ("sulfonamide_NH",             "[SX4](=O)(=O)[NX3;H1,H2]",                  10.1,  "acid"),
    ("barbiturate_NH",             "[NX3;H1;R]1[CX3;R](=O)[NX3;H1,H0;R][CX3;R](=O)[CX4;R][CX3;R]1=O", 7.4, "acid"),
    ("imide_NH",                   "[CX3](=O)[NX3;H1][CX3]=O",                   9.6,  "acid"),
    ("acylhydrazone_NH",   "[CX3](=O)[NX3;H1][NX2]=[CX3]",                   10.5,  "acid"),
    ("hydrazide_NH",       "[CX3](=O)[NX3;H1][NX3;H2]",                       10.5,  "acid"),
    ("urea_NH",            "[NX3;H1][CX3](=O)[NX3;H1,H2]",                    13.0,  "acid"),
    ("amide_NH",           "[CX3](=O)[NX3;H1,H2;!$([N]~N)]",                  15.0,  "acid"),
    ("phenol_diacyl",      "[OX2H1][c;R]1[c;R][c;R](=O)[c;R][c;R][c;R]1=O",   3.5,  "acid"),
    ("phenol_ortho_CO",    "[OX2H1][c;R]:[c;R][CX3;R](=O)",                    7.8,  "acid"),
    ("catechol_OH",        "[OX2H1][c;R]:[c;R][OX2H1]",                        9.2,  "acid"),
    ("phenol_EWG",         "[OX2H1][c;R]:[c;R][$([NX3](=O)=O),$([CX3]=O),"
                           "$(C#N),$([SX4](=O)(=O))]",                         7.2,  "acid"),
    ("phenol",             "c[OX2H1]",                                         10.0,  "acid"),
    ("thiol_arom",         "c[SX2H1]",                                          6.5,  "acid"),
    ("thiol_aliph",        "[CX4][SX2H1]",                                      9.8,  "acid"),
    ("aniline",            "c[NX3;H1,H2;!$(N~[!#6])]",                         4.6,  "base"),
    ("pyridine_like",      "[$([nX2]1:[c,n]:c:[c,n]:c1),$([nX2]:c:n)]",        5.2,  "base"),
    ("morpholine_N",       "[NX3;H0;R;$(N1CC[O,S]CC1)]",                       4.9,  "base"),
    ("piperazine_NH",      "[NX3;H1;R;$(N1CCNCC1)]",                           8.1,  "base"),
    ("piperazine_N_sub",   "[NX3;H0;R;$(N1CCNCC1)]",                           3.5,  "base"),
    ("amidine",            "[CX3](=[NX2;H0,H1])[NX3;H1,H2;"
                           "!$([NX3][CX3](=[NX2])[NX3])]",                    12.4,  "base"),
    ("guanidine",          "[NX3][CX3](=[NX2;!$(N~C#N)])[NX3]",               12.5,  "base"),
    ("aliphatic_amine",    "[NX3;H1,H2;!$(NC=O);!$(N~[!#6;!H]);!$([nH]);"
                           "!$(Nc);!$([NX3][CX3](=[NX2]))]",                  9.5,  "base"),
    ("aliphatic_amine_t",  "[NX3;H0;!$(NC=O);!$(Nc);!$([nH]);!$([N]~[!#6]);"
                           "!$([N;R]1CC[O,S]CC1);!$([N;R]1CCNCC1)]",          9.0,  "base"),
]


def _compile_ionizable_sites():
    from rdkit import Chem
    compiled = []
    for lbl, sma, pka, stype in _IONIZABLE_SITE_DEF:
        pat = Chem.MolFromSmarts(sma)
        if pat is not None:
            compiled.append((lbl, pat, pka, stype))
    return compiled

_IONIZABLE_SITES_COMPILED = _compile_ionizable_sites()


def _find_ionizable_sites(mol):
    sites         = []
    seen_ion      = set()
    claimed_atoms = set()

    for site in _find_flavone_A_ring_phenols(mol):
        ion_idx = site["ionizable_idx"]
        if ion_idx in seen_ion:
            continue
        seen_ion.add(ion_idx)
        claimed_atoms.update(site["atom_indices"])
        sites.append(site)

    _GROUP_SITE_LABELS = frozenset({"guanidine", "amidine"})

    for lbl, pat, pka, stype in _IONIZABLE_SITES_COMPILED:
        for match in mol.GetSubstructMatches(pat):
            if any(a in claimed_atoms for a in match):
                continue
            ion_atoms = [
                idx for idx in match
                if mol.GetAtomWithIdx(idx).GetAtomicNum() in (7, 8, 16)
                and mol.GetAtomWithIdx(idx).GetTotalNumHs() > 0
                and idx not in seen_ion
            ]
            if not ion_atoms:
                continue
            for ion_idx in ion_atoms:
                seen_ion.add(ion_idx)
                sites.append({
                    "label":         lbl,
                    "atom_indices":  [ion_idx],
                    "ionizable_idx": ion_idx,
                    "heuristic_pka": pka,
                    "site_type":     stype,
                })
                if lbl in _GROUP_SITE_LABELS:
                    for other_idx in match:
                        if (other_idx != ion_idx
                                and mol.GetAtomWithIdx(other_idx).GetAtomicNum() == 7):
                            seen_ion.add(other_idx)
                    break
    return sites


def _hh_fraction_charged(pka, ph, site_type):
    if site_type == "acid":
        return 1.0 / (1.0 + 10.0 ** (pka - ph))
    return 1.0 / (1.0 + 10.0 ** (ph - pka))


def _hh_match_score(pka, ph, site_type, actual_charge):
    f_charged = _hh_fraction_charged(pka, ph, site_type)
    dpH       = abs(ph - pka)
    decisive  = (f_charged >= 0.65) or (f_charged <= 0.35)
    rwd_mul   = 1.6 if decisive else 1.0
    pen_mul   = 1.6 if decisive else 1.0
    if site_type == "acid":
        expected_neg = f_charged > 0.5
        if expected_neg and actual_charge < 0:
            return  min(1.5, dpH * 0.55 * rwd_mul) + 0.15
        elif expected_neg:
            return -min(1.5, dpH * 0.45 * pen_mul) - 0.15
        elif actual_charge >= 0:
            return  0.15
        else:
            return -min(1.5, dpH * 0.45 * pen_mul) - 0.15
    else:
        expected_pos = f_charged > 0.5
        if expected_pos and actual_charge > 0:
            return  min(1.5, dpH * 0.55 * rwd_mul) + 0.15
        elif expected_pos:
            return -min(1.5, dpH * 0.45 * pen_mul) - 0.15
        elif actual_charge <= 0:
            return  0.15
        else:
            return -min(1.5, dpH * 0.45 * pen_mul) - 0.15


_CHEM_BONUS_DEF = [
    ("amide",            +2.5, "[CX3](=O)[NX3;H1,H2]"),
    ("lactam",           +2.5, "[C;R](=O)[N;R]"),
    ("urea_NH",          +1.5, "[NX3;H1][CX3](=O)[NX3;H1,H2]"),
    ("thioamide",        +1.0, "[CX3](=S)[NX3;H1,H2]"),
    ("aromatic_ring",    +0.3, "c1ccccc1"),
    ("phenol_preserved", _W_PHENOL_PRESERVED_BONUS, "c[OX2H1]"),
]
_CHEM_PENALTY_DEF = [
    ("lactim_ring",      -4.0, "[C;R](=[NX2])[OX2H1]"),
    ("iminol_general",   -3.5, "[NX2]=[CX3][OX2H1]"),
    ("amide_N_deproton", -5.0, "[$([NX3-]C=O),$([NX3-]c=O)]"),
    ("enol_simple",      -1.2, "[CX3](=[CX3])[OX2H1]"),
    ("exo_imine_arom",   -2.5, "[NX2;!r]=[cX3]"),
    ("pyrogallol_triketo", -_W_PYROGALLOL_TRIKETO,
        "[#6;!a;R]1(=O)[#6;!a;R](=O)[#6;!a;R](=O)[#6;R][#6;R][#6;R]1"),
    ("catechol_diketo",    -_W_CATECHOL_DIKETO,
        "[#6;!a;R]1(=O)[#6;!a;R](=O)[#6;R][#6;R][#6;R][#6;R]1"),
]


def _compile_chem_rules():
    from rdkit import Chem
    rules = []
    for lbl, wt, sma in _CHEM_BONUS_DEF + _CHEM_PENALTY_DEF:
        pat = Chem.MolFromSmarts(sma)
        if pat is not None:
            rules.append((lbl, wt, pat))
    return rules

_CHEM_RULES = _compile_chem_rules()


def _n_aromatic_rings(mol):
    if mol is None:
        return 0
    try:
        from rdkit.Chem import rdMolDescriptors
        return int(rdMolDescriptors.CalcNumAromaticRings(mol))
    except Exception:
        return 0


def _count_phenolic_OH(mol):
    if mol is None:
        return 0
    from rdkit import Chem
    pat = Chem.MolFromSmarts("c[OX2H1]")
    if pat is None:
        return 0
    return len(mol.GetSubstructMatches(pat))


def _score_tautomer(smiles, ref_mol=None):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -999.0
    total = 0.0
    for lbl, wt, pat in _CHEM_RULES:
        n = len(mol.GetSubstructMatches(pat))
        if n:
            total += wt * n
    if ref_mol is not None:
        rings_lost   = max(0, _n_aromatic_rings(ref_mol) - _n_aromatic_rings(mol))
        phenols_lost = max(0, _count_phenolic_OH(ref_mol) - _count_phenolic_OH(mol))
        if rings_lost > 0:
            total += -_W_AROM_RING_LOST * rings_lost
        if phenols_lost > 0:
            total += -_W_PHENOL_TO_KETO_FLIP * phenols_lost
    return total


def _score_microstate(smiles, ph, taut_score, pubchem, ref_mol=None, ion_sites=None):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1e9

    if ion_sites is None:
        ion_sites = _find_ionizable_sites(ref_mol if ref_mol is not None else mol)
    fc_map    = {a.GetIdx(): int(a.GetFormalCharge()) for a in mol.GetAtoms()}
    net       = sum(fc_map.values())
    n_pos     = sum(1 for v in fc_map.values() if v > 0)
    n_neg     = sum(1 for v in fc_map.values() if v < 0)

    pat_bad = Chem.MolFromSmarts("[$([NX3-]C=O),$([NX3-]c=O)]")
    s1 = -5.0 * len(mol.GetSubstructMatches(pat_bad)) if pat_bad else 0.0

    s2 = 0.0
    if ref_mol is not None:
        rings_lost = max(0, _n_aromatic_rings(ref_mol) - _n_aromatic_rings(mol))
        if rings_lost > 0:
            s2 = -_W_AROM_RING_LOST * rings_lost

    s3 = 0.65 * taut_score

    s4 = 0.0
    for site in ion_sites:
        pka = site["heuristic_pka"]
        if pubchem.get("available") and pubchem.get("confidence") in ("high", "medium"):
            pc_vals = pubchem.get("pka_values", [])
            if pc_vals:
                pka = min(pc_vals, key=lambda v: abs(v - pka))
        site_charge = sum(fc_map.get(i, 0) for i in site["atom_indices"])
        s4 += _hh_match_score(pka, ph, site["site_type"], site_charge)

    s5 = 0.0
    if pubchem.get("available"):
        w = {"high": 1.0, "medium": 0.6, "low": 0.2}.get(
            pubchem.get("confidence", "low"), 0.2)
        for pv in pubchem.get("pka_values", []):
            exp = -1 if _hh_fraction_charged(pv, ph, "acid") > 0.5 else 0
            s5 += 0.25 * w if net == exp else -0.15 * w
        s5 = max(-0.4, min(0.5, s5))

    has_acid = any(s["site_type"] == "acid"
                   and (ph - s["heuristic_pka"]) > 1.0 for s in ion_sites)
    has_base = any(s["site_type"] == "base"
                   and (s["heuristic_pka"] - ph) > 1.0 for s in ion_sites)
    is_zw = (n_pos > 0 and n_neg > 0 and net == 0)
    if is_zw:
        s6 = 0.8 if (has_acid and has_base) else -0.6
    else:
        s6 = -0.4 if (has_acid and has_base and net == 0 and n_pos == 0) else 0.0

    strong_acid   = [s for s in ion_sites if s["site_type"] == "acid"
                     and (ph - s["heuristic_pka"]) > 2.0]
    strong_base   = [s for s in ion_sites if s["site_type"] == "base"
                     and (s["heuristic_pka"] - ph) > 2.0]
    probable_acid = [s for s in ion_sites if s["site_type"] == "acid"
                     and 0.0 < (ph - s["heuristic_pka"]) <= 2.0]
    probable_base = [s for s in ion_sites if s["site_type"] == "base"
                     and 0.0 < (s["heuristic_pka"] - ph) <= 2.0]
    s7 = 0.0
    if strong_acid  and net >= 0 and n_neg == 0:
        s7 -= 0.5  * len(strong_acid)
    if strong_base  and net <= 0 and n_pos == 0:
        s7 -= 0.5  * len(strong_base)
    if probable_acid and net >= 0 and n_neg == 0:
        s7 -= 0.35 * len(probable_acid)
    if probable_base and net <= 0 and n_pos == 0:
        s7 -= 0.35 * len(probable_base)

    s8 = -0.12 * max(0, n_pos + n_neg - 2)

    return s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8


def _manual_deprotonate_site(smiles, site):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    target_idx = None
    for idx in site["atom_indices"]:
        if idx >= rw.GetNumAtoms():
            continue
        atom = rw.GetAtomWithIdx(idx)
        sym, nh = atom.GetSymbol(), atom.GetTotalNumHs()
        if site["site_type"] == "acid":
            if sym in ("O", "S") and nh >= 1:
                target_idx = idx; break
            if sym == "N" and nh >= 1 and target_idx is None:
                target_idx = idx
        else:
            if sym == "N" and atom.GetFormalCharge() == 0:
                target_idx = idx; break
    if target_idx is None:
        return None
    try:
        atom = rw.GetAtomWithIdx(target_idx)
        if site["site_type"] == "acid":
            atom.SetFormalCharge(-1)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
        else:
            atom.SetFormalCharge(+1)
            atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
            atom.SetNoImplicit(False)
        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
        return Chem.MolToSmiles(new_mol, isomericSmiles=True, canonical=True)
    except Exception:
        return None


def _supplement_dimorphite(tautomer_smiles, dimorphite_results, ion_sites, target_ph):
    supplemented = list(dimorphite_results)
    existing     = set(dimorphite_results)
    for site in ion_sites:
        pka   = site.get("heuristic_pka", 10.0)
        stype = site.get("site_type", "acid")
        if stype == "acid" and (target_ph - pka) < -1.5:
            continue
        if stype == "base" and (pka - target_ph) < -1.5:
            continue
        new_smi = _manual_deprotonate_site(tautomer_smiles, site)
        if new_smi and new_smi not in existing:
            supplemented.append(new_smi)
            existing.add(new_smi)
    return supplemented


def _pubchem_pka_lookup(smiles: str) -> dict:
    from rdkit import Chem
    result = {"available": False, "pka_values": [], "confidence": "low",
              "source": "pubchem"}
    try:
        import requests as _rq
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return result
        ik = Chem.MolToInchiKey(mol)
        if not ik:
            return result
        if ik in _PKANET_CACHE:
            return _PKANET_CACHE[ik]
        time.sleep(0.25)
        r = _rq.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{ik}/cids/JSON",
            timeout=8)
        if r.status_code != 200:
            _PKANET_CACHE[ik] = result; return result
        cid = r.json()["IdentifierList"]["CID"][0]
        time.sleep(0.25)
        r2 = _rq.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
            "?heading=Dissociation+Constants", timeout=10)
        if r2.status_code != 200:
            _PKANET_CACHE[ik] = result; return result
        texts = []
        def _collect(sec):
            if "dissociation" in sec.get("TOCHeading", "").lower():
                for info in sec.get("Information", []):
                    for swm in info.get("Value", {}).get("StringWithMarkup", []):
                        s = swm.get("String", "").strip()
                        if s: texts.append(s)
            for sub in sec.get("Section", []): _collect(sub)
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


def _generate_ranked_microstates(
    base_smiles,
    target_ph=7.4,
    ph_window=1.0,
    max_tautomers=8,
    pubchem=None,
):
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    if pubchem is None:
        pubchem = {}

    ref_mol = Chem.MolFromSmiles(base_smiles)
    if ref_mol is None:
        return []

    enum   = rdMolStandardize.TautomerEnumerator()
    seen_t = set()
    tautomers = []
    input_canon = Chem.MolToSmiles(ref_mol, isomericSmiles=True, canonical=True)
    seen_t.add(input_canon)
    tautomers.append({
        "smiles": input_canon,
        "score":  _score_tautomer(input_canon, ref_mol=ref_mol),
    })
    try:
        for tmol in enum.Enumerate(ref_mol):
            smi = Chem.MolToSmiles(tmol, isomericSmiles=True, canonical=True)
            if smi in seen_t:
                continue
            seen_t.add(smi)
            tautomers.append({
                "smiles": smi,
                "score":  _score_tautomer(smi, ref_mol=ref_mol),
            })
    except Exception:
        pass

    tautomers = sorted(tautomers, key=lambda x: -x["score"])[:max_tautomers]
    best_t    = tautomers[0]["score"]
    kept      = [t for t in tautomers
                 if t["score"] >= best_t - _TAUTOMER_PLAUSIBILITY_CUTOFF]
    if not kept:
        kept = [tautomers[0]]

    ion_sites = _find_ionizable_sites(ref_mol)

    ph_lo = max(0.0,  target_ph - ph_window / 2)
    ph_hi = min(14.0, target_ph + ph_window / 2)

    all_micro = []
    seen_smi  = set()

    for taut in kept:
        raw_states = [taut["smiles"]]
        try:
            from dimorphite_dl import protonate_smiles as _dim
            _raw = None
            for kw in [
                {"ph_min": ph_lo, "ph_max": ph_hi, "max_variants": 32},
                {"min_ph": ph_lo, "max_ph": ph_hi, "max_variants": 32},
            ]:
                try:
                    _raw = _dim(taut["smiles"], **kw)
                    break
                except TypeError:
                    continue
            if _raw:
                seen_d = {taut["smiles"]}
                for s in (_raw if isinstance(_raw, list) else [_raw]):
                    if not s:
                        continue
                    m = Chem.MolFromSmiles(s)
                    if m is None:
                        continue
                    c = Chem.MolToSmiles(m, canonical=True)
                    if c and c not in seen_d:
                        seen_d.add(c)
                        raw_states.append(c)
        except Exception:
            pass

        microstates = _supplement_dimorphite(
            taut["smiles"], raw_states, ion_sites, target_ph
        )

        for smi in microstates:
            if smi in seen_smi:
                continue
            mol_check = Chem.MolFromSmiles(smi)
            if mol_check is None:
                continue
            seen_smi.add(smi)
            sc  = _score_microstate(smi, target_ph, taut["score"], pubchem,
                                    ref_mol=ref_mol, ion_sites=ion_sites)
            net = int(Chem.GetFormalCharge(mol_check))
            all_micro.append({
                "microstate_smiles": smi,
                "tautomer_smiles":   taut["smiles"],
                "selection_score":   float(sc),
                "net_charge":        net,
            })

    all_micro.sort(key=lambda x: (
        -x["selection_score"],
        abs(x["net_charge"]),
        x["microstate_smiles"],
    ))
    return all_micro


def _apply_ionizable_site_correction(original_smiles: str, current_smiles: str,
                                      ph: float, log: list) -> str:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(current_smiles)
    if mol is None:
        return current_smiles

    fc_map = {a.GetIdx(): int(a.GetFormalCharge()) for a in mol.GetAtoms()}
    sites  = _find_ionizable_sites(mol)

    missed = sorted(
        [s for s in sites
         if s["site_type"] == "acid"
         and s["heuristic_pka"] < ph
         and s.get("ionizable_idx") is not None
         and mol.GetAtomWithIdx(s["ionizable_idx"]).GetTotalNumHs() > 0
         and fc_map.get(s["ionizable_idx"], 0) >= 0],
        key=lambda s: s["heuristic_pka"]
    )
    if not missed:
        return current_smiles

    site = missed[0]
    try:
        rw = Chem.RWMol(mol)
        rw.GetAtomWithIdx(site["ionizable_idx"]).SetFormalCharge(-1)
        Chem.SanitizeMol(rw)
        corrected = Chem.MolToSmiles(rw, canonical=True)
        log.append(
            f"✓ pKa site correction: deprotonated {site['label']} "
            f"(pKa={site['heuristic_pka']:.1f} < pH {ph:.1f})"
        )
        return corrected
    except Exception as e:
        log.append(f"⚠ Site correction failed ({site['label']}): {e}")
        return current_smiles


def _load_local_pkanet_module(log=None):
    import importlib.util
    import sys
    import hashlib
    from pathlib import Path

    pkanet_path = Path(__file__).resolve().with_name("pKaNET.py")

    if not pkanet_path.exists():
        raise FileNotFoundError(f"pKaNET.py not found next to core.py: {pkanet_path}")

    module_name = "pKaNET_local_ACD"

    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(pkanet_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pKaNET.py from: {pkanet_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    file_hash = hashlib.md5(pkanet_path.read_bytes()).hexdigest()[:12]

    if log is not None:
        log.append(f"✓ pKaNET loaded from: {pkanet_path.name}  (md5: {file_hash})")

    return mod


def _pkanet_stereo_status(smiles: str) -> dict:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "assigned": 0, "unassigned": 0, "centers": []}
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    assigned = [(i, lab) for i, lab in centers if lab in ("R", "S")]
    unassigned = [(i, lab) for i, lab in centers if lab == "?"]
    return {"valid": True, "assigned": len(assigned), "unassigned": len(unassigned), "centers": centers}


def _select_pkanet_stereoisomer(smiles: str, stereo_choice: str = "keep_input") -> tuple[str, list[str], dict]:
    from rdkit import Chem
    from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

    log = []
    choice = (stereo_choice or "keep_input").strip().upper()
    if choice in {"KEEP", "KEEP_INPUT", "AS_IS", "AUTO", ""}:
        choice = "KEEP_INPUT"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles[:60]}")

    status = _pkanet_stereo_status(smiles)
    if status["assigned"] > 0:
        log.append("✓ Stereochemistry already defined in input SMILES (@/@@); keeping input stereochemistry")
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True), log, status

    if choice not in {"R", "S"}:
        if status["unassigned"] > 0:
            log.append(f"ℹ Undefined stereocenter(s) detected: {status['unassigned']}; keeping input without forced R/S")
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True), log, status

    if status["unassigned"] == 0:
        log.append(f"ℹ No undefined stereocenter detected; requested {choice} ignored")
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True), log, status

    if status["unassigned"] != 1:
        log.append(f"⚠ {status['unassigned']} undefined stereocenters detected; R/S selector is only applied to single-center ligands. Keeping input.")
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True), log, status

    opts = StereoEnumerationOptions(onlyUnassigned=True, unique=True)
    isomers = list(EnumerateStereoisomers(mol, options=opts))
    for iso in isomers:
        centers = Chem.FindMolChiralCenters(iso, includeUnassigned=True, useLegacyImplementation=False)
        labels = [lab for _, lab in centers if lab in ("R", "S")]
        if len(labels) == 1 and labels[0] == choice:
            smi = Chem.MolToSmiles(iso, isomericSmiles=True, canonical=True)
            log.append(f"✓ Undefined stereocenter resolved as {choice} for docking")
            return smi, log, status

    log.append(f"⚠ Could not generate requested {choice} stereoisomer; keeping input")
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True), log, status


def protonate_pkanet(
    smiles: str,
    ph: float,
    use_pubchem: bool = False,
    max_tautomers: int = 8,
    ph_window: float = 1.0,
    selection_mode: str = "auto_recommended",
    manual_rank: int | None = None,
    stereo_choice: str = "keep_input",
    return_details: bool = False,
) -> tuple:
    from rdkit import Chem

    log = []

    try:
        pk = _load_local_pkanet_module(log)

        _pkanet_standardize = pk.standardize_smiles
        _pkanet_pubchem = pk.pubchem_lookup
        _pkanet_rank = pk.generate_ranked_microstates

        log.append("✓ pKaNET imported from local pKaNET.py")
    except Exception as e:
        log.append(f"❌ pKaNET import failed: {e}")
        raise RuntimeError(
            "pKaNET mode was selected, but local pKaNET.py could not be loaded. "
            "Please check that pKaNET.py is in the same folder as core.py."
        ) from e

    stereo_input, stereo_log, stereo_status = _select_pkanet_stereoisomer(smiles.strip(), stereo_choice)
    log.extend(stereo_log)

    canonical, std_status = _pkanet_standardize(stereo_input)
    if canonical is None:
        log.append(f"❌ Standardization failed ({std_status})")
        raise ValueError(f"pKaNET standardization failed: {std_status}")

    log.append("✓ RDKit standardized (pKaNET)")

    pubchem_result = {}
    if use_pubchem:
        try:
            pubchem_result = _pkanet_pubchem(canonical)
            if pubchem_result.get("available"):
                log.append(
                    f"✓ PubChem pKa: {pubchem_result.get('pka_values', [])} "
                    f"(conf: {pubchem_result.get('confidence', '?')})"
                )
            else:
                log.append(
                    f"ℹ PubChem: no usable pKa data — heuristic table "
                    f"({pubchem_result.get('error', '')})"
                )
        except Exception as e:
            pubchem_result = {}
            log.append(f"⚠ PubChem lookup failed: {e}")

    try:
        top, ambiguous, all_micro, tr_flag, tr_motifs, ml_preds = _pkanet_rank(
            canonical,
            target_ph=ph,
            ph_window=ph_window,
            max_tautomers=max_tautomers,
            top_n=5,
            pubchem_result=pubchem_result,
        )
    except Exception as e:
        log.append(f"❌ pKaNET ranking failed: {e}")
        raise RuntimeError("pKaNET ranking failed. No fallback was used.") from e

    if not top:
        log.append("❌ pKaNET returned no microstates")
        raise RuntimeError("pKaNET returned no microstates. No fallback was used.")

    for r in all_micro[:5]:
        marker = " ← selected" if r.get("recommended_default") else ""
        log.append(
            f"  rank {r.get('microstate_rank','?')}: "
            f"score={float(r.get('selection_score',0)):.3f}  "
            f"charge={int(r.get('net_charge',0)):+d}  "
            f"{r.get('microstate_smiles','')}{marker}"
        )

    selection_mode = (selection_mode or "auto_recommended").strip().lower()
    if selection_mode in {"auto", "recommended", "auto recommended"}:
        selection_mode = "auto_recommended"
    elif selection_mode in {"highest", "highest_score", "rank1", "rank_1"}:
        selection_mode = "highest_score"
    elif selection_mode in {"manual", "manual_rank"}:
        selection_mode = "manual_rank"

    if selection_mode == "manual_rank" and manual_rank is not None:
        best = next((r for r in all_micro if int(r.get("microstate_rank", -1)) == int(manual_rank)), None)
        if best is None:
            log.append(f"⚠ Requested manual microstate rank {manual_rank} not found; using auto recommendation")
            best = next((r for r in all_micro if r.get("recommended_default")), top[0])
            selection_mode = "auto_recommended"
    elif selection_mode == "highest_score":
        best = top[0]
    else:
        best = next((r for r in all_micro if r.get("recommended_default")), top[0])
        selection_mode = "auto_recommended"

    best_smi = best["microstate_smiles"]
    charge = int(best["net_charge"])
    selected_rank = int(best.get("microstate_rank", 1))

    log.append(
        f"✓ pKaNET: ranked {len(all_micro)} microstates — "
        f"selected rank: {selected_rank}  "
        f"score: {float(best.get('selection_score', 0.0)):.3f}  "
        f"charge: {charge:+d}  "
        f"mode: {selection_mode}  "
        f"backend: {best.get('decision_backend', '?')} ({best.get('decision_mode', '?')})"
    )
    if selected_rank != 1:
        log.append(
            f"ℹ Rank-1 was not used for docking; conservative/manual selected rank {selected_rank}. "
            f"Reason: {best.get('recommendation_reason', 'user/auto selection')}"
        )
    log.append(f"✓ pKa source: {best.get('pKa_source', 'heuristic')}")

    if ambiguous:
        if len(top) > 1:
            log.append(f"⚠ Top assignment ambiguous — alt charge {int(top[1].get('net_charge', 0)):+d}")
        else:
            log.append("⚠ Top assignment ambiguous")

    if tr_flag:
        log.append(f"⚠ Tautomer-rich motifs: {', '.join(tr_motifs)}")

    log.append(f"✓ Final pKaNET SMILES: {best_smi}")
    log.append(f"✓ Formal charge: {charge:+d}")

    mol = Chem.MolFromSmiles(best_smi)
    if mol is None:
        raise RuntimeError(f"pKaNET returned invalid SMILES: {best_smi}")
    actual_charge = int(Chem.GetFormalCharge(mol))
    if actual_charge != charge:
        log.append(
            f"⚠ Charge mismatch corrected: pKaNET reported {charge:+d}, "
            f"RDKit formal charge is {actual_charge:+d}"
        )
        charge = actual_charge

    details = {
        "selection_mode": selection_mode,
        "stereo_choice": stereo_choice,
        "stereo_status": stereo_status,
        "selected_rank": int(best.get("microstate_rank", 1)),
        "selected_microstate": {k: v for k, v in best.items() if k != "charged_atom_rows"},
        "ranked_microstates": [{k: v for k, v in r.items() if k != "charged_atom_rows"} for r in all_micro],
        "top_microstates": [{k: v for k, v in r.items() if k != "charged_atom_rows"} for r in top],
        "ambiguous": bool(ambiguous),
        "tautomer_rich": bool(tr_flag),
        "tautomer_rich_motifs": tr_motifs,
        "pubchem_result": pubchem_result,
        "ml_predictions": ml_preds,
    }
    if return_details:
        return best_smi, charge, log, details
    return best_smi, charge, log


# ══════════════════════════════════════════════════════════════════════════════
#  LIGAND PREPARATION  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════


def _meeko_to_pdbqt(mol, out_path: str):
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


def _ligand_charge_summary(smiles: str) -> dict:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles[:60]}")
    net = n_pos = n_neg = 0
    rows = []
    for atom in mol.GetAtoms():
        fc = int(atom.GetFormalCharge())
        net += fc
        if fc > 0:
            n_pos += 1
        elif fc < 0:
            n_neg += 1
        if fc != 0:
            rows.append({
                "atom_idx": atom.GetIdx(),
                "symbol": atom.GetSymbol(),
                "formal_charge": fc,
            })
    return {
        "net_charge": int(net),
        "charged_atoms": rows,
        "is_zwitterion": bool(n_pos > 0 and n_neg > 0 and net == 0),
    }

def _charged_atoms_text(rows: list) -> str:
    if not rows:
        return "none"
    return ", ".join(f"{r['symbol']}{r['atom_idx']}({r['formal_charge']:+d})" for r in rows)


def prepare_ligand(
    smiles: str,
    name: str,
    ph: float,
    wdir,
    mode: str = "pkanet",
    use_pubchem: bool = True,
    max_tautomers: int = 8,
    ph_window: float = 1.0,
    pkanet_selection_mode: str = "auto_recommended",
    pkanet_manual_rank: int | None = None,
    pkanet_stereo_choice: str = "keep_input",
) -> dict:
    _rdkit_six_patch()
    from rdkit import Chem
    from rdkit.Chem import AllChem

    wdir      = Path(wdir)
    log       = []
    out_pdbqt = str(wdir / f"{name}.pdbqt")
    out_sdf   = str(wdir / f"{name}_3d.sdf")

    try:
        raw = smiles.strip()
        prot = raw
        pkanet_details = {}
        pkanet_ranked_csv = ""
        pkanet_decision_log = ""
        actual_mode = mode or "dimorphite"

        if actual_mode == "neutral":
            mol_check = Chem.MolFromSmiles(raw)
            if mol_check is None:
                raise ValueError(f"RDKit could not parse SMILES: {raw[:60]}")
            prot = Chem.MolToSmiles(mol_check, isomericSmiles=True, canonical=True)
            log.append("✓ Neutral mode (keep input charge state)")

        elif actual_mode == "pkanet":
            try:
                prot, _, pka_log, pkanet_details = protonate_pkanet(
                    raw, ph,
                    use_pubchem=use_pubchem,
                    max_tautomers=max_tautomers,
                    ph_window=ph_window,
                    selection_mode=pkanet_selection_mode,
                    manual_rank=pkanet_manual_rank,
                    stereo_choice=pkanet_stereo_choice,
                    return_details=True,
                )
                try:
                    import csv as _csv
                    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name)) or "ligand"
                    pkanet_ranked_csv = str(wdir / f"{safe_name}_pkanet_ranked_microstates.csv")
                    rows = pkanet_details.get("ranked_microstates", []) or []
                    if rows:
                        preferred = [
                            "microstate_rank", "recommended_default", "recommendation", "selection_score",
                            "delta_from_best", "net_charge", "microstate_smiles", "charged_atoms",
                            "pKa_source", "decision_backend", "decision_mode", "ambiguous_top_assignment",
                            "flag_heuristic_only", "flag_polyphenol_like", "flag_coumarin_like", "flag_flavonoid_like",
                            "flag_possible_overdeprotonation", "flag_borderline_pka", "flag_tautomer_rich",
                            "recommendation_reason",
                        ]
                        extra = sorted({k for r in rows for k in r.keys()} - set(preferred))
                        fieldnames = preferred + extra
                        with open(pkanet_ranked_csv, "w", newline="", encoding="utf-8") as fh:
                            wr = _csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                            wr.writeheader(); wr.writerows(rows)
                    pkanet_decision_log = str(wdir / f"{safe_name}_pkanet_decision_log.txt")
                    with open(pkanet_decision_log, "w", encoding="utf-8") as fh:
                        fh.write("\n".join(pka_log))
                    log.append(f"✓ pKaNET ranked microstates exported: {pkanet_ranked_csv}")
                    log.append(f"✓ pKaNET decision log exported: {pkanet_decision_log}")
                except Exception as _csv_e:
                    log.append(f"⚠ Could not export pKaNET ranked table: {_csv_e}")
                log.extend(pka_log)
                log.append("✓ pKaNET Cloud protonation applied")
            except Exception as _pke:
                log.append(f"⚠ pKaNET failed ({_pke}) — falling back to Dimorphite-DL")
                try:
                    from dimorphite_dl import protonate_smiles
                    _win = ph_window * 0.5
                    vs = protonate_smiles(
                        raw,
                        ph_min=ph - _win, ph_max=ph + _win,
                        max_variants=16,
                    )
                    if vs:
                        _vs_list = vs if isinstance(vs, list) else [vs]
                        from rdkit import Chem as _Chem
                        def _abs_charge(s):
                            m = _Chem.MolFromSmiles(s)
                            return abs(int(_Chem.GetFormalCharge(m))) if m else 99
                        prot = min(_vs_list, key=_abs_charge)
                        log.append(
                            f"✓ Dimorphite-DL fallback pH {ph:.1f} "
                            f"(picked charge {_abs_charge(prot):+d} from {len(_vs_list)} variants)"
                        )
                    prot = _apply_ionizable_site_correction(raw, prot, ph, log)
                except Exception as e2:
                    log.append(f"⚠ Dimorphite-DL fallback skipped: {e2}")

        else:
            try:
                from dimorphite_dl import protonate_smiles
                _win = ph_window * 0.5
                vs = protonate_smiles(
                    prot,
                    ph_min=ph - _win, ph_max=ph + _win,
                    max_variants=16,
                )
                if vs:
                    _vs_list = vs if isinstance(vs, list) else [vs]
                    from rdkit import Chem as _Chem
                    def _abs_charge(s):
                        m = _Chem.MolFromSmiles(s)
                        return abs(int(_Chem.GetFormalCharge(m))) if m else 99
                    prot = min(_vs_list, key=_abs_charge)
                    log.append(f"✓ Dimorphite-DL pH {ph:.1f}")
                else:
                    log.append("⚠ Dimorphite-DL returned no variants — using input SMILES")
            except Exception as e:
                log.append(f"⚠ Dimorphite-DL skipped: {e}")
            prot = _apply_ionizable_site_correction(raw, prot, ph, log)

        mol = Chem.MolFromSmiles(prot)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {prot[:60]}")

        charge_info = _ligand_charge_summary(prot)
        charge = int(charge_info["net_charge"])
        log.append(f"✓ Formal charge: {charge:+d}")
        log.append(f"✓ Charged atoms: {_charged_atoms_text(charge_info['charged_atoms'])}")

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

        try:
            _meeko_to_pdbqt(mol, out_pdbqt)
            log.append("✓ PDBQT written (Meeko)")
        except Exception as e_meeko:
            log.append(f"⚠ Meeko failed ({e_meeko}), trying OpenBabel…")
            subprocess.run(
                ["obabel", out_sdf, "-O", out_pdbqt, "-xh"],
                capture_output=True, timeout=30,
            )
            if not Path(out_pdbqt).exists() or Path(out_pdbqt).stat().st_size < 10:
                raise ValueError(f"Both Meeko and OpenBabel failed: {e_meeko}")
            log.append("✓ PDBQT written (OpenBabel fallback)")

        return {
            "success":           True,
            "pdbqt":             out_pdbqt,
            "sdf":               out_sdf,
            "input_smiles":      raw,
            "prepared_smiles":   prot,
            "prot_smiles":       prot,
            "charge":            charge,
            "net_charge":        charge,
            "charge_method":     "rdkit_formal_charge",
            "charged_atoms":     charge_info["charged_atoms"],
            "is_zwitterion":     charge_info["is_zwitterion"],
            "protonation_mode":  actual_mode,
            "pkanet_selection_mode": pkanet_details.get("selection_mode", pkanet_selection_mode),
            "pkanet_selected_rank": pkanet_details.get("selected_rank"),
            "pkanet_selected_microstate": pkanet_details.get("selected_microstate", {}),
            "pkanet_ranked_microstates": pkanet_details.get("ranked_microstates", []),
            "pkanet_ranked_csv": pkanet_ranked_csv,
            "pkanet_decision_log": pkanet_decision_log,
            "pkanet_ambiguous": pkanet_details.get("ambiguous", False),
            "log":               log,
        }

    except Exception as e:
        log.append(f"ERROR: {e}")
        return {"success": False, "error": str(e), "log": log}

def prepare_ligand_from_file(file_path: str, name: str, wdir) -> dict:
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
            subprocess.run(
                ["obabel", file_path, "-O", _ob_sdf],
                capture_output=True, timeout=30,
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
            subprocess.run(
                ["obabel", out_sdf, "-O", out_pdbqt, "-xh"],
                capture_output=True, timeout=30,
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
#  CO-CRYSTAL LIGAND  —  SMILES ACQUISITION
# ══════════════════════════════════════════════════════════════════════════════

def _rcsb_ccd_smiles(resname: str):
    """
    Fetch ideal isomeric SMILES for a PDB Chemical Component (3-letter code)
    from the RCSB Chemical Component Dictionary (CCD) REST API.

    Priority: SMILES_STEREO  >  SMILES  (isomeric before canonical).

    Returns the SMILES string on success, None on any failure (network error,
    unknown residue, missing descriptor field).
    """
    import requests
    url = (
        f"https://data.rcsb.org/rest/v1/core/chemcomp/"
        f"{resname.strip().upper()}"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        descriptors = data.get("rcsb_chem_comp_descriptor", [])
        if isinstance(descriptors, dict):
            descriptors = [descriptors]
        smiles_by_type: dict = {}
        for item in descriptors:
            t = (item.get("type") or "").upper()
            d = (item.get("descriptor") or "").strip()
            if d and t in ("SMILES_STEREO", "SMILES"):
                smiles_by_type.setdefault(t, d)
        return smiles_by_type.get("SMILES_STEREO") or smiles_by_type.get("SMILES")
    except Exception:
        return None


def _smiles_from_cif_block(cif_path: str, resname: str):
    """
    Extract SMILES from the ``_chem_comp.pdbx_smiles`` field of an mmCIF
    structure file.

    Handles three CIF layouts:

    *  **loop_** block  — a table whose header contains ``_chem_comp.id`` and
       ``_chem_comp.pdbx_smiles``.  We find the row whose ``id`` matches
       *resname* (exact) **or** starts with ``resname + "_"`` (e.g. ``LIG_B``
       for resname ``LIG``).
    *  **Inline single-value** — ``_chem_comp.pdbx_smiles   <value>`` on the
       same line (horizontal whitespace only between key and value).
    *  **Two-line single-value** — key on one line, value token on the next
       (but NOT another ``_``-key or a ``;``-block start).

    Returns the SMILES string on success, None otherwise.
    Ignores placeholder values ``"."`` and ``"?"``.
    """
    target = resname.strip().upper()
    try:
        with open(cif_path, errors="replace") as fh:
            content = fh.read()

        # ── Strategy A: loop_ containing _chem_comp columns ──────────────
        loop_pat = _re.compile(
            r"loop_\s*((?:_chem_comp\.\S+\s*)+)(.*?)(?=loop_|data_|\Z)",
            _re.DOTALL,
        )
        for m in loop_pat.finditer(content):
            header_block = m.group(1)
            data_block   = m.group(2)
            headers = header_block.split()
            if "_chem_comp.id" not in headers:
                continue
            if "_chem_comp.pdbx_smiles" not in headers:
                continue
            id_idx  = headers.index("_chem_comp.id")
            smi_idx = headers.index("_chem_comp.pdbx_smiles")
            n       = len(headers)
            tokens  = _re.findall(r"'[^']*'|\"[^\"]*\"|[^\s]+", data_block)
            for i in range(0, len(tokens) - n + 1, n):
                row = tokens[i : i + n]
                if len(row) < n:
                    break
                comp_id = row[id_idx].strip("'\"").upper()
                # Exact match OR prefix match: "LIG_B" → target "LIG"
                if comp_id == target or comp_id.startswith(target + "_"):
                    smi = row[smi_idx].strip("'\"")
                    if smi and smi not in (".", "?"):
                        return smi

        # ── Strategy B: inline single-value (same line, no newline allowed) ──
        # Use [ \t]+ (horizontal whitespace only) to stop at end of line so
        # we never accidentally capture the next field name on the line below.
        m = _re.search(
            r"_chem_comp\.pdbx_smiles[ \t]+([^\s#][^\r\n]*)", content
        )
        if m:
            smi = m.group(1).strip().strip("'\"")
            if smi and smi not in (".", "?"):
                return smi

        # ── Strategy C: two-line single-value  ────────────────────────────
        # key on one line, plain value token on the very next non-empty line,
        # but only if that line does NOT start with another key (_), a
        # comment (#), or a semicolon-block (;).
        m = _re.search(
            r"_chem_comp\.pdbx_smiles\s*\r?\n[ \t]*([^_#;\s][^\r\n]*)",
            content,
        )
        if m:
            smi = m.group(1).strip().strip("'\"")
            if smi and smi not in (".", "?"):
                return smi

        # ── Strategy D: semicolon text block ─────────────────────────────
        # _chem_comp.pdbx_smiles
        # ;
        # <value>
        # ;
        m = _re.search(
            r"_chem_comp\.pdbx_smiles\s*\r?\n;\r?\n([^\n]+)\r?\n;",
            content,
        )
        if m:
            smi = m.group(1).strip().strip("'\"")
            if smi and smi not in (".", "?"):
                return smi

    except Exception:
        pass
    return None


def get_cocrystal_smiles(
    ligand_pdb_path: str,
    cocrystal_ligand_id: str,
    raw_pdb: str = "",
) -> tuple:
    """
    Return ``(smiles, source, warning)`` for the extracted co-crystal ligand.

    Acquisition strategy (in priority order):

    1. **RCSB CCD REST API** — ideal, stereo-correct, curated SMILES.
       Uses the 3-letter residue code parsed from *cocrystal_ligand_id*
       (format ``RESNAME_CHAIN_RESID``).

    2. **CIF _chem_comp.pdbx_smiles** — parsed directly from the structure
       CIF if *raw_pdb* is a CIF file.  No network call required.

    3. **3D → SMILES conversion** (last resort) — RDKit reads the extracted
       ``LIG.pdb``.  PDB format carries no bond-order information, so
       aromatic/double-bond assignment is heuristic and may be wrong for
       complex ligands.  A warning is always attached to this source.

    Returns
    -------
    smiles  : str — SMILES string, empty on total failure.
    source  : str — ``"rcsb_ccd"`` | ``"cif_block"`` | ``"3d_conversion"`` | ``""``.
    warning : str — non-empty when the result may be unreliable, or on failure.
    """
    resname = (
        cocrystal_ligand_id.split("_")[0].upper()
        if cocrystal_ligand_id
        else ""
    )

    # ── 1. RCSB CCD ──────────────────────────────────────────────────────
    if resname:
        try:
            smi = _rcsb_ccd_smiles(resname)
            if smi:
                return smi, "rcsb_ccd", ""
        except Exception:
            pass

    # ── 2. CIF embedded block ────────────────────────────────────────────
    if raw_pdb and is_cif_file(raw_pdb) and resname:
        try:
            smi = _smiles_from_cif_block(raw_pdb, resname)
            if smi:
                return smi, "cif_block", ""
        except Exception:
            pass

    # ── 3. 3D → SMILES from extracted LIG.pdb (last resort) ─────────────
    #
    # Why this is hard:
    #   Crystal PDB files carry no explicit bond-order information — bonds are
    #   inferred from inter-atomic distances.  Without hydrogen atoms present,
    #   valence clues are missing, so RDKit's sanitiser often assigns wrong
    #   bond orders (e.g. all-single aromatic rings, missed double bonds).
    #
    # Fix:
    #   A. Use OpenBabel to convert PDB → SDF *with hydrogen addition* (-h).
    #      Obabel adds H first, then perceives bond orders — much more
    #      reliable than reading the bare PDB.  Read the SDF with RDKit to
    #      get a clean canonical SMILES.
    #   B. RDKit fallback: read PDB with removeHs=False so H are present
    #      during sanitisation; try DetermineBonds (RDKit ≥ 2022.09) before
    #      falling back to standard SanitizeMol; only strip H for the final
    #      SMILES step.

    warn_3d = (
        "SMILES was derived from 3D coordinates (PDB has no bond-order data). "
        "Bond orders were inferred heuristically — verify the structure, "
        "especially for aromatic rings or unusual valences, before trusting "
        "docking scores."
    )

    if ligand_pdb_path and os.path.exists(ligand_pdb_path):
        _rdkit_six_patch()
        from rdkit import Chem

        # ── A. OpenBabel PDB → SDF (adds H, then perceives bonds) ────────
        # This is the most reliable path: obabel adds H before bond
        # perception, so valence information guides the assignment.
        try:
            import tempfile, os as _os
            with tempfile.NamedTemporaryFile(
                suffix=".sdf", delete=False
            ) as tf:
                ob_sdf = tf.name
            # -h  : add hydrogens
            # -p 7.4 : at physiological pH so protonation state is sensible
            rc, _ = run_cmd(
                ["obabel", ligand_pdb_path, "-O", ob_sdf, "-h", "-p", "7.4"]
            )
            if rc == 0 and _os.path.exists(ob_sdf) and _os.path.getsize(ob_sdf) > 10:
                supp = Chem.SDMolSupplier(ob_sdf, removeHs=False, sanitize=True)
                mols = [m for m in supp if m is not None]
                if not mols:
                    supp = Chem.SDMolSupplier(ob_sdf, removeHs=False, sanitize=False)
                    mols = [m for m in supp if m is not None]
                if mols:
                    mol = mols[0]
                    try:
                        mol_noh = Chem.RemoveHs(mol)
                        smi = Chem.MolToSmiles(mol_noh, isomericSmiles=True)
                    except Exception:
                        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                    if smi:
                        return smi, "3d_conversion", warn_3d
        except Exception:
            pass

        # ── B. RDKit with H kept during sanitisation ──────────────────────
        # Read with removeHs=False so H atoms provide valence clues.
        # Try the modern DetermineBonds API first (RDKit >= 2022.09),
        # which uses a distance+charge model and is far more accurate than
        # the legacy distance-only sanitiser.
        try:
            mol = Chem.MolFromPDBFile(
                ligand_pdb_path, removeHs=False, sanitize=False
            )
            if mol is not None:
                bonded = False

                # DetermineBonds (RDKit >= 2022.09)
                try:
                    from rdkit.Chem import DetermineBonds
                    DetermineBonds(mol, charge=0)
                    bonded = True
                except (ImportError, Exception):
                    pass

                if not bonded:
                    try:
                        Chem.SanitizeMol(mol)
                        bonded = True
                    except Exception:
                        # Partial sanitisation: keep going with what we have
                        try:
                            Chem.SanitizeMol(
                                mol,
                                Chem.SanitizeFlags.SANITIZE_ALL
                                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                            )
                            bonded = True
                        except Exception:
                            pass

                if bonded or mol.GetNumAtoms() > 0:
                    try:
                        mol_noh = Chem.RemoveHs(mol, sanitize=False)
                        Chem.SanitizeMol(mol_noh)
                        smi = Chem.MolToSmiles(mol_noh, isomericSmiles=True)
                    except Exception:
                        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                    if smi:
                        return smi, "3d_conversion", warn_3d
        except Exception:
            pass

        # ── C. OpenBabel direct → SMILES (last-ditch) ────────────────────
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(
                suffix=".smi", delete=False
            ) as tf:
                smi_tmp = tf.name
            rc, _ = run_cmd(
                ["obabel", ligand_pdb_path, "-O", smi_tmp, "-h", "--canonical"]
            )
            if rc == 0 and os.path.exists(smi_tmp):
                for line in open(smi_tmp):
                    pts = line.strip().split(None, 1)
                    if pts and pts[0]:
                        return pts[0], "3d_conversion", warn_3d
        except Exception:
            pass

    return "", "", (
        f"Could not obtain SMILES for co-crystal ligand "
        f"'{resname or cocrystal_ligand_id}'. "
        "All strategies failed: RCSB CCD, CIF block, "
        "OpenBabel PDB→SDF, RDKit DetermineBonds, OpenBabel direct."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  DOCKING  (unchanged from original)
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
    wdir      = Path(wdir)
    out_pdbqt = str(wdir / f"{out_name}_out.pdbqt")
    out_sdf   = str(wdir / f"{out_name}_out.sdf")


    rc, vlog = run_cmd(
        f'"{vina_path}" '
        f'--receptor "{os.path.abspath(receptor_pdbqt)}" '
        f'--ligand "{os.path.abspath(ligand_pdbqt)}" '
        f'--config "{os.path.abspath(config_txt)}" '
        f'--exhaustiveness {exhaustiveness} '
        f'--num_modes {n_modes} '
        f'--energy_range {energy_range} '
        f'--out "{os.path.abspath(out_pdbqt)}"',
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
#  BOND ORDER CORRECTION  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def _bo_template(smiles: str):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            raise ValueError(f"Cannot parse SMILES: {smiles!r}")
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(mol)
            Chem.SetAromaticity(mol)
        except Exception:
            pass
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        pass
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
                from rdkit.Chem import RWMol
                rw = RWMol(fixed)
                charged_idx = []
                for atom in rw.GetAtoms():
                    if atom.GetFormalCharge() != 0:
                        atom.SetNoImplicit(True)
                        charged_idx.append(atom.GetIdx())
                fixed_protected = rw.GetMol()
                fixed_h = Chem.AddHs(fixed_protected, addCoords=True)
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
#  SDF UTILITIES  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def load_mols_from_sdf(sdf_path: str, sanitize: bool = True) -> list:
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
    from rdkit import Chem
    with Chem.SDWriter(path) as w:
        w.write(mol)


def write_single_pose_pdb(mol, path: str) -> None:
    from rdkit import Chem
    Chem.MolToPDBFile(mol, path)


def convert_sdf_to_v2000(sdf_path: str) -> str:
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
#  STRUCTURAL ANALYSIS  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def get_interacting_residues(receptor_pdb: str, lig_mol, cutoff: float = 3.5) -> list:
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
                    crystal_pdb_path, sanitize=sanitize,
                    removeHs=removeHs, proximityBonding=proxBonding,
                )
                if cryst is not None and cryst.GetNumConformers() > 0:
                    if not sanitize:
                        try: Chem.SanitizeMol(cryst)
                        except Exception: pass
                    break
                cryst = None
            except Exception:
                cryst = None
        if cryst is None or cryst.GetNumConformers() == 0:
            return None
        pose = Chem.RemoveHs(pose_mol, sanitize=False)
        try: Chem.SanitizeMol(pose)
        except Exception: pass
        if pose.GetNumConformers() == 0:
            return None
        n_smaller = min(pose.GetNumAtoms(), cryst.GetNumAtoms())
        mcs = rdFMCS.FindMCS(
            [pose, cryst], timeout=10,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            completeRingsOnly=False, matchValences=False,
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
#  POSEVIEW REST API  (unchanged from original)
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
    import requests
    job    = requests.get(poll_url + job_id + "/", headers=_PP_HEADERS, timeout=15).json()
    status = str(job.get("status", "")).lower()
    polls  = 0
    while status in ("pending", "running", "processing", "queued", ""):
        if polls >= max_polls:
            raise RuntimeError(f"Job {job_id} still '{status}' after {max_polls * poll_interval} s")
        time.sleep(poll_interval)
        polls += 1
        job    = requests.get(poll_url + job_id + "/", headers=_PP_HEADERS, timeout=15).json()
        status = str(job.get("status", "")).lower()
    return job


def _prepare_pdb_for_poseview(receptor_pdb: str) -> str:
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
                    headers=_PP_HEADERS, timeout=30,
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
            last_error = str(e); continue
        except Exception as e:
            last_error = f"Polling error (attempt {attempt}): {e}"; continue
        if status in ("failed", "failure", "error"):
            last_error = f"PoseView rejected job (attempt {attempt}). Full response: {pv_job}"
            continue
        if status != "success":
            last_error = f"Unexpected status '{status}' (attempt {attempt}). Full response: {pv_job}"
            continue
        img_url = pv_job.get("image")
        if not img_url:
            last_error = f"Job succeeded but 'image' key missing. Keys: {list(pv_job.keys())}"
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
    return True, "Direct POST mode — no pre-upload needed"


def clear_poseview_cache():
    _PP_PROTEIN_CACHE.clear()


def call_poseview2_ref(pdb_code: str, ligand_id: str) -> tuple:
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
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                timeout=30,
            )
            data = r.json()
            if r.status_code not in (200, 202):
                last_error = f"Submission failed ({r.status_code}), attempt {attempt}: {data}"
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
                poll        = requests.get(location, headers={"Accept": "application/json"}, timeout=15).json()
                status_code = poll.get("status_code")
                if status_code == 200:
                    svg_url = poll.get("result_svg", "")
                    if not svg_url:
                        last_error = f"Job finished but result_svg is empty. Full response: {poll}"
                        job_failed = True; break
                    resp = requests.get(svg_url, timeout=20)
                    resp.raise_for_status()
                    return resp.content, None
                elif status_code == 202:
                    continue
                else:
                    last_error = f"Unexpected poll status {status_code} (attempt {attempt}). Full response: {poll}"
                    job_failed = True; break
            except Exception as e:
                last_error = f"Polling error (attempt {attempt}, poll {poll_i+1}): {e}"
                continue
        if not job_failed:
            last_error = f"Timed out after {_PV_POLL_ATTEMPTS * 2} s (attempt {attempt})"
    return None, last_error


def diagnose_poseview() -> dict:
    import requests
    result = {
        "server_reachable": False, "upload_ok": False,
        "poseview_ok": False, "status": "", "job_response": {},
        "image_url": "", "error": "", "log": [],
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
        r = requests.post(_PP_UPLOAD, data={"pdb_code": "4agn"}, timeout=30)
        r.raise_for_status()
        job_id = r.json().get("job_id")
        log.append(f"✓ Upload job submitted: {job_id}")
        job = _pp_poll(job_id, _PP_UPLOAD_JOBS, poll_interval=1, max_polls=30)
        log.append(f"✓ Upload job: {job.get('status')}")
        protein_id   = job["output_protein"]
        protein_json = requests.get(_PP_BASE + "molecule_handler/proteins/" + protein_id + "/", timeout=15).json()
        pdb_text     = protein_json["file_string"]
        ligand_id    = protein_json["ligand_set"][0]
        ligand_json  = requests.get(_PP_BASE + "molecule_handler/ligands/" + ligand_id + "/", timeout=15).json()
        sdf_text     = ligand_json["file_string"]
        log.append(f"✓ Got protein ({len(pdb_text)} chars) + ligand {ligand_json.get('name')} ({len(sdf_text)} chars)")
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
            result["error"] = f"PoseView returned '{result['status']}'. Full response: {pv_job}"
            log.append(f"✗ {result['error']}")
    except Exception as e:
        result["error"] = f"PoseView step failed: {e}"
        log.append(f"✗ PoseView failed: {e}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE UTILITIES  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def svg_to_png(svg_bytes: bytes):
    try:
        import cairosvg
        return cairosvg.svg2png(bytestring=svg_bytes, scale=2, background_color="white")
    except Exception:
        return None


def stamp_png(png_bytes: bytes, text: str) -> bytes:
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io as _io
        img  = Image.open(_io.BytesIO(png_bytes)).convert("RGBA")
        draw = ImageDraw.Draw(img)
        font = None
        _font_candidates = [
            # Linux
            ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 28),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 26),
            # macOS (system + Homebrew)
            ("/System/Library/Fonts/Supplemental/Arial.ttf", 28),
            ("/System/Library/Fonts/Helvetica.ttc", 28),
            ("/Library/Fonts/Arial.ttf", 28),
            ("/opt/homebrew/share/fonts/liberation-fonts/LiberationSans-Regular.ttf", 28),
            # Windows
            ("C:/Windows/Fonts/arial.ttf", 28),
            ("C:/Windows/Fonts/segoeui.ttf", 28),
            ("C:/Windows/Fonts/calibri.ttf", 28),
        ]
        for fp, sz in _font_candidates:
            try:
                font = ImageFont.truetype(fp, sz); break
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
            radius=pill_r, fill=(232, 232, 232, 230),
        )
        draw.text(
            (px + pill_w // 2, py_ + pill_h // 2),
            text, font=font, fill=(26, 26, 26, 255), anchor="mm",
        )
        buf = _io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return png_bytes


# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM 2D INTERACTION DIAGRAM  (unchanged from original)
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
}

_ATOM_CLR = {
    "C":"#1a1a1a", "N":"#1a5fa8", "O":"#cc2222",
    "S":"#c8a800", "P":"#e07000", "F":"#1a7a1a",
    "CL":"#1a7a1a", "BR":"#8b2500", "I":"#5c2d8a", "H":"#555555",
}
_AROM_DOT_CLR = "#1a7a1a"

_AROM_ATOMS = {"PHE","TYR","TRP","HIS","HEM","HEC","HEA","HEB","HDD","HDM"}

_AROM_ATOM_NAMES = {
    "CG","CD1","CD2","CE1","CE2","CZ",
    "ND1","NE2","CE3","CZ2","CZ3","CH2",
    "C1","C2","C3","C4","C5","C6","C7","C8",
    "C10","C11","C12","C13","C14","C15","C16","C17",
    "C19","C20",
    "CA","CB",
    "CAA","CAB","CAC","CAD",
    "CBA","CBB","CBC","CBD",
    "C2A","C3A","C4A",
    "C2B","C3B","C4B",
    "C2C","C3C","C4C",
    "C2D","C3D","C4D",
    "CHA","CHB","CHC","CHD",
}

_HYDR_BASE     = {"ALA","VAL","ILE","LEU","MET","PHE","TRP","PRO","GLY","TYR","HIS"}
_HYDR_EXTENDED = _HYDR_BASE | HEME_RESNAMES


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
                        best = max(_angle3(rp, rc[hj], lxyz[i]) for hj in hs)
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
                        best = max(_angle3(lxyz[i], lxyz[hi], rp) for hi in lig_hs)
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
    for j in range(nr):
        rn = r_rn[j]; ch = r_ch[j]; ri = int(r_ri[j]); el = r_el[j]; rp = rc[j]
        dists_j = np.linalg.norm(lxyz - rp, axis=1)
        md = float(dists_j.min()); mi = int(dists_j.argmin())
        if md > max(cutoff + 1.0, 5.6): continue
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
        _is_heme_fe = (r_rn[j] in HEME_RESNAMES and r_an[j].strip().upper() == "FE")
        if rn.strip().upper() in METAL_RESNAMES or el in METAL_RESNAMES or _is_heme_fe:
            if md < 2.8:
                results.append(dict(resname=rn,chain=ch,resid=ri,
                    itype="metal",distance=round(md,1),lig_atom_idx=mi,
                    prot_el=el,is_donor=False,ring_atom_indices=None))
    lr = _get_aromatic_ring_data(lig_mol_3d, conf)
    if lr:
        arom_mask = np.array([
            r_rn[j] in _AROM_ATOMS and r_an[j] in _AROM_ATOM_NAMES and r_el[j] == "C"
            for j in range(nr)
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
        cat_mask = np.array([r_rn[j] in {"LYS","ARG"} and r_el[j] == "N" for j in range(nr)])
        for j in np.where(cat_mask)[0]:
            rp = rc[j]
            for lc, _, ring_idxs in lr:
                d = float(np.linalg.norm(lc - rp))
                if d < 5.0:
                    results.append(dict(resname=r_rn[j],chain=r_ch[j],resid=int(r_ri[j]),
                        itype="cation_pi",distance=round(d,1),
                        lig_atom_idx=ring_idxs[0],prot_el="N",is_donor=True,
                        ring_atom_indices=ring_idxs)); break
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
        return [{**items[0], "bx": cx + R * _math.cos(a), "by": cy + R * _math.sin(a), "slot_angle": a}]
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
        result.append({**item, "bx": cx + R * _math.cos(a), "by": cy + R * _math.sin(a), "slot_angle": a})
    return result


def _place_residues_pca(interactions, svg_coords, mol3d, cx, cy, R=210):
    if not interactions: return []
    try:
        import numpy as _np_rc
        NODE_R = 24.55; GAP = 45.0; ATOM_R = 20.0
        lx, ly  = _rl_ligand_center(svg_coords)
        atom_xy = list(svg_coords.values())
        anchor_a = [_rl_anchor_angle(ix, svg_coords, lx, ly) for ix in interactions]
        order    = sorted(range(len(interactions)), key=lambda k: anchor_a[k])
        sorted_ix = [interactions[k] for k in order]
        rng       = _np_rc.random.default_rng(42)
        positions = _rl_place_radially(sorted_ix, svg_coords, lx, ly, atom_xy, ATOM_R, GAP, NODE_R, rng)
        positions = _rl_resolve_overlaps(positions, atom_xy, lx, ly, node_r=NODE_R, excl_r=NODE_R + ATOM_R)
        positions = _rl_reduce_crossings(sorted_ix, positions, svg_coords, lx, ly)
        result = []
        for k, (bx, by) in enumerate(positions):
            ang = _math.atan2(by - ly, bx - lx)
            result.append({**sorted_ix[k], "angle": ang, "bx": float(bx), "by": float(by), "slot_angle": ang})
        return result
    except Exception:
        return _place_residues_no_cross(interactions, svg_coords, cx, cy, R)


def _rl_ligand_center(svg_coords):
    pts = list(svg_coords.values())
    if not pts: return 400.0, 380.0
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
        vx = ax - cx; vy = ay - cy
        t_proj = vx * dx + vy * dy
        if t_proj < -atom_r: continue
        perp2 = vx * vx + vy * vy - t_proj * t_proj
        if perp2 >= r2: continue
        t_exit = t_proj + _math.sqrt(max(r2 - perp2, 0.0))
        if t_exit > t_max: t_max = t_exit
    return t_max if t_max > 1.0 else 20.0


def _rl_place_radially(sorted_interactions, svg_coords, cx, cy, atom_xy, atom_r, gap, node_r, rng):
    JITTER_T = 7.0; JITTER_R = 4.0
    n = len(sorted_interactions)
    jit_t = rng.uniform(-JITTER_T, JITTER_T, n)
    jit_r = rng.uniform(-JITTER_R, JITTER_R, n)
    positions = []
    for k, ix in enumerate(sorted_interactions):
        theta  = _rl_anchor_angle(ix, svg_coords, cx, cy)
        bdist  = _rl_ray_boundary(cx, cy, theta, atom_xy, atom_r)
        r_place = bdist + gap + node_r + jit_r[k]
        cos_t = _math.cos(theta); sin_t = _math.sin(theta)
        tx, ty = -sin_t, cos_t
        bx = cx + r_place * cos_t + jit_t[k] * tx
        by = cy + r_place * sin_t + jit_t[k] * ty
        positions.append((bx, by))
    return positions


def _rl_resolve_overlaps(positions, atom_xy, cx, cy, node_r=24.55, excl_r=46.0, max_iters=400):
    n = len(positions)
    if n <= 1: return list(positions)
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
        dx_acc = [0.0] * n; dy_acc = [0.0] * n
        any_overlap = False
        for i in range(n):
            for j in range(i + 1, n):
                ddx = bx[j] - bx[i]; ddy = by[j] - by[i]
                d   = _math.sqrt(ddx * ddx + ddy * ddy) + 1e-8
                if d < min_sep:
                    frac = (min_sep - d) * 0.55 / d
                    dx_acc[i] -= frac * ddx; dy_acc[i] -= frac * ddy
                    dx_acc[j] += frac * ddx; dy_acc[j] += frac * ddy
                    any_overlap = True
        for i in range(n):
            bx[i] += dx_acc[i]; by[i] += dy_acc[i]
        for i in range(n):
            for (ax, ay) in atom_xy:
                ddx = bx[i] - ax; ddy = by[i] - ay
                d   = _math.sqrt(ddx * ddx + ddy * ddy) + 1e-8
                if d < excl_r:
                    push = (excl_r - d) * 0.5 / d
                    if d < 0.5:
                        ddx = bx[i] - cx; ddy = by[i] - cy
                        d   = _math.sqrt(ddx * ddx + ddy * ddy) + 1e-8
                        push = excl_r * 0.5 / d
                    bx[i] += push * ddx; by[i] += push * ddy
        if not any_overlap: break
    return list(zip(bx, by))


def _rl_reduce_crossings(ix_list, bx_by, svg_coords, cx, cy):
    n = len(ix_list); bx_by = list(bx_by)
    def _anchor(ix):
        ai = ix.get("lig_atom_idx", 0)
        ax, ay = svg_coords.get(ai, (cx, cy))
        ri = ix.get("ring_atom_indices")
        if ri:
            rx, ry = _ring_centroid_2d(ri, svg_coords)
            if rx is not None: return (rx, ry)
        return (ax, ay)
    def _cross(a1, b1, a2, b2):
        def _side(O, A, B):
            return (B[0]-O[0])*(A[1]-O[1]) - (B[1]-O[1])*(A[0]-O[0])
        d1 = _side(a2, b2, a1); d2 = _side(a2, b2, b1)
        d3 = _side(a1, b1, a2); d4 = _side(a1, b1, b2)
        return (d1 * d2 < 0) and (d3 * d4 < 0)
    for _pass in range(n * 2):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if _cross(_anchor(ix_list[i]), bx_by[i], _anchor(ix_list[j]), bx_by[j]):
                    bx_by[i], bx_by[j] = bx_by[j], bx_by[i]; improved = True
        if not improved: break
    return bx_by


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
            parts.append(f'<line x1="{x1s:.2f}" y1="{y1s:.2f}" x2="{x2s:.2f}" y2="{y2s:.2f}" stroke="#1a1a1a" stroke-width="1.77" opacity="0.9"/>')
        elif bt==Chem.BondType.DOUBLE:
            dx,dy=x2s-x1s,y2s-y1s; L=_math.sqrt(dx*dx+dy*dy)+1e-9
            px,py=-dy/L*2.5,dx/L*2.5
            for sg in (1,-1):
                parts.append(f'<line x1="{x1s+px*sg:.2f}" y1="{y1s+py*sg:.2f}" x2="{x2s+px*sg:.2f}" y2="{y2s+py*sg:.2f}" stroke="#1a1a1a" stroke-width="2.44" opacity="0.9"/>')
        elif bt==Chem.BondType.TRIPLE:
            dx,dy=x2s-x1s,y2s-y1s; L=_math.sqrt(dx*dx+dy*dy)+1e-9
            px,py=-dy/L*3.8,dx/L*3.8
            for m in (-1,0,1):
                parts.append(f'<line x1="{x1s+px*m:.2f}" y1="{y1s+py*m:.2f}" x2="{x2s+px*m:.2f}" y2="{y2s+py*m:.2f}" stroke="#1a1a1a" stroke-width="2.0" opacity="0.9"/>')
        else:
            bd=bond.GetBondDir()
            if bd==Chem.BondDir.BEGINWEDGE:
                dx,dy=x2s-x1s,y2s-y1s; L=_math.sqrt(dx*dx+dy*dy)+1e-9
                px,py=-dy/L*3.0,dx/L*3.0
                parts.append(f'<polygon points="{x1s:.2f},{y1s:.2f} {x2s+px:.2f},{y2s+py:.2f} {x2s-px:.2f},{y2s-py:.2f}" fill="#1a1a1a" stroke="none"/>')
            elif bd==Chem.BondDir.BEGINDASH:
                dx,dy=x2s-x1s,y2s-y1s; L=_math.sqrt(dx*dx+dy*dy)+1e-9
                px,py=-dy/L,dx/L
                for step in range(1,6):
                    t=step/7; mx2=x1s+dx*t; my2=y1s+dy*t; w=t*5.0
                    parts.append(f'<line x1="{mx2-px*w:.2f}" y1="{my2-py*w:.2f}" x2="{mx2+px*w:.2f}" y2="{my2+py*w:.2f}" stroke="#1a1a1a" stroke-width="1.6"/>')
            else:
                parts.append(f'<line x1="{x1s:.2f}" y1="{y1s:.2f}" x2="{x2s:.2f}" y2="{y2s:.2f}" stroke="#1a1a1a" stroke-width="1.77" opacity="0.9"/>')
    for ring in arom_rings:
        rcoords=[svg_coords.get(i,(0,0)) for i in ring]
        rcx=sum(x for x,y in rcoords)/len(rcoords)
        rcy=sum(y for x,y in rcoords)/len(rcoords)
        avg=sum(_math.sqrt((x-rcx)**2+(y-rcy)**2) for x,y in rcoords)/len(rcoords)
        cr=avg*0.58
        parts.append(f'<circle cx="{rcx:.2f}" cy="{rcy:.2f}" r="{cr:.2f}" fill="none" stroke="#1a1a1a" stroke-width="1.77" stroke-dasharray="5.43 2.72" opacity="0.7"/>')
        parts.append(f'<circle cx="{rcx:.2f}" cy="{rcy:.2f}" r="5.43" fill="{_AROM_DOT_CLR}"/>')
    for i in range(mol2d.GetNumAtoms()):
        atom=mol2d.GetAtomWithIdx(i); sym=atom.GetSymbol()
        if sym=="C": continue
        ax,ay=svg_coords.get(i,(0,0))
        clr=_ATOM_CLR.get(sym.upper(),"#555")
        fs={"H":16}.get(sym,17.65)
        hw={"H":7,"N":9,"O":9,"S":11,"P":11,"F":8,"CL":16,"BR":16,"I":11}.get(sym.upper(),9)
        parts.append(f'<rect x="{ax-hw:.1f}" y="{ay-11:.1f}" width="{hw*2:.0f}" height="22" fill="white" stroke="none"/>')
        parts.append(f'<text x="{ax:.2f}" y="{ay:.2f}" text-anchor="middle" dominant-baseline="central" font-family="Arial,sans-serif" font-size="{fs}" font-weight="700" fill="{clr}">{sym}</text>')
        fc=atom.GetFormalCharge()
        if fc!=0:
            fcs="+" if fc==1 else "\u2212" if fc==-1 else f"{fc:+d}"
            parts.append(f'<text x="{ax+hw:.1f}" y="{ay-hw+2:.1f}" font-family="Arial,sans-serif" font-size="10" fill="{clr}">{fcs}</text>')
    return "".join(parts)


def _render_diagram_svg(mol2d, svg_coords, placements, title, W, H):
    parts=[]
    parts.append(f'<svg width="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">')
    parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    if title:
        esc=title.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        tw=len(esc)*14.5+48; tw=max(tw,240); tw=min(tw,W-40)
        px=(W-tw)/2; ph=46; pr=ph/2
        parts.append(f'<rect x="{px:.1f}" y="14" width="{tw:.0f}" height="{ph}" rx="{pr:.1f}" ry="{pr:.1f}" fill="#f2f2f2" stroke="none"/>')
        parts.append(f'<text x="{W/2:.1f}" y="37" text-anchor="middle" dominant-baseline="central" font-family="Arial,sans-serif" font-size="24.93" font-weight="700" fill="#1a1a1a">{esc}</text>')
    for p in placements:
        itype=p["itype"]; bx,by=p["bx"],p["by"]
        cbx=max(50,min(bx,W-50)); cby=max(70,min(by,H-65))
        bg=_RES_CIRCLE.get(itype,dict(fill="#cccccc",opacity=0.2))
        parts.append(f'<circle cx="{cbx:.1f}" cy="{cby:.1f}" r="24.55" fill="{bg["fill"]}" opacity="{bg["opacity"]}"/>')
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
            parts.append(f'<line x1="{lx:.2f}" y1="{ly:.2f}" x2="{cbx:.2f}" y2="{cby:.2f}" stroke="{clr}" stroke-width="1.6"{dash_attr} opacity="0.85"/>')
        elif itype in ("hbond","hbond_to_halogen"):
            da=dash_attr if itype=="hbond" else dash_attr_hbx
            parts.append(f'<line x1="{lx:.2f}" y1="{ly:.2f}" x2="{cbx:.2f}" y2="{cby:.2f}" stroke="{clr}" stroke-width="1.6"{da} opacity="0.85"/>')
            if p.get("distance") is not None:
                t_along=0.40; bx_l=lx+(cbx-lx)*t_along; by_l=ly+(cby-ly)*t_along
                dx_l=cbx-lx; dy_l=cby-ly; _len=_math.sqrt(dx_l*dx_l+dy_l*dy_l)+1e-9
                px_l=-dy_l/_len*14; py_l=dx_l/_len*14; mx2=bx_l+px_l*14; my2=by_l+py_l*14
                ds=f"{p['distance']}\u00c5"; tw2=len(ds)*7+8
                parts.append(f'<rect x="{mx2-tw2/2:.1f}" y="{my2-9:.1f}" width="{tw2:.0f}" height="17" rx="4" fill="white" stroke="{clr}" stroke-width="0.5"/>')
                parts.append(f'<text x="{mx2:.1f}" y="{my2:.1f}" text-anchor="middle" dominant-baseline="central" font-family="Arial,sans-serif" font-size="14" font-weight="700" fill="{clr}">{ds}</text>')
        else:
            da={"ionic":"6 2 2 2","metal":"3 2","halogen":"5 2"}.get(itype,"5 3")
            parts.append(f'<line x1="{lx:.2f}" y1="{ly:.2f}" x2="{cbx:.2f}" y2="{cby:.2f}" stroke="{clr}" stroke-width="1.8" stroke-dasharray="{da}" opacity="0.85"/>')
            if p.get("distance") is not None:
                t_along=0.40; bx_l=lx+(cbx-lx)*t_along; by_l=ly+(cby-ly)*t_along
                dx_l=cbx-lx; dy_l=cby-ly; _len=_math.sqrt(dx_l*dx_l+dy_l*dy_l)+1e-9
                px_l=-dy_l/_len*14; py_l=dx_l/_len*14; mx2=bx_l+px_l*14; my2=by_l+py_l*14
                ds=f"{p['distance']}\u00c5"; tw2=len(ds)*7+8
                parts.append(f'<rect x="{mx2-tw2/2:.1f}" y="{my2-9:.1f}" width="{tw2:.0f}" height="17" rx="4" fill="white" stroke="{clr}" stroke-width="0.5"/>')
                parts.append(f'<text x="{mx2:.1f}" y="{my2:.1f}" text-anchor="middle" dominant-baseline="central" font-family="Arial,sans-serif" font-size="14" font-weight="700" fill="{clr}">{ds}</text>')
    parts.append(_render_ligand_svg(mol2d, svg_coords))
    for p in placements:
        itype=p["itype"]; bx,by=p["bx"],p["by"]
        cbx=max(50,min(bx,W-50)); cby=max(70,min(by,H-65))
        rn=p["resname"]; ri=p["resid"]; ch=p.get("chain","")
        _NO_NUM = HEME_RESNAMES | METAL_RESNAMES
        lbl = rn.upper() if rn.upper() in _NO_NUM else f"{rn.upper()} {ri}{ch}"
        lbl_clr=_LBL_CLR.get(itype,"#333")
        parts.append(f'<text x="{cbx:.1f}" y="{cby:.1f}" text-anchor="middle" dominant-baseline="central" font-family="Arial,sans-serif" font-size="14.29" font-weight="700" fill="{lbl_clr}">{lbl}</text>')
    _LEG_ORDER=["hydrophobic","hbond","pi_pi","cation_pi","hbond_to_halogen","ionic","metal","halogen"]
    _LEG_LABEL={"hydrophobic":"Hydrophobic","hbond":"Hydrogen bond","hbond_to_halogen":"H\u00b7\u00b7\u00b7Halogen","pi_pi":"\u03c0-\u03c0 stacking","cation_pi":"Cation-\u03c0","ionic":"Ionic","metal":"Metal","halogen":"Halogen bond"}
    active=[t for t in _LEG_ORDER if any(p["itype"]==t for p in placements)]
    if active:
        ly0=H-52
        def _entry_w(t):
            txt=_LEG_LABEL.get(t,t)
            return 9.54*2+(4+28 if t!="hydrophobic" else 0)+6+len(txt)*9.5+20
        total_w=sum(_entry_w(t) for t in active); total_w=min(total_w,W-40)
        lx0=(W-total_w)/2
        parts.append(f'<rect x="{lx0-8:.0f}" y="{ly0-5}" width="{total_w+16:.0f}" height="44" fill="white" stroke="#e0e0e0" stroke-width="0.8" rx="6"/>')
        for k,it in enumerate(active):
            bg=_RES_CIRCLE.get(it,dict(fill="#ccc",opacity=0.2))
            clr=_LINE_CLR.get(it,bg["fill"])
            lbl=_LEG_LABEL.get(it,it)
            text_w=len(lbl)*9.5+6
            glyph_w=9.54*2+(4+28 if it!="hydrophobic" else 0)
            entry_w=glyph_w+6+text_w
            circ_cx=lx0+sum(
                (9.54*2+(4+28 if active[kk]!="hydrophobic" else 0)+6+len(_LEG_LABEL.get(active[kk],active[kk]))*9.5+6+20)
                for kk in range(k)
            )+9.54
            parts.append(f'<circle cx="{circ_cx:.1f}" cy="{ly0+10}" r="9.54" fill="{bg["fill"]}" opacity="{bg["opacity"]}"/>')
            line_x1=circ_cx+9.54+4; line_x2=line_x1+28
            if it!="hydrophobic":
                parts.append(f'<line x1="{line_x1:.1f}" y1="{ly0+10}" x2="{line_x2:.1f}" y2="{ly0+10}" stroke="{clr}" stroke-width="2" stroke-dasharray="5 3" opacity="0.85"/>')
            text_x=(line_x2 if it!="hydrophobic" else circ_cx+9.54)+6
            parts.append(f'<text x="{text_x:.1f}" y="{ly0+10}" text-anchor="start" dominant-baseline="central" font-family="Arial,sans-serif" font-size="16" font-weight="700" fill="#555">{lbl}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def _build_diagram_data(receptor_pdb, pose_sdf, smiles, cutoff, max_residues, size=(800,759)):
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

    def _parse_smi_robust(smi):
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

    mol2d = None
    if smiles and smiles.strip():
        mol2d = _parse_smi_robust(smiles.strip())
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
        def _strip_charges(mol):
            rw = Chem.RWMol(mol)
            for a in rw.GetAtoms(): a.SetFormalCharge(0)
            Chem.SanitizeMol(rw)
            return rw.GetMol()
        m3_neutral    = _strip_charges(m3)
        mol2d_neutral = _strip_charges(mol2d)
        mt = m3_neutral.GetSubstructMatch(mol2d_neutral)
        if len(mt) == mol2d.GetNumAtoms():
            for i2, i3 in enumerate(mt): m3to2d[i3] = i2
    except Exception:
        try:
            mt = m3.GetSubstructMatch(mol2d)
            if len(mt) == mol2d.GetNumAtoms():
                for i2, i3 in enumerate(mt): m3to2d[i3] = i2
        except Exception:
            pass
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


def draw_interaction_diagram(
    receptor_pdb: str,
    pose_sdf: str,
    smiles: str,
    title: str = "",
    cutoff: float = 4.5,
    size: tuple = (800, 759),
    max_residues: int = 14,
) -> bytes:
    try:
        mol2d, sc, pl, W, H = _build_diagram_data(receptor_pdb, pose_sdf, smiles, cutoff, max_residues, size)
    except Exception as e:
        W, H = size
        return (f'<svg viewBox="0 0 {W} 80" xmlns="http://www.w3.org/2000/svg">'
                f'<rect width="{W}" height="80" fill="white"/>'
                f'<text x="{W//2}" y="44" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#cc2222">'
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
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDepictor
    import json as _json
    RDLogger.DisableLog("rdApp.*")
    W, H = size
    try:
        mol2d, sc, pl, W, H = _build_diagram_data(receptor_pdb, pose_sdf, smiles, cutoff, max_residues, size)
    except Exception:
        return None
    lig_svg = _render_ligand_svg(mol2d, sc)
    sc_serial = {str(k): [round(v[0], 2), round(v[1], 2)] for k, v in sc.items()}
    pl_serial = []
    for p in pl:
        lx, ly = sc.get(p.get("lig_atom_idx", 0), (W//2, H//2))
        if p.get("ring_atom_indices"):
            rx, ry = _ring_centroid_2d(p["ring_atom_indices"], sc)
            if rx is not None: lx, ly = rx, ry
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


def draw_interactions_rdkit_classic(
    lig_mol,
    receptor_pdb: str,
    smiles: str,
    title: str = "",
    cutoff: float = 3.5,
    size: tuple = (650, 620),
    max_residues: int = 10,
) -> bytes:
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
        def _strip_charges_rdk(mol):
            rw = Chem.RWMol(mol)
            for a in rw.GetAtoms(): a.SetFormalCharge(0)
            Chem.SanitizeMol(rw)
            return rw.GetMol()
        m3n   = _strip_charges_rdk(mol3d_noH)
        m2d_n = _strip_charges_rdk(mol2d)
        match = m3n.GetSubstructMatch(m2d_n)
        if len(match) == mol2d.GetNumAtoms():
            for i2d, i3d in enumerate(match):
                idx3d_to_2d[i3d] = i2d
    except Exception:
        try:
            match = mol3d_noH.GetSubstructMatch(mol2d)
            if len(match) == mol2d.GetNumAtoms():
                for i2d, i3d in enumerate(match):
                    idx3d_to_2d[i3d] = i2d
        except Exception:
            pass
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
        if itype in ("hbond", "hbond_to_halogen"): color = _C_HB
        elif itype == "hydrophobic":                color = _C_HP
        else:                                       color = _C_OT
        rn = ix["resname"]; ri = ix["resid"]; ch = ix.get("chain","")
        _NO_NUM = HEME_RESNAMES | METAL_RESNAMES
        lbl = rn.upper() if rn.upper() in _NO_NUM else f"{rn}{ri}{ch}"
        res_atom = Chem.Atom(0)
        res_atom.SetProp("atomLabel", lbl)
        aid = lig_ext.AddAtom(res_atom)
        pts.append(aid); clrs[aid] = color
        lig_ext.AddBond(aid, ai, Chem.BondType.ZERO)
    rdDepictor.Compute2DCoords(lig_ext)
    d2d = Draw.MolDraw2DSVG(W, H)
    opts = d2d.drawOptions()
    opts.circleAtoms = True; opts.fillHighlights = True
    opts.continuousHighlight = False; opts.highlightRadius = 0.5
    opts.addAtomIndices = False; opts.padding = 0.15
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


def draw_interaction_diagram_interactive(
    receptor_pdb: str,
    pose_sdf: str,
    smiles: str,
    title: str = "",
    cutoff: float = 4.5,
    size: tuple = (800, 759),
    max_residues: int = 14,
) -> str:
    import json
    try:
        mol2d, sc, pl, W, H = _build_diagram_data(receptor_pdb, pose_sdf, smiles, cutoff, max_residues, size)
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
            "dist":    str(p["distance"]) + "\u00c5" if p.get("distance") else "",
            "lx":      round(lx, 2), "ly": round(ly, 2),
            "bx":      round(p["bx"], 2), "by": round(p["by"], 2),
        })
    return json.dumps({"placements": residues_js, "W": W, "H": H,
                       "ligand_svg": lig_svg, "title": title})
