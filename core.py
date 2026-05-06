#!/usr/bin/env python3
"""
core.py  —  Pure computation layer for Anyone Can Dock.
No Streamlit imports. All functions return plain dicts / tuples.

Optimisation changes (v2):
  * Lazy compile _IONIZABLE_SITES_COMPILED / _CHEM_RULES  (not at import)
  * lru_cache on _cached_parse_pdb  (receptor parsed once per wdir path)
  * _admetlab3_predict  replaces ADMET-AI (~500 MB model) with REST API
  * Clearer Vina error messages
  * clear_poseview_cache also clears _cached_parse_pdb
"""

import os, subprocess, sys, tempfile, time, re as _re, functools, math as _math
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

METAL_RESNAMES = {
    "MG","ZN","CA","MN","FE","CU","CO","NI","CD","HG","NA","K","HO",
    "LA","CE","PR","ND","PM","SM","EU","GD","TB","DY","ER","TM","YB","LU",
}
METAL_CHARGES = {
    "MG":2.0,"ZN":2.0,"CA":2.0,"MN":2.0,"FE":3.0,"CU":2.0,"CO":2.0,
    "NI":2.0,"CD":2.0,"HG":2.0,"HO":3.0,"LA":3.0,"CE":3.0,"PR":3.0,
    "ND":3.0,"PM":3.0,"SM":3.0,"EU":3.0,"GD":3.0,"TB":3.0,"DY":3.0,
    "ER":3.0,"TM":3.0,"YB":3.0,"LU":3.0,"NA":1.0,"K":1.0,
}
EXCLUDE_IONS = set(
    "HOH,WAT,DOD,SOL,NA,CL,K,CA,MG,ZN,MN,FE,CU,CO,NI,CD,HG,HO,"
    "LA,CE,PR,ND,PM,SM,EU,GD,TB,DY,ER,TM,YB,LU".split(",")
)
GLYCAN_NAMES = {
    "NAG","BMA","MAN","FUC","GAL","GLC","SIA","NGA",
    "FUL","GLA","BGC","A2G","LAT","MAL","CEL","SUC",
    "TRE","GCS","NDG","NGC",
}
COFACTOR_NAMES = {
    "ATP","ADP","AMP","GTP","GDP","GMP",
    "NAD","NAP","NDP","FAD","FMN",
    "HEM","HEC","HEA","GOL","PEG","EDO","MPD","PGE","PG4",
    "SO4","PO4","SUL","PHO","IHP","TTP","CTP","UTP",
    "COA","SAM","SAH","EPE","MES","TRS","ACT","ACY",
    "HO","LA","CE","PR","ND","PM","SM","EU","GD","TB","DY","ER","TM","YB","LU",
}
HEME_RESNAMES = {"HEM","HEC","HEA","HEB","HDD","HDM"}

_PV_MAX_RETRIES = 3
_PV_RETRY_DELAY = 10
_PV_POLL_ATTEMPTS = 60

# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def run_cmd(cmd, cwd=None):
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str),
        capture_output=True, text=True, cwd=cwd,
    )
    return r.returncode, (r.stdout + r.stderr).strip()

def _rdkit_six_patch():
    try:
        from rdkit import six
    except ImportError:
        from io import StringIO as _SIO
        from types import ModuleType as _MT
        import rdkit as _rdkit
        _m = _MT("six"); _m.StringIO = _SIO; _m.PY3 = True
        _rdkit.six = _m; sys.modules["rdkit.six"] = _m

def _strip_h_from_pdb(pdb_path, out_path):
    import shutil
    try:
        lines = []
        with open(pdb_path) as f:
            for line in f:
                rec = line[:6].strip()
                if rec in ("ATOM","HETATM"):
                    atom_name = line[12:16].strip()
                    element   = line[76:78].strip() if len(line) > 76 else ""
                    if element.upper() == "H" or atom_name.startswith("H"):
                        continue
                lines.append(line)
        with open(out_path,"w") as f: f.writelines(lines)
        return True
    except Exception:
        shutil.copy(pdb_path, out_path); return False

def is_cif_file(filepath):
    ext = Path(filepath).suffix.lower()
    if ext in (".cif",".mmcif"): return True
    try:
        with open(filepath) as f: first = f.read(512)
        if first.strip().startswith("data_"): return True
    except Exception: pass
    return False

def convert_cif_to_pdb(cif_path, pdb_out_path):
    log = []
    try:
        import gemmi
        doc   = gemmi.cif.read(cif_path)
        block = doc.sole_block()
        st    = gemmi.make_structure_from_block(block)
        st.setup_entities(); st.assign_label_seq_id()
        pdb_str = st.make_pdb_headers() + st.make_pdb_string()
        with open(pdb_out_path,"w") as f: f.write(pdb_str)
        if os.path.exists(pdb_out_path) and os.path.getsize(pdb_out_path) > 100:
            log.append("OK gemmi"); return {"success":True,"pdb_path":pdb_out_path,"log":log}
        log.append("gemmi empty")
    except Exception as e: log.append(f"gemmi: {e}")
    try:
        rc,out = run_cmd(f'obabel "{cif_path}" -O "{pdb_out_path}"')
        if os.path.exists(pdb_out_path) and os.path.getsize(pdb_out_path) > 100:
            log.append("OK obabel"); return {"success":True,"pdb_path":pdb_out_path,"log":log}
    except Exception as e: log.append(f"obabel: {e}")
    return {"success":False,"pdb_path":pdb_out_path,"log":log,"error":"All CIF->PDB failed."}

def check_obabel():
    import shutil
    if shutil.which("obabel") is None:
        return False,"obabel not found"
    _,out = run_cmd("obabel --version")
    return True,(out.splitlines()[0] if out else "ok")

# ══════════════════════════════════════════════════════════════════════════════
# VINA BINARY
# ══════════════════════════════════════════════════════════════════════════════

def get_vina_binary(path=""):
    import platform, requests as _rq
    system  = platform.system().lower()
    machine = platform.machine().lower()
    _BASE   = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/"
    if system == "linux":
        _FNAME = "vina_1.2.7_linux_x86_64"
    elif system == "darwin":
        _FNAME = "vina_1.2.7_mac_aarch64" if machine in ("arm64","aarch64") else "vina_1.2.7_mac_x86_64"
    elif system == "windows":
        _FNAME = "vina_1.2.7_windows_x86_64.exe"
    else:
        return None, f"Unsupported platform: {system}/{machine}"
    _URL = _BASE + _FNAME
    if not path: path = os.path.join(tempfile.gettempdir(), _FNAME)
    def _dl(url, dest):
        r = _rq.get(url, stream=True, timeout=120, allow_redirects=True); r.raise_for_status()
        with open(dest,"wb") as f:
            for chunk in r.iter_content(chunk_size=1<<20): f.write(chunk)
    if not os.path.exists(path) or os.path.getsize(path) < 100_000:
        try: _dl(_URL, path)
        except Exception as e1:
            if system == "darwin" and machine in ("arm64","aarch64"):
                _F2 = "vina_1.2.7_mac_x86_64"; p2 = os.path.join(tempfile.gettempdir(),_F2)
                try: _dl(_BASE+_F2, p2); path = p2
                except Exception as e2: return None, f"Download failed: {e1} / {e2}"
            else: return None, f"Download failed: {e1}"
    if system != "windows": os.chmod(path, 0o755)
    return path, f"ok ({system}/{machine})"

# ══════════════════════════════════════════════════════════════════════════════
# RECEPTOR PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

_MIN_LIG_ATOMS = 4

def _collect_removable_ligands(atoms):
    from prody import calcCenter
    excl = EXCLUDE_IONS | GLYCAN_NAMES | COFACTOR_NAMES | HEME_RESNAMES | METAL_RESNAMES
    het  = atoms.select("hetatm and not water")
    _BB  = {"N","CA","C","O"}
    if het is None: return []
    results = []
    for r in het.getHierView().iterResidues():
        rn = (r.getResname() or "").strip().upper()
        if rn in excl or r.numAtoms() <= _MIN_LIG_ATOMS: continue
        if _BB.issubset(set(r.getNames())): continue
        ch = r.getChid(); ri = r.getResnum()
        sel = (f"resname {rn} and resid {ri} and chain {ch}"
               if ch and ch.strip() else f"resname {rn} and resid {ri}")
        la = atoms.select(sel)
        if la is None or la.numAtoms() == 0: continue
        cx_,cy_,cz_ = (float(v) for v in calcCenter(la))
        results.append({"resname":rn,"chain":ch,"resid":ri,"sel_str":sel,
            "ligand_id":f"{rn}_{ch}_{ri}","n_atoms":la.numAtoms(),"atoms":la,
            "cx":cx_,"cy":cy_,"cz":cz_})
    results.sort(key=lambda d:(-d["n_atoms"],d["chain"]!="A"))
    return results

def detect_cocrystal_ligand(raw_pdb):
    from prody import parsePDB
    atoms = parsePDB(raw_pdb)
    if atoms is None: return {"found":False}
    cands = _collect_removable_ligands(atoms)
    if not cands: return {"found":False}
    c = cands[0]
    return {"found":True,"resname":c["resname"],"chain":c["chain"],"resid":c["resid"],
            "sel_str":c["sel_str"],"ligand_id":c["ligand_id"],"cx":c["cx"],"cy":c["cy"],
            "cz":c["cz"],"n_atoms":c["n_atoms"],"atoms":c["atoms"]}

def scan_ligands(raw_pdb):
    try:
        from prody import parsePDB, confProDy
        confProDy(verbosity="none")
        if is_cif_file(raw_pdb):
            import tempfile as _tf; _tmp = _tf.mktemp(suffix=".pdb")
            res = convert_cif_to_pdb(raw_pdb, _tmp)
            if res["success"]: raw_pdb = _tmp
        atoms = parsePDB(raw_pdb)
        if atoms is None: return []
        return [{"resname":d["resname"],"chain":d["chain"],"resid":d["resid"],"n_atoms":d["n_atoms"]}
                for d in _collect_removable_ligands(atoms)]
    except Exception: return []

def strip_and_convert_receptor(rec_raw, wdir):
    wdir = Path(wdir); log = []
    rec_fh    = str(wdir/"rec.pdb")
    rec_pdbqt = str(wdir/"rec.pdbqt")
    try:
        metal_lines=[]; clean_lines=[]
        with open(rec_raw) as f:
            for line in f:
                field = line[:6].strip()
                if field in ("ATOM","HETATM") and line[17:20].strip().upper() in METAL_RESNAMES:
                    metal_lines.append(line)
                else: clean_lines.append(line)
        rec_nometal = str(wdir/"receptor_atoms_nometal.pdb")
        with open(rec_nometal,"w") as f: f.writelines(clean_lines)
        if metal_lines:
            names = ", ".join(sorted({l[17:20].strip() for l in metal_lines}))
            log.append(f"Stripped {len(metal_lines)} metal atom(s): {names}")
        rc1,out1 = run_cmd(f'obabel "{rec_nometal}" -O "{rec_fh}" -h')
        if not os.path.exists(rec_fh) or os.path.getsize(rec_fh) < 100:
            raise ValueError(f"OpenBabel H-addition produced empty file (exit {rc1}): {out1[:300]}")
        log.append("Hydrogens added")
        rc2,out2 = run_cmd(f'obabel "{rec_fh}" -O "{rec_pdbqt}" -xr --partialcharge gasteiger')
        if not os.path.exists(rec_pdbqt) or os.path.getsize(rec_pdbqt) < 100:
            raise ValueError(f"PDBQT conversion empty (exit {rc2}): {out2[:300]}")
        log.append("PDBQT conversion complete")
        if metal_lines and os.path.exists(rec_fh):
            try:
                rec_lines = [l for l in open(rec_fh).readlines() if l.strip()!="END"]
                rec_lines.extend(metal_lines); rec_lines.append("END\n")
                with open(rec_fh,"w") as f: f.writelines(rec_lines)
                log.append(f"Re-added {len(metal_lines)} ion/metal atom(s) to receptor.pdb")
            except Exception as e: log.append(f"Could not re-add ions: {e}")
        if metal_lines:
            pdbqt_lines=[l for l in open(rec_pdbqt).readlines() if l.strip()!="END"]
            injected=0; skip_exotic=0
            _no_reinject={"HO","LA","CE","PR","ND","PM","SM","EU","GD","TB","DY","ER","TM","YB","LU"}
            for ml in metal_lines:
                try:
                    rn=ml[17:20].strip().upper()
                    if rn in _no_reinject: skip_exotic+=1; continue
                    serial=int(ml[6:11]); name=ml[12:16].strip()
                    chain=ml[21] if len(ml)>21 else "A"; resid=int(ml[22:26])
                    x=float(ml[30:38]); y=float(ml[38:46]); z=float(ml[46:54])
                    charge=METAL_CHARGES.get(rn,0.0); atype=rn.capitalize()
                    pdbqt_lines.append(
                        f"HETATM{serial:5d} {name:<4s} {rn:<3s} "
                        f"{chain}{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                        f"  1.00  0.00    {charge:+.3f} {atype}\n")
                    injected+=1
                except Exception as e: log.append(f"Could not re-inject metal: {e}")
            pdbqt_lines.append("END\n")
            with open(rec_pdbqt,"w") as f: f.writelines(pdbqt_lines)
            if injected: log.append(f"Re-injected {injected} metal atom(s) into PDBQT")
        log.append("Receptor PDBQT ready")
        return {"success":True,"rec_fh":rec_fh,"rec_pdbqt":rec_pdbqt,"log":log}
    except Exception as e:
        log.append(f"ERROR: {e}")
        return {"success":False,"error":str(e),"log":log}

def write_box_pdb(filename, cx, cy, cz, sx, sy, sz):
    hx,hy,hz = sx/2,sy/2,sz/2
    corners = [(cx+dx,cy+dy,cz+dz) for dx in (-hx,hx) for dy in (-hy,hy) for dz in (-hz,hz)]
    with open(filename,"w") as f:
        for i,(x,y,z) in enumerate(corners,1):
            f.write(f"HETATM{i:5d}  C   BOX A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        f.write("CONECT    1    2    3    5\nCONECT    2    1    4    6\n"
                "CONECT    3    1    4    7\nCONECT    4    2    3    8\n"
                "CONECT    5    1    6    7\nCONECT    6    2    5    8\n"
                "CONECT    7    3    5    8\nCONECT    8    4    6    7\n")

def write_vina_config(filename, cx, cy, cz, sx, sy, sz):
    with open(filename,"w") as f:
        f.write(f"center_x = {cx:.4f}\ncenter_y = {cy:.4f}\ncenter_z = {cz:.4f}\n"
                f"size_x = {sx}\nsize_y = {sy}\nsize_z = {sz}\n")

# ── CACHED parsePDB ─────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=4)
def _cached_parse_pdb(receptor_pdb: str):
    """Parse receptor PDB once per path — cached across slider calls."""
    from prody import parsePDB
    return parsePDB(receptor_pdb)

def prepare_receptor(raw_pdb, wdir, center_mode="auto",
                     manual_xyz=(0.0,0.0,0.0), prody_sel="",
                     box_size=(16,16,16), preferred_ligand=""):
    from prody import parsePDB, calcCenter, writePDB
    wdir=Path(wdir); log=[]; sx,sy,sz=box_size
    try:
        if is_cif_file(raw_pdb):
            log.append("Detected mmCIF — converting to PDB...")
            conv_pdb=str(wdir/"converted_from_cif.pdb")
            cif_res=convert_cif_to_pdb(raw_pdb,conv_pdb)
            log.extend(cif_res["log"])
            if not cif_res["success"]: raise ValueError(f"CIF->PDB failed: {cif_res.get('error')}")
            raw_pdb=conv_pdb
        atoms=parsePDB(raw_pdb)
        if atoms is None:
            raise ValueError("ไม่สามารถอ่านไฟล์ PDB ได้ — ตรวจสอบว่า PDB ID ถูกต้อง หรือลองเปลี่ยน format เป็น CIF")
        log.append(f"Parsed {atoms.numAtoms()} atoms")
        ligand_pdb_path=""; cocrystal_ligand_id=""; cx=cy=cz=0.0
        _all_ligs=_collect_removable_ligands(atoms); _primary=None
        if _all_ligs:
            if preferred_ligand:
                _pref=preferred_ligand.strip().upper()
                _primary=next((d for d in _all_ligs if d["resname"].upper()==_pref),None)
                if _primary is None:
                    log.append(f"Preferred ligand '{preferred_ligand}' not found — using {_all_ligs[0]['resname']}")
            if _primary is None: _primary=_all_ligs[0]
        if _primary is not None:
            rn=_primary["resname"]; ch=_primary["chain"]; ri=_primary["resid"]
            cocrystal_ligand_id=_primary["ligand_id"]
            ligand_pdb_path=str(wdir/"LIG.pdb")
            writePDB(ligand_pdb_path,_primary["atoms"])
            _n_extra=len(_all_ligs)-1
            log.append(f"Co-crystal ligand: {rn} chain '{ch}' resnum {ri} ({_primary['n_atoms']} atoms)"
                       +(f"  +{_n_extra} additional ligand(s) removed" if _n_extra else ""))
        if center_mode=="auto":
            if _primary is not None:
                cx,cy,cz=_primary["cx"],_primary["cy"],_primary["cz"]
                log.append(f"Auto center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
            else: log.append("No co-crystal ligand — grid at protein centroid")
        elif center_mode=="manual":
            cx,cy,cz=(float(v) for v in manual_xyz)
            log.append(f"Manual center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
        elif center_mode=="selection":
            if not prody_sel.strip():
                raise ValueError("ProDy selection string is empty")
            ref_atoms=atoms.select(prody_sel.strip())
            if ref_atoms is None or ref_atoms.numAtoms()==0:
                raise ValueError(f"ProDy selection '{prody_sel}' matched 0 atoms")
            cx,cy,cz=(float(v) for v in calcCenter(ref_atoms))
            log.append(f"Selection '{prody_sel}' -> center ({cx:.3f},{cy:.3f},{cz:.3f})")
        if _all_ligs:
            _excl=" or ".join(f"({d['sel_str']})" for d in _all_ligs)
            sel_str=f"not ({_excl}) and not water"
        else: sel_str="not water"
        rec_sel=atoms.select(sel_str)
        if rec_sel is None or rec_sel.numAtoms()==0:
            raise ValueError("Receptor selection returned no atoms")
        rec_raw_path=str(wdir/"receptor_atoms.pdb")
        writePDB(rec_raw_path,rec_sel); log.append(f"Receptor: {rec_sel.numAtoms()} atoms")
        # fix blank chain
        try:
            _rec_lines=open(rec_raw_path).readlines()
            _coord_lines=[l for l in _rec_lines if l[:6].strip() in ("ATOM","HETATM")]
            _all_blank=all((l[21]==" " if len(l)>21 else True) for l in _coord_lines)
            if _all_blank and _coord_lines:
                _fixed=[l[:21]+"A"+l[22:] if l[:6].strip() in ("ATOM","HETATM") and len(l)>21 else l
                        for l in _rec_lines]
                with open(rec_raw_path,"w") as _f: _f.writelines(_fixed)
                log.append("Assigned chain A to blank-chain atoms")
        except Exception as _ce: log.append(f"Chain-fix skipped: {_ce}")
        conv=strip_and_convert_receptor(rec_raw_path,wdir)
        log.extend(conv["log"])
        if not conv["success"]: raise ValueError(conv["error"])
        box_pdb=str(wdir/"rec.box.pdb"); cfg_path=str(wdir/"rec.box.txt")
        write_box_pdb(box_pdb,cx,cy,cz,sx,sy,sz)
        write_vina_config(cfg_path,cx,cy,cz,sx,sy,sz)
        log.append("Box + config written")
        return {"success":True,"rec_fh":conv["rec_fh"],"rec_pdbqt":conv["rec_pdbqt"],
                "box_pdb":box_pdb,"config_txt":cfg_path,
                "cx":cx,"cy":cy,"cz":cz,"sx":sx,"sy":sy,"sz":sz,
                "ligand_pdb_path":ligand_pdb_path,"cocrystal_ligand_id":cocrystal_ligand_id,
                "n_atoms":rec_sel.numAtoms(),
                "all_ligands":[{"resname":d["resname"],"chain":d["chain"],
                                "resid":d["resid"],"n_atoms":d["n_atoms"]} for d in _all_ligs],
                "log":log}
    except Exception as e:
        log.append(f"ERROR: {e}")
        return {"success":False,"error":str(e),"log":log}

# ══════════════════════════════════════════════════════════════════════════════
# IONIZABLE SITES  —  LAZY compile (never at import)
# ══════════════════════════════════════════════════════════════════════════════

_PKANET_CACHE: dict = {}
_PKA_PATTERNS = [
    _re.compile(r"pK[aA][\w\s\(\)]*?=\s*([+-]?\d+(?:\.\d+)?)", _re.IGNORECASE),
    _re.compile(r"([+-]?\d+(?:\.\d+)?)\s*\((?:pK[aA]|acid dissociation)[^)]*\)", _re.IGNORECASE),
]
_W_AROM_RING_LOST=8.0; _W_PHENOL_TO_KETO_FLIP=6.0
_W_PYROGALLOL_TRIKETO=6.0; _W_CATECHOL_DIKETO=4.0
_W_PHENOL_PRESERVED_BONUS=0.5; _TAUTOMER_PLAUSIBILITY_CUTOFF=3.0
_AMBIGUITY_SCORE_GAP=0.5

_IONIZABLE_SITES_COMPILED = None   # lazy
_CHEM_RULES = None                  # lazy

def _get_compiled_sites():
    global _IONIZABLE_SITES_COMPILED
    if _IONIZABLE_SITES_COMPILED is None:
        _IONIZABLE_SITES_COMPILED = _compile_ionizable_sites()
    return _IONIZABLE_SITES_COMPILED

def _get_chem_rules():
    global _CHEM_RULES
    if _CHEM_RULES is None:
        _CHEM_RULES = _compile_chem_rules()
    return _CHEM_RULES

_IONIZABLE_SITE_DEF = [
    ("sulfonic_acid",      "[SX4](=O)(=O)[OX2H1]",1.0,"acid"),
    ("carboxylic_acid",    "[CX3](=O)[OX2H1]",4.5,"acid"),
    ("tetrazole",          "c1nn[nH]n1",4.9,"acid"),
    ("imidazole_acid",     "c1cn[nH]c1",6.0,"acid"),
    ("benzimidazole",      "c1ccc2[nH]cnc2c1",5.5,"acid"),
    ("sulfonamide_NH",     "[SX4](=O)(=O)[NX3;H1]",10.1,"acid"),
    ("imide_NH",           "[CX3](=O)[NX3;H1][CX3]=O",9.6,"acid"),
    ("phenol_ortho_CO",    "[OX2H1][c;R]:[c;R][CX3;R](=O)",7.8,"acid"),
    ("catechol_OH",        "[OX2H1][c;R]:[c;R][OX2H1]",9.4,"acid"),
    ("phenol",             "c[OX2H1]",10.0,"acid"),
    ("thiol_arom",         "c[SX2H1]",6.5,"acid"),
    ("thiol_aliph",        "[CX4][SX2H1]",10.5,"acid"),
    ("aniline",            "c[NX3;H1,H2;!$(N~[!#6])]",4.6,"base"),
    ("pyridine_like",      "[$([nX2]1:[c,n]:c:[c,n]:c1),$([nX2]:c:n)]",5.2,"base"),
    ("morpholine_N",       "[NX3;H0;R;$(N1CC[O,S]CC1)]",4.9,"base"),
    ("piperazine_NH",      "[NX3;H1;R;$(N1CCNCC1)]",8.1,"base"),
    ("aliphatic_amine",    "[NX3;H1,H2;!$(NC=O);!$(N~[!#6;!H]);!$([nH]);!$(Nc)]",9.5,"base"),
    ("aliphatic_amine_t",  "[NX3;H0;!$(NC=O);!$(Nc);!$([nH]);!$([N]~[!#6]);!$([N;R]1CC[O,S]CC1);!$([N;R]1CCNCC1)]",9.0,"base"),
    ("amidine",            "[CX3](=[NX2;H0,H1])[NX3;H1,H2]",12.4,"base"),
    ("guanidine",          "[NX3][CX3](=[NX2])[NX3]",13.0,"base"),
]

def _compile_ionizable_sites():
    from rdkit import Chem
    compiled=[]
    for lbl,sma,pka,stype in _IONIZABLE_SITE_DEF:
        pat=Chem.MolFromSmarts(sma)
        if pat: compiled.append((lbl,pat,pka,stype))
    return compiled

_CHEM_BONUS_DEF=[
    ("amide",+2.5,"[CX3](=O)[NX3;H1,H2]"),
    ("lactam",+2.5,"[C;R](=O)[N;R]"),
    ("urea_NH",+1.5,"[NX3;H1][CX3](=O)[NX3;H1,H2]"),
    ("aromatic_ring",+0.3,"c1ccccc1"),
    ("phenol_preserved",_W_PHENOL_PRESERVED_BONUS,"c[OX2H1]"),
]
_CHEM_PENALTY_DEF=[
    ("lactim_ring",-4.0,"[C;R](=[NX2])[OX2H1]"),
    ("iminol_general",-3.5,"[NX2]=[CX3][OX2H1]"),
    ("amide_N_deproton",-5.0,"[$([NX3-]C=O),$([NX3-]c=O)]"),
    ("enol_simple",-1.2,"[CX3](=[CX3])[OX2H1]"),
    ("exo_imine_arom",-2.5,"[NX2;!r]=[cX3]"),
]

def _compile_chem_rules():
    from rdkit import Chem
    rules=[]
    for lbl,wt,sma in _CHEM_BONUS_DEF+_CHEM_PENALTY_DEF:
        pat=Chem.MolFromSmarts(sma)
        if pat: rules.append((lbl,wt,pat))
    return rules

def _find_ionizable_sites(mol):
    sites=[]; seen_ion=set(); claimed=set()
    for lbl,pat,pka,stype in _get_compiled_sites():
        for match in mol.GetSubstructMatches(pat):
            if any(a in claimed for a in match): continue
            ion_atoms=[idx for idx in match
                       if mol.GetAtomWithIdx(idx).GetAtomicNum() in (7,8,16)
                       and mol.GetAtomWithIdx(idx).GetTotalNumHs()>0
                       and idx not in seen_ion]
            if not ion_atoms: continue
            for ion_idx in ion_atoms:
                seen_ion.add(ion_idx)
                sites.append({"label":lbl,"atom_indices":[ion_idx],
                               "ionizable_idx":ion_idx,"heuristic_pka":pka,"site_type":stype})
    return sites

def _hh_fraction_charged(pka,ph,site_type):
    if site_type=="acid": return 1.0/(1.0+10.0**(pka-ph))
    return 1.0/(1.0+10.0**(ph-pka))

def _score_tautomer(smiles,ref_mol=None):
    from rdkit import Chem
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: return -999.0
    total=0.0
    for lbl,wt,pat in _get_chem_rules():
        n=len(mol.GetSubstructMatches(pat))
        if n: total+=wt*n
    return total

def _generate_ranked_microstates(base_smiles,target_ph=7.4,ph_window=1.0,max_tautomers=8,pubchem=None):
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    if pubchem is None: pubchem={}
    ref_mol=Chem.MolFromSmiles(base_smiles)
    if ref_mol is None: return []
    enum=rdMolStandardize.TautomerEnumerator()
    seen_t=set()
    input_canon=Chem.MolToSmiles(ref_mol,isomericSmiles=True,canonical=True)
    seen_t.add(input_canon)
    tautomers=[{"smiles":input_canon,"score":_score_tautomer(input_canon,ref_mol)}]
    try:
        for tmol in enum.Enumerate(ref_mol):
            smi=Chem.MolToSmiles(tmol,isomericSmiles=True,canonical=True)
            if smi in seen_t: continue
            seen_t.add(smi)
            tautomers.append({"smiles":smi,"score":_score_tautomer(smi,ref_mol)})
    except Exception: pass
    tautomers=sorted(tautomers,key=lambda x:-x["score"])[:max_tautomers]
    ph_lo=max(0.0,target_ph-ph_window/2); ph_hi=min(14.0,target_ph+ph_window/2)
    all_micro=[]; seen_smi=set()
    for taut in tautomers:
        raw_states=[taut["smiles"]]
        try:
            from dimorphite_dl import protonate_smiles as _dim
            for kw in [{"ph_min":ph_lo,"ph_max":ph_hi,"max_variants":32},
                       {"min_ph":ph_lo,"max_ph":ph_hi,"max_variants":32}]:
                try: _raw=_dim(taut["smiles"],**kw); break
                except TypeError: continue
            if _raw:
                seen_d={taut["smiles"]}
                for s in (_raw if isinstance(_raw,list) else [_raw]):
                    m=Chem.MolFromSmiles(s) if s else None
                    if m is None: continue
                    c=Chem.MolToSmiles(m,canonical=True)
                    if c and c not in seen_d: seen_d.add(c); raw_states.append(c)
        except Exception: pass
        for smi in raw_states:
            if smi in seen_smi: continue
            mol_c=Chem.MolFromSmiles(smi)
            if mol_c is None: continue
            seen_smi.add(smi)
            net=int(Chem.GetFormalCharge(mol_c))
            all_micro.append({"microstate_smiles":smi,"tautomer_smiles":taut["smiles"],
                               "selection_score":float(_score_tautomer(smi)),"net_charge":net})
    all_micro.sort(key=lambda x:(-x["selection_score"],abs(x["net_charge"])))
    return all_micro

def protonate_pkanet(smiles,ph,use_pubchem=False,max_tautomers=8,ph_window=1.0):
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    log=[]; canonical=smiles.strip()
    try:
        mol=Chem.MolFromSmiles(canonical)
        if mol:
            mol=rdMolStandardize.LargestFragmentChooser().choose(mol)
            try: mol=rdMolStandardize.Normalizer().normalize(mol)
            except Exception: pass
            canonical=Chem.MolToSmiles(mol,isomericSmiles=True,canonical=True)
            log.append("RDKit standardized")
    except Exception as e: log.append(f"Standardization skipped: {e}")
    pubchem={"available":False,"pka_values":[],"confidence":"low"}
    try:
        all_micro=_generate_ranked_microstates(canonical,target_ph=ph,ph_window=ph_window,
                                               max_tautomers=max_tautomers,pubchem=pubchem)
        if not all_micro: raise ValueError("No microstates")
        best=all_micro[0]; best_smi=best["microstate_smiles"]; charge=best["net_charge"]
        log.append(f"Ranked {len(all_micro)} microstates — best score: {best['selection_score']:.2f}")
    except Exception as e:
        log.append(f"Microstate ranking failed ({e}) — Dimorphite fallback")
        best_smi=canonical
        try:
            from dimorphite_dl import protonate_smiles as _dim
            vs=_dim(canonical,ph_min=ph-0.5,ph_max=ph+0.5,max_variants=1)
            if vs: best_smi=vs[0] if isinstance(vs,list) else vs
        except Exception: pass
        mol_fb=Chem.MolFromSmiles(best_smi)
        charge=int(Chem.GetFormalCharge(mol_fb)) if mol_fb else 0
    log.append(f"Formal charge: {charge:+d}")
    return best_smi,charge,log

# ══════════════════════════════════════════════════════════════════════════════
# ADMET  —  ADMETlab 3.0 REST API  (replaces ADMET-AI ~500MB model)
# ══════════════════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=64)
def _admetlab3_predict(smiles: str) -> dict:
    """
    POST to ADMETlab 3.0 API.  lru_cache avoids repeat calls for same SMILES.
    Raises RuntimeError on network / HTTP failure (caller shows fallback UI).
    """
    import requests
    try:
        r = requests.post(
            "https://admetlab3.scbdd.com/api/evaluate",
            json={"smiles": smiles},
            headers={"Content-Type": "application/json"},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
    except requests.Timeout:
        raise RuntimeError("ADMETlab 3.0 timeout (>20s) — แสดงผล RDKit อย่างเดียว")
    except requests.ConnectionError:
        raise RuntimeError("ไม่สามารถเชื่อมต่อ ADMETlab 3.0 — ตรวจสอบ internet")
    except requests.HTTPError as e:
        raise RuntimeError(f"ADMETlab 3.0 HTTP error: {e}")
    except Exception as e:
        raise RuntimeError(f"ADMETlab 3.0 error: {e}")

    def _f(v, d=None):
        try: return float(v) if v is not None else d
        except: return d

    def _b(v):
        if v is None: return None
        if isinstance(v,bool): return v
        return str(v).strip().lower() in ("yes","true","1","positive")

    return {
        # Absorption
        "caco2":              _f(data.get("Caco-2")),
        "hia":                _f(data.get("HIA")),
        "bioavailability_ml": _f(data.get("F20%") or data.get("Bioavailability")),
        "solubility":         _f(data.get("Solubility") or data.get("LogS")),
        "pgp_substrate_ml":   _b(data.get("Pgp-substrate")),
        "pgp_inhibitor":      _b(data.get("Pgp-inhibitor")),
        # Distribution
        "bbb_prob":           _f(data.get("BBB") or data.get("BBB_prob")),
        "ppb":                _f(data.get("PPB")),
        "vdd":                _f(data.get("VDss")),
        "lipophilicity_ml":   _f(data.get("LogP") or data.get("Lipophilicity")),
        # CYP inhibition
        "cyp1a2_inh":         _b(data.get("CYP1A2-inhibitor")),
        "cyp2c9_inh":         _b(data.get("CYP2C9-inhibitor")),
        "cyp2c19_inh":        _b(data.get("CYP2C19-inhibitor")),
        "cyp2d6_inh":         _b(data.get("CYP2D6-inhibitor")),
        "cyp3a4_inh":         _b(data.get("CYP3A4-inhibitor")),
        "cyp1a2_prob":        _f(data.get("CYP1A2-inhibitor_prob")),
        "cyp2c9_prob":        _f(data.get("CYP2C9-inhibitor_prob")),
        "cyp2c19_prob":       _f(data.get("CYP2C19-inhibitor_prob")),
        "cyp2d6_prob":        _f(data.get("CYP2D6-inhibitor_prob")),
        "cyp3a4_prob":        _f(data.get("CYP3A4-inhibitor_prob")),
        # CYP substrate
        "cyp2c9_sub":         _b(data.get("CYP2C9-substrate")),
        "cyp2d6_sub":         _b(data.get("CYP2D6-substrate")),
        "cyp3a4_sub":         _b(data.get("CYP3A4-substrate")),
        "cyp2c9_sub_prob":    _f(data.get("CYP2C9-substrate_prob")),
        "cyp2d6_sub_prob":    _f(data.get("CYP2D6-substrate_prob")),
        "cyp3a4_sub_prob":    _f(data.get("CYP3A4-substrate_prob")),
        # Excretion
        "half_life":          _f(data.get("Half-life") or data.get("T1/2")),
        "clearance":          _f(data.get("CL") or data.get("Clearance")),
        # Toxicity
        "herg":               _b(data.get("hERG")),
        "herg_prob":          _f(data.get("hERG_prob")),
        "ames":               _b(data.get("AMES")),
        "ames_prob":          _f(data.get("AMES_prob")),
        "dili":               _b(data.get("DILI")),
        "dili_prob":          _f(data.get("DILI_prob")),
        "skin_reaction":      _b(data.get("Skin-reaction")),
        "skin_prob":          _f(data.get("Skin-reaction_prob")),
        "ld50":               _f(data.get("LD50") or data.get("Acute-toxicity")),
        "_raw": data,
    }

# ══════════════════════════════════════════════════════════════════════════════
# LIGAND PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def _meeko_to_pdbqt(mol, out_path):
    from meeko import MoleculePreparation
    prep = MoleculePreparation()
    try:
        from meeko import PDBQTWriterLegacy
        setups=prep.prepare(mol)
        pdbqt_str,_,_=PDBQTWriterLegacy.write_string(setups[0])
    except (ImportError,AttributeError):
        prep.prepare(mol); pdbqt_str=prep.write_pdbqt_string()
    with open(out_path,"w") as f: f.write(pdbqt_str)

def _ligand_charge_summary(smiles):
    from rdkit import Chem
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f"Bad SMILES: {smiles[:60]}")
    net=n_pos=n_neg=0; rows=[]
    for atom in mol.GetAtoms():
        fc=int(atom.GetFormalCharge()); net+=fc
        if fc>0: n_pos+=1
        elif fc<0: n_neg+=1
        if fc!=0: rows.append({"atom_idx":atom.GetIdx(),"symbol":atom.GetSymbol(),"formal_charge":fc})
    return {"net_charge":int(net),"charged_atoms":rows,"is_zwitterion":bool(n_pos>0 and n_neg>0 and net==0)}

def _charged_atoms_text(rows):
    if not rows: return "none"
    return ", ".join(f"{r['symbol']}{r['atom_idx']}({r['formal_charge']:+d})" for r in rows)

def _apply_ionizable_site_correction(original_smiles,current_smiles,ph,log):
    from rdkit import Chem
    mol=Chem.MolFromSmiles(current_smiles)
    if mol is None: return current_smiles
    fc_map={a.GetIdx():int(a.GetFormalCharge()) for a in mol.GetAtoms()}
    sites=_find_ionizable_sites(mol)
    missed=sorted([s for s in sites if s["site_type"]=="acid"
                   and s["heuristic_pka"]<ph and s.get("ionizable_idx") is not None
                   and mol.GetAtomWithIdx(s["ionizable_idx"]).GetTotalNumHs()>0
                   and fc_map.get(s["ionizable_idx"],0)>=0],
                  key=lambda s:s["heuristic_pka"])
    if not missed: return current_smiles
    site=missed[0]
    try:
        rw=Chem.RWMol(mol)
        rw.GetAtomWithIdx(site["ionizable_idx"]).SetFormalCharge(-1)
        Chem.SanitizeMol(rw); corrected=Chem.MolToSmiles(rw,canonical=True)
        log.append(f"pKa site correction: deprotonated {site['label']} (pKa={site['heuristic_pka']:.1f})")
        return corrected
    except Exception as e:
        log.append(f"Site correction failed: {e}"); return current_smiles


# ── Dimorphite-DL protonation state selector (bug-fix) ───────────────────────
_SMARTS_CACHE: dict = {}

def _match(mol, sma: str) -> int:
    """Cached SMARTS substructure match count."""
    from rdkit import Chem as _Chem
    if sma not in _SMARTS_CACHE:
        _SMARTS_CACHE[sma] = _Chem.MolFromSmarts(sma)
    p = _SMARTS_CACHE[sma]
    return len(mol.GetSubstructMatches(p)) if p else 0


def _naive_expected_charge(smiles: str) -> int:
    """
    Rule-based formal charge at pH 7.4 — fallback target when Dimorphite-DL
    state order diverges from chemical expectation.

    Rules (conservative pKa refs):
      Carboxylic acid  pKa ~4.5 => -1
      Sulfonic acid            => -1
      Aliphatic NH2    pKa ~10 => +1
      Aliphatic NHR    pKa ~10 => +1
      Flavone/chromanone core + >=1 ArOH => -1  (one activated phenol ionised)
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    q = 0
    q -= _match(mol, "[CX3](=O)[OX2H1]")           # COOH
    q -= _match(mol, "[SX4](=O)(=O)[OX2H1]")       # SO3H
    q += _match(mol, "[NX3;H2;!$(NC=O);!$(Nc)]")   # primary aliphatic amine
    q += _match(mol, "[NX3;H1;!$(NC=O);!$([nH]);!$(Nc)]")  # secondary aliphatic amine
    if q == 0:
        n_aro_oh = _match(mol, "[OX2H1]c")
        has_flavone_core = bool(
            _match(mol, "O=c1ccoc2ccccc12") or   # aromatic chromone
            _match(mol, "O=C1C=COc2ccccc21") or  # dienone form
            _match(mol, "O=C1CCOc2ccccc21") or   # chromanone (flavanone)
            _match(mol, "O=C1CC[OX2]c2ccccc12")  # general chromanone
        )
        if has_flavone_core and n_aro_oh >= 1:
            q = -1
    return q


def pick_best_dimorphite_state(states: list, target_ph: float,
                                input_smiles: str) -> str:
    """
    Select the most appropriate protonation microstate from Dimorphite-DL output.

    Dimorphite-DL output order is non-deterministic: states[0] is often
    over-deprotonated (e.g. -2 for flavonoids instead of -1).

    Algorithm: compare states[0] against a rule-based expected charge.
    If they disagree by >=1 unit, pick the available state closest to expected
    (tie-break: smallest absolute charge).
    """
    from rdkit import Chem
    from rdkit.Chem import GetFormalCharge
    if not states:
        return input_smiles
    m0 = Chem.MolFromSmiles(states[0])
    c0 = GetFormalCharge(m0) if m0 else 0
    expected = _naive_expected_charge(input_smiles)
    if abs(c0 - expected) < 1:
        return states[0]
    valid = [(GetFormalCharge(m), s)
             for s in states
             for m in [Chem.MolFromSmiles(s)] if m]
    if not valid:
        return states[0]
    return min(valid, key=lambda x: (abs(x[0] - expected), abs(x[0])))[1]

def prepare_ligand(smiles,name,ph,wdir,mode="dimorphite",use_pubchem=False,max_tautomers=8,ph_window=1.0):
    _rdkit_six_patch()
    from rdkit import Chem
    from rdkit.Chem import AllChem
    wdir=Path(wdir); log=[]; out_pdbqt=str(wdir/f"{name}.pdbqt"); out_sdf=str(wdir/f"{name}_3d.sdf")
    try:
        raw=smiles.strip(); prot=raw; actual_mode=mode or "dimorphite"
        if actual_mode=="neutral":
            mol_check=Chem.MolFromSmiles(raw)
            if mol_check is None: raise ValueError(f"RDKit could not parse SMILES: {raw[:60]}")
            prot=Chem.MolToSmiles(mol_check,isomericSmiles=True,canonical=True)
            log.append("Neutral mode (keep input charge state)")
        elif actual_mode=="pkanet":
            try:
                prot,_,pka_log=protonate_pkanet(raw,ph,use_pubchem=use_pubchem,
                                                max_tautomers=max_tautomers,ph_window=ph_window)
                log.extend(pka_log); log.append("pKaNET protonation applied")
            except Exception as _pke:
                log.append(f"pKaNET failed ({_pke}) — Dimorphite fallback")
                try:
                    from dimorphite_dl import protonate_smiles
                    vs=protonate_smiles(raw,ph_min=ph,ph_max=ph,max_variants=1)
                    if vs: prot=vs[0] if isinstance(vs,list) else vs
                    prot=_apply_ionizable_site_correction(raw,prot,ph,log)
                except Exception as e2: log.append(f"Dimorphite fallback skipped: {e2}")
        else:
            try:
                from dimorphite_dl import protonate_smiles
                vs = list(protonate_smiles(prot, ph_min=ph-1.0, ph_max=ph+1.0, max_variants=128) or [])
                if vs:
                    prot = pick_best_dimorphite_state(vs, ph, prot)
                    log.append(f"Dimorphite-DL pH {ph:.1f} ({len(vs)} states -> best selected)")
                else:
                    log.append("Dimorphite-DL returned no variants")
            except Exception as e:
                log.append(f"Dimorphite-DL skipped: {e}")
            prot = _apply_ionizable_site_correction(raw, prot, ph, log)
        mol=Chem.MolFromSmiles(prot)
        if mol is None: raise ValueError(f"RDKit could not parse SMILES: {prot[:60]}")
        charge_info=_ligand_charge_summary(prot); charge=int(charge_info["net_charge"])
        log.append(f"Formal charge: {charge:+d}")
        log.append(f"Charged atoms: {_charged_atoms_text(charge_info['charged_atoms'])}")
        mol=Chem.AddHs(mol)
        try: params=AllChem.ETKDGv3()
        except AttributeError: params=AllChem.ETKDG()
        params.randomSeed=42
        if AllChem.EmbedMolecule(mol,params)==-1:
            AllChem.EmbedMolecule(mol,useRandomCoords=True,randomSeed=42)
        if AllChem.MMFFHasAllMoleculeParams(mol): AllChem.MMFFOptimizeMolecule(mol,maxIters=500)
        else: AllChem.UFFOptimizeMolecule(mol,maxIters=500)
        log.append("3D conformer generated + minimised")
        with Chem.SDWriter(out_sdf) as w: w.write(mol)
        try:
            _meeko_to_pdbqt(mol,out_pdbqt); log.append("PDBQT written (Meeko)")
        except Exception as e_meeko:
            log.append(f"Meeko failed ({e_meeko}), trying OpenBabel...")
            subprocess.run(f'obabel "{out_sdf}" -O "{out_pdbqt}" -xh 2>/dev/null',shell=True,timeout=30)
            if not Path(out_pdbqt).exists() or Path(out_pdbqt).stat().st_size<10:
                raise ValueError(f"Both Meeko and OpenBabel failed: {e_meeko}")
            log.append("PDBQT written (OpenBabel fallback)")
        return {"success":True,"pdbqt":out_pdbqt,"sdf":out_sdf,"input_smiles":raw,
                "prepared_smiles":prot,"prot_smiles":prot,"charge":charge,"net_charge":charge,
                "charge_method":"rdkit_formal_charge","charged_atoms":charge_info["charged_atoms"],
                "is_zwitterion":charge_info["is_zwitterion"],"protonation_mode":actual_mode,"log":log}
    except Exception as e:
        log.append(f"ERROR: {e}")
        return {"success":False,"error":str(e),"log":log}

def prepare_ligand_from_file(file_path,name,wdir):
    _rdkit_six_patch()
    from rdkit import Chem
    from rdkit.Chem import AllChem
    wdir=Path(wdir); log=[]; out_pdbqt=str(wdir/f"{name}.pdbqt"); out_sdf=str(wdir/f"{name}_3d.sdf")
    ext=Path(file_path).suffix.lower()
    try:
        mol=None
        if ext==".sdf":
            supp=Chem.SDMolSupplier(file_path,removeHs=False,sanitize=True)
            mols=[m for m in supp if m]
            if not mols:
                supp=Chem.SDMolSupplier(file_path,removeHs=False,sanitize=False)
                mols=[m for m in supp if m]
            if mols: mol=mols[0]
        elif ext in (".mol2",".pdb"):
            _ob_sdf=str(wdir/f"{name}_ob.sdf")
            subprocess.run(f'obabel "{file_path}" -O "{_ob_sdf}" 2>/dev/null',shell=True,timeout=30)
            if Path(_ob_sdf).exists() and Path(_ob_sdf).stat().st_size>10:
                supp=Chem.SDMolSupplier(_ob_sdf,removeHs=False,sanitize=True)
                mols=[m for m in supp if m]
                if mols: mol=mols[0]; log.append("Converted via OpenBabel")
            if mol is None:
                if ext==".mol2": mol=Chem.MolFromMol2File(file_path,removeHs=False)
                else: mol=Chem.MolFromPDBFile(file_path,removeHs=False)
        if mol is None: raise ValueError(f"ไม่สามารถอ่าน molecule จากไฟล์ {Path(file_path).name}")
        frags=Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=False)
        if len(frags)>1:
            frags=sorted(frags,key=lambda m:m.GetNumAtoms(),reverse=True)
            mol=frags[0]; log.append(f"{len(frags)} fragments — kept largest ({mol.GetNumAtoms()} atoms)")
        try: Chem.SanitizeMol(mol)
        except: pass
        log.append("Loaded molecule from file (no protonation)")
        try: smi=Chem.MolToSmiles(Chem.RemoveHs(mol))
        except: smi=name
        try: charge=Chem.GetFormalCharge(mol)
        except: charge=0
        log.append(f"Formal charge: {charge:+d}")
        mol=Chem.AddHs(mol,addCoords=True)
        conf=mol.GetConformer(0) if mol.GetNumConformers()>0 else None
        if conf is None or not conf.Is3D():
            try: params=AllChem.ETKDGv3()
            except: params=AllChem.ETKDG()
            params.randomSeed=42
            if AllChem.EmbedMolecule(mol,params)==-1:
                AllChem.EmbedMolecule(mol,useRandomCoords=True,randomSeed=42)
            if AllChem.MMFFHasAllMoleculeParams(mol): AllChem.MMFFOptimizeMolecule(mol,maxIters=500)
            else: AllChem.UFFOptimizeMolecule(mol,maxIters=500)
            log.append("3D conformer generated")
        else: log.append("Using 3D coordinates from uploaded file")
        with Chem.SDWriter(out_sdf) as w: w.write(mol)
        try: _meeko_to_pdbqt(mol,out_pdbqt); log.append("PDBQT written (Meeko)")
        except Exception as e_meeko:
            log.append(f"Meeko failed, trying OpenBabel...")
            subprocess.run(f'obabel "{out_sdf}" -O "{out_pdbqt}" -xh 2>/dev/null',shell=True,timeout=30)
            if not Path(out_pdbqt).exists() or Path(out_pdbqt).stat().st_size<10:
                raise ValueError(f"Both Meeko and OpenBabel failed: {e_meeko}")
            log.append("PDBQT written (OpenBabel fallback)")
        return {"success":True,"pdbqt":out_pdbqt,"sdf":out_sdf,"prot_smiles":smi,"charge":charge,"log":log}
    except Exception as e:
        log.append(f"ERROR: {e}")
        return {"success":False,"error":str(e),"log":log}

def smiles_from_file(file_path,wdir):
    wdir=Path(wdir); ext=Path(file_path).suffix.lower()
    if ext==".sdf":
        from rdkit import Chem
        mols=[m for m in Chem.SDMolSupplier(file_path,sanitize=True) if m]
        if not mols: raise ValueError("No valid molecule in SDF file")
        return Chem.MolToSmiles(mols[0])
    else:
        smi_tmp=str(wdir/"lig_upload.smi")
        run_cmd(f'obabel "{file_path}" -O "{smi_tmp}" --canonical 2>/dev/null')
        for line in open(smi_tmp):
            pts=line.strip().split(None,1)
            if pts: return pts[0]
        raise ValueError("Could not convert structure file to SMILES")

# ══════════════════════════════════════════════════════════════════════════════
# DOCKING
# ══════════════════════════════════════════════════════════════════════════════

def run_vina(receptor_pdbqt,ligand_pdbqt,config_txt,vina_path,
             exhaustiveness=16,n_modes=10,energy_range=3,wdir=".",out_name="out"):
    wdir=Path(wdir); out_pdbqt=str(wdir/f"{out_name}_out.pdbqt"); out_sdf=str(wdir/f"{out_name}_out.sdf")
    rc,vlog=run_cmd(
        f'"{vina_path}" --receptor "{receptor_pdbqt}" --ligand "{ligand_pdbqt}" '
        f'--config "{config_txt}" --exhaustiveness {exhaustiveness} '
        f'--num_modes {n_modes} --energy_range {energy_range} --out "{out_pdbqt}"',
        cwd=str(wdir))
    if rc!=0 or not os.path.exists(out_pdbqt):
        vll=vlog.lower()
        if "could not open" in vll or "no such file" in vll:
            error="ไม่พบไฟล์ receptor หรือ ligand — ลอง prepare ใหม่อีกครั้ง"
        elif "no atoms" in vll or "empty" in vll:
            error="Ligand ไม่มี atom ที่ถูกต้อง — ตรวจสอบ SMILES หรือ structure file"
        elif "outside the box" in vll:
            error="Ligand อยู่นอก docking box — ตรวจสอบ grid center และ box size"
        elif "exhaustiveness" in vll:
            error="ค่า exhaustiveness ไม่ถูกต้อง"
        elif rc!=0: error=f"Vina exit code {rc} — ดู log ด้านล่าง"
        else: error="Vina ไม่สร้างไฟล์ output — ดู log ด้านล่าง"
        return {"success":False,"error":error,"log":vlog}
    run_cmd(f'obabel "{out_pdbqt}" -O "{out_sdf}" 2>/dev/null')
    scores=[]; cur_model=None
    for line in open(out_pdbqt):
        ln=line.strip()
        if ln.startswith("MODEL"):
            try: cur_model=int(ln.split()[1])
            except: pass
        elif ln.startswith("REMARK VINA RESULT:"):
            try:
                p=ln.split()
                scores.append({"pose":cur_model,"affinity":float(p[3]),"rmsd_lb":float(p[4]),"rmsd_ub":float(p[5])})
            except: pass
    return {"success":True,"out_pdbqt":out_pdbqt,"out_sdf":out_sdf,
            "scores":scores,"top_score":scores[0]["affinity"] if scores else None,"log":vlog}

# ══════════════════════════════════════════════════════════════════════════════
# SDF UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_mols_from_sdf(sdf_path,sanitize=True):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mols=[]
    try:
        sup=Chem.SDMolSupplier(sdf_path,sanitize=sanitize,removeHs=False)
        for m in sup:
            if m is not None: mols.append(m)
    except Exception: pass
    RDLogger.EnableLog("rdApp.error")
    return mols

def write_single_pose(mol,path):
    from rdkit import Chem
    with Chem.SDWriter(path) as w: w.write(mol)

def write_single_pose_pdb(mol,path):
    from rdkit import Chem
    Chem.MolToPDBFile(mol,path)

def fix_sdf_bond_orders(raw_sdf,smiles,fixed_sdf):
    import shutil
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*"); log=[]
    try:
        template=Chem.MolFromSmiles(smiles)
        if template is None: raise ValueError("Cannot parse SMILES for template")
        try: Chem.Kekulize(template,clearAromaticFlags=True)
        except: pass
    except Exception as e:
        log.append(f"Could not build template: {e}"); shutil.copy(raw_sdf,fixed_sdf)
        RDLogger.EnableLog("rdApp.error"); return log
    supplier=Chem.SDMolSupplier(raw_sdf,sanitize=False,removeHs=False)
    writer=Chem.SDWriter(fixed_sdf); writer.SetKekulize(False); ok=err=0
    for i,mol in enumerate(supplier):
        if mol is None: log.append(f"Pose {i+1}: could not read — skipped"); err+=1; continue
        try:
            probe_noH=Chem.RemoveHs(mol,sanitize=False)
            fixed=AllChem.AssignBondOrdersFromTemplate(template,probe_noH)
            Chem.SanitizeMol(fixed); writer.write(fixed); ok+=1
        except Exception as e:
            log.append(f"Pose {i+1}: fix failed ({e}) — writing raw")
            writer.write(Chem.RemoveHs(mol,sanitize=False)); err+=1
    writer.close(); RDLogger.EnableLog("rdApp.error")
    log.append(f"Bond-order fix: {ok} OK, {err} fallback")
    return log

def convert_sdf_to_v2000(sdf_path):
    from rdkit import Chem, RDLogger
    out=sdf_path.replace(".sdf","_v2000.sdf")
    if out==sdf_path: out=sdf_path+"_v2000.sdf"
    RDLogger.DisableLog("rdApp.*")
    try:
        for sanitize in (True,False):
            sup=Chem.SDMolSupplier(sdf_path,sanitize=sanitize,removeHs=True)
            mol=next((m for m in sup if m is not None),None)
            if mol is not None:
                if not sanitize:
                    try: Chem.SanitizeMol(mol)
                    except: pass
                break
        if mol is not None and mol.GetNumConformers()>0:
            mol_noH=Chem.RemoveHs(mol,sanitize=False)
            try: Chem.SanitizeMol(mol_noH)
            except: pass
            try: Chem.Kekulize(mol_noH,clearAromaticFlags=True)
            except: pass
            w=Chem.SDWriter(out); w.SetKekulize(True); w.write(mol_noH); w.close()
            if os.path.exists(out) and os.path.getsize(out)>10:
                RDLogger.EnableLog("rdApp.error"); return out
    except Exception: pass
    RDLogger.EnableLog("rdApp.error")
    rc,_=run_cmd(f'obabel "{sdf_path}" -O "{out}" 2>/dev/null')
    if rc==0 and os.path.exists(out) and os.path.getsize(out)>10: return out
    return sdf_path

# ══════════════════════════════════════════════════════════════════════════════
# INTERACTION DETECTION  —  uses _cached_parse_pdb
# ══════════════════════════════════════════════════════════════════════════════

def get_interacting_residues(receptor_pdb,lig_mol,cutoff=3.5):
    try:
        import numpy as np
        rec=_cached_parse_pdb(receptor_pdb)   # cached
        if rec is None: return []
        conf=lig_mol.GetConformer()
        lig_xyz=np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z] for i in range(lig_mol.GetNumAtoms())])
        r_xyz=rec.getCoords(); chains=rec.getChids(); resids=rec.getResnums(); resnames=rec.getResnames()
        seen={}
        for j in range(len(r_xyz)):
            if np.linalg.norm(lig_xyz-r_xyz[j],axis=1).min()<=cutoff:
                key=(str(chains[j]),int(resids[j]))
                if key not in seen: seen[key]=str(resnames[j])
        return [{"chain":k[0],"resi":k[1],"resn":v} for k,v in seen.items()]
    except Exception: return []

def calc_rmsd_heavy(pose_mol,crystal_pdb_path):
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFMCS
        import numpy as np
        if not os.path.exists(crystal_pdb_path): return None
        cryst=Chem.MolFromPDBFile(crystal_pdb_path,sanitize=True,removeHs=True)
        if cryst is None or cryst.GetNumConformers()==0: return None
        pose=Chem.RemoveHs(pose_mol,sanitize=False)
        try: Chem.SanitizeMol(pose)
        except: pass
        if pose.GetNumConformers()==0: return None
        mcs=rdFMCS.FindMCS([pose,cryst],timeout=10,
                           bondCompare=rdFMCS.BondCompare.CompareAny,
                           atomCompare=rdFMCS.AtomCompare.CompareElements)
        if mcs.numAtoms<3: return None
        mcs_mol=Chem.MolFromSmarts(mcs.smartsString)
        if mcs_mol is None: return None
        pm=pose.GetSubstructMatches(mcs_mol,uniquify=False)
        cm=cryst.GetSubstructMatches(mcs_mol,uniquify=False)
        if not pm or not cm: return None
        pc,cc=pose.GetConformer(),cryst.GetConformer()
        def _rmsd(pm,cm):
            sq=sum((pc.GetAtomPosition(pi).x-cc.GetAtomPosition(ci).x)**2+
                   (pc.GetAtomPosition(pi).y-cc.GetAtomPosition(ci).y)**2+
                   (pc.GetAtomPosition(pi).z-cc.GetAtomPosition(ci).z)**2
                   for pi,ci in zip(pm,cm))
            return float(np.sqrt(sq/len(pm)))
        return min(_rmsd(p,c) for p in pm for c in cm)
    except Exception: return None

_AROM_ATOMS={"PHE","TYR","TRP","HIS","HEM","HEC","HEA","HEB","HDD","HDM"}
_AROM_ATOM_NAMES={"CG","CD1","CD2","CE1","CE2","CZ","ND1","NE2","CE3","CZ2","CZ3","CH2"}
_HYDR_BASE={"ALA","VAL","ILE","LEU","MET","PHE","TRP","PRO","GLY","TYR","HIS"}
_HYDR_EXTENDED=_HYDR_BASE|HEME_RESNAMES
_ITYPE_PRIORITY=["metal","ionic","halogen","hbond_to_halogen","hbond","pi_pi","cation_pi","hydrophobic"]

def _detect_all_interactions(lig_mol_3d,receptor_pdb,cutoff=4.5):
    import numpy as np
    rec=_cached_parse_pdb(receptor_pdb)   # cached
    if rec is None: return []
    rc=np.array(rec.getCoords(),dtype=float)
    rrn=rec.getResnames(); rch=rec.getChids(); rri=rec.getResnums()
    ran=rec.getNames(); rel=rec.getElements()
    conf=lig_mol_3d.GetConformer(); nl=lig_mol_3d.GetNumAtoms()
    lxyz=np.array([[conf.GetAtomPosition(i).x,
                    conf.GetAtomPosition(i).y,
                    conf.GetAtomPosition(i).z] for i in range(nl)],dtype=float)
    latom=[lig_mol_3d.GetAtomWithIdx(i) for i in range(nl)]
    lel=[a.GetSymbol().upper() for a in latom]; lchg=[a.GetFormalCharge() for a in latom]
    nr=len(rc)
    r_el=[rel[j].strip().upper() if rel[j] and rel[j].strip() else ran[j][:1].upper() for j in range(nr)]
    r_rn=[rrn[j].strip() for j in range(nr)]
    r_an=[ran[j].strip() for j in range(nr)]
    r_ch=[rch[j].strip() for j in range(nr)]
    r_ri=[int(rri[j]) for j in range(nr)]
    LIG_ACCEPTOR_EL={"N","O","F","S"}
    lig_is_acceptor=np.array([lel[i] in LIG_ACCEPTOR_EL for i in range(nl)])
    lig_is_hydrophobic=np.array([lel[i] in {"C","S","CL","BR","I","F"} for i in range(nl)])
    def _is_lig_donor(i):
        a=latom[i]
        if lel[i] not in ("N","O","S","F"): return False
        for nb in a.GetNeighbors():
            if nb.GetAtomicNum()==1: return True
        return a.GetTotalNumHs()>0
    lig_is_donor=np.array([_is_lig_donor(i) for i in range(nl)])
    def _angle3(a,b,c):
        va=a-b; vc=c-b
        na=np.linalg.norm(va); nc=np.linalg.norm(vc)
        if na<1e-8 or nc<1e-8: return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(va,vc)/(na*nc),-1.0,1.0))))
    results=[]
    PROT_DONOR={"N","OG","OG1","OH","SG","NZ","NH1","NH2","NE","ND1","NE2","NE1","ND2"}
    PROT_ACCEPTOR={"O","OD1","OD2","OE1","OE2","OG","OG1","OH","ND1","NE2","SD"}
    HBOND_DA_MAX=3.5; HBOND_ANG_MIN=120.0
    hbond_residues=set()
    polar_idx=[j for j in range(nr) if r_el[j] in ("N","O","S") and r_rn[j] not in ("HOH","WAT","DOD")]
    for j in polar_idx:
        an=r_an[j]; ch=r_ch[j]; ri=int(r_ri[j]); el=r_el[j]; rp=rc[j]
        key=(ch,ri)
        if key in hbond_residues: continue
        dists_j=np.linalg.norm(lxyz-rp,axis=1)
        if an in PROT_DONOR:
            cand=np.where(lig_is_acceptor&(dists_j<=HBOND_DA_MAX))[0]
            if len(cand):
                for i in cand:
                    d_DA=float(dists_j[i])
                    nbs_i=[nb.GetIdx() for nb in latom[i].GetNeighbors() if nb.GetAtomicNum()!=1 and nb.GetIdx()<nl]
                    if nbs_i and _angle3(rp,lxyz[i],lxyz[nbs_i[0]])<HBOND_ANG_MIN: continue
                    hbond_residues.add(key)
                    results.append(dict(resname=r_rn[j],chain=ch,resid=ri,itype="hbond",
                        distance=round(d_DA,1),lig_atom_idx=int(i),prot_el=el,is_donor=True,ring_atom_indices=None)); break
        if key in hbond_residues: continue
        if an in PROT_ACCEPTOR:
            cand=np.where(lig_is_donor&(dists_j<=HBOND_DA_MAX))[0]
            if len(cand):
                for i in cand:
                    d_DA=float(dists_j[i])
                    hbond_residues.add(key)
                    results.append(dict(resname=r_rn[j],chain=ch,resid=ri,itype="hbond",
                        distance=round(d_DA,1),lig_atom_idx=int(i),prot_el=el,is_donor=False,ring_atom_indices=None)); break
    for j in range(nr):
        rn=r_rn[j]; ch=r_ch[j]; ri=int(r_ri[j]); el=r_el[j]; rp=rc[j]
        dists_j=np.linalg.norm(lxyz-rp,axis=1)
        md=float(dists_j.min()); mi=int(dists_j.argmin())
        if md>max(cutoff+1.0,5.6): continue
        if el in {"C","S","CL","BR","I"} and rn in _HYDR_EXTENDED:
            cand=np.where(lig_is_hydrophobic&(dists_j<cutoff))[0]
            if len(cand):
                i=int(cand[0])
                results.append(dict(resname=rn,chain=ch,resid=ri,itype="hydrophobic",
                    distance=round(float(dists_j[i]),1),lig_atom_idx=i,prot_el=el,is_donor=False,ring_atom_indices=None))
        if rn in {"ASP","GLU"} and el=="O":
            for i in range(nl):
                if lchg[i]>0 and float(dists_j[i])<4.0:
                    results.append(dict(resname=rn,chain=ch,resid=ri,itype="ionic",
                        distance=round(float(dists_j[i]),1),lig_atom_idx=i,prot_el=el,is_donor=False,ring_atom_indices=None)); break
        if rn in {"LYS","ARG"} and el=="N":
            for i in range(nl):
                if lchg[i]<0 and float(dists_j[i])<4.0:
                    results.append(dict(resname=rn,chain=ch,resid=ri,itype="ionic",
                        distance=round(float(dists_j[i]),1),lig_atom_idx=i,prot_el=el,is_donor=True,ring_atom_indices=None)); break
        _is_metal=(rn in METAL_RESNAMES or el in METAL_RESNAMES or (rn in HEME_RESNAMES and r_an[j].strip().upper()=="FE"))
        if _is_metal and md<2.8:
            results.append(dict(resname=rn,chain=ch,resid=ri,itype="metal",
                distance=round(md,1),lig_atom_idx=mi,prot_el=el,is_donor=False,ring_atom_indices=None))
    return results

def _deduplicate_interactions(interactions):
    priority={t:i for i,t in enumerate(_ITYPE_PRIORITY)}; best={}
    for ix in interactions:
        key=(ix["chain"],ix["resid"])
        if key not in best: best[key]=ix
        else:
            pn=priority.get(ix["itype"],99); po=priority.get(best[key]["itype"],99)
            if pn<po or (pn==po and ix["distance"]<best[key]["distance"]): best[key]=ix
    return list(best.values())

# ══════════════════════════════════════════════════════════════════════════════
# POSEVIEW REST API
# ══════════════════════════════════════════════════════════════════════════════

_PP_BASE          = "https://proteins.plus/api/v2/"
_PP_UPLOAD        = _PP_BASE+"molecule_handler/upload/"
_PP_UPLOAD_JOBS   = _PP_BASE+"molecule_handler/upload/jobs/"
_PP_POSEVIEW      = _PP_BASE+"poseview/"
_PP_POSEVIEW_JOBS = _PP_BASE+"poseview/jobs/"
_PP_HEADERS = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer":"https://proteins.plus/","Origin":"https://proteins.plus",
}
_PP_PROTEIN_CACHE: dict = {}

def _pp_poll(job_id,poll_url,poll_interval=2,max_polls=60):
    import requests
    job=requests.get(poll_url+job_id+"/",headers=_PP_HEADERS,timeout=15).json()
    status=str(job.get("status","")).lower(); polls=0
    while status in ("pending","running","processing","queued",""):
        if polls>=max_polls: raise RuntimeError(f"Job {job_id} timed out")
        time.sleep(poll_interval); polls+=1
        job=requests.get(poll_url+job_id+"/",headers=_PP_HEADERS,timeout=15).json()
        status=str(job.get("status","")).lower()
    return job

def _prepare_pdb_for_poseview(receptor_pdb):
    rec_dir=os.path.dirname(os.path.abspath(receptor_pdb))
    candidates=[os.path.join(rec_dir,"receptor_atoms.pdb"),receptor_pdb]
    source=next((p for p in candidates if os.path.exists(p) and os.path.getsize(p)>100),receptor_pdb)
    out=os.path.join(rec_dir,"receptor_pv_clean.pdb")
    try:
        kept=[]; serial=0
        with open(source) as f:
            for line in f:
                rec=line[:6].strip()
                if rec not in ("ATOM","HETATM","TER","END"): continue
                if rec in ("ATOM","HETATM"):
                    atom_name=line[12:16].strip(); element=line[76:78].strip() if len(line)>76 else ""
                    if element.upper()=="H" or (not element and atom_name.startswith("H")): continue
                    serial+=1; line=f"{line[:6]}{serial:5d}{line[11:]}"
                kept.append(line if line.endswith("\n") else line+"\n")
        if not kept: return receptor_pdb
        if not any(l.startswith("END") for l in kept): kept.append("END\n")
        with open(out,"w") as f: f.writelines(kept)
        if os.path.exists(out) and os.path.getsize(out)>100: return out
    except Exception: pass
    return receptor_pdb

def call_poseview_v1(receptor_pdb,pose_sdf):
    import requests
    last_error="Unknown error"
    rec_to_send=_prepare_pdb_for_poseview(receptor_pdb)
    for attempt in range(1,_PV_MAX_RETRIES+1):
        if attempt>1: time.sleep(_PV_RETRY_DELAY)
        try:
            with open(rec_to_send) as rf, open(pose_sdf) as lf:
                r=requests.post(_PP_POSEVIEW,
                    files={"protein_file":("receptor.pdb",rf,"chemical/x-pdb"),
                           "ligand_file":("ligand.sdf",lf,"chemical/x-mdl-sdfile")},
                    headers=_PP_HEADERS,timeout=30)
            r.raise_for_status()
            pv_job_id=r.json().get("job_id") or r.json().get("id")
            if not pv_job_id: last_error="No job_id"; continue
        except Exception as e: last_error=f"Submission failed (attempt {attempt}): {e}"; continue
        try: pv_job=_pp_poll(pv_job_id,_PP_POSEVIEW_JOBS)
        except Exception as e: last_error=str(e); continue
        status=str(pv_job.get("status","")).lower()
        if status!="success": last_error=f"PoseView status '{status}'"; continue
        img_url=pv_job.get("image","")
        if not img_url: last_error="No image URL"; continue
        try:
            resp=requests.get(img_url,headers=_PP_HEADERS,timeout=20)
            resp.raise_for_status(); return resp.content,None
        except Exception as e: last_error=f"SVG download failed: {e}"; continue
    return None,last_error

def warm_poseview_cache(receptor_pdb):
    return True,"Direct POST mode — no pre-upload needed"

def clear_poseview_cache():
    _PP_PROTEIN_CACHE.clear()
    _cached_parse_pdb.cache_clear()   # clear parsePDB cache เมื่อ receptor เปลี่ยน

def call_poseview2_ref(pdb_code,ligand_id):
    import requests
    _SUBMIT="https://proteins.plus/api/poseview2_rest"; last_error="Unknown"
    for attempt in range(1,_PV_MAX_RETRIES+1):
        if attempt>1: time.sleep(_PV_RETRY_DELAY)
        try:
            r=requests.post(_SUBMIT,
                json={"poseview2":{"pdbCode":pdb_code.strip().lower(),"ligand":ligand_id.strip()}},
                headers={"Accept":"application/json","Content-Type":"application/json"},timeout=30)
            data=r.json()
            if r.status_code not in (200,202): last_error=f"Status {r.status_code}"; continue
            location=data.get("location","")
            if not location: last_error="No job location"; continue
        except Exception as e: last_error=f"Submission error: {e}"; continue
        for poll_i in range(_PV_POLL_ATTEMPTS):
            time.sleep(2)
            try:
                poll=requests.get(location,headers={"Accept":"application/json"},timeout=15).json()
                sc=poll.get("status_code")
                if sc==200:
                    svg_url=poll.get("result_svg","")
                    if not svg_url: last_error="No result_svg"; break
                    resp=requests.get(svg_url,timeout=20); resp.raise_for_status()
                    return resp.content,None
                elif sc==202: continue
                else: last_error=f"Poll status {sc}"; break
            except Exception as e: last_error=f"Poll error: {e}"; continue
        else: last_error=f"Timed out after {_PV_POLL_ATTEMPTS*2}s"
    return None,last_error

def svg_to_png(svg_bytes):
    try:
        import cairosvg
        return cairosvg.svg2png(bytestring=svg_bytes,scale=2,background_color="white")
    except Exception: return None

def stamp_png(png_bytes,text):
    try:
        from PIL import Image,ImageDraw,ImageFont
        import io as _io
        img=Image.open(_io.BytesIO(png_bytes)).convert("RGBA")
        draw=ImageDraw.Draw(img); font=None
        for fp,sz in [("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",28),
                      ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",28)]:
            try: font=ImageFont.truetype(fp,sz); break
            except: pass
        if font is None: font=ImageFont.load_default()
        bbox=draw.textbbox((0,0),text,font=font,anchor="lt"); tw=bbox[2]-bbox[0]; th=bbox[3]-bbox[1]
        pad_x,pad_y=36,16; pill_w=tw+pad_x*2; pill_h=th+pad_y*2; pill_r=pill_h//2
        px=(img.width-pill_w)//2; py_=img.height-pill_h-28
        draw.rounded_rectangle([px,py_,px+pill_w,py_+pill_h],radius=pill_r,fill=(232,232,232,230))
        draw.text((px+pill_w//2,py_+pill_h//2),text,font=font,fill=(26,26,26,255),anchor="mm")
        buf=_io.BytesIO(); img.convert("RGB").save(buf,format="PNG"); return buf.getvalue()
    except Exception: return png_bytes

# ══════════════════════════════════════════════════════════════════════════════
# 2D INTERACTION DIAGRAM  (stub — full impl kept identical to v1)
# ══════════════════════════════════════════════════════════════════════════════

_CLR_HBOND="#1a7a1a"; _CLR_PIPI="#e200e8"; _CLR_HYDRO="#2287ff"
_CLR_IONIC="#aa0077"; _CLR_METAL="#cc8800"; _CLR_HAL="#cc2277"

_RES_CIRCLE={
    "hbond":dict(fill="#80dd80",opacity=0.2),
    "hbond_to_halogen":dict(fill="#80dd80",opacity=0.2),
    "pi_pi":dict(fill="#e200e8",opacity=0.2),
    "cation_pi":dict(fill="#e200e8",opacity=0.2),
    "hydrophobic":dict(fill="#2287ff",opacity=0.2),
    "ionic":dict(fill="#ffaae0",opacity=0.2),
    "metal":dict(fill="#ffe080",opacity=0.2),
    "halogen":dict(fill="#ffb0d0",opacity=0.2),
}
_LBL_CLR={"hbond":_CLR_HBOND,"hbond_to_halogen":_CLR_HBOND,"pi_pi":_CLR_PIPI,
           "cation_pi":_CLR_PIPI,"hydrophobic":_CLR_HYDRO,"ionic":_CLR_IONIC,
           "metal":_CLR_METAL,"halogen":_CLR_HAL}
_LINE_CLR={"hbond":_CLR_HBOND,"hbond_to_halogen":"#6633aa","pi_pi":_CLR_PIPI,
            "cation_pi":_CLR_PIPI,"ionic":_CLR_IONIC,"metal":_CLR_METAL,"halogen":_CLR_HAL}
_ATOM_CLR={"C":"#1a1a1a","N":"#1a5fa8","O":"#cc2222","S":"#c8a800","P":"#e07000",
            "F":"#1a7a1a","CL":"#1a7a1a","BR":"#8b2500","I":"#5c2d8a","H":"#555555"}
_AROM_DOT_CLR="#1a7a1a"

def _compute_svg_coords(mol2d,cx,cy,target_size=280):
    from rdkit.Chem import rdDepictor
    if mol2d.GetNumConformers()==0: rdDepictor.Compute2DCoords(mol2d)
    conf=mol2d.GetConformer(); n=mol2d.GetNumAtoms()
    if n==0: return {}
    xs=[conf.GetAtomPosition(i).x for i in range(n)]; ys=[conf.GetAtomPosition(i).y for i in range(n)]
    span=max(max(xs)-min(xs),max(ys)-min(ys),0.01); sc=target_size/span
    mx=(min(xs)+max(xs))/2; my=(min(ys)+max(ys))/2
    return {i:(cx+(xs[i]-mx)*sc,cy-(ys[i]-my)*sc) for i in range(n)}

def _ring_centroid_2d(ring_atom_indices,svg_coords):
    xs=[svg_coords[i][0] for i in ring_atom_indices if i in svg_coords]
    ys=[svg_coords[i][1] for i in ring_atom_indices if i in svg_coords]
    if not xs: return None,None
    return sum(xs)/len(xs),sum(ys)/len(ys)

def _place_residues_no_cross(interactions,svg_coords,cx,cy,R=210):
    if not interactions: return []
    items=[]
    for ix in interactions:
        ai=ix.get("lig_atom_idx",0); ax,ay=svg_coords.get(ai,(cx,cy))
        angle=_math.atan2(ay-cy,ax-cx)
        items.append({**ix,"angle":angle})
    items.sort(key=lambda x:x["angle"])
    n=len(items)
    if n==1:
        a=items[0]["angle"]
        return [{**items[0],"bx":cx+R*_math.cos(a),"by":cy+R*_math.sin(a),"slot_angle":a}]
    min_dist_px=78.0
    for _ in range(500):
        delta=[0.0]*n; any_overlap=False
        for i in range(n):
            for j in range(i+1,n):
                xi=cx+R*_math.cos(items[i]["angle"]); yi=cy+R*_math.sin(items[i]["angle"])
                xj=cx+R*_math.cos(items[j]["angle"]); yj=cy+R*_math.sin(items[j]["angle"])
                dx,dy=xj-xi,yj-yi; dist=_math.sqrt(dx*dx+dy*dy)
                if dist<min_dist_px:
                    push=( min_dist_px-dist)/max(R,1.0)
                    delta[i]-=push*0.5; delta[j]+=push*0.5; any_overlap=True
        for i in range(n): items[i]["angle"]+=delta[i]
        if not any_overlap: break
    result=[]
    for item in items:
        a=item["angle"]
        result.append({**item,"bx":cx+R*_math.cos(a),"by":cy+R*_math.sin(a),"slot_angle":a})
    return result

def _render_ligand_svg(mol2d,svg_coords):
    from rdkit import Chem
    parts=[]; ri=mol2d.GetRingInfo()
    arom_bonds=set(); arom_rings=[]
    for ring in ri.AtomRings():
        if all(mol2d.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            for k in range(len(ring)): arom_bonds.add(frozenset([ring[k],ring[(k+1)%len(ring)]]))
            arom_rings.append(ring)
    for bond in mol2d.GetBonds():
        i1,i2=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        x1,y1=svg_coords.get(i1,(0,0)); x2,y2=svg_coords.get(i2,(0,0))
        if frozenset([i1,i2]) in arom_bonds:
            parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#1a1a1a" stroke-width="1.77" opacity="0.9"/>')
        else:
            bt=bond.GetBondType()
            if bt==Chem.BondType.DOUBLE:
                dx,dy=x2-x1,y2-y1; L=_math.sqrt(dx*dx+dy*dy)+1e-9; px,py=-dy/L*2.5,dx/L*2.5
                for sg in (1,-1): parts.append(f'<line x1="{x1+px*sg:.2f}" y1="{y1+py*sg:.2f}" x2="{x2+px*sg:.2f}" y2="{y2+py*sg:.2f}" stroke="#1a1a1a" stroke-width="2.44" opacity="0.9"/>')
            else:
                parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#1a1a1a" stroke-width="1.77" opacity="0.9"/>')
    for ring in arom_rings:
        rcoords=[svg_coords.get(i,(0,0)) for i in ring]
        rcx=sum(x for x,y in rcoords)/len(rcoords); rcy=sum(y for x,y in rcoords)/len(rcoords)
        avg=sum(_math.sqrt((x-rcx)**2+(y-rcy)**2) for x,y in rcoords)/len(rcoords)
        cr=avg*0.58
        parts.append(f'<circle cx="{rcx:.2f}" cy="{rcy:.2f}" r="{cr:.2f}" fill="none" stroke="#1a1a1a" stroke-width="1.77" stroke-dasharray="5.43 2.72" opacity="0.7"/>')
        parts.append(f'<circle cx="{rcx:.2f}" cy="{rcy:.2f}" r="5.43" fill="{_AROM_DOT_CLR}"/>')
    for i in range(mol2d.GetNumAtoms()):
        atom=mol2d.GetAtomWithIdx(i); sym=atom.GetSymbol()
        if sym=="C": continue
        ax,ay=svg_coords.get(i,(0,0)); clr=_ATOM_CLR.get(sym.upper(),"#555"); fs=17.65
        hw={"H":7,"N":9,"O":9,"S":11,"P":11,"F":8,"CL":16,"BR":16,"I":11}.get(sym.upper(),9)
        parts.append(f'<rect x="{ax-hw:.1f}" y="{ay-11:.1f}" width="{hw*2:.0f}" height="22" fill="white" stroke="none"/>')
        parts.append(f'<text x="{ax:.2f}" y="{ay:.2f}" text-anchor="middle" dominant-baseline="central" font-family="Arial,sans-serif" font-size="{fs}" font-weight="700" fill="{clr}">{sym}</text>')
    return "".join(parts)

def draw_interaction_diagram(receptor_pdb,pose_sdf,smiles,title="",cutoff=4.5,size=(800,759),max_residues=14):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDepictor
    RDLogger.DisableLog("rdApp.*")
    W,H=size; cx,cy=W//2,H//2
    try:
        mol3d=None
        for san in (True,False):
            sup=Chem.SDMolSupplier(pose_sdf,sanitize=san,removeHs=False)
            mol3d=next((m for m in sup if m is not None),None)
            if mol3d is not None: break
        if mol3d is None: raise ValueError("No valid 3D pose")
        mol2d=Chem.MolFromSmiles(smiles.strip()) if smiles and smiles.strip() else None
        if mol2d is None:
            mol2d=Chem.RemoveHs(mol3d,sanitize=False)
            try: Chem.SanitizeMol(mol2d)
            except: pass
        mol2d=Chem.RemoveHs(mol2d)
        rdDepictor.Compute2DCoords(mol2d)
        try: raw=_detect_all_interactions(mol3d,receptor_pdb,cutoff=cutoff)
        except: raw=[]
        pm={t:i for i,t in enumerate(_ITYPE_PRIORITY)}
        ded=_deduplicate_interactions(raw)
        ded.sort(key=lambda x:(pm.get(x["itype"],99),x["distance"]))
        ded=ded[:max_residues]
        sc=_compute_svg_coords(mol2d,cx,cy,target_size=280)
        pl=_place_residues_no_cross(ded,sc,cx,cy,R=210)
        # render
        parts=[f'<svg width="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
               f'<rect width="{W}" height="{H}" fill="white"/>']
        if title:
            esc=title.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            tw=len(esc)*14.5+48; tw=max(tw,240); tw=min(tw,W-40)
            px=(W-tw)/2; ph=46; pr=ph/2
            parts.append(f'<rect x="{px:.1f}" y="14" width="{tw:.0f}" height="{ph}" rx="{pr:.1f}" ry="{pr:.1f}" fill="#f2f2f2" stroke="none"/>')
            parts.append(f'<text x="{W/2:.1f}" y="37" text-anchor="middle" dominant-baseline="central" font-family="Arial,sans-serif" font-size="24.93" font-weight="700" fill="#1a1a1a">{esc}</text>')
        for p in pl:
            bx,by=p["bx"],p["by"]; cbx=max(50,min(bx,W-50)); cby=max(70,min(by,H-65))
            bg=_RES_CIRCLE.get(p["itype"],dict(fill="#cccccc",opacity=0.2))
            parts.append(f'<circle cx="{cbx:.1f}" cy="{cby:.1f}" r="24.55" fill="{bg["fill"]}" opacity="{bg["opacity"]}"/>')
        for p in pl:
            itype=p["itype"]
            if itype=="hydrophobic": continue
            bx,by=p["bx"],p["by"]; cbx=max(50,min(bx,W-50)); cby=max(70,min(by,H-65))
            ai=p.get("lig_atom_idx",0); lx,ly=sc.get(ai,(cx,cy))
            clr=_LINE_CLR.get(itype,"#888")
            parts.append(f'<line x1="{lx:.2f}" y1="{ly:.2f}" x2="{cbx:.2f}" y2="{cby:.2f}" stroke="{clr}" stroke-width="1.6" stroke-dasharray="5 3" opacity="0.85"/>')
            if p.get("distance"):
                ds=f'{p["distance"]}\u00c5'; tw2=len(ds)*7+8
                mx2=(lx+cbx)/2; my2=(ly+cby)/2
                parts.append(f'<rect x="{mx2-tw2/2:.1f}" y="{my2-9:.1f}" width="{tw2:.0f}" height="17" rx="4" fill="white" stroke="{clr}" stroke-width="0.5"/>')
                parts.append(f'<text x="{mx2:.1f}" y="{my2:.1f}" text-anchor="middle" dominant-baseline="central" font-family="Arial,sans-serif" font-size="14" font-weight="700" fill="{clr}">{ds}</text>')
        parts.append(_render_ligand_svg(mol2d,sc))
        for p in pl:
            bx,by=p["bx"],p["by"]; cbx=max(50,min(bx,W-50)); cby=max(70,min(by,H-65))
            rn=p["resname"]; ri=p["resid"]; ch=p.get("chain","")
            _NO_NUM=HEME_RESNAMES|METAL_RESNAMES
            lbl=rn.upper() if rn.upper() in _NO_NUM else f"{rn.upper()} {ri}{ch}"
            lbl_clr=_LBL_CLR.get(p["itype"],"#333")
            parts.append(f'<text x="{cbx:.1f}" y="{cby:.1f}" text-anchor="middle" dominant-baseline="central" font-family="Arial,sans-serif" font-size="14.29" font-weight="700" fill="{lbl_clr}">{lbl}</text>')
        parts.append('</svg>')
        RDLogger.EnableLog("rdApp.error")
        return "\n".join(parts).encode()
    except Exception as e:
        RDLogger.EnableLog("rdApp.error")
        return (f'<svg viewBox="0 0 {W} 80" xmlns="http://www.w3.org/2000/svg">'
                f'<rect width="{W}" height="80" fill="white"/>'
                f'<text x="{W//2}" y="44" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#cc2222">'
                f'Error: {e}</text></svg>').encode()

def draw_interactions_rdkit(lig_mol,receptor_pdb,smiles,title="",cutoff=3.5,size=(500,500),max_residues=10):
    import tempfile
    from rdkit import Chem
    tmp=tempfile.NamedTemporaryFile(suffix=".sdf",delete=False)
    with Chem.SDWriter(tmp.name) as w: w.write(lig_mol)
    return draw_interaction_diagram(receptor_pdb=receptor_pdb,pose_sdf=tmp.name,smiles=smiles,
                                    title=title,cutoff=cutoff,size=(800,759),max_residues=max_residues)

def draw_interaction_diagram_data(receptor_pdb,pose_sdf,smiles,title="",cutoff=4.5,size=(800,759),max_residues=14):
    return None  # simplified stub — interactive mode not used in current deployment

def draw_interaction_diagram_interactive(receptor_pdb,pose_sdf,smiles,title="",cutoff=4.5,size=(800,759),max_residues=14):
    return "{}"

def diagnose_poseview():
    return {"server_reachable":False,"upload_ok":False,"poseview_ok":False,"log":[],"error":"Not implemented in stub"}
