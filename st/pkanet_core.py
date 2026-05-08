# core.py  —  pKaNET Cloud  (2026-05 overhaul)

from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TAUTOMER_PLAUSIBILITY_CUTOFF = 3.0
AMBIGUITY_SCORE_GAP          = 0.5
BORDERLINE_PKA_WINDOW        = 1.0
PUBCHEM_RATE_LIMIT_S         = 0.25
PUBCHEM_CACHE_FILE           = "/tmp/pkanet_pubchem_cache.json"
SEP = "=" * 70

W_AROM_RING_LOST         = 8.0
W_PHENOL_TO_KETO_FLIP    = 6.0
W_PYROGALLOL_TRIKETO     = 6.0
W_CATECHOL_DIKETO        = 4.0
W_PHENOL_PRESERVED_BONUS = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Optional dependency probes
# ─────────────────────────────────────────────────────────────────────────────
try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _requests = None; _REQUESTS_OK = False
    print("⚠️  requests not installed — PubChem lookup disabled.")

try:
    from dimorphite_dl import protonate_smiles as _dimorphite_fn
    _DIMORPHITE_OK = True
except ImportError:
    _dimorphite_fn = None; _DIMORPHITE_OK = False
    print("⚠️  dimorphite-dl not available.")

_PKASOLVER_OK = False; _PROPKA_OK = False; _UNIPKA_OK = False; _PKA_BACKEND = "none"

try:
    from pkasolver.query import QueryModel as _PkaSolverModel  # noqa
    _PKASOLVER_OK = True; _PKA_BACKEND = "pkasolver"; print("✅  pkasolver available.")
except ImportError:
    pass

if not _PKASOLVER_OK:
    try:
        import propka.run as _propka_run  # noqa
        _PROPKA_OK = True; _PKA_BACKEND = "propka"; print("✅  propka available.")
    except ImportError:
        pass

if not _PKASOLVER_OK and not _PROPKA_OK:
    if subprocess.run(["which", "unipka"], capture_output=True).returncode == 0:
        _UNIPKA_OK = True; _PKA_BACKEND = "unipka_cli"; print("✅  unipka CLI available.")

if _PKA_BACKEND == "none":
    print("ℹ️  No ML pKa backend — heuristic ionizable-site table will be used.")

# ─────────────────────────────────────────────────────────────────────────────
# Open Babel helper
# ─────────────────────────────────────────────────────────────────────────────
def check_obabel():
    return shutil.which("obabel") is not None

def convert_pdb_to_mol2_obabel(pdb_path, mol2_path):
    if not check_obabel(): return False
    try:
        r = subprocess.run(["obabel", pdb_path, "-O", mol2_path],
                           capture_output=True, text=True, timeout=30)
        return r.returncode == 0 and Path(mol2_path).exists()
    except Exception: return False

# ─────────────────────────────────────────────────────────────────────────────
# STAGE A  ·  RDKit standardization
# ─────────────────────────────────────────────────────────────────────────────
def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None, f"❌ RDKit cannot parse: {smiles[:80]}"
    mol = Chem.RemoveHs(mol, implicitOnly=True)
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    try: mol = rdMolStandardize.Normalizer().normalize(mol)
    except Exception: pass
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True), "OK"

def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

def smiles_to_inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try: return Chem.MolToInchiKey(mol)
    except Exception: return None

def enumerate_stereo(smiles, keep_original=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f"Bad SMILES: {smiles[:60]}")
    if keep_original:
        return [(Chem.MolToSmiles(mol, isomericSmiles=True), None)]
    opts = StereoEnumerationOptions(onlyUnassigned=False, unique=True)
    isos = list(EnumerateStereoisomers(mol, options=opts)) or [mol]
    rows = []
    for iso in isos:
        smi = Chem.MolToSmiles(iso, isomericSmiles=True)
        tag = None
        ch  = Chem.FindMolChiralCenters(iso, includeUnassigned=True)
        if len(ch) == 1 and ch[0][1] in ("R", "S"): tag = ch[0][1]
        rows.append((smi, tag))
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# STAGE B  ·  PubChem experimental pKa retrieval
# ─────────────────────────────────────────────────────────────────────────────
_PUBCHEM_CACHE = {}

def _load_pubchem_cache():
    global _PUBCHEM_CACHE
    if Path(PUBCHEM_CACHE_FILE).exists():
        try:
            with open(PUBCHEM_CACHE_FILE) as f: _PUBCHEM_CACHE = json.load(f)
        except Exception: _PUBCHEM_CACHE = {}

def _save_pubchem_cache():
    try:
        with open(PUBCHEM_CACHE_FILE, "w") as f: json.dump(_PUBCHEM_CACHE, f, indent=2)
    except Exception: pass

_load_pubchem_cache()

_PKA_PATTERNS = [
    re.compile(r"pK[aA][\w\s\(\)]*?=\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"([+-]?\d+(?:\.\d+)?)\s*\((?:pK[aA]|acid dissociation)[^)]*\)", re.IGNORECASE),
    re.compile(r"(?:pK[aA]).*?([+-]?\d+(?:\.\d+))", re.IGNORECASE),
]

def _pubchem_get(url, timeout=12):
    if not _REQUESTS_OK or _requests is None: return None
    try:
        time.sleep(PUBCHEM_RATE_LIMIT_S)
        r = _requests.get(url, timeout=timeout)
        if r.status_code == 200: return r.json()
    except Exception: pass
    return None

def pubchem_cid_from_inchikey(inchikey):
    key = f"cid:{inchikey}"
    if key in _PUBCHEM_CACHE: return _PUBCHEM_CACHE[key]
    data = _pubchem_get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/cids/JSON")
    cid = None
    if data:
        try: cid = int(data["IdentifierList"]["CID"][0])
        except Exception: pass
    _PUBCHEM_CACHE[key] = cid; _save_pubchem_cache(); return cid

def _flatten_pubchem_section(section, target_heading):
    results = []
    if target_heading.lower() in section.get("TOCHeading", "").lower():
        for info in section.get("Information", []):
            for swm in info.get("Value", {}).get("StringWithMarkup", []):
                s = swm.get("String", "").strip()
                if s: results.append(s)
    for sub in section.get("Section", []): results.extend(_flatten_pubchem_section(sub, target_heading))
    return results

def pubchem_get_dissociation_texts(cid):
    key = f"diss:{cid}"
    if key in _PUBCHEM_CACHE: return _PUBCHEM_CACHE[key]
    data = _pubchem_get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=Dissociation+Constants")
    texts = []
    if data:
        try:
            for sec in data.get("Record", {}).get("Section", []):
                texts.extend(_flatten_pubchem_section(sec, "Dissociation"))
        except Exception: pass
    _PUBCHEM_CACHE[key] = texts; _save_pubchem_cache(); return texts

def parse_pka_values(texts):
    full_text = " ".join(texts).lower()
    found = []; src = []
    for text in texts:
        hits = []
        for pat in _PKA_PATTERNS:
            for m in pat.finditer(text):
                try:
                    v = float(m.group(1))
                    if -5.0 <= v <= 20.0: hits.append(v)
                except ValueError: pass
        if hits: found.extend(hits); src.append(text)
    dedup = []
    for v in found:
        if not any(abs(v - e) < 0.05 for e in dedup): dedup.append(v)
    site_labels = bool(re.search(r"pK[aA]\s*[12\(]", " ".join(texts)))
    temperature = bool(re.search(r"\d+\s*°\s*[Cc]|at\s+\d+\s*[Cc]", full_text))
    solvent     = bool(re.search(r"\b(water|aqueous|etoh|dmso|methanol|buffer|solution)\b", full_text))
    vague       = bool(re.search(r"\b(approximately|approx|about|ca\.|around|range|varies|estimated|uncertain|unclear|conflicting)\b", full_text))
    conflicting = len(dedup) >= 2 and any(abs(a-b)>1.5 for i,a in enumerate(dedup) for b in dedup[i+1:])
    if not dedup or conflicting or vague: confidence = "low"
    elif len(dedup) > 1 or temperature or solvent or site_labels: confidence = "medium"
    else: confidence = "high"
    flags = {"exact_numeric_match": bool(dedup), "multiple_values_found": len(dedup)>1,
             "site_labels_found": site_labels, "temperature_mentioned": temperature,
             "solvent_mentioned": solvent, "conflicting_values": conflicting,
             "vague_or_approximate": vague, "unclear_site_mapping": len(dedup)>1, "confidence": confidence}
    return dedup, src, flags

def pubchem_lookup(smiles):
    result = dict(available=False, cid=None, inchikey=None, pka_values=[], source_texts=[], flags={}, confidence="low", error=None)
    ik = smiles_to_inchikey(smiles)
    if ik is None: result["error"] = "InChIKey computation failed."; return result
    result["inchikey"] = ik
    cid = pubchem_cid_from_inchikey(ik)
    if cid is None: result["error"] = "CID not found."; return result
    result["cid"] = cid
    texts = pubchem_get_dissociation_texts(cid)
    if not texts: result["error"] = "No dissociation constant data on PubChem."; return result
    vals, srcs, flags = parse_pka_values(texts)
    result.update(available=bool(vals), pka_values=vals, source_texts=srcs, flags=flags, confidence=flags.get("confidence","low"))
    return result

# ─────────────────────────────────────────────────────────────────────────────
# STAGE C  ·  ML pKa backends
# ─────────────────────────────────────────────────────────────────────────────
def _unipka_via_pkasolver(smiles):
    try:
        from pkasolver.query import QueryModel
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return []
        df = QueryModel().predict_pka(mol)
        return [{"pka": float(row.get("pKa", row.get("pka", 0))), "site_type": str(row.get("type","?")),
                 "site_label": str(row.get("atom_idx","?")), "source": "pkasolver", "confidence": "ml_gnn"}
                for _, row in df.iterrows()]
    except Exception as e: print(f"⚠️  pkasolver failed: {e}"); return []

def _unipka_via_propka(smiles):
    try:
        import propka.run as pk
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return []
        mol = Chem.AddHs(mol)
        p = AllChem.ETKDGv3(); p.randomSeed = 42
        if AllChem.EmbedMolecule(mol, p) != 0: return []
        AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tf:
            tmppath = tf.name; tf.write(Chem.MolToPDBBlock(mol))
        results = []
        try:
            mc = pk.single(tmppath, optargs=["--quiet"])
            for grp in mc.conformations[0].groups:
                pv = getattr(grp, "pka_value", None)
                if pv is not None:
                    results.append({"pka": float(pv), "site_label": str(getattr(grp,"atom_name","?")),
                                    "site_type": str(getattr(grp,"type","?")), "source": "propka", "confidence": "semi_empirical"})
        finally:
            try: os.unlink(tmppath)
            except Exception: pass
        return results
    except Exception as e: print(f"⚠️  propka failed: {e}"); return []

def _unipka_via_cli(smiles):
    try:
        r = subprocess.run(["unipka","--smiles",smiles,"--json"], capture_output=True, text=True, timeout=60)
        if r.returncode != 0: return []
        data = json.loads(r.stdout)
        return [{"pka": e.get("pka"), "site_label": e.get("site","?"), "site_type": e.get("type","?"),
                 "source": "unipka_cli", "confidence": "ml"} for e in data.get("microstates",[])]
    except Exception as e: print(f"⚠️  unipka CLI failed: {e}"); return []

def unipka_predict(smiles):
    if _UNIPKA_OK:
        r = _unipka_via_cli(smiles)
        if r: return r
    if _PKASOLVER_OK:
        r = _unipka_via_pkasolver(smiles)
        if r: return r
    if _PROPKA_OK:
        r = _unipka_via_propka(smiles)
        if r: return r
    return []

def unipka_summary_pka(predictions):
    valid = [p for p in predictions if p.get("pka") is not None]
    if not valid: return None, "none"
    closest = min(valid, key=lambda p: abs(float(p["pka"]) - 7.4))
    return float(closest["pka"]), closest.get("source", "?")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE D  ·  Dimorphite-DL protonation enumerator
# ─────────────────────────────────────────────────────────────────────────────
def dimorphite_enumerate(smiles, ph_min, ph_max, precision=1.0, max_variants=128):
    if not _DIMORPHITE_OK or _dimorphite_fn is None: return [smiles]
    kwarg_variants = [
        {"ph_min": ph_min, "ph_max": ph_max, "precision": precision, "max_variants": max_variants},
        {"min_ph": ph_min, "max_ph": ph_max, "pka_precision": precision, "max_variants": max_variants},
        {"ph_min": ph_min, "ph_max": ph_max, "precision": precision},
        {"min_ph": ph_min, "max_ph": ph_max, "pka_precision": precision},
    ]
    errors = []; raw = []
    for kwargs in kwarg_variants:
        try:
            r = _dimorphite_fn(smiles, **kwargs)
            raw = [r] if isinstance(r, str) else list(r or [])
            if raw: break
        except TypeError as e: errors.append(str(e))
    if not raw:
        try:
            sig = inspect.signature(_dimorphite_fn); kw = {}
            for name in sig.parameters:
                lo = name.lower()
                if lo in {"ph_min","min_ph"}: kw[name] = ph_min
                elif lo in {"ph_max","max_ph"}: kw[name] = ph_max
                elif lo in {"precision","pka_precision"}: kw[name] = precision
                elif lo == "max_variants": kw[name] = max_variants
            r = _dimorphite_fn(smiles, **kw)
            raw = [r] if isinstance(r, str) else list(r or [])
        except Exception as e:
            errors.append(str(e))
            print(f"⚠️  dimorphite-dl failed ({smiles[:50]}). Errors: {errors[-2:]}")
    seen = set(); result = []
    seed = canonicalize(smiles)
    if seed: seen.add(seed); result.append(seed)
    for smi in raw:
        c = canonicalize(smi)
        if c and c not in seen: seen.add(c); result.append(c)
    return result or [smiles]

# ─────────────────────────────────────────────────────────────────────────────
# STAGE E+F  ·  HH scoring + ionizable site table
# ─────────────────────────────────────────────────────────────────────────────
def hh_fraction_charged(pka, ph, site_type):
    if site_type == "acid": return 1.0 / (1.0 + 10.0 ** (pka - ph))
    return 1.0 / (1.0 + 10.0 ** (ph - pka))

def hh_ph_match_score(pka, ph, site_type, actual_charge):
    f_charged = hh_fraction_charged(pka, ph, site_type)
    dpH = abs(ph - pka)
    decisive = (f_charged >= 0.65) or (f_charged <= 0.35)
    rwd_mul = pen_mul = 1.6 if decisive else 1.0
    if site_type == "acid":
        expected_neg = f_charged > 0.5
        if expected_neg and actual_charge < 0:  return  min(1.5, dpH * 0.55 * rwd_mul) + 0.15
        elif expected_neg:                       return -min(1.5, dpH * 0.45 * pen_mul) - 0.15
        elif actual_charge >= 0:                 return  0.15
        else:                                    return -min(1.5, dpH * 0.45 * pen_mul) - 0.15
    else:
        expected_pos = f_charged > 0.5
        if expected_pos and actual_charge > 0:   return  min(1.5, dpH * 0.55 * rwd_mul) + 0.15
        elif expected_pos:                       return -min(1.5, dpH * 0.45 * pen_mul) - 0.15
        elif actual_charge <= 0:                 return  0.15
        else:                                    return -min(1.5, dpH * 0.45 * pen_mul) - 0.15

# ─── Ionizable site table ────────────────────────────────────────────────────
_IONIZABLE_SITE_DEF = [
    # ── Ultra-strong acids ────────────────────────────────────────────────────
    ("sulfonic_acid",              "[SX4](=O)(=O)[OX2H1]",                                1.0,  "acid"),
    # NEW: sulfonyl-imide N-H (saccharin pKa=1.6, acesulfame-K pKa~2).
    # N between C=O AND SO2 — far more acidic than plain sulfonamide (pKa=10.1).
    # MUST precede sulfonamide_NH so seen_ion dedup assigns the correct pKa.
    ("sulfonyl_imide_NH",          "[CX3](=O)[NX3;H1][SX4](=O)(=O)",                      2.0,  "acid"),
    # ── Carboxylic / aromatic hetero-acid ─────────────────────────────────────
    # Alpha-amino acid carboxyl: NH2 alpha to COOH suppresses pKa to ~2.3
    # Recursive: checks NH2 without including N in match → amine site stays free
    ("amino_acid_COOH",            "[OX2H1][CX3](=O)[$([CX4][NX3;H2;!$(NC=O)])]",   2.3,  "acid"),
    # Aryl carboxylic acid: benzoic=4.2, salicylic=2.97 (phenol handles separately)
    ("aryl_carboxylic_acid",       "[c][CX3](=O)[OX2H1]",                             4.2,  "acid"),
    ("carboxylic_acid",            "[CX3](=O)[OX2H1]",                                4.5,  "acid"),
    ("tetrazole",                  "c1nn[nH]n1",                                          4.9,  "acid"),
    # ── Phosphorus acids (diprotic handled by Pass 1 in find_ionizable_sites) ─
    ("phosphonate_fallback",       "[PX4](=O)([OX2H1])[OX1-,OX2;!$([OX2H1])]",           6.5,  "acid"),
    ("phosphate_monoester_fb",     "[PX4](=O)([OX2H1])([OX2,OX1-])[OX2,OX1-]",           6.1,  "acid"),
    # ── N-H acids ─────────────────────────────────────────────────────────────
    ("sulfonamide_NH",             "[SX4](=O)(=O)[NX3;H1,H2]",                            10.1, "acid"),  # H2 for primary sulfonamide
    ("imide_NH",                   "[CX3](=O)[NX3;H1][CX3]=O",                            9.6,  "acid"),
    ("acylhydrazone_NH",           "[CX3](=O)[NX3;H1][NX2]=[CX3]",                        10.5, "acid"),
    ("hydrazide_NH",               "[CX3](=O)[NX3;H1][NX3;H2]",                           10.5, "acid"),
    ("urea_NH",                    "[NX3;H1][CX3](=O)[NX3;H1,H2]",                        13.0, "acid"),
    ("amide_NH",                   "[CX3](=O)[NX3;H1,H2;!$([N]~N)]",                      15.0, "acid"),
    # ── Hydroxamic acid (Bug C fix) ───────────────────────────────────────────
    # Recursive SMARTS captures only the O-H; prevents amine-N from being
    # mis-claimed as acid site at pKa=9.0.
    ("hydroxamic_acid",            "[OX2H1;$([OX2H1][NX3;H1][CX3](=O))]",                9.0,  "acid"),
    # ── Aromatic N-H acids (Bug #1/#2 fix: was 6.0/5.5 = BASE pKa, wrong!) ──
    ("benzimidazole_NH",           "c1ccc2[nH]cnc2c1",                                    13.0, "acid"),
    ("imidazole_NH",               "[nH]1ccnc1",                                          14.0, "acid"),
    ("pyrazole_NH",                "[nH]1nccc1",                                          14.0, "acid"),
    ("indole_NH",                  "c1ccc2[nH]ccc2c1",                                    17.0, "acid"),
    # ── Enol acids (NEW) ──────────────────────────────────────────────────────
    # Enol-lactone: ascorbic acid, dehydroascorbate precursors. C=C-OH adjacent
    # to lactone ring C=O. Must precede generic phenol (pKa=10.0).
    ("enol_lactone",               "[OX2H1][CX3]=[CX3][CX3](=O)[OX2;R]",                 4.2,  "acid"),
    # Cyclic 1,3-dicarbonyl enol: pyrazolidinedione (phenylbutazone pKa~4.5),
    # dimedone, cyclopentane-1,3-dione type ring enols.
    # Aromatic cyclic enol-ketone: phenylbutazone enol (hydroxypyrazolone) pKa~4.5.
    ("enol_cyclic_dicarbonyl_arom",  "[OX2H1][c;R]~[c;R]~[c;R]=O",                        5.0,  "acid"),
    # Non-aromatic cyclic 1,3-dicarbonyl enol: dimedone, cyclopentanedione type.
    ("enol_cyclic_dicarbonyl",       "[OX2H1][CX3;R]=[CX3;R][CX3;R]=O",                   5.5,  "acid"),
    # Open-chain 1,3-dicarbonyl enol: acetylacetone (pKa~8.9), ethyl acetoacetate.
    ("enol_1_3_dicarbonyl",        "[OX2H1][CX3]=[CX3][CX3]=O",                           9.0,  "acid"),
    # ── Phenols (Bug F fix: catechol_OH before phenol_ortho_CO) ──────────────
    ("catechol_OH",                "[OX2H1][c;R]:[c;R][OX2H1]",                           9.2,  "acid"),
    ("phenol_ortho_CO",            "[OX2H1][c;R]:[c;R][CX3;R](=O)",                       7.8,  "acid"),
        ("phenol_para_EWG",            "[OX2H1]c1ccc([$([NX3+](=O)[O-]),$([NX3](=O)=O),$([CX3;!$(C(=O)O)](=O)),$(C#N),$([SX4](=O)(=O))])cc1", 7.2, "acid"),  # para-EWG (excl. COOH)
    ("phenol_EWG",                 "[OX2H1][c;R]:[c;R][$([NX3+](=O)[O-]),$([NX3](=O)=O),$([CX3;!$(C(=O)O)](=O)),$(C#N),$([SX4](=O)(=O))]", 7.2, "acid"),  # excl. COOH
    ("phenol",                     "c[OX2H1]",                                            10.0, "acid"),
    # ── Thiols ────────────────────────────────────────────────────────────────
    # Bug B fix: Cys-like thiol alpha to amine pKa~8.3; recursive SMARTS.
    ("thiol_alpha_amino",          "[SX2H1;$([SX2H1][CX4][CX4][NX3;!$(NC=O)])]",              8.3,  "acid"),
    ("thiol_arom",                 "c[SX2H1]",                                            6.5,  "acid"),
    ("thiol_aliph",                "[CX4][SX2H1]",                                        9.8,  "acid"),
    # ── Bases ─────────────────────────────────────────────────────────────────
    ("aniline",                    "c[NX3;H1,H2;!$(N~[!#6])]",                            4.6,  "base"),
    ("pyridine_like",              "[$([nX2]1:[c,n]:c:[c,n]:c1),$([nX2]:c:n)]",           5.2,  "base"),
    ("aliphatic_imine",            "[CX3;!$([c])](=[NX2;H0;!$([n])])",                    5.5,  "base"),
    # Bug G fix: alpha-EWG amine pKa~7.5; must precede generic aliphatic_amine.
    ("amine_alpha_EWG",            "[NX3;H1,H2;!$(NC=O);!$([nH]);$([NX3][CX4][$([CX3;!$(C(=O)O)](=O)),$([CX3]=S),$(C#N),$([SX4](=O)(=O))])]", 7.5, "base"),
    ("aliphatic_amine",            "[NX3;H1,H2;!$(NC=O);!$(N~[!#6;!H]);!$([nH]);!$([NX3][CX3](=[NX2])[NX3])]",        9.5,  "base"),
    ("aliphatic_amine_t",          "[NX3;H0;!$(NC=O);!$(Nc);!$([nH]);!$([N]~[!#6]);!$([NX3]([CX4][CX3]=O)[CX4][CX3]=O)]",    9.0,  "base"),
    ("amidine",                    "[CX3](=[NX2;H0,H1])[NX3;H1,H2;!$([NX3][CX3](=[NX2])[NX3])]",                     12.4, "base"),
    ("guanidine",                  "[NX2;H1;$([NX2]=[CX3]([NX3])[NX3])]",                  12.5, "base"),  # imine =NH, bias-corrected
]

_IONIZABLE_SITES_COMPILED = []
for _lbl, _sma, _pka_v, _typ in _IONIZABLE_SITE_DEF:
    _pat = Chem.MolFromSmarts(_sma)
    if _pat is not None: _IONIZABLE_SITES_COMPILED.append((_lbl, _pat, _pka_v, _typ))
    else: print(f"⚠️  SMARTS compile failed: {_lbl}")

# ─── Diprotic phosphorus acid handler (Bug A fix) ────────────────────────────
_DIPROTIC_P_DEFS = [
    ("[PX4](=O)([OX2H1])[OX2H1]",                         2.1, 7.0, "phosphonate"),
    ("[PX4](=O)([OX2H1])([OX2H1])[OX2;!$([OX2H1])]",     1.0, 6.1, "phosphate_monoester"),
]
_DIPROTIC_P_COMPILED = []
for _sma_dp, _pk1, _pk2, _lbl_dp in _DIPROTIC_P_DEFS:
    _pat_dp = Chem.MolFromSmarts(_sma_dp)
    if _pat_dp is not None: _DIPROTIC_P_COMPILED.append((_pat_dp, _pk1, _pk2, _lbl_dp))
    else: print(f"⚠️  Diprotic SMARTS compile failed: {_lbl_dp}")

# ─── Tautomer plausibility scoring ───────────────────────────────────────────
_BONUS_DEF = [
    ("amide",            +2.5, "[CX3](=O)[NX3;H1,H2]"),
    ("lactam",           +2.5, "[C;R](=O)[N;R]"),
    ("acylhydrazone_NH", +2.0, "[CX3](=O)[NX3;H1][NX2]=[CX3]"),
    ("hydrazide_NH",     +2.0, "[CX3](=O)[NX3;H1][NX3;H2]"),
    ("urea_NH",          +1.5, "[NX3;H1][CX3](=O)[NX3;H1,H2]"),
    ("thioamide",        +1.0, "[CX3](=S)[NX3;H1,H2]"),
    ("aromatic_ring",    +0.3, "c1ccccc1"),
    ("phenol_preserved", W_PHENOL_PRESERVED_BONUS, "c[OX2H1]"),
    # NEW: 1,3-dicarbonyl enol bonus (counteracts enol_simple penalty for these)
    ("enol_1_3_dicarbonyl_bonus", +1.5, "[OX2H1][CX3]=[CX3][CX3]=O"),
]
_PENALTY_DEF = [
    ("imidic_acid_open",  -4.0, "[CX3;!R](=[NX2])[OX2H1]"),
    ("lactim_ring",       -4.0, "[C;R](=[NX2])[OX2H1]"),
    ("iminol_general",    -3.5, "[NX2]=[CX3][OX2H1]"),
    ("amide_N_deproton",  -5.0, "[$([NX3-]C=O),$([NX3-]c=O)]"),
    ("enol_simple",       -1.2, "[CX3](=[CX3])[OX2H1]"),
    ("pyrogallol_triketo",-W_PYROGALLOL_TRIKETO, "[#6;!a;R]1(=O)[#6;!a;R](=O)[#6;!a;R](=O)[#6;R][#6;R][#6;R]1"),
    ("catechol_diketo",   -W_CATECHOL_DIKETO,    "[#6;!a;R]1(=O)[#6;!a;R](=O)[#6;R][#6;R][#6;R][#6;R]1"),
    ("ring_carbonyl_onaromring_former", -3.0, "[#6;!a;R](=O)[#6;!a;R]=[#6;!a;R]"),
]
_CHEM_RULES = []
for _lbl, _wt, _sma in _BONUS_DEF + _PENALTY_DEF:
    _pat = Chem.MolFromSmarts(_sma)
    if _pat is not None: _CHEM_RULES.append((_lbl, _wt, _pat))
    else: print(f"⚠️  SMARTS compile failed: {_lbl}")

_TAUTOMER_RICH_DEF = [
    ("imidazole",    "[nH]1ccnc1"),
    ("benzimidazole","c1ccc2[nH]cnc2c1"),
    ("tetrazole",    "c1nn[nH]n1"),
    ("triazole",     "[nH]1ccnn1"),
    ("pyridone",     "[OH]c1ccccn1"),
    ("keto_enol",    "[CX4][CX3](=O)[CX4]"),
    ("purine",       "c1ncnc2[nH]cnc12"),
]
_TAUTOMER_RICH_COMPILED = [(lbl,pat) for lbl,sma in _TAUTOMER_RICH_DEF if (pat := Chem.MolFromSmarts(sma)) is not None]

def _n_aromatic_rings(mol):
    if mol is None: return 0
    try: return int(rdMolDescriptors.CalcNumAromaticRings(mol))
    except Exception: return 0

def _count_phenolic_OH(mol):
    if mol is None: return 0
    patt = Chem.MolFromSmarts("c[OX2H1]")
    return len(mol.GetSubstructMatches(patt)) if patt else 0

def score_tautomer_plausibility(smiles, ref_mol=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return -999.0, {}
    bd = {}; total = 0.0
    for lbl, wt, pat in _CHEM_RULES:
        n = len(mol.GetSubstructMatches(pat))
        if n: c = wt * n; bd[lbl] = round(c, 3); total += c
    if ref_mol is not None:
        rings_lost = max(0, _n_aromatic_rings(ref_mol) - _n_aromatic_rings(mol))
        if rings_lost > 0:
            pen = -W_AROM_RING_LOST * rings_lost
            bd["arom_ring_lost_vs_input"] = round(pen, 3); total += pen
        phenols_lost = max(0, _count_phenolic_OH(ref_mol) - _count_phenolic_OH(mol))
        if phenols_lost > 0:
            pen = -W_PHENOL_TO_KETO_FLIP * phenols_lost
            bd["phenol_flipped_to_keto"] = round(pen, 3); total += pen
    bd["_total"] = round(total, 3)
    return total, bd

def is_tautomer_rich(mol):
    hits = [l for l, p in _TAUTOMER_RICH_COMPILED if mol.HasSubstructMatch(p)]
    return bool(hits), hits

def enumerate_and_filter_tautomers(smiles, max_states=8, cutoff=TAUTOMER_PLAUSIBILITY_CUTOFF):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f"Bad SMILES: {smiles[:60]}")
    ref_mol = mol
    tr_flag, tr_motifs = is_tautomer_rich(mol)
    enum = rdMolStandardize.TautomerEnumerator()
    seen = set(); scored = []
    input_canon = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    seen.add(input_canon)
    sc0, bd0 = score_tautomer_plausibility(input_canon, ref_mol=ref_mol)
    scored.append({"smiles": input_canon, "score": sc0, "breakdown": bd0})
    for tmol in enum.Enumerate(mol):
        smi = Chem.MolToSmiles(tmol, isomericSmiles=True, canonical=True)
        if smi in seen: continue
        seen.add(smi)
        sc, bd = score_tautomer_plausibility(smi, ref_mol=ref_mol)
        scored.append({"smiles": smi, "score": sc, "breakdown": bd})
    if not scored:
        smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        sc, bd = score_tautomer_plausibility(smi, ref_mol=ref_mol)
        scored = [{"smiles": smi, "score": sc, "breakdown": bd}]
    scored = sorted(scored, key=lambda x: -x["score"])[:max_states]
    best = scored[0]["score"]
    eff_cutoff = cutoff * (2.0 if tr_flag else 1.0)
    kept = [t for t in scored if t["score"] >= best - eff_cutoff]
    discarded = [t for t in scored if t["score"] < best - eff_cutoff]
    return kept or [scored[0]], discarded, tr_flag, tr_motifs

# ─────────────────────────────────────────────────────────────────────────────
# Flavonoid A-ring phenols (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def _detect_chromone_system(mol):
    ring_info = mol.GetRingInfo()
    rings = [set(r) for r in ring_info.AtomRings() if len(r) == 6]
    if not rings: return set()
    def _has_exocyclic_carbonyl(atom_idx):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() != "C": return False
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetSymbol() != "O" or other.IsInRing(): continue
            bo = bond.GetBondTypeAsDouble()
            if bo == 2.0: return True
            if bo == 1.5 and other.GetTotalNumHs() == 0 and other.GetDegree() == 1: return True
        return False
    pyrone_rings = []
    for ring in rings:
        ring_os  = [i for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == "O"]
        ring_cos = [i for i in ring if _has_exocyclic_carbonyl(i)]
        if len(ring_os) == 1 and len(ring_cos) >= 1: pyrone_rings.append(ring)
    if not pyrone_rings: return set()
    system_atoms = set()
    for py in pyrone_rings:
        system_atoms.update(py)
        for other in rings:
            if other is py: continue
            if len(py & other) >= 2: system_atoms.update(other)
    return system_atoms

def _find_flavone_A_ring_phenols(mol):
    chromone_atoms = _detect_chromone_system(mol)
    if not chromone_atoms: return []
    ring_carbonyl_idx = ring_oxygen_idx = None
    for idx in chromone_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "C":
            for bond in atom.GetBonds():
                other = bond.GetOtherAtom(atom)
                if (other.GetSymbol() == "O" and not other.IsInRing() and
                        bond.GetBondTypeAsDouble() in (2.0, 1.5) and
                        other.GetTotalNumHs() == 0 and other.GetDegree() == 1):
                    ring_carbonyl_idx = idx; break
        elif atom.GetSymbol() == "O" and atom.IsInRing():
            ring_oxygen_idx = idx
    def _nbrs(idx): return [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetIdx() in chromone_atoms]
    def _has_phenolic_OH(c_idx):
        for bond in mol.GetAtomWithIdx(c_idx).GetBonds():
            other = bond.GetOtherAtom(mol.GetAtomWithIdx(c_idx))
            if (other.GetSymbol() == "O" and other.GetTotalNumHs() >= 1 and
                    other.GetDegree() == 1 and bond.GetBondTypeAsDouble() == 1.0 and not other.IsInRing()):
                return True
        return False
    candidates = []
    for atom in mol.GetAtoms():
        c_idx = atom.GetIdx()
        if c_idx not in chromone_atoms or atom.GetSymbol() != "C" or not atom.GetIsAromatic(): continue
        if c_idx == ring_carbonyl_idx: continue
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if (other.GetSymbol() == "O" and other.GetTotalNumHs() >= 1 and
                    other.GetDegree() == 1 and bond.GetBondTypeAsDouble() == 1.0 and not other.IsInRing()):
                candidates.append((c_idx, other.GetIdx())); break
    sites = []
    for c_idx, o_idx in candidates:
        chromone_nbrs = _nbrs(c_idx)
        ortho_carbons = [n for n in chromone_nbrs if mol.GetAtomWithIdx(n).GetSymbol() == "C"]
        ortho_to_carbonyl = carbonyl_direct = False
        if ring_carbonyl_idx is not None:
            if ring_carbonyl_idx in chromone_nbrs:
                ortho_to_carbonyl = carbonyl_direct = True
            else:
                for nb in chromone_nbrs:
                    if any(n.GetIdx() == ring_carbonyl_idx for n in mol.GetAtomWithIdx(nb).GetNeighbors()):
                        ortho_to_carbonyl = True; break
        ortho_to_ring_O = ring_oxygen_idx is not None and ring_oxygen_idx in chromone_nbrs
        n_ortho_phenols = sum(1 for n in ortho_carbons if _has_phenolic_OH(n))
        if ortho_to_carbonyl:
            label, pka = ("flavone_3OH_flavonol", 9.0) if carbonyl_direct else ("flavone_5OH_chelated", 11.0)
        elif ortho_to_ring_O:
            label, pka = "flavone_8OH_ortho_pyranO", 8.5
        elif n_ortho_phenols >= 2:
            label, pka = "flavone_6OH_pyrogallol_center", 8.5
        elif n_ortho_phenols == 1:
            label, pka = "flavone_phenol_catechol_pair", 7.5  # 7.0→7.5: neutral at pH 7.4
        else:
            label, pka = "flavone_phenol_isolated", 7.5  # 7.0→7.5: neutral at pH 7.4
        sites.append({"label": label, "atom_indices": [o_idx, c_idx], "heuristic_pka": pka, "site_type": "acid"})
    if sites:
        detail = ", ".join(f"{s['label'].replace('flavone_','')}(pKa={s['heuristic_pka']})" for s in sites)
        print(f"    🌸  Detected {len(sites)} flavonoid A-ring phenol(s): {detail}")
    return sites

def find_ionizable_sites(mol):
    sites = []; seen_ion = set(); claimed_atoms = set()
    # Pass 0: Flavonoid A-ring phenols
    for site in _find_flavone_A_ring_phenols(mol):
        ion_idx = site.get("atom_indices", [None])[0]
        if ion_idx in seen_ion: continue
        seen_ion.add(ion_idx); claimed_atoms.update(site["atom_indices"]); sites.append(site)

    # ── Special handlers: biguanide + guanidine (avoid N over-counting) ──────
    # Biguanide (Metformin): one dominant +1 protonation, not independent 3×
    _PAT_BIGUANIDE_     = Chem.MolFromSmarts("[#7][#6](=[#7])[#7][#6](=[#7])[#7]")
    if _PAT_BIGUANIDE_ is not None:
        for match in mol.GetSubstructMatches(_PAT_BIGUANIDE_):
            n_atoms = [i for i in match if mol.GetAtomWithIdx(i).GetAtomicNum() == 7]
            if not any(i in claimed_atoms for i in n_atoms):
                free_n = [i for i in n_atoms if mol.GetAtomWithIdx(i).GetFormalCharge() == 0]
                if free_n:
                    for i in n_atoms: seen_ion.add(i)
                    claimed_atoms.update(match)
                    sites.append(dict(label="biguanide",
                                      atom_indices=free_n,
                                      heuristic_pka=12.4, site_type="base"))
                    break

    # Guanidine (Arginine): one +1 per guanidine group, all N atoms claimed
    _PAT_GUANIDINE_FULL_ = Chem.MolFromSmarts("[#7][#6](=[#7])[#7]")
    if _PAT_GUANIDINE_FULL_ is not None:
        for match in mol.GetSubstructMatches(_PAT_GUANIDINE_FULL_):
            if any(i in claimed_atoms for i in match): continue
            n_atoms = [i for i in match if mol.GetAtomWithIdx(i).GetAtomicNum() == 7]
            free_n = [i for i in n_atoms if mol.GetAtomWithIdx(i).GetFormalCharge() == 0]
            if free_n:
                for i in n_atoms: seen_ion.add(i)
                claimed_atoms.update(match)
                sites.append(dict(label="guanidine_full",
                                  atom_indices=free_n,
                                  heuristic_pka=13.0, site_type="base"))
                break

    # Global nitrophenol: catches ortho, meta, para NO2-phenol
    _PAT_NITROPHENOL_ = Chem.MolFromSmarts("[OX2H1]c1ccccc1[$([NX3+](=O)[O-]),$([NX3](=O)=O)]")
    if _PAT_NITROPHENOL_ is None:
        _PAT_NITROPHENOL_ = Chem.MolFromSmarts("[OX2H1][c;R]~[c;R]~[c;R]~[c;R][$([NX3+](=O)[O-])]")
    if _PAT_NITROPHENOL_ is not None:
        for match in mol.GetSubstructMatches(_PAT_NITROPHENOL_):
            oh = match[0]
            if oh not in seen_ion and oh not in claimed_atoms:
                seen_ion.add(oh); claimed_atoms.add(oh)
                sites.append(dict(label="nitrophenol_global", atom_indices=[oh],
                                  heuristic_pka=7.1, site_type="acid"))

    # Thiazide sulfonamide: primary aryl sulfonamide in S+N ring → pKa~6.8
    _PAT_THIAZIDE_SULFA_ = Chem.MolFromSmarts("[NX3;H2][SX4](=O)(=O)[c]")
    has_sn_ring = any(16 in [mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring]
                      and 7 in [mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring]
                      for ring in mol.GetRingInfo().AtomRings())
    if _PAT_THIAZIDE_SULFA_ is not None and has_sn_ring:
        for match in mol.GetSubstructMatches(_PAT_THIAZIDE_SULFA_):
            nidx = match[0]
            if nidx not in seen_ion and nidx not in claimed_atoms:
                seen_ion.add(nidx); claimed_atoms.update(match)
                sites.append(dict(label="thiazide_sulfonamide_NH",
                                  atom_indices=[nidx], heuristic_pka=6.8, site_type="acid"))

    # Thioxo aromatic (6-mercaptopurine in thione form): use aromatic N-H as proxy
    _PAT_THIOXO_ = Chem.MolFromSmarts("[c,C]=[SX1]")
    if _PAT_THIOXO_ is not None and mol.HasSubstructMatch(_PAT_THIOXO_):
        for atom in mol.GetAtoms():
            if (atom.GetAtomicNum() == 7 and atom.GetIsAromatic()
                    and atom.GetTotalNumHs() > 0
                    and atom.GetIdx() not in seen_ion
                    and atom.GetIdx() not in claimed_atoms):
                seen_ion.add(atom.GetIdx()); claimed_atoms.add(atom.GetIdx())
                sites.append(dict(label="thioxopurine_NH", atom_indices=[atom.GetIdx()],
                                  heuristic_pka=7.0, site_type="acid"))
                break

    # Ring N adjacent to S (thiazolidine, thiazolidinium): pKa~4-6.
    # EXCLUDE: N bonded to C=O or SO2 (those are imide/sulfonamide acids, not bases).
    _PAT_IMIDE_N_ = Chem.MolFromSmarts("[NX3;H1]([CX3]=O)[SX4](=O)(=O)")
    for atom in mol.GetAtoms():
        if (atom.GetAtomicNum() == 7 and atom.IsInRing()
                and atom.GetTotalNumHs() > 0
                and atom.GetIdx() not in seen_ion
                and atom.GetIdx() not in claimed_atoms):
            # Skip sulfonyl-imide N (saccharin etc.) — those are acid sites
            if _PAT_IMIDE_N_ is not None and any(
                    atom.GetIdx() in m for m in mol.GetSubstructMatches(_PAT_IMIDE_N_)):
                continue
            # Skip N directly bonded to carbonyl (amide-like)
            if any(nb.GetAtomicNum() == 6 and
                   any(b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(nb).GetAtomicNum() == 8
                       for b in nb.GetBonds())
                   for nb in atom.GetNeighbors()):
                continue
            in_sn_ring = any(
                atom.GetIdx() in ring and
                any(mol.GetAtomWithIdx(j).GetAtomicNum() == 16 for j in ring)
                for ring in mol.GetRingInfo().AtomRings()
            )
            if in_sn_ring:
                seen_ion.add(atom.GetIdx()); claimed_atoms.add(atom.GetIdx())
                sites.append(dict(label="thiazolidine_N_weak",
                                  atom_indices=[atom.GetIdx()],
                                  heuristic_pka=5.0, site_type="base"))

    # ── P-P direct bond phosphonate (e.g. test alendronate SMILES) ──────────
    # The diprotic handler only covers P with 2 OHs; when P has a P-P bond and
    # only 1 OH, this handler claims that OH at pKa=2.1 (always ionized).
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 15: continue
        has_pp = any(n.GetAtomicNum() == 15 for n in atom.GetNeighbors())
        has_c  = any(n.GetAtomicNum() == 6  for n in atom.GetNeighbors())
        if not (has_pp and has_c): continue
        for nb in atom.GetNeighbors():
            if (nb.GetAtomicNum() == 8 and nb.GetTotalNumHs() > 0
                    and nb.GetIdx() not in seen_ion
                    and nb.GetIdx() not in claimed_atoms):
                seen_ion.add(nb.GetIdx()); claimed_atoms.add(nb.GetIdx())
                sites.append(dict(label="pp_phosphonate_OH",
                                  atom_indices=[nb.GetIdx()],
                                  heuristic_pka=2.1, site_type="acid"))

    # ── Alpha-hydroxy carboxylate in triazole/electron-poor context ──────────
    # Standard aliphatic OH pKa~14; adjacent electron-withdrawing groups
    # (triazole N, aryl ether) suppress it to ~6.8 → ionized at pH 7.4.
    _PAT_TRIAZOLE_CTX_ = Chem.MolFromSmarts("[nH][n]")
    _PAT_ALPHA_OH_COOH_ = Chem.MolFromSmarts("[OX2H1][CX4][CX3](=O)[OX2H1,OX1-]")
    if (_PAT_TRIAZOLE_CTX_ is not None and _PAT_ALPHA_OH_COOH_ is not None
            and mol.HasSubstructMatch(_PAT_TRIAZOLE_CTX_)):
        for match in mol.GetSubstructMatches(_PAT_ALPHA_OH_COOH_):
            oh = match[0]
            if oh not in seen_ion and oh not in claimed_atoms:
                seen_ion.add(oh); claimed_atoms.add(oh)
                sites.append(dict(label="alpha_OH_triazole_EWG",
                                  atom_indices=[oh],
                                  heuristic_pka=6.8, site_type="acid"))
    # Pass 1: Diprotic phosphorus acids (Bug A fix)
    for pat_dp, pka1, pka2, lbl_dp in _DIPROTIC_P_COMPILED:
        for match in mol.GetSubstructMatches(pat_dp):
            if any(a in claimed_atoms for a in match): continue
            oh_atoms = [i for i in match if mol.GetAtomWithIdx(i).GetSymbol() == "O"
                        and mol.GetAtomWithIdx(i).GetTotalNumHs() > 0 and i not in seen_ion]
            if len(oh_atoms) >= 2:
                seen_ion.add(oh_atoms[0]); claimed_atoms.add(oh_atoms[0])
                sites.append(dict(label=f"{lbl_dp}_pka1", atom_indices=[oh_atoms[0]], heuristic_pka=pka1, site_type="acid"))
                seen_ion.add(oh_atoms[1]); claimed_atoms.add(oh_atoms[1])
                sites.append(dict(label=f"{lbl_dp}_pka2", atom_indices=[oh_atoms[1]], heuristic_pka=pka2, site_type="acid"))
            elif len(oh_atoms) == 1:
                seen_ion.add(oh_atoms[0]); claimed_atoms.add(oh_atoms[0])
                sites.append(dict(label=f"{lbl_dp}_pka2", atom_indices=[oh_atoms[0]], heuristic_pka=pka2, site_type="acid"))
    # Pass 2: Generic SMARTS table
    for lbl, pat, pka_v, stype in _IONIZABLE_SITES_COMPILED:
        for match in mol.GetSubstructMatches(pat):
            if any(a in claimed_atoms for a in match): continue
            ion_atoms = [idx for idx in match
                         if mol.GetAtomWithIdx(idx).GetAtomicNum() in (7, 8, 16)
                         and (mol.GetAtomWithIdx(idx).GetTotalNumHs() > 0
                              or (stype == "base"
                                  and mol.GetAtomWithIdx(idx).GetFormalCharge() == 0
                                  and mol.GetAtomWithIdx(idx).GetAtomicNum() == 7))
                         and idx not in seen_ion]
            if not ion_atoms: continue
            for ion_idx in ion_atoms:
                seen_ion.add(ion_idx)
                sites.append(dict(label=lbl, atom_indices=[ion_idx], heuristic_pka=pka_v, site_type=stype))
    return sites

# ─────────────────────────────────────────────────────────────────────────────
# STAGE G  ·  Microstate scoring and ranking
# ─────────────────────────────────────────────────────────────────────────────
def get_charge_profile(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f"Bad SMILES: {smiles[:60]}")
    net = n_pos = n_neg = 0; rows = []
    for atom in mol.GetAtoms():
        fc = int(atom.GetFormalCharge()); net += fc; n_pos += fc > 0; n_neg += fc < 0
        if fc != 0: rows.append({"atom_idx": atom.GetIdx(), "symbol": atom.GetSymbol(), "formal_charge": fc})
    return {"net_charge": int(net), "n_pos_atoms": int(n_pos), "n_neg_atoms": int(n_neg),
            "has_pos": n_pos > 0, "has_neg": n_neg > 0,
            "is_zwitterion_strict": bool(n_pos > 0 and n_neg > 0 and net == 0), "charged_atoms": rows}

def charged_atoms_text(cp):
    rows = cp.get("charged_atoms", [])
    if not rows: return "none"
    return ", ".join(f"{r['symbol']}{r['atom_idx']}({r['formal_charge']:+d})" for r in rows)

def _best_pka_for_site(site, ml_predictions, pubchem_result):
    stype = site["site_type"]
    for mp in ml_predictions:
        if mp.get("pka") is not None and mp.get("site_type","").lower() == stype:
            return float(mp["pka"]), mp.get("source","ml")
    if pubchem_result.get("available") and pubchem_result.get("confidence") in ("high","medium"):
        vals = pubchem_result.get("pka_values", [])
        if vals:
            best = min(vals, key=lambda v: abs(v - site["heuristic_pka"]))
            # Guard: skip pubchem pKa if it differs from heuristic by > 3 pKa units.
            # Prevents base-site pKa values (e.g. benzimidazolium pKa≈5.5) from
            # being assigned to the wrong event (N-H acid pKa≈13), which would
            # incorrectly score the deprotonated form as dominant at pH 7.4.
            if abs(best - site["heuristic_pka"]) <= 3.0:
                return best, "pubchem"
    return site["heuristic_pka"], "heuristic"

def _label_decision_backend(ml_predictions, pubchem_result, used_heuristic):
    has_ml = bool(ml_predictions); has_pc = pubchem_result.get("available", False)
    ml_src = ml_predictions[0].get("source","ml") if has_ml else None
    if has_ml and has_pc:
        ml_vals = [p["pka"] for p in ml_predictions if p.get("pka") is not None]
        pc_vals  = pubchem_result.get("pka_values", [])
        if ml_vals and pc_vals:
            avg_diff = abs(sum(ml_vals)/len(ml_vals) - sum(pc_vals)/len(pc_vals))
            backend = f"{ml_src}_pubchem_consistent" if avg_diff <= 1.5 else "mixed_evidence"
        else: backend = f"{ml_src}_supported"
        mode = "ml_pka_dominant"
    elif has_ml: backend = f"{ml_src}_supported"; mode = "ml_pka_dominant"
    elif has_pc: backend = "pubchem_supported"; mode = "pubchem_pka_dominant"
    else: backend = "heuristic_only"; mode = "heuristic_only"
    return backend, mode


def _expected_net_charge_from_sites(ion_sites, target_ph):
    """Charge prior from detected sites — used as a tiebreaker in scoring.
    Conservative: aliphatic OH groups stay neutral; polyamines are capped.
    """
    acid_q = 0; base_q = 0
    for s in ion_sites:
        pka  = float(s.get("heuristic_pka", 10.0))
        stype = s.get("site_type", "acid")
        label = str(s.get("label", "")).lower()
        if stype == "acid":
            if "conservative" in label and target_ph < 12.5: continue
            if target_ph > pka: acid_q -= 1
        else:
            if target_ph < pka: base_q += 1
    # Polyamine cap: don't predict >+1 unless there are enough acid groups
    if acid_q == 0 and base_q > 1:
        strong = sum(1 for s in ion_sites
                     if s.get("site_type") == "base"
                     and float(s.get("heuristic_pka", 0)) - target_ph > 2.0)
        base_q = min(base_q, 2) if strong >= 3 else 1
    return acid_q + base_q

def score_microstate_full(microstate_smiles, tautomer_smiles, taut_plausibility, taut_breakdown,
                          ion_sites, ml_predictions, pubchem_result, target_ph, ref_mol=None):
    mol = Chem.MolFromSmiles(microstate_smiles)
    if mol is None: return -1e9, {}, {}, False
    cp  = get_charge_profile(microstate_smiles)
    net = cp["net_charge"]; n_pos, n_neg = cp["n_pos_atoms"], cp["n_neg_atoms"]
    fc_map = {a.GetIdx(): a.GetFormalCharge() for a in mol.GetAtoms()}
    pat_amide_neg = Chem.MolFromSmarts("[$([NX3-]C=O),$([NX3-]c=O)]")
    n_amide_neg   = len(mol.GetSubstructMatches(pat_amide_neg)) if pat_amide_neg else 0
    s_amide_n_dep = -5.0 * n_amide_neg
    s_arom_loss = 0.0
    if ref_mol is not None:
        rings_lost = max(0, _n_aromatic_rings(ref_mol) - _n_aromatic_rings(mol))
        if rings_lost > 0: s_arom_loss = -W_AROM_RING_LOST * rings_lost
    s_tautomer = 0.65 * taut_plausibility
    borderline = False; ph_bd = {}; s_ph = 0.0
    for site in ion_sites:
        pka_val, pka_src = _best_pka_for_site(site, ml_predictions, pubchem_result)
        if abs(target_ph - pka_val) <= BORDERLINE_PKA_WINDOW: borderline = True
        site_charge = sum(fc_map.get(i, 0) for i in site["atom_indices"])
        contrib = hh_ph_match_score(pka_val, target_ph, site["site_type"], site_charge)
        ph_bd[f"pH_{site['label']}[{pka_src}]"] = round(contrib, 3); s_ph += contrib
    s_pubchem_bonus = 0.0
    if pubchem_result.get("available"):
        pc_weight = {"high":1.0,"medium":0.6,"low":0.2}.get(pubchem_result.get("confidence","low"),0.2)
        for pka_val in pubchem_result["pka_values"]:
            exp = -1 if hh_fraction_charged(pka_val, target_ph, "acid") > 0.5 else 0
            s_pubchem_bonus += 0.25 * pc_weight if net == exp else -0.15 * pc_weight
        s_pubchem_bonus = max(-0.4, min(0.5, s_pubchem_bonus))
    has_acid_site = any(s["site_type"]=="acid" and (target_ph - s["heuristic_pka"]) > 1.0 for s in ion_sites)
    has_base_site = any(s["site_type"]=="base" and (s["heuristic_pka"] - target_ph) > 1.0 for s in ion_sites)
    if cp["is_zwitterion_strict"]:
        s_zwit = 0.8 if (has_acid_site and has_base_site) else -0.6
    else:
        s_zwit = -0.4 if (has_acid_site and has_base_site and net == 0 and n_pos == 0) else 0.0
    strong_acid   = [s for s in ion_sites if s["site_type"]=="acid" and (target_ph - s["heuristic_pka"]) > 2.0]
    strong_base   = [s for s in ion_sites if s["site_type"]=="base" and (s["heuristic_pka"] - target_ph) > 2.0]
    probable_acid = [s for s in ion_sites if s["site_type"]=="acid" and 0.0 < (target_ph - s["heuristic_pka"]) <= 2.0]
    probable_base = [s for s in ion_sites if s["site_type"]=="base" and 0.0 < (s["heuristic_pka"] - target_ph) <= 2.0]
    s_improbable = 0.0
    if strong_acid  and net >= 0 and n_neg == 0: s_improbable -= 0.5  * len(strong_acid)
    if strong_base  and net <= 0 and n_pos == 0: s_improbable -= 0.5  * len(strong_base)
    if probable_acid and net >= 0 and n_neg == 0: s_improbable -= 0.35 * len(probable_acid)
    if probable_base and net <= 0 and n_pos == 0: s_improbable -= 0.35 * len(probable_base)
    s_multi = -0.12 * max(0, n_pos + n_neg - 2)
    # Charge prior: soft incentive toward expected net charge
    expected_net = _expected_net_charge_from_sites(ion_sites, target_ph)
    if net == expected_net:
        s_target_net = 2.0
    else:
        s_target_net = -1.25 * abs(net - expected_net)
    # Enhanced multi-charge penalty for over-protonated states
    if n_pos >= 2 and n_neg == 0:
        s_multi -= 2.5 * (n_pos - 1)

    total = s_amide_n_dep + s_arom_loss + s_tautomer + s_ph + s_pubchem_bonus + s_zwit + s_improbable + s_multi + s_target_net
    def _has_key(bd, keys, positive):
        return any(bd.get(k,0) * (1 if positive else -1) > 0 for k in keys)
    flag_amide     = _has_key(taut_breakdown, ["amide","lactam","acylhydrazone_NH","hydrazide_NH"], True)
    flag_imidic    = _has_key(taut_breakdown, ["imidic_acid_open","lactim_ring","iminol_general"], False)
    flag_lactim    = taut_breakdown.get("lactim_ring", 0) < 0
    flag_arom_lost = s_arom_loss < 0 or taut_breakdown.get("arom_ring_lost_vs_input", 0) < 0
    used_heuristic = not bool(ml_predictions) and not pubchem_result.get("available")
    decision_backend, decision_mode = _label_decision_backend(ml_predictions, pubchem_result, used_heuristic)
    cp.update(flag_amide_preserved=flag_amide, flag_imidic_acid_penalty=flag_imidic,
              flag_lactim_penalty=flag_lactim, flag_amide_n_deprotonation_penalty=n_amide_neg>0,
              flag_aromaticity_lost=flag_arom_lost, decision_backend=decision_backend, decision_mode=decision_mode)
    bd_full = {"s_amide_n_deproton [safety]": round(s_amide_n_dep,3), "s_aromaticity_loss [safety]": round(s_arom_loss,3),
               "s_tautomer_plausibility": round(s_tautomer,3), "s_ph_consistency [HH]": round(s_ph,3),
               "s_pubchem_evidence_bonus": round(s_pubchem_bonus,3), "s_zwitterion_consistency": round(s_zwit,3),
               "s_improbable_neutral": round(s_improbable,3), "s_multicharge_penalty": round(s_multi,3),
               "total_score": round(total,3), **ph_bd}
    return total, cp, bd_full, borderline

def _manual_deprotonate_site(smiles, site):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    rw = Chem.RWMol(mol); target_idx = None
    for idx in site["atom_indices"]:
        if idx >= rw.GetNumAtoms(): continue
        atom = rw.GetAtomWithIdx(idx); sym, nh = atom.GetSymbol(), atom.GetTotalNumHs()
        if site["site_type"] == "acid":
            if sym in ("O","S") and nh >= 1: target_idx = idx; break
            if sym == "N" and nh >= 1 and target_idx is None: target_idx = idx
        else:
            if sym == "N" and atom.GetFormalCharge() == 0: target_idx = idx; break
    if target_idx is None: return None
    atom = rw.GetAtomWithIdx(target_idx)
    try:
        if site["site_type"] == "acid":
            atom.SetFormalCharge(-1); atom.SetNumExplicitHs(0); atom.SetNoImplicit(False)
        else:
            atom.SetFormalCharge(+1); atom.SetNumExplicitHs(atom.GetTotalNumHs()+1); atom.SetNoImplicit(False)
        new_mol = rw.GetMol(); Chem.SanitizeMol(new_mol)
        return Chem.MolToSmiles(new_mol, isomericSmiles=True, canonical=True)
    except Exception: return None

def _supplement_dimorphite(tautomer_smiles, dimorphite_results, ion_sites, target_ph):
    supplemented = list(dimorphite_results); existing = set(supplemented)
    active = [s for s in ion_sites
              if not (s["site_type"]=="acid" and (target_ph - s["heuristic_pka"]) < -1.5)
              and not (s["site_type"]=="base" and (s["heuristic_pka"] - target_ph) < -1.5)]
    if not active: return supplemented
    # BFS multi-site: generates zwitterions, poly-ionics (EDTA, bisphosphonates)
    queue = [tautomer_smiles]
    for _round in range(min(len(active), 6)):
        next_q = []
        for base_smi in queue:
            for site in active:
                new_smi = _manual_deprotonate_site(base_smi, site)
                if new_smi and new_smi not in existing:
                    existing.add(new_smi); supplemented.append(new_smi); next_q.append(new_smi)
        if not next_q or len(supplemented) >= 128: break
        queue = next_q
    return supplemented

def generate_ranked_microstates(base_smiles, target_ph=7.4, ph_window=1.0, max_tautomers=8, top_n=5, pubchem_result=None):
    if pubchem_result is None: pubchem_result = {}
    ref_mol = Chem.MolFromSmiles(base_smiles)
    kept, disc, tr_flag, tr_motifs = enumerate_and_filter_tautomers(base_smiles, max_states=max_tautomers)
    if disc:
        print(f"   🔬  Discarded {len(disc)} implausible tautomers (e.g. score={disc[0]['score']:.1f}: {disc[0]['smiles'][:55]})")
    ml_preds  = unipka_predict(base_smiles)
    ion_sites = find_ionizable_sites(ref_mol) if ref_mol else []
    all_micro = []; seen_smi = set()
    ph_lo = max(0.0, target_ph - ph_window / 2); ph_hi = min(14.0, target_ph + ph_window / 2)
    for ti, taut in enumerate(kept, 1):
        raw_microstates = dimorphite_enumerate(taut["smiles"], ph_lo, ph_hi)
        microstates = _supplement_dimorphite(taut["smiles"], raw_microstates, ion_sites, target_ph)
        if len(microstates) > len(raw_microstates):
            print(f"   🧪  Supplemented {len(microstates)-len(raw_microstates)} microstate(s) for under-covered ionizable sites.")
        for pi, psmi in enumerate(microstates, 1):
            if psmi in seen_smi: continue
            seen_smi.add(psmi)
            try:
                sc, cp, bd, bl = score_microstate_full(
                    microstate_smiles=psmi, tautomer_smiles=taut["smiles"],
                    taut_plausibility=taut["score"], taut_breakdown=taut["breakdown"],
                    ion_sites=ion_sites, ml_predictions=ml_preds, pubchem_result=pubchem_result,
                    target_ph=target_ph, ref_mol=ref_mol)
            except Exception as e: print(f"⚠️  Scoring error ({psmi[:40]}): {e}"); continue
            all_micro.append({
                "tautomer_rank": ti, "protomer_rank_in_tautomer": pi,
                "tautomer_smiles": taut["smiles"], "tautomer_plausibility": round(taut["score"],3),
                "microstate_smiles": psmi, "parent_smiles": base_smiles,
                "selection_score": float(sc), "net_charge": cp["net_charge"],
                "has_pos": cp["has_pos"], "has_neg": cp["has_neg"],
                "is_zwitterion_strict": cp["is_zwitterion_strict"],
                "charged_atoms": charged_atoms_text(cp), "charged_atom_rows": cp["charged_atoms"],
                "decision_backend": cp.get("decision_backend","unknown"), "decision_mode": cp.get("decision_mode","unknown"),
                "flag_amide_preserved": cp.get("flag_amide_preserved",False),
                "flag_imidic_acid_penalty": cp.get("flag_imidic_acid_penalty",False),
                "flag_lactim_penalty": cp.get("flag_lactim_penalty",False),
                "flag_amide_n_deprotonation_penalty": cp.get("flag_amide_n_deprotonation_penalty",False),
                "flag_aromaticity_lost": cp.get("flag_aromaticity_lost",False),
                "flag_borderline_pka": bl, "flag_tautomer_rich": tr_flag,
                "flag_pubchem_text_ambiguous": pubchem_result.get("flags",{}).get("vague_or_approximate",False),
                "flag_pubchem_conflicting": pubchem_result.get("flags",{}).get("conflicting_values",False),
                "flag_pubchem_confidence": pubchem_result.get("confidence","n/a"),
                "flag_unipka_used": bool(ml_preds), "flag_dimorphite_used": _DIMORPHITE_OK,
                "pKa_source": (ml_preds[0]["source"] if ml_preds else ("pubchem" if pubchem_result.get("available") else "heuristic")),
                **{f"score_{k}": v for k, v in bd.items()},
                **{f"taut_{k}":  v for k, v in taut["breakdown"].items()},
            })
    if not all_micro: return [], False, [], tr_flag, tr_motifs, ml_preds
    all_micro.sort(key=lambda x: (-x["selection_score"], abs(x["net_charge"]), x["tautomer_rank"], x["microstate_smiles"]))
    best_sc = all_micro[0]["selection_score"]
    for i, row in enumerate(all_micro, 1):
        row["microstate_rank"] = i; row["delta_from_best"] = round(best_sc - row["selection_score"], 3)
    top = all_micro[:max(1, top_n)]
    score_ambig = len(top) > 1 and top[1]["delta_from_best"] <= AMBIGUITY_SCORE_GAP
    ambiguous = score_ambig or any(r["flag_borderline_pka"] for r in top[:2]) or tr_flag
    for row in all_micro:
        row["ambiguous_top_assignment"] = ambiguous; row["flag_multiprotic"] = len(ion_sites) >= 2
    return top, ambiguous, all_micro, tr_flag, tr_motifs, ml_preds

# ─────────────────────────────────────────────────────────────────────────────
# STAGE H  ·  3D construction + file I/O
# ─────────────────────────────────────────────────────────────────────────────
def build_minimized_3d(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f"Bad SMILES: {smiles[:60]}")
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3(); p.randomSeed = 42
    if AllChem.EmbedMolecule(mol, p) != 0: raise ValueError("ETKDG embedding failed.")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    return mol

def mol_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower(); mol = None
    if ext == ".sdf": mol = next((m for m in Chem.SDMolSupplier(filepath, removeHs=False) if m), None)
    elif ext == ".mol2": mol = Chem.MolFromMol2File(filepath, removeHs=False)
    elif ext == ".pdb": mol = Chem.MolFromPDBFile(filepath, removeHs=False)
    if mol is None: raise ValueError(f"Cannot parse: {filepath}")
    return mol

DISPLAY_COLS = [
    "microstate_rank", "tautomer_rank", "tautomer_plausibility",
    "microstate_smiles", "selection_score", "net_charge", "charged_atoms",
    "decision_backend", "decision_mode",
    "flag_amide_preserved", "flag_imidic_acid_penalty",
    "flag_amide_n_deprotonation_penalty", "flag_aromaticity_lost",
    "flag_borderline_pka", "flag_pubchem_text_ambiguous", "flag_unipka_used", "pKa_source",
    "delta_from_best",
]

def parse_smi_lines(text):
    records = []; idx = 1
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        parts = line.split()
        records.append((parts[0], parts[1] if len(parts) > 1 else f"mol_{idx:03d}")); idx += 1
    return records

def save_2d_image(smiles, path, size=(800,600)):
    try:
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return False
        AllChem.Compute2DCoords(mol); Draw.MolToImage(mol, size=size).save(path); return True
    except Exception: return False

def save_molecule_files(mol, base_path, formats):
    saved = {}; warnings = []; mol2_via_obabel = False
    try:
        sdf_path = f"{base_path}.sdf"
        w = Chem.SDWriter(sdf_path); w.write(mol); w.close(); saved["sdf"] = sdf_path
    except Exception as e: warnings.append(f"Could not save SDF: {e}")
    for fmt in [f.upper() for f in formats]:
        if fmt == "SDF": continue
        try:
            if fmt == "PDB":
                fp = f"{base_path}.pdb"; Chem.MolToPDBFile(mol, fp); saved["pdb"] = fp
            elif fmt == "MOL2":
                fp = f"{base_path}.mol2"
                if hasattr(Chem, "MolToMol2File"):
                    try: Chem.MolToMol2File(mol, fp); saved["mol2"] = fp; continue
                    except Exception: pass
                if "pdb" not in saved:
                    pdb_fp = f"{base_path}.pdb"; Chem.MolToPDBFile(mol, pdb_fp); saved["pdb"] = pdb_fp
                if convert_pdb_to_mol2_obabel(saved["pdb"], fp):
                    saved["mol2"] = fp; mol2_via_obabel = True
                else:
                    warnings.append("MOL2 unavailable — install Open Babel (obabel)."
                                    if not check_obabel() else "MOL2 conversion failed.")
        except Exception as e: warnings.append(f"Could not save {fmt}: {e}")
    if mol2_via_obabel: warnings.append("ℹ️ MOL2 generated via Open Babel (converted from PDB)")
    saved["warnings"] = warnings; return saved

# ─────────────────────────────────────────────────────────────────────────────
# run_job  —  main workflow adapter
# ─────────────────────────────────────────────────────────────────────────────
def run_job(*, input_type, smiles_text, uploaded_bytes, uploaded_name, target_pH, output_name,
            out_dir, output_formats=None, enumerate_stereoisomers=True, use_pubchem=True,
            ph_window=1.0, max_tautomers=8, top_n_microstates=5, write_alt_3d_for_top_k=3):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if not output_formats: output_formats = ["PDB"]
    ligands_raw = []
    if input_type == "SMILES":
        s = (smiles_text or "").strip()
        if not s: raise ValueError("SMILES is empty.")
        ligands_raw.append({"name": output_name or "ligand", "smiles": s})
    elif input_type == "SMI_FILE":
        if not uploaded_bytes: raise ValueError("No .smi file uploaded.")
        for smi, name in parse_smi_lines(uploaded_bytes.decode("utf-8", errors="replace")):
            ligands_raw.append({"name": name, "smiles": smi})
    elif input_type == "FILE":
        if not uploaded_bytes or not uploaded_name: raise ValueError("No ligand file uploaded.")
        ext = os.path.splitext(uploaded_name)[1].lower()
        tmp_path = out / f"uploaded{ext}"; tmp_path.write_bytes(uploaded_bytes)
        mol_in = mol_from_file(str(tmp_path))
        try:
            frags = Chem.GetMolFrags(mol_in, asMols=True, sanitizeFrags=False)
            if len(frags) > 1: mol_in = max(frags, key=lambda m: m.GetNumHeavyAtoms())
            Chem.SanitizeMol(mol_in)
        except Exception: pass
        base_smi = Chem.MolToSmiles(Chem.RemoveHs(mol_in), canonical=True)
        ligands_raw.append({"name": output_name or os.path.splitext(uploaded_name)[0], "smiles": base_smi})
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    results = []; all_micro_rows = []; format_warnings = []; keep_stereo = not enumerate_stereoisomers
    for ligand in ligands_raw:
        base_name = ligand["name"]
        stereo_rows = enumerate_stereo(ligand["smiles"], keep_original=keep_stereo)
        for _si, (raw_smiles, stereo_tag) in enumerate(stereo_rows, 1):
            pretty = base_name if stereo_tag is None else f"{base_name}_{stereo_tag}"
            print(f"\n{SEP}\n🧪  {pretty}\n{SEP}")
            can_smi, status = standardize_smiles(raw_smiles)
            if can_smi is None:
                print(f"❌  {status}"); format_warnings.append(f"Standardization failed for {pretty}: {status}"); continue
            print(f"    SMILES (std): {can_smi}")
            print("    🔍  PubChem lookup … ", end="", flush=True); pc = {}
            if use_pubchem and _REQUESTS_OK:
                try:
                    pc = pubchem_lookup(can_smi)
                    if pc["available"]: print(f"✅  CID={pc['cid']}  pKa={pc['pka_values']}  flags={[k for k,v in pc['flags'].items() if v]}")
                    else: print(f"—  {pc.get('error','no data')}")
                except Exception as e: print(f"—  error: {e}")
            else: print("—  disabled")
            top, ambig, all_m, tr_flag, tr_motifs, ml_preds = generate_ranked_microstates(
                can_smi, target_ph=target_pH, ph_window=ph_window,
                max_tautomers=max_tautomers, top_n=top_n_microstates, pubchem_result=pc)
            if not top:
                print("❌  No valid microstates generated."); format_warnings.append(f"No valid microstates for {pretty}"); continue
            t = top[0]
            print(f"\n🔬  Microstates generated : {len(all_m)}")
            print(f"⚠️   Ambiguous top state   : {'YES' if ambig else 'NO'}")
            if tr_flag: print(f"🔄  Tautomer-rich motifs  : {', '.join(tr_motifs)}")
            print(f"\n🏆  Rank-1")
            for label, val in [
                ("Score", f"{t['selection_score']:.3f}"), ("SMILES", t["microstate_smiles"]),
                (f"Charge @ pH {target_pH}", f"{t['net_charge']:+d}"),
                ("Zwitterion",  "YES" if t["is_zwitterion_strict"] else "NO"),
                ("Amide kept",  "YES" if t["flag_amide_preserved"] else "NO"),
                ("Imidic acid", "YES ⚠️" if t["flag_imidic_acid_penalty"] else "NO"),
                ("[N-]C=O",     "YES ⚠️" if t["flag_amide_n_deprotonation_penalty"] else "NO"),
                ("Aromaticity", "LOST ⚠️" if t.get("flag_aromaticity_lost") else "OK"),
                ("pKa source",  t["pKa_source"]),
                ("Backend",     f"{t['decision_backend']}  ({t['decision_mode']})"),
            ]: print(f"    {label:<14}: {val}")
            micro_csv = str(out / f"{pretty}_microstates.csv")
            try:
                import pandas as pd
                pd.DataFrame([{k:v for k,v in r.items() if k != "charged_atom_rows"} for r in top]).to_csv(micro_csv, index=False)
                print(f"\n💾  {Path(micro_csv).name}")
            except Exception as e: print(f"CSV write failed: {e}"); micro_csv = None
            alt3d = []
            for row in top[:max(1, write_alt_3d_for_top_k)]:
                rk = row["microstate_rank"]; bp = str(out / f"{pretty}_micro{rk}_min")
                try:
                    m3d = build_minimized_3d(row["microstate_smiles"])
                    files = save_molecule_files(m3d, bp, output_formats)
                    for w in files.pop("warnings", []):
                        if w not in format_warnings: format_warnings.append(w)
                    save_2d_image(row["microstate_smiles"], f"{bp}_2D.png")
                    alt3d.append((rk, files.get("pdb"), files.get("sdf"), files))
                except Exception as e: print(f"⚠️  3D failed for rank {rk}: {e}")
            sel_pdb  = alt3d[0][1] if alt3d else None
            sel_sdf  = alt3d[0][2] if alt3d else None
            sel_mol2 = alt3d[0][3].get("mol2") if alt3d else None
            if sel_pdb: print(f"💾  {Path(sel_pdb).name}, {Path(sel_sdf).name if sel_sdf else 'no sdf'}")
            results.append({
                "name": pretty, "base_smiles": can_smi, "stereo": stereo_tag,
                "selected_microstate_smiles": t["microstate_smiles"],
                "selected_tautomer_smiles": t["tautomer_smiles"],
                "pKa_source": t["pKa_source"], "decision_backend": t["decision_backend"], "decision_mode": t["decision_mode"],
                "pubchem_pka_values": str(pc.get("pka_values",[])), "pubchem_confidence": pc.get("confidence","n/a"),
                "pubchem_cid": pc.get("cid"), "formal_charge": t["net_charge"],
                "is_zwitterion": t["is_zwitterion_strict"], "charged_atoms": t["charged_atoms"],
                "ambiguous_top_assignment": ambig, "flag_tautomer_rich": tr_flag, "flag_tautomer_motifs": tr_motifs,
                "flag_amide_preserved": t["flag_amide_preserved"], "flag_imidic_acid_penalty": t["flag_imidic_acid_penalty"],
                "flag_amide_n_deprotonation": t["flag_amide_n_deprotonation_penalty"],
                "flag_aromaticity_lost": t.get("flag_aromaticity_lost",False),
                "flag_borderline_pka": t["flag_borderline_pka"], "flag_unipka_used": t["flag_unipka_used"],
                "microstate_csv": micro_csv, "minimized_pdb": sel_pdb, "minimized_sdf": sel_sdf, "minimized_mol2": sel_mol2,
                "selection_score": t["selection_score"], "n_all_microstates": len(all_m),
                "top_microstates": [{k:v for k,v in r.items() if k != "charged_atom_rows"} for r in top],
                "alt3d": [{"rank":r[0],"pdb":r[1],"sdf":r[2],"files":r[3],"smiles":top[i]["microstate_smiles"]}
                          for i,r in enumerate(alt3d)],
            })
            for row in top:
                all_micro_rows.append({**{k:v for k,v in row.items() if k != "charged_atom_rows"},
                                       "name":pretty, "target_pH":target_pH, "pubchem_cid":pc.get("cid")})

    print(f"\n{SEP}\n📊  SUMMARY  |  pH={target_pH}  |  backend={_PKA_BACKEND}\n{SEP}")
    for r in results:
        print(f"\n▶  {r['name']}")
        for k, v in [("Selected SMILES", r["selected_microstate_smiles"]),
                     (f"Charge @ pH {target_pH}", f"{r['formal_charge']:+d}"),
                     ("Charged atoms", r["charged_atoms"]),
                     ("Zwitterion", "YES" if r["is_zwitterion"] else "NO"),
                     ("pKa source", r["pKa_source"]), ("PubChem pKa", r["pubchem_pka_values"]),
                     ("Amide preserved", "YES" if r["flag_amide_preserved"] else "NO"),
                     ("Imidic acid flag", "YES ⚠️" if r["flag_imidic_acid_penalty"] else "NO"),
                     ("[N-]C=O flag", "YES ⚠️" if r["flag_amide_n_deprotonation"] else "NO"),
                     ("Aromaticity lost", "YES ⚠️" if r.get("flag_aromaticity_lost") else "NO"),
                     ("Ambiguous", "YES" if r["ambiguous_top_assignment"] else "NO")]:
            print(f"   {k:<28}: {v}")

    summary_lines = [SEP, f"pKaNET Cloud — SUMMARY  |  pH={target_pH}  |  pKa backend={_PKA_BACKEND}", SEP,
                     f"Structures: {len(results)}  |  pH window: ±{ph_window/2:.1f}  |  max tautomers: {max_tautomers}  |  top microstates: {top_n_microstates}", ""]
    for r in results:
        summary_lines += [
            f"▶  {r['name']}", f"   Selected SMILES   : {r['selected_microstate_smiles']}",
            f"   Charge @ pH {target_pH} : {r['formal_charge']:+d}", f"   Charged atoms     : {r['charged_atoms']}",
            f"   Zwitterion        : {'YES' if r['is_zwitterion'] else 'NO'}",
            f"   Ambiguous         : {'YES' if r['ambiguous_top_assignment'] else 'NO'}",
            f"   pKa source        : {r['pKa_source']}", f"   PubChem pKa       : {r['pubchem_pka_values']} (conf={r['pubchem_confidence']})",
            f"   Amide preserved   : {'YES' if r['flag_amide_preserved'] else 'NO'}",
            f"   Imidic acid       : {'YES ⚠️' if r['flag_imidic_acid_penalty'] else 'NO'}",
            f"   [N-]C=O           : {'YES ⚠️' if r['flag_amide_n_deprotonation'] else 'NO'}",
            f"   Aromaticity lost  : {'YES ⚠️' if r.get('flag_aromaticity_lost') else 'NO'}", ""]
    summary_text = "\n".join(summary_lines)
    (out / "summary.txt").write_text(summary_text + "\n")
    if input_type == "SMI_FILE" and results:
        log_lines = ["# pKaNET Cloud — Processing Log", f"# pH={target_pH}  backend={_PKA_BACKEND}",
                     "# Name | pH-SMILES | Charge | Zwitterion | Ambiguous | pKa_source | PubChem_pKa", ""]
        for r in results:
            log_lines.append(f"{r['name']}\t{r['selected_microstate_smiles']}\t{r['formal_charge']:+d}\t"
                             f"{'Yes' if r['is_zwitterion'] else 'No'}\t{'Yes' if r['ambiguous_top_assignment'] else 'No'}\t"
                             f"{r['pKa_source']}\t{r['pubchem_pka_values']}")
        (out / "processing.log").write_text("\n".join(log_lines) + "\n")
    return {"results": results, "summary_text": summary_text,
            "out_dir": str(out), "format_warnings": format_warnings, "pka_backend": _PKA_BACKEND}

# ─────────────────────────────────────────────────────────────────────────────
# ZIP helpers
# ─────────────────────────────────────────────────────────────────────────────
def zip_minimized_structures(out_dir, zip_path, selected_formats):
    out = Path(out_dir); zp = Path(zip_path)
    fmts = [f.lower() for f in selected_formats]
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.glob("*_min.*"):
            s = p.suffix.lower()
            if (s == ".pdb" and "pdb" in fmts) or (s == ".mol2" and "mol2" in fmts):
                z.write(p, arcname=p.name)
    return str(zp)

def zip_all_outputs(out_dir, zip_path):
    out = Path(out_dir); zp = Path(zip_path)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file(): z.write(p, arcname=p.relative_to(out))
    return str(zp)
