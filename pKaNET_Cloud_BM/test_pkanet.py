#!/usr/bin/env python3
"""
test_pkanet.py — Internal regression test suite for pKaNET Cloud+
65 chemically curated cases across 12 functional-group groups.

Part of the pKaNET Cloud+ validation framework for Anyone Can Dock.

Usage:
    python3 test_pkanet.py pKaNET.py          # run all 65 tests
    python3 test_pkanet.py pKaNET.py G8       # flavonoid regression only
    python3 test_pkanet.py pKaNET.py G12      # drug regression panel only

    The first argument is the path to the pKaNET core module
    (core-pkaNET-v80.py or equivalent).
    The optional second argument filters by group label (G1–G12).

Groups:
    G1   Imidazole-type N-H              (10 tests)
    G2   Phosphonate / Phosphate         ( 7 tests)
    G3   Thiol ArSH / AlkSH              ( 5 tests)
    G4   Carboxylic acid                 ( 5 tests)
    G5   Phenol variants                 ( 6 tests)  incl. warfarin enol acid
    G6   Amine bases                     ( 5 tests)
    G7   Sulfonamide / Saccharin         ( 4 tests)
    G8   Flavonoid regression            ( 4 tests)  ← MUST NOT change
    G9   Zwitterion / Multi-site         ( 5 tests)
    G10  PubChem pKa guard               ( 3 tests)
    G11  Truly neutral                   ( 4 tests)
    G12  Drug regression panel           ( 7 tests)  incl. EGFR inhibitors

Test environment:
    Tests run in heuristic-only mode when dimorphite-dl and ML pKa backends
    are unavailable.  PubChem calls are replaced by inline mock dicts where
    relevant (G10, T02, T48).

Expected outcome on a passing build:
    ✅ ALL PASS
"""
import sys
import importlib.util
from pathlib import Path

GROUP_LABELS = {
    "G1":  "Imidazole-type N-H",
    "G2":  "Phosphonate/Phosphate",
    "G3":  "Thiol ArSH/AlkSH",
    "G4":  "Carboxylic acid",
    "G5":  "Phenol variants",
    "G6":  "Amine bases",
    "G7":  "Sulfonamide/Saccharin",
    "G8":  "Flavonoid (regression — MUST NOT change)",
    "G9":  "Zwitterion/Multi-site",
    "G10": "PubChem pKa guard",
    "G11": "Truly neutral",
    "G12": "Drug regression panel",
}

# ── Test definitions ───────────────────────────────────────────────────────
# (id, group, name, smiles, ph, expected_charge, must_not_contain, pubchem_mock)
TESTS_RAW = [
    # ── G1: Imidazole-type N-H ────────────────────────────────────────────
    # NH pKa 12-14 → neutral pH 7.4; PubChem gives base pKa ~5-7 → must guard
    ("T01","G1","Imidazole",                "C1=CN=CN1",                             7.4, 0, "[n-]", {}),
    ("T02","G1","Benzimidazole",            "C1=CC=C2C(=C1)NC=N2",                   7.4, 0, "[n-]", dict(available=True,confidence="medium",pka_values=[5.48,5.3])),
    ("T03","G1","Pyrazole",                 "C1=CC=NN1",                             7.4, 0, "[n-]", {}),
    ("T04","G1","Indazole",                 "C1=CC2=CC=CC=C2N1",                     7.4, 0, "[n-]", {}),
    ("T05","G1","Purine",                   "C1=NC2=NC=NC=C2N1",                     7.4, 0, "[n-]", {}),
    ("T06","G1","Adenine",                  "Nc1ncnc2[nH]cnc12",                     7.4, 0, "[n-]", {}),
    ("T07","G1","1-Methylbenzimidazole",    "Cn1cnc2ccccc21",                        7.4, 0, "[n-]", {}),
    ("T08","G1","Clotrimazole",             "Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1ccnc1",7.4, 0, "[n-]", {}),
    ("T09","G1","Omeprazole",              "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",7.4,0,"[n-]",{}),
    ("T10","G1","Metronidazole",            "Cc1ncc([N+](=O)[O-])n1CCO",             7.4, 0, "[n-]", {}),

    # ── G2: Phosphonate / Phosphate ───────────────────────────────────────
    # pKa1=2.1 fully ionized; pKa2=6.5 also ionized at pH 7.4 → -2 per P
    ("T11","G2","Methylphosphonic acid",    "CP(=O)(O)O",                            7.4,-2, None,   {}),
    ("T12","G2","Phenylphosphonic acid",    "OP(=O)(O)c1ccccc1",                     7.4,-2, None,   {}),
    ("T13","G2","Fosfomycin",              "CC1OC1P(=O)(O)O",                        7.4,-2, None,   {}),
    ("T14","G2","Alendronate",             "NCCCP(=O)(O)P(=O)(O)O",                  7.4,-2, None,   {}),
    ("T15","G2","Tenofovir",               "Cc1cn([C@@H]2C[C@H](COP(=O)(O)O)O2)c(=O)nc1N",7.4,-2,None,{}),
    ("T16","G2","Phosphate monoester",     "COP(=O)(O)O",                            7.4,-2, None,   {}),
    ("T17","G2","Glyphosate",              "OC(=O)CNCP(=O)(O)O",                    7.4,-2, None,   {}),

    # ── G3: Thiol ─────────────────────────────────────────────────────────
    # ArSH pKa~6.5 → ionized -1; AlkSH pKa~10.5 → neutral
    ("T18","G3","Thiophenol",              "Sc1ccccc1",                              7.4,-1, None,   {}),
    ("T19","G3","4-Chlorothiophenol",      "Sc1ccc(Cl)cc1",                         7.4,-1, None,   {}),
    ("T20","G3","6-Mercaptopurine",        "S=c1[nH]cnc2nc[nH]c12",                 7.4,-1, None,   {}),
    ("T21","G3","Captopril",               "SC[C@@H]1N[C@@H](C(=O)O)CS1",           7.4,-1, None,   {}),
    ("T22","G3","Ethanethiol",             "CCS",                                   7.4, 0, None,   {}),

    # ── G4: Carboxylic acid ───────────────────────────────────────────────
    ("T23","G4","Acetic acid",             "CC(=O)O",                               7.4,-1, None,   {}),
    ("T24","G4","Ibuprofen",               "CC(C)Cc1ccc(CC(C)C(=O)O)cc1",           7.4,-1, None,   {}),
    ("T25","G4","Aspirin",                 "CC(=O)Oc1ccccc1C(=O)O",                 7.4,-1, None,   {}),
    ("T26","G4","Diclofenac",              "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",        7.4,-1, None,   {}),
    ("T27","G4","Trichloroacetic acid",    "OC(=O)C(Cl)(Cl)Cl",                     7.4,-1, None,   {}),

    # ── G5: Phenol variants ───────────────────────────────────────────────
    ("T28","G5","Phenol",                  "Oc1ccccc1",                             7.4, 0, None,   {}),
    ("T29","G5","4-Nitrophenol",           "Oc1ccc([N+](=O)[O-])cc1",               7.4,-1, None,   {}),
    ("T30","G5","Pentafluorophenol",       "Oc1c(F)c(F)c(F)c(F)c1F",               7.4,-1, None,   {}),
    ("T31","G5","Acetaminophen",           "CC(=O)Nc1ccc(O)cc1",                    7.4, 0, None,   {}),
    ("T32","G5","Catechol",                "Oc1ccccc1O",                            7.4, 0, None,   {}),
    ("T33","G5","Warfarin",                "CC(=O)CC1C(=O)c2ccccc2OC1c1ccccc1",    7.4,-1, None,   {}),

    # ── G6: Amine bases ───────────────────────────────────────────────────
    ("T34","G6","Aniline",                 "Nc1ccccc1",                             7.4, 0, None,   {}),
    ("T35","G6","Pyridine",                "c1ccncc1",                              7.4, 0, None,   {}),
    ("T36","G6","Methylamine",             "CN",                                    7.4,+1, None,   {}),
    ("T37","G6","Metformin",               "CN(C)C(=N)NC(=N)N",                     7.4,+1, None,   {}),
    ("T38","G6","Amlodipine",              "CCOC(=O)C1=C(CCl)NC(C)=C(C(=O)OCC)C1c1ccccc1Cl",7.4,+1,None,{}),

    # ── G7: Sulfonamide / Saccharin ───────────────────────────────────────
    ("T39","G7","Methanesulfonamide",      "CS(=O)(=O)N",                           7.4, 0, None,   {}),
    ("T40","G7","Saccharin",               "O=C1NS(=O)(=O)c2ccccc21",               7.4,-1, None,   {}),
    ("T41","G7","Chlorothiazide",          "NS(=O)(=O)c1cc2c(cc1Cl)NCNS2=O",       7.4,-1, None,   {}),
    ("T42","G7","Furosemide",              "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl",7.4,-2,None,  {}),

    # ── G8: Flavonoid (regression — MUST NOT change) ──────────────────────
    ("T43","G8","Baicalein",               "O=c1cc(-c2ccccc2)oc2cc(O)c(O)c(O)c12",  7.4, 0,"[O-]", {}),
    ("T44","G8","Apigenin",                "O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",  7.4, 0,"[O-]", {}),
    ("T45","G8","Luteolin",                "O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",7.4,0,"[O-]", {}),
    ("T46","G8","Kaempferol",              "O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",7.4,0,"[O-]", {}),

    # ── G9: Zwitterion / Multi-site ───────────────────────────────────────
    ("T47","G9","Glycine",                 "NCC(=O)O",                              7.4, 0, None,   {}),
    ("T48","G9","Histidine",               "N[C@@H](Cc1c[nH]cn1)C(=O)O",            7.4, 0,"[n-]", dict(available=True,confidence="high",pka_values=[1.8,6.0,9.2])),
    ("T49","G9","Glutamic acid",           "N[C@@H](CCC(=O)O)C(=O)O",              7.4,-1, None,   {}),
    ("T50","G9","Lysine",                  "NCCCC[C@@H](N)C(=O)O",                  7.4,+1, None,   {}),
    ("T51","G9","Cysteine",                "N[C@@H](CS)C(=O)O",                     7.4, 0, None,   {}),

    # ── G10: PubChem pKa guard ────────────────────────────────────────────
    # guard: if |pubchem_pKa - heuristic_pKa| > 3.0 → skip pubchem
    ("T52","G10","Benzimidazole+PubChem base pKa","C1=CC=C2C(=C1)NC=N2",            7.4, 0,"[n-]", dict(available=True,confidence="medium",pka_values=[5.48,5.3])),
    ("T53","G10","Imidazole+PubChem base pKa",    "C1=CN=CN1",                      7.4, 0,"[n-]", dict(available=True,confidence="high",pka_values=[7.0,6.99])),
    ("T54","G10","Phenol+PubChem correct pKa",    "Oc1ccccc1",                      7.4, 0, None,  dict(available=True,confidence="high",pka_values=[9.99,10.0])),

    # ── G11: Truly neutral ────────────────────────────────────────────────
    ("T55","G11","Caffeine",               "Cn1cnc2c1c(=O)n(C)c(=O)n2C",            7.4, 0, None,   {}),
    ("T56","G11","Cholesterol",            "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",7.4,0,None,{}),
    ("T57","G11","Glucose",                "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",7.4, 0, None,   {}),
    ("T58","G11","Benzene",                "c1ccccc1",                              7.4, 0, None,   {}),

    # ── G12: Drug regression panel ────────────────────────────────────────
    ("T59","G12","Erlotinib",              "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",7.4,0,"[n-]",{}),
    ("T60","G12","Gefitinib",              "COc1cc2c(cc1OCCCN1CCOCC1)ncnc2Nc1ccc(F)c(Cl)c1",7.4,+1,None,{}),
    ("T61","G12","Imatinib",               "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",7.4,+1,None,{}),
    ("T62","G12","Osimertinib",            "C=CC(=O)Nc1cc(-c2cn(C)c3ncnc(Nc4ccc(F)c(Cl)c4)c23)ccn1",7.4,0,None,{}),
    ("T63","G12","Atorvastatin",           "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c(-c2ccccc2)c(C(=O)Nc2ccccc2F)c1CC(=O)O",7.4,-2,None,{}),
    ("T64","G12","Methotrexate",           "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1",7.4,-2,None,{}),
    ("T65","G12","Ciprofloxacin",          "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",7.4,0,None,{}),
]

TESTS = [
    dict(id=t[0], group=t[1], name=t[2], smiles=t[3],
         ph=t[4], expected_charge=t[5],
         must_not_contain=t[6], pubchem_mock=t[7])
    for t in TESTS_RAW
]


# ── Loader ─────────────────────────────────────────────────────────────────
def load_module(path: str):
    spec = importlib.util.spec_from_file_location("pkanet", path)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"❌ Failed to load {path}:\n   {e}")
        sys.exit(1)
    return mod


# ── Runner ─────────────────────────────────────────────────────────────────
def run_group(mod, tests: list[dict]) -> tuple[int, int]:
    passed = failed = 0
    W_id, W_name = 4, 30
    print(f"\n  {'ID':<{W_id}} {'Name':<{W_name}} {'Exp':>4}  {'Got':>4}  "
          f"{'Fragment':<18}  Result")
    print("  " + "-" * 72)

    for t in tests:
        tid     = t["id"]
        name    = t["name"]
        smi     = t["smiles"]
        ph      = t["ph"]
        exp_q   = t["expected_charge"]
        no_frag = t.get("must_not_contain")
        pubchem = t.get("pubchem_mock") or {}

        try:
            ranked, *_ = mod.generate_ranked_microstates(
                smi,
                target_ph=ph,
                pubchem_result=pubchem if pubchem else None,
            )
        except Exception as e:
            print(f"  {tid:<{W_id}} {name:<{W_name}} {exp_q:>+4}   ERR  "
                  f"{'exception':<18}  ❌ ERROR")
            print(f"    {str(e)[:70]}")
            failed += 1
            continue

        if not ranked:
            print(f"  {tid:<{W_id}} {name:<{W_name}} {exp_q:>+4}   N/A  "
                  f"{'no microstates':<18}  ❌ FAIL")
            failed += 1
            continue

        top     = ranked[0]
        got_q   = top.get("charge", top.get("net_charge", "?"))
        top_smi = top.get("smiles", "")

        charge_ok = (got_q == exp_q)
        frag_ok   = True
        frag_msg  = "—"
        if no_frag:
            if no_frag in top_smi:
                frag_ok  = False
                frag_msg = f"HAS {no_frag} ❌"
            else:
                frag_msg = f"no {no_frag} ✅"

        ok = charge_ok and frag_ok
        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        name_trunc = (name[:W_name-1] + "…") if len(name) > W_name else name
        print(f"  {tid:<{W_id}} {name_trunc:<{W_name}} {exp_q:>+4}  {got_q:>+4}  "
              f"{frag_msg:<18}  {status}")

        if not ok:
            s = (top_smi[:65] + "…") if len(top_smi) > 65 else top_smi
            print(f"    smiles: {s}")

    return passed, failed


# ── Summary ────────────────────────────────────────────────────────────────
def print_summary(results: dict[str, tuple[int,int]]):
    total_p = total_f = 0
    print(f"\n{'='*60}")
    print(f"{'Group':<12} {'Description':<35} {'Pass':>5}  {'Fail':>5}")
    print("-"*60)
    for grp, (p, f) in sorted(results.items()):
        total_p += p; total_f += f
        bar   = "✅" if f == 0 else "❌"
        label = GROUP_LABELS.get(grp, grp)
        label_t = (label[:34] + "…") if len(label) > 34 else label
        print(f"{grp:<12} {label_t:<35} {p:>5}  {f:>5}  {bar}")
    total = total_p + total_f
    print("-"*60)
    print(f"{'TOTAL':<48} {total_p:>5}  {total_f:>5}")
    print(f"\n{'✅ ALL PASS' if total_f == 0 else f'❌ {total_f}/{total} FAILED'}")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    src_path = sys.argv[1]
    if not Path(src_path).exists():
        print(f"❌ File not found: {src_path}")
        sys.exit(1)

    filter_group = sys.argv[2].upper() if len(sys.argv) > 2 else None
    if filter_group and filter_group not in GROUP_LABELS:
        print(f"❌ Unknown group '{filter_group}'. Valid: {', '.join(sorted(GROUP_LABELS))}")
        sys.exit(1)

    print(f"Loading {src_path} …")
    mod = load_module(src_path)

    selected = [t for t in TESTS if filter_group is None or t["group"] == filter_group]
    groups   = sorted(set(t["group"] for t in selected))
    total    = len(selected)
    print(f"Running {total} tests"
          + (f" (group {filter_group})" if filter_group else " across 12 groups") + " …")

    results: dict[str, tuple[int,int]] = {}
    for grp in groups:
        label = GROUP_LABELS[grp]
        grp_tests = [t for t in selected if t["group"] == grp]
        print(f"\n── {grp}: {label} ({len(grp_tests)} tests) " + "─"*20)
        p, f = run_group(mod, grp_tests)
        results[grp] = (p, f)

    print_summary(results)
    sys.exit(0 if all(f == 0 for _, f in results.values()) else 1)
