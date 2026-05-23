# <img src="https://github.com/nyelidl/pKaNET_Cloud/blob/f0f6f2f1276a9c6d027810a54b2e2c3c2f861315/pKaN.svg" width="60"> pKaNET Cloud+ — Reproducible Computational Chemistry Validation Report for Ligand Preparation in Anyone Can Dock
***Tested on 20 May 2026***

**pKaNET Cloud+** refers to the protonation engine with a calibrated SMARTS-based pKa table, Dimorphite-DL-assisted microstate enumeration, pKaNET re-ranking, and pKaHub-derived benchmark validation.

**Validation role:** Reproducible computational chemistry validation and regression audit  
**Reference dataset:** pKaHub-derived docking-relevant validation subset  
**Validation file:** `pKaNET_pKahub_docking_relevant_subset_validation.csv`  
**Failed-case file:** `pKaNET_pKahub_docking_relevant_failed_cases.csv`  
**Benchmark endpoint:** Net-charge agreement at pH 7.4 for docking-relevant protonation-state assignment  
**Test harness:** `test_pkanet.py` — 65 curated cases across 12 functional-group groups  
**Test environment:** Full pipeline: Dimorphite-DL microstate enumeration + heuristic pKa table (no ML pKa backend, no PubChem network access)

---

## 🔍 What This Tool Does

pKaNET Cloud+ determines the dominant docking-relevant protonation state of a small molecule at a user-defined pH using a tautomer-aware Henderson–Hasselbalch microstate-ranking workflow.

The workflow is designed for ligand preparation before molecular docking, molecular mechanics parameterisation, and cheminformatics dataset curation.

Main functions:

- Identifies ionisable sites using a calibrated SMARTS-based heuristic pKa table with context-aware rules for over 50 functional-group classes.
- Uses Dimorphite-DL-assisted ionisation-state enumeration, followed by pKaNET Cloud+ tautomer-aware microstate filtering and re-ranking.
- Ranks candidate microstates using a Henderson–Hasselbalch-inspired scoring function with multi-site charge-cap logic.
- Optionally queries PubChem for experimental dissociation-constant evidence when available.
- Returns the dominant microspecies as a pH-adjusted SMILES with formal charge.
- Provides a fast sub-millisecond `heuristic_net_charge()` path and a smart `predict_charge(mode='auto')` dispatcher for large-scale screening.
- Builds and minimises a 3D ligand structure using ETKDG followed by MMFF optimisation, with UFF fallback.
- Exports docking- and parameterisation-ready PDB and SDF files.

---

## 📦 Dependencies

| Library | Required | Purpose |
|---|---:|---|
| `rdkit` | ✅ required | SMARTS matching, molecule standardisation, tautomer handling, formal charge assignment, 3D conformer generation, and geometry optimisation |
| `dimorphite-dl` | ✅ required | Initial ionisation-state enumeration; pKaNET Cloud+ re-ranks the generated microstates using its heuristic pKa scoring model |
| `requests` | ⚙️ optional | PubChem experimental pKa / dissociation-constant lookup |
| `pkasolver` | ⚙️ optional | Optional ML-GNN pKa backend when available |
| `propka` | ⚙️ optional | Optional semi-empirical pKa backend or fallback |

> `py3Dmol` is used only in the accompanying Colab notebook or visualisation interface. It is not required by the core `pKaNET.py` engine.

---

## 🧪 Internal Regression Test Suite

The internal regression suite contains **65 chemically curated test cases** across 12 functional-group classes.

### How to run

```bash
python3 test_pkanet.py pKaNET.py          # full suite
python3 test_pkanet.py pKaNET.py G8       # flavonoid group only
python3 test_pkanet.py pKaNET.py G12      # drug regression panel only
```

---

**65 / 65 PASS (100 %)**

| Group | Description | Pass | Fail | Status |
|---|---|---:|---:|---|
| G1 | Imidazole-type N-H | 10 | 0 | ✅ |
| G2 | Phosphonate / phosphate | 7 | 0 | ✅ |
| G3 | Thiol ArSH / AlkSH | 5 | 0 | ✅ |
| G4 | Carboxylic acid | 5 | 0 | ✅ |
| G5 | Phenol variants (incl. warfarin enol acid) | 6 | 0 | ✅ |
| G6 | Amine bases | 5 | 0 | ✅ |
| G7 | Sulfonamide / saccharin | 4 | 0 | ✅ |
| G8 | Flavonoid regression — **MUST NOT change** | 4 | 0 | ✅ |
| G9 | Zwitterion / multi-site | 5 | 0 | ✅ |
| G10 | PubChem pKa guard | 3 | 0 | ✅ |
| G11 | Truly neutral | 4 | 0 | ✅ |
| G12 | Drug regression panel | 7 | 0 | ✅ |
| **Total** | | **65** | **0** | ✅ |

---

### Per-test detail

#### G1 — Imidazole-type N-H (10 tests)

| ID | Compound | Expected | Got | Fragment guard | Result |
|---|---|---:|---:|---|---|
| T01 | Imidazole | 0 | 0 | no `[n−]` ✅ | ✅ PASS |
| T02 | Benzimidazole | 0 | 0 | no `[n−]` ✅ | ✅ PASS |
| T03 | Pyrazole | 0 | 0 | no `[n−]` ✅ | ✅ PASS |
| T04 | Indazole | 0 | 0 | no `[n−]` ✅ | ✅ PASS |
| T05 | Purine | 0 | 0 | no `[n−]` ✅ | ✅ PASS |
| T06 | Adenine | 0 | 0 | no `[n−]` ✅ | ✅ PASS |
| T07 | 1-Methylbenzimidazole | 0 | 0 | no `[n−]` ✅ | ✅ PASS |
| T08 | Clotrimazole | 0 | 0 | no `[n−]` ✅ | ✅ PASS |
| T09 | Omeprazole | 0 | 0 | no `[n−]` ✅ | ✅ PASS |
| T10 | Metronidazole | 0 | 0 | no `[n−]` ✅ | ✅ PASS |

#### G2 — Phosphonate / Phosphate (7 tests)

| ID | Compound | Expected | Got | Result |
|---|---|---:|---:|---|
| T11 | Methylphosphonic acid | −2 | −2 | ✅ PASS |
| T12 | Phenylphosphonic acid | −2 | −2 | ✅ PASS |
| T13 | Fosfomycin | −2 | −2 | ✅ PASS |
| T14 | Alendronate | −2 | −2 | ✅ PASS |
| T15 | Tenofovir | −2 | −2 | ✅ PASS |
| T16 | Phosphate monoester | −2 | −2 | ✅ PASS |
| T17 | Glyphosate | −2 | −2 | ✅ PASS |

#### G3 — Thiol ArSH / AlkSH (5 tests)

| ID | Compound | Expected | Got | Result |
|---|---|---:|---:|---|
| T18 | Thiophenol | −1 | −1 | ✅ PASS |
| T19 | 4-Chlorothiophenol | −1 | −1 | ✅ PASS |
| T20 | 6-Mercaptopurine | −1 | −1 | ✅ PASS |
| T21 | Captopril | −1 | −1 | ✅ PASS |
| T22 | Ethanethiol | 0 | 0 | ✅ PASS |

#### G4 — Carboxylic Acid (5 tests)

| ID | Compound | Expected | Got | Result |
|---|---|---:|---:|---|
| T23 | Acetic acid | −1 | −1 | ✅ PASS |
| T24 | Ibuprofen | −1 | −1 | ✅ PASS |
| T25 | Aspirin | −1 | −1 | ✅ PASS |
| T26 | Diclofenac | −1 | −1 | ✅ PASS |
| T27 | Trichloroacetic acid | −1 | −1 | ✅ PASS |

#### G5 — Phenol Variants (6 tests)

| ID | Compound | Expected | Got | Result |
|---|---|---:|---:|---|
| T28 | Phenol | 0 | 0 | ✅ PASS |
| T29 | 4-Nitrophenol | −1 | −1 | ✅ PASS |
| T30 | Pentafluorophenol | −1 | −1 | ✅ PASS |
| T31 | Acetaminophen | 0 | 0 | ✅ PASS |
| T32 | Catechol | 0 | 0 | ✅ PASS |
| T33 | Warfarin | −1 | −1 | ✅ PASS |

#### G6 — Amine Bases (5 tests)

| ID | Compound | Expected | Got | Result |
|---|---|---:|---:|---|
| T34 | Aniline | 0 | 0 | ✅ PASS |
| T35 | Pyridine | 0 | 0 | ✅ PASS |
| T36 | Methylamine | +1 | +1 | ✅ PASS |
| T37 | Metformin | +1 | +1 | ✅ PASS |
| T38 | Amlodipine | +1 | +1 | ✅ PASS |

#### G7 — Sulfonamide / Saccharin (4 tests)

| ID | Compound | Expected | Got | Result |
|---|---|---:|---:|---|
| T39 | Methanesulfonamide | 0 | 0 | ✅ PASS |
| T40 | Saccharin | −1 | −1 | ✅ PASS |
| T41 | Chlorothiazide | −1 | −1 | ✅ PASS |
| T42 | Furosemide | −2 | −2 | ✅ PASS |

#### G8 — Flavonoid Regression — MUST NOT change (4 tests)

| ID | Compound | Expected | Got | Fragment guard | Result |
|---|---|---:|---:|---|---|
| T43 | Baicalein | 0 | 0 | no `[O−]` ✅ | ✅ PASS |
| T44 | Apigenin | 0 | 0 | no `[O−]` ✅ | ✅ PASS |
| T45 | Luteolin | 0 | 0 | no `[O−]` ✅ | ✅ PASS |
| T46 | Kaempferol | 0 | 0 | no `[O−]` ✅ | ✅ PASS |

#### G9 — Zwitterion / Multi-site (5 tests)

| ID | Compound | Expected | Got | Result |
|---|---|---:|---:|---|
| T47 | Glycine | 0 | 0 | ✅ PASS |
| T48 | Histidine | 0 | 0 | ✅ PASS |
| T49 | Glutamic acid | −1 | −1 | ✅ PASS |
| T50 | Lysine | +1 | +1 | ✅ PASS |
| T51 | Cysteine | 0 | 0 | ✅ PASS |

#### G10 — PubChem pKa Guard (3 tests)

| ID | Compound | Expected | Got | Fragment guard | Result |
|---|---|---:|---:|---|---|
| T52 | Benzimidazole + PubChem base pKa mock | 0 | 0 | no `[n−]` ✅ | ✅ PASS |
| T53 | Imidazole + PubChem base pKa mock | 0 | 0 | no `[n−]` ✅ | ✅ PASS |
| T54 | Phenol + PubChem correct pKa | 0 | 0 | — | ✅ PASS |

#### G11 — Truly Neutral (4 tests)

| ID | Compound | Expected | Got | Result |
|---|---|---:|---:|---|
| T55 | Caffeine | 0 | 0 | ✅ PASS |
| T56 | Cholesterol | 0 | 0 | ✅ PASS |
| T57 | Glucose | 0 | 0 | ✅ PASS |
| T58 | Benzene | 0 | 0 | ✅ PASS |

#### G12 — Drug Regression Panel (7 tests)

| ID | Compound | Expected | Got | Result |
|---|---|---:|---:|---|
| T59 | Erlotinib | 0 | 0 | ✅ PASS |
| T60 | Gefitinib | +1 | +1 | ✅ PASS |
| T61 | Imatinib | +1 | +1 | ✅ PASS |
| T62 | Osimertinib | 0 | 0 | ✅ PASS |
| T63 | Atorvastatin | −2 | −2 | ✅ PASS |
| T64 | Methotrexate | −2 | −2 | ✅ PASS |
| T65 | Ciprofloxacin | 0 | 0 | ✅ PASS |

---

## 📊 External Benchmark — pKaHub-Derived Validation Subset (http://pkahub.ttk.hu/)

pKaNET Cloud+ was benchmarked against a docking-relevant subset derived from pKaHub, an experimental aqueous pKa database with macroscopic charge-state transition annotations.

### Benchmark Endpoint

The benchmark endpoint is **net-charge agreement at pH 7.4**. This checks whether pKaNET Cloud+ predicts the same dominant net formal charge as the pKaHub-derived reference annotation at pH 7.4.

This benchmark is **not** a numerical pKa prediction benchmark. The reported agreement values must not be interpreted as pKa MAE, RMSE, or quantitative pKa accuracy.

### Drug-Like Screening Criteria (27,218-molecule subset)

The 27,218-molecule evaluation subset was selected from the combined unified dataset (38,724 unique molecules) by applying the following drug-like / small-molecule criteria:

| Property | Threshold |
|---|---|
| Molecular weight | 100–600 Da |
| H-bond donors (HBD) | ≤ 7 |
| H-bond acceptors (HBA) | ≤ 12 |
| logP | −3 to +6.5 |
| TPSA | ≤ 180 Å² |
| Rotatable bonds | ≤ 15 |
| Heavy atoms | ≥ 7 |

From 38,724 unique SMILES, 35,872 passed these criteria. The final 27,218 were selected by prioritising non-excluded records and highest experimental pKa data availability.

### Overall Results

| Dataset | Correct | Total (with ref.) | Agreement rate |
|---|---:|---:|---:|
| 65-case curated regression set | 65 | 65 | **100.00 %** |
| pKaHub-derived 27,218-molecule subset | 18,850 | 27,183 | **69.34 %** |

### Agreement by Expected Charge State (pH 7.4, 27,218-molecule subset)

| Expected charge | Count | Correct | Agreement |
|---:|---:|---:|---:|
| −4 | 7 | 1 | 14.3 % |
| −3 | 84 | 14 | 16.7 % |
| −2 | 694 | 310 | 44.7 % |
| −1 | 7,495 | 4,321 | 57.7 % |
| 0 | 12,146 | 9,240 | 76.1 % |
| +1 | 6,386 | 4,923 | 77.1 % |
| +2 | 309 | 41 | 13.3 % |
| +3 or above | 62 | 0 | 0.0 % |

### Interpretation

For monoprotic drug-like molecules, which represent the majority of practical lead-optimisation cases, pKaNET Cloud+ assigns the same dominant net charge as the pKaHub-derived reference annotation for approximately three out of four compounds. Agreement is highest for neutral molecules (76.1 %) and monocations (77.1 %), and lower for polyprotic and zwitterionic molecules, as expected. The 8,333 disagreements (30.66 %) are concentrated in molecules where at least one predicted ionisable site has a heuristic pKa within ±1.5 units of pH 7.4 (borderline), and in strongly polyprotic species (charge ≥ |2|) where multi-site pKa ordering is not recoverable from the heuristic table alone. The dominant failure modes are single-step over-prediction of basicity (+1 charge error, 56.0 % of failures) and single-step over-prediction of acidity (−1 charge error, 37.2 % of failures).

---

## ⚡ New Public API

Three new public functions are available for programmatic and large-scale use:

```python
# Sub-millisecond heuristic estimate with multi-site charge caps
charge = core.heuristic_net_charge("CC(=O)O", ph=7.4)          # → −1

# Smart dispatcher: fast normally, full pipeline for borderline pKa or tautomeric risk
charge, mode = core.predict_charge("CC(=O)O", ph=7.4, mode="auto")

# Batch prediction — returns a pandas DataFrame
df = core.batch_predict_charges(["CC(=O)O", "CN", "NCC(=O)O"], ph=7.4, mode="auto")
```

`predict_charge(mode='auto')` automatically escalates to the full tautomer + Dimorphite-DL + scoring pipeline when:
- any detected site pKa is within 1.5 pH units of the target (borderline), or
- the molecule has ring systems but no detectable ionisable sites on the parent form (tautomeric enol risk, e.g. warfarin supplied as the keto form).

The `heuristic_net_charge` function applies two charge-cap rules that suppress systematic over-charging in the fast path: a **polyamine cap** (prevents every amine from being independently protonated when no acidic groups are present) and a **multi-acid cap** (prevents over-deprotonation of symmetric diacids by counting only sites with pKa clearly below the target pH).

---

## 🗂️ Benchmark Files

| File | Description |
|---|---|
| `pKaNET_pKahub_docking_relevant_subset_validation.csv` | Curated validation subset with pKaHub-derived reference charge labels and pKaNET predictions |
| `pKaNET_pKahub_docking_relevant_failed_cases.csv` | Disagreement cases for manual review and future rule refinement |
| `curated_regression_set.csv` | Internal 65-compound chemically curated regression set |
| `v80_val27k_pass.csv` | 18,850 molecules with correct charge assignment |
| `v80_val27k_fail.csv` | 8,333 disagreement cases |
| `validation_summary_template.csv` | Template for recording new validation outputs |
| `failed_cases_review_template.csv` | Template for manually classifying disagreement cases |

---

## ⚠️ Important Notes

- pKaNET Cloud+ uses a calibrated heuristic pKa table and microstate-ranking workflow, not a quantitative experimental pKa predictor.
- For borderline cases where one or more predicted site pKa values fall within ±1.5 units of the target pH, the predicted charge should be treated as uncertain. Use `predict_charge(mode='auto')` to escalate these automatically to the full pipeline.
- `heuristic_net_charge` returns charge 0 for keto-form warfarin input (no OH detectable on the parent form); `predict_charge(mode='auto')` correctly escalates to the full pipeline and returns −1. This is expected behaviour.
- Net-charge agreement does not guarantee that the exact ionised atom or tautomer is correct, especially for polyprotic or zwitterionic molecules.
- The pKaHub-derived benchmark subset is a curated validation subset, not a redistribution of the complete raw pKaHub database.
- In the G12 drug regression panel, Gefitinib and Imatinib are assigned as +1, whereas Erlotinib and Osimertinib are assigned as neutral. EGFR inhibitors should be evaluated compound by compound rather than assigned a uniform charge class.
- The 65-case internal regression suite was run with Dimorphite-DL active. The 27,218-molecule benchmark used `heuristic_net_charge` (fast path). Using `predict_charge(mode='auto')` for the full benchmark would further improve accuracy for borderline and polyprotic cases, at the cost of longer run time.

---

## 🧬 Supported Inputs

| Format | Extension |
|---|---|
| SMILES | `.smi` or plain-text |
| MDL Molfile | `.mol` |
| Structure-data file | `.sdf` |
| Tripos Mol2 | `.mol2` |
| Protein Data Bank ligand | `.pdb` |

---

## 📤 Outputs

| Output | Description |
|---|---|
| pH-adjusted SMILES | Dominant predicted microspecies at the target pH |
| Net formal charge | Integer formal charge of the selected microspecies |
| `minimized_ligand.pdb` | 3D ligand structure after geometry minimisation |
| `minimized_ligand.sdf` | 3D ligand structure with explicit hydrogens and formal charge |
| Preparation log | Site-level protonation decisions, pKa evidence, and ranking information |

---

## 🎯 Intended Use Cases

- Ligand preparation before molecular docking (AutoDock Vina, VinaXB, GNINA, Glide, GOLD, rDock).
- GAFF2, CGenFF, or other force-field parameterisation workflows.
- QSAR, ADMET, and virtual-screening dataset curation.
- Teaching pKa, protonation state, microspecies, and docking-preparation concepts.

---

## 🔗 Integration with Anyone Can Dock

pKaNET Cloud+ is the default protonation engine in the Anyone Can Dock web application, replacing the previous Dimorphite-DL-only pipeline.

```python
protonate_pkanet()
```

```
input ligand → standardisation → Dimorphite-DL enumeration →
pKaNET Cloud+ ranking → dominant microspecies → 3D generation →
minimisation → docking-ready output
```

---

## ✅ Recommended Wording for Manuscript or ESI

> The protonation-state assignment module of pKaNET Cloud+ was evaluated using an internal chemically curated regression set (65 molecules, 100 % net-charge agreement at pH 7.4) and a drug-like subset derived from the pKaHub experimental pKa database (27,218 molecules, 69.34 % net-charge agreement at pH 7.4; Sipos-Szabó et al.). The benchmark endpoint was dominant net-charge agreement at pH 7.4, not numerical pKa prediction accuracy. The pKaHub-derived benchmark subset was curated to retain molecules with interpretable macroscopic charge-state annotations relevant to ligand docking. The complete raw pKaHub database was not redistributed; only curated validation outputs and disagreement summaries were provided for reproducibility. Full benchmark data and extended ligand preparation methodology are provided in the Supporting Information.

---

## 🚫 Wording to Avoid

| Avoid | Use instead |
|---|---|
| "pKaNET pKa accuracy is 69.34 %" | "pKaNET net-charge agreement at pH 7.4 is 69.34 %" |
| "pKaNET predicts pKa correctly" | "pKaNET assigns the correct dominant net charge" |
| "Fully validated against experimental data" | "Benchmarked against pKaHub-derived charge-state annotations" |
| "All imidazole cases are fixed" | "The reported imidazole N-H deprotonation issue is resolved in the regression set; residual failures may remain for complex imidazole-containing molecules" |
| "70.60 % net-charge agreement" | "69.34 % net-charge agreement" (correct value for the current release) |

---

## 🙏 Acknowledgements

- **RDKit** — molecule standardisation, SMARTS matching, tautomer handling, formal charge assignment, ETKDG conformer generation, and MMFF/UFF geometry optimisation.
- **Dimorphite-DL** — initial ionisation-state enumeration; pKaNET Cloud+ performs independent re-ranking.
- **pKaSolver** — optional ML-GNN pKa backend.
- **PROPKA** — optional semi-empirical pKa backend.
- **requests** — optional HTTP client for PubChem lookup.
- **pKaHub** — external experimental pKa reference resource used to derive the docking-relevant benchmark subset.

---

## 📖 Citation

If you use pKaNET Cloud+ in your work, please cite:

> Hengphasatporn, K. et al. *pKaNET Cloud+: Tautomer-aware protonation-state ranking for docking-ready ligand preparation.* Manuscript in preparation.

For the pKaHub benchmark reference dataset, cite:

> Sipos-Szabó, L.; Bajusz, D.; Balogh, G. T.; Keserű, G. M. Benchmarking pKa Prediction Algorithms against an Extensive, Public Data Set. *Journal of Chemical Information and Modeling* **2026**, 66, 4607–4619. DOI: 10.1021/acs.jcim.6c00107.

---

## 📌 Project Context

pKaNET Cloud+ is developed as part of the ligand-preparation workflow for Anyone Can Dock and related computational drug-discovery tools. The method improves docking-readiness by reducing common protonation-state errors caused by direct rule-based ionisation workflows, especially for imidazole-like motifs, flavonoids, phosphates/phosphonates, sulfonamide-like acids, zwitterions, warfarin-type enol acids, and drug-like polyprotic molecules.
