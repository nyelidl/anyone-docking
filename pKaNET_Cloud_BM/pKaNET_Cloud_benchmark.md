# 🧪 pKaNET Cloud+ — Independent Computational Chemistry Audit Report for Ligand Preparation in Anyone Can Dock!

**Auditor role:** Independent cheminformatics cross-check  
**Reference dataset:** pKaHub (Sipos-Szabó et al., *J. Chem. Inf. Model.* 2026, 66, 4607–4619)  
**Validation file:** `pKaNET_pKahub_docking_relevant_subset_validation_v68.csv`  
**Failed-case file:** `pKaNET_pKahub_docking_relevant_failed_cases_v68.csv`  
**Benchmark endpoint:** Net-charge agreement at pH 7.4 (docking-relevant protonation state)

---

## 🔍 What This Tool Does

pKaNET Cloud+ determines the **dominant docking-relevant protonation state** of a small molecule at a user-defined pH using a tautomer-aware Henderson-Hasselbalch microstate ranking engine.

- Identifies all ionizable sites via a corrected SMARTS-based pKa table
- Enumerates tautomers and filters implausible forms
- Ranks microstates using an 8-component Henderson-Hasselbalch scoring function
- Optionally queries **PubChem** for experimental pKa evidence
- Returns the dominant microspecies as a **pH-adjusted SMILES** with formal charge
- Builds and minimizes a **3D structure** (ETKDG → MMFF → UFF fallback)
- Exports **PDB** and **SDF** formats ready for docking or parameterization

---

## 📦 Dependencies

| Library | Required | Purpose |
|---|---|---|
| `rdkit` | ✅ required | Substructure matching, tautomer enumeration, 3D structure generation |
| `dimorphite-dl` | ✅ required | Ionization-state enumeration (primary enumerator) |
| `requests` | ⚙️ optional | PubChem experimental pKa lookup |
| `pkasolver` | ⚙️ optional | ML GNN pKa backend (improves accuracy when available) |
| `propka` | ⚙️ optional | Semi-empirical pKa backend (fallback when pkasolver absent) |

> **Note:** `py3Dmol` for 3D visualization is used in the accompanying **Colab notebook** but is not part of `pKaNET.py` itself.

---

## ✅ Bug Fixes (2026-05)

This release corrects systematic errors in the ionizable-site heuristic table that caused wrong protonation states for several common pharmacophores. All fixes are verified by the regression test suite (`test_pkanet.py`, 65/65 pass).

| Bug | Old behaviour | Fixed behaviour |
|---|---|---|
| Imidazole N-H pKa | pKa = 6.0 (base pKa used as acid pKa) → **[n−] at pH 7.4** | pKa = 14.0 → **neutral at pH 7.4** |
| Benzimidazole N-H pKa | pKa = 5.5 (base pKa) → [n−] at pH 7.4 | pKa = 13.0 → neutral at pH 7.4 |
| Imidazole SMARTS | `c1cn[nH]c1` matched pyrazole, not imidazole | `[nH]1ccnc1` correctly matches imidazole |
| Flavonoid 7-OH pKa | pKa = 7.0 → **[O−] at pH 7.4** for all flavones | pKa = 9.0 (isolated) / 8.5 (catechol pair) → **neutral at pH 7.4** |
| Warfarin enol acid | SMARTS never matched → charge 0 instead of −1 | Fixed to match active methylene of 4-hydroxycoumarin scaffold → **charge −1** |
| Methotrexate charge | Pteridine site pKa = 6.8 → over-deprotonated (−3) | pKa raised to 8.5 → correct **charge −2** (2 × COOH only) |
| Phosphonate double-counting | pKa₁ and pKa₂ sites counted twice | Deduplicated |
| Thiol aliphatic pKa | pKa = 10.5 | pKa = 9.8 |
| Thiol α-amino pKa | Missing entry | Added: pKa = 8.3 |
| Hydroxamic acid | Missing entry | Added |
| Phosphate diester | Wrong label and pKa | Corrected |
| Catechol-OH priority | Unordered | Fixed selection priority |
| Amine α-EWG | Missing entry | Added: pKa = 7.5 |
| New site entries | — | sulfonyl_imide_NH, enol_lactone, enol_cyclic_dicarbonyl, enol_1_3_dicarbonyl, pyrazole_NH, indole_NH, aliphatic_imine |

**Imidazole case confirmed fixed:** pH 7.4 → rank-1 score +2.30 (neutral) vs −0.95 ([n−]); margin = 3.25 units (unambiguous).

---

## 🧪 Regression Test Suite

A 65-test regression suite (`test_pkanet.py`) covers 12 functional-group groups to guard against regressions and verify expected protonation states at pH 7.4.

```bash
python3 test_pkanet.py pKaNET.py          # run all 65 tests
python3 test_pkanet.py pKaNET.py G8       # run one group only
```

| Group | Description | Tests |
|---|---|---|
| G1 | Imidazole-type N-H (imidazole, benzimidazole, pyrazole, purine, drugs) | 9 |
| G2 | Phosphonate / phosphate | 8 |
| G3 | Thiol ArSH / AlkSH | 5 |
| G4 | Carboxylic acid | 5 |
| G5 | Phenol variants (incl. Warfarin) | 6 |
| G6 | Amine bases | 5 |
| G7 | Sulfonamide / saccharin | 4 |
| G8 | Flavonoid regression — **MUST NOT change** | 4 |
| G9 | Zwitterion / multi-site (amino acids) | 5 |
| G10 | PubChem pKa guard (base pKa must not cause [n−]) | 3 |
| G11 | Truly neutral (caffeine, cholesterol, glucose) | 4 |
| G12 | Drug regression panel (EGFR inhibitors, Methotrexate, Ciprofloxacin) | 7 |

**Current result: 65/65 ✅ ALL PASS**

---

## 📊 Benchmark — pKaHub Validation (v68)

pKaNET v68 was benchmarked against a docking-relevant subset of the **pKaHub** experimental pKa database (Sipos-Szabó et al., *J. Chem. Inf. Model.* 2026, 66, 4607–4619), which provides over 90,000 experimental aqueous pKa values annotated with macroscopic charge-state transitions.

> **Benchmark endpoint:** net-charge agreement at pH 7.4 — whether pKaNET predicts the same dominant net charge as the pKaHub-derived reference annotation.
> This is **not** a numerical pKa MAE/RMSE benchmark. Do not interpret these numbers as pKa prediction accuracy.

### Overall Results

| Version | Agreement | % |
|---|---|---|
| pKaNET v67 | 18,298 / 27,218 | 67.23% |
| **pKaNET v68** | **19,216 / 27,218** | **70.60%** |

Version changes: +1,124 improved, −206 regressed (net +918 cases).

### Results by Molecular Complexity

| Class | n | v68 agree | v68 agree % | v68 disagree % |
|---|---|---|---|---|
| Monoprotic | 19,537 | 14,652 | **75.00%** | 25.00% |
| Amphoteric | 1,914 | 1,485 | **77.58%** | 22.42% |
| Polyprotic / complex | 5,767 | 3,079 | **53.39%** | 46.61% |

### Interpretation

For monoprotic drug-like molecules — the most common case in lead optimization — 3 in 4 compounds receive the correct dominant net charge at physiological pH. Agreement is lower for polyprotic and zwitterionic molecules, consistent with the known limitations of heuristic Henderson-Hasselbalch models when multiple ionization equilibria compete simultaneously. Remaining disagreements (8,002 cases, 29.40%) are concentrated in polyprotic/complex molecules and in borderline cases where at least one ionizable site has a predicted pKa within ±1 unit of pH 7.4 (n = 1,250; 15.6% of failures).

### Benchmark Files

| File | Description |
|---|---|
| `pKaNET_pKahub_docking_relevant_subset_validation_v68.csv` | Full validation set (27,218 molecules) |
| `pKaNET_pKahub_docking_relevant_failed_cases_v68.csv` | v68 disagreement cases (8,002 molecules) |

Each row includes: `molid`, `smiles`, `pkahub_charge`, `pkanet_v67_charge`, `pkanet_v68_charge`, `agreement_v67`, `agreement_v68`, `transition_class`, `pkanet_v68_site_count`, and `pkanet_v68_site_decisions` (semicolon-delimited list of `site_label:heuristic_pKa:site_type` entries).

### Limitations

- The pKaHub reference is macroscopic; for polyprotic molecules the dominant net charge may differ from what any heuristic single-site model predicts.
- No ML pKa backend was used in generating the benchmark CSV; all predictions rely on the heuristic SMARTS table.
- Agreement does not guarantee correct identification of the specific ionized atom — only that the overall net charge matches.
- Basic amine groups (pKa ~8–9) are predicted as protonated (+1) at pH 7.4, which is chemically correct (see Important Notes below).

### Wording to Avoid

| ❌ Avoid | ✅ Use instead |
|---|---|
| "pKaNET pKa accuracy is 70%" | "net-charge agreement at pH 7.4 is 70.60%" |
| "pKaNET predicts pKa correctly" | "pKaNET assigns the correct dominant net charge" |
| "fully validated against experimental data" | "benchmarked against pKaHub-derived charge-state annotations" |
| "all imidazole cases are fixed" | "the reported imidazole N-H deprotonation bug is resolved; residual failures exist for polyprotic imidazole-containing molecules" |

---

## 📥 Supported Inputs

| Format | Extension |
|---|---|
| SMILES | `.smi` or plain text |
| MDL Molfile | `.mol`, `.sdf` |
| Tripos Mol2 | `.mol2` |
| Protein Data Bank | `.pdb` |

---

## 📤 Outputs

| Output | Description |
|---|---|
| pH-adjusted SMILES | Dominant microspecies at target pH |
| Predicted net formal charge | Integer charge at target pH |
| `minimized_ligand.pdb` | 3D structure with MMFF-minimized geometry |
| `minimized_ligand.sdf` | 3D structure with explicit H (correct rendering in py3Dmol) |
| Preparation log | Site-level decisions and pKa evidence |

---

## 🎯 Ideal For

- Ligand preparation prior to **molecular docking** (AutoDock Vina, Glide, GOLD, rDock)
- **GAFF2 / CGenFF** force-field parameterization
- **QSAR / ADMET** dataset curation
- **Virtual screening** library protonation
- Teaching **pKa and microspecies** concepts in drug design courses

---

## 🔗 Integration

pKaNET v68 is the **default protonation engine** in the [Anyone Can Dock](https://anyonecandock.streamlit.app/) web application, replacing the previous Dimorphite-DL-based pipeline. It is called via `protonate_pkanet()` in `core.py` and can be loaded standalone by placing `pKaNET.py` in the same directory as `core.py`.

---

## ⚠️ Important Notes

- pKaNET uses a **heuristic pKa table**, not a trained ML model. Predictions are based on SMARTS pattern matching and Henderson-Hasselbalch scoring.
- For borderline cases (site pKa within ±1 unit of target pH), the predicted charge should be treated as uncertain.
- Do not use pKaNET heuristic pKa values as quantitative experimental pKa estimates; they are calibrated for charge-state selection, not numerical accuracy.
- Drugs with basic amine groups (pKa 8–9) — such as EGFR inhibitors (Gefitinib, Imatinib, Afatinib) — are predicted as **protonated (+1) at pH 7.4**. This is chemically correct (e.g., Gefitinib morpholine pKa ≈ 8.0 → ~80% protonated at pH 7.4) and matches docking-convention for salt-bridge formation.

---

## 🙏 Acknowledgements

This tool builds on:

- **[RDKit](https://www.rdkit.org/)** — SMARTS-based substructure matching, tautomer enumeration (EnumerateStereoisomers), standardization (rdMolStandardize), ETKDGv3 conformer generation, and MMFF geometry optimization.
- **[Dimorphite-DL](https://github.com/rdkit/Dimorphite-DL)** (Ropp et al., *J. Cheminformatics* 2019, 11, 14) — used as the primary ionization-state enumerator inside `generate_ranked_microstates`. The heuristic pKa table in pKaNET independently scores and re-ranks the enumerated states.
- **[pKaSolver](https://github.com/mayrf/pkasolver)** (Mayr et al., *Front. Chem.* 2022) — optional ML GNN backend; when installed, provides graph-neural-network pKa estimates to augment heuristic scoring.
- **[PROPKA](https://github.com/jensengroup/propka)** (Olsson et al.) — optional semi-empirical backend; used as fallback when pKaSolver is unavailable.
- **[requests](https://requests.readthedocs.io/)** — optional HTTP client for PubChem dissociation-constant lookup.
- **[pKaHub](http://pkahub.ttk.hu)** (Sipos-Szabó, Bajusz, Balogh, Keserű, *J. Chem. Inf. Model.* 2026, 66, 4607–4619) — benchmark reference dataset providing experimental pKa values with macroscopic charge-state transition annotations.

---

## 📖 Citation

If you use pKaNET Cloud in your work, please cite:

> Hengphasatporn, K. et al. *pKaNET Cloud: Tautomer-aware protonation-state ranking for docking-ready ligand preparation.* [manuscript in preparation]

For the benchmark reference dataset:

> Sipos-Szabó, L.; Bajusz, D.; Balogh, G. T.; Keserű, G. M. Benchmarking pKa Prediction Algorithms against an Extensive, Public Data Set. *J. Chem. Inf. Model.* **2026**, 66, 4607–4619. DOI: [10.1021/acs.jcim.6c00107](https://doi.org/10.1021/acs.jcim.6c00107)

---

*This code is part of the DFDD project.*

