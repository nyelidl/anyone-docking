# 🧪 pKaNET Cloud+ — Reproducible Computational Chemistry Validation Report for Ligand Preparation in Anyone Can Dock

**pKaNET Cloud+** refers to the v68 protonation engine with a corrected SMARTS-based pKa table, Dimorphite-DL-assisted microstate enumeration, pKaNET re-ranking, and pKaHub-derived benchmark validation.

**Validation role:** Reproducible computational chemistry validation and regression audit  
**Reference dataset:** pKaHub-derived docking-relevant validation subset  
**Validation file:** `pKaNET_pKahub_docking_relevant_subset_validation_v68.csv`  
**Failed-case file:** `pKaNET_pKahub_docking_relevant_failed_cases_v68.csv`  
**Benchmark endpoint:** Net-charge agreement at pH 7.4 for docking-relevant protonation-state assignment

---

## 🔍 What This Tool Does

pKaNET Cloud+ determines the dominant docking-relevant protonation state of a small molecule at a user-defined pH using a tautomer-aware Henderson–Hasselbalch microstate-ranking workflow.

The workflow is designed for ligand preparation before molecular docking, molecular mechanics parameterization, and cheminformatics dataset curation.

Main functions:

- Identifies ionizable sites using a corrected SMARTS-based heuristic pKa table.
- Uses Dimorphite-DL-assisted ionization-state enumeration, followed by pKaNET Cloud+ tautomer-aware microstate filtering and re-ranking.
- Ranks candidate microstates using a Henderson–Hasselbalch-inspired scoring function.
- Optionally queries PubChem for experimental dissociation-constant evidence when available.
- Returns the dominant microspecies as a pH-adjusted SMILES with formal charge.
- Builds and minimizes a 3D ligand structure using ETKDG followed by MMFF optimization, with UFF fallback.
- Exports docking- and parameterization-ready PDB and SDF files.

---

## 📦 Dependencies

| Library | Required | Purpose |
|---|---:|---|
| `rdkit` | ✅ required | SMARTS matching, molecule standardization, tautomer handling, formal charge assignment, 3D conformer generation, and geometry optimization |
| `dimorphite-dl` | ✅ required | Initial ionization-state enumeration; pKaNET Cloud+ re-ranks the generated microstates using its heuristic pKa scoring model |
| `requests` | ⚙️ optional | PubChem experimental pKa / dissociation-constant lookup |
| `pkasolver` | ⚙️ optional | Optional ML-GNN pKa backend when available |
| `propka` | ⚙️ optional | Optional semi-empirical pKa backend or fallback |

> `py3Dmol` is used only in the accompanying Colab notebook or visualization interface. It is not required by the core `pKaNET.py` engine.

---

## ✅ Bug Fixes in pKaNET Cloud+ v68

This release corrects systematic errors in the previous ionizable-site heuristic table that caused incorrect protonation states for several common pharmacophores. The corrected rules are verified by the internal regression test suite.

| Issue | Previous behavior | Corrected behavior |
|---|---|---|
| Imidazole N-H pKa | Base pKa was incorrectly used as acid pKa, causing artificial `[n−]` formation at pH 7.4 | Imidazole N-H is treated as weakly acidic and remains neutral at pH 7.4 |
| Benzimidazole N-H pKa | Base pKa was incorrectly interpreted as N-H acidity | Benzimidazole remains neutral at pH 7.4 |
| Imidazole SMARTS | SMARTS pattern incorrectly matched pyrazole-like motifs | Corrected imidazole-specific SMARTS pattern |
| Flavonoid phenolic OH | Over-deprotonation of flavone/polyphenol scaffolds at pH 7.4 | Flavonoid phenols remain neutral unless strong activating features justify ionization |
| Warfarin enol acid | 4-hydroxycoumarin acidic enol was missed | Warfarin-like enol acid is assigned as anionic at pH 7.4 |
| Methotrexate charge | Pteridine site caused over-deprotonation | Dominant charge is assigned from the carboxylate groups only |
| Phosphonate/phosphate | Duplicate counting of acidic sites | Deduplicated charge assignment |
| Thiol groups | Aliphatic and aromatic thiols were treated with overly generic pKa values | Separate thiol rules added or refined |
| Sulfonamide/saccharin | Several acidic sulfonamide-like motifs were missing or under-prioritized | Additional SMARTS entries added |
| Hydroxamic acid | Missing entry | Added |
| Phosphate diester | Incorrect site label and pKa behavior | Corrected |
| Catechol OH priority | Ambiguous or unordered assignment | Priority rules refined |
| Amine near electron-withdrawing groups | Missing local correction | Added reduced-basicity rule |
| Additional motifs | Several ionizable pharmacophores were absent | Added entries for sulfonyl imide N-H, enol lactone, cyclic dicarbonyl enol, 1,3-dicarbonyl enol, pyrazole N-H, indole N-H, and aliphatic imine |

The imidazole regression case is now resolved: the neutral imidazole form is ranked above the artificial deprotonated `[n−]` form at pH 7.4.

---

## 🧪 Internal Regression Test Suite

The internal regression suite contains 65 chemically curated test cases across 12 functional-group classes. It is designed to prevent regression in common docking-relevant ionization motifs.

Run all tests:

```bash
python3 test_pkanet.py pKaNET.py
```

Run one group only, for example flavonoids:

```bash
python3 test_pkanet.py pKaNET.py G8
```

Run the drug regression panel:

```bash
python3 test_pkanet.py pKaNET.py G12
```

### Regression Groups

| Group | Description | Number of tests |
|---|---|---:|
| G1 | Imidazole-type N-H motifs | 10 |
| G2 | Phosphonate / phosphate groups | 7 |
| G3 | Thiol groups | 5 |
| G4 | Carboxylic acids | 5 |
| G5 | Phenol variants, including warfarin-like enol acid | 6 |
| G6 | Amine bases | 5 |
| G7 | Sulfonamide / saccharin motifs | 4 |
| G8 | Flavonoid regression panel | 4 |
| G9 | Zwitterionic and multi-site molecules | 5 |
| G10 | PubChem pKa guard cases | 3 |
| G11 | Truly neutral molecules | 4 |
| G12 | Drug regression panel, including EGFR inhibitors and other drugs | 7 |

Current internal regression status:

```text
65 / 65 PASS
```

---

## 📊 External Benchmark — pKaHub-Derived Validation Subset

pKaNET Cloud+ v68 was benchmarked against a docking-relevant subset derived from pKaHub, an experimental aqueous pKa database with macroscopic charge-state transition annotations.

### Benchmark Endpoint

The benchmark endpoint is **net-charge agreement at pH 7.4**.

This means the validation checks whether pKaNET Cloud+ predicts the same dominant net formal charge as the pKaHub-derived reference annotation at pH 7.4.

This benchmark is **not** a numerical pKa prediction benchmark. The reported agreement values must not be interpreted as pKa MAE, RMSE, or quantitative pKa accuracy.

### Overall Results

| Version | Agreement | Agreement rate |
|---|---:|---:|
| pKaNET v67 | 18,298 / 27,218 | 67.23% |
| pKaNET Cloud+ v68 | 19,216 / 27,218 | 70.60% |

Version-level change from v67 to v68:

| Category | Number of cases |
|---|---:|
| Improved | 1,124 |
| Regressed | 206 |
| Net gain | 918 |

### Results by Molecular Complexity

| Class | n | v68 agree | v68 agree % | v68 disagree % |
|---|---:|---:|---:|---:|
| Monoprotic | 19,537 | 14,652 | 75.00% | 25.00% |
| Amphoteric | 1,914 | 1,485 | 77.58% | 22.42% |
| Polyprotic / complex | 5,767 | 3,079 | 53.39% | 46.61% |

### Interpretation

For monoprotic drug-like molecules, which represent many practical lead-optimization cases, pKaNET Cloud+ v68 assigns the same dominant net charge as the pKaHub-derived reference annotation for approximately three out of four compounds.

Agreement is lower for polyprotic and zwitterionic molecules. This is expected because multiple ionization equilibria, tautomeric alternatives, and macroscopic charge-state transitions can compete near physiological pH.

The remaining disagreements consist of 8,002 cases, corresponding to 29.40% of the benchmark subset. A subset of failures includes borderline molecules where at least one predicted ionizable site has a heuristic pKa within ±1 unit of pH 7.4.

---

## 🗂️ Benchmark Files

The pKaHub-derived subset was curated to retain molecules with interpretable macroscopic charge-state annotations relevant to ligand docking at pH 7.4. The filtering prioritized compounds with clear reference charge transitions, docking-relevant ionizable groups, and chemically interpretable SMILES.

The complete raw pKaHub database is **not redistributed** in this repository. Only the curated validation table, pKaNET predictions, agreement labels, and failed-case review file are provided for reproducibility.

| File | Description |
|---|---|
| `pKaNET_pKahub_docking_relevant_subset_validation_v68.csv` | Curated validation subset with pKaHub-derived reference charge labels and pKaNET predictions |
| `pKaNET_pKahub_docking_relevant_failed_cases_v68.csv` | Disagreement cases from pKaNET Cloud+ v68 for manual review and future rule refinement |
| `curated_regression_set.csv` | Internal 65-compound chemically curated regression set |
| `validation_summary_template.csv` | Template for recording new validation outputs |
| `failed_cases_review_template.csv` | Template for manually classifying disagreement cases |

Recommended key columns for validation output:

```text
molid, smiles, pkahub_charge, pkanet_v67_charge, pkanet_v68_charge,
agreement_v67, agreement_v68, transition_class,
pkanet_v68_site_count, pkanet_v68_site_decisions
```

The `pkanet_v68_site_decisions` field may contain a semicolon-delimited list of site-level annotations, for example:

```text
site_label:heuristic_pKa:site_type
```

---

## ⚠️ Important Notes

- pKaNET Cloud+ uses a heuristic pKa table and microstate-ranking workflow, not a quantitative experimental pKa predictor.
- Dimorphite-DL is used for initial ionization-state enumeration, while pKaNET Cloud+ performs tautomer-aware filtering, heuristic pKa scoring, and final microstate re-ranking.
- For borderline cases where one or more predicted site pKa values fall within ±1 unit of the target pH, the predicted charge should be treated as uncertain.
- Net-charge agreement does not guarantee that the exact ionized atom or tautomer is correct, especially for polyprotic or zwitterionic molecules.
- The pKaHub-derived benchmark subset is a curated validation subset, not a redistribution of the complete raw pKaHub database.
- Ligands containing aliphatic basic amines are often predicted as monocationic at pH 7.4. However, inhibitors from the same pharmacological class can have different dominant charge states. For example, in the current EGFR/drug regression panel, Gefitinib and Imatinib are assigned as +1, whereas Erlotinib and Osimertinib are assigned as neutral. Therefore, EGFR inhibitors should be evaluated compound by compound rather than assigned a uniform charge class.

---

## 🧬 Supported Inputs

| Format | Extension / input type |
|---|---|
| SMILES | `.smi` or plain-text SMILES |
| MDL Molfile | `.mol` |
| Structure-data file | `.sdf` |
| Tripos Mol2 | `.mol2` |
| Protein Data Bank ligand file | `.pdb` |

---

## 📤 Outputs

| Output | Description |
|---|---|
| pH-adjusted SMILES | Dominant predicted microspecies at the target pH |
| Net formal charge | Integer formal charge of the selected microspecies |
| `minimized_ligand.pdb` | 3D ligand structure after geometry minimization |
| `minimized_ligand.sdf` | 3D ligand structure with explicit hydrogens and formal charge information |
| Preparation log | Site-level protonation decisions, pKa evidence, and ranking information |

---

## 🎯 Intended Use Cases

pKaNET Cloud+ is intended for:

- Ligand preparation before molecular docking.
- AutoDock Vina, VinaXB, GNINA, Glide, GOLD, rDock, and similar docking workflows.
- GAFF2, CGenFF, or other force-field parameterization workflows where a reasonable ligand protonation state is required before charge assignment.
- QSAR, ADMET, and virtual-screening dataset curation.
- Teaching pKa, protonation state, microspecies, and docking-preparation concepts.

---

## 🔗 Integration with Anyone Can Dock

pKaNET Cloud+ v68 is the default protonation engine in the Anyone Can Dock web application.

It replaces the previous direct Dimorphite-DL-only pipeline with a Dimorphite-DL-assisted enumeration plus pKaNET re-ranking workflow.

In the Anyone Can Dock codebase, pKaNET Cloud+ is called through:

```python
protonate_pkanet()
```

Typical integration pattern:

```text
input ligand → standardization → Dimorphite-DL-assisted enumeration →
pKaNET Cloud+ ranking → dominant microspecies → 3D generation →
minimization → docking-ready output
```

To use pKaNET Cloud+ as a standalone module, place `pKaNET.py` in the same directory as `core.py` or import it directly in a Python workflow.

---

## 🧪 Minimal Command-Line Testing Workflow

Run the internal regression suite:

```bash
python3 test_pkanet.py pKaNET.py
```

Run only the flavonoid regression group:

```bash
python3 test_pkanet.py pKaNET.py G8
```

Run only the drug regression panel:

```bash
python3 test_pkanet.py pKaNET.py G12
```

A successful run should end with:

```text
✅ ALL PASS
```

---

## ✅ Recommended Wording for Manuscript or ESI

The following wording is recommended when describing this benchmark:

> pKaNET Cloud+ was evaluated using an internal chemically curated regression set and an external pKaHub-derived docking-relevant validation subset. The benchmark endpoint was dominant net-charge agreement at pH 7.4, not numerical pKa prediction accuracy. The pKaHub-derived subset was curated to retain molecules with interpretable macroscopic charge-state annotations relevant to ligand docking. The complete raw pKaHub database was not redistributed; only curated validation outputs and disagreement summaries were provided for reproducibility.

---

## 🚫 Wording to Avoid

| Avoid | Use instead |
|---|---|
| “pKaNET pKa accuracy is 70.60%” | “pKaNET net-charge agreement at pH 7.4 is 70.60%” |
| “pKaNET predicts pKa correctly” | “pKaNET assigns the correct dominant net charge” |
| “Fully validated against experimental data” | “Benchmarked against pKaHub-derived charge-state annotations” |
| “All imidazole cases are fixed” | “The reported imidazole N-H deprotonation bug is resolved; residual failures may remain for complex imidazole-containing molecules” |
| “All EGFR inhibitors are +1” | “EGFR inhibitors should be evaluated compound by compound” |

---

## 🙏 Acknowledgements

pKaNET Cloud+ builds on the following open scientific software and data resources:

- **RDKit** — molecule standardization, SMARTS-based substructure matching, tautomer handling, formal charge assignment, ETKDG conformer generation, and MMFF/UFF geometry optimization.
- **Dimorphite-DL** — initial ionization-state enumeration. pKaNET Cloud+ uses Dimorphite-DL-assisted enumeration and then performs independent scoring and re-ranking.
- **pKaSolver** — optional machine-learning pKa backend when available.
- **PROPKA** — optional semi-empirical pKa backend or fallback.
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

pKaNET Cloud+ is developed as part of the ligand-preparation workflow for Anyone Can Dock and related computational drug-discovery tools.

The method is intended to improve docking-readiness by reducing common protonation-state errors caused by direct rule-based ionization workflows, especially for imidazole-like motifs, flavonoids, phosphates/phosphonates, sulfonamide-like acids, zwitterions, and drug-l
