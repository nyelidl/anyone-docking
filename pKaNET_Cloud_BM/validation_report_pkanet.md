# pKaNET Cloud+ — Validation Report
**Version:** core.py (calibrated heuristic + fast predict API, with `amine_gamma_ring_sulfonyl` patch)  
**Date:** 2026-07-14  
**Test harness:** `test_pkanet.py` — 67 cases across 13 functional groups  
**Environment:** heuristic path + Dimorphite-DL (no ML pKa backend, no PubChem network)

---

## Result Summary

| Metric | Value |
|--------|-------|
| **Total tests** | 67 |
| **PASS** | 67 |
| **FAIL** | 0 |
| **Pass rate** | **100 %** |

---

## Changes Since Previous Report (2026-05-20, 65 tests)

| Change | Detail |
|--------|--------|
| **New rule** | `amine_gamma_ring_sulfonyl` (pKa 6.5, base) — catches secondary/primary amines γ to a ring sulfone via three ring-constrained CX4 carbons. Fixes dorzolamide false positive (+1 → 0). |
| **SMARTS** | `[NX3;H1,H2;!$(NC=O);!$([nH])][CX4;R][CX4;R][CX4;R][SX4;R](=O)(=O)` |
| **New tests** | T66 (dorzolamide, expected 0) and T67 (NAH bis-acylhydrazone, negative control, expected 0) |
| **New group** | G13: EWG-suppressed amine (ring sulfonyl) |
| **Regression** | All 65 prior tests unchanged ✅ |

---

## Results by Group

| Group | Description | Pass | Fail | Status |
|-------|-------------|------|------|--------|
| G1  | Imidazole-type N-H (imidazole, benzimidazole, pyrazole, purine, adenine, drugs) | 10 | 0 | ✅ |
| G2  | Phosphonate / Phosphate (diprotic acids, bisphosphonate, tenofovir, glyphosate) | 7  | 0 | ✅ |
| G3  | Thiol — ArSH / AlkSH (thiophenol, 6-mercaptopurine, captopril, ethanethiol) | 5  | 0 | ✅ |
| G4  | Carboxylic acid (acetic, ibuprofen, aspirin, diclofenac, trichloroacetic) | 5  | 0 | ✅ |
| G5  | Phenol variants (phenol, p-nitrophenol, pentafluorophenol, acetaminophen, catechol, warfarin) | 6  | 0 | ✅ |
| G6  | Amine bases (aniline, pyridine, methylamine, metformin, amlodipine) | 5  | 0 | ✅ |
| G7  | Sulfonamide / Saccharin (methanesulfonamide, saccharin, chlorothiazide, furosemide) | 4  | 0 | ✅ |
| G8  | Flavonoid regression — **MUST NOT change** (baicalein, apigenin, luteolin, kaempferol) | 4  | 0 | ✅ |
| G9  | Zwitterion / Multi-site (glycine, histidine, glutamic acid, lysine, cysteine) | 5  | 0 | ✅ |
| G10 | PubChem pKa guard (benzimidazole+base pKa mock, imidazole+base pKa mock, phenol+correct pKa) | 3  | 0 | ✅ |
| G11 | Truly neutral (caffeine, cholesterol, glucose, benzene) | 4  | 0 | ✅ |
| G12 | Drug regression panel (erlotinib, gefitinib, imatinib, osimertinib, atorvastatin, methotrexate, ciprofloxacin) | 7  | 0 | ✅ |
| G13 | EWG-suppressed amine — ring sulfonyl (dorzolamide, NAH negative control) | 2  | 0 | ✅ |

---

## Per-Test Detail

### G1 — Imidazole-type N-H

| ID | Compound | Expected | Got | Result |
|----|----------|----------|-----|--------|
| T01 | Imidazole | 0 | 0 | ✅ PASS |
| T02 | Benzimidazole | 0 | 0 | ✅ PASS |
| T03 | Pyrazole | 0 | 0 | ✅ PASS |
| T04 | Indazole | 0 | 0 | ✅ PASS |
| T05 | Purine | 0 | 0 | ✅ PASS |
| T06 | Adenine | 0 | 0 | ✅ PASS |
| T07 | 1-Methylbenzimidazole | 0 | 0 | ✅ PASS |
| T08 | Clotrimazole | 0 | 0 | ✅ PASS |
| T09 | Omeprazole | 0 | 0 | ✅ PASS |
| T10 | Metronidazole | 0 | 0 | ✅ PASS |

### G2 — Phosphonate / Phosphate

| ID | Compound | Expected | Got | Result |
|----|----------|----------|-----|--------|
| T11 | Methylphosphonic acid | −2 | −2 | ✅ PASS |
| T12 | Phenylphosphonic acid | −2 | −2 | ✅ PASS |
| T13 | Fosfomycin | −2 | −2 | ✅ PASS |
| T14 | Alendronate | −2 | −2 | ✅ PASS |
| T15 | Tenofovir | −2 | −2 | ✅ PASS |
| T16 | Phosphate monoester | −2 | −2 | ✅ PASS |
| T17 | Glyphosate | −2 | −2 | ✅ PASS |

### G3 — Thiol

| ID | Compound | Expected | Got | Result |
|----|----------|----------|-----|--------|
| T18 | Thiophenol | −1 | −1 | ✅ PASS |
| T19 | 4-Chlorothiophenol | −1 | −1 | ✅ PASS |
| T20 | 6-Mercaptopurine | −1 | −1 | ✅ PASS |
| T21 | Captopril | −1 | −1 | ✅ PASS |
| T22 | Ethanethiol | 0 | 0 | ✅ PASS |

### G4 — Carboxylic Acid

| ID | Compound | Expected | Got | Result |
|----|----------|----------|-----|--------|
| T23 | Acetic acid | −1 | −1 | ✅ PASS |
| T24 | Ibuprofen | −1 | −1 | ✅ PASS |
| T25 | Aspirin | −1 | −1 | ✅ PASS |
| T26 | Diclofenac | −1 | −1 | ✅ PASS |
| T27 | Trichloroacetic acid | −1 | −1 | ✅ PASS |

### G5 — Phenol Variants

| ID | Compound | Expected | Got | Result |
|----|----------|----------|-----|--------|
| T28 | Phenol | 0 | 0 | ✅ PASS |
| T29 | 4-Nitrophenol | −1 | −1 | ✅ PASS |
| T30 | Pentafluorophenol | −1 | −1 | ✅ PASS |
| T31 | Acetaminophen | 0 | 0 | ✅ PASS |
| T32 | Catechol | 0 | 0 | ✅ PASS |
| T33 | Warfarin | −1 | −1 | ✅ PASS |

### G6 — Amine Bases

| ID | Compound | Expected | Got | Result |
|----|----------|----------|-----|--------|
| T34 | Aniline | 0 | 0 | ✅ PASS |
| T35 | Pyridine | 0 | 0 | ✅ PASS |
| T36 | Methylamine | +1 | +1 | ✅ PASS |
| T37 | Metformin | +1 | +1 | ✅ PASS |
| T38 | Amlodipine | +1 | +1 | ✅ PASS |

### G7 — Sulfonamide / Saccharin

| ID | Compound | Expected | Got | Result |
|----|----------|----------|-----|--------|
| T39 | Methanesulfonamide | 0 | 0 | ✅ PASS |
| T40 | Saccharin | −1 | −1 | ✅ PASS |
| T41 | Chlorothiazide | −1 | −1 | ✅ PASS |
| T42 | Furosemide | −2 | −2 | ✅ PASS |

### G8 — Flavonoid Regression (MUST NOT change)

| ID | Compound | Expected | Got | Fragment guard | Result |
|----|----------|----------|-----|----------------|--------|
| T43 | Baicalein | 0 | 0 | no [O−] ✅ | ✅ PASS |
| T44 | Apigenin | 0 | 0 | no [O−] ✅ | ✅ PASS |
| T45 | Luteolin | 0 | 0 | no [O−] ✅ | ✅ PASS |
| T46 | Kaempferol | 0 | 0 | no [O−] ✅ | ✅ PASS |

### G9 — Zwitterion / Multi-site

| ID | Compound | Expected | Got | Result |
|----|----------|----------|-----|--------|
| T47 | Glycine | 0 | 0 | ✅ PASS |
| T48 | Histidine | 0 | 0 | ✅ PASS |
| T49 | Glutamic acid | −1 | −1 | ✅ PASS |
| T50 | Lysine | +1 | +1 | ✅ PASS |
| T51 | Cysteine | 0 | 0 | ✅ PASS |

### G10 — PubChem pKa Guard

| ID | Compound | Expected | Got | Fragment guard | Result |
|----|----------|----------|-----|----------------|--------|
| T52 | Benzimidazole + PubChem base pKa mock | 0 | 0 | no [n−] ✅ | ✅ PASS |
| T53 | Imidazole + PubChem base pKa mock | 0 | 0 | no [n−] ✅ | ✅ PASS |
| T54 | Phenol + PubChem correct pKa | 0 | 0 | — | ✅ PASS |

### G11 — Truly Neutral

| ID | Compound | Expected | Got | Result |
|----|----------|----------|-----|--------|
| T55 | Caffeine | 0 | 0 | ✅ PASS |
| T56 | Cholesterol | 0 | 0 | ✅ PASS |
| T57 | Glucose | 0 | 0 | ✅ PASS |
| T58 | Benzene | 0 | 0 | ✅ PASS |

### G12 — Drug Regression Panel

| ID | Compound | Expected | Got | Result |
|----|----------|----------|-----|--------|
| T59 | Erlotinib | 0 | 0 | ✅ PASS |
| T60 | Gefitinib | +1 | +1 | ✅ PASS |
| T61 | Imatinib | +1 | +1 | ✅ PASS |
| T62 | Osimertinib | 0 | 0 | ✅ PASS |
| T63 | Atorvastatin | −2 | −2 | ✅ PASS |
| T64 | Methotrexate | −2 | −2 | ✅ PASS |
| T65 | Ciprofloxacin | 0 | 0 | ✅ PASS |

### G13 — EWG-Suppressed Amine (ring sulfonyl) — NEW

| ID | Compound | Expected | Got | Notes | Result |
|----|----------|----------|-----|-------|--------|
| T66 | Dorzolamide | 0 | 0 | `amine_gamma_ring_sulfonyl` pKa 6.5 fires; previously `aliphatic_amine` pKa 9.5 gave +1 | ✅ PASS |
| T67 | NAH (bis-acylhydrazone) | 0 | 0 | Negative control — no sulfur, no amine rules fire; unaffected by new rule | ✅ PASS |

---

## Benchmark Accuracy (27,218 Drug-Like Molecules, pH 7.4) from pKahub is hosted at: pkahub.ttk.hu.

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Previous heuristic (binary H-H, no caps) | 18,128 | 27,183 | 66.70 % |
| `heuristic_net_charge` (this release, with caps) | **18,850** | 27,183 | **69.34 %** |
| `predict_charge(mode='auto')` with full pipeline for borderline | ≥ 18,850 | 27,183 | ≥ 69.34 % |

Largest per-charge improvements:

| Expected charge | Previous | This release | Δ |
|---|---|---|---|
| +1 | 67.8 % | **77.1 %** | +9.3 pp |
| −1 | 54.8 % | **57.7 %** | +2.9 pp |
| 0  | 76.0 % | **76.1 %** | stable |

---

## Notes

- All 67-case tests run with the full tautomer + Dimorphite-DL + Henderson–Hasselbalch scoring pipeline. Dimorphite-DL is installed and active.
- In production with PubChem lookup enabled, additional experimental pKa anchors further improve accuracy for well-characterised compounds.
- Expected charges reflect the dominant protonation state at **pH 7.4** using published pKa values.
- The flavonoid group (G8) is a hard regression constraint — any change that produces [O−] on a flavonoid at pH 7.4 is treated as a failure regardless of overall charge.
- `heuristic_net_charge` returns charge 0 for keto-form warfarin (no OH detectable on the parent); `predict_charge(mode='auto')` correctly escalates to the full pipeline and returns −1.
- The `amine_gamma_ring_sulfonyl` rule uses ring-constrained SMARTS (`[CX4;R]` and `[SX4;R]`) to avoid false positives on open-chain amines. Brinzolamide (exp amine pKa 5.9) has an unsaturated carbon in the γ-path and requires a separate `CX3` variant not yet implemented.

---

*Generated against `core.py` (calibrated heuristic + fast predict API, `amine_gamma_ring_sulfonyl` patch) — 2026-07-14*
