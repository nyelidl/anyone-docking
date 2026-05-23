# <img src="https://raw.githubusercontent.com/nyelidl/anyone-docking/main/any-L.svg" width="32"> Anyone can dock, everyone can do!

**Anyone docking: Browser-based molecular docking — no installation required.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nyelidl.github.io/anyone-docking/)

> Paste a SMILES, draw a structure, or upload a file. Pick a PDB or CIF. Dock in seconds.


> Recent update: receptor setup now auto-scans ligand-like HETATM records and shows a reference-ligand dropdown only when needed. Ligand preparation now supports pKaNET-ranked microstates with manual rank selection for ambiguous protonation/tautomer cases.

---

Batch docking with 4 docking engines: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1e23e0145ja6zJi0HibA6_JBO7p78Xvyw?usp=sharing)

---

## ✨ What it does

| | |
|---|---|
| 🔬 | **Single & batch docking** via AutoDock Vina 1.2.7 |
| 🏗️ | **Guided receptor prep** — download any PDB/CIF, review ligand candidates automatically, choose the co-crystal reference ligand when needed, strip solvent, add hydrogens |
| 📄 | **PDB & mmCIF support** — upload `.pdb` or `.cif` files, or download either format from RCSB (auto-fallback to CIF for large/newer entries) |
| 🎯 | **Smart grid detection with user control** — auto-detects ligand-like HETATM records, auto-selects clear single-ligand cases, shows a dropdown only when multiple ligand candidates exist, and uses the selected reference ligand centroid for the docking box |
| ✏️ | **3-way ligand input** — SMILES text, file upload (`.pdb` / `.sdf` / `.mol2`), or **draw in Ketcher** |
| 🧬 | **pKaNET-ranked ligand microstates** — ranks protonation/tautomer states at the target pH, recommends a docking-ready default, and allows manual rank selection from a dropdown for ambiguous cases |
| ⚖️ | **Conservative docking-state selection** — for heuristic-only polyphenol, coumarin, flavonoid, and related systems, ACD avoids blindly choosing the highest-scoring over-deprotonated state |
| 🧬 | **Heme-aware preparation** — HEM/HEC/HEA/HEB stripped before OpenBabel (avoids Fe-porphyrin failures), re-injected into PDBQT with correct AD4 atom types, and shown as orange sticks in all 3D viewers |
| ⚗️ | **Water, cofactor & metal control** — remove waters, keep selected metal ions, and keep/strip cofactors such as HEM, FAD, NAD, ATP, CoA, or SAM through receptor-preparation options |
| 🔗 | **Modified amino acid safety** — HETATM residues with full protein backbone (CYP, MSE, TPO…) are correctly kept in the receptor, not mistakenly removed as ligands |
| ♻️ | **Redocking validation** in both single & batch mode — dock the co-crystal ligand as a reference with RMSD vs crystal, score comparison, and reference line in plots |
| 🧪 | **Bond-order correction** — fixes PDBQT aromaticity artifacts before visualization |
| 🗺️ | **Two local 2D diagram engines** + PoseView download tab |
| 🖱️ | **Interactive drag mode** — freely reposition residue labels in real time, export PNG (up to 600 dpi) or SVG |
| 🔭 | **Binding pocket viewer** — interacting residues (orange sticks) around the docked ligand, with toggleable labels and adjustable distance cutoff |
| 🤖 | **AI-ready prompt** — auto-filled context for GPT-4o, Claude, Gemini, or DeepSeek; adapts based on whether redocking was performed |
| 📊 | **3D viewers** — animated multi-pose sweep, interactive pose selector, and dedicated binding pocket view |
| 📁 | **One-click ZIP** — all poses, bond-order-corrected SDFs, 2D diagrams, score plot, receptor-preparation log, and pKaNET decision log when available |

---

## 🗺️ 2D Interaction Diagrams

Three tabs — each with a different rendering engine:

### 🧬 Anyone Can Dock 2D Diagram *(default)*

A custom PoseView-style SVG diagram rendered entirely locally (no server required).

| Feature | Detail |
|---|---|
| **8 interaction types** | H-bond (distance on line), hydrophobic, π-π stacking, cation-π, ionic, metal coordination (heme Fe, ZN, MG…), halogen bond, H-bond to halogen |
| **Heme/metal labels** | HEM, ZN, MG, FE etc. shown without residue number for clarity |
| **Geometry-based detection** | All interactions computed from 3D coordinates — no server, works on Streamlit Cloud |
| **ACS ChemDraw bond style** | Bond widths, double-bond spacing, and wedge geometry follow ACS publication standards |
| **Smart layout** | Residue circles placed radially by natural interaction angle; simultaneous-delta push-apart prevents overlap |
| **Interactive drag mode** | Click 🖱 Interactive to reposition any residue circle — lines and distance labels update in real time |
| **Export** | ↓ SVG (vector) · ↓ PNG at Screen (1×) / 150 dpi (2×) / 300 dpi (3×) / 600 dpi (4×) |

### 🔬 RDKit 2D Diagram

Classic RDKit `MolDraw2DSVG` highlight-circle style.

| Feature | Detail |
|---|---|
| **Interaction types** | H-bond / polar (blue) · Hydrophobic (green) · Other including metal/heme (pink) |
| **Layout** | RDKit's own force-field layout — residue pseudo-atoms added as `BondType.ZERO` bonds |
| **Side-by-side** | Docked pose (left) + co-crystal reference (right) |
| **Heme/metal labels** | HEM, ZN, MG etc. shown without residue number |
| **Export** | PNG + SVG download under each diagram |
| **AI prompt** | Auto-filled prompt adapts to single diagram vs. comparison |

### 🔬 PoseView (proteins.plus)

Submits receptor + docked pose to the [proteins.plus](https://proteins.plus/help/poseview) PoseView REST API.

| Feature | Detail |
|---|---|
| **PoseView v1** | Docked pose diagram — submitted directly, no round-trip via MoleculeHandler |
| **PoseView2** | Co-crystal reference diagram fetched by PDB code + ligand ID (when available) |
| **Manual fallback** | Download `receptor.pdb` + `docked_pose.sdf` for manual upload at proteins.plus |
| **API test** | Built-in diagnostic button tests server availability with PDB 4AGN |
| **Retries** | Automatic 3× retry with 10 s delay on server-side failures |

---

## 🧬 Supported protein types

| Protein class | Support | Notes |
|---|---|---|
| Standard single-chain proteins | ✅ Full | Primary use case |
| Multi-chain / homo-oligomers | ✅ Full | Duplicate chains can be deduplicated; if the same ligand appears in equivalent chains, the chain A ligand is auto-selected without unnecessary dropdowns |
| **Heme proteins** (CYP450, peroxidases, Hb, Mb) | ✅ Full | Fe-porphyrin handled separately; grid auto-centers on Fe; shown in all viewers |
| Metal-binding proteins (zinc fingers, carbonic anhydrase) | ✅ Full | ZN, MG, CA, MN, FE, CU re-injected with correct charges |
| MD simulation outputs (GROMACS, AMBER) | ✅ Full | Blank chain IDs auto-assigned to chain A |
| Non-standard ligand names (MOL, LIG, UNL, INH) | ✅ Full | `hetatm` keyword bypasses ProDy misclassification |
| Modified amino acids (CYP, MSE, TPO, SEP) | ✅ Full | Backbone atom check keeps them in receptor |
| Multiple co-crystal ligands | ✅ Full | Ligand-like HETATM records are auto-scanned; a dropdown appears only when multiple reference-ligand candidates exist; the selected ligand defines the grid center and is removed from the receptor |
| Cofactor-binding proteins (FAD, NAD, ATP, CoA) | ✅ Full | Cofactors are classified separately from ligand candidates and can be kept or stripped independently |
| Glycoproteins | ⚠️ Partial | Glycans kept in receptor (correct), but not shown in 2D interaction diagram |
| Antibodies / very large proteins | ⚠️ Partial | Works; 3D viewer and interaction detection may be slow |
| Membrane proteins | ⚠️ Partial | Dockable without lipids; lipids not auto-filtered |
| RNA / DNA targets | ⚠️ Partial | H-bond/hydrophobic detection runs; no nucleic-acid-specific interactions in 2D diagram |
| **Covalent docking** | ❌ No | Vina is non-covalent only |
| Dimer interface binding sites | ❌ No | Chain deduplication removes the second chain |

---

## 🤖 AI Prompt Section

Located below the 2D diagram. The prompt auto-adapts to the session state:

| Scenario | Prompt content |
|---|---|
| **Docked ligand only** | Plain-language explanation of interactions + ready-to-use summary paragraph |
| **With co-crystal reference (no redocking)** | Comparison prompt · Reference: `ligand` co-crystallised in PDB `ID` (see 2D diagram) |
| **With co-crystal reference (redocking performed)** | Comparison prompt · Reference binding energy from re-docking included |
| **Heme targets** | Gold dashed line = metal/heme coordination included in legend description |

Copy the prompt + a screenshot of your diagram into Claude, GPT-4o, or Gemini. The prompt ends with a **"Ready-to-use summary:"** section — a 3–4 sentence paragraph ready to paste into a report or slide.

---

## 💻 Platform compatibility

| Platform | Vina binary | OpenBabel | Status |
|---|---|---|---|
| **Linux x86_64** | ✅ Auto-download | `apt install openbabel` | Fully supported (primary) |
| **macOS Intel** | ✅ Auto-download | `brew install open-babel` | Fully supported |
| **macOS Apple Silicon** (M1–M4) | ✅ Native `aarch64` binary | `brew install open-babel` | Fully supported |
| **Windows x86_64** | ✅ Auto-download | [Installer](https://openbabel.org/wiki/Category:Installation) | Supported (WSL2 recommended) |
| **Streamlit Cloud** | ✅ Auto-download | via `packages.txt` | Fully supported |
| **Google Colab** | ✅ Auto-download | `!apt install openbabel` | Fully supported in both Google Colab and web-based interfaces|

> **Easiest option:** Use the [hosted Streamlit app](https://nyelidl.github.io/anyone-docking/) — no installation needed.

---

## 🏗️ Receptor input formats

| Format | Source | Notes |
|---|---|---|
| **PDB** | Upload `.pdb` or download from RCSB | Standard format, works for most entries |
| **mmCIF** | Upload `.cif` / `.mmcif` or download from RCSB | Recommended for large structures or newer entries that lack `.pdb` files |
| **MD outputs** | Upload `.pdb` from GROMACS / AMBER | Blank chain IDs auto-assigned to chain A |

CIF files are automatically converted to PDB using a multi-strategy cascade: **gemmi** → **OpenBabel** → **ProDy**. If PDB download from RCSB fails, the app automatically falls back to CIF format.

---


## 🧫 Receptor setup and ligand-reference selection

The receptor setup panel is designed to prevent blind automatic selection of the wrong HETATM record.

| Step | Behavior |
|---|---|
| **Structure loading** | For RCSB mode, entering a valid 4-character PDB ID automatically downloads/scans the structure. Upload mode scans immediately after a structure file is provided. No extra scan button is required. |
| **Ligand candidates** | Only ligand-like small molecules are shown as candidates for defining the binding site. Waters, buffer molecules, ions, metals, glycans, and cofactors are filtered or handled separately. |
| **Single ligand case** | If only one ligand-like candidate is present, it is selected automatically. |
| **Homo-multimer case** | If the same ligand appears in equivalent chains, the chain A ligand is selected automatically to avoid redundant choices. |
| **Multiple ligand case** | If multiple ligand-like candidates are present in the relevant chain or assembly, a dropdown appears and the user selects the reference ligand. |
| **Grid definition** | The selected reference ligand is removed from the docking receptor but its centroid is used to define the AutoDock Vina search box. |
| **Waters / metals / cofactors** | Waters, metal ions, and cofactors are controlled separately through receptor-preparation options. |

Example ligand-selection behavior:

| Residue name | Chain | Residue ID | Atom count | Type guess | UI behavior |
|---|---:|---:|---:|---|---|
| ERL | A | 901 | 45 | ligand | shown as binding-site reference candidate |
| GOL | A | 902 | 6 | buffer/additive | not shown in ligand dropdown; removed by default |
| ZN | A | 903 | 1 | metal | controlled by metal-ion option |
| FAD | A | 904 | 53 | cofactor | controlled by cofactor option |


## 🖥️ Ligand input modes

| Mode | Description |
|---|---|
| **SMILES string** | Type or paste any valid SMILES — standardized and prepared at the target pH |
| **Upload file** | `.pdb`, `.sdf`, `.mol2` — use as-is or re-protonate at the target pH |
| **Draw in Ketcher** | Full 2D chemical sketcher in the browser → SMILES exported automatically |
| **pKaNET microstate control** | Ranked protonation/tautomer states are available through a dropdown. Users can use the recommended state, the highest-scoring state, or manually select another rank before docking. |

---


## 🧬 pKaNET microstate selection

For ligand preparation, ACD reports the selected protonation/tautomer state instead of silently forcing a single hidden form.

| Mode | Meaning |
|---|---|
| **Auto recommended** | Uses the pKaNET-recommended docking state. For chemically ambiguous heuristic-only systems such as polyphenols, coumarins, flavonoids, catechols, and related scaffolds, this may select a conservative state rather than the highest score. |
| **Highest-scoring microstate** | Uses the top-ranked pKaNET state directly. |
| **Manual rank** | Lets the user choose a ranked microstate from a dropdown before preparing the ligand for docking. |

The ligand panel displays the selected rank, formal charge, score, recommendation status, and selected microstate SMILES. Detailed reasoning, charged atoms, and ambiguity notes are stored in the **Preparation log** to keep the main interface clean.

This design is intended for docking-relevant ligand preparation, not absolute microscopic pKa prediction. Ambiguous cases should be inspected manually when the predicted charge state may strongly affect docking.


## ⚗️ Water, cofactor & metal options

Accessible in receptor preparation:

| Option | Default | Effect |
|---|---|---|
| **Remove waters** | ✅ On | Removes crystallographic waters unless the user chooses to keep them |
| **Keep metal ions in receptor** | ✅ On | Keeps common ions/metals such as ZN, MG, CA, MN, FE, CU, CO, NI, CD, HG, NA, and K when appropriate |
| **Keep cofactors in receptor** | ✅ On | Keeps cofactors such as ATP, ADP, FAD, FMN, NAD, CoA, and SAM when appropriate |

Buffer/additive molecules such as GOL, EDO, PEG, SO4, and PO4 are not treated as binding-site reference ligands. They are removed by default unless intentionally handled as part of the receptor setup.

Heme (HEM/HEC/HEA/HEB/HDD/HDM) is handled separately — stripped before OpenBabel when needed and re-injected with AD4 atom types for docking/visualization.

---

## ♻️ Redocking validation

Available in **both single and batch** docking modes:

| Feature | Description |
|---|---|
| **Co-crystal reference docking** | Dock the known co-crystal ligand alongside your candidate(s) |
| **RMSD vs crystal** | Heavy-atom RMSD calculated against the original crystal pose via MCS matching |
| **Reference score line** | Dashed red line on the affinity plot for quick visual comparison |
| **Pose confirmation** | Browse reference poses, confirm which one to use as the baseline |
| **Download** | Export reference poses as SDF/PDBQT |

---

## 🔭 3D visualization layers

| Viewer | What you see |
|---|---|
| **Receptor prep** | Protein cartoon · selected reference ligand (magenta) · heme/metal context when present · docking grid box (cyan wireframe) · XYZ axis arrows |
| **Animated pose viewer** | All poses swept as frames · protein surface · co-crystal overlay · heme (orange) |
| **Interactive pose selector** | Single selected pose · protein cartoon + surface · co-crystal overlay · heme (orange) |
| **Binding pocket view** | Faint full-protein cartoon · docked pose (cyan) · heme (orange) · interacting residues (orange sticks) · optional residue labels |
| **Redocking browser** | Reference ligand poses · crystal overlay · heme (orange) · RMSD per pose |

---

---

## 🖥️ Run locally

### Google Colab (web-based interface)
Run locally on Google colab with web-based interface: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WtWYUUB1AREZMeB5qEJ9OD84AvWk1z4z?usp=sharing)


### Linux (Ubuntu / Debian)
```bash
sudo apt install python3.11 python3.11-venv openbabel libcairo2-dev libpangocairo-1.0-0 && \
git clone https://github.com/nyelidl/anyone-docking-local.git && \
cd anyone-docking-local && \
python3.11 -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt && \
streamlit run app.py
```

### macOS
```bash
brew install python@3.11 open-babel cairo pango && \
git clone https://github.com/nyelidl/anyone-docking-local.git && \
cd anyone-docking-local && \
python3.11 -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt && \
streamlit run app.py
```

> **Apple Silicon (M1/M2/M3/M4):** Fully supported — the app auto-downloads the correct `aarch64` Vina binary.

### Windows

> **Recommended:** Use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/) with Ubuntu and follow the Linux instructions above — it's the simplest and most reliable path.

For native Windows, install dependencies manually first:
1. **OpenBabel** — download the installer from [openbabel.org](https://openbabel.org/wiki/Category:Installation) and add it to PATH
2. **Cairo & Pango** — easiest via conda: `conda install -c conda-forge cairo pango`

Then:
```bash
git clone https://github.com/nyelidl/anyone-docking-local.git && \
cd anyone-docking-local && \
python -m venv venv && \
venv\Scripts\activate && \
pip install -r requirements.txt && \
streamlit run app.py
```

### All platforms

- Python 3.10+ required
- AutoDock Vina 1.2.7 binary is **downloaded automatically** on first launch (Linux, macOS Intel/ARM, Windows)
- The app auto-detects your OS and CPU architecture

### Optional: CIF support

For best mmCIF → PDB conversion quality, install [gemmi](https://gemmi.readthedocs.io/):

```bash
pip install gemmi
```

Without gemmi, the app falls back to OpenBabel and ProDy (both already in the dependency stack).

### Streamlit Cloud deployment

Place these files in your repo root:

```
anyone-docking/
├── app.py
├── core.py
├── requirements.txt   # Python packages
└── packages.txt       # System apt packages (openbabel, libcairo2-dev, …)
```

---

## 📄 Citation

If you use this tool in research, please cite the relevant software and resources used in this workflow:

> **AutoDock Vina 1.2.7**
> Eberhardt et al., *Journal of Chemical Information and Modeling*, 2021
> DOI: https://doi.org/10.1021/acs.jcim.1c00203

> **DFDD**
> Hengphasatporn, K.; Duan, L.; Harada, R.; Shigeta, Y., *Journal of Chemical Information and Modeling*, 2026
> DOI: https://doi.org/10.1021/acs.jcim.5c02852

> **RDKit**
> Landrum, G. (2023). RDKit: Open-source cheminformatics.
> https://www.rdkit.org

> **ProDy**
> Bakan et al., *Bioinformatics*, 2011
> DOI: https://doi.org/10.1093/bioinformatics/btr168

> **stmol**
> Nápoles-Duarte et al., *Frontiers in Molecular Biosciences*, 2022
> DOI: https://doi.org/10.3389/fmolb.2022.990846

> **Dimorphite-DL**
> Ropp et al., *Journal of Cheminformatics*, 2019
> DOI: https://doi.org/10.1186/s13321-019-0336-9

> **pKaNET Cloud**
> Please cite the corresponding pKaNET/ACD software paper or manuscript when available.

> **gemmi** *(optional, for CIF support)*
> Wojdyr, M., *Journal of Open Source Software*, 2022
> DOI: https://doi.org/10.21105/joss.04200

---

## 📜 License

This project is licensed under the MIT License.
See the LICENSE file for details.
