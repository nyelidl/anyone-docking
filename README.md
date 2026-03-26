# <img src="https://raw.githubusercontent.com/nyelidl/anyone-docking/main/any-L.svg" width="32"> nyone can dock, everyone can do!

**Anyone docking: Browser-based molecular docking — no installation required.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nyelidl.github.io/anyone-docking/)

> Paste a SMILES, draw a structure, or upload a file. Pick a PDB or CIF. Dock in seconds.

---

Batch docking with 4 docking engines: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1e23e0145ja6zJi0HibA6_JBO7p78Xvyw?usp=sharing)

---

## ✨ What it does

| | |
|---|---|
| 🔬 | **Single & batch docking** via AutoDock Vina 1.2.7 |
| 🏗️ | **Automatic receptor prep** — download any PDB/CIF, strip solvent, add hydrogens |
| 📄 | **PDB & mmCIF support** — upload `.pdb` or `.cif` files, or download either format from RCSB (auto-fallback to CIF for large/newer entries) |
| 🎯 | **Auto grid detection** from co-crystal ligand centroid with XYZ axis & box overlay |
| ✏️ | **3-way ligand input** — SMILES text, file upload (.pdb), or **draw in Ketcher** |
| ♻️ | **Redocking validation** in both single & batch mode — dock the co-crystal ligand as a reference with RMSD vs crystal, score comparison, and reference line in plots |
| 🧪 | **Bond-order correction** — fixes PDBQT aromaticity artifacts before visualization |
| 🗺️ | **Three 2D diagram engines** in separate tabs — see below for full details |
| 🖱️ | **Interactive drag mode** — freely reposition residue labels in real time, export PNG (up to 600 dpi) or SVG |
| 🔭 | **Binding pocket viewer** — interacting residues (orange sticks) around the docked ligand, with toggleable residue labels and adjustable distance cutoff |
| 🤖 | **AI-ready prompt** — auto-filled context for GPT, Claude, Gemini, or DeepSeek; adapts based on whether redocking was performed |
| 📊 | **3D viewers** — animated multi-pose sweep, interactive pose selector, and dedicated binding pocket view |
| 📁 | **One-click ZIP** — all poses, bond-order-corrected SDFs, 2D diagrams, and score plot |

---

## 🗺️ 2D Interaction Diagrams

Three tabs — each with a different rendering engine:

### 🧬 Anyone Can Dock 2D Diagram *(default)*

A custom PoseView-style SVG diagram rendered entirely locally (no server required).

| Feature | Detail |
|---|---|
| **8 interaction types** | H-bond (distance shown on line), hydrophobic, π-π stacking, cation-π, ionic, metal coordination, halogen bond, H-bond to halogen |
| **PLIP-powered detection** | When PLIP is installed, interactions are validated with full geometry checks — proper D-H···A angles for H-bonds, ring planarity for π-stacking, both angles for halogen bonds. Falls back to built-in distance-based detector automatically. |
| **Default cutoff** | **4.5 Å** — matches PoseView/LigPlot+ convention; captures the full first shell of interacting residues |
| **Radial-collapse layout** | Residue circles collapse inward from outside the ligand along each interaction's natural direction; positions are initialised from PCA projection of the real 3D binding site |
| **ACS ChemDraw bond style** | Bond widths, double-bond spacing, and wedge geometry follow ACS publication standards |
| **Interactive drag mode** | Click 🖱 Interactive to reposition any residue circle — lines and distance labels update in real time |
| **Export** | ↓ SVG (vector) · ↓ PNG at Screen (1×) / 150 dpi (2×) / 300 dpi (3×) / 600 dpi (4×) |

### 🔬 RDKit 2D Diagram

Classic RDKit `MolDraw2DSVG` highlight-circle style.

| Feature | Detail |
|---|---|
| **Interaction types** | H-bond / polar (blue) · Hydrophobic (green) · Other (pink) |
| **Default cutoff** | **4.5 Å** |
| **Layout** | RDKit's own force-field layout — residue pseudo-atoms added as `BondType.ZERO` bonds |
| **Side-by-side** | Docked pose (left) + co-crystal reference (right) |
| **Export** | PNG + SVG download under each diagram |
| **AI prompt** | Auto-filled prompt adapts to single diagram vs. comparison |

### ⬇ PoseView

Download-only tab — no API calls made from this app.

| File | Description |
|---|---|
| `receptor.pdb` | Cleaned receptor (hydrogens added) |
| `docked_pose.sdf` | Selected docked pose |
| `cocrystal.sdf` | Co-crystal ligand (converted from PDB if needed) |

Upload these files manually at [proteins.plus/poseview](https://proteins.plus/help/poseview) to generate a server-side PoseView diagram.

---

## 🔬 PLIP Integration (optional, recommended)

When **PLIP** (Protein-Ligand Interaction Profiler) is installed, the 🧬 Anyone Can Dock diagram uses geometry-validated interaction detection instead of the built-in distance-only detector.

| Interaction | Built-in detector | PLIP detector |
|---|---|---|
| **H-bond** | polar atoms < 3.5 Å | dist(D···A) ≤ 4.1 Å **+ D-H···A angle ≥ 120°** + explicit H required |
| **Hydrophobic** | dist < cutoff | hydrophobic atom typing + dist < cutoff |
| **π-π stacking** | centroid dist < 5.5 Å | centroid dist + **ring planarity angle** + **offset < 2.0 Å** |
| **Halogen bond** | C-X angle ≥ 140° | + **X···A-R angle ≥ 90°** (both angles checked) |

The app **auto-detects** whether PLIP is available and shows a status badge:
- ✅ **PLIP active** — geometry-validated interactions
- ℹ️ **Built-in detector** — with install instructions shown in the UI

**To enable PLIP on Streamlit Cloud**, use these dependency files:

`packages.txt`:
```
openbabel
libopenbabel-dev
libcairo2-dev
libpango1.0-dev
pkg-config
python3-dev
libgirepository1.0-dev
```

`requirements.txt` (add these three lines):
```
lxml>=4.9
openbabel>=3.1.1
plip>=3.0.0
```

**To enable PLIP locally** (Linux):
```bash
sudo apt install libopenbabel-dev
pip install lxml openbabel plip
```

---

## 🤖 AI Prompt Section

Located below the 2D diagram. The prompt auto-adapts to the session state:

| Scenario | Prompt content |
|---|---|
| **Docked ligand only** | Plain-language explanation of interactions + ready-to-use summary paragraph |
| **With co-crystal reference (no redocking)** | Comparison prompt · Reference: `ligand` co-crystallised in PDB `ID` (see 2D diagram) |
| **With co-crystal reference (redocking performed)** | Comparison prompt · Reference binding energy from re-docking included |

Copy the prompt + a screenshot of your diagram into Claude, GPT-4o, or Gemini to get a plain-English explanation of your results. The prompt ends with a "Ready-to-use summary:" section — a 3–4 sentence paragraph ready to paste into a report or slide.

---

## 💻 Platform compatibility

| Platform | Vina binary | OpenBabel | Status |
|---|---|---|---|
| **Linux x86_64** | ✅ Auto-download | `apt install openbabel` | Fully supported (primary) |
| **macOS Intel** | ✅ Auto-download | `brew install open-babel` | Fully supported |
| **macOS Apple Silicon** (M1–M4) | ✅ Native arm64 | `brew install open-babel` | Fully supported |
| **Windows x86_64** | ✅ Auto-download | [Installer](https://openbabel.org/wiki/Category:Installation) | Supported (WSL2 recommended) |
| **Streamlit Cloud** | ✅ Auto-download | via `packages.txt` | Fully supported |
| **Google Colab** | ✅ Auto-download | `!apt install openbabel` | Fully supported |

> **Easiest option:** Use the [hosted Streamlit app](https://nyelidl.github.io/anyone-docking/) — no installation needed.

---

## 🏗️ Receptor input formats

| Format | Source | Notes |
|---|---|---|
| **PDB** | Upload `.pdb` or download from RCSB | Standard format, works for most entries |
| **mmCIF** | Upload `.cif` / `.mmcif` or download from RCSB | Recommended for large structures or newer PDB entries that lack `.pdb` files |

CIF files are automatically converted to PDB using a multi-strategy cascade: **gemmi** → **OpenBabel** → **ProDy**. If PDB download from RCSB fails, the app automatically falls back to CIF format.

---

## 🖥️ Ligand input modes

| Mode | Description |
|---|---|
| **SMILES string** | Type or paste any valid SMILES |
| **Upload file** | `.pdb` — converted automatically |
| **Draw in Ketcher** | Full 2D chemical sketcher in the browser → SMILES exported automatically |

---

## ♻️ Redocking validation

Available in **both single and batch** docking modes:

| Feature | Description |
|---|---|
| **Co-crystal reference docking** | Dock the known co-crystal ligand alongside your candidate(s) |
| **RMSD vs crystal** | Heavy-atom RMSD calculated against the original crystal pose |
| **Reference score line** | Dashed line on the affinity plot for quick visual comparison |
| **Pose confirmation** | Browse reference poses, confirm which one to use as the baseline |
| **Download** | Export reference poses as SDF/PDBQT |

---

## 🔭 3D visualization layers

| Viewer | What you see |
|---|---|
| **Receptor prep** | Protein cartoon · co-crystal ligand (magenta) · docking grid box (cyan wireframe) · XYZ axis arrows |
| **Animated pose viewer** | All poses swept as frames · protein surface · co-crystal overlay |
| **Interactive pose selector** | Single selected pose · protein cartoon + surface · co-crystal overlay |
| **Binding pocket view** | Faint full-protein cartoon · docked pose (cyan) · interacting residues (orange sticks) · optional residue labels |
| **Redocking browser** | Reference ligand poses · crystal overlay · RMSD per pose |

---

## 🖥️ Run locally

### Linux (Ubuntu / Debian)
```bash
sudo apt install python3.11 python3.11-venv openbabel libopenbabel-dev \
  libcairo2-dev libpangocairo-1.0-0 pkg-config python3-dev && \
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

> **Anyone Can Dock**
> Hengphasatporn, K.; Bunchuay T.; Duan, L.; Shigeta, Y., *Journal of Chemical Information and Modeling*, 2026
> https://github.com/nyelidl/anyone-docking/

> **PLIP** *(optional, for geometry-validated interactions)*
> Salentin et al., *Nucleic Acids Research*, 2015
> DOI: https://doi.org/10.1093/nar/gkv315

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
> Hengphasatporn, K. et al., *J. Chem. Inf. Model.* 2026, **66** (4), 1955–1963
> DOI: https://doi.org/10.1021/acs.jcim.5c02852

> **gemmi** *(optional, for CIF support)*
> Wojdyr, M., *Journal of Open Source Software*, 2022
> DOI: https://doi.org/10.21105/joss.04200

---

## 📜 License

This project is licensed under the MIT License.
See the LICENSE file for details.
