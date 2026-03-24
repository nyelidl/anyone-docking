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
| 🗺️ | **2D interaction diagrams** — docked pose via RDKit (local) or PoseView v1 (proteins.plus) + co-crystal reference via PoseView2, side-by-side (PNG + SVG download) |
| 🔭 | **Binding pocket viewer** — interacting residues (orange sticks) around the docked ligand, with toggleable residue labels and adjustable distance cutoff |
| 🤖 | **AI-ready prompt** — auto-filled context for GPT, Claude, Gemini, or DeepSeek; legend adapts when both diagrams are generated |
| 📊 | **3D viewers** — animated multi-pose sweep, interactive pose selector, and dedicated binding pocket view |
| 📁 | **One-click ZIP** — all poses, bond-order-corrected SDFs, 2D diagrams, and score plot |

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

```bash
git clone https://github.com/nyelidl/anyone-docking.git
cd anyone-docking
pip install -r requirements.txt
streamlit run app.py
```

> Requires Python 3.10+, OpenBabel (`apt install openbabel`), and libcairo2 (`apt install libcairo2-dev libpangocairo-1.0-0`).
> AutoDock Vina binary is downloaded automatically on first launch.

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

If you use this tool in research, please cite the following software and resources used in this workflow:

> **AutoDock Vina 1.2.7**
> Eberhardt et al., *Journal of Chemical Information and Modeling*, 2021
> DOI: https://doi.org/10.1021/acs.jcim.1c00203

> **ProteinsPlus / PoseView & PoseView2**
> Schöning-Stierand et al., *Nucleic Acids Research*, 2022
> DOI: https://doi.org/10.1093/nar/gkac258

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
