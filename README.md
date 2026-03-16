# <img src="https://raw.githubusercontent.com/nyelidl/anyone-docking/main/any-L.svg" width="32"> nyone can dock, everyone can do!

**Anyone docking: Browser-based molecular docking — no installation required.**

<a href="https://nyelidl.github.io/anyone-docking/" target="_blank">
  <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App"/>
</a>

> Paste a SMILES, draw a structure, or upload a file. Pick a PDB. Dock in seconds.

---

## 🚀 Try it live

**[anyone-docking.streamlit.app](https://anyone-docking.streamlit.app/)**

---

## ✨ What it does

| | |
|---|---|
| 🔬 | **Single & batch docking** via AutoDock Vina 1.2.7 |
| 🏗️ | **Automatic receptor prep** — download any PDB, strip solvent, add hydrogens |
| 🎯 | **Auto grid detection** from co-crystal ligand centroid with XYZ axis & box overlay |
| ✏️ | **3-way ligand input** — SMILES text, file upload (.pdb), or **draw in Ketcher** |
| ♻️ | **Redocking validation** with reference score line in batch plot |
| 🧪 | **Bond-order correction** — fixes PDBQT aromaticity artifacts before visualization |
| 🗺️ | **2D interaction diagrams** — docked pose via PoseView v1 + co-crystal reference via PoseView2, side-by-side (PNG + SVG download) |
| 🔭 | **Binding pocket viewer** — 4.5 Å shell of interacting residues (orange sticks) around the docked ligand, with toggleable residue labels |
| 🤖 | **AI-ready prompt** — auto-filled context for GPT, Claude, Gemini, or DeepSeek; legend adapts when both diagrams are generated |
| 📊 | **3D viewers** — animated multi-pose sweep, interactive pose selector, and dedicated binding pocket view |
| 📁 | **One-click ZIP** — all poses, bond-order-corrected SDFs, 2D diagrams, and batch score plot |

---

## 🖥️ Ligand input modes

| Mode | Description |
|---|---|
| **SMILES string** | Type or paste any valid SMILES |
| **Upload file** | `.pdb` — converted automatically |
| **Draw in Ketcher** | Full 2D chemical sketcher in the browser → SMILES exported automatically |

---

## 🔭 3D visualization layers

| Viewer | What you see |
|---|---|
| **Receptor prep** | Protein cartoon · co-crystal ligand (magenta) · docking grid box (cyan wireframe) · XYZ axis arrows |
| **Animated pose viewer** | All poses swept as frames · protein surface · co-crystal overlay |
| **Interactive pose selector** | Single selected pose · protein cartoon + surface · co-crystal overlay |
| **Binding pocket view** | Faint full-protein cartoon · docked pose (cyan) · interacting residues ≤4.5 Å (orange sticks) · optional residue labels |


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

### Streamlit Cloud deployment

Place these files in your repo root:

```
anyone-docking/
├── app.py
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

---

## 📜 License

This project is licensed under the MIT License.  
See the LICENSE file for details.
