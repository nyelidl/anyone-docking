# <img src="https://raw.githubusercontent.com/nyelidl/anyone-docking/main/any-L.svg" width="32"> nyone can dock, everyone can do!

**Anyone docking: Browser-based molecular docking — no installation required.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://anyone-docking.streamlit.app/)

> Paste a SMILES. Pick a PDB. Dock in seconds.

---

## 🚀 Try it live

**[anyone-docking.streamlit.app](https://anyone-docking.streamlit.app/)**

---

## ✨ What it does

| | |
|---|---|
| 🔬 | **Single & batch docking** via AutoDock Vina 1.2.7 |
| 🏗️ | **Automatic receptor prep** — download any PDB, strip solvent, add hydrogens |
| 🎯 | **Auto grid detection** from co-crystal ligand centroid |
| ♻️ | **Redocking validation** with RMSD against the crystal pose |
| 🧪 | **Bond-order correction** — fixes PDBQT aromaticity artifacts before visualization |
| 🗺️ | **2D interaction diagrams** via Proteins.Plus PoseView (PNG + SVG download) |
| 🤖 | **AI-ready prompt** — copy–paste into GPT, Claude, Gemini, or DeepSeek |
| 📊 | **3D viewer** — whole-protein view, ligand-centered camera, binding-pocket surface |
| 📁 | **One-click ZIP** — all poses, corrected SDFs, and interaction diagrams |

---

## 🛠️ Stack

`AutoDock Vina 1.2.7` · `Streamlit` · `RDKit` · `Meeko` · `OpenBabel` · `py3Dmol` · `Proteins.Plus`

---

## 🖥️ Run locally

```bash
git clone https://github.com/your-username/anyone-docking.git
cd anyone-docking
pip install -r requirements.txt
streamlit run app.py
```

> Requires Python 3.10+, OpenBabel, and libcairo2.  
> AutoDock Vina binary is downloaded automatically on first launch.

---

## 📄 Citation

If you use this tool in research, please cite the following software and resources used in this workflow:

> **AutoDock Vina 1.2.7**  
> Eberhardt et al., *Journal of Chemical Information and Modeling*, 2021  
> DOI: https://doi.org/10.1021/acs.jcim.1c00203  

> **ProteinsPlus / PoseView**  
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

## 📝 License

MIT
