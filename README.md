# 🧬 Anyone Docking

**Anyone can dock, everyone can do: Browser-based molecular docking — no installation required.**

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

If you use this tool in research, please cite AutoDock Vina:

> Eberhardt et al. (2021) *J. Chem. Inf. Model.* 61(8), 3891–3898. https://doi.org/10.1021/acs.jcim.1c00203

---

## 📝 License

MIT
