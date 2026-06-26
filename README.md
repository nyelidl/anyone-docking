# <img src="https://raw.githubusercontent.com/nyelidl/anyone-docking/main/any-L.svg" width="32"> Anyone can dock, everyone can do!

**Anyone docking: Browser-based molecular docking — no installation required.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nyelidl.github.io/anyone-docking/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/anyonecandock/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Paste a SMILES, draw a structure, or upload a file. Pick a PDB or CIF. Dock in seconds.

> **Recent updates:** New **CLI** and **Python API** via [`anyonecandock`](https://pypi.org/project/anyonecandock/) on PyPI. New **`redock` command** — automatically re-docks the co-crystal ligand with zero SMILES input (RCSB CCD lookup → CIF block → 3D fallback). Receptor setup now auto-scans ligand-like HETATM records. Ligand preparation supports pKaNET-ranked microstates with manual rank selection.

---

## 🚀 Six ways to use Anyone Can Dock

| Mode | Best for | How |
|---|---|---|
| 🌐 **Streamlit Web App** | Quickest start, no setup | [Open in browser →](https://nyelidl.github.io/anyone-docking/) |
| ☁️ **Streamlit via Colab** | Web UI on free GPU/CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WtWYUUB1AREZMeB5qEJ9OD84AvWk1z4z?usp=sharing) |
| 🖥️ **Streamlit locally** | Full control, own machine | `pip install anyonecandock && streamlit run app.py` |
| 📓 **Colab notebook** | Batch docking, scripting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1e23e0145ja6zJi0HibA6_JBO7p78Xvyw?usp=sharing) |
| ⌨️ **CLI (`acd`)** | Automation, pipelines, scripts | `pip install anyonecandock` then `acd dock ...` |
| 🐍 **Python API** | Programmatic integration | `from anyonecandock import core` |
| 🤖 **GPT / Claude plugin** | Natural-language docking | See below ↓ |

---

## 🤖 AI Interfaces

### Anyone Can Dock GPT
Ask GPT-4o to dock molecules using natural language.

[![ChatGPT](https://img.shields.io/badge/ChatGPT-Anyone_Can_Dock_GPT-10a37f?logo=openai&logoColor=white)](https://chatgpt.com/g/g-6a0455faa96481918503be2b696e13ce-anyone-can-dock-gpt)

### Anyone Can Dock in Claude
Connect Claude to the ACD API as a custom MCP connector — ask Claude to dock molecules, search PDB targets, and interpret results directly in the chat.

[![Add to Claude](https://img.shields.io/badge/Claude-Add_Anyone_Can_Dock-cc7b4b?logo=anthropic&logoColor=white)](https://claude.ai/settings/connectors)

**Example prompts after connecting:**
```
"baicalein จับกับ SARS-CoV-2 main protease ได้ไหม docking มาที"
"dock quercetin into JAK2 and report binding affinity"
"compare erlotinib vs gefitinib binding to EGFR (1M17)"
```

**Powered by:** AutoDock Vina 1.2.7 · pKaNET protonation at pH 7.4 · MCP Streamable HTTP

> MCP server URL: `https://anyone-can-dock-mcp.anyonecandock.workers.dev`

---

## ⌨️ CLI — `acd` command

Install from PyPI:

```bash
pip install anyonecandock
```

### Commands

```
acd <command> [options]

  dock      Full pipeline: receptor + ligand + Vina (single ligand)
  redock    Self-docking validation — no SMILES input needed
  batch     Batch docking from a .smi file or SMILES list
  receptor  Prepare receptor only (download/convert → .pdb + .pdbqt + box)
  ligand    Prepare ligand only (SMILES/file → .pdbqt + .sdf)
  diagram   Generate a 2D interaction diagram SVG for a completed run
```

### Quick examples

```bash
# Dock a single ligand by compound name
acd dock --receptor 1M17 --compound erlotinib

# Dock by SMILES
acd dock --receptor 4AGN --smiles "CCO" --name ethanol

# Dock from an existing receptor preparation (skips re-prep)
acd dock --receptor-json ./rec/receptor_summary.json --compound baicalein

# Dock from a structure file
acd dock --receptor structure.cif --ligand-file ligand.sdf --name mylig

# ── Redocking (self-docking validation) — NEW ──────────────────────────
# Re-docks the co-crystal ligand automatically; no SMILES required.
# SMILES acquired via: RCSB CCD → CIF _chem_comp.pdbx_smiles → 3D fallback
acd redock --receptor 4AGN
acd redock --receptor structure.cif --fmt CIF
acd redock --receptor 4AGN --resname DC3 --diagram
acd redock --receptor-json ./rec/receptor_summary.json

# Batch docking
acd batch --receptor 1M17 --ligands compounds.smi
acd batch --receptor 4AGN --smiles-list "CCO ethanol" "c1ccccc1O phenol"
```

### `acd redock` — self-docking validation

The `redock` command automatically identifies and re-docks the co-crystal ligand present in a PDB/CIF structure — **no SMILES input required**.

```
acd redock --receptor <PDB_ID_or_FILE> [options]

Options:
  --resname     Override the auto-detected ligand residue name (e.g. DC3, ATP)
  --name        Output name for the re-docked ligand (default: residue code)
  --ph          Target pH for protonation (default: 7.4)
  --diagram     Generate 2D interaction diagram after docking
  --save-poses  Save each docked pose as an individual SDF/PDB file
  -e / --exhaustiveness   Vina exhaustiveness (default: 16)
  -n / --poses            Max poses to output (default: 10)
  -o / --output           Output directory (default: ./acd_redock)
```

**SMILES acquisition order:**
1. **RCSB CCD REST API** — ideal, stereo-correct, curated SMILES from the PDB chemical component dictionary
2. **CIF `_chem_comp.pdbx_smiles`** — parsed directly from the structure CIF (no network required)
3. **3D coordinate conversion** — RDKit + OpenBabel with hydrogen-aware bond perception (last resort; a warning is shown)

**Redocking verdict** (printed automatically):
- ✓ **PASS** — best pose RMSD ≤ 2.0 Å vs crystal
- ⚠ **BORDERLINE** — RMSD 2–3 Å
- ✗ **FAIL** — RMSD > 3.0 Å

### All `acd dock` options

```bash
acd dock --help

  --receptor       PDB ID (auto-downloaded) or path to .pdb/.cif
  --receptor-json  JSON from a previous acd receptor run (skips re-prep)
  --fmt            PDB or CIF (default: PDB)
  --smiles         Ligand SMILES string
  --compound       Compound name to search on PubChem
  --ligand-file    Ligand structure file (.sdf/.mol2/.pdb)
  --name           Output name (default: LIG)
  --ph             Target pH (default: 7.4)
  --neutral        Neutral mode: keep input charge, add H only
  --no-pubchem     Skip PubChem pKa lookup
  --center         auto / manual / selection (default: auto)
  --cx/cy/cz       Manual box centre coordinates
  --bx/by/bz       Box size in Å (default: 20 × 20 × 20)
  -e               Vina exhaustiveness (default: 16)
  -n               Max poses (default: 10)
  --redock-smiles  Dock a reference co-crystal ligand alongside yours
  --save-poses     Save each pose as individual SDF/PDB
  --diagram        Generate 2D interaction diagram after docking
  -o               Output directory (default: ./acd_results)
```

---

## 🐍 Python API

```python
from anyonecandock import core

# Prepare receptor from PDB ID
result = core.prepare_receptor("raw.pdb", wdir="./rec")

# Prepare ligand from SMILES at pH 7.4
lig = core.prepare_ligand(
    smiles="c1ccc(cc1)O",
    name="phenol",
    ph=7.4,
    wdir="./lig",
)

# Run docking
dock = core.run_vina(
    receptor_pdbqt=result["rec_pdbqt"],
    ligand_pdbqt=lig["pdbqt"],
    config_txt=result["config_txt"],
    vina_path=vina_bin,
    exhaustiveness=16,
    n_modes=10,
    wdir="./out",
    out_name="phenol",
)
print(dock["top_score"])

# Auto-detect co-crystal SMILES (used by redock)
smiles, source, warning = core.get_cocrystal_smiles(
    ligand_pdb_path = result["ligand_pdb_path"],
    cocrystal_ligand_id = result["cocrystal_ligand_id"],
    raw_pdb = "raw.cif",   # used for CIF block parsing
)
print(f"SMILES ({source}): {smiles}")

# 2D interaction diagram
svg_bytes = core.draw_interaction_diagram(
    receptor_pdb=result["rec_fh"],
    pose_sdf="phenol_out.sdf",
    smiles=lig["prot_smiles"],
    title="Phenol · 4AGN",
)
open("diagram.svg", "wb").write(svg_bytes)
```

---

## 🌐 Streamlit Web App

The simplest entry point — no installation, runs in the browser.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nyelidl.github.io/anyone-docking/)

Supports single and batch docking, all 2D diagram engines, interactive drag layout, ADME predictions, and ready-to-use figure export.

---

## ☁️ Streamlit via Google Colab

Run the full Streamlit web interface on Colab's free compute tier — no local install needed.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WtWYUUB1AREZMeB5qEJ9OD84AvWk1z4z?usp=sharing)

---

## 📓 Colab Notebook (batch docking)

Batch docking with 4 docking engines in a Python notebook environment:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1e23e0145ja6zJi0HibA6_JBO7p78Xvyw?usp=sharing)

---

## 🖥️ Run Streamlit locally

### Linux (Ubuntu / Debian)
```bash
sudo apt install python3.11 python3.11-venv openbabel libcairo2-dev libpangocairo-1.0-0
pip install anyonecandock
git clone https://github.com/nyelidl/anyone-docking-local.git
cd anyone-docking-local
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### macOS
```bash
brew install python@3.11 open-babel cairo pango
git clone https://github.com/nyelidl/anyone-docking-local.git
cd anyone-docking-local
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

> **Apple Silicon (M1–M4):** Fully supported — the app auto-downloads the correct `aarch64` Vina binary.

### Windows

> **Recommended:** Use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/) with Ubuntu and follow the Linux instructions above.

For native Windows:
1. Install **OpenBabel** from [openbabel.org](https://openbabel.org/wiki/Category:Installation) and add to PATH
2. Install **Cairo/Pango** via conda: `conda install -c conda-forge cairo pango`

```bash
git clone https://github.com/nyelidl/anyone-docking-local.git
cd anyone-docking-local
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud deployment

```
anyone-docking/
├── app.py
├── core.py
├── requirements.txt   # Python packages
└── packages.txt       # System packages (openbabel, libcairo2-dev, …)
```

---

## ✨ What it does

| | |
|---|---|
| 🔬 | **Single & batch docking** via AutoDock Vina 1.2.7 |
| ♻️ | **`redock` — self-docking validation** — auto-fetches co-crystal SMILES (RCSB CCD → CIF → 3D), docks, reports RMSD verdict |
| 🏗️ | **Guided receptor prep** — download any PDB/CIF, review ligand candidates automatically, strip solvent, add hydrogens |
| 📄 | **PDB & mmCIF support** — upload `.pdb` or `.cif` files or download from RCSB (auto-fallback to CIF for large entries) |
| 🎯 | **Smart grid detection** — auto-detects ligand-like HETATM records; shows dropdown only when multiple candidates exist |
| ✏️ | **3-way ligand input** — SMILES text, file upload, or draw in **Ketcher** |
| 🧬 | **pKaNET-ranked microstates** — tautomer-aware protonation at target pH with manual rank selection [Open in browser →](https://nyelidl.github.io/pKaNET_Cloud/)|
| ⚖️ | **Conservative docking-state selection** — avoids over-deprotonated states for polyphenols, flavonoids, coumarins |
| 🧬 | **Heme-aware preparation** — HEM/HEC/HEA/HEB stripped before OpenBabel, re-injected with correct AD4 atom types |
| ⚗️ | **Water/cofactor/metal control** — remove waters, keep metals, keep/strip FAD/NAD/ATP/CoA independently |
| 🗺️ | **Three 2D diagram engines**: ACD custom SVG · RDKit · PoseView (proteins.plus) |
| 🖱️ | **Interactive drag mode** — reposition residue labels in real time, export PNG (up to 600 dpi) or SVG |
| 🔭 | **Binding pocket viewer** — interacting residues as orange sticks with adjustable distance cutoff |
| 🤖 | **AI-ready prompt** — auto-filled for Claude, GPT-4o, Gemini; adapts to redocking context |
| 📊 | **3D viewers** — animated multi-pose sweep, pose selector, binding pocket view |
| 📁 | **One-click ZIP** — all poses, corrected SDFs, diagrams, score plot, pKaNET log |

---

## 🗺️ 2D Interaction Diagrams

Three tabs — each with a different rendering engine:

### 🧬 Anyone Can Dock 2D Diagram *(default)*

| Feature | Detail |
|---|---|
| **8 interaction types** | H-bond (distance on line), hydrophobic, π-π, cation-π, ionic, metal/heme, halogen bond, H···halogen |
| **Geometry-based** | All interactions computed from 3D coordinates — no server, works offline |
| **ACS-style bonds** | Bond widths, double-bond spacing, and wedge geometry follow ACS publication standards |
| **Smart layout** | Radial placement by interaction angle; push-apart prevents overlap |
| **Interactive drag** | Reposition any residue label in real time; distance labels update live |
| **Export** | SVG (vector) · PNG at 1× / 2× (150 dpi) / 3× (300 dpi) / 4× (600 dpi) |

### 🔬 RDKit 2D Diagram

Classic highlight-circle style. H-bond (blue) · Hydrophobic (green) · Other/metal (pink). Side-by-side with co-crystal reference when available.

### 🔬 PoseView (proteins.plus)

REST API submission of receptor + docked pose. PoseView v1 (docked pose) + PoseView2 (co-crystal reference by PDB code + ligand ID). Built-in API test and manual fallback download.

> ⚠️ **PoseView limitation:** charged species are shown in neutral form. Use ACD or RDKit diagrams for ligands with formal charges.

---

## 🧬 Supported protein types

| Protein class | Support | Notes |
|---|---|---|
| Standard single-chain proteins | ✅ Full | Primary use case |
| Multi-chain / homo-oligomers | ✅ Full | Duplicate chains deduplicated; chain A ligand auto-selected |
| Heme proteins (CYP450, peroxidases, Hb, Mb) | ✅ Full | Fe-porphyrin handled separately; grid auto-centers on Fe |
| Metal-binding proteins (zinc fingers, carbonic anhydrase) | ✅ Full | ZN, MG, CA, MN, FE, CU re-injected with correct charges |
| MD simulation outputs (GROMACS, AMBER) | ✅ Full | Blank chain IDs auto-assigned to chain A |
| Non-standard ligand names (MOL, LIG, UNL, INH) | ✅ Full | |
| Modified amino acids (CYP, MSE, TPO, SEP) | ✅ Full | Backbone atom check keeps them in receptor |
| Multiple co-crystal ligands | ✅ Full | Dropdown shown only when needed |
| Cofactor-binding proteins (FAD, NAD, ATP, CoA) | ✅ Full | Kept or stripped independently |
| Glycoproteins | ⚠️ Partial | Glycans kept in receptor; not in 2D diagram |
| Membrane proteins | ⚠️ Partial | Dockable without lipids; lipids not auto-filtered |
| RNA / DNA targets | ⚠️ Partial | Basic interaction detection; no nucleic-acid-specific types |
| Covalent docking | ❌ No | Vina is non-covalent only |

---

## ♻️ Redocking validation

Available in **all modes** (Streamlit, CLI `acd redock`, API, Colab):

| Feature | Description |
|---|---|
| **Auto SMILES acquisition** | RCSB CCD API → CIF `_chem_comp.pdbx_smiles` → 3D conversion (no manual input needed) |
| **RMSD vs crystal** | Heavy-atom RMSD via MCS matching against original crystal pose |
| **Verdict** | PASS ≤ 2.0 Å · BORDERLINE 2–3 Å · FAIL > 3.0 Å |
| **Reference score line** | Dashed red line on affinity plot |
| **Pose confirmation** | Browse reference poses, pin as baseline for score comparisons |
| **Download** | Export reference poses as SDF/PDBQT |

---

## 💻 Platform compatibility

| Platform | Vina binary | OpenBabel | Status |
|---|---|---|---|
| **Linux x86_64** | ✅ Auto-download | `apt install openbabel` | Fully supported (primary) |
| **macOS Intel** | ✅ Auto-download | `brew install open-babel` | Fully supported |
| **macOS Apple Silicon** (M1–M4) | ✅ Native `aarch64` | `brew install open-babel` | Fully supported |
| **Windows x86_64** | ✅ Auto-download | [Installer](https://openbabel.org/wiki/Category:Installation) | Supported (WSL2 recommended) |
| **Streamlit Cloud** | ✅ Auto-download | via `packages.txt` | Fully supported |
| **Google Colab** | ✅ Auto-download | `!apt install openbabel` | Fully supported |

---

## 🧬 pKaNET microstate selection

| Mode | Meaning |
|---|---|
| **Auto recommended** | pKaNET recommendation. For ambiguous systems (polyphenols, coumarins, flavonoids), may choose a conservative state rather than the highest score |
| **Highest-scoring** | Top-ranked pKaNET state directly |
| **Manual rank** | Choose any ranked microstate from a dropdown before docking |

---

## ⚗️ Water, cofactor & metal options

| Option | Default | Effect |
|---|---|---|
| **Remove waters** | ✅ On | Removes crystallographic waters |
| **Keep metal ions** | ✅ On | Keeps ZN, MG, CA, MN, FE, CU, CO, NI, CD, HG, NA, K |
| **Keep cofactors** | ✅ On | Keeps ATP, ADP, FAD, FMN, NAD, CoA, SAM, HEM |

Buffers and additives (GOL, EDO, PEG, SO4, PO4) are removed by default.

---

## 📄 Citation

If you use this tool in research, please cite:

> **AutoDock Vina 1.2.7**
> Eberhardt et al., *J. Chem. Inf. Model.*, 2021 · DOI: [10.1021/acs.jcim.1c00203](https://doi.org/10.1021/acs.jcim.1c00203)

> **DFDD**
> Hengphasatporn, K.; Duan, L.; Harada, R.; Shigeta, Y., *J. Chem. Inf. Model.*, 2026 · DOI: [10.1021/acs.jcim.5c02852](https://doi.org/10.1021/acs.jcim.5c02852)

> **RDKit** · Landrum, G. (2023) · https://www.rdkit.org

> **ProDy** · Bakan et al., *Bioinformatics*, 2011 · DOI: [10.1093/bioinformatics/btr168](https://doi.org/10.1093/bioinformatics/btr168)

> **stmol** · Nápoles-Duarte et al., *Front. Mol. Biosci.*, 2022 · DOI: [10.3389/fmolb.2022.990846](https://doi.org/10.3389/fmolb.2022.990846)

> **Dimorphite-DL** · Ropp et al., *J. Cheminform.*, 2019 · DOI: [10.1186/s13321-019-0336-9](https://doi.org/10.1186/s13321-019-0336-9)

> **pKaNET Cloud** · Please cite the corresponding manuscript when available.

> **gemmi** *(optional, for CIF support)* · Wojdyr, M., *JOSS*, 2022 · DOI: [10.21105/joss.04200](https://doi.org/10.21105/joss.04200)

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
