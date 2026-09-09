# <img src="https://raw.githubusercontent.com/nyelidl/anyone-docking/main/any-L.svg" width="32"> Anyone Can Dock

**Anyone can dock, Everyone can do!**

***One molecular docking workflow, four ways to run.***

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nyelidl.github.io/anyone-docking/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/anyonecandock/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Anyone Can Dock makes protein-ligand docking accessible through a **zero-install web app**, a **private local interface**, an **automation-friendly CLI**, and a **GPU-ready Google Colab notebook**.

> From structure preparation to validated docking results, without making molecular docking harder than it needs to be.

---

## 🚀 Four ways to use Anyone Can Dock (ACD)

| Mode | Best for | Start here |
|---|---|---|
| 🌐 **ACD Online** | Beginners, teaching, demonstrations, quick docking | [Open in browser →](https://nyelidl.github.io/anyone-docking/) |
| 🖥️ **ACD Local** | Private research and unrestricted local workflows | [Local installation](#-acd-local) |
| ⌨️ **ACD CLI** | Automation, reproducible research, servers, screening | `pip install anyonecandock` |
| 📓 **ACD Google Colab** | Workshops, portable research, GPU docking | [Open in Colab →](https://colab.research.google.com/drive/1tApXZyT3CGziMTLG86oQe6Q7WSycK196?usp=sharing) |

---

## 🌐 ACD Online

**Best for:** beginners, teaching, demonstrations, and quick docking without installation.

- Runs directly in a web browser
- Interactive **Basic Dock** and **Batch Dock** modes
- Download structures from RCSB or upload PDB/mmCIF files
- Automatic, manual, residue-selection, and blind-docking box placement
- Meeko receptor preparation with Open Babel fallback
- Metal, heme, cofactor, water, and HETATM handling
- SMILES, PubChem name, SDF, MOL2, and PDB ligand input
- pH-aware ligand protonation with pKaNET
- AutoDock Vina docking and reproducible random seeds
- Co-crystal redocking and RMSD validation
- Interactive 3D receptor, binding-box, ligand, and pose visualization
- Pose Browser with individual pose export
- Download SDF or PDB poses with explicit hydrogens
- Preserve all docked poses in the original PDBQT output
- Interaction diagrams using ACD, RDKit, ProLIF, and PoseView
- Batch ranking, score tables, ProLIF interaction barcodes, and ZIP export
- RDKit physicochemical descriptors, drug-likeness rules, and structural alerts
- Optional ADMET-AI predictions

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nyelidl.github.io/anyone-docking/)

---

## 🖥️ ACD Local

**Best for:** private research, full graphical functionality, and unrestricted local workflows.

Includes all major Online features, plus:

- Runs entirely on the user's own computer
- Local control of receptor, ligand, result, and temporary files
- Basic Dock and Batch Dock interfaces
- Validated ferric-heme and Compound-I preparation
- Automatic `HEME_FERRIC` versus `CPD_I` geometry detection
- Geometric ferryl oxygen and proximal cysteine identification
- Fe-O and Fe-S validation gates
- Ferryl oxygen hydrogen removal
- JSON-derived RESP charges for Fe, OXO, heme, and proximal cysteine
- Final PDBQT reread and charge validation before docking
- Fail-closed heme validation: docking starts only after required checks pass
- Receptor PDB, scores, individual poses, hydrogenated poses, diagrams, and complete result archives
- Optional local ADMET-AI analysis
- No hosted-server runtime or storage restrictions

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y \
  python3.11 \
  python3.11-venv \
  openbabel \
  libcairo2-dev \
  libpango1.0-dev \
  libpangocairo-1.0-0

git clone https://github.com/nyelidl/anyone-docking-local.git
cd anyone-docking-local

python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

streamlit run app.py
```

### macOS

```bash
brew update
brew install python@3.11 open-babel cairo pango

git clone https://github.com/nyelidl/anyone-docking-local.git
cd anyone-docking-local

python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

streamlit run app.py
```

> **Apple Silicon (M1-M4):** supported with the native `aarch64` Vina binary.

### Windows

**Recommended:** use WSL2 with Ubuntu and follow the Linux instructions above.

For native Windows:

1. Install Open Babel and add it to `PATH`.
2. Install Cairo/Pango, for example with `conda install -c conda-forge cairo pango`.
3. Clone and run the app:

```bash
git clone https://github.com/nyelidl/anyone-docking-local.git
cd anyone-docking-local
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## ⌨️ ACD CLI

**Best for:** automation, reproducible research, scripting, servers, and large screening projects.

- Commands for docking, batch docking, receptor preparation, ligand preparation, redocking, and diagrams
- Accepts PDB IDs or local PDB/mmCIF structures
- Accepts SMILES, PubChem names, SDF, MOL2, and PDB ligands
- Automatic co-crystal, manual, selection-based, and blind-docking boxes
- No artificial docking-box size limit
- Preparation-only mode without starting Vina
- Prepared-receptor JSON export and reuse
- pH-aware, tautomer-aware ligand preparation using pKaNET
- Microstate selection and configurable tautomer limits
- Deterministic conformer and Vina random seeds
- AutoDock Vina exhaustiveness, pose count, and energy-range controls
- Batch-safe noninteractive operation
- Automatic co-crystal redocking and heavy-atom RMSD verdicts
- Validated ferric-heme and Compound-I preparation
- RESP charge assignment and final PDBQT validation
- Multi-heme-center support
- PDBQT, SDF, bond-order-corrected SDF, PDB, CSV, and SVG output
- Eight interaction classes, including hydrogen bonding, hydrophobic, aromatic, ionic, halogen, and metal/heme coordination
- Python API access for integration into custom pipelines

### Install from PyPI

```bash
pip install anyonecandock
```

### Commands

```text
acd <command> [options]

  dock      Full pipeline: receptor + ligand + Vina (single ligand)
  redock    Self-docking validation; no SMILES input needed
  batch     Batch docking from a .smi file or SMILES list
  receptor  Prepare receptor only
  ligand    Prepare ligand only
  diagram   Generate a 2D interaction diagram SVG
```

### Quick examples

```bash
# Dock a single ligand by compound name
acd dock --receptor 1M17 --compound erlotinib

# Dock by SMILES
acd dock --receptor 4AGN --smiles "CCO" --name ethanol

# Reuse a prepared receptor
acd dock --receptor-json ./rec/receptor_summary.json --compound baicalein

# Dock from local structure files
acd dock --receptor structure.cif --ligand-file ligand.sdf --name mylig

# Self-docking validation
acd redock --receptor 4AGN
acd redock --receptor structure.cif --fmt CIF
acd redock --receptor 4AGN --resname DC3 --diagram

# Batch docking
acd batch --receptor 1M17 --ligands compounds.smi
acd batch --receptor 4AGN --smiles-list "CCO ethanol" "c1ccccc1O phenol"
```

### Redocking verdict

The `redock` workflow automatically identifies and re-docks the co-crystal ligand. SMILES are obtained in this order:

1. RCSB Chemical Component Dictionary (CCD)
2. CIF `_chem_comp.pdbx_smiles`
3. 3D coordinate conversion as a fallback

The heavy-atom RMSD verdict is reported automatically:

- **PASS:** RMSD <= 2.0 Å
- **BORDERLINE:** RMSD > 2.0 Å and <= 3.0 Å
- **FAIL:** RMSD > 3.0 Å

### Python API

```python
from anyonecandock import core

SEED = 72
BOX_SIZE = (18, 18, 18)
vina_bin = "/path/to/vina"

result = core.prepare_receptor(
    raw_pdb="raw.pdb",
    wdir="./rec",
    box_size=BOX_SIZE,
)

lig = core.prepare_ligand(
    smiles="c1ccc(cc1)O",
    name="phenol",
    ph=7.4,
    wdir="./lig",
    mode="pkanet",
    use_pubchem=False,
    max_tautomers=8,
    ph_window=1.0,
    conformer_seed=SEED,
)

dock = core.run_vina(
    receptor_pdbqt=result["rec_pdbqt"],
    ligand_pdbqt=lig["pdbqt"],
    config_txt=result["config_txt"],
    vina_path=vina_bin,
    exhaustiveness=16,
    n_modes=10,
    energy_range=3,
    seed=SEED,
    wdir="./out",
    out_name="phenol",
)

print("Top score:", dock["top_score"])
```

---

## 📓 ACD Google Colab

**Best for:** workshops, education, portable research, and GPU docking without local setup.

- Guided notebook workflow with step-by-step preparation and analysis
- Upload PDB/mmCIF structures or download them from RCSB
- Receptor and pH-aware ligand preparation
- Validated ferric-heme and Compound-I handling
- Four supported docking engines:
  - AutoDock Vina 1.2.7
  - VinaXB
  - GNINA
  - Vina-GPU 2.1
- Google T4 GPU support for Vina-GPU
- Co-crystal redocking before production docking
- Interactive ligand and pose selection
- Binding-box validation
- Heavy-atom RMSD analysis across poses
- Engine-aware score extraction
- Ranked CSV results and score plots
- Individual result download or complete ZIP packaging
- Mobile- and tablet-friendly result export
- No additional `prepare_receptor.py` upload required; required preparation logic is included in the notebook/package

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tApXZyT3CGziMTLG86oQe6Q7WSycK196?usp=sharing)

---

## 📊 Choose Your Version

| Capability | Online | Local | CLI | Colab |
|---|---:|---:|---:|---:|
| Graphical workflow | Yes | Yes | No | Notebook |
| No local installation | Yes | No | No | Yes |
| Basic docking | Yes | Yes | Yes | Yes |
| Batch docking | Yes | Yes | Yes | Yes |
| AutoDock Vina | Yes | Yes | Yes | Yes |
| VinaXB, GNINA, and Vina-GPU | No | No | No | Yes |
| Automated scripting | No | No | Yes | Partial |
| Interactive Pose Browser | Yes | Yes | File output | Yes |
| ADME/ADMET analysis | Yes | Yes | Optional tools | No |
| Strict ferric/Compound-I validation | Yes | Yes | Yes | Yes |
| Private local processing | No | Yes | Yes | Colab runtime |

---

## 🧬 Heme-aware docking

ACD Local, CLI, and Colab support validated preparation of ferric heme and Compound-I systems.

For supported heme centers, ACD can:

- Detect `HEME_FERRIC` versus `CPD_I` from local Fe coordination geometry
- Identify a ferryl oxygen geometrically rather than relying only on atom names
- Recognize ferryl oxygen labels such as `O`, `O1`, `OXY`, or `OXO`
- Identify the proximal cysteine sulfur geometrically
- Validate Fe-O and Fe-S distances before docking
- Remove an incorrectly retained O-H hydrogen from the ferryl oxygen
- Apply JSON-derived RESP charges to Fe, OXO, the heme group, and proximal cysteine
- Re-read the final PDBQT and validate required charges before docking
- Fail closed if required heme-state checks do not pass
- Handle multiple heme centers in CLI workflows

---

## 🗺️ Interaction analysis

ACD supports eight interaction classes in its geometry-based analysis, including:

- Hydrogen bonds
- Hydrophobic contacts
- Aromatic interactions
- Cation-pi interactions
- Ionic interactions
- Metal/heme coordination
- Halogen bonds
- H···halogen contacts

Depending on the interface, results can also be visualized with **RDKit**, **ProLIF**, and **PoseView**.

---

## 🧬 Supported protein types

| Protein class | Support | Notes |
|---|---|---|
| Standard single-chain proteins | ✅ Full | Primary use case |
| Multi-chain / homo-oligomers | ✅ Full | Multi-chain structures supported |
| Heme proteins | ✅ Full | Ferric heme and Compound-I supported in Local/CLI/Colab |
| Metal-binding proteins | ✅ Full | Common metal ions can be retained during preparation |
| MD simulation outputs | ✅ Full | PDB structures from common MD workflows can be used |
| Non-standard ligand names | ✅ Full | Ligand-like HETATM records can be detected |
| Modified amino acids | ✅ Full | Supported when retained as part of the receptor |
| Multiple co-crystal ligands | ✅ Full | Ligand selection supported |
| Cofactor-binding proteins | ✅ Full | Cofactors can be retained or removed as appropriate |
| RNA / DNA targets | ⚠️ Partial | No nucleic-acid-specific interaction model |
| Covalent docking | ❌ No | Standard Vina docking is non-covalent |

---

## 🤖 AI interfaces

AI interfaces are optional front ends to ACD rather than separate docking engines.

### Anyone Can Dock GPT

Ask ChatGPT to dock molecules using natural-language instructions.

[![ChatGPT](https://img.shields.io/badge/ChatGPT-Anyone_Can_Dock_GPT-10a37f?logo=openai&logoColor=white)](https://chatgpt.com/g/g-6a0455faa96481918503be2b696e13ce-anyone-can-dock-gpt)

### Anyone Can Dock in Claude

Connect Claude to the ACD API as a custom MCP connector.

**MCP server URL:**

```text
https://anyone-can-dock-mcp.anyonecandock.workers.dev
```

[![Add to Claude](https://img.shields.io/badge/Claude-Add_Anyone_Can_Dock-cc7b4b?logo=anthropic&logoColor=white)](https://claude.ai/customize/connectors)

Example prompts:

```text
"dock quercetin into JAK2 and report binding affinity"
"compare erlotinib vs gefitinib binding to EGFR (1M17)"
```

---

## 💻 Platform compatibility

| Platform | Vina binary | Open Babel | Status |
|---|---|---|---|
| Linux x86_64 | Auto-download | `apt install openbabel` | Fully supported |
| macOS Intel | Auto-download | `brew install open-babel` | Fully supported |
| macOS Apple Silicon | Native `aarch64` | `brew install open-babel` | Fully supported |
| Windows x86_64 | Auto-download | Installer / WSL2 | Supported; WSL2 recommended |
| Streamlit Cloud | Auto-download | via `packages.txt` | Fully supported |
| Google Colab | Auto-download | `apt install openbabel` | Fully supported |

---

## 📄 Citation

If you use ACD in research, please cite the relevant methods used in your workflow.

> **AutoDock Vina 1.2.7**  
> Eberhardt et al., *J. Chem. Inf. Model.*, 2021. DOI: [10.1021/acs.jcim.1c00203](https://doi.org/10.1021/acs.jcim.1c00203)

> **DFDD**  
> Hengphasatporn, K.; Duan, L.; Harada, R.; Shigeta, Y., *J. Chem. Inf. Model.*, 2026. DOI: [10.1021/acs.jcim.5c02852](https://doi.org/10.1021/acs.jcim.5c02852)

> **Anyone Can Dock: An Online Molecular Docking Tool for Everyone**  
> Hengphasatporn, K.; Bunchuay T.; Duan, L.; ; Shigeta, Y., *J. Cheminformatics.*, 2026. DOI: [10.21203/rs.3.rs-9763995/v1](https://doi.org/10.21203/rs.3.rs-9763995/v1)

> **RDKit** · Landrum, G. (2023) · https://www.rdkit.org

> **ProDy** · Bakan et al., *Bioinformatics*, 2011. DOI: [10.1093/bioinformatics/btr168](https://doi.org/10.1093/bioinformatics/btr168)

> **stmol** · Nápoles-Duarte et al., *Front. Mol. Biosci.*, 2022. DOI: [10.3389/fmolb.2022.990846](https://doi.org/10.3389/fmolb.2022.990846)

> **Dimorphite-DL** · Ropp et al., *J. Cheminform.*, 2019. DOI: [10.1186/s13321-019-0336-9](https://doi.org/10.1186/s13321-019-0336-9)

> **pKaNET Cloud** · Please cite the corresponding manuscript when available.

> **gemmi** *(optional, for CIF support)* · Wojdyr, M., *JOSS*, 2022. DOI: [10.21105/joss.04200](https://doi.org/10.21105/joss.04200)

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Anyone Can Dock: from structure preparation to validated docking results, without making molecular docking harder than it needs to be.**
