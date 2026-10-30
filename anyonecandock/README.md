# anyone can dock — CLI

Molecular docking from the command line, powered by **AutoDock Vina 1.2.7**, **pKaNET Cloud+**, and **RDKit**.

## Installation

### System dependencies (Ubuntu / Debian)

```bash
sudo apt-get install -y openbabel libopenbabel-dev \
  libcairo2-dev libpango1.0-dev libpangocairo-1.0-0
```

### pip (recommended)

```bash
# 1. clone the repository
git clone https://github.com/nyelidl/anyone-docking.git
cd anyone-docking

# 2. install in editable mode  (add [all] for ADMET-AI + PNG export)
pip install -e .
pip install -e ".[all]"
```

### conda

```bash
conda create -n acd python=3.11 -y
conda activate acd

# system libs via conda-forge
conda install -c conda-forge openbabel cairosvg rdkit prody -y

# package itself
pip install -e .
```

After installation the `acd` command is available in your shell.

---

## Commands

```
acd --help
acd dock     --help
acd batch    --help
acd codock   --help
acd ensemble --help
acd receptor --help
acd ligand   --help
acd diagram  --help
```

---

## Quick examples

### 1. Full single-ligand docking (PDB ID + SMILES)

```bash
acd dock \
  --receptor 1M17 \
  --smiles "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC" \
  --name erlotinib \
  --ph 7.4 \
  --exhaustiveness 16 \
  --output ./erlotinib_dock
```

Output in `./erlotinib_dock/`:
```
receptor/rec.pdb
receptor/rec.pdbqt
receptor/receptor_summary.json
ligand/erlotinib.pdbqt
erlotinib_out.pdbqt      ← all poses
erlotinib_out.sdf
erlotinib_pv_ready.sdf   ← bond-order corrected
erlotinib_scores.csv
```

### 2. Add 2D interaction diagram and save per-pose files

```bash
acd dock \
  --receptor 1M17 \
  --smiles "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC" \
  --name erlotinib \
  --diagram \
  --save-poses \
  --output ./erlotinib_full
```

### 3. Use a local PDB/CIF file

```bash
acd dock \
  --receptor my_homology_model.pdb \
  --smiles "CCc1nn(C)c2cc(-c3cncnc3Nc3cccc(S(N)(=O)=O)c3)ccc12" \
  --name compound_1 \
  --center manual --cx 14.2 --cy -3.8 --cz 22.1 \
  --bx 22 --by 22 --bz 22
```

### 4. Batch docking from a .smi file

```bash
# compounds.smi: one "SMILES name" per line
cat > compounds.smi << 'EOF'
O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12 Apigenin
O=c1cc(-c2ccccc2)oc2cc(O)c(O)c(O)c12 Baicalein
COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC Erlotinib
EOF

acd batch \
  --receptor 1M17 \
  --ligands compounds.smi \
  --ph 7.4 \
  --exhaustiveness 8 \
  --redock-smiles "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC Erlotinib" \
  --output ./batch_egfr
```

Output `./batch_egfr/batch_scores.csv`:
```
Name,SMILES,Prepared SMILES,Top Score,Charge,Status,...
Erlotinib,...,-9.12,0,OK,...
Apigenin,...,-8.44,0,OK,...
Baicalein,...,-8.01,0,OK,...
```

### 4a. Simultaneous docking of a ligand set

Use `codock` when two or more ligands should bind **together** in one Vina search.
This is different from `batch`, which docks and ranks each ligand independently.
The resulting affinity is for the joint ligand complex; it is not a per-ligand score.

```bash
acd codock --receptor 5X72 --ligand ligand_p59.pdbqt ligand_p69.pdbqt \
  --center manual --cx -15 --cy 15 --cz 129 --bx 30 --by 24 --bz 24 \
  --exhaustiveness 32 --output ./joint_5x72
```

`--ligand` accepts two or more quoted SMILES (`"CCO ethanol"`), compound names
(`"aspirin"`, resolved through PubChem), ligand structure paths (`.sdf`, `.mol2`,
`.pdb`, `.pdbqt`), or a `.smi` file with one `SMILES [name]` per line. Inputs can
be mixed. The older `--ligands`, `--smiles-list`, and `--ligand-files` options
remain available for existing scripts.
The output includes `joint_out.pdbqt` (joint poses) and `joint_results.json`
(ligand labels, files, and joint scores). Check both ligands in each pose before
interpreting the result.

### 5. Prepare receptor once, re-use for many batch runs

```bash
# Step A — prepare receptor (saved as JSON)
acd receptor --pdb 1M17 --output ./rec_1m17

# Step B — batch with saved receptor (no re-download or re-prep)
acd batch \
  --receptor-json ./rec_1m17/receptor_summary.json \
  --ligands library.smi \
  --output ./screen_run_1
```

### 6. Dock against a receptor ensemble

```bash
acd ensemble \
  --receptors "ensemble/*.pdb" \
  --ligands compounds.smi \
  --center manual --cx 14.2 --cy -3.8 --cz 22.1 \
  --bx 22 --by 22 --bz 22 \
  --seed 42 \
  --output ./ensemble_screen
```

Each receptor is prepared and validated independently, including ferric-heme
and Compound-I validation when present. A failed receptor is recorded and
skipped without bypassing validation. Use one manual box only for structurally
aligned receptors; otherwise use automatic or selection-based centering.

Output in `./ensemble_screen/`:

```text
001_model_a/                 complete per-receptor batch output
002_model_b/
ensemble_receptors.csv       preparation and run status for each receptor
ensemble_scores.csv          all receptor-ligand score rows
ensemble_summary.csv         best, mean, median, and population SD per ligand
```

Ensemble score summaries are ranking aids and are not population-weighted
thermodynamic binding energies.

### 7. Stand-alone 2D diagram for an existing docking result

```bash
acd diagram \
  --receptor ./rec_1m17/receptor/rec.pdb \
  --pose-sdf  ./erlotinib_dock/erlotinib_pv_ready.sdf \
  --smiles "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC" \
  --name erlotinib \
  --pose 1 \
  --score -9.4 \
  --output ./diagrams
```

---

## Programmatic use (Python API)

```python
from anyonecandock import (
    prepare_receptor,
    prepare_ligand,
    run_vina,
    get_vina_binary,
    fix_sdf_bond_orders,
    draw_interaction_diagram,
)
from pathlib import Path

wdir = Path("./my_run")
wdir.mkdir(exist_ok=True)

# 1. Receptor
rec = prepare_receptor(
    raw_pdb="1M17.pdb",
    wdir=wdir,
    center_mode="auto",
    box_size=(18, 18, 18),
)

# 2. Ligand
lig = prepare_ligand(
    smiles="COCCOC1=C(...)OCCOC",
    name="erlotinib",
    ph=7.4,
    wdir=wdir,
    mode="pkanet",
    conformer_seed=None,  # default: random; set a positive integer to reproduce 3D coordinates
)

# 3. Vina
vina, _ = get_vina_binary()
dock = run_vina(
    receptor_pdbqt=rec["rec_pdbqt"],
    ligand_pdbqt=lig["pdbqt"],
    config_txt=rec["config_txt"],
    vina_path=vina,
    exhaustiveness=16,
    n_modes=10,
    energy_range=3,
    wdir=wdir,
    out_name="erlotinib",
)
print(f"Best score: {dock['top_score']:.2f} kcal/mol")

# 4. 2D diagram
svg = draw_interaction_diagram(
    receptor_pdb=rec["rec_fh"],
    pose_sdf=dock["out_sdf"],
    smiles=lig["prot_smiles"],
    title="Erlotinib · 1M17",
)
Path("diagram.svg").write_bytes(svg)
```

---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `VINA_PATH` | auto-downloaded | Point to a custom Vina binary |
| `ACD_WORKDIR` | temp dir per run | Persistent working directory |

---

## Optional extras

| Feature | Install |
|---|---|
| PNG export for diagrams | `pip install cairosvg` |
| ADMET predictions | `pip install admet-ai` |
| Pretty terminal tables | `pip install rich` |
| ML pKa backend | `pip install pkasolver` |

---

## Citation

If you use Anyone Can Dock in your research, please cite:

> Hengphasatporn, K.; Bunchuay, T.; Duan, L.; Shigeta, Y. Anyone Can Dock: an online molecular docking tool for everyone. *J. Cheminform.* 2026. DOI: [10.1186/s13321-026-01306-2](https://doi.org/10.1186/s13321-026-01306-2)

Please also cite AutoDock Vina:

> Eberhardt, J.; Santos-Martins, D.; Tillack, A. F.; Forli, S. *J. Chem. Inf. Model.* 2021, 61, 3891–3898. DOI: [10.1021/acs.jcim.1c00203](https://doi.org/10.1021/acs.jcim.1c00203)

## Ligand structure files

`--ligand-file` accepts `.sdf`, `.mol2`, `.pdb`, and `.pdbqt` (case insensitive).

```bash
# Prepared PDBQT: original file passed directly to Vina, unchanged
acd dock \
  --receptor-json receptor_summary.json \
  --ligand-file ligand.pdbqt \
  --name ligand \
  --output results

# PDB: existing file preparation and Meeko writer, without adding hydrogen
acd dock \
  --receptor-json receptor_summary.json \
  --ligand-file ligand.pdb \
  --name ligand \
  --no-add-h \
  --output results
```

PDBQT direct mode performs read-only format checks and verifies the original
SHA256 against the file supplied to Vina immediately before launch. It performs
no ligand conversion, protonation, hydrogen changes, or atom/type/charge edits.
`--name` controls output filenames only. Missing, empty, malformed, or multi-pose
PDBQT inputs fail before docking. Supply one prepared ligand with a valid torsion
tree (`ROOT`, `ENDROOT`, optional branches, and `TORSDOF`). Validation checks file
structure; Vina remains responsible for engine-specific chemical/type support.

PDB files use `prepare_ligand_from_file` and the existing `_meeko_to_pdbqt`
implementation (RDKit → Meeko). They do not enter the SMILES `prepare_ligand`
protonation/pKaNET path. Open Babel is not used for ligand PDB → PDBQT conversion,
including on failure. Normal PDB mode adds missing hydrogens; `--no-add-h`
disables that addition and requires an input suitable for Meeko, normally with
explicit hydrogens already present. For PDB input, this option now also disables
Meeko hydrogen merging (`merge_these_atom_types=()`), preserving the original
atom set. Meeko still assigns AutoDock types, Gasteiger partial charges, and
torsions. Normal PDB mode retains Meeko’s default nonpolar hydrogen merging.

`results/ligand/atom_mapping.tsv` records original atom identity, names, residue
name/number, chain, insertion code, coordinates, input formal charge, output
partial charge/type, and Meeko's explicit atom mapping. Original coordinates are
kept, including planar PDB input. New hydrogen atoms are identified separately.
In normal mode, nonpolar hydrogens merged by Meeko are reported with their parent;
Meeko's torsion-tree reordering is reported without rearranging the PDBQT tree.
Original four-column atom-name fields are restored from the input PDB using
Meeko’s explicit RDKit-index-to-output-serial mapping, after identity validation.
The TSV includes zero-based `input_rdkit_index` and names before/after restoration.
Only the atom-name field is changed by this restoration; charges, types,
coordinates, serials, and torsion records remain as Meeko wrote them. Unexpected
drops, duplicates, coordinate changes, or ambiguous mapping stop docking. Alternate locations, multiple models, disconnected PDB fragments,
and Meeko-generated pseudoatoms are rejected when unique mapping cannot be
established. PDB bond-order information is limited: some ligands require a better
specified input or externally prepared PDBQT rather than automatic repair.

SDF/MOL2 preparation and their existing Open Babel paths remain available.
Open Babel may also still be used for receptor preparation and conversion of
**docked output** PDBQT to SDF. These are separate from ligand input preparation.
See [PDB atom-name restoration](PDB_ATOM_NAME_RESTORATION.md) and
[the implementation audit](LIGAND_WORKFLOW_AUDIT.md) for source references,
validation results, and before/after workflows.
