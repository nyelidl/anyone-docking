"""
anyonecandock — molecular docking from the command line.

Quick-start
-----------
    acd dock --receptor 1M17 --smiles "CCO" --name ethanol
    acd batch --receptor 1M17 --ligands compounds.smi
    acd receptor --pdb 4AGN --output ./rec
    acd ligand --smiles "c1ccccc1O" --name phenol --ph 7.4
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("anyonecandock")
except PackageNotFoundError:
    __version__ = "dev"

from .core import (          # noqa: F401  (re-export for programmatic use)
    prepare_receptor,
    calculate_blind_box,
    prepare_ligand,
    prepare_ligand_from_file,
    run_vina,
    run_vina_multi,
    get_vina_binary,
    fix_sdf_bond_orders,
    load_mols_from_sdf,
    write_single_pose,
    write_single_pose_pdb,
    get_interacting_residues,
    calc_rmsd_heavy,
    draw_interaction_diagram,
)

__all__ = [
    "__version__",
    "prepare_receptor",
    "calculate_blind_box",
    "prepare_ligand",
    "prepare_ligand_from_file",
    "run_vina",
    "run_vina_multi",
    "get_vina_binary",
    "fix_sdf_bond_orders",
    "load_mols_from_sdf",
    "write_single_pose",
    "write_single_pose_pdb",
    "get_interacting_residues",
    "calc_rmsd_heavy",
    "draw_interaction_diagram",
]
