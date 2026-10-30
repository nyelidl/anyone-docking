"""Interaction classification on saved poses, and the empty-box sanity gate.

Fixture: caffeine docked into adenosine A2A (PDB 5MZP) by the deployed API,
receptor trimmed to residues within 10 Å of the pose. The crystal structure
shows an H-bond to Asn253 and π-stacking with Phe168.
"""
import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

pytest.importorskip("rdkit")
pytest.importorskip("prody")
sys.path.insert(0, str(ROOT / "anyonecandock"))  # CLI package source with core.py

spec = importlib.util.spec_from_file_location("pose_access_ix_test", ROOT / "st/pose_access.py")
pose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pose)


def _classify(sdf=True):
    return pose.classify_interactions(
        DATA / "5mzp_caffeine_pocket.pdb",
        DATA / "5mzp_caffeine_pose1.sdf" if sdf else None,
        DATA / "5mzp_caffeine_pose1.pdb",
        CAFFEINE,
    )


@pytest.mark.parametrize("use_sdf", [True, False])
def test_caffeine_a2a_key_interactions(use_sdf):
    interactions, source, warning = _classify(use_sdf)
    assert source.startswith("ACD geometric interaction classifier")
    assert "not experimentally observed" in warning
    found = {(i["type"], i["protein_residue"]) for i in interactions}
    assert ("hbond", "ASN253") in found
    assert ("pi_pi", "PHE168") in found


def test_interactions_are_unique_and_sorted():
    interactions, _, _ = _classify()
    keys = [(i["type"], i["protein_chain"], i["protein_residue"]) for i in interactions]
    assert len(keys) == len(set(keys))
    dists = [i["distance_angstrom"] for i in interactions]
    assert dists == sorted(dists)
    for i in interactions:
        assert set(i) >= {"type", "label", "protein_residue", "protein_chain", "distance_angstrom"}


def test_classifier_never_raises_on_bad_input(tmp_path):
    bad = tmp_path / "empty.pdb"
    bad.write_text("END\n")
    interactions, source, warning = pose.classify_interactions(bad, None, bad, "")
    assert interactions == []
    assert source in ("not_available", "not_configured")
    assert "Proximity contacts are distances only" in warning


def _box_counter():
    tree = ast.parse((ROOT / "st/api.py").read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_receptor_heavy_atoms_in_box")
    ns = {"Path": Path, "Dict": Dict, "Any": Any}
    exec(compile(ast.Module([fn], []), "api.py", "exec"), ns)
    return ns["_receptor_heavy_atoms_in_box"]


def test_box_gate_counts_pocket_atoms():
    count = _box_counter()
    rec = str(DATA / "5mzp_caffeine_pocket.pdb")
    in_box, total = count(dict(rec_fh=rec, cx=-21.5376, cy=6.2062, cz=17.5341, sx=20, sy=20, sz=20))
    assert total > 0 and in_box >= 50


def test_box_gate_detects_box_outside_protein():
    count = _box_counter()
    rec = str(DATA / "5mzp_caffeine_pocket.pdb")
    # The 1M17 grid placed on the 5MZP receptor: the silent 0.000 kcal/mol failure mode.
    in_box, total = count(dict(rec_fh=rec, cx=22.0137, cy=0.2528, cz=52.7940, sx=20, sy=20, sz=20))
    assert total > 0 and in_box == 0


def test_box_gate_is_permissive_when_box_unknown():
    count = _box_counter()
    assert count(dict(rec_fh=str(DATA / "5mzp_caffeine_pocket.pdb"))) == (None, None)
