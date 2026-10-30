import importlib.util
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pose_access_under_test", ROOT/"st/pose_access.py")
pose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pose)

@pytest.mark.parametrize("kind,element", [("Na","Na"),("NA","N"),("K","K"),("Co","Co"),("CO","Co"),("Ni","Ni"),("NI","Ni"),("Cd","Cd"),("CD","Cd"),("Hg","Hg"),("HG","Hg"),("Ho","Ho"),("HO","Ho"),("La","La"),("CE","Ce"),("Pr","Pr"),("ND","Nd"),("Pm","Pm"),("SM","Sm"),("Eu","Eu"),("GD","Gd"),("Tb","Tb"),("DY","Dy"),("Er","Er"),("TM","Tm"),("Yb","Yb"),("LU","Lu")])
def test_ion_type(kind,element):
    line=f"HETATM{1:5d}  NA  ION A   1    {1.:8.3f}{2.:8.3f}{3.:8.3f}  1.00  0.00     0.000 {kind}"
    atoms=pose.atom_records(line,pdbqt=True)
    assert atoms[0]["element"]==element
