import importlib.util
from pathlib import Path
import pytest
from anyonecandock import core
ROOT=Path(__file__).resolve().parents[1]
@pytest.mark.parametrize('resname,expected',[('OLC','lipid/additive'),('OLA','lipid/additive'),('TAR','buffer/ion'),('CFF','ligand'),('AQ4','ligand'),('NA','metal'),('PLM','ligand'),('MYR','ligand'),('STE','ligand'),('MAL','ligand'),('BEZ','ligand'),('SIN','ligand')])
def test_classification(resname,expected):
    assert core._guess_hetatm_type(resname,20)==expected

def test_core_constants_synchronized():
    spec=importlib.util.spec_from_file_location('core_sync',ROOT/'tools/check_core_sync.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    constants=module.check()
    assert not {'PLM','MYR','STE','MAL','BEZ','SIN'} & constants['LIPID_ADDITIVE_RESNAMES']

def test_all_retained_metals_have_pose_mapping():
    spec=importlib.util.spec_from_file_location('ion_pose',ROOT/'st/pose_access.py')
    pose=importlib.util.module_from_spec(spec);spec.loader.exec_module(pose)
    for name in core.METAL_RESNAMES:
        symbol=name.title() if name!='NA' else 'Na'
        line=f'HETATM{1:5d}  NA  ION A   1    {1.:8.3f}{2.:8.3f}{3.:8.3f}  1.00  0.00     0.000 {symbol}'
        assert pose.atom_records(line,pdbqt=True)[0]['element']==symbol

@pytest.mark.network
@pytest.mark.parametrize('pdb_id,reference',[('5MZP','CFF_A_2401'),('1M17','AQ4_A_999')])
def test_real_reference_selection(tmp_path,pdb_id,reference):
    from urllib.request import urlopen
    source=tmp_path/f'{pdb_id}.pdb'
    source.write_bytes(urlopen(f'https://files.rcsb.org/download/{pdb_id}.pdb',timeout=60).read())
    result=core.prepare_receptor(str(source),tmp_path)
    assert result['success'],result
    assert result['cocrystal_ligand_id']==reference
    assert not result['reference_warning']

@pytest.mark.parametrize('additive',[True,False])
def test_auto_without_genuine_ligand_warns(tmp_path,monkeypatch,additive):
    def line(serial,name,res,resid,x,het=False):
        return f"{'HETATM' if het else 'ATOM  '}{serial:5d} {name:>4} {res:>3} A{resid:4d}    {x:8.3f}{1.:8.3f}{2.:8.3f}  1.00 20.00           C\n"
    text=''.join(line(i,n,'ALA',1,float(i)) for i,n in enumerate(['N','CA','C','O'],1))
    if additive:text+=''.join(line(i,'C'+str(i),'OLC',2,float(i),True) for i in range(5,10))
    source=tmp_path/'input.pdb';source.write_text(text+'END\n')
    def convert(raw,wdir,*args):
        return {'success':True,'rec_fh':raw,'rec_pdbqt':raw,'log':[]}
    monkeypatch.setattr(core,'strip_and_convert_receptor',convert)
    result=core.prepare_receptor(str(source),tmp_path)
    assert result['success'],result
    assert result['reference_warning']
    if additive:assert result['cocrystal_ligand_id']=='OLC_A_2'
    else:assert (result['cx'],result['cy'],result['cz'])!=(0,0,0)
