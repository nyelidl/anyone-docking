import importlib.util
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'st'))
spec=importlib.util.spec_from_file_location('backend_reference_api',ROOT/'st/api.py')
api=importlib.util.module_from_spec(spec);sys.modules[spec.name]=api;spec.loader.exec_module(api)

def test_reference_warning_in_compact_job_result():
    meta={'receptor':{'cocrystal_ligand_id':'OLC_A_2','reference_warning':'Crystallization additive reference'},'results':[]}
    result=api._ultra_compact_from_meta('test-job',meta)
    assert result['receptor']['reference_warning']=='Crystallization additive reference'
    assert result['receptor']['cocrystal_ligand_id']=='OLC_A_2'
