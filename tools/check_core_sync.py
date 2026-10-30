"""Compare classification constants without importing scientific dependencies."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PATHS=[ROOT/'core.py',ROOT/'st/core.py',ROOT/'anyonecandock/anyonecandock/core.py']
NAMES={'METAL_RESNAMES','BUFFER_RESNAMES','LIPID_ADDITIVE_RESNAMES'}
def constants(path):
    return {n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in NAMES}
def check():
    data=[constants(p) for p in PATHS]
    assert all(d==data[0] for d in data), 'HETATM classification constants differ'
    return data[0]
if __name__=='__main__':print('Classification sets synchronized:',{k:len(v) for k,v in check().items()})
