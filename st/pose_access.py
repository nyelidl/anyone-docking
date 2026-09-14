"""Expose saved docking coordinates. No embedding, minimization, or redocking."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import quote, urlencode

from fastapi import HTTPException


COMPACT_FIELDS = (
    'selected_pose_pdb_url', 'selected_pose_sdf_url', 'complex_pdb_url',
    'viewer_3d_url', 'render_3d_png_url', 'interactions',
    'interaction_analysis_source', 'interaction_analysis_warning', 'structure_warning',
    'receptor_sha256', 'pose_sha256', 'complex_sha256',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_write(path, text):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.pose_')
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def saved_file(wdir, value):
    """Resolve only actual files belonging to this job, including old relative paths."""
    if not value:
        raise HTTPException(404, 'Saved docking coordinate file is unavailable')
    path = Path(value)
    if not path.is_absolute():
        path = wdir / path
    if not path.resolve().is_relative_to(wdir.resolve()) or not path.is_file():
        raise HTTPException(404, 'Saved docking coordinate file is unavailable in this job')
    return path


def pdbqt_models(path):
    models, current, lines = {}, None, []
    for line in Path(path).read_text().splitlines(keepends=True):
        if line.startswith('MODEL '):
            if current is not None:
                raise HTTPException(409, 'Nested Vina MODEL records')
            try:
                current = int(line.split()[1])
            except (ValueError, IndexError):
                raise HTTPException(409, 'Invalid Vina pose rank')
            if current < 1 or current in models:
                raise HTTPException(409, 'Duplicate or invalid Vina pose rank')
            lines = [line]
        elif line.startswith('ENDMDL'):
            if current is None:
                raise HTTPException(409, 'Unmatched Vina ENDMDL')
            lines.append(line)
            text = ''.join(lines)
            scores = re.findall(r'^REMARK VINA RESULT:\s+(\S+)', text, re.M)
            if len(scores) != 1:
                raise HTTPException(409, 'Pose must have exactly one Vina score')
            try:
                score = float(scores[0])
                if not math.isfinite(score):
                    raise ValueError()
            except ValueError:
                raise HTTPException(409, 'Invalid Vina affinity')
            models[current] = {'text': text, 'affinity': score}
            current, lines = None, []
        elif current is not None:
            lines.append(line)
        elif line.startswith(('ATOM  ', 'HETATM')):
            raise HTTPException(409, 'Vina output atoms have no MODEL rank')
    if current is not None or not models:
        raise HTTPException(409, 'No complete ranked Vina poses available')
    return models


def atom_records(text, pdbqt=False):
    types = {'A':'C', 'C':'C', 'N':'N', 'NA':'N', 'NS':'N', 'O':'O', 'OA':'O',
             'OS':'O', 'S':'S', 'SA':'S', 'H':'H', 'HD':'H', 'HS':'H', 'P':'P',
             'F':'F', 'Cl':'Cl', 'CL':'Cl', 'Br':'Br', 'BR':'Br', 'I':'I',
             'Mg':'Mg', 'MG':'Mg', 'Mn':'Mn', 'MN':'Mn', 'Zn':'Zn', 'ZN':'Zn',
             'Ca':'Ca', 'CA':'Ca', 'Fe':'Fe', 'FE':'Fe', 'Cu':'Cu', 'CU':'Cu',
             'Si':'Si', 'B':'B', 'Se':'Se'}
    atoms = []
    for line in text.splitlines():
        if not line.startswith(('ATOM  ', 'HETATM')):
            continue
        try:
            element = line[76:78].strip()
            if pdbqt:
                kind = line[77:].strip()
                if re.fullmatch(r'G\d+', kind):
                    continue  # Meeko macrocycle glue is not a physical atom.
                element = 'C' if re.fullmatch(r'CG\d+', kind) else types.get(kind)
                if not element:
                    raise ValueError('Unsupported AutoDock element type '+kind)
            if not element:
                raise ValueError('Missing explicit atom element')
            xyz = tuple(float(line[i:i+8]) for i in (30,38,46))
            if not all(map(math.isfinite, xyz)):
                raise ValueError('Non-finite coordinates')
            atoms.append(dict(line=line, serial=int(line[6:11]), name=line[12:16].strip(),
                              element=element.title(), xyz=xyz, resname=line[17:20].strip(),
                              chain=line[21:22], resnum=int(line[22:26]), icode=line[26:27]))
        except (ValueError, IndexError) as exc:
            raise HTTPException(409, 'Cannot validate saved coordinates: '+str(exc)) from exc
    if not atoms:
        raise HTTPException(409, 'No physical atoms in saved coordinates')
    return atoms


def to_pdb(atoms, ligand=False):
    lines = []
    for atom in atoms:
        line = atom['line'].ljust(80)
        # Copy coordinates/identity columns verbatim; discard PDBQT charge/type columns.
        lines.append(('HETATM' if ligand else line[:6]) + line[6:66] + ' '*10 +
                     atom['element'].rjust(2) + '  \n')
    return ''.join(lines)


def complex_pdb(receptor, ligand):
    if len(receptor)+len(ligand) > 99999:
        raise HTTPException(409, 'Complex exceeds standard PDB serial field capacity')
    used = {(a['chain'],a['resnum'],a['icode']) for a in receptor}
    occupied = {a['chain'] for a in receptor+ligand}
    chain_map = {}
    for a in ligand:
        if (a['chain'],a['resnum'],a['icode']) in used and a['chain'] not in chain_map:
            free = next((c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789' if c not in occupied), None)
            if free is None:
                raise HTTPException(409, 'Cannot assign an unambiguous complex ligand chain')
            chain_map[a['chain']] = free
            occupied.add(free)
    out = []
    for serial, a in enumerate(receptor+ligand,1):
        is_lig = serial > len(receptor)
        line = to_pdb([a], ligand=is_lig).rstrip('\n')
        line = line[:6]+f'{serial:5d}'+line[11:]
        if is_lig and a['chain'] in chain_map:
            line = line[:21]+chain_map[a['chain']]+line[22:]
        if serial == len(receptor)+1:
            out.append('TER\n')
        out.append(line+'\n')
    return ''.join(out)+'END\n', chain_map


def checked_scores(row, models):
    scores = {}
    for entry in row.get('scores', []):
        if not isinstance(entry,dict):
            continue
        rank = entry.get('pose')
        if rank in scores:
            raise HTTPException(409, 'Duplicate rank in docking score table')
        scores[rank] = entry
        try:
            affinity = float(entry['affinity'])
        except (ValueError, TypeError, KeyError):
            raise HTTPException(409, 'Invalid saved docking score')
        if not math.isfinite(affinity) or rank not in models or abs(affinity-models[rank]['affinity']) > .0011:
            raise HTTPException(409, 'Docking score table does not match saved Vina MODEL scores')
    return scores


def select_rank(row, models):
    scores = checked_scores(row,models)
    top = min(models,key=lambda rank:(models[rank]['affinity'],rank))
    redocking = row.get('is_redocking',False) or row.get('pose_selection_method','') in (
        'lowest_rmsd_vs_cocrystal_ligand','top_score_fallback')
    if redocking:
        # The existing selector stores the unrounded chosen rank and RMSD.
        rank = row.get('selected_pose_rank')
        if row.get('pose_selection_method')=='lowest_rmsd_vs_cocrystal_ligand' and rank in models:
            rmsd = row.get('selected_pose_rmsd')
            if isinstance(rmsd,(int,float)) and math.isfinite(rmsd) and rmsd >= 0:
                return rank,'lowest_rmsd_vs_cocrystal_ligand',''
        candidates = [(e['rmsd_vs_crystal'],rank) for rank,e in scores.items()
                      if isinstance(e.get('rmsd_vs_crystal'),(int,float)) and math.isfinite(e['rmsd_vs_crystal']) and e['rmsd_vs_crystal']>=0]
        if candidates:
            return min(candidates)[1],'lowest_rmsd_vs_cocrystal_ligand',''
        return top,'top_score_fallback',row.get('pose_selection_warning') or 'Redocking RMSD unavailable; selected the lowest Vina affinity.'
    return top,'top_score',''


def proximity(receptor, ligand, cutoff):
    """Nearest ligand-heavy-atom distance per receptor heavy atom; not bond typing."""
    import numpy as np
    heavy = [a for a in ligand if a['element'] not in ('H','D','T')]
    if not heavy:
        return [],[]
    coords = np.array([a['xyz'] for a in heavy])
    contacts,residues = [],{}
    for atom in receptor:
        if atom['element'] in ('H','D','T'):
            continue
        distances = np.linalg.norm(coords-np.array(atom['xyz']),axis=1)
        idx = int(distances.argmin()); distance = float(distances[idx])
        if distance > cutoff:
            continue
        other = heavy[idx]
        key = (atom['chain'],atom['resnum'],atom['icode'],atom['resname'])
        residues[key] = dict(chain=key[0].strip(),resnum=key[1],icode=key[2].strip(),resname=key[3])
        contacts.append(dict(type='proximity_contact',protein_residue=f"{atom['resname']}{atom['resnum']}{atom['icode'].strip()}",
            protein_chain=atom['chain'].strip(),protein_atom=atom['name'],protein_serial=atom['serial'],
            ligand_atom=other['name'],ligand_serial=other['serial'],distance_angstrom=round(distance,3),
            protein_xyz=atom['xyz'],ligand_xyz=other['xyz']))
    contacts.sort(key=lambda c:c['distance_angstrom'])
    return contacts, list(residues.values())


def validate_sdf_coordinates(mol, ligand):
    expected = [a for a in ligand if a['element'] not in ('H','D','T')]
    actual = [(a.GetSymbol(), tuple(mol.GetConformer().GetAtomPosition(a.GetIdx())))
              for a in mol.GetAtoms() if a.GetAtomicNum()!=1]
    if len(actual)!=len(expected):
        raise ValueError('SDF/PDBQT heavy-atom counts differ')
    used = set()
    for element, xyz in actual:
        matches = [i for i,a in enumerate(expected) if a['element']==element and math.dist(a['xyz'],xyz)<=.0011]
        if len(matches)!=1 or matches[0] in used:
            raise ValueError('SDF coordinates do not uniquely match this Vina pose')
        used.add(matches[0])


def export_sdf(wdir, row, model_order, rank, ligand, target):
    """Extract an existing SDF record only after verifying exact docked heavy coordinates."""
    from rdkit import Chem
    warnings = []
    for key in ('pv_sdf','out_sdf'):
        try:
            path = saved_file(wdir,row.get(key))
            blocks = path.read_text().split('$$$$')
            index = model_order.index(rank)
            if index >= len(blocks) or not blocks[index].strip():
                raise ValueError('SDF record for this pose is missing')
            block = blocks[index]
            if index:
                block = block.removeprefix('\r\n').removeprefix('\n')
            mol = Chem.MolFromMolBlock(block,sanitize=False,removeHs=False,strictParsing=True)
            if mol is None or mol.GetNumConformers()!=1:
                raise ValueError('SDF record is unreadable')
            validate_sdf_coordinates(mol, ligand)
            atomic_write(target,block.rstrip('\r\n')+'\n$$$$\n')
            return ''
        except (HTTPException,ValueError,OSError) as exc:
            warnings.append(getattr(exc,'detail',str(exc)))
    Path(target).unlink(missing_ok=True)
    return 'No validated SDF for this pose; use exact PDB/PDBQT. '+ '; '.join(warnings)


def structure(wdir, meta, row, rank, public_url, cutoff=4.5):
    source = saved_file(wdir,row.get('out_pdbqt'))
    models = pdbqt_models(source)
    checked_scores(row,models)
    if rank not in models:
        raise HTTPException(404, 'Requested Vina pose rank does not exist')
    selected,method,fallback = select_rank(row,models)
    receptor_meta = meta.get('receptor',{})
    receptor_source = saved_file(wdir,receptor_meta.get('rec_pdbqt') or receptor_meta.get('rec_fh'))
    rec_atoms = atom_records(receptor_source.read_text(),pdbqt=receptor_source.suffix.lower()=='.pdbqt')
    lig_atoms = atom_records(models[rank]['text'],pdbqt=True)
    name = row.get('name','ligand')
    # Generated stems have a namespace separate from docking/input filenames.
    stem = f"structure_{hashlib.sha256(name.encode()).hexdigest()[:12]}_pose_{rank}"
    files = {'receptor':wdir/'structure_receptor_prepared.pdb',
             'pdb':wdir/(stem+'.pdb'),'pdbqt':wdir/(stem+'.pdbqt'),
             'sdf':wdir/(stem+'.sdf'),'complex':wdir/(stem+'_complex.pdb')}
    complex_text,chain_map = complex_pdb(rec_atoms,lig_atoms)
    atomic_write(files['receptor'],to_pdb(rec_atoms)+'END\n')
    atomic_write(files['pdb'],to_pdb(lig_atoms,ligand=True)+'END\n')
    atomic_write(files['pdbqt'],models[rank]['text'])
    atomic_write(files['complex'],complex_text)
    sdf_warning = export_sdf(wdir,row,list(models),rank,lig_atoms,files['sdf'])
    contacts,residues = proximity(rec_atoms,lig_atoms,cutoff)
    jid = meta['job_id']
    def url(kind):
        return public_url(f'/jobs/{jid}/files/{files[kind].name}')
    query = urlencode({'ligand_name':name,'pocket_distance':cutoff})
    result = dict(success=True,job_id=jid,ligand_name=name,pose_rank=rank,selected_pose_rank=selected,
        pose_selection_method=method,pose_selection_warning=fallback,affinity=models[rank]['affinity'],
        affinity_units='kcal/mol',affinity_interpretation='Vina computational score, not experimental binding affinity',
        receptor_pdb_id=receptor_meta.get('pdb_id'),prepared_smiles=row.get('prepared_smiles',''),
        charge=row.get('charge'),protonation_mode=row.get('protonation_mode_used',''),
        grid_center=receptor_meta.get('center'),grid_size=receptor_meta.get('size'),
        receptor_pdb_url=url('receptor'),pose_pdb_url=url('pdb'),pose_pdbqt_url=url('pdbqt'),
        pose_sdf_url='' if sdf_warning else url('sdf'),complex_pdb_url=url('complex'),
        receptor_url=url('receptor'),ligand_url=url('pdb'),ligand_pdb_url=url('pdb'),
        ligand_sdf_url='' if sdf_warning else url('sdf'),
        viewer_url=public_url(f'/jobs/{jid}/view3d/{rank}?{query}'),render_3d_png_url='',
        receptor_sha256=digest(files['receptor']),pose_sha256=digest(files['pdbqt']),complex_sha256=digest(files['complex']),
        receptor_source_sha256=digest(receptor_source),vina_output_sha256=digest(source),
        pose_pdb_sha256=digest(files['pdb']),pose_sdf_sha256='' if sdf_warning else digest(files['sdf']),
        hash_definitions={'receptor_sha256':'exported receptor PDB','pose_sha256':'extracted ranked Vina PDBQT',
                          'complex_sha256':'exported receptor plus pose PDB'},
        complex_ligand_chain_map=chain_map,coordinates_source='saved Vina MODEL and prepared receptor; no conformer generation',
        interactions=[],interaction_analysis_source='not_configured',
        interaction_analysis_warning='No chemical interaction classifier is configured. Proximity contacts are distances only, not hydrogen bonds or other classified interactions.',
        proximity_analysis_source='ACD Euclidean nearest heavy-atom distance per receptor atom',
        proximity_contacts=contacts[:200],proximity_contacts_total=len(contacts),proximity_contacts_truncated=len(contacts)>200,
        pocket_residues=residues[:200],pocket_residues_total=len(residues),pocket_residues_truncated=len(residues)>200,
        pocket_distance_angstrom=cutoff,structure_warning=sdf_warning)
    return result


def compact_fields(data):
    return dict(selected_pose_rank=data['selected_pose_rank'],pose_selection_method=data['pose_selection_method'],
                pose_selection_warning=data['pose_selection_warning'],selected_pose_pdb_url=data['pose_pdb_url'],
                selected_pose_sdf_url=data['pose_sdf_url'],complex_pdb_url=data['complex_pdb_url'],
                viewer_3d_url=data['viewer_url'],render_3d_png_url='',interactions=data['interactions'],
                interaction_analysis_source=data['interaction_analysis_source'],
                interaction_analysis_warning=data['interaction_analysis_warning'],structure_warning=data['structure_warning'],
                receptor_sha256=data['receptor_sha256'],pose_sha256=data['pose_sha256'],complex_sha256=data['complex_sha256'])
