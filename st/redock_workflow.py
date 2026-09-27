"""Redocking operations shared by CLI and GUI adapters; no UI dependencies.

Deploy this file with each standalone app. Copies are checked for parity in tests.
The chemistry, Vina and RMSD algorithms remain in the existing core module.
"""
from pathlib import Path
import hashlib
import math


def prepare_reference_ligand(core, *, smiles, name, ph, wdir, mode="pkanet", **options):
    return core.prepare_ligand(smiles=smiles, name=name, ph=ph, wdir=wdir, mode=mode, **options)


def run_reference_vina(core, **options):
    return core.run_vina(**options)


def calculate_pose_rmsds(core, scores, pose_mols, crystal_pdb):
    """Preserve CLI atom mapping and two-decimal reporting, including missing RMSDs."""
    rmsds = {}
    if crystal_pdb and Path(crystal_pdb).is_file():
        for score in scores:
            idx = (score['pose'] - 1) if score['pose'] else 0
            if idx < len(pose_mols):
                try:
                    value = core.calc_rmsd_heavy(pose_mols[idx], crystal_pdb)
                    rmsds[score['pose']] = round(value, 2) if value is not None else None
                except Exception:
                    rmsds[score['pose']] = None
    return rmsds


def rank_redock_scores(scores, rmsds):
    """Sort by reported RMSD as CLI tables do; retain original Vina pose numbers."""
    rows = [dict(s, rmsd=rmsds.get(s['pose'])) for s in scores]
    return sorted(rows, key=lambda r: (r['rmsd'] if r['rmsd'] is not None and math.isfinite(r['rmsd']) else math.inf, r['pose']))


def inspect_redock_structure(core, raw_path, wdir, cutoff=4.5):
    """Eligibility is based on supplied coordinates, before any chain deduplication."""
    import numpy as np
    from prody import parsePDB
    from scipy.spatial import cKDTree
    wdir = Path(wdir); wdir.mkdir(parents=True, exist_ok=True)
    raw_path = Path(raw_path)
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    pdb = raw_path
    if core.is_cif_file(str(raw_path)):
        pdb = wdir / 'inspection.pdb'
        converted = core.convert_cif_to_pdb(str(raw_path), str(pdb))
        if not converted.get('success'):
            raise ValueError(f"Structure conversion failed: {converted.get('error')}")
    atoms = parsePDB(str(pdb), model=1)
    if atoms is None:
        raise ValueError('Structure could not be parsed.')
    protein = atoms.select('protein and noh')
    if protein is None:
        raise ValueError('No protein heavy atoms found in the structure.')
    # Same ligand classification as CLI; retain residue instances, not just names.
    rows = core._collect_hetatm_residues(atoms)
    candidates = [r for r in rows if r['type_guess'] == 'ligand']
    chains = sorted(set(str(c).strip() for c in protein.getChids()))
    result = dict(ready=False, raw_path=str(raw_path), source_sha256=digest,
                  pdb_path=str(pdb), chains=chains, cutoff=cutoff, contacts=[],
                  candidate_count=len(candidates), structure_state='holo' if candidates else 'no suitable ligand')
    if not candidates:
        result['message'] = 'Redock is not ready: no bound ligand was detected in this structure.'
        return result
    if len(candidates) != 1:
        result['message'] = 'Redock is not ready: multiple relevant bound ligands were detected.'
        return result
    chosen = candidates[0]
    # The historical key omits insertion codes. Do not silently combine distinct residues.
    ligmask = ((atoms.getResnames() == chosen['resname']) &
               (np.char.strip(atoms.getChids().astype(str)) == chosen['chain']) &
               (atoms.getResnums() == chosen['resid']) & atoms.getFlags('hetatm'))
    ligand = atoms[np.flatnonzero(ligmask)]
    if len(set(ligand.getResindices())) != 1:
        raise ValueError('Ambiguous ligand residue identifiers/insertion codes; cannot select one reference safely.')
    ligand_heavy = ligand.select('noh')
    if ligand_heavy is None:
        raise ValueError('Bound ligand has no heavy atoms.')
    result['ligand'] = {k: chosen[k] for k in ('key', 'resname', 'chain', 'resid', 'type_guess')}
    lig_tree = cKDTree(ligand_heavy.getCoords())
    for chain in chains:
        mask = np.char.strip(protein.getChids().astype(str)) == chain
        part = protein[np.flatnonzero(mask)]
        distances, _ = lig_tree.query(part.getCoords())
        near = distances <= cutoff
        result['contacts'].append(dict(chain=chain, min_distance=float(distances.min()),
                                       contacting_atoms=int(near.sum()),
                                       contacting_residues=len(set(part.getResindices()[near]))))
    bound = [r['chain'] for r in result['contacts'] if r['contacting_atoms']]
    if len(bound) > 1:
        result['message'] = 'Redock is not ready: the bound ligand interacts with multiple protein chains.'
        return result
    if not bound:
        result['message'] = 'Redock is not ready: no protein chain contacts the ligand within 4.5 Å.'
        return result
    chain = bound[0]
    # Keep the selected protein chain plus the exact ligand, irrespective of ligand chain ID.
    prot_all = atoms.select('protein')
    keep = set(prot_all.getIndices()[np.char.strip(prot_all.getChids().astype(str)) == chain])
    keep.update(ligand.getIndices())
    selected_protein = protein[np.flatnonzero(np.char.strip(protein.getChids().astype(str)) == chain)]
    selected_tree = cKDTree(selected_protein.getCoords())
    # Retain complete associated metal/cofactor residues; never keep unrelated chains' compounds.
    for row in rows:
        if row['type_guess'] not in ('metal', 'heme/cofactor', 'cofactor'):
            continue
        residue = row['atoms']
        heavy = residue.select('noh')
        if heavy is not None and selected_tree.query(heavy.getCoords())[0].min() <= cutoff:
            keep.update(residue.getIndices())
    # Preserve original lines/coordinates/atom names and applicable connectivity annotations.
    kept_atoms = atoms[sorted(keep)]
    serials = set(int(s) for s in kept_atoms.getSerials())
    residue_ids = {(str(c).strip(), int(r), str(i).strip()) for c, r, i in zip(kept_atoms.getChids(), kept_atoms.getResnums(), kept_atoms.getIcodes())}
    scoped = wdir / 'ligand_bound_chain.pdb'
    original = pdb.read_text().splitlines(True)
    lines = []; in_first = True; saw_model = False
    for line in original:
        tag = line[:6].strip()
        if tag == 'MODEL':
            if saw_model: in_first = False
            saw_model = True
        elif tag == 'ENDMDL':
            in_first = False
        elif tag in ('ATOM', 'HETATM') and in_first:
            if int(line[6:11]) in serials:
                lines.append(line)
        elif tag in ('LINK', 'SSBOND'):
            # Preserve annotations only when both residue endpoints survive filtering.
            try:
                if tag == 'LINK':
                    endpoints = [(line[21].strip(), int(line[22:26]), line[26].strip()),
                                 (line[51].strip(), int(line[52:56]), line[56].strip())]
                else:
                    endpoints = [(line[15].strip(), int(line[17:21]), line[21].strip()),
                                 (line[29].strip(), int(line[31:35]), line[35].strip())]
                if all(endpoint in residue_ids for endpoint in endpoints): lines.append(line)
            except (ValueError, IndexError):
                raise ValueError('Malformed connectivity annotation; cannot safely filter the receptor.')
        elif tag == 'CONECT':
            ids = [int(line[i:i+5]) for i in range(6, len(line.rstrip()), 5) if line[i:i+5].strip()]
            if ids and ids[0] in serials:
                ids = [i for i in ids if i in serials]
                if len(ids) > 1: lines.append('CONECT' + ''.join(f'{i:5d}' for i in ids) + '\n')
    scoped.write_text(''.join(lines) + 'END\n')
    reference = wdir / 'crystal_reference.pdb'
    ligand_serials = set(int(s) for s in ligand.getSerials())
    reference.write_text(''.join(l for l in lines if l.startswith(('ATOM  ', 'HETATM')) and int(l[6:11]) in ligand_serials) + 'END\n')
    policy = {r['key']: ('reference' if r['key'] == chosen['key'] else
                        'keep' if r['type_guess'] in ('metal', 'heme/cofactor', 'cofactor') else 'remove') for r in rows}
    result.update(ready=True, message='Ready: one bound ligand contacts one protein chain.',
                  selected_chain=chain, scoped_path=str(scoped), reference_path=str(reference),
                  reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(), hetatm_policy=policy)
    return result


def execute_prepared_redock(core, inspection, receptor, wdir, vina_path, *, ph=7.4,
                            mode='pkanet', exhaustiveness=16, n_modes=10, energy_range=3,
                            seed=None, conformer_seed=None):
    """GUI orchestration using the same operations called by cmd_redock."""
    import csv
    wdir = Path(wdir); wdir.mkdir(parents=True, exist_ok=True)
    ref = inspection['reference_path']
    if hashlib.sha256(Path(ref).read_bytes()).hexdigest() != inspection['reference_sha256']:
        raise ValueError('Crystal reference changed; prepare receptor again.')
    ligand = inspection['ligand']; name = ligand['resname']
    cocrystal_id = receptor.get('cocrystal_ligand_id', '')
    smiles, source, warning = core.get_cocrystal_smiles(ref, cocrystal_id, inspection['raw_path'])
    if not smiles:
        raise ValueError(f'Could not obtain reference ligand SMILES: {warning}')
    prep = prepare_reference_ligand(core, smiles=smiles, name=name, ph=ph, wdir=wdir,
                                    mode=mode, conformer_seed=conformer_seed)
    if not prep.get('success'): raise ValueError(prep.get('error', 'Ligand preparation failed.'))
    dock = run_reference_vina(core, receptor_pdbqt=str(Path(receptor['rec_pdbqt']).resolve()),
                             ligand_pdbqt=str(Path(prep['pdbqt']).resolve()),
                             config_txt=str(Path(receptor['config_txt']).resolve()), vina_path=vina_path,
                             exhaustiveness=exhaustiveness, n_modes=n_modes, energy_range=energy_range,
                             wdir=wdir, out_name=name, seed=seed)
    if not dock.get('success'): raise ValueError(dock.get('error', 'Vina failed.'))
    fixed = wdir / f'{name}_pv_ready.sdf'
    if prep.get('prot_smiles'): core.fix_sdf_bond_orders(dock['out_sdf'], prep['prot_smiles'], str(fixed))
    sdf = str(fixed) if fixed.is_file() and fixed.stat().st_size >= 10 else dock['out_sdf']
    mols = core.load_mols_from_sdf(sdf, sanitize=False)
    # Fail rather than associating scores with shifted indices after a bad SDF record.
    from rdkit import Chem
    raw_mols = list(Chem.SDMolSupplier(sdf, removeHs=False, sanitize=False))
    if any(m is None for m in raw_mols) or len(raw_mols) < max(s['pose'] for s in dock['scores']):
        raise ValueError('Docked SDF pose records are incomplete; score-to-pose mapping is unsafe.')
    if hashlib.sha256(Path(ref).read_bytes()).hexdigest() != inspection['reference_sha256']:
        raise ValueError('Crystal reference changed during docking; results cannot be validated.')
    rmsds = calculate_pose_rmsds(core, dock['scores'], mols, ref)
    ranked = rank_redock_scores(dock['scores'], rmsds)
    csv_path = wdir / 'redock_scores.csv'
    with csv_path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['pose', 'affinity', 'rmsd'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(ranked)
    return dict(dock=dock, prep=prep, sdf=sdf, scores=ranked, csv=str(csv_path),
                smiles_source=source, warning=warning, reference=ref)
