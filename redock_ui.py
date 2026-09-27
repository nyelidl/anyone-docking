"""Streamlit adapter for the dedicated Redock tab."""
import hashlib
import json
from pathlib import Path

from redock_workflow import inspect_redock_structure, execute_prepared_redock


def receptor_setup(st, core, wdir, pfx, src, pdb_id, upload_file):
    """Called inside the existing receptor component after its shared source widgets."""
    wdir = Path(wdir); wdir.mkdir(parents=True, exist_ok=True)
    inspection = None
    try:
        if src == 'Download from RCSB':
            token = (pdb_id or '').strip().upper()
            if len(token) != 4 or not token.isalnum():
                raise ValueError('Enter a valid four-character PDB ID.')
            path = wdir / f'raw_{token}.cif'
            if not path.is_file() or path.stat().st_size < 200:
                rc, _ = core.run_cmd(['curl', '-sf', '--max-time', '60', f'https://files.rcsb.org/download/{token}.cif', '-o', str(path)])
                if rc != 0 or not path.is_file() or path.stat().st_size < 200:
                    path = wdir / f'raw_{token}.pdb'
                    rc, _ = core.run_cmd(['curl', '-sf', '--max-time', '60', f'https://files.rcsb.org/download/{token}.pdb', '-o', str(path)])
                    if rc != 0 or not path.is_file() or path.stat().st_size < 200:
                        raise ValueError('Could not download this structure in CIF or PDB format.')
        else:
            if upload_file is None:
                raise ValueError('Upload a PDB or mmCIF structure to inspect it.')
            extension = '.cif' if Path(upload_file.name).suffix.lower() in ('.cif', '.mmcif') else '.pdb'
            path = wdir / ('uploaded' + extension)
            path.write_bytes(upload_file.getvalue())
        signature = hashlib.sha256(path.read_bytes()).hexdigest()
        previous = st.session_state.get(pfx + 'inspection')
        if previous and previous['source_sha256'] == signature and previous['raw_path'] == str(path):
            inspection = previous
        else:
            st.session_state.pop(pfx + 'prepared', None)
            st.session_state.pop(pfx + 'result', None)
            with st.spinner('Inspecting ligand and protein-chain contacts…'):
                inspection = inspect_redock_structure(core, path, wdir / signature[:16])
            st.session_state[pfx + 'inspection'] = inspection
    except Exception as exc:
        st.session_state.pop(pfx + 'inspection', None)
        st.session_state.pop(pfx + 'prepared', None)
        st.session_state.pop(pfx + 'result', None)
        st.info(str(exc))

    with st.expander('⚗️ Receptor setup panel', expanded=True):
        st.caption('One ligand instance only. Heavy-atom contacts ≤4.5 Å determine the receptor chain. Only supplied chains/model 1 are inspected; no symmetry mates or biological assemblies are generated.')
        if inspection:
            st.caption('Structure inspection: ' + inspection['structure_state'])
            st.write('Protein chains: ' + ', '.join(c or '(blank)' for c in inspection['chains']))
            if inspection.get('ligand'):
                ligand = inspection['ligand']
                st.write(f"Detected ligand: **{ligand['resname']}**")
                st.write(f"Ligand residue: **{ligand['resname']} {ligand['chain'] or '(blank)'} {ligand['resid']}**")
            if inspection.get('contacts'):
                st.dataframe(inspection['contacts'], hide_index=True)
            if inspection['ready']:
                st.success(inspection['message'])
                st.write(f"Selected receptor chain: **{inspection['selected_chain'] or '(blank)'}**")
            else:
                st.warning(inspection['message'])
        box = tuple(st.slider(f'{axis} size (Å)', 10, 40, 18, 1, key=pfx + axis) for axis in ('X', 'Y', 'Z'))
        st.caption('The reference ligand centers the search box, matching the CLI auto redock mode (default 18 × 18 × 18 Å).')
        settings = (inspection or {}).get('source_sha256'), box
        if st.session_state.get(pfx + 'prepared_settings') != settings:
            st.session_state.pop(pfx + 'prepared', None)
            st.session_state.pop(pfx + 'result', None)
        ready = bool(inspection and inspection['ready'])
        if st.button('▶ Prepare Receptor', key=pfx + 'btn_receptor', type='primary', disabled=not ready):
            st.session_state.pop(pfx + 'prepared', None)
            st.session_state.pop(pfx + 'result', None)
            try:
                out = Path(inspection['scoped_path']).parent / 'prepared'
                out.mkdir(exist_ok=True)
                with st.spinner('Preparing the ligand-bound receptor chain…'):
                    result = core.prepare_receptor(raw_pdb=inspection['scoped_path'], wdir=out,
                                                   center_mode='auto', box_size=box,
                                                   hetatm_policy=inspection['hetatm_policy'],
                                                   reference_hetatm_key=inspection['ligand']['key'])
                if not result.get('success'):
                    raise ValueError(result.get('error', 'Receptor preparation failed.'))
                if hashlib.sha256(Path(inspection['reference_path']).read_bytes()).hexdigest() != inspection['reference_sha256']:
                    raise ValueError('Crystal reference changed unexpectedly during preparation.')
                st.session_state[pfx + 'prepared'] = result
                st.session_state[pfx + 'prepared_settings'] = settings
            except Exception as exc:
                st.error(str(exc))
        prepared = st.session_state.get(pfx + 'prepared')
        if prepared:
            st.success('Receptor prepared. The original crystal reference is preserved separately.')
            with st.expander('Receptor preparation log', expanded=False):
                st.code('\n'.join(prepared.get('log', [])))
    return inspection


def render_results(st, core, result, prepared, show_pose, pfx='r_'):
    from rdkit import Chem
    import pandas as pd
    st.markdown('### 🔎 Pose Browser')
    rows = result['scores']
    table = pd.DataFrame([{'Pose': r['pose'], 'Binding affinity (kcal/mol)': r.get('affinity'),
                           'RMSD vs crystal (Å)': r['rmsd']} for r in rows])
    st.dataframe(table, hide_index=True)
    st.caption('Ordered by lowest crystal RMSD, with original Vina pose numbers retained. RMSD uses the CLI common-substructure atom mapping without coordinate alignment; missing values are placed last.')
    if all(r['rmsd'] is None for r in rows):
        st.warning('No valid RMSD could be calculated. No best redocking pose can be identified.')
    rank = st.selectbox('Pose (RMSD order)', [r['pose'] for r in rows], key=pfx + 'pose',
                        format_func=lambda p: f"Pose {p}")
    row = next(r for r in rows if r['pose'] == rank)
    st.write(f"Pose **{rank}** · Affinity **{row.get('affinity')} kcal/mol** · RMSD **{row['rmsd'] if row['rmsd'] is not None else 'unavailable'} Å**")
    mols = core.load_mols_from_sdf(result['sdf'], sanitize=False)
    mol = mols[rank - 1]
    show_pose(mol, prepared['rec_fh'], result['reference'])
    st.download_button('⬇ Selected pose SDF', Chem.MolToMolBlock(mol) + '\n$$$$\n',
                       file_name=f'redock_pose{rank}.sdf', key=pfx + 'sdf_download')
    st.download_button('⬇ Selected pose PDB', Chem.MolToPDBBlock(mol),
                       file_name=f'redock_pose{rank}.pdb', key=pfx + 'pdb_download')
    st.download_button('⬇ RMSD-ranked scores CSV', Path(result['csv']).read_bytes(),
                       file_name='redock_scores.csv', key=pfx + 'csv_download')
    st.download_button('⬇ Original crystal reference', Path(result['reference']).read_bytes(),
                       file_name='crystal_reference.pdb', key=pfx + 'reference_download')
    if result.get('warning'): st.warning(result['warning'])
    st.caption(f"Ligand SMILES source: {result['smiles_source']}")
    prep = result.get('prep', {})
    st.caption(f"Actual protonation: {prep.get('protonation_mode', 'not reported')} · Formal charge: {prep.get('net_charge', prep.get('charge', 'not reported'))}")
    with st.expander('Ligand preparation log', expanded=False):
        st.code('\n'.join(prep.get('log', [])))
    with st.expander('Vina output log', expanded=False):
        st.code(result['dock'].get('log', ''))


def docking_controls(st, core, wdir, vina_path, show_pose, pfx='r_'):
    st.markdown('### Step 2 — Redock the extracted ligand')
    inspection = st.session_state.get(pfx + 'inspection')
    prepared = st.session_state.get(pfx + 'prepared')
    ph = st.number_input('pH', min_value=0.0, max_value=14.0, value=7.4, key=pfx + 'ph')
    mode = st.selectbox('Protonation mode', ['pkanet', 'neutral'], key=pfx + 'prot')
    exh = st.slider('Exhaustiveness', 1, 128, 16, key=pfx + 'exh')
    modes = st.slider('Maximum poses', 1, 50, 10, key=pfx + 'modes')
    energy = st.slider('Energy range (kcal/mol)', 1, 20, 3, key=pfx + 'energy')
    with st.expander('Reproducibility (random seed)', expanded=False):
        seed = st.number_input('Vina seed (0 = automatic)', min_value=0, value=0, key=pfx + 'seed')
        conformer = st.number_input('Conformer seed (0 = automatic)', min_value=0, value=0, key=pfx + 'conformer_seed')
    ready = bool(inspection and inspection['ready'] and prepared)
    st.markdown('<style>.st-key-r_btn_redock button:enabled {background-color:#198754!important;color:white!important;border-color:#198754!important;}</style>', unsafe_allow_html=True)
    if st.button('Redock', type='primary', disabled=not ready, key=pfx + 'btn_redock'):
        st.session_state.pop(pfx + 'result', None)
        st.session_state.pop(pfx + 'pose', None)
        try:
            import tempfile
            out = Path(tempfile.mkdtemp(prefix='run_', dir=str(wdir)))
            with st.spinner('Preparing extracted ligand and redocking with AutoDock Vina…'):
                result = execute_prepared_redock(core, inspection, prepared, out, vina_path,
                                                ph=ph, mode=mode, exhaustiveness=exh, n_modes=modes,
                                                energy_range=energy, seed=int(seed) or None,
                                                conformer_seed=int(conformer) or None)
            result['settings'] = dict(ph=ph, mode=mode, exhaustiveness=exh, n_modes=modes,
                                      energy_range=energy, seed=seed, conformer_seed=conformer)
            (out / 'redock_manifest.json').write_text(json.dumps({'inspection': inspection, 'settings': result['settings']}, indent=2))
            st.session_state[pfx + 'result'] = result
        except Exception as exc:
            st.error(f'Redocking failed: {exc}')
    if not ready:
        st.caption('Redock becomes available after an eligible structure has been prepared.')
    result = st.session_state.get(pfx + 'result')
    if result and prepared:
        st.caption('Completed-run parameters: ' + json.dumps(result['settings']))
        render_results(st, core, result, prepared, show_pose, pfx)
