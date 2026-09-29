"""Local, annotation-based missing-residue assessment. No loop modeling or API calls."""
from pathlib import Path
import hashlib
import json

CUTOFF = 5.0


def _clean(value):
    return '' if value in (None, '.', '?') else str(value).strip()


def _annotations(path, model):
    import gemmi
    missing, sequences, warnings = [], {}, []
    if path.suffix.lower() in ('.cif', '.mmcif') or path.read_text(errors='replace').lstrip().startswith('data_'):
        block = gemmi.cif.read(str(path)).sole_block()
        category = block.get_mmcif_category('_pdbx_unobs_or_zero_occ_residues.')
        for i in range(len(next(iter(category.values()), []))):
            def val(key): return _clean(category.get(key, [''] * (i+1))[i])
            if val('polymer_flag').upper() == 'N': continue
            if val('PDB_model_num') and val('PDB_model_num') != str(model): continue
            chain, number = val('auth_asym_id'), val('auth_seq_id')
            try:
                number = int(number)
            except ValueError:
                warnings.append('An unobserved residue has no usable author residue number.'); continue
            missing.append((chain, number, val('PDB_ins_code'), val('auth_comp_id') or val('label_comp_id')))
        scheme = block.get_mmcif_category('_pdbx_poly_seq_scheme.')
        for i in range(len(next(iter(scheme.values()), []))):
            chain = _clean(scheme.get('pdb_strand_id', [''] * (i+1))[i])
            name = _clean(scheme.get('mon_id', [''] * (i+1))[i])
            if chain and name: sequences.setdefault(chain, []).append(name)
    else:
        for line in path.read_text(errors='replace').splitlines():
            if line.startswith('REMARK 465') and len(line) >= 26:
                try:
                    number = int(line[21:26])
                except ValueError: continue
                named_model = line[11:14].strip()
                if named_model.isdigit() and int(named_model) != model: continue
                missing.append((line[19:20].strip(), number, line[26:27].strip(), line[15:18].strip()))
            elif line.startswith('SEQRES'):
                sequences.setdefault(line[11:12].strip(), []).extend(line[19:70].split())
    return sorted(set(missing)), sequences, warnings


def _missing_atom_annotations(path, model):
    """Read deposited missing-atom annotations; do not infer missing positions."""
    import gemmi
    rows, warnings = [], []
    if path.suffix.lower() in ('.cif', '.mmcif') or path.read_text(errors='replace').lstrip().startswith('data_'):
        category = gemmi.cif.read(str(path)).sole_block().get_mmcif_category('_pdbx_unobs_or_zero_occ_atoms.')
        for i in range(len(next(iter(category.values()), []))):
            def val(key): return _clean(category.get(key, [''] * (i+1))[i])
            if val('polymer_flag').upper() == 'N': continue
            if val('PDB_model_num') and val('PDB_model_num') != str(model): continue
            try:
                number = int(val('auth_seq_id'))
            except ValueError:
                warnings.append('A missing-atom annotation has no usable author residue number.'); continue
            name = val('auth_atom_id') or val('label_atom_id')
            if not name:
                warnings.append('A missing-atom annotation has no atom name.'); continue
            rows.append((val('auth_asym_id'), number, val('PDB_ins_code'),
                         val('auth_comp_id') or val('label_comp_id'), name, val('label_alt_id')))
    else:
        # wwPDB REMARK 470: model 12–14, name 16–18, chain 20,
        # author sequence 21–24, insertion 25, atom names 26 onward.
        for line in path.read_text(errors='replace').splitlines():
            if not line.startswith('REMARK 470'): continue
            try:
                number = int(line[20:24])
            except ValueError: continue
            named_model = line[11:14].strip()
            if named_model.isdigit() and int(named_model) != model: continue
            for name in line[25:].split():
                rows.append((line[19:20].strip(), number, line[24:25].strip(), line[15:18].strip(), name, ''))
    return sorted(set(rows)), warnings


def normalize_report(report):
    """Retire legacy far/complete claims without changing docking permission."""
    import copy
    report = copy.deepcopy(report)
    for gap in report.get('gaps', []):
        if gap.get('status') == 'far':
            gap['status'] = 'unknown'
            gap['note'] = 'Observed boundaries are outside the cutoff; the missing segment location is unknown.'
    if report.get('status') == 'far' or (report.get('status') == 'complete' and not report.get('atom_completeness_verified')):
        report['status'] = 'unknown'
        report['reason'] = 'Binding-site completeness unknown: this earlier report does not establish the location of missing segments or atom completeness. Re-run receptor preparation for the updated report.'
    return report


def _heavy(residue):
    return [[a.pos.x, a.pos.y, a.pos.z] for a in residue if not a.element.is_hydrogen and a.occ > 0]


def assess_missing_residues(raw_path, ligand_path=None, receptor_path=None, cutoff=CUTOFF):
    """Distances belong to observed gap boundaries, never to the missing atoms.

    Near => block. Distant boundaries do not locate the missing segment: unknown.
    Terminal/unmapped gaps, absent sequence evidence, and missing sites => unknown.
    """
    result = dict(report_version=2, status='unknown', blocked=False, cutoff_A=cutoff, gaps=[],
                  missing_atoms=[], atom_completeness_verified=False, warnings=[],
                  reason='Missing-residue status unknown: insufficient annotations or sequence information.')
    try:
        import gemmi
        import numpy as np
        from scipy.spatial import cKDTree
        raw = Path(raw_path)
        result['source_sha256'] = hashlib.sha256(raw.read_bytes()).hexdigest()
        structure = gemmi.read_structure(str(raw))
        if not len(structure): raise ValueError('No coordinate model found.')
        model = structure[0]
        missing, sequences, warnings = _annotations(raw, model.num)
        result['warnings'].extend(warnings)
        atom_rows, atom_warnings = _missing_atom_annotations(raw, model.num)
        result['warnings'].extend(atom_warnings)
        observed_atoms = set()
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.occ > 0:
                        key = (chain.name.strip(), residue.seqid.num, residue.seqid.icode.strip(), residue.name, atom.name.strip())
                        observed_atoms.add(key + ('',))
                        observed_atoms.add(key + (atom.altloc.strip('\x00 '),))
        for chain, number, ins, name, atom_name, alt in atom_rows:
            if (chain, number, ins, name, atom_name, alt) in observed_atoms: continue
            result['missing_atoms'].append(dict(chain=chain, residue=f'{name} {number}{ins}',
                                                atom=atom_name, alternate=alt or None))
        result['missing_atom_count'] = len(result['missing_atoms'])
        if result['missing_atoms']:
            result['warnings'].append('Annotated missing atoms are listed separately. Their positions and binding-site impact are unknown; this is not an exhaustive atom-completeness check.')
        observed = {}
        for chain in model:
            for residue in chain:
                if not gemmi.find_tabulated_residue(residue.name).is_amino_acid(): continue
                xyz = _heavy(residue)
                if xyz:
                    observed.setdefault(chain.name.strip(), {})[(residue.seqid.num, residue.seqid.icode.strip())] = (residue.name, xyz)
        # Assess all original protein chains: CIF conversion/filtering can rename chains,
        # and a removed nearby chain's unresolved segment must not be declared safe.
        relevant_chains = set(observed) | {m[0] for m in missing}
        result['scope'] = 'all original protein chains, first coordinate model'
        declared = {(c, n, i) for c, n, i, _ in missing}
        # Rebuilt annotations may be stale: resolved residues are not missing now.
        missing = [m for m in missing if (m[1], m[2]) not in observed.get(m[0], {})]
        result['resolved_annotation_count'] = len(declared) - len(missing)
        if not missing:
            complete = bool(relevant_chains) and all(
                sequences.get(c) == [v[0] for _, v in sorted(observed.get(c, {}).items())]
                for c in relevant_chains)
            result['sequence_coverage_verified'] = bool(complete and not warnings)
            if complete and not warnings:
                result['reason'] = 'Sequence coverage matches the observed residues, but binding-site completeness is unknown: atom completeness has not been verified.'
            if result['missing_atoms']:
                result['reason'] = 'Annotated missing atoms detected; binding-site impact is unknown. Docking remains available, but binding-site completeness is not confirmed.'
            return result
        result['missing_residue_count'] = len(missing)
        tree = None
        if ligand_path and Path(ligand_path).is_file():
            ligand = gemmi.read_structure(str(ligand_path))
            xyz = [p for c in ligand[0] for r in c for p in _heavy(r)] if len(ligand) else []
            if xyz:
                tree = cKDTree(xyz)
                result['ligand_sha256'] = hashlib.sha256(Path(ligand_path).read_bytes()).hexdigest()
        if tree is None:
            result['warnings'].append('No co-crystal ligand coordinates available; binding-site proximity cannot be assessed.')
        # Group annotated missing positions between the same observed sequence boundaries.
        groups = {}
        for chain, number, ins, name in missing:
            keys = sorted(observed.get(chain, {}))
            before = max((k for k in keys if k < (number, ins)), default=None)
            after = min((k for k in keys if k > (number, ins)), default=None)
            groups.setdefault((chain, before, after), []).append((number, ins, name))
        for (chain, before, after), residues in groups.items():
            distances = []
            boundary_labels = []
            for key in (before, after):
                boundary_labels.append(f'{chain or "(blank)"}:{key[0]}{key[1]}' if key else None)
                distances.append(float(tree.query(np.array(observed[chain][key][1]))[0].min()) if key and tree is not None else None)
            if any(d is not None and d <= cutoff for d in distances): status = 'near'
            else: status = 'unknown'
            result['gaps'].append(dict(chain=chain, missing_residues=', '.join(f'{name} {n}{i}' for n,i,name in residues),
                                       before=boundary_labels[0], after=boundary_labels[1],
                                       before_distance_A=round(distances[0],3) if distances[0] is not None else None,
                                       after_distance_A=round(distances[1],3) if distances[1] is not None else None,
                                       status=status, note=('Observed boundaries are outside the cutoff; the missing segment location and binding-site impact are unknown.'
                                                            if all(d is not None and d > cutoff for d in distances) else 'Missing residues have no observed coordinates.')))
        statuses = {g['status'] for g in result['gaps']}
        if 'near' in statuses:
            result.update(status='near', blocked=True, reason='Missing residue found close to the binding site: please rebuild and validate the missing region before docking. Docking is disabled.')
        else:
            result['reason'] = 'Missing residues detected; binding-site impact is unknown. Observed boundary distances do not establish where the missing segments lie. Docking remains available under the allow-unknown setting; binding-site completeness is not confirmed.'
        return result
    except Exception as exc:
        result['warnings'].append(str(exc))
        return result


def save_report(report, folder):
    path = Path(folder) / 'pocket_completeness.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    return str(path)


def show_report(st, report):
    if not report: return
    report = normalize_report(report)
    if report['blocked']: st.error(report['reason'])
    elif report['status'] == 'complete': st.success(report['reason'])
    else: st.warning(report['reason'])
    if report.get('gaps'):
        st.dataframe(report['gaps'], hide_index=True)
        st.caption(f"Distances are measured from observed flanking residues to co-crystal ligand heavy atoms. A boundary within {report['cutoff_A']:g} Å blocks docking. Missing residues themselves have no coordinates.")
    if report.get('missing_atoms'):
        st.caption('Annotated missing atoms (reported separately from entirely missing residues)')
        st.dataframe(report['missing_atoms'], hide_index=True)
    for warning in report.get('warnings', []): st.caption(warning)
