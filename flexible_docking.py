"""Optional Vina side-chain flexibility, shared by ACD front ends.

The original receptor charges/types and all retained cofactors are preserved.
Meeko supplies only selected side-chain torsion trees. No loop/backbone modeling.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

PROPERTY = "ACD_RECEPTOR_PDB"
SIDECHAINS = {
    "ARG": "CB CG CD NE CZ NH1 NH2", "ASN": "CB CG OD1 ND2",
    "ASP": "CB CG OD1 OD2", "CYS": "CB SG", "GLN": "CB CG CD OE1 NE2",
    "GLU": "CB CG CD OE1 OE2", "HIS": "CB CG ND1 CD2 CE1 NE2",
    "ILE": "CB CG1 CG2 CD1", "LEU": "CB CG CD1 CD2", "LYS": "CB CG CD CE NZ",
    "MET": "CB CG SD CE", "PHE": "CB CG CD1 CD2 CE1 CE2 CZ",
    "SER": "CB OG", "THR": "CB OG1 CG2", "TRP": "CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
    "TYR": "CB CG CD1 CD2 CE1 CE2 CZ OH", "VAL": "CB CG1 CG2",
}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atoms(path):
    path = Path(path)
    if path.suffix.lower() in (".cif", ".mmcif"):
        import gemmi
        text = gemmi.read_structure(str(path)).make_pdb_string()
    else:
        text = path.read_text()
    rows = []
    for line in text.splitlines(True):
        if line.startswith("ENDMDL"):
            break
        if line.startswith(("ATOM  ", "HETATM")):
            if len(line) < 54:
                raise ValueError("Truncated receptor atom record")
            rows.append(line)
    return rows


def _key(line):
    return (line[21:22].strip(), line[22:26].strip(), line[26:27].strip(), line[12:16].strip())


def _xyz(line):
    return tuple(float(line[i:i+8]) for i in (30, 38, 46))


def _dist(a, b):
    return sum((x-y)**2 for x, y in zip(a, b)) ** 0.5


def _renumber(rows):
    if len(rows) > 99999:
        raise ValueError("Pose export exceeds the PDB atom-serial limit")
    return [line[:6] + f"{i:5d}" + line[11:] for i,line in enumerate(rows, 1)]


def _ligand_pdb(line):
    atom_type = line[77:].strip().split()[0]
    element = {"A":"C", "NA":"N", "OA":"O", "SA":"S", "HD":"H", "HS":"H"}.get(atom_type, atom_type)
    if atom_type.startswith("CG"):
        element = "C"
    if re.fullmatch(r"G\d+", atom_type):
        return ""  # Vina/Meeko macrocycle closure pseudoatom, not a chemical atom.
    return line[:66].ljust(76) + element.upper().rjust(2) + "\n"


def parse_residues(value):
    """Chain:number (including insertion code); _:number means a blank chain."""
    tokens = re.split(r"[\s,;]+", value.strip()) if isinstance(value, str) else list(value or [])
    result = []
    for token in tokens:
        if not token:
            continue
        match = re.fullmatch(r"([^:\s]):(-?\d+)([A-Za-z]?)", token)
        if not match:
            raise ValueError(f"Invalid flexible residue '{token}'; use A:315 or A:315B (blank chain: _:315)")
        chain, number, icode = match.groups()
        canonical = f"{chain}:{int(number)}{icode}"
        if canonical not in result:
            result.append(canonical)
    return result


def candidate_residues(pdb):
    groups = {}
    for line in _atoms(pdb):
        if not line.startswith("ATOM  "):
            continue
        chain, number, icode, name = _key(line)
        resname = line[17:20].strip()
        if resname not in SIDECHAINS:
            continue
        key = f"{chain or '_'}:{number}{icode}"
        group = groups.setdefault(key, {"key": key, "resname": resname, "atoms": set(), "coords": []})
        group["atoms"].add(name)
        group["coords"].append(_xyz(line))
    return [dict(key=g["key"], resname=g["resname"], coords=g["coords"],
                 complete=set(("N CA C O " + SIDECHAINS[g["resname"]]).split()) <= g["atoms"])
            for g in groups.values()]


def metadata(receptor_pdbqt):
    path = Path(str(receptor_pdbqt) + ".flex.json")
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    for field, digest in (("rigid_pdbqt", "rigid_sha256"), ("flex_pdbqt", "flex_sha256"),
                          ("receptor_pdb", "receptor_sha256")):
        if _hash(data[field]) != data[digest]:
            raise ValueError("Flexible receptor inputs changed; prepare the receptor again")
    return data


def prepare_flexible(receptor_pdb, receptor_pdbqt, residues, *, max_residues=None, meeko_command=None):
    residues = parse_residues(residues)
    sidecar = Path(str(receptor_pdbqt) + ".flex.json")
    sidecar.unlink(missing_ok=True)
    if not residues:
        return None
    if max_residues and len(residues) > max_residues:
        raise ValueError(f"This interface allows at most {max_residues} flexible residues")
    pdb = Path(receptor_pdb).resolve()
    full = Path(receptor_pdbqt).resolve()
    directory = full.parent / (full.stem + "_flexible")
    directory.mkdir(exist_ok=True)
    rows = _atoms(pdb)
    candidates = {row["key"]: row for row in candidate_residues(pdb)}
    selected = set()
    metals = {"ZN", "MG", "CA", "MN", "FE", "CU", "CO", "NI", "CD", "HG"}
    metal_xyz = [_xyz(l) for l in rows if l[76:78].strip().upper() in metals or l[17:20].strip() in metals]
    for residue in residues:
        candidate = candidates.get(residue)
        if not candidate or not candidate["complete"]:
            raise ValueError(f"{residue}: missing, incomplete, or unsupported flexible side chain (ALA/GLY/PRO are excluded)")
        chain, number = residue.split(":")
        match = re.fullmatch(r"(-?\d+)([A-Za-z]?)", number)
        key = ("" if chain == "_" else chain, match[1], match[2])
        selected.add(key)
        residue_rows = [l for l in rows if _key(l)[:3] == key]
        if any(l[16:17].strip() for l in residue_rows):
            raise ValueError(f"{residue}: resolve alternate conformations before flexible preparation")
        if any(l[12:16].strip() not in {"N", "CA", "C", "O"} and
               l[76:78].strip().upper() in {"N", "O", "S"} and
               any(_dist(_xyz(l), m) < 2.8 for m in metal_xyz) for l in residue_rows):
            raise ValueError(f"{residue}: metal-coordinating side chain; flexible metal coordination is unsupported")
        if candidate["resname"] == "CYS":
            sulfurs = [l for l in residue_rows if l[12:16].strip() == "SG"]
            if any(_key(other)[:3] != key and other[12:16].strip() == "SG" and
                   _dist(_xyz(s), _xyz(other)) < 2.5 for s in sulfurs for other in rows):
                raise ValueError(f"{residue}: disulfide-linked cysteine cannot be made flexible")
    # Protein-only Meeko input prevents unsupported cofactors being retargeted.
    protein = directory / "protein.pdb"
    protein.write_text("".join(l for l in rows if l.startswith("ATOM  ") and
                               l[76:78].strip().upper() != "H" and not l[12:16].strip().startswith("H")) + "END\n")
    prefix = directory / "meeko"
    command = list(meeko_command or [sys.executable, "-m", "meeko.cli.mk_prepare_receptor"])
    spec = ",".join(("" if r.startswith("_:") else r.split(":")[0]) + ":" + r.split(":")[1] for r in residues)
    command += ["--read_pdb", str(protein), "-o", str(prefix), "-p", "-f", spec, "--allow_bad_res"]
    for old in directory.glob("meeko*.pdbqt"):
        old.unlink()
    run = subprocess.run(command, capture_output=True, text=True, timeout=300)
    flex = Path(str(prefix) + "_flex.pdbqt")
    if run.returncode or not flex.is_file():
        raise ValueError("Meeko flexible preparation failed; no rigid fallback is allowed. " + (run.stderr + run.stdout)[-2500:])
    flex_text = flex.read_text()
    flex_atoms = [l for l in flex_text.splitlines(True) if l.startswith(("ATOM  ", "HETATM"))]
    flex_keys = {_key(l) for l in flex_atoms}
    if not flex_atoms or {k[:3] for k in flex_keys} != selected:
        raise ValueError("Meeko flexible residue identities do not match the selected receptor residues")
    original = full.read_text().splitlines(True)
    original_atoms = {_key(l): l for l in original if l.startswith(("ATOM  ", "HETATM"))}
    is_h = lambda line: line[77:].strip().startswith("H")
    original_heavy = [l for l in original_atoms.values() if not is_h(l)]
    flex_heavy = [l for l in flex_atoms if not is_h(l)]
    flex_heavy_keys = {_key(l) for l in flex_heavy}
    old_h, new_h = {}, {}
    for atoms, heavy, groups in ((list(original_atoms.values()), original_heavy, old_h), (flex_atoms, flex_heavy, new_h)):
        for line in atoms:
            if not is_h(line) or _key(line)[:3] not in selected:
                continue
            parents = [l for l in heavy if _key(l)[:3] == _key(line)[:3]]
            if not parents:
                raise ValueError("Flexible hydrogen has no identifiable parent atom")
            parent = min(parents, key=lambda p: _dist(_xyz(p), _xyz(line)))
            if _key(parent) in flex_heavy_keys:
                groups.setdefault(_key(parent), []).append(line)
    if {k:len(v) for k,v in old_h.items()} != {k:len(v) for k,v in new_h.items()}:
        raise ValueError("Meeko changed selected side-chain protonation; prepare a consistent receptor protonation state first")
    hydrogen_sources = {}
    for parent in new_h:
        for new, old in zip(sorted(new_h[parent], key=_key), sorted(old_h[parent], key=_key)):
            hydrogen_sources[_key(new)] = old
    removed_keys = flex_keys | {_key(l) for group in old_h.values() for l in group}
    # Keep ACD's validated atom types/charges, including heme and catalytic ions.
    # Meeko may supply hydrogens missing in the original PDBQT; retain those.
    rewritten = []
    for line in flex_text.splitlines(True):
        if line.startswith(("ATOM  ", "HETATM")):
            key = _key(line)
            source = hydrogen_sources.get(key) if is_h(line) else original_atoms.get(key)
            if source:
                if not line[77:].strip().startswith("H") and _dist(_xyz(source), _xyz(line)) > 0.02:
                    raise ValueError("Meeko changed initial heavy-atom coordinates")
                line = line[:66] + source[66:].rstrip() + "\n"
            elif not line[77:].strip().startswith("H"):
                raise ValueError(f"Flexible atom {key} is absent from the prepared receptor")
        rewritten.append(line)
    flex.write_text("".join(rewritten))
    rigid = directory / "receptor_rigid.pdbqt"
    rigid.write_text("".join(l for l in original if not l.startswith(("ATOM  ", "HETATM")) or _key(l) not in removed_keys))
    data = {"residues": residues, "receptor_pdb": str(pdb), "full_pdbqt": str(full),
            "rigid_pdbqt": str(rigid), "flex_pdbqt": str(flex), "meeko_command": command, "meeko_log": (run.stdout + run.stderr)[-8000:],
            "receptor_sha256": _hash(pdb), "rigid_sha256": _hash(rigid), "flex_sha256": _hash(flex)}
    sidecar.write_text(json.dumps(data, indent=2))
    return data


def split_output(raw_output, data):
    """Separate ligand torsion trees and restore each pose's receptor coordinates."""
    raw_output = Path(raw_output)
    ligand_path = raw_output.with_name(raw_output.stem + "_ligands.pdbqt")
    expected = {_key(l) for l in _atoms(data["flex_pdbqt"])}
    base = _atoms(data["receptor_pdb"])
    models, current, ligands, flex, inside = [], None, [], [], False
    for line in raw_output.read_text().splitlines(True):
        if line.startswith("MODEL"):
            if current is not None:
                raise ValueError("Nested Vina MODEL record")
            current = int(line.split()[1]); ligands = [line]; flex = []
        elif line.startswith("BEGIN_RES"):
            if current is None or inside:
                raise ValueError("Invalid flexible residue output")
            inside = True
        elif line.startswith("END_RES"):
            if not inside:
                raise ValueError("Unmatched END_RES")
            inside = False
        elif line.startswith("ENDMDL"):
            if current is None or inside:
                raise ValueError("Incomplete flexible pose")
            ligands.append(line)
            models.append((current, ligands, flex)); current = None
        elif inside:
            if line.startswith(("ATOM  ", "HETATM")):
                flex.append(line)
        elif current is not None:
            ligands.append(line)
    if current is not None or not models:
        raise ValueError("Vina output has no complete flexible docking poses")
    receptors = {}; complexes = {}
    for pose, ligand_lines, moved in models:
        keys = [_key(l) for l in moved]
        if len(set(keys)) != len(keys) or set(keys) != expected:
            raise ValueError(f"Pose {pose}: flexible atom identities/counts do not match the input")
        updates = {_key(l): l for l in moved}
        receptor_lines = []
        seen = set()
        for line in base:
            key = _key(line)
            if key in updates:
                line = line[:30] + updates[key][30:54] + line[54:]
                seen.add(key)
            receptor_lines.append(line)
        # Remove stale nonpolar side-chain H (Vina does not move those atoms).
        selected = {k[:3] for k in expected}
        receptor_lines = [l for l in receptor_lines if not (
            _key(l)[:3] in selected and _key(l) not in updates and
            (l[76:78].strip().upper() == "H" or l[12:16].strip().startswith("H")) and
            l[12:16].strip() not in {"H", "HN", "HA", "HA2", "HA3"})]
        if any(k not in seen and not updates[k][77:].strip().startswith("H") for k in expected):
            raise ValueError("Flexible heavy atom missing from receptor PDB")
        for key in expected - seen:
            l = updates[key]
            receptor_lines.append(l[:66].ljust(76) + " H\n")
        rec = raw_output.with_name(raw_output.stem + f"_pose{pose}_receptor.pdb")
        receptor_lines = _renumber(receptor_lines)
        rec.write_text("".join(receptor_lines) + "END\n")
        complex_path = rec.with_name(rec.name.replace("_receptor.pdb", "_complex.pdb"))
        ligand_atoms = [_ligand_pdb(l) for l in ligand_lines if l.startswith(("ATOM  ", "HETATM"))]
        ligand_atoms = [l for l in ligand_atoms if l]
        complex_atoms = _renumber(receptor_lines + ligand_atoms)
        complex_path.write_text("".join(complex_atoms[:len(receptor_lines)]) + "TER\n" +
                                "".join(complex_atoms[len(receptor_lines):]) + "END\n")
        receptors[str(pose)] = str(rec); complexes[str(pose)] = str(complex_path)
    ligand_path.write_text("".join(l for _, lines, _ in models for l in lines))
    return {"out_pdbqt": str(ligand_path), "combined_out_pdbqt": str(raw_output),
            "pose_receptors": receptors, "pose_complexes": complexes, "flexibility": data}


def validate_box(data, config):
    box = {}
    for line in Path(config).read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key.strip() in {f"{prefix}_{axis}" for prefix in ("center", "size") for axis in "xyz"}:
                box[key.strip()] = float(value.split("#", 1)[0].strip())
    if len(box) != 6 or any(box["size_" + a] <= 0 for a in "xyz"):
        raise ValueError("Flexible docking requires a complete, positive docking box")
    outside = []
    for line in _atoms(data["flex_pdbqt"]):
        if any(abs(x-box["center_" + a]) > box["size_" + a]/2 for a,x in zip("xyz", _xyz(line))):
            outside.append(":".join(_key(line)[:2]))
    if outside:
        raise ValueError("Flexible side-chain atoms lie outside the docking box: " + ", ".join(sorted(set(outside))) + ". Enlarge or reposition the box and prepare again.")


def annotate_sdf(path, pose_receptors):
    """Store pose-specific receptor provenance on every ligand, retained by RDKit."""
    path = Path(path)
    records = path.read_text().split("$$$$")
    records = [r for r in records if r.strip()]
    if len(records) != len(pose_receptors):
        raise ValueError("Ligand SDF and flexible receptor pose counts differ")
    path.write_text("".join(r.rstrip() + f"\n\n> <{PROPERTY}>\n{receptor}\n\n$$$$\n"
                            for r, receptor in zip(records, pose_receptors.values())))


def receptor_for_mol(default, mol):
    if mol is not None and mol.HasProp(PROPERTY):
        path = mol.GetProp(PROPERTY)
        if not Path(path).is_file():
            raise ValueError("Pose-specific flexible receptor is missing; restore the full result directory")
        return path
    return default


def receptor_for_sdf(default, sdf, index=0):
    from rdkit import Chem
    mols = list(Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False))
    if index >= len(mols) or mols[index] is None:
        raise ValueError("Cannot resolve receptor for unreadable ligand pose")
    if mols[index].HasProp(PROPERTY) and not Path(mols[index].GetProp(PROPERTY)).is_absolute():
        mols[index].SetProp(PROPERTY, str(Path(sdf).resolve().parent / mols[index].GetProp(PROPERTY)))
    return receptor_for_mol(default, mols[index])


def export_pose_context(mol, output_stem):
    if not mol.HasProp(PROPERTY):
        return
    source = Path(receptor_for_mol(None, mol))
    output_stem = Path(output_stem)
    shutil.copy2(source, str(output_stem) + "_receptor.pdb")
    complex_path = source.with_name(source.name.replace("_receptor.pdb", "_complex.pdb"))
    if complex_path.is_file():
        shutil.copy2(complex_path, str(output_stem) + "_complex.pdb")


def run_fingerprint(fp, ligands, protein, rd_mols, receptor_pdb, plf):
    """ProLIF must use each pose's matching receptor, including batch profiles."""
    if not any(m.HasProp(PROPERTY) for m in rd_mols):
        fp.run_from_iterable(ligands, protein, progress=False, n_jobs=1)
        return
    from rdkit import Chem
    frames = {}
    for i, (ligand, mol) in enumerate(zip(ligands, rd_mols)):
        path = receptor_for_mol(receptor_pdb, mol)
        protein_mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=True)
        if protein_mol is None:
            raise ValueError("Pose-specific flexible receptor cannot be parsed for ProLIF")
        fp.run_from_iterable([ligand], plf.Molecule.from_rdkit(protein_mol), progress=False, n_jobs=1)
        frames[i] = fp.ifp[0]
    fp.ifp = frames


def run_flexible(core, options, data, *, multiple=False):
    validate_box(data, options["config_txt"])
    work = Path(options.get("wdir", ".")).resolve(); work.mkdir(parents=True, exist_ok=True)
    name = options.get("out_name", "joint" if multiple else "out")
    raw = work / f"{name}_out.pdbqt"
    sdf = work / f"{name}_out.sdf"
    ligands = options["ligand_pdbqts"] if multiple else [options["ligand_pdbqt"]]
    inputs = [Path(p).resolve() for p in [data["rigid_pdbqt"], data["flex_pdbqt"], options["config_txt"], *ligands]]
    if raw.resolve() in inputs or sdf.resolve() in inputs:
        raise ValueError("Flexible output would overwrite an input")
    if options.get("ligand_input_sha256") and _hash(ligands[0]) != options["ligand_input_sha256"]:
        raise ValueError("Input PDBQT checksum changed; docking was not started")
    for path in [raw, sdf, raw.with_name(raw.stem + "_ligands.pdbqt"),
                 raw.with_name(raw.stem + "_flex_results.json"),
                 *work.glob(raw.stem + "_pose*_receptor.pdb"),
                 *work.glob(raw.stem + "_pose*_complex.pdb")]:
        path.unlink(missing_ok=True)
    command = [str(options["vina_path"]), "--receptor", data["rigid_pdbqt"], "--flex", data["flex_pdbqt"],
               "--ligand", *[str(Path(p).resolve()) for p in ligands], "--config", str(Path(options["config_txt"]).resolve()),
               "--exhaustiveness", str(options.get("exhaustiveness", 16)), "--num_modes", str(options.get("n_modes", 10)),
               "--energy_range", str(options.get("energy_range", 3)), "--out", str(raw)]
    if options.get("seed") is not None:
        command += ["--seed", str(options["seed"])]
    if options.get("cpu") is not None:
        command += ["--cpu", str(options["cpu"])]
    rc, log = core["run_cmd"](command, cwd=str(work))
    if rc or not raw.is_file():
        return {"success": False, "error": f"Flexible Vina failed (exit {rc})", "log": log}
    result = split_output(raw, data)
    scores = []
    pose = None
    for line in Path(result["out_pdbqt"]).read_text().splitlines():
        if line.startswith("MODEL"):
            pose = int(line.split()[1])
        elif line.startswith("REMARK VINA RESULT:"):
            p = line.split()
            scores.append(dict(pose=pose, affinity=float(p[3]), rmsd_lb=float(p[4]), rmsd_ub=float(p[5])))
    if [str(s["pose"]) for s in scores] != list(result["pose_receptors"]):
        raise ValueError("Flexible scores and receptor poses do not match")
    result.update(success=True, scores=scores, top_score=scores[0]["affinity"], seed=options.get("seed"), log=log,
                  command=command,
                  ligand_pdbqts=[str(Path(p).resolve()) for p in ligands], score_scope="joint_ligand_complex" if multiple else "ligand_with_flexible_receptor")
    if not multiple:
        rc, conversion = core["run_cmd"](["obabel", result["out_pdbqt"], "-O", str(sdf)])
        if rc or not sdf.is_file():
            raise ValueError("Flexible ligand SDF conversion failed: " + conversion)
        annotate_sdf(sdf, result["pose_receptors"])
        result["out_sdf"] = str(sdf)
    manifest = raw.with_name(raw.stem + "_flex_results.json")
    result["flex_manifest"] = str(manifest)
    manifest.write_text(json.dumps(result, indent=2))
    return result


def install(namespace):
    """Extend existing core entry points without changing rigid docking behavior."""
    if namespace.get("_flexible_docking_installed"):
        return
    namespace["_flexible_docking_installed"] = True
    original_prepare = namespace["prepare_receptor"]

    @functools.wraps(original_prepare)
    def prepare(*args, flexible_residues=None, **kwargs):
        result = original_prepare(*args, **kwargs)
        if result.get("success"):
            try:
                command = namespace.get("_meeko_prepare_receptor_cmd", lambda: None)()
                result["flexibility"] = prepare_flexible(result["rec_fh"], result["rec_pdbqt"], flexible_residues,
                                                          meeko_command=command)
                if result["flexibility"]:
                    validate_box(result["flexibility"], result["config_txt"])
                    result.setdefault("log", []).append("Flexible side chains: " + ", ".join(result["flexibility"]["residues"]))
            except Exception as exc:
                result.update(success=False, error=str(exc))
                result.setdefault("log", []).append(str(exc))
        return result
    namespace["prepare_receptor"] = prepare
    original_load = namespace.get("load_mols_from_sdf")
    if original_load is not None:
        @functools.wraps(original_load)
        def load(sdf_path, *args, **kwargs):
            mols = original_load(sdf_path, *args, **kwargs)
            for mol in mols:
                if mol.HasProp(PROPERTY) and not Path(mol.GetProp(PROPERTY)).is_absolute():
                    mol.SetProp(PROPERTY, str(Path(sdf_path).resolve().parent / mol.GetProp(PROPERTY)))
            return mols
        namespace["load_mols_from_sdf"] = load
    for name in ("run_vina", "run_vina_multi"):
        original = namespace[name]
        def make_wrapper(original, multiple):
            @functools.wraps(original)
            def run(*args, **kwargs):
                options = inspect.signature(original).bind(*args, **kwargs)
                options.apply_defaults()
                try:
                    data = metadata(options.arguments["receptor_pdbqt"])
                    if data:
                        return run_flexible(namespace, options.arguments, data, multiple=multiple)
                except Exception as exc:
                    return {"success": False, "error": str(exc), "log": "Flexible docking stopped; rigid fallback was not used."}
                return original(*args, **kwargs)
            return run
        namespace[name] = make_wrapper(original, name.endswith("multi"))
    # Route scientific interaction APIs to the receptor stored on the ligand.
    for name in ("get_interacting_residues", "_detect_all_interactions", "_enrich_with_res_xyz",
                 "_prolif_interacting_residue_centers_from_pose", "draw_interactions_rdkit",
                 "draw_interactions_rdkit_classic", "call_poseview_v1", "_build_diagram_data"):
        original = namespace.get(name)
        if original is None:
            continue
        def make_interaction_wrapper(original):
            @functools.wraps(original)
            def call(*args, **kwargs):
                bound = inspect.signature(original).bind(*args, **kwargs)
                if "receptor_pdb" in bound.arguments:
                    receptor = bound.arguments["receptor_pdb"]
                    for key in ("lig_mol", "lig_mol_3d", "mol3d", "rd_mol"):
                        if key in bound.arguments:
                            receptor = receptor_for_mol(receptor, bound.arguments[key]); break
                    else:
                        for key in ("pose_sdf", "pose_sdf_path"):
                            if key in bound.arguments:
                                receptor = receptor_for_sdf(receptor, bound.arguments[key]); break
                    bound.arguments["receptor_pdb"] = receptor
                return original(*bound.args, **bound.kwargs)
            return call
        namespace[name] = make_interaction_wrapper(original)
