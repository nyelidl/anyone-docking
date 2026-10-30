"""Ligand validation and mapped PDB name restoration (no chemistry imports)."""
from __future__ import annotations

import csv
import hashlib
import math
import re
from pathlib import Path

SUPPORTED_FORMATS = (".sdf", ".mol2", ".pdb", ".pdbqt")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_atom_records(text, *, pdbqt=False):
    atoms = []
    serials = set()
    for line_no, line in enumerate(text.splitlines(), 1):
        if line[:6].strip() not in ("ATOM", "HETATM"):
            continue
        try:
            atom = dict(index=len(atoms) + 1, serial=int(line[6:11]),
                        atom_name=line[12:16].strip(), atom_name_field=line[12:16], resname=line[17:20].strip(),
                        chain=line[21:22].strip(), resnum=int(line[22:26]),
                        icode=line[26:27].strip(), altloc=line[16:17].strip(),
                        xyz=tuple(float(line[i:i + 8]) for i in (30, 38, 46)),
                        element="" if pdbqt else line[76:78].strip(),
                        formal_charge="" if pdbqt else line[78:80].strip())
            if not atom["atom_name"] or atom["serial"] <= 0 or atom["serial"] in serials:
                raise ValueError("missing atom name or invalid/duplicate serial")
            if not all(math.isfinite(v) for v in atom["xyz"]):
                raise ValueError("non-finite coordinates")
            if pdbqt:
                atom["charge"] = float(line[70:76])
                atom["type"] = line[77:].strip()
                if not math.isfinite(atom["charge"]) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,3}", atom["type"]):
                    raise ValueError("missing/invalid AutoDock type or partial charge")
            atoms.append(atom)
            serials.add(atom["serial"])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid ligand {'PDBQT' if pdbqt else 'PDB'} atom at line {line_no}: {exc}") from exc
    if not atoms:
        raise ValueError(f"Invalid ligand {'PDBQT' if pdbqt else 'PDB'}: no ATOM/HETATM records were found.")
    return atoms


def validate_ligand_file(file_path):
    path = Path(file_path).expanduser().resolve()
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported ligand format: {path.suffix or '(none)'}\nSupported formats: " + " ".join(SUPPORTED_FORMATS))
    if not path.is_file():
        raise ValueError(f"Ligand file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Invalid ligand {path.suffix}: file is empty: {path}")
    result = {"path": str(path)}
    if path.suffix.lower() != ".pdbqt":
        return result
    raw = path.read_bytes()
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Invalid ligand PDBQT: expected an ASCII PDBQT file.") from exc
    atoms = read_atom_records(text, pdbqt=True)
    # Validate one ligand torsion tree without rewriting or sanitizing anything.
    root = endroot = torsdof = 0
    stack = []
    seen_atoms = set()
    branch_children = []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        tag = fields[0]
        if tag == "REMARK":
            continue
        if tag == "ROOT":
            if root or len(fields) != 1 or torsdof:
                raise ValueError("Invalid ligand PDBQT: duplicate/misplaced ROOT.")
            root += 1
        elif tag == "ENDROOT":
            if root != 1 or endroot or stack or not seen_atoms or len(fields) != 1:
                raise ValueError("Invalid ligand PDBQT: malformed ENDROOT.")
            endroot += 1
        elif tag == "BRANCH":
            try:
                pair = tuple(map(int, fields[1:]))
                if len(pair) != 2 or not endroot or torsdof or pair[0] not in seen_atoms or pair[1] in seen_atoms:
                    raise ValueError()
            except ValueError as exc:
                raise ValueError("Invalid ligand PDBQT: malformed BRANCH.") from exc
            stack.append(pair)
            branch_children.append(pair[1])
        elif tag == "ENDBRANCH":
            try:
                pair = tuple(map(int, fields[1:]))
                if not stack or stack.pop() != pair or pair[1] not in seen_atoms:
                    raise ValueError()
            except ValueError as exc:
                raise ValueError("Invalid ligand PDBQT: unmatched ENDBRANCH.") from exc
        elif tag == "TORSDOF":
            if len(fields) != 2 or not fields[1].isdigit() or not endroot or stack or torsdof:
                raise ValueError("Invalid ligand PDBQT: malformed TORSDOF.")
            torsdof += 1
        elif tag in ("ATOM", "HETATM"):
            if not root or torsdof or (endroot and not stack):
                raise ValueError("Invalid ligand PDBQT: atom outside ROOT/BRANCH.")
            seen_atoms.add(int(line[6:11]))
        else:
            raise ValueError(f"Invalid ligand PDBQT: unsupported record {tag!r}; provide one prepared ligand, not a multi-pose file.")
    if root != 1 or endroot != 1 or torsdof != 1 or stack or len(branch_children) != len(set(branch_children)):
        raise ValueError("Invalid ligand PDBQT: incomplete or ambiguous torsion tree.")
    result.update(atom_count=len(atoms), input_sha256=hashlib.sha256(raw).hexdigest())
    return result


_MAPPING_FIELDS = [
    "input_rdkit_index", "output_atom_name_before_restore", "output_atom_name_after_restore",
    "input_index", "input_serial", "output_index", "output_serial", "prepared_index",
    "input_atom_name", "output_atom_name", "input_element", "output_type",
    "input_resname", "output_resname", "input_resnum", "output_resnum",
    "input_chain", "output_chain", "input_icode", "output_icode",
    "input_x", "input_y", "input_z", "output_x", "output_y", "output_z",
    "input_formal_charge", "output_partial_charge", "coordinate_difference",
    "mapping_status", "order_status", "reason",
]


def mapping_row(atom):
    row = {"input_rdkit_index": atom["index"] - 1, "input_index": atom["index"], "input_serial": atom["serial"],
           "mapping_status": "not_prepared"}
    for field in ("atom_name", "element", "resname", "resnum", "chain", "icode", "formal_charge"):
        row["input_" + field] = atom.get(field, "")
    row.update(zip(("input_x", "input_y", "input_z"), atom["xyz"]))
    return row


def write_mapping_report(path, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_MAPPING_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def validate_pdb_mapping(original, mol, setup, pdbqt_path, report_path, *, preserve_atom_set=False):
    """Validate Meeko identity independently of names, then restore original PDB labels."""
    text = Path(pdbqt_path).read_text()
    rows = [mapping_row(atom) for atom in original]
    errors = []
    restored_names = {}
    try:
        output = read_atom_records(text, pdbqt=True)
        index_map = {}
        for line in text.splitlines():
            if line.startswith("REMARK INDEX MAP "):
                numbers = list(map(int, line.split()[3:]))
                if len(numbers) % 2:
                    raise ValueError("odd number of values in Meeko INDEX MAP")
                for i, serial in zip(numbers[::2], numbers[1::2]):
                    if i in index_map or serial in index_map.values():
                        raise ValueError("duplicate input/output in Meeko INDEX MAP")
                    index_map[i] = serial
        by_serial = {atom["serial"]: atom for atom in output}
        if set(index_map.values()) != set(by_serial):
            raise ValueError("unmapped, duplicated or unexpected output atoms (including unsupported pseudoatoms)")
        if [atom["serial"] for atom in output] != list(range(1, len(output) + 1)):
            raise ValueError("unexpected output atom ordering: serial sequence differs from the Meeko writer")
        expected = {atom.index + 1 for atom in setup.atoms if not atom.is_ignore}
        if set(index_map) != expected or any(i < 1 or i > mol.GetNumAtoms() for i in expected):
            raise ValueError("INDEX MAP does not match Meeko's retained input atoms")
        if len(setup.atoms) != mol.GetNumAtoms():
            raise ValueError("Meeko added pseudoatoms; unique physical-atom mapping is not supported for this ligand")
        retained = [i for i in index_map if i <= len(original)]
        reordered = sorted(retained, key=index_map.get) != sorted(retained)
        for i, atom in enumerate(mol.GetAtoms(), 1):
            row = rows[i - 1] if i <= len(original) else {"mapping_status": "added_hydrogen"}
            if i > len(original):
                rows.append(row)
                if atom.GetAtomicNum() != 1:
                    errors.append(f"Unexpected generated non-hydrogen atom {i}")
            row["prepared_index"] = i
            prepared = setup.atoms[i - 1]
            if prepared.atomic_num != atom.GetAtomicNum():
                raise ValueError(f"Meeko element changed at RDKit atom {i - 1}")
            if prepared.is_ignore:
                if preserve_atom_set:
                    raise ValueError(f"Original atom {i} was dropped despite exact atom-set preservation")
                parent = prepared.graph[0] + 1 if len(prepared.graph) == 1 else None
                if atom.GetAtomicNum() == 1 and prepared.atom_type == "H" and parent in index_map:
                    row.update(mapping_status="merged_nonpolar_h" if i <= len(original) else "added_then_merged_h",
                               order_status="not_written", reason=f"Meeko nonpolar H merged; charge transferred to prepared atom {parent}")
                else:
                    row.update(mapping_status="FAIL", reason="unexpectedly dropped atom")
                    errors.append(f"Unexpectedly dropped atom {i}: {atom.GetSymbol()}")
                continue
            out = by_serial[index_map[i]]
            info = atom.GetPDBResidueInfo()
            point = mol.GetConformer().GetAtomPosition(i - 1)
            baseline = original[i - 1] if i <= len(original) else dict(
                atom_name=info.GetName().strip(), resname=info.GetResidueName().strip(),
                resnum=info.GetResidueNumber(), chain=info.GetChainId().strip(),
                icode=info.GetInsertionCode().strip(), xyz=(point.x, point.y, point.z))
            delta = math.dist(baseline["xyz"], out["xyz"])
            row.update(output_index=out["index"], output_serial=out["serial"], output_type=out["type"],
                       output_partial_charge=out["charge"], coordinate_difference=delta,
                       order_status="mapped_torsion_tree_order" if reordered else "preserved")
            for key in ("atom_name", "resname", "resnum", "chain", "icode"):
                row["output_" + key] = out[key]
            row.update(zip(("output_x", "output_y", "output_z"), out["xyz"]))
            differences = [key for key in ("resname", "resnum", "chain", "icode")
                           if baseline[key] != out[key]]
            if delta > 0.0011:
                differences.append("coordinates")
            # Confirm that the emitted AutoDock type is the one assigned to this exact atom.
            if out["type"] != prepared.atom_type:
                differences.append("AutoDock type")
            if i <= len(original):
                restored_names[out["serial"]] = baseline["atom_name_field"]
                row.update(output_atom_name_before_restore=out["atom_name"],
                           output_atom_name_after_restore=baseline["atom_name"],
                           output_atom_name=baseline["atom_name"])
            elif baseline["atom_name"] != out["atom_name"]:
                differences.append("generated H atom name")
            if differences:
                row.update(mapping_status="FAIL", reason="Changed " + ", ".join(differences))
                errors.append(f"Atom {i} ({baseline['atom_name']}): {row['reason']}")
            else:
                row["mapping_status"] = "mapped" if i <= len(original) else "added_hydrogen"
        if errors:
            raise ValueError("; ".join(errors))
        # Only after proving the index mapping and chemical/coordinate identity,
        # restore the exact original fixed-width name field. Never reorder records.
        lines = []
        for line in text.splitlines(keepends=True):
            if line.startswith(("ATOM  ", "HETATM")):
                name = restored_names.get(int(line[6:11]))
                if name is not None:
                    line = line[:12] + name + line[16:]
            lines.append(line)
        restored = "".join(lines)
        final_atoms = {a["serial"]: a for a in read_atom_records(restored, pdbqt=True)}
        for serial, name in restored_names.items():
            if final_atoms[serial]["atom_name_field"] != name:
                raise ValueError(f"Final atom-name restoration failed for output serial {serial}")
        Path(pdbqt_path).write_text(restored)

    except Exception as exc:
        for row in rows:
            if row["mapping_status"] == "not_prepared":
                row.update(mapping_status="FAIL", reason=str(exc))
        write_mapping_report(report_path, rows)
        raise ValueError(f"Ligand atom mapping failed: {exc}. Report: {report_path}. Docking was not started.") from exc
    write_mapping_report(report_path, rows)
    return {"input_atoms": len(original), "output_atoms": len(output),
            "mapped_original_atoms": sum(r["mapping_status"] == "mapped" for r in rows),
            "merged_original_h": sum(r["mapping_status"] == "merged_nonpolar_h" for r in rows),
            "added_hydrogens": mol.GetNumAtoms() - len(original), "reordered": reordered,
            "restored_names": sum(r.get("output_atom_name_before_restore") != r.get("output_atom_name_after_restore")
                                  for r in rows if r["mapping_status"] == "mapped")}
