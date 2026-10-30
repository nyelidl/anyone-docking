"""Strict ferric-heme/Compound-I detection and final PDBQT charge validation."""

from __future__ import annotations

import json
import math
from pathlib import Path

HEME_RESNAMES = {"HEM", "HEC", "HEA", "HEB", "HDD", "HDM"}
CYS_RESNAMES = {"CYS", "CYM", "CYP"}
WATER_RESNAMES = {"HOH", "WAT", "DOD", "SOL"}
FEO_MIN = 1.45
FEO_MAX = 1.90
FE_S_MIN = 1.90
FE_S_MAX = 2.60
OH_MAX = 1.20
SH_MAX = 1.50
PROPIONATE_O_NAMES = {"O1A", "O2A", "O1D", "O2D"}
HYDROGEN_ALIASES = {
    **{f"{index}HM{ring}": f"HM{ring}{index}" for ring in "ABCD" for index in (1, 2, 3)},
    **{f"{index}H{group}": f"H{group}{index}" for group in ("AA", "AD", "BA", "BB", "BC", "BD") for index in (1, 2)},
    "HN": "H", "1HB": "HB2", "2HB": "HB3",
}


class HemeStateError(ValueError):
    pass


def _atom(line: str) -> dict:
    name = line[12:16].strip().upper()
    raw_element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if raw_element not in {"H", "C", "N", "O", "S", "P", "F", "CL", "BR", "I", "FE", "MG", "MN", "ZN", "CA", "CU", "CO", "NI"}:
        stripped_name = name.lstrip("0123456789")
        raw_element = "FE" if stripped_name.startswith("FE") else stripped_name[:1]
    return {
        "line": line,
        "record": line[:6].strip(),
        "serial": int(line[6:11]),
        "name": name,
        "resname": line[17:20].strip().upper(),
        "chain": (line[21:22].strip() or "_"),
        "resid": int(line[22:26]),
        "x": float(line[30:38]),
        "y": float(line[38:46]),
        "z": float(line[46:54]),
        "element": raw_element,
    }


def _distance(a: dict, b: dict) -> float:
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in ("x", "y", "z")))


def _atoms_from_path(path: str) -> list[dict]:
    atoms = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if line[:6].strip() not in {"ATOM", "HETATM"}:
                continue
            try:
                atoms.append(_atom(line))
            except (ValueError, IndexError):
                continue
    return atoms


def center_key(center: dict) -> str:
    return f"{center['resname']}|{center['chain']}|{center['resid']}|{center['fe_serial']}"


def detect_heme_centers(path: str) -> list[dict]:
    atoms = _atoms_from_path(path)
    residue_heavy_counts = {}
    for atom in atoms:
        if atom["element"] != "H" and not atom["name"].startswith("H"):
            key = (atom["resname"], atom["chain"], atom["resid"])
            residue_heavy_counts[key] = residue_heavy_counts.get(key, 0) + 1
    centers = []
    for fe in atoms:
        if fe["resname"] not in HEME_RESNAMES:
            continue
        if not (fe["name"].startswith("FE") or fe["element"] == "FE"):
            continue

        oxo_candidates = []
        for oxygen in atoms:
            if oxygen["element"] != "O" and not oxygen["name"].startswith("O"):
                continue
            if oxygen["serial"] == fe["serial"] or oxygen["resname"] in WATER_RESNAMES:
                continue
            if oxygen["record"] == "ATOM":
                continue
            if oxygen["resname"] in HEME_RESNAMES and oxygen["name"] in PROPIONATE_O_NAMES:
                continue
            oxygen_residue = (oxygen["resname"], oxygen["chain"], oxygen["resid"])
            same_heme = oxygen_residue == (fe["resname"], fe["chain"], fe["resid"])
            standalone_oxo = residue_heavy_counts.get(oxygen_residue, 0) == 1
            if not same_heme and not standalone_oxo:
                continue
            distance = _distance(fe, oxygen)
            if FEO_MIN <= distance <= FEO_MAX:
                oxo_candidates.append((oxygen, distance))
        if len(oxo_candidates) > 1:
            labels = ", ".join(f"{o['name']}#{o['serial']} ({d:.2f} A)" for o, d in oxo_candidates)
            raise HemeStateError(f"Ambiguous ferryl oxygen candidates for Fe {fe['serial']}: {labels}")

        all_sulfurs = [atom for atom in atoms if atom["element"] == "S" or atom["name"] == "SG"]
        if all_sulfurs:
            nearest_sulfur = min(all_sulfurs, key=lambda atom: _distance(fe, atom))
            if _distance(fe, nearest_sulfur) <= FE_S_MAX and not (
                nearest_sulfur["resname"] in CYS_RESNAMES and nearest_sulfur["name"] == "SG"
            ):
                raise HemeStateError(
                    f"Nearest Fe-bound sulfur to Fe {fe['serial']} is not proximal cysteine SG: "
                    f"{nearest_sulfur['resname']} {nearest_sulfur['chain']}:{nearest_sulfur['resid']} "
                    f"{nearest_sulfur['name']}"
                )
        cys_sulfurs = [atom for atom in all_sulfurs if atom["resname"] in CYS_RESNAMES and atom["name"] == "SG"]
        if not cys_sulfurs:
            raise HemeStateError(f"No proximal cysteine SG found for heme Fe {fe['serial']}")
        sulfur = min(cys_sulfurs, key=lambda atom: _distance(fe, atom))
        fe_s = _distance(fe, sulfur)
        if not FE_S_MIN <= fe_s <= FE_S_MAX:
            raise HemeStateError(
                f"Invalid proximal Fe-S geometry for Fe {fe['serial']}: "
                f"nearest {sulfur['resname']} {sulfur['chain']}:{sulfur['resid']} SG "
                f"is {fe_s:.2f} A (required {FE_S_MIN:.2f}-{FE_S_MAX:.2f} A)"
            )

        oxo, fe_o = oxo_candidates[0] if oxo_candidates else (None, None)
        oxo_h = []
        if oxo:
            for hydrogen in atoms:
                if hydrogen["element"] == "H" or hydrogen["name"].startswith("H"):
                    oh = _distance(oxo, hydrogen)
                    if oh <= OH_MAX:
                        oxo_h.append((hydrogen, oh))
            if len(oxo_h) > 1:
                labels = ", ".join(f"{h['name']}#{h['serial']} ({d:.2f} A)" for h, d in oxo_h)
                raise HemeStateError(f"Ambiguous oxo hydrogens for O {oxo['serial']}: {labels}")

        center = {
            "resname": fe["resname"], "chain": fe["chain"], "resid": fe["resid"],
            "fe_serial": fe["serial"], "fe_name": fe["name"],
            "state": "CPD_I" if oxo else "HEME_FERRIC",
            "oxo_serial": oxo["serial"] if oxo else None,
            "oxo_name": oxo["name"] if oxo else None,
            "oxo_resname": oxo["resname"] if oxo else None,
            "oxo_chain": oxo["chain"] if oxo else None,
            "oxo_resid": oxo["resid"] if oxo else None,
            "fe_o_distance": fe_o,
            "cys_resname": sulfur["resname"], "cys_chain": sulfur["chain"],
            "cys_resid": sulfur["resid"], "sg_serial": sulfur["serial"],
            "fe_s_distance": fe_s,
            "oxo_h_serial": oxo_h[0][0]["serial"] if oxo_h else None,
            "oxo_h_name": oxo_h[0][0]["name"] if oxo_h else None,
            "oh_distance": oxo_h[0][1] if oxo_h else None,
        }
        center["key"] = center_key(center)
        centers.append(center)
    return centers


def remove_oxo_hydrogens(path: str, centers: list[dict]) -> list[dict]:
    remove_serials = {c["oxo_h_serial"] for c in centers if c.get("state") == "CPD_I" and c.get("oxo_h_serial")}
    if not remove_serials:
        return centers
    lines = Path(path).read_text(errors="replace").splitlines(keepends=True)
    kept = []
    for line in lines:
        if line[:6].strip() in {"ATOM", "HETATM"}:
            try:
                if int(line[6:11]) in remove_serials:
                    continue
            except ValueError:
                pass
        kept.append(line)
    Path(path).write_text("".join(kept))
    return centers


def load_charge_parameters(data_path: str | None = None) -> dict:
    path = Path(data_path) if data_path else Path(__file__).resolve().parent / "data" / "heme_charges.json"
    if not path.exists():
        raise HemeStateError(f"heme_charges.json is missing: {path}. Docking aborted.")
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        raise HemeStateError(f"heme_charges.json is malformed: {exc}. Docking aborted.") from exc
    source = data.get("source")
    if not isinstance(source, dict) or not source.get("reference") or not source.get("doi"):
        raise HemeStateError("heme_charges.json is missing provenance. Docking aborted.")
    for state in ("HEME_FERRIC", "CPD_I"):
        block = data.get(state)
        if not isinstance(block, dict) or not block.get("heme_atoms") or not block.get("proximal_cysteine"):
            raise HemeStateError(f"{state} RESP parameter set is incomplete. Docking aborted.")
        for group in ("heme_atoms", "proximal_cysteine"):
            if len(block[group]) != len(set(block[group])):
                raise HemeStateError(f"{state} contains duplicate atom mappings in {group}.")
        for charge_group, parent_group in (
            ("heme_atoms", "heme_hydrogen_parents"),
            ("proximal_cysteine", "cysteine_hydrogen_parents"),
        ):
            parents = block.get(parent_group)
            if not isinstance(parents, dict):
                raise HemeStateError(f"{state} is missing {parent_group} topology.")
            if not set(parents).issubset(block[charge_group]) or not set(parents.values()).issubset(block[charge_group]):
                raise HemeStateError(f"{state} contains invalid {parent_group} atom mappings.")
        expected = float(block["expected_total_charge"])
        actual = sum(float(v) for v in block["heme_atoms"].values()) + sum(float(v) for v in block["proximal_cysteine"].values())
        if abs(actual - expected) > float(block.get("charge_tolerance", 0.001)):
            raise HemeStateError(f"{state} parameter total {actual:.4f} does not match expected {expected:.4f}.")
    return data


def prepare_heme_centers(
    path: str,
    requested_states: dict | None = None,
    data_path: str | None = None,
    remove_oxo_h: bool = True,
) -> dict:
    centers = detect_heme_centers(path)
    if not centers:
        return {"centers": [], "parameters": None, "log": []}
    params = load_charge_parameters(data_path)
    requested_states = requested_states or {}
    log = []
    for center in centers:
        requested = requested_states.get(center["key"], "auto")
        selected = center["state"] if requested == "auto" else requested
        if selected != center["state"]:
            raise HemeStateError(
                f"Heme {center['key']} was detected as {center['state']} but {selected} was requested. "
                "This preparation mode validates existing coordinates and does not synthesize or remove ferryl oxygen atoms."
            )
        center["selected_state"] = selected
        center["parameter_set"] = params[selected]["name"]
        log.extend(_center_log(center))
    if remove_oxo_h:
        remove_oxo_hydrogens(path, centers)
    return {"centers": centers, "parameters": params, "log": log}


def _center_log(center: dict) -> list[str]:
    label = f"{center['resname']} {center['chain']} {center['resid']}"
    out = [
        f"Heme center: {label}",
        f"Fe atom: {center['fe_name']} {center['fe_serial']}",
        f"Detected state: {center['state']}",
        f"Proximal Cys: {center['cys_resname']} {center['cys_chain']}:{center['cys_resid']} SG {center['sg_serial']}",
        f"Fe-S distance: {center['fe_s_distance']:.2f} A",
    ]
    if center["state"] == "CPD_I":
        out += [f"Ferryl oxo: {center['oxo_name']} {center['oxo_serial']}", f"Fe-O distance: {center['fe_o_distance']:.2f} A"]
        if center.get("oxo_h_serial"):
            out.append(f"Removed oxo H: {center['oxo_h_name']} {center['oxo_h_serial']} (O-H {center['oh_distance']:.2f} A)")
        else:
            out.append("Axial O-H: none")
    else:
        out.append("Ferryl oxo: none")
    out.append(f"Charge parameter set: {center['parameter_set']}")
    return out


def apply_and_validate_pdbqt(path: str, preparation: dict) -> dict:
    centers = preparation.get("centers", [])
    if not centers:
        return {"centers": [], "log": []}
    params = preparation["parameters"]
    lines = Path(path).read_text(errors="replace").splitlines(keepends=True)
    final_cleanup_log = []
    for center in centers:
        atom_rows = []
        for index, line in enumerate(lines):
            if line[:6].strip() not in {"ATOM", "HETATM"}:
                continue
            try:
                atom = _atom(line)
            except (ValueError, IndexError):
                continue
            atom["index"] = index
            atom_rows.append(atom)
        if center.get("state") == "CPD_I":
            oxos = _find_final_oxo(atom_rows, center)
            if len(oxos) != 1:
                raise HemeStateError("Final PDBQT does not contain exactly one geometrically validated ferryl oxo.")
            attached_h = [
                atom for atom in atom_rows
                if atom["element"] == "H" and _distance(oxos[0], atom) <= OH_MAX
            ]
            if len(attached_h) > 1:
                raise HemeStateError(f"Final PDBQT contains multiple hydrogens attached to ferryl oxo {center['oxo_serial']}.")
            if attached_h:
                hydrogen = attached_h[0]
                distance = _distance(oxos[0], hydrogen)
                lines.pop(hydrogen["index"])
                final_cleanup_log.append(
                    f"Removed converter-added oxo H: {hydrogen['name']} {hydrogen['serial']} (O-H {distance:.2f} A)"
                )

        # Protein converters can protonate the Fe-bound cysteinate sulfur.
        # The published proximal-Cys RESP topology has no SG hydrogen.
        atom_rows = []
        for index, line in enumerate(lines):
            if line[:6].strip() not in {"ATOM", "HETATM"}:
                continue
            try:
                atom = _atom(line)
            except (ValueError, IndexError):
                continue
            atom["index"] = index
            atom_rows.append(atom)
        sulfurs = [
            atom for atom in atom_rows
            if atom["chain"] == center["cys_chain"] and atom["resid"] == center["cys_resid"]
            and atom["resname"] in CYS_RESNAMES and atom["name"] == "SG"
        ]
        if len(sulfurs) != 1:
            raise HemeStateError("Final PDBQT does not contain exactly one validated proximal cysteine SG.")
        sg_h = [
            atom for atom in atom_rows
            if atom["element"] == "H"
            and atom["chain"] == center["cys_chain"]
            and atom["resid"] == center["cys_resid"]
            and atom["resname"] in CYS_RESNAMES
            and _distance(sulfurs[0], atom) <= SH_MAX
        ]
        if len(sg_h) > 1:
            raise HemeStateError("Final PDBQT contains multiple hydrogens attached to proximal cysteine SG.")
        if sg_h:
            hydrogen = sg_h[0]
            distance = _distance(sulfurs[0], hydrogen)
            lines.pop(hydrogen["index"])
            final_cleanup_log.append(
                f"Removed converter-added proximal Cys SG-H: {hydrogen['serial']} (S-H {distance:.2f} A)"
            )

        # The RESP models use deprotonated heme propionates. OpenBabel can add
        # O-H atoms to them when the deposited structure has no hydrogens.
        atom_rows = []
        for index, line in enumerate(lines):
            if line[:6].strip() not in {"ATOM", "HETATM"}:
                continue
            try:
                atom = _atom(line)
            except (ValueError, IndexError):
                continue
            atom["index"] = index
            atom_rows.append(atom)
        propionate_oxygens = [
            atom for atom in atom_rows
            if atom["resname"] == center["resname"] and atom["chain"] == center["chain"]
            and atom["resid"] == center["resid"] and atom["name"] in PROPIONATE_O_NAMES
        ]
        propionate_h = {
            atom["index"]: (atom, oxygen, _distance(oxygen, atom))
            for oxygen in propionate_oxygens
            for atom in atom_rows
            if atom["element"] == "H" and _distance(oxygen, atom) <= OH_MAX
        }
        for index in sorted(propionate_h, reverse=True):
            hydrogen, oxygen, distance = propionate_h[index]
            lines.pop(index)
            final_cleanup_log.append(
                f"Removed converter-added heme propionate H: {hydrogen['serial']} "
                f"from {oxygen['name']} (O-H {distance:.2f} A)"
            )
    parsed = []
    for index, line in enumerate(lines):
        if line[:6].strip() not in {"ATOM", "HETATM"}:
            continue
        try:
            atom = _atom(line)
        except (ValueError, IndexError):
            continue
        atom["index"] = index
        try:
            atom["charge"] = float(line[70:76])
        except ValueError:
            fields = line.split()
            atom["charge"] = float(fields[-2])
        parsed.append(atom)

    log = list(final_cleanup_log)
    for center in centers:
        block = params[center["selected_state"]]
        targets = {}
        heme_atoms = [a for a in parsed if a["resname"] == center["resname"] and a["chain"] == center["chain"] and a["resid"] == center["resid"]]
        cys_atoms = [a for a in parsed if a["chain"] == center["cys_chain"] and a["resid"] == center["cys_resid"] and a["resname"] in CYS_RESNAMES]
        heme_charges, heme_folded = _normalize_hydrogen_names(
            lines, heme_atoms, block["heme_atoms"], block.get("heme_hydrogen_parents", {}), "heme"
        )
        cys_charges, cys_folded = _normalize_hydrogen_names(
            lines, cys_atoms, block["proximal_cysteine"], block.get("cysteine_hydrogen_parents", {}), "proximal Cys"
        )
        if heme_folded or cys_folded:
            log.append(
                f"Collapsed absent nonpolar H RESP charges into parent atoms: "
                f"heme {heme_folded}, proximal Cys {cys_folded}"
            )
        by_name = {}
        for atom in heme_atoms:
            by_name.setdefault(atom["name"], []).append(atom)
        if center["state"] == "CPD_I":
            oxos = _find_final_oxo(parsed, center)
            if len(oxos) != 1:
                raise HemeStateError("Final PDBQT does not contain exactly one geometrically validated ferryl oxo.")
            by_name["OXO"] = oxos
        allowed_heme_names = set(heme_charges) - {"OXO"}
        unknown_heme = [
            atom["name"] for atom in heme_atoms
            if atom["name"] not in allowed_heme_names and atom["serial"] != center.get("oxo_serial")
        ]
        if unknown_heme:
            raise HemeStateError(
                f"Unknown/unparameterized {center['selected_state']} heme atoms: "
                + ", ".join(sorted(set(unknown_heme)))
            )
        for name, charge in heme_charges.items():
            matches = by_name.get(name, [])
            if len(matches) != 1:
                raise HemeStateError(f"{center['selected_state']} atom {name} mapped {len(matches)} times; expected exactly once.")
            targets[matches[0]["index"]] = float(charge)
        cys_by_name = {}
        for atom in cys_atoms:
            cys_by_name.setdefault(atom["name"], []).append(atom)
        unknown_cys = [atom["name"] for atom in cys_atoms if atom["name"] not in cys_charges]
        if unknown_cys:
            raise HemeStateError(
                "Unknown/unparameterized proximal-Cys atoms: " + ", ".join(sorted(set(unknown_cys)))
            )
        for name, charge in cys_charges.items():
            matches = cys_by_name.get(name, [])
            if len(matches) != 1:
                raise HemeStateError(f"Proximal Cys atom {name} mapped {len(matches)} times; expected exactly once.")
            if matches[0]["index"] in targets:
                raise HemeStateError(f"Atom {name} received duplicate heme-state parameters.")
            targets[matches[0]["index"]] = float(charge)
        for index, charge in targets.items():
            line = lines[index].rstrip("\n")
            if len(line) < 76:
                line = line.ljust(76)
            lines[index] = line[:70] + f"{charge:6.3f}" + line[76:] + "\n"
        center["mapped_heme_atoms"] = len(heme_charges)
        center["mapped_cys_atoms"] = len(cys_charges)
        center["mapped_atoms"] = len(targets)
        center["expected_charge"] = float(block["expected_total_charge"])
        center["effective_heme_charges"] = heme_charges
        center["effective_cys_charges"] = cys_charges

    Path(path).write_text("".join(lines))
    reread = _reread_charges(path)
    for center in centers:
        block = params[center["selected_state"]]
        required = list(center["effective_heme_charges"].items()) + list(center["effective_cys_charges"].items())
        actual_total = 0.0
        for name, expected in required:
            logical_name = center["oxo_name"] if name == "OXO" else name
            group = "cys" if name in center["effective_cys_charges"] else "heme"
            matches = [a for a in reread if a["name"] == logical_name and (
                (group == "heme" and ((name == "OXO" and _is_oxo_identity(a, center)) or (a["resname"] == center["resname"] and a["chain"] == center["chain"] and a["resid"] == center["resid"]))) or
                (group == "cys" and a["chain"] == center["cys_chain"] and a["resid"] == center["cys_resid"])
            )]
            if len(matches) != 1 or abs(matches[0]["charge"] - float(expected)) > 0.0006:
                raise HemeStateError(f"Final PDBQT charge validation failed for {name}.")
            actual_total += matches[0]["charge"]
        tolerance = max(float(block.get("charge_tolerance", 0.001)), 0.00051 * len(required))
        if abs(actual_total - center["expected_charge"]) > tolerance:
            raise HemeStateError(f"Final charge sum {actual_total:.4f} does not match expected {center['expected_charge']:.4f}.")
        center["final_charge"] = actual_total
        log += [f"Mapped heme/Cys atoms: {center['mapped_atoms']}/{len(required)}", f"Charge sum: {actual_total:.4f}", "Final receptor PDBQT validation: PASS"]
    return {"centers": centers, "log": log}


def _is_oxo_identity(atom: dict, center: dict) -> bool:
    return (
        atom["name"] == center["oxo_name"]
        and atom["resname"] == center["oxo_resname"]
        and atom["chain"] == center["oxo_chain"]
        and atom["resid"] == center["oxo_resid"]
    )


def _find_final_oxo(atoms: list[dict], center: dict) -> list[dict]:
    matches = [atom for atom in atoms if _is_oxo_identity(atom, center)]
    irons = [
        atom for atom in atoms
        if atom["resname"] == center["resname"] and atom["chain"] == center["chain"]
        and atom["resid"] == center["resid"] and atom["name"] == center["fe_name"]
    ]
    if len(matches) != 1 or len(irons) != 1:
        return []
    return matches if FEO_MIN <= _distance(irons[0], matches[0]) <= FEO_MAX else []


def _normalize_hydrogen_names(lines, atoms, expected_charges, hydrogen_parents, label):
    expected_h = set(hydrogen_parents)
    actual_h = [a for a in atoms if a["element"] == "H" or a["name"].startswith("H")]
    if len(actual_h) > len(expected_h):
        raise HemeStateError(
            f"{label} has {len(actual_h)} hydrogens, more than the validated model's {len(expected_h)}."
        )
    heavy_by_name = {
        atom["name"]: atom for atom in atoms
        if atom["name"] in expected_charges and atom["name"] not in expected_h
    }
    for atom in actual_h:
        expected_name = HYDROGEN_ALIASES.get(atom["name"])
        if expected_name not in expected_h:
            continue
        parent = heavy_by_name.get(hydrogen_parents[expected_name])
        if parent is None or _distance(parent, atom) > 1.25:
            raise HemeStateError(
                f"{label} alias {atom['name']} -> {expected_name} failed parent-connectivity validation."
            )
        atom["name"] = expected_name
        line = lines[atom["index"]].rstrip("\n")
        lines[atom["index"]] = line[:12] + f"{expected_name:>4s}" + line[16:] + "\n"
    assigned = set()
    for expected_name in sorted(expected_h):
        exact = [a for a in actual_h if a["name"] == expected_name and a["index"] not in assigned]
        if exact:
            assigned.add(exact[0]["index"])
    folded = 0
    effective = {
        name: float(charge) for name, charge in expected_charges.items()
        if name not in expected_h
    }
    for atom in actual_h:
        if atom["index"] in assigned:
            effective[atom["name"]] = float(expected_charges[atom["name"]])
    for parent_name in sorted(set(hydrogen_parents.values())):
        parent = heavy_by_name.get(parent_name)
        if parent is None:
            raise HemeStateError(f"{label} hydrogen parent {parent_name} is missing.")
        wanted = sorted(
            name for name, mapped_parent in hydrogen_parents.items()
            if mapped_parent == parent_name and not any(a["name"] == name and a["index"] in assigned for a in actual_h)
        )
        candidates = sorted(
            (a for a in actual_h if a["index"] not in assigned and _distance(parent, a) <= 1.25),
            key=lambda atom: (atom["serial"], atom["name"]),
        )
        if len(candidates) > len(wanted):
            raise HemeStateError(
                f"{label} hydrogen mapping for {parent_name} found too many candidate hydrogens."
            )
        mapped_names = wanted[:len(candidates)]
        for atom, expected_name in zip(candidates, mapped_names):
            atom["name"] = expected_name
            line = lines[atom["index"]].rstrip("\n")
            lines[atom["index"]] = line[:12] + f"{expected_name:>4s}" + line[16:] + "\n"
            assigned.add(atom["index"])
            effective[expected_name] = float(expected_charges[expected_name])
        for missing_name in wanted[len(candidates):]:
            effective[parent_name] += float(expected_charges[missing_name])
            folded += 1
    return effective, folded


def _reread_charges(path: str) -> list[dict]:
    out = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if line[:6].strip() not in {"ATOM", "HETATM"}:
            continue
        atom = _atom(line)
        try:
            atom["charge"] = float(line[70:76])
        except ValueError:
            atom["charge"] = float(line.split()[-2])
        out.append(atom)
    return out
