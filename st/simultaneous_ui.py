"""Streamlit interface for Vina's simultaneous multi-ligand docking mode."""

from __future__ import annotations

import json
import re
from pathlib import Path


def _is_smiles(value: str) -> bool:
    """Distinguish typed SMILES from compound names without noisy RDKit logs."""
    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(value) is not None


def _parse_typed_ligand(line: str, index: int) -> tuple[str, str, None, str]:
    """Accept a name, a SMILES, or a SMILES followed by a display label."""
    line = line.strip()
    if line.lower().startswith("name:"):
        name = line[5:].strip()
        return "name", name, None, name
    if line.lower().startswith("smiles:"):
        line = line[7:].strip()
    first, _, rest = line.partition(" ")
    if rest and _is_smiles(first):
        return "smiles", first, None, rest.strip()
    if _is_smiles(line):
        return "smiles", line, None, f"lig_{index}"
    return "name", line, None, line


def _resolve_compound_name(name: str, lookup) -> tuple[str, dict]:
    """Use Single Dock's PubChem lookup and reject missing or invalid SMILES."""
    if not name or lookup is None:
        raise ValueError("PubChem name lookup is unavailable for this entry.")
    result = lookup(name)
    if not result.get("found"):
        raise ValueError(result.get("error") or f"'{name}' was not found in PubChem.")
    smiles = (result.get("smiles") or "").strip()
    if not smiles or not _is_smiles(smiles):
        raise ValueError(f"PubChem did not return a usable SMILES for '{name}'.")
    return smiles, {
        "input_name": name,
        "cid": result.get("cid"),
        "iupac": result.get("iupac", ""),
        "smiles": smiles,
        "url": result.get("url", ""),
    }


def _pose_blocks(pdbqt_text: str) -> list[str]:
    blocks = []
    current = []
    in_model = False
    for line in pdbqt_text.splitlines(keepends=True):
        if line.startswith("MODEL"):
            if current:
                blocks.append("".join(current))
            current = [line]
            in_model = True
        elif line.startswith("ENDMDL") and in_model:
            current.append(line)
            blocks.append("".join(current))
            current = []
            in_model = False
        elif in_model:
            current.append(line)
    if current:
        blocks.append("".join(current))
    return blocks or [pdbqt_text]


def _ligand_components(pose_text: str) -> list[dict]:
    """Read each ligand and its actual identity from a Vina joint pose."""
    ligands: list[dict] = []
    current: list[str] = []
    smiles = ""
    for line in pose_text.splitlines(keepends=True):
        if line.startswith("REMARK SMILES ") and not line.startswith("REMARK SMILES IDX"):
            smiles = line.removeprefix("REMARK SMILES ").strip()
        if line.startswith("ROOT") and not current:
            current = [line]
        elif current:
            current.append(line)
            if line.startswith("TORSDOF"):
                pdbqt = "".join(current)
                ligands.append({"pdbqt": pdbqt, "smiles": smiles,
                                "atom_count": sum(row.startswith(("ATOM  ", "HETATM"))
                                                  for row in current)})
                current = []
                smiles = ""
    return ligands


def _ligand_blocks(pose_text: str) -> list[str]:
    """Split one Vina joint pose at each ligand's TORSDOF record."""
    return [component["pdbqt"] for component in _ligand_components(pose_text)]


def _pdbqt_view_pdb(pdbqt: str) -> str:
    """Give 3Dmol an explicitly supported PDB model with correct atom elements."""
    ad4_elements = {"A": "C", "C": "C", "N": "N", "NA": "N", "O": "O",
                    "OA": "O", "S": "S", "SA": "S", "H": "H", "HD": "H"}
    rows = []
    for row in pdbqt.splitlines():
        if not row.startswith(("ATOM  ", "HETATM")):
            continue
        atom_type = row[76:].split()[-1] if row[76:].split() else ""
        element = ad4_elements.get(atom_type, atom_type.title())
        if not element:
            raise ValueError("Joint pose contains an atom without an AutoDock element type")
        rows.append(row[:66].ljust(76) + element.rjust(2))
    if not rows:
        raise ValueError("Joint pose contains no ligand atoms")
    return "\n".join(rows) + "\nEND\n"


def _input_smiles(path: str) -> str:
    for line in Path(path).read_text().splitlines():
        if line.startswith("REMARK SMILES ") and not line.startswith("REMARK SMILES IDX"):
            return line.removeprefix("REMARK SMILES ").strip()
    return ""


def _identity_errors(components: list[dict], labels: list[str], paths: list[str]) -> list[str]:
    errors = []
    if len(components) != len(labels):
        errors.append(f"Expected {len(labels)} ligand components, found {len(components)}.")
    for index, component in enumerate(components[:len(paths)]):
        try:
            expected_smiles = _input_smiles(paths[index])
        except OSError:
            errors.append(f"Prepared input for {labels[index]} is no longer available.")
            continue
        if expected_smiles and component["smiles"] != expected_smiles:
            errors.append(f"Component {index + 1} does not match prepared {labels[index]}.")
    return errors


def render_simultaneous_docking(st, core, receptor_section, vina_path,
                                 wdir: Path, show3d, viewer_bg, compound_lookup=None):
    """Prepare 2–4 ligands and show the joint Vina poses without per-ligand scores."""
    wdir.mkdir(parents=True, exist_ok=True)
    receptor_section(pfx="m_", wdir=wdir, step_label="Step M1 of M2")
    st.markdown("### Co-Dock: dock ligands together")
    st.caption("Co-Dock places 2–4 ligands together in one pocket. The score is for "
               "their combined pose. Each pose has one copy of every input ligand; "
               "Co-Dock does not enumerate all pairings. **Hint:** Use Batch Dock to "
               "compare ligands one by one.")
    mode = st.radio(
        "Ligand input",
        ["Compound name", "SMILES list (text)", "Upload .smi file",
         "Upload structure files (.pdb/.mol2)"],
        horizontal=True, key="m_input_mode",
    )
    if mode == "Compound name":
        st.caption("Search PubChem and add 2–4 compounds to the editable SMILES list below.")
        name_col, search_col = st.columns([5, 1])
        with name_col:
            query = st.text_input("Compound name", placeholder="e.g. aspirin, baicalein…",
                                  key="m_compound_name").strip()
        with search_col:
            st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
            search = st.button("Search", key="m_compound_search")
        if search:
            st.session_state.pop("m_compound_result", None)
            if not query:
                st.warning("Enter a compound name before searching.")
            else:
                try:
                    with st.spinner(f"Searching PubChem for '{query}'…"):
                        match_result = compound_lookup(query) if compound_lookup else None
                        smiles, match = _resolve_compound_name(
                            query, lambda _name: match_result)
                    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", query).strip("._-")
                    if not label:
                        raise ValueError("The compound name cannot be used as a ligand label.")
                    lines = st.session_state.get("m_compound_smiles_text", "").splitlines()
                    lines = [line for line in lines if not (
                        line.strip() and not line.lstrip().startswith("#")
                        and len(line.split(None, 1)) > 1
                        and line.split(None, 1)[1].strip() == label
                    )]
                    if len([line for line in lines if line.strip() and not line.lstrip().startswith("#")]) >= 4:
                        raise ValueError("Co-Dock accepts at most four ligands; remove one from the list first.")
                    lines.append(f"{smiles} {label}")
                    st.session_state["m_compound_smiles_text"] = "\n".join(lines)
                    st.session_state["m_compound_result"] = {**match_result, "input_name": query}
                    st.session_state.setdefault("m_compound_catalog", {})[label] = match
                except Exception as exc:
                    st.error(f"PubChem search failed for {query}: {exc}")
        searched = st.session_state.get("m_compound_result")
        if searched and searched.get("input_name") == query:
            info_col, image_col = st.columns([3, 1])
            with info_col:
                st.markdown(f"**{searched.get('iupac') or query}**  \n"
                            f"{searched.get('formula', '')} · "
                            f"[PubChem CID {searched.get('cid')}]({searched.get('url')})")
                st.caption("Added to the SMILES list. Search another name to add it.")
            with image_col:
                if searched.get("img_url"):
                    st.image(searched["img_url"], width=140)
        st.text_area("One `SMILES [name]` per line", height=300,
                     placeholder="Search a compound above to add its SMILES here.",
                     key="m_compound_smiles_text")
    elif mode == "SMILES list (text)":
        st.text_area("One `SMILES [name]` per line", height=300, key="m_smiles_text",
                     placeholder="CC(=O)Oc1ccccc1C(=O)O Aspirin\n"
                                 "O=c1cc(-c2ccccc2)oc2cc(O)c(O)c(O)c12 Baicalein")
    elif mode == "Upload .smi file":
        st.file_uploader("Upload .smi file", type=["smi", "txt"], key="m_smi_upload")
    else:
        st.file_uploader("Upload 2–4 ligand files", type=["sdf", "mol2", "pdb", "pdbqt"],
                         accept_multiple_files=True, key="m_structure_uploads")
    col1, col2, col3 = st.columns(3)
    with col1:
        ph = st.number_input("pH", 0.0, 14.0, 7.4, 0.1, key="m_ph")
        neutral = st.checkbox("Keep input charge state", key="m_neutral")
    with col2:
        exhaustiveness = st.slider("Exhaustiveness", 8, 64, 32, 4, key="m_exh")
        n_modes = st.slider("Combined poses", 1, 20, 10, key="m_modes")
    with col3:
        energy_range = st.slider("Energy range (kcal/mol)", 1, 10, 3, key="m_range")
        seed = st.number_input("Random seed (0 = random)", 0, value=0, key="m_seed")

    ready = st.session_state.get("m_receptor_done", False)
    if st.button("▶ Run Co-Dock", type="primary", key="m_run", disabled=not ready):
        st.session_state["m_joint_done"] = False
        st.session_state.pop("m_joint_result", None)
        entries = []
        if mode == "Upload structure files (.pdb/.mol2)":
            for item in st.session_state.get("m_structure_uploads") or []:
                entries.append(("file", item.name, item.getvalue(), Path(item.name).stem))
        else:
            if mode == "Upload .smi file":
                item = st.session_state.get("m_smi_upload")
                lines = item.getvalue().decode("utf-8").splitlines() if item else []
            else:
                text_key = ("m_compound_smiles_text" if mode == "Compound name"
                            else "m_smiles_text")
                lines = st.session_state.get(text_key, "").splitlines()
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split(None, 1)
                    if not _is_smiles(parts[0]):
                        st.error(f"Invalid SMILES on line {len(entries) + 1}: {parts[0]}")
                        return
                    entries.append(("smiles", parts[0], None,
                                    parts[1] if len(parts) > 1 else f"lig_{len(entries)+1}"))
        if not 2 <= len(entries) <= 4:
            st.error("Provide 2–4 ligands for one simultaneous search.")
            return
        if not vina_path:
            st.error("Vina is unavailable; check the installation step.")
            return
        paths = []
        labels = []
        resolved_compounds = []
        compound_catalog = (st.session_state.get("m_compound_catalog", {})
                            if mode == "Compound name" else {})
        ligand_count = len(entries)
        progress = st.progress(0, text=f"Preparing ligands 0/{ligand_count}…")
        for idx, (kind, value, data, label) in enumerate(entries, 1):
            progress.progress((idx - 1) / (ligand_count + 1),
                              text=f"Preparing {label} ({idx}/{ligand_count})…")
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._-") or f"lig_{idx}"
            stem = f"{idx:02d}_{safe}"
            lig_dir = wdir / "ligands" / stem
            lig_dir.mkdir(parents=True, exist_ok=True)
            if kind == "file":
                source = lig_dir / (stem + Path(value).suffix.lower())
                source.write_bytes(data)
                prep = core.prepare_ligand_from_file(str(source), stem, lig_dir)
            else:
                if label in compound_catalog and compound_catalog[label]["smiles"] == value:
                    resolved_compounds.append(compound_catalog[label])
                if kind == "name":
                    try:
                        with st.spinner(f"Looking up '{value}' in PubChem…"):
                            value, match = _resolve_compound_name(value, compound_lookup)
                    except Exception as exc:
                        progress.empty()
                        st.error(f"Could not resolve {label} from PubChem: {exc}")
                        return
                    resolved_compounds.append(match)
                prep = core.prepare_ligand(value, stem, ph, lig_dir,
                                           mode="neutral" if neutral else "pkanet")
            if not prep.get("success"):
                progress.empty()
                st.error(f"Could not prepare {label}: {prep.get('error', 'unknown error')}")
                return
            paths.append(str(Path(prep["pdbqt"]).resolve()))
            labels.append(label)
            progress.progress(idx / (ligand_count + 1),
                              text=f"Prepared {idx}/{ligand_count} ligands")
        progress.progress(ligand_count / (ligand_count + 1),
                          text="Running one joint Vina search… (no live Vina percentage)")
        try:
            result = core.run_vina_multi(
                receptor_pdbqt=st.session_state["m_receptor_pdbqt"],
                ligand_pdbqts=paths,
                config_txt=st.session_state["m_config_txt"],
                vina_path=vina_path,
                exhaustiveness=exhaustiveness,
                n_modes=n_modes,
                energy_range=energy_range,
                wdir=wdir,
                out_name="joint",
                seed=int(seed) or None,
            )
        except Exception:
            progress.empty()
            raise
        if not result["success"]:
            progress.progress(1.0, text="Joint Vina search finished with an error")
            st.error(f"Simultaneous docking failed: {result['error']}")
            st.code(result.get("log", ""))
            return
        progress.progress(1.0, text="Done — joint docking complete")
        manifest = {"score_scope": "joint_ligand_complex", "ligand_labels": labels,
                    "ligand_pdbqts": paths, "scores": result["scores"],
                    "top_score": result["top_score"], "out_pdbqt": result["out_pdbqt"],
                    "pubchem_matches": resolved_compounds}
        (wdir / "joint_results.json").write_text(json.dumps(manifest, indent=2) + "\n")
        st.session_state["m_joint_result"] = {**result, "labels": labels,
                                               "pubchem_matches": resolved_compounds}
        st.session_state["m_joint_done"] = True

    result = st.session_state.get("m_joint_result") if st.session_state.get("m_joint_done") else None
    if result:
        st.markdown("### Co-Dock results")
        st.write("Ligands: " + ", ".join(result["labels"]))
        for match in result.get("pubchem_matches", []):
            st.caption(f"PubChem match: {match['input_name']} → CID {match['cid']} "
                       f"({match['iupac']})")
        st.info("All affinities below are for the complete ligand set. They cannot "
                "be assigned to an individual ligand or compared with Batch Dock scores.")
        if result["scores"]:
            st.dataframe(result["scores"], use_container_width=True)
        else:
            st.warning("Vina output was written, but no joint score could be parsed.")
        output = Path(result["out_pdbqt"])
        if output.is_file():
            pdbqt = output.read_text()
            st.download_button("Download combined poses (.pdbqt)", pdbqt,
                               file_name="joint_out.pdbqt", mime="text/plain", key="m_download_pdbqt")
            blocks = _pose_blocks(pdbqt)
            pose_idx = st.selectbox("View combined pose", range(len(blocks)),
                                    format_func=lambda i: f"Combined pose {i+1}", key="m_pose_idx")
            st.download_button("Download selected combined pose (.pdbqt)", blocks[pose_idx],
                               file_name=f"joint_pose{pose_idx+1}.pdbqt",
                               mime="text/plain", key="m_download_pose")
            components = _ligand_components(blocks[pose_idx])
            expected_paths = result.get("ligand_pdbqts", [])
            identity_errors = _identity_errors(components, result["labels"], expected_paths)
            if identity_errors:
                st.error("The combined pose does not match the prepared ligand inputs: "
                         + " ".join(identity_errors))
            else:
                st.caption("This pose contains one copy of each ligand: " + " + ".join(
                    f"{label} ({component['atom_count']} atoms)"
                    for label, component in zip(result["labels"], components)))
                with st.expander("Check ligand identities in this pose"):
                    st.dataframe([
                        {"Ligand": label, "SMILES in Vina output": component["smiles"],
                         "PDBQT atoms": component["atom_count"]}
                        for label, component in zip(result["labels"], components)
                    ], use_container_width=True)
            try:
                if identity_errors:
                    raise ValueError("ligand identities in the joint pose do not match the inputs")
                import py3Dmol
                view = py3Dmol.view(width="100%", height=480)
                view.setBackgroundColor(viewer_bg())
                rec_pdb = st.session_state.get("m_receptor_fh")
                model_offset = 0
                if rec_pdb and Path(rec_pdb).is_file():
                    view.addModel(Path(rec_pdb).read_text(), "pdb")
                    view.setStyle({"model": 0}, {"cartoon": {"color": "lightgray", "opacity": 0.65}})
                    model_offset = 1
                colors = ("cyanCarbon", "magentaCarbon", "greenCarbon", "orangeCarbon")
                st.caption("3D colors: " + ", ".join(
                    f"{label} ({color.replace('Carbon', '')})"
                    for label, color in zip(result["labels"], colors)))
                for index, component in enumerate(components):
                    model_index = model_offset + index
                    view.addModel(_pdbqt_view_pdb(component["pdbqt"]), "pdb")
                    view.setStyle({"model": model_index},
                                  {"stick": {"colorscheme": colors[index], "radius": 0.22}})
                view.zoomTo({"model": list(range(model_offset, model_offset + len(components)))})
                show3d(view, height=480)
            except Exception as exc:
                st.caption(f"3D preview unavailable: {exc}")
        report = wdir / "joint_results.json"
        if report.is_file():
            st.download_button("Download joint results (.json)", report.read_bytes(),
                               file_name="joint_results.json", mime="application/json",
                               key="m_download_json")
        with st.expander("Vina log"):
            st.code(result.get("log", ""))
