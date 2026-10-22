"""Streamlit interface for Vina's simultaneous multi-ligand docking mode."""

from __future__ import annotations

import json
import re
from pathlib import Path


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


def _ligand_blocks(pose_text: str) -> list[str]:
    """Split one Vina joint pose at each ligand's TORSDOF record for viewing."""
    ligands = []
    current = []
    for line in pose_text.splitlines(keepends=True):
        if line.startswith("ROOT") and not current:
            current = [line]
        elif current:
            current.append(line)
            if line.startswith("TORSDOF"):
                ligands.append("".join(current))
                current = []
    return ligands


def render_simultaneous_docking(st, core, receptor_section, vina_path,
                                 wdir: Path, show3d, viewer_bg):
    """Prepare 2–4 ligands and show the joint Vina poses without per-ligand scores."""
    wdir.mkdir(parents=True, exist_ok=True)
    receptor_section(pfx="m_", wdir=wdir, step_label="Step M1 of M2")
    st.markdown("### Co-Dock: dock ligands together")
    st.caption("Co-Dock places 2–4 ligands together in one pocket. The score is for "
               "their combined pose. **Hint:** Use Batch Dock to compare ligands one by one.")
    mode = st.radio("Ligand input", ["SMILES list", "Upload .smi", "Upload structures"],
                    horizontal=True, key="m_input_mode")
    if mode == "SMILES list":
        st.text_area("One SMILES and label per line", key="m_smiles_text",
                     placeholder="CCO ethanol\nCCN ethylamine")
    elif mode == "Upload .smi":
        st.file_uploader("Upload .smi", type=["smi", "txt"], key="m_smi_upload")
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
        entries = []
        if mode == "Upload structures":
            for item in st.session_state.get("m_structure_uploads") or []:
                entries.append(("file", item.name, item.getvalue(), Path(item.name).stem))
        else:
            if mode == "Upload .smi":
                item = st.session_state.get("m_smi_upload")
                lines = item.getvalue().decode("utf-8").splitlines() if item else []
            else:
                lines = st.session_state.get("m_smiles_text", "").splitlines()
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split(None, 1)
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
        for idx, (kind, value, data, label) in enumerate(entries, 1):
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._-") or f"lig_{idx}"
            stem = f"{idx:02d}_{safe}"
            lig_dir = wdir / "ligands" / stem
            lig_dir.mkdir(parents=True, exist_ok=True)
            if kind == "file":
                source = lig_dir / (stem + Path(value).suffix.lower())
                source.write_bytes(data)
                prep = core.prepare_ligand_from_file(str(source), stem, lig_dir)
            else:
                prep = core.prepare_ligand(value, stem, ph, lig_dir,
                                           mode="neutral" if neutral else "pkanet")
            if not prep.get("success"):
                st.error(f"Could not prepare {label}: {prep.get('error', 'unknown error')}")
                return
            paths.append(str(Path(prep["pdbqt"]).resolve()))
            labels.append(label)
        with st.spinner("Running one joint Vina search…"):
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
        if not result["success"]:
            st.error(f"Simultaneous docking failed: {result['error']}")
            st.code(result.get("log", ""))
            return
        manifest = {"score_scope": "joint_ligand_complex", "ligand_labels": labels,
                    "ligand_pdbqts": paths, "scores": result["scores"],
                    "top_score": result["top_score"], "out_pdbqt": result["out_pdbqt"]}
        (wdir / "joint_results.json").write_text(json.dumps(manifest, indent=2) + "\n")
        st.session_state["m_joint_result"] = {**result, "labels": labels}
        st.session_state["m_joint_done"] = True

    result = st.session_state.get("m_joint_result") if st.session_state.get("m_joint_done") else None
    if result:
        st.markdown("### Co-Dock results")
        st.write("Ligands: " + ", ".join(result["labels"]))
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
            try:
                import py3Dmol
                view = py3Dmol.view(width="100%", height=480)
                view.setBackgroundColor(viewer_bg())
                rec_pdb = st.session_state.get("m_receptor_fh")
                model_offset = 0
                if rec_pdb and Path(rec_pdb).is_file():
                    view.addModel(Path(rec_pdb).read_text(), "pdb")
                    view.setStyle({"model": 0}, {"cartoon": {"color": "lightgray", "opacity": 0.65}})
                    model_offset = 1
                ligand_models = _ligand_blocks(blocks[pose_idx])
                if len(ligand_models) != len(result["labels"]):
                    raise ValueError("Joint pose could not be separated into its ligand components")
                colors = ("cyanCarbon", "magentaCarbon", "greenCarbon", "orangeCarbon")
                st.caption("3D colors: " + ", ".join(
                    f"{label} ({color.replace('Carbon', '')})"
                    for label, color in zip(result["labels"], colors)))
                for index, ligand_text in enumerate(ligand_models):
                    model_index = model_offset + index
                    view.addModel(ligand_text, "pdbqt")
                    view.setStyle({"model": model_index},
                                  {"stick": {"colorscheme": colors[index], "radius": 0.22}})
                view.zoomTo({"model": model_offset})
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
