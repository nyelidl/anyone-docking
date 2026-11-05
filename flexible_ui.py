"""Small shared Streamlit controls for optional Vina receptor flexibility."""
from pathlib import Path
import io
import zipfile

from flexible_docking import candidate_residues, metadata


def controls(st, source, pfx, *, online=False):
    with st.expander("Flexible docking: receptor flexibility — optional", expanded=False):
        enabled = st.checkbox("Allow selected protein side chains to move", key=pfx + "flex_enabled")
        selected = []
        if enabled:
            st.caption("Vina only. The backbone stays fixed. More movable bonds increase search cost; no runtime estimate is implied.")
            if source and Path(source).is_file():
                try:
                    import hashlib
                    source_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
                    if st.session_state.get(pfx + "flex_source_hash") != source_hash:
                        st.session_state[pfx + "flex_selection"] = []
                    st.session_state[pfx + "flex_source_hash"] = source_hash
                    rows = candidate_residues(source)
                    rows = [r for r in rows if r["complete"]]
                    labels = {r["key"]: f"{r['key']} {r['resname']}" for r in rows}
                    # Restrict the displayed list around the actual selected reference when available.
                    cutoff = st.slider("Candidate distance from reference ligand (Å)", 3.0, 10.0, 5.0, 0.5, key=pfx + "flex_cutoff")
                    reference = None
                    points = []
                    from flexible_docking import _atoms, _xyz, _dist
                    key = st.session_state.get(pfx + "reference_hetatm_key", "")
                    if key and "|" in key:
                        name, chain, number = key.split("|")
                        chain = "" if chain == "_" else chain
                        points = [_xyz(l) for l in _atoms(source) if l[17:20].strip() == name and
                                  l[21:22].strip() == chain and l[22:26].strip() == number and
                                  l[76:78].strip() != "H"]
                    if pfx == "r_":
                        reference = (st.session_state.get(pfx + "inspection") or {}).get("reference_path")
                    if reference and Path(reference).is_file():
                        points = [_xyz(l) for l in _atoms(reference) if l[76:78].strip() != "H"]
                    if points:
                        nearby = {r["key"] for r in rows if any(_dist(a,b) <= cutoff for a in r["coords"] for b in points)}
                        old = set(st.session_state.get(pfx + "flex_selection", []))
                        labels = {k:v for k,v in labels.items() if k in nearby or k in old}
                    else:
                        st.caption("No extracted reference ligand yet: all complete supported residues are listed. Select by chain and residue number.")
                    old = st.session_state.get(pfx + "flex_selection", [])
                    if any(k not in labels for k in old):
                        st.session_state[pfx + "flex_selection"] = [k for k in old if k in labels]
                    # Scope the yellow tags to this selector; other ligand tags keep their styling.
                    st.markdown(
                        f"""<style>
                        .st-key-{pfx}flex_selection [data-baseweb="tag"] {{
                            background-color: #FFD600 !important;
                            color: #1A1A1A !important;
                        }}
                        .st-key-{pfx}flex_selection [data-baseweb="tag"] span,
                        .st-key-{pfx}flex_selection [data-baseweb="tag"] svg {{
                            color: #1A1A1A !important;
                            fill: currentColor;
                        }}
                        </style>""",
                        unsafe_allow_html=True,
                    )
                    selected = st.multiselect("Flexible residues", list(labels), format_func=labels.get, key=pfx + "flex_selection")
                    st.caption("Selected flexible residues are shown as yellow sticks in the receptor 3D view.")
                except Exception as exc:
                    st.error("Cannot list flexible residues: " + str(exc))
            else:
                st.info("Load a receptor structure first.")
            if online and len(selected) > 3:
                st.error("Online allows at most 3 flexible residues. Use Local or CLI for larger searches.")
            if not selected:
                st.warning("Select at least one residue before preparing a flexible receptor.")
            st.caption("Missing side-chain atoms, disulfides and metal-coordinating side chains are checked during preparation.")
        signature = (enabled, tuple(selected))
        previous = st.session_state.get(pfx + "flex_signature", (False, ()))
        if signature != previous:
            st.session_state[pfx + "receptor_done"] = False
            st.session_state.pop(pfx + "prepared", None)
            st.session_state.pop(pfx + "result", None)
            st.session_state[{"b_": "b_batch_done", "m_": "m_joint_done"}.get(pfx, "docking_done")] = False
        st.session_state[pfx + "flex_signature"] = signature
        return selected


def require_selection(st, pfx, *, online=False):
    selected = st.session_state.get(pfx + "flex_selection", []) if st.session_state.get(pfx + "flex_enabled") else []
    if st.session_state.get(pfx + "flex_enabled") and not selected:
        raise ValueError("Select at least one flexible residue")
    if online and len(selected) > 3:
        raise ValueError("Online allows at most 3 flexible residues")
    return selected


def download(st, receptor_pdbqt, pfx, *, result_root=None):
    """Include input trees plus all completed pose receptor/complex files."""
    try:
        data = metadata(receptor_pdbqt)
        if not data:
            return
        root = Path(receptor_pdbqt).resolve().parent
        files = {Path(data[k]) for k in ("rigid_pdbqt", "flex_pdbqt", "receptor_pdb", "full_pdbqt")}
        files.add(Path(str(receptor_pdbqt) + ".flex.json"))
        # ACD run outputs live in this work directory or its descendants.
        search_root = Path(result_root).resolve() if result_root else root
        for pattern in ("*_pose*_receptor.pdb", "*_pose*_complex.pdb", "*_flex_results.json", "*_out.pdbqt"):
            files.update(search_root.rglob(pattern))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for f in sorted(files):
                if f.is_file():
                    archive.write(f, str(f.relative_to(search_root)) if f.is_relative_to(search_root) else f.name)
        st.download_button("Download flexible receptor & pose complexes", buffer.getvalue(),
                           file_name="flexible_docking_results.zip", mime="application/zip", key=pfx + "flex_download")
    except Exception as exc:
        st.warning("Flexible result download unavailable: " + str(exc))
