"""One prepared receptor/box shared by the web docking modes.

Aliases preserve existing docking/result code. This module never prepares a
second receptor. Redock eligibility is assessed against the actual shared input.
"""
import copy
import hashlib
import json
from pathlib import Path

PREFIX = "shared_"
FIELDS = ("raw_pdb_path", "receptor_fh", "receptor_pdbqt", "box_pdb", "config_txt",
          "cx", "cy", "cz", "box_sx", "box_sy", "box_sz", "ligand_pdb_path",
          "cocrystal_ligand_id", "pdb_token", "receptor_log", "reference_hetatm_key",
          "hetatm_policy", "heme_states", "flex_enabled", "flex_selection")


def _reset_results(state):
    for key in ("docking_done", "redock_done", "b_batch_done", "m_joint_done"):
        state[key] = False
    for key in ("redock_result", "b_redock_result", "r_result", "m_joint_result",
                "b_batch_results", "confirmed_ref_score", "confirmed_ref_pose",
                "confirmed_ref_name", "confirmed_ref_rmsd", "b_confirmed_ref_score",
                "b_confirmed_ref_pose", "b_confirmed_ref_name", "b_confirmed_ref_rmsd"):
        state.pop(key, None)
    # Pose/diagram reference caches must not outlive their receptor.
    for key in list(state):
        if key.startswith(("pv_", "b_pv", "rtf_", "_poseview_", "_b_pose_synced", "_pose_synced")):
            state.pop(key, None)
    for key in ("redock_score", "b_redock_score", "score_df", "pose_mols", "output_pdbqt",
                "output_sdf", "output_pv_sdf", "b_plot_png"):
        state[key] = None


def _disable(state):
    state.pop("shared_active_results_dir", None)
    for prefix in ("", "b_", "m_"):
        state[prefix + "receptor_done"] = False
        for field in FIELDS:
            state.pop(prefix + field, None)
    state.pop("r_prepared", None)
    state.pop("r_inspection", None)
    state.pop("r_ligand_choice", None)
    _reset_results(state)


def _signature(state, prepared):
    paths = [prepared[k] for k in ("rec_fh", "rec_pdbqt", "config_txt")]
    paths += [state[PREFIX + "raw_pdb_path"]]
    sidecar = Path(str(prepared["rec_pdbqt"]) + ".flex.json")
    if sidecar.is_file():
        paths.append(str(sidecar))
        data = json.loads(sidecar.read_text())
        paths += [data[k] for k in ("rigid_pdbqt", "flex_pdbqt")]
    payload = {"revision": state.get(PREFIX + "preparation_revision", 0),
               "reference": state.get(PREFIX + "reference_hetatm_key", ""),
               "inputs": [(str(Path(p).resolve()), hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in paths]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _inspect_redock(core, raw, reference_key, prepared, directory):
    """Inspect the selected crystal ligand; reuse the full shared receptor."""
    from redock_workflow import inspect_redock_structure
    from pocket_completeness import assess_missing_residues, save_report
    inspection = inspect_redock_structure(core, raw, directory, ligand_key=reference_key)
    if not inspection.get("ready"):
        return inspection, None
    prepared_chains = {line[21:22].strip() for line in Path(prepared["rec_fh"]).read_text().splitlines()
                       if line.startswith("ATOM  ")}
    missing = set(inspection.get("selected_chains", [])) - prepared_chains
    if missing:
        inspection.update(ready=False, message="Redock is unavailable: a contacting chain is absent from the shared prepared receptor. Prepare the complete receptor again.")
        return inspection, None
    report = assess_missing_residues(raw, inspection["reference_path"], prepared["rec_fh"])
    save_report(report, directory)
    inspection["pocket_report"] = report
    if report.get("blocked"):
        inspection.update(ready=False, message="Redock is blocked: missing residues or atoms affect the selected binding site. Rebuild and validate the region, then prepare the receptor again.")
        return inspection, None
    redock_prepared = dict(prepared, pocket_report=report)
    inspection["message"] = "Ready: redocking will reuse the shared receptor and docking box."
    return inspection, redock_prepared


def sync(state, core, wdir):
    """Publish one receptor to every mode; invalidate all runs when it changes."""
    prepared = state.get(PREFIX + "prepared_result")
    if not state.get(PREFIX + "receptor_done") or not prepared:
        _disable(state)
        state.pop("shared_active_signature", None)
        return {"ready": False, "message": "Prepare the receptor above to enable docking."}
    try:
        from flexible_docking import metadata, parse_residues
        selected = parse_residues(state.get(PREFIX + "flex_selection", [])) if state.get(PREFIX + "flex_enabled") else []
        if state.get(PREFIX + "flex_enabled") and not selected:
            raise ValueError("Select flexible residues and prepare the receptor again.")
        flex = metadata(prepared["rec_pdbqt"])
        actual = parse_residues(flex.get("residues", [])) if flex else []
        if set(selected) != set(actual):
            raise ValueError("Flexible residue selection does not match the prepared receptor. Click Prepare Receptor again; rigid fallback is blocked.")
        if selected and not flex:
            raise ValueError("Flexible receptor files are missing. Click Prepare Receptor again; rigid fallback is blocked.")
        signature = _signature(state, prepared)
    except (OSError, ValueError, KeyError) as exc:
        state[PREFIX + "receptor_done"] = False
        _disable(state)
        return {"ready": False, "message": "Shared receptor files are unavailable; prepare again. " + str(exc)}
    revision = state.get(PREFIX + "preparation_revision", 0)
    previous = state.get("shared_active_signature")
    if previous and previous != signature and state.get("shared_active_revision") == revision:
        state[PREFIX + "receptor_done"] = False
        _disable(state)
        return {"ready": False, "message": "Shared receptor files changed after preparation. Prepare again before docking."}
    changed = previous != signature
    if changed:
        _reset_results(state)
    for prefix in ("", "b_", "m_"):
        for field in FIELDS:
            state[prefix + field] = copy.deepcopy(state.get(PREFIX + field))
        state[prefix + "receptor_done"] = True
    if changed:
        state.pop("r_result", None)
        state.pop("r_prepared", None)
        reference_key = state.get(PREFIX + "reference_hetatm_key", "")
        if reference_key:
            try:
                inspection, redock_prepared = _inspect_redock(
                    core, state[PREFIX + "raw_pdb_path"], reference_key, prepared,
                    Path(wdir) / "shared_redock_inspection")
            except Exception as exc:
                inspection = {"ready": False, "message": "Redock inspection failed: " + str(exc)}
                redock_prepared = None
        else:
            inspection = {"ready": False, "message": "Redock requires a bound reference ligand. Select one in the shared receptor panel and prepare again. Other modes can use the prepared receptor."}
            redock_prepared = None
        state["r_inspection"] = inspection
        state["r_ligand_choice"] = inspection.get("selected_ligand_key")
        if redock_prepared:
            state["r_prepared"] = redock_prepared
    state["shared_active_signature"] = signature
    state["shared_active_revision"] = revision
    results = Path(wdir) / "docking_runs" / signature[:20]
    results.mkdir(parents=True, exist_ok=True)
    state["shared_active_results_dir"] = str(results)
    mode_message = ("Flexible docking ready — selected side chains: " + ", ".join(actual) + ". "
                    if flex else "Rigid docking ready — receptor side chains are fixed. ")
    return {"ready": True, "message": mode_message + "All docking modes use this prepared receptor and box.",
            "redock_ready": bool(state.get("r_prepared")), "results_dir": str(results)}


def show_redock_status(st):
    inspection = st.session_state.get("r_inspection") or {}
    if inspection.get("ready") and st.session_state.get("r_prepared"):
        st.success("Using the shared receptor, box, and selected crystal reference ligand.")
    else:
        st.info(inspection.get("message", "Prepare the shared receptor above first."))
    if inspection.get("contacts"):
        with st.expander("Reference ligand contacts and missing-residue assessment", expanded=False):
            st.dataframe(inspection["contacts"], hide_index=True)
            if inspection.get("pocket_report"):
                from pocket_completeness import show_report
                show_report(st, inspection["pocket_report"])
