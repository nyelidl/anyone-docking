#!/usr/bin/env python3
"""
api.py — FastAPI wrapper for Anyone Can Dock.

Revision 0.1.2 changes (marked inline with `# [REV]`):
- Escape user-controlled filename/URL in the /view HTML page (reflected-XSS fix).
- Guarantee unique output file stems when ligands share the same name.
- Build the result zip BEFORE flipping job status to "completed" (no download race).
- _make_zip no longer embeds itself or the transient status.json.
- _compute_blind_box now also parses mmCIF (RCSB fallback can return .cif).
- RMSD-over-poses loop guards against empty SDF and non-dict score rows.

Earlier fixes (retained):
- Defines ligand_pdb_path before RMSD calculation.
- RMSD calculation only runs for true redocking cases.
- Compact persistent job status.
- PoseView / 2D interaction output support.
"""

from __future__ import annotations

import csv
import html
import json
import os
import shutil
import threading
import traceback
import uuid
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from anyonecandock.core import (
    check_obabel,
    fix_sdf_bond_orders,
    get_vina_binary,
    load_mols_from_sdf,
    prepare_ligand,
    prepare_receptor,
    run_vina,
    scan_hetatm_residues,
)

try:
    from anyonecandock.core import (
        calc_rmsd_heavy,
        write_single_pose,
        call_poseview_v1,
        call_poseview2_ref,
        draw_interaction_diagram,
        svg_to_png,
    )
except Exception:
    calc_rmsd_heavy = None
    write_single_pose = None
    call_poseview_v1 = None
    call_poseview2_ref = None
    draw_interaction_diagram = None
    svg_to_png = None


API_TITLE = "Anyone Can Dock API"
API_VERSION = "0.1.3"  # [REV] env-configurable resource limits
BASE_WORKDIR = Path(os.getenv("ACD_API_WORKDIR", "/tmp/anyone_can_dock_api")).resolve()
BASE_WORKDIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("DOCKGPT_API_KEY", "").strip()
JOBS: Dict[str, Dict[str, Any]] = {}


# ----------------------------------------------------------------------------
# [REV] Runtime resource limits, configurable via Render environment variables.
# A value of 0 / empty disables that particular limit (backward compatible).
# ----------------------------------------------------------------------------
def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(float(os.getenv(name, "") or default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float = 0.0) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


MAX_JOBS_PER_DAY = _env_int("MAX_JOBS_PER_DAY", 0)
MAX_LIGANDS_PER_JOB = _env_int("MAX_LIGANDS_PER_JOB", 0)
MAX_EXHAUSTIVENESS = _env_int("MAX_EXHAUSTIVENESS", 0)
MAX_NUM_MODES = _env_int("MAX_NUM_MODES", 0)
MAX_ENERGY_RANGE = _env_int("MAX_ENERGY_RANGE", 0)
MAX_BOX_SIZE = _env_float("MAX_BOX_SIZE", 0.0)

_USAGE_LOCK = threading.Lock()
_USAGE_FILE = BASE_WORKDIR / "usage_counter.json"


def _active_limits() -> Dict[str, Any]:
    return {
        "max_jobs_per_day": MAX_JOBS_PER_DAY or None,
        "max_ligands_per_job": MAX_LIGANDS_PER_JOB or None,
        "max_exhaustiveness": MAX_EXHAUSTIVENESS or None,
        "max_num_modes": MAX_NUM_MODES or None,
        "max_energy_range": MAX_ENERGY_RANGE or None,
        "max_box_size": MAX_BOX_SIZE or None,
    }


def _clamp_box_size(size_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if MAX_BOX_SIZE and MAX_BOX_SIZE > 0:
        return tuple(min(float(s), MAX_BOX_SIZE) for s in size_xyz)  # type: ignore[return-value]
    return tuple(float(s) for s in size_xyz)  # type: ignore[return-value]


def _daily_jobs_used() -> int:
    today = date.today().isoformat()
    if _USAGE_FILE.exists():
        try:
            loaded = json.loads(_USAGE_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and loaded.get("date") == today:
                return int(loaded.get("count", 0))
        except Exception:
            pass
    return 0


def _reserve_daily_job_slot() -> None:
    """Best-effort per-day submission quota.

    The counter lives on disk under BASE_WORKDIR. On an ephemeral filesystem
    (e.g. Render free /tmp) it resets on restart/redeploy, which is acceptable
    as a soft guard. Raises HTTP 429 once the daily cap is reached.
    """
    if MAX_JOBS_PER_DAY <= 0:
        return

    today = date.today().isoformat()
    with _USAGE_LOCK:
        count = 0
        if _USAGE_FILE.exists():
            try:
                loaded = json.loads(_USAGE_FILE.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and loaded.get("date") == today:
                    count = int(loaded.get("count", 0))
            except Exception:
                count = 0

        if count >= MAX_JOBS_PER_DAY:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Daily docking limit reached ({MAX_JOBS_PER_DAY} jobs/day). "
                    "Please try again tomorrow."
                ),
            )

        try:
            _USAGE_FILE.write_text(
                json.dumps({"date": today, "count": count + 1}), encoding="utf-8"
            )
        except Exception:
            pass


def _apply_runtime_caps(req: "DockRequest") -> List[str]:
    """Clamp cost-driving knobs down to the configured maxima. Returns notes."""
    notes: List[str] = []

    if MAX_EXHAUSTIVENESS and req.exhaustiveness > MAX_EXHAUSTIVENESS:
        notes.append(f"exhaustiveness {req.exhaustiveness}->{MAX_EXHAUSTIVENESS}")
        req.exhaustiveness = MAX_EXHAUSTIVENESS

    if MAX_NUM_MODES and req.num_modes > MAX_NUM_MODES:
        notes.append(f"num_modes {req.num_modes}->{MAX_NUM_MODES}")
        req.num_modes = MAX_NUM_MODES

    if MAX_ENERGY_RANGE and req.energy_range > MAX_ENERGY_RANGE:
        notes.append(f"energy_range {req.energy_range}->{MAX_ENERGY_RANGE}")
        req.energy_range = MAX_ENERGY_RANGE

    if MAX_BOX_SIZE and MAX_BOX_SIZE > 0:
        for axis in ("size_x", "size_y", "size_z"):
            val = float(getattr(req.grid, axis))
            if val > MAX_BOX_SIZE:
                notes.append(f"grid.{axis} {val}->{MAX_BOX_SIZE}")
                setattr(req.grid, axis, MAX_BOX_SIZE)

    return notes


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="API wrapper for Anyone Can Dock.",
)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


class GridBox(BaseModel):
    center_x: Optional[float] = Field(default=None)
    center_y: Optional[float] = Field(default=None)
    center_z: Optional[float] = Field(default=None)
    size_x: float = Field(default=20.0, gt=0)
    size_y: float = Field(default=20.0, gt=0)
    size_z: float = Field(default=20.0, gt=0)

    def has_manual_center(self) -> bool:
        return self.center_x is not None and self.center_y is not None and self.center_z is not None

    def center_tuple(self) -> Tuple[float, float, float]:
        if not self.has_manual_center():
            raise ValueError("center_x, center_y, and center_z are required for manual center mode")
        return float(self.center_x), float(self.center_y), float(self.center_z)

    def size_tuple(self) -> Tuple[float, float, float]:
        return float(self.size_x), float(self.size_y), float(self.size_z)


class LigandInput(BaseModel):
    smiles: str
    name: str = "ligand"


class DockRequest(BaseModel):
    pdb_id: Optional[str] = None
    receptor_pdb_text: Optional[str] = None
    receptor_name: str = "receptor"

    ligands: List[LigandInput]

    grid: GridBox = Field(default_factory=GridBox)
    center_mode: Literal["auto", "manual", "selection", "blind"] = "auto"
    prody_sel: str = ""
    preferred_ligand: str = ""
    hetatm_policy: Dict[str, str] = Field(default_factory=dict)
    reference_hetatm_key: str = ""

    ph: float = 7.4
    protonation_mode: Literal["pkanet", "neutral", "dimorphite"] = "pkanet"
    use_pubchem: bool = True
    max_tautomers: int = Field(default=8, ge=1, le=64)
    ph_window: float = Field(default=1.0, ge=0.0, le=4.0)
    pkanet_selection_mode: Literal["auto_recommended", "highest_score", "manual_rank"] = "auto_recommended"
    pkanet_manual_rank: Optional[int] = Field(default=None, ge=1)

    exhaustiveness: int = Field(default=8, ge=1, le=128)
    num_modes: int = Field(default=10, ge=1, le=50)
    energy_range: int = Field(default=3, ge=1, le=20)

    fix_bond_orders: bool = True

    @model_validator(mode="after")
    def validate_inputs(self) -> "DockRequest":
        if not self.pdb_id and not self.receptor_pdb_text:
            raise ValueError("Either pdb_id or receptor_pdb_text is required")
        if not self.ligands:
            raise ValueError("At least one ligand is required")
        if self.center_mode == "manual" and not self.grid.has_manual_center():
            raise ValueError("Manual center mode requires center_x, center_y, and center_z")
        if self.center_mode == "selection" and not self.prody_sel.strip():
            raise ValueError("Selection center mode requires prody_sel")
        return self


class DockSubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str
    status_url: str
    download_url: str


class ScanRequest(BaseModel):
    pdb_id: Optional[str] = None
    receptor_pdb_text: Optional[str] = None

    @model_validator(mode="after")
    def validate_inputs(self) -> "ScanRequest":
        if not self.pdb_id and not self.receptor_pdb_text:
            raise ValueError("Either pdb_id or receptor_pdb_text is required")
        return self


def _safe_name(name: str, default: str = "item") -> str:
    cleaned = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name or ""))
    return cleaned[:80] or default


def _public_url(path: str) -> str:
    base = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}" if base else path


def _job_url(job_id: str, endpoint: str) -> str:
    if endpoint == "download":
        return _public_url(f"/jobs/{job_id}/download")
    return _public_url(f"/jobs/{job_id}")


def _status_path(job_id: str) -> Path:
    return BASE_WORKDIR / job_id / "status.json"


def _write_status_file(job_id: str, payload: Dict[str, Any]) -> None:
    wdir = BASE_WORKDIR / job_id
    wdir.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("job_id", job_id)
    payload.setdefault("download_url", _public_url(f"/jobs/{job_id}/download"))
    _status_path(job_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_status_file(job_id: str) -> Optional[Dict[str, Any]]:
    path = _status_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_receptor_input(req: DockRequest | ScanRequest, wdir: Path) -> str:
    if req.pdb_id:
        pdb_id = req.pdb_id.strip().upper()
        if not pdb_id or len(pdb_id) > 12:
            raise ValueError("Invalid PDB ID")

        pdb_path = wdir / f"{pdb_id}.pdb"
        cif_path = wdir / f"{pdb_id}.cif"

        for url, out_path in [
            (f"https://files.rcsb.org/download/{pdb_id}.pdb", pdb_path),
            (f"https://files.rcsb.org/download/{pdb_id}.cif", cif_path),
        ]:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and len(r.text) > 200:
                out_path.write_text(r.text, encoding="utf-8")
                return str(out_path)

        raise ValueError(f"Could not download PDB/mmCIF for {pdb_id}")

    suffix = ".cif" if (req.receptor_pdb_text or "").lstrip().startswith("data_") else ".pdb"
    rec_path = wdir / f"{_safe_name(getattr(req, 'receptor_name', 'receptor'), 'receptor')}{suffix}"
    rec_path.write_text(req.receptor_pdb_text or "", encoding="utf-8")
    return str(rec_path)


def _compute_blind_box(raw_pdb: str, padding: float = 4.0) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    # [REV] Support mmCIF as well as PDB; _write_receptor_input may fall back to a .cif download.
    if str(raw_pdb).lower().endswith(".cif"):
        from prody import parseMMCIF

        atoms = parseMMCIF(raw_pdb)
    else:
        from prody import parsePDB

        atoms = parsePDB(raw_pdb)

    if atoms is None:
        raise ValueError("Could not parse receptor for blind docking box")

    prot = atoms.select("protein") or atoms
    coords = prot.getCoords()
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)

    center = tuple(float(x) for x in ((mn + mx) / 2.0))
    size = tuple(float(x) for x in ((mx - mn) + 2.0 * padding))
    return center, size


def _csv_from_scores(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "input_smiles",
        "prepared_smiles",
        "charge",
        "status",
        "top_score",
        "num_poses",
        "out_pdbqt",
        "out_sdf",
        "pv_sdf",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_zip(wdir: Path, zip_path: Path) -> None:
    # [REV] Never include the zip itself or the transient status.json.
    skip = {zip_path.resolve()}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in wdir.rglob("*"):
            if p.is_file() and p.resolve() not in skip and p.name != "status.json":
                zf.write(p, arcname=p.relative_to(wdir))


def _looks_like_protonation_valence_error(text: str) -> bool:
    t = (text or "").lower()
    triggers = [
        "explicit valence",
        "greater than permitted",
        "valence for atom",
        "sanitize",
        "kekulize",
    ]
    return any(x in t for x in triggers)


def _is_redocking_case(lig_name: str, rec: Dict[str, Any], req: DockRequest) -> bool:
    name = (lig_name or "").strip().upper()
    if not name:
        return False

    cocrystal_id = str(rec.get("cocrystal_ligand_id") or "").strip().upper()
    cocrystal_resname = cocrystal_id.split("_")[0] if cocrystal_id else ""
    preferred = (req.preferred_ligand or "").strip().upper()
    ref_key = (req.reference_hetatm_key or "").strip().upper()

    candidates = {x for x in [cocrystal_id, cocrystal_resname, preferred, ref_key] if x}
    return name in candidates


def _select_pose_for_interaction(
    pose_sdf: str,
    rec: Dict[str, Any],
    req: DockRequest,
    lig_name: str,
) -> Dict[str, Any]:
    pose_info: Dict[str, Any] = {
        "selected_pose_rank": None,
        "pose_selection_method": "",
        "selected_pose_rmsd": None,
        "warning": "",
        "selected_pose_sdf": "",
    }

    mols = load_mols_from_sdf(pose_sdf, sanitize=False) if pose_sdf else []
    if not mols:
        pose_info["pose_selection_method"] = "unavailable"
        pose_info["warning"] = "No readable docking poses were available for 2D interaction generation."
        return pose_info

    ligand_pdb_path = rec.get("ligand_pdb_path") or ""
    is_redocking = _is_redocking_case(lig_name, rec, req)

    selected_idx = 0

    if is_redocking and calc_rmsd_heavy is not None and ligand_pdb_path and Path(ligand_pdb_path).exists():
        rmsd_rows = []
        for i, mol in enumerate(mols):
            try:
                rmsd = calc_rmsd_heavy(mol, ligand_pdb_path)
            except Exception:
                rmsd = None
            if rmsd is not None:
                rmsd_rows.append((float(rmsd), i))

        if rmsd_rows:
            rmsd_rows.sort(key=lambda x: x[0])
            best_rmsd, selected_idx = rmsd_rows[0]
            pose_info["pose_selection_method"] = "lowest_rmsd_vs_cocrystal_ligand"
            pose_info["selected_pose_rmsd"] = round(float(best_rmsd), 4)
        else:
            pose_info["pose_selection_method"] = "top_score_fallback"
            pose_info["warning"] = "Redocking detected but RMSD could not be computed; top-ranked pose used."

    elif is_redocking:
        pose_info["pose_selection_method"] = "top_score_fallback"
        pose_info["warning"] = "Redocking detected but RMSD utilities/reference ligand unavailable; top-ranked pose used."

    else:
        pose_info["pose_selection_method"] = "top_score"

    selected_pose_sdf = str(Path(pose_sdf).with_name(f"{Path(pose_sdf).stem}_selected_for_2d.sdf"))

    try:
        if write_single_pose is not None:
            write_single_pose(mols[selected_idx], selected_pose_sdf)
        else:
            from rdkit import Chem
            with Chem.SDWriter(selected_pose_sdf) as writer:
                writer.write(mols[selected_idx])
        pose_info["selected_pose_sdf"] = selected_pose_sdf
    except Exception as e:
        pose_info["warning"] = (
            (pose_info.get("warning") + " ") if pose_info.get("warning") else ""
        ) + f"Could not write selected pose SDF: {e}"

    pose_info["selected_pose_rank"] = int(selected_idx + 1)
    return pose_info


def _generate_2d_interaction(
    job_id: str,
    wdir: Path,
    rec: Dict[str, Any],
    pose_info: Dict[str, Any],
    smiles: str,
    lig_name: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "poseview_available": False,
        "poseview_svg_url": "",
        "poseview_png_url": "",
        "poseview_svg_file": "",
        "poseview_png_file": "",
        "poseview_error": "",
        "poseview_source": "",
        "two_d_interaction_available": False,
        "two_d_interaction_svg_url": "",
        "two_d_interaction_png_url": "",
        "two_d_interaction_svg_file": "",
        "two_d_interaction_png_file": "",
        "two_d_interaction_error": "",
    }

    selected_pose_sdf = pose_info.get("selected_pose_sdf") or ""
    receptor_pdb = rec.get("rec_fh") or ""

    if not selected_pose_sdf or not Path(selected_pose_sdf).exists():
        err = "Selected pose SDF is unavailable."
        out["poseview_error"] = err
        out["two_d_interaction_error"] = err
        return out

    if not receptor_pdb or not Path(receptor_pdb).exists():
        err = "Prepared receptor PDB is unavailable."
        out["poseview_error"] = err
        out["two_d_interaction_error"] = err
        return out

    safe = _safe_name(lig_name, "ligand")
    svg_name = f"{safe}_interaction2d.svg"
    png_name = f"{safe}_interaction2d.png"
    svg_path = wdir / svg_name
    png_path = wdir / png_name

    svg_bytes = None
    source = ""
    warnings: List[str] = []

    try:
        if call_poseview_v1 is not None:
            svg_bytes, pv_err = call_poseview_v1(receptor_pdb=receptor_pdb, pose_sdf=selected_pose_sdf)
            if svg_bytes:
                source = "ProteinsPlus PoseView v1"
            else:
                warnings.append(pv_err or "PoseView did not return an SVG diagram.")
        else:
            warnings.append("PoseView helper is not available in anyonecandock.")
    except Exception as e:
        warnings.append(f"PoseView failed: {e}")

    if not svg_bytes:
        try:
            if draw_interaction_diagram is None:
                warnings.append("draw_interaction_diagram is not available in anyonecandock.")
            else:
                svg_bytes = draw_interaction_diagram(
                    receptor_pdb=receptor_pdb,
                    pose_sdf=selected_pose_sdf,
                    smiles=smiles,
                    title=f"{safe} selected pose",
                )
                if svg_bytes:
                    source = "Anyone Can Dock 2D Diagram"
        except Exception as e:
            warnings.append(f"Anyone Can Dock 2D Diagram failed: {e}")

    if not svg_bytes:
        err = " ; ".join([w for w in warnings if w]) or "Could not generate any 2D interaction diagram."
        out["poseview_error"] = err
        out["two_d_interaction_error"] = err
        return out

    svg_path.write_bytes(svg_bytes)

    png_ok = False
    png_err = ""

    try:
        png_bytes = svg_to_png(svg_bytes) if svg_to_png is not None else None
        if png_bytes:
            png_path.write_bytes(png_bytes)
            png_ok = png_path.exists() and png_path.stat().st_size > 100
        else:
            png_err = "PNG conversion failed for the interaction diagram."
    except Exception as png_e:
        png_err = f"PNG conversion failed: {png_e}"

    combined_warning = " ; ".join([w for w in warnings if w] + ([png_err] if png_err else []))
    svg_url = _public_url(f"/jobs/{job_id}/files/{svg_name}")
    png_url = _public_url(f"/jobs/{job_id}/files/{png_name}") if png_ok else ""

    out.update({
        "poseview_available": True,
        "poseview_svg_url": svg_url,
        "poseview_png_url": png_url,
        "poseview_svg_file": str(svg_path),
        "poseview_png_file": str(png_path) if png_ok else "",
        "poseview_error": combined_warning,
        "poseview_source": source,
        "two_d_interaction_available": True,
        "two_d_interaction_svg_url": svg_url,
        "two_d_interaction_png_url": png_url,
        "two_d_interaction_svg_file": str(svg_path),
        "two_d_interaction_png_file": str(png_path) if png_ok else "",
        "two_d_interaction_error": combined_warning,
    })

    return out


def _ultra_compact_from_meta(job_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    receptor = meta.get("receptor", {}) if isinstance(meta, dict) else {}
    results = []

    for r in meta.get("results", []) if isinstance(meta, dict) else []:
        scores = r.get("scores", [])
        if not isinstance(scores, list):
            scores = []

        results.append({
            "name": r.get("name", ""),
            "status": r.get("status", ""),
            "top_score": r.get("top_score", None),
            "scores": scores[:10],
            "num_poses": r.get("num_poses", None),
            "charge": r.get("charge", None),
            "prepared_smiles": r.get("prepared_smiles", ""),
            "protonation_mode_used": r.get("protonation_mode_used", ""),
            "protonation_fallback": r.get("protonation_fallback", ""),
            "selected_pose_rank": r.get("selected_pose_rank", None),
            "pose_selection_method": r.get("pose_selection_method", ""),
            "selected_pose_rmsd": r.get("selected_pose_rmsd", None),
            "pose_selection_warning": r.get("pose_selection_warning", ""),
            "two_d_interaction_available": bool(r.get("two_d_interaction_available", False)),
            "two_d_interaction_svg_url": r.get("two_d_interaction_svg_url", ""),
            "two_d_interaction_png_url": r.get("two_d_interaction_png_url", ""),
            "two_d_interaction_error": r.get("two_d_interaction_error", ""),
            "poseview_available": bool(r.get("poseview_available", False)),
            "poseview_svg_url": r.get("poseview_svg_url", ""),
            "poseview_png_url": r.get("poseview_png_url", ""),
            "poseview_error": r.get("poseview_error", ""),
            "error": r.get("error", ""),
        })

    return {
        "job_id": job_id,
        "status": meta.get("status", "completed"),
        "message": "Docking job completed.",
        "receptor": {
            "pdb_id": receptor.get("pdb_id"),
            "center": receptor.get("center", {}),
            "size": receptor.get("size", {}),
            "cocrystal_ligand_id": receptor.get("cocrystal_ligand_id"),
        },
        "results": results,
        "download_url": _public_url(f"/jobs/{job_id}/download"),
    }


def _run_docking_job(job_id: str, req: DockRequest) -> None:
    wdir = BASE_WORKDIR / job_id
    wdir.mkdir(parents=True, exist_ok=True)

    JOBS[job_id].update(status="running", workdir=str(wdir), error=None)
    _write_status_file(job_id, {
        "job_id": job_id,
        "status": "running",
        "message": "Docking job is running.",
        "download_url": _public_url(f"/jobs/{job_id}/download"),
    })

    try:
        raw_receptor = _write_receptor_input(req, wdir)

        center_mode = req.center_mode
        grid = req.grid
        manual_xyz = (0.0, 0.0, 0.0)
        box_size = grid.size_tuple()
        prody_sel = req.prody_sel

        if center_mode == "manual":
            manual_xyz = grid.center_tuple()
            core_center_mode = "manual"
        elif center_mode == "selection":
            core_center_mode = "selection"
        elif center_mode == "blind":
            blind_center, blind_size = _compute_blind_box(raw_receptor)
            manual_xyz = blind_center
            box_size = _clamp_box_size(blind_size)  # [REV] respect MAX_BOX_SIZE on blind boxes too
            core_center_mode = "manual"
        else:
            core_center_mode = "auto"

        rec = prepare_receptor(
            raw_pdb=raw_receptor,
            wdir=wdir,
            center_mode=core_center_mode,
            manual_xyz=manual_xyz,
            prody_sel=prody_sel,
            box_size=box_size,
            preferred_ligand=req.preferred_ligand,
            hetatm_policy=req.hetatm_policy,
            reference_hetatm_key=req.reference_hetatm_key,
        )

        if not rec.get("success"):
            raise RuntimeError(f"Receptor preparation failed: {rec.get('error')}")

        vina_path, vina_msg = get_vina_binary()
        if not vina_path:
            raise RuntimeError(f"Vina binary unavailable: {vina_msg}")

        results: List[Dict[str, Any]] = []
        all_logs: List[str] = []
        used_names: set = set()  # [REV] guarantee unique output stems for duplicate ligand names

        for idx, lig in enumerate(req.ligands, start=1):
            # [REV] Disambiguate duplicate/blank names so output files never collide.
            base_name = _safe_name(lig.name, f"lig_{idx}")
            name = base_name
            _dup = 2
            while name in used_names:
                name = f"{base_name}_{_dup}"
                _dup += 1
            used_names.add(name)

            row: Dict[str, Any] = {
                "name": name,
                "input_smiles": lig.smiles,
                "prepared_smiles": "",
                "charge": None,
                "status": "pending",
                "top_score": None,
                "num_poses": 0,
                "out_pdbqt": "",
                "out_sdf": "",
                "pv_sdf": "",
                "error": "",
            }

            try:
                prep = prepare_ligand(
                    smiles=lig.smiles,
                    name=name,
                    ph=req.ph,
                    wdir=wdir,
                    mode=req.protonation_mode,
                    use_pubchem=req.use_pubchem,
                    max_tautomers=req.max_tautomers,
                    ph_window=req.ph_window,
                    pkanet_selection_mode=req.pkanet_selection_mode,
                    pkanet_manual_rank=req.pkanet_manual_rank,
                )

                all_logs.append(f"\n===== {name}: ligand preparation =====")
                all_logs.extend(prep.get("log", []))

                if not prep.get("success"):
                    prep_error = str(prep.get("error", "Ligand preparation failed"))
                    prep_log = "\n".join(map(str, prep.get("log", [])))

                    if (
                        req.protonation_mode != "neutral"
                        and _looks_like_protonation_valence_error(prep_error + "\n" + prep_log)
                    ):
                        all_logs.append(
                            f"\n===== {name}: ligand preparation fallback =====\n"
                            "pKa/protonation mode produced a valence/sanitization error. "
                            "Retrying once with protonation_mode='neutral'."
                        )

                        prep = prepare_ligand(
                            smiles=lig.smiles,
                            name=f"{name}_neutral_fallback",
                            ph=req.ph,
                            wdir=wdir,
                            mode="neutral",
                            use_pubchem=False,
                            max_tautomers=1,
                            ph_window=0.0,
                            pkanet_selection_mode="auto_recommended",
                            pkanet_manual_rank=None,
                        )

                        all_logs.extend(prep.get("log", []))
                        row["protonation_fallback"] = "neutral"

                    if not prep.get("success"):
                        raise RuntimeError(prep.get("error", prep_error))

                dock = run_vina(
                    receptor_pdbqt=rec["rec_pdbqt"],
                    ligand_pdbqt=prep["pdbqt"],
                    config_txt=rec["config_txt"],
                    vina_path=vina_path,
                    exhaustiveness=req.exhaustiveness,
                    n_modes=req.num_modes,
                    energy_range=req.energy_range,
                    wdir=wdir,
                    out_name=name,
                )

                all_logs.append(f"\n===== {name}: docking =====")
                all_logs.append(dock.get("log", ""))

                if not dock.get("success"):
                    raise RuntimeError(dock.get("error", "Docking failed"))

                pv_sdf = ""
                if req.fix_bond_orders and dock.get("out_sdf"):
                    pv_path = wdir / f"{name}_pv_ready.sdf"
                    bo_log = fix_sdf_bond_orders(
                        dock["out_sdf"],
                        prep.get("prot_smiles", lig.smiles),
                        str(pv_path),
                    )
                    all_logs.append(f"\n===== {name}: bond-order correction =====")
                    all_logs.extend(bo_log)

                    if pv_path.exists() and pv_path.stat().st_size > 10:
                        pv_sdf = str(pv_path)

                pose_sdf_for_reading = pv_sdf or dock.get("out_sdf", "")
                n_poses = len(load_mols_from_sdf(pose_sdf_for_reading, sanitize=False)) if pose_sdf_for_reading else 0
                prepared_smiles = prep.get("prot_smiles") or prep.get("prepared_smiles") or lig.smiles

                # ------------------------------------------------------------------
                # ligand_pdb_path must be defined before use.
                # RMSD over poses is only computed for true redocking cases.
                # ------------------------------------------------------------------
                ligand_pdb_path = rec.get("ligand_pdb_path") or ""

                raw_scores = dock.get("scores", [])
                if (
                    pose_sdf_for_reading  # [REV] skip if no pose SDF was produced
                    and _is_redocking_case(name, rec, req)
                    and calc_rmsd_heavy is not None
                    and ligand_pdb_path
                    and Path(ligand_pdb_path).exists()
                ):
                    all_mols = load_mols_from_sdf(pose_sdf_for_reading, sanitize=False)
                    for i, score_row in enumerate(raw_scores):
                        if not isinstance(score_row, dict):
                            continue  # [REV] guard: score rows may not be mutable dicts
                        if i < len(all_mols):
                            try:
                                rmsd = calc_rmsd_heavy(all_mols[i], ligand_pdb_path)
                                score_row["rmsd_vs_crystal"] = round(float(rmsd), 2) if rmsd is not None else None
                            except Exception:
                                score_row["rmsd_vs_crystal"] = None

                pose_info = _select_pose_for_interaction(
                    pose_sdf=pose_sdf_for_reading,
                    rec=rec,
                    req=req,
                    lig_name=name,
                )

                interaction2d = _generate_2d_interaction(
                    job_id=job_id,
                    wdir=wdir,
                    rec=rec,
                    pose_info=pose_info,
                    smiles=prepared_smiles,
                    lig_name=name,
                )

                row.update(
                    prepared_smiles=prepared_smiles,
                    charge=prep.get("charge"),
                    status="ok",
                    top_score=dock.get("top_score"),
                    num_poses=n_poses,
                    out_pdbqt=dock.get("out_pdbqt", ""),
                    out_sdf=dock.get("out_sdf", ""),
                    pv_sdf=pose_sdf_for_reading,
                    scores=raw_scores,
                    pkanet_ranked_csv=prep.get("pkanet_ranked_csv", ""),
                    pkanet_decision_log=prep.get("pkanet_decision_log", ""),
                    pkanet_ambiguous=prep.get("pkanet_ambiguous", False),
                    protonation_mode_used=(
                        "neutral" if row.get("protonation_fallback") == "neutral" else req.protonation_mode
                    ),
                    protonation_fallback=row.get("protonation_fallback", ""),
                    selected_pose_rank=pose_info.get("selected_pose_rank"),
                    pose_selection_method=pose_info.get("pose_selection_method", ""),
                    selected_pose_rmsd=pose_info.get("selected_pose_rmsd"),
                    pose_selection_warning=pose_info.get("warning", ""),
                    selected_pose_sdf=pose_info.get("selected_pose_sdf", ""),
                    **interaction2d,
                )

            except Exception as lig_error:
                row.update(status="failed", error=str(lig_error))
                all_logs.append(f"\n===== {name}: ERROR =====\n{lig_error}")

            results.append(row)

        csv_path = wdir / "docking_summary.csv"
        _csv_from_scores(csv_path, results)

        log_path = wdir / "workflow_log.txt"
        log_path.write_text("\n".join(str(x) for x in (rec.get("log", []) + all_logs)), encoding="utf-8")

        meta = {
            "job_id": job_id,
            "status": "completed",
            "receptor": {
                "pdb_id": req.pdb_id,
                "rec_fh": rec.get("rec_fh"),
                "rec_pdbqt": rec.get("rec_pdbqt"),
                "config_txt": rec.get("config_txt"),
                "box_pdb": rec.get("box_pdb"),
                "center": {"x": rec.get("cx"), "y": rec.get("cy"), "z": rec.get("cz")},
                "size": {"x": rec.get("sx"), "y": rec.get("sy"), "z": rec.get("sz")},
                "cocrystal_ligand_id": rec.get("cocrystal_ligand_id"),
                "hetatm_table": rec.get("hetatm_table", []),
            },
            "results": results,
            "summary_csv": str(csv_path),
            "log_file": str(log_path),
        }

        meta_path = wdir / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        # [REV] Build the result zip BEFORE flipping status to "completed",
        # so a client that sees "completed" is guaranteed a downloadable zip.
        zip_path = wdir / f"{job_id}_results.zip"
        _make_zip(wdir, zip_path)

        _write_status_file(job_id, _ultra_compact_from_meta(job_id, meta))

        JOBS[job_id].update(
            status="completed",
            result=meta,
            zip_path=str(zip_path),
            summary_csv=str(csv_path),
            log_file=str(log_path),
        )

    except Exception as e:
        err_text = str(e)
        tb = traceback.format_exc()

        (wdir / "error_traceback.txt").write_text(tb, encoding="utf-8")

        _write_status_file(job_id, {
            "job_id": job_id,
            "status": "failed",
            "message": "Docking job failed.",
            "error": err_text,
            "download_url": _public_url(f"/jobs/{job_id}/download"),
        })

        JOBS[job_id].update(status="failed", error=err_text, traceback=tb)


def _restore_completed_job_from_disk(job_id: str) -> Optional[Dict[str, Any]]:
    wdir = BASE_WORKDIR / job_id
    meta_path = wdir / "metadata.json"

    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    zip_path = wdir / f"{job_id}_results.zip"

    job = {
        "job_id": job_id,
        "status": meta.get("status", "completed"),
        "request": {},
        "result": meta,
        "error": None,
        "zip_path": str(zip_path) if zip_path.exists() else "",
        "summary_csv": meta.get("summary_csv", ""),
        "log_file": meta.get("log_file", ""),
        "workdir": str(wdir),
    }

    JOBS[job_id] = job
    return job


def _compact_job_response(job_id: str, job: Dict[str, Any]) -> Dict[str, Any]:
    status = job.get("status", "unknown")

    out: Dict[str, Any] = {
        "job_id": job_id,
        "status": status,
        "message": "",
        "download_url": _public_url(f"/jobs/{job_id}/download"),
    }

    if status in ("queued", "running"):
        out["message"] = "Docking job is still running. Check again later."
        return out

    if status == "failed":
        out["message"] = "Docking job failed."
        out["error"] = job.get("error", "")
        return out

    result = job.get("result") or {}
    receptor = result.get("receptor", {}) if isinstance(result, dict) else {}

    compact_results = []

    for r in result.get("results", []) if isinstance(result, dict) else []:
        scores = r.get("scores", [])
        if not isinstance(scores, list):
            scores = []

        compact_results.append({
            "name": r.get("name"),
            "status": r.get("status"),
            "top_score": r.get("top_score"),
            "num_poses": r.get("num_poses"),
            "charge": r.get("charge"),
            "prepared_smiles": r.get("prepared_smiles"),
            "protonation_mode_used": r.get("protonation_mode_used", ""),
            "protonation_fallback": r.get("protonation_fallback", ""),
            "selected_pose_rank": r.get("selected_pose_rank", None),
            "pose_selection_method": r.get("pose_selection_method", ""),
            "selected_pose_rmsd": r.get("selected_pose_rmsd", None),
            "pose_selection_warning": r.get("pose_selection_warning", ""),
            "two_d_interaction_available": bool(r.get("two_d_interaction_available", False)),
            "two_d_interaction_svg_url": r.get("two_d_interaction_svg_url", ""),
            "two_d_interaction_png_url": r.get("two_d_interaction_png_url", ""),
            "two_d_interaction_error": r.get("two_d_interaction_error", ""),
            "poseview_available": bool(r.get("poseview_available", False)),
            "poseview_svg_url": r.get("poseview_svg_url", ""),
            "poseview_png_url": r.get("poseview_png_url", ""),
            "poseview_error": r.get("poseview_error", ""),
            "scores": scores[:10],
            "error": r.get("error", ""),
        })

    out.update({
        "message": "Docking job completed.",
        "receptor": {
            "pdb_id": receptor.get("pdb_id"),
            "center": receptor.get("center"),
            "size": receptor.get("size"),
            "cocrystal_ligand_id": receptor.get("cocrystal_ligand_id"),
        },
        "results": compact_results,
        "summary_csv": result.get("summary_csv") if isinstance(result, dict) else "",
        "log_file": result.get("log_file") if isinstance(result, dict) else "",
    })

    return out


@app.get("/health")
def health() -> Dict[str, Any]:
    ob_ok, ob_msg = check_obabel()
    vina_path, vina_msg = get_vina_binary()

    return {
        "status": "ok",
        "api": API_TITLE,
        "version": API_VERSION,
        "workdir": str(BASE_WORKDIR),
        "public_base_url": os.getenv("PUBLIC_BASE_URL", ""),
        "openbabel": {"available": ob_ok, "message": ob_msg},
        "vina": {"available": bool(vina_path), "message": vina_msg},
        "auth_enabled": bool(API_KEY),
        "limits": _active_limits(),
        "daily_jobs_used": _daily_jobs_used(),
    }


@app.get("/ping")
def ping() -> Dict[str, Any]:
    ob_ok, ob_msg = check_obabel()
    vina_path, vina_msg = get_vina_binary()

    return {
        "ok": True,
        "api": "Anyone Can Dock API",
        "version": API_VERSION,
        "openbabel_available": bool(ob_ok),
        "openbabel_message": str(ob_msg),
        "vina_available": bool(vina_path),
        "vina_message": str(vina_msg),
        "auth_enabled": bool(API_KEY),
    }


_BUILTIN_LIGANDS: Dict[str, Dict[str, str]] = {
    "gefitinib": {
        "name": "Gefitinib",
        "smiles": "COc1cc2c(cc1OCCCN1CCOCC1)ncnc2Nc1ccc(F)c(Cl)c1",
        "source": "built-in example ligand list",
    },
    "erlotinib": {
        "name": "Erlotinib",
        "smiles": "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
        "source": "built-in example ligand list",
    },
    "osimertinib": {
        "name": "Osimertinib",
        "smiles": "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C",
        "source": "built-in example ligand list",
    },
    "afatinib": {
        "name": "Afatinib",
        "smiles": "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1",
        "source": "built-in example ligand list",
    },
}


def _pick_smiles_from_pubchem_props(props: Dict[str, Any]) -> str:
    for key in ("IsomericSMILES", "CanonicalSMILES", "ConnectivitySMILES", "SMILES"):
        val = props.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


@app.get("/compound/smiles", dependencies=[Depends(require_api_key)])
def compound_smiles(
    name: str = Query(...),
    prefer_builtin: bool = Query(True),
) -> Dict[str, Any]:
    q = (name or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Compound name is required.")

    key = q.lower().strip()

    if prefer_builtin and key in _BUILTIN_LIGANDS:
        item = _BUILTIN_LIGANDS[key]
        return {
            "found": True,
            "query": q,
            "name": item["name"],
            "smiles": item["smiles"],
            "source": item["source"],
            "cid": None,
            "formula": "",
            "molecular_weight": None,
            "url": "",
            "warning": "Built-in example SMILES. Verify stereochemistry/salt form for research use.",
        }

    try:
        from urllib.parse import quote

        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(q)}/cids/JSON"
        cid_resp = requests.get(cid_url, timeout=12)

        if cid_resp.status_code != 200:
            return {
                "found": False,
                "query": q,
                "source": "PubChem",
                "error": f"Compound not found in PubChem or PubChem returned HTTP {cid_resp.status_code}.",
            }

        cid = cid_resp.json().get("IdentifierList", {}).get("CID", [None])[0]
        if cid is None:
            return {"found": False, "query": q, "source": "PubChem", "error": "No CID found."}

        prop_blocks = [
            "IUPACName,MolecularFormula,MolecularWeight,IsomericSMILES,CanonicalSMILES,ConnectivitySMILES",
            "IUPACName,MolecularFormula,MolecularWeight,CanonicalSMILES,ConnectivitySMILES",
        ]

        props: Dict[str, Any] = {}

        for block in prop_blocks:
            prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{block}/JSON"
            prop_resp = requests.get(prop_url, timeout=12)
            if prop_resp.status_code == 200:
                rows = prop_resp.json().get("PropertyTable", {}).get("Properties", [])
                if rows:
                    props = rows[0]
                    if _pick_smiles_from_pubchem_props(props):
                        break

        smiles = _pick_smiles_from_pubchem_props(props)

        if not smiles:
            return {
                "found": False,
                "query": q,
                "cid": cid,
                "source": "PubChem",
                "error": "PubChem CID found, but no usable SMILES was returned.",
            }

        return {
            "found": True,
            "query": q,
            "name": props.get("IUPACName", q),
            "smiles": smiles,
            "canonical_smiles": props.get("CanonicalSMILES", smiles),
            "source": "PubChem",
            "cid": cid,
            "formula": props.get("MolecularFormula", ""),
            "molecular_weight": props.get("MolecularWeight", None),
            "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            "warning": "Verify stereochemistry, salt form, tautomer, and protonation state before final docking.",
        }

    except Exception as e:
        return {"found": False, "query": q, "source": "PubChem", "error": str(e)}


@app.get("/pdb/search", dependencies=[Depends(require_api_key)])
def pdb_search(
    query: str = Query(...),
    top_n: int = Query(10, ge=1, le=25),
) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Search query is required.")

    try:
        payload = {
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {"value": q},
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": int(top_n)},
                "results_verbosity": "compact",
            },
        }

        r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=payload, timeout=15)

        if r.status_code != 200:
            return {
                "query": q,
                "found": False,
                "results": [],
                "error": f"RCSB search returned HTTP {r.status_code}.",
            }

        hits = r.json().get("result_set", []) or []
        results: List[Dict[str, Any]] = []

        for hit in hits[:top_n]:
            if isinstance(hit, dict):
                pdb_id = str(hit.get("identifier", "") or hit.get("entry_id", "")).upper().strip()
            else:
                pdb_id = str(hit).upper().strip()

            if not pdb_id or any(ch in pdb_id for ch in "{}[]:, "):
                continue

            title = ""
            method = ""
            resolution = None
            deposited_ligands: List[str] = []
            protein_name = ""

            try:
                e = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}", timeout=12)
                if e.status_code == 200:
                    ej = e.json()
                    title = (ej.get("struct", {}) or {}).get("title", "") or ""
                    exptl = ej.get("exptl") or []
                    if isinstance(exptl, list) and exptl:
                        method = str((exptl[0] or {}).get("method", "") or "")
                    info = ej.get("rcsb_entry_info", {}) or {}
                    res_comb = info.get("resolution_combined")
                    if isinstance(res_comb, list) and res_comb:
                        try:
                            resolution = float(res_comb[0])
                        except Exception:
                            resolution = None
            except Exception:
                pass

            try:
                np_r = requests.get(f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/1", timeout=8)
                if np_r.status_code == 200:
                    nj = np_r.json()
                    comp = (((nj.get("pdbx_entity_nonpoly", {}) or {}).get("comp_id", "")) or "").strip()
                    if comp:
                        deposited_ligands.append(comp)
            except Exception:
                pass

            try:
                pe = requests.get(f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1", timeout=8)
                if pe.status_code == 200:
                    pj = pe.json()
                    protein_name = ((pj.get("rcsb_polymer_entity", {}) or {}).get("pdbx_description", "") or "").strip()
            except Exception:
                pass

            results.append({
                "pdb_id": pdb_id,
                "title": title,
                "method": method,
                "resolution": resolution,
                "protein_name": protein_name,
                "example_ligands": deposited_ligands[:5],
                "url": f"https://www.rcsb.org/structure/{pdb_id}",
            })

        return {
            "query": q,
            "found": bool(results),
            "results": results,
            "source": "RCSB PDB Search/Data API",
            "warning": "Review biological relevance, ligand identity, mutations, missing residues, and resolution before docking.",
        }

    except Exception as e:
        return {
            "query": q,
            "found": False,
            "results": [],
            "source": "RCSB PDB Search/Data API",
            "error": str(e),
        }


@app.post("/scan_hetatm", dependencies=[Depends(require_api_key)])
def scan_hetatm(req: ScanRequest) -> Dict[str, Any]:
    job_id = f"scan_{uuid.uuid4().hex[:10]}"
    wdir = BASE_WORKDIR / job_id
    wdir.mkdir(parents=True, exist_ok=True)

    try:
        raw = _write_receptor_input(req, wdir)
        table = scan_hetatm_residues(raw)
        return {"success": True, "hetatm_table": table}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # [REV] scan jobs are throwaway; don't leak temp dirs on the free instance.
        shutil.rmtree(wdir, ignore_errors=True)


@app.post("/dock", response_model=DockSubmitResponse, dependencies=[Depends(require_api_key)])
def submit_docking(req: DockRequest, background_tasks: BackgroundTasks) -> DockSubmitResponse:
    # [REV] Enforce env-configured resource limits before queueing.
    if MAX_LIGANDS_PER_JOB and len(req.ligands) > MAX_LIGANDS_PER_JOB:
        raise HTTPException(
            status_code=422,
            detail=f"Too many ligands ({len(req.ligands)}). Maximum is {MAX_LIGANDS_PER_JOB} per job.",
        )

    cap_notes = _apply_runtime_caps(req)
    _reserve_daily_job_slot()  # raises 429 if the daily cap is reached

    job_id = f"dock_{uuid.uuid4().hex[:12]}"

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "request": req.model_dump(),
        "result": None,
        "error": None,
    }

    message = (
        "Docking job submitted. Please come back in about 1–2 minutes "
        "and ask DockGPT to check the status of this job_id. "
        "Do not report docking scores until the job status is completed."
    )
    if cap_notes:
        message += " Note: some parameters were capped to instance limits (" + "; ".join(cap_notes) + ")."

    _write_status_file(job_id, {
        "job_id": job_id,
        "status": "queued",
        "message": message,
        "download_url": _public_url(f"/jobs/{job_id}/download"),
    })

    background_tasks.add_task(_run_docking_job, job_id, req)

    return DockSubmitResponse(
        job_id=job_id,
        status="queued",
        message=message,
        status_url=_job_url(job_id, "status"),
        download_url=_job_url(job_id, "download"),
    )


@app.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str) -> Dict[str, Any]:
    status_payload = _read_status_file(job_id)
    if status_payload:
        return status_payload

    job = JOBS.get(job_id) or _restore_completed_job_from_disk(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=(
                "Job not found. The service may have restarted or the free instance "
                "may have lost its in-memory job registry. Please resubmit the docking job."
            ),
        )

    compact = _compact_job_response(job_id, job)
    _write_status_file(job_id, compact)
    return compact


@app.get("/jobs/{job_id}/summary", dependencies=[Depends(require_api_key)])
def get_job_summary(job_id: str) -> Dict[str, Any]:
    return get_job(job_id)


@app.get("/jobs/{job_id}/files/{filename}", dependencies=[Depends(require_api_key)])
def get_job_file(job_id: str, filename: str) -> FileResponse:
    safe_filename = Path(filename).name
    file_path = BASE_WORKDIR / job_id / safe_filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    suffix = file_path.suffix.lower()

    if suffix == ".svg":
        media_type = "image/svg+xml"
    elif suffix == ".png":
        media_type = "image/png"
    else:
        media_type = "application/octet-stream"

    return FileResponse(
        str(file_path),
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{safe_filename}"'},
    )


@app.get("/jobs/{job_id}/view/{filename}", dependencies=[Depends(require_api_key)])
def view_job_file(job_id: str, filename: str):
    from fastapi.responses import HTMLResponse

    safe_filename = Path(filename).name
    file_url = _public_url(f"/jobs/{job_id}/files/{safe_filename}")

    # [REV] Escape user-controlled values before embedding in HTML (reflected-XSS fix).
    esc_name = html.escape(safe_filename)
    esc_url = html.escape(file_url, quote=True)

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{esc_name}</title>
  <style>
    body {{ margin: 0; padding: 24px; font-family: Arial, sans-serif; background: #f7f7f7; }}
    .card {{ max-width: 1200px; margin: auto; background: white; padding: 18px; border-radius: 12px; box-shadow: 0 2px 14px rgba(0,0,0,0.10); }}
    img {{ width: 100%; height: auto; display: block; border: 1px solid #ddd; background: white; }}
    a {{ color: #2563eb; word-break: break-all; }}
  </style>
</head>
<body>
  <div class="card">
    <h2>{esc_name}</h2>
    <p><a href="{esc_url}" target="_blank">Open image directly</a></p>
    <img src="{esc_url}" alt="{esc_name}">
  </div>
</body>
</html>"""

    return HTMLResponse(content=html_doc)


@app.get("/jobs/{job_id}/download", dependencies=[Depends(require_api_key)])
def download_job(job_id: str) -> FileResponse:
    job = JOBS.get(job_id) or _restore_completed_job_from_disk(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail=f"Job is not completed: {job.get('status')}")

    zip_path = job.get("zip_path")

    if not zip_path or not Path(zip_path).exists():
        raise HTTPException(status_code=404, detail="Result zip not found")

    return FileResponse(zip_path, media_type="application/zip", filename=Path(zip_path).name)


@app.delete("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def delete_job(job_id: str) -> Dict[str, Any]:
    job = JOBS.pop(job_id, None)
    wdir = BASE_WORKDIR / job_id

    if wdir.exists():
        shutil.rmtree(wdir, ignore_errors=True)

    return {"success": True, "deleted": bool(job)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
