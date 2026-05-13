#!/usr/bin/env python3
"""
api.py — FastAPI wrapper for Anyone Can Dock / core.py.

Place this file next to core.py and pKaNET.py, then run:
    uvicorn api:app --host 0.0.0.0 --port 8000

Main GPT-friendly workflow:
    POST /dock
    GET  /jobs/{job_id}
    GET  /jobs/{job_id}/download
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from core import (
    check_obabel,
    fix_sdf_bond_orders,
    get_vina_binary,
    load_mols_from_sdf,
    prepare_ligand,
    prepare_receptor,
    run_vina,
    scan_hetatm_residues,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_TITLE = "Anyone Can Dock API"
API_VERSION = "0.1.0"
BASE_WORKDIR = Path(os.getenv("ACD_API_WORKDIR", "/tmp/anyone_can_dock_api")).resolve()
BASE_WORKDIR.mkdir(parents=True, exist_ok=True)

# Optional security. If DOCKGPT_API_KEY is set, clients must send X-API-Key.
API_KEY = os.getenv("DOCKGPT_API_KEY", "").strip()

# In-memory job registry. For production, replace this with Redis/DB/object storage.
JOBS: Dict[str, Dict[str, Any]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "API wrapper for Anyone Can Dock. It prepares receptor/ligand inputs, "
        "runs AutoDock Vina through core.py, and returns docking files and scores."
    ),
)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Optional API-key protection. Enabled only when DOCKGPT_API_KEY is set."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class GridBox(BaseModel):
    center_x: Optional[float] = Field(default=None, description="Docking box center X in Å")
    center_y: Optional[float] = Field(default=None, description="Docking box center Y in Å")
    center_z: Optional[float] = Field(default=None, description="Docking box center Z in Å")
    size_x: float = Field(default=20.0, gt=0, description="Docking box size X in Å")
    size_y: float = Field(default=20.0, gt=0, description="Docking box size Y in Å")
    size_z: float = Field(default=20.0, gt=0, description="Docking box size Z in Å")

    def has_manual_center(self) -> bool:
        return self.center_x is not None and self.center_y is not None and self.center_z is not None

    def center_tuple(self) -> Tuple[float, float, float]:
        if not self.has_manual_center():
            raise ValueError("center_x, center_y, and center_z are required for manual center mode")
        return float(self.center_x), float(self.center_y), float(self.center_z)

    def size_tuple(self) -> Tuple[float, float, float]:
        return float(self.size_x), float(self.size_y), float(self.size_z)


class LigandInput(BaseModel):
    smiles: str = Field(description="Ligand SMILES string")
    name: str = Field(default="ligand", description="Short ligand name used for output files")


class DockRequest(BaseModel):
    pdb_id: Optional[str] = Field(default=None, description="RCSB PDB ID, e.g. 1M17")
    receptor_pdb_text: Optional[str] = Field(
        default=None,
        description="Raw PDB/mmCIF text. Use this when no PDB ID is available.",
    )
    receptor_name: str = Field(default="receptor", description="Name for receptor file")

    ligands: List[LigandInput] = Field(description="One or more ligands to dock")

    grid: GridBox = Field(default_factory=GridBox)
    center_mode: Literal["auto", "manual", "selection", "blind"] = Field(
        default="auto",
        description=(
            "auto = use co-crystal/reference HETATM; manual = use grid center; "
            "selection = use ProDy selection; blind = whole-protein box"
        ),
    )
    prody_sel: str = Field(default="", description="ProDy atom selection when center_mode='selection'")
    preferred_ligand: str = Field(default="", description="Optional co-crystal ligand resname/key hint")
    hetatm_policy: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional HETATM policy table: key -> reference/keep/remove",
    )
    reference_hetatm_key: str = Field(default="", description="Specific HETATM key used as reference")

    ph: float = Field(default=7.4, description="pH for ligand protonation")
    protonation_mode: Literal["pkanet", "neutral", "dimorphite"] = Field(default="pkanet")
    use_pubchem: bool = Field(default=True, description="Allow pKaNET PubChem pKa lookup")
    max_tautomers: int = Field(default=8, ge=1, le=64)
    ph_window: float = Field(default=1.0, ge=0.0, le=4.0)
    pkanet_selection_mode: Literal["auto_recommended", "highest_score", "manual_rank"] = "auto_recommended"
    pkanet_manual_rank: Optional[int] = Field(default=None, ge=1)

    exhaustiveness: int = Field(default=8, ge=1, le=128)
    num_modes: int = Field(default=10, ge=1, le=50)
    energy_range: int = Field(default=3, ge=1, le=20)

    fix_bond_orders: bool = Field(default=True, description="Generate *_pv_ready.sdf using input SMILES template")

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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_name(name: str, default: str = "item") -> str:
    cleaned = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name or ""))
    return cleaned[:80] or default


def _write_receptor_input(req: DockRequest | ScanRequest, wdir: Path) -> str:
    if req.pdb_id:
        pdb_id = req.pdb_id.strip().upper()
        if not pdb_id or len(pdb_id) > 12:
            raise ValueError("Invalid PDB ID")
        pdb_path = wdir / f"{pdb_id}.pdb"
        cif_path = wdir / f"{pdb_id}.cif"
        # Try PDB first; fall back to mmCIF.
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
    from prody import parsePDB
    import numpy as np

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
        "name", "input_smiles", "prepared_smiles", "charge", "status", "top_score",
        "num_poses", "out_pdbqt", "out_sdf", "pv_sdf", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_zip(wdir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in wdir.rglob("*"):
            if p.is_file() and p != zip_path:
                zf.write(p, arcname=p.relative_to(wdir))


def _job_url(job_id: str, endpoint: str) -> str:
    # Relative URLs are easier for GPT Actions and reverse proxies.
    if endpoint == "download":
        return f"/jobs/{job_id}/download"
    return f"/jobs/{job_id}"


# ─────────────────────────────────────────────────────────────────────────────
# Background workflow
# ─────────────────────────────────────────────────────────────────────────────

def _run_docking_job(job_id: str, req: DockRequest) -> None:
    wdir = BASE_WORKDIR / job_id
    wdir.mkdir(parents=True, exist_ok=True)
    JOBS[job_id].update(status="running", workdir=str(wdir), error=None)

    try:
        raw_receptor = _write_receptor_input(req, wdir)

        # Resolve grid mode.
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
            box_size = blind_size
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

        for lig in req.ligands:
            name = _safe_name(lig.name, f"lig_{len(results)+1}")
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
                    raise RuntimeError(prep.get("error", "Ligand preparation failed"))

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
                    bo_log = fix_sdf_bond_orders(dock["out_sdf"], prep.get("prot_smiles", lig.smiles), str(pv_path))
                    all_logs.append(f"\n===== {name}: bond-order correction =====")
                    all_logs.extend(bo_log)
                    if pv_path.exists() and pv_path.stat().st_size > 10:
                        pv_sdf = str(pv_path)

                n_poses = len(load_mols_from_sdf(dock.get("out_sdf", ""), sanitize=False)) if dock.get("out_sdf") else 0
                row.update(
                    prepared_smiles=prep.get("prot_smiles") or prep.get("prepared_smiles") or lig.smiles,
                    charge=prep.get("charge"),
                    status="ok",
                    top_score=dock.get("top_score"),
                    num_poses=n_poses,
                    out_pdbqt=dock.get("out_pdbqt", ""),
                    out_sdf=dock.get("out_sdf", ""),
                    pv_sdf=pv_sdf or dock.get("out_sdf", ""),
                    scores=dock.get("scores", []),
                    pkanet_ranked_csv=prep.get("pkanet_ranked_csv", ""),
                    pkanet_decision_log=prep.get("pkanet_decision_log", ""),
                    pkanet_ambiguous=prep.get("pkanet_ambiguous", False),
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

        zip_path = wdir / f"{job_id}_results.zip"
        _make_zip(wdir, zip_path)

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
        JOBS[job_id].update(status="failed", error=err_text, traceback=tb)



# ─────────────────────────────────────────────────────────────────────────────
# GPT-friendly compact job responses
# ─────────────────────────────────────────────────────────────────────────────

def _restore_completed_job_from_disk(job_id: str) -> Optional[Dict[str, Any]]:
    """Restore a completed job from metadata.json if in-memory JOBS was lost."""
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
    """Return a small JSON response suitable for GPT Actions.

    The full metadata can be large because it may include HETATM tables,
    pKaNET logs, file paths, and pose-level information. GPT Actions can reject
    very large responses, so this endpoint returns only essential fields.
    """
    status = job.get("status", "unknown")
    out: Dict[str, Any] = {
        "job_id": job_id,
        "status": status,
        "message": "",
        "download_url": f"/jobs/{job_id}/download",
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
        compact_results.append({
            "name": r.get("name"),
            "status": r.get("status"),
            "top_score": r.get("top_score"),
            "num_poses": r.get("num_poses"),
            "charge": r.get("charge"),
            "prepared_smiles": r.get("prepared_smiles"),
            "scores": r.get("scores", [])[:10] if isinstance(r.get("scores", []), list) else [],
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


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    ob_ok, ob_msg = check_obabel()
    vina_path, vina_msg = get_vina_binary()
    return {
        "status": "ok",
        "api": API_TITLE,
        "version": API_VERSION,
        "workdir": str(BASE_WORKDIR),
        "openbabel": {"available": ob_ok, "message": ob_msg},
        "vina": {"available": bool(vina_path), "message": vina_msg},
        "auth_enabled": bool(API_KEY),
    }
    
@app.get("/ping")
def ping():
    ob_ok, ob_msg = check_obabel()
    vina_path, vina_msg = get_vina_binary()

    return {
        "ok": True,
        "api": "Anyone Can Dock API",
        "openbabel_available": bool(ob_ok),
        "openbabel_message": str(ob_msg),
        "vina_available": bool(vina_path),
        "vina_message": str(vina_msg),
        "auth_enabled": bool(API_KEY)
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


@app.post("/dock", response_model=DockSubmitResponse, dependencies=[Depends(require_api_key)])
def submit_docking(req: DockRequest, background_tasks: BackgroundTasks) -> DockSubmitResponse:
    job_id = f"dock_{uuid.uuid4().hex[:12]}"
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "request": req.model_dump(),
        "result": None,
        "error": None,
    }
    background_tasks.add_task(_run_docking_job, job_id, req)
    return DockSubmitResponse(
        job_id=job_id,
        status="queued",
        message="Docking job submitted. Poll status_url until status is completed or failed.",
        status_url=_job_url(job_id, "status"),
        download_url=_job_url(job_id, "download"),
    )


@app.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id) or _restore_completed_job_from_disk(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=(
                "Job not found. The service may have restarted or the free instance "
                "may have lost its in-memory job registry. Please resubmit the docking job."
            ),
        )
    return _compact_job_response(job_id, job)


@app.get("/jobs/{job_id}/download", dependencies=[Depends(require_api_key)])
def download_job(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
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


# ─────────────────────────────────────────────────────────────────────────────
# Local debug entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
