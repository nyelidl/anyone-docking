#!/usr/bin/env python3
"""
Anyone Can Dock (ACD)
Simple molecular docking from the command line.

Usage:
  acd <command> [options]
  acd --help
  acd --version

Commands:
  dock        Prepare a receptor and ligand, then run molecular docking
  redock      Re-dock the co-crystallized ligand to validate the setup
  batch       Dock multiple ligands against one receptor
  codock      Dock multiple ligands simultaneously in one Vina search
  ensemble    Dock multiple ligands against multiple receptor conformations
  receptor    Prepare a receptor only without docking
  ligand      Prepare a ligand only without docking
  diagram     Generate a 2D interaction diagram from docking results

Quick examples:
  # Re-dock the built-in crystal ligand to confirm the setup works
  acd redock --receptor 4AGN

  # Dock a compound using a PDB ID receptor
  acd dock --receptor 1M17 --compound erlotinib

  # Dock a ligand file
  acd dock --receptor protein.pdb --ligand-file ligand.sdf \
    --center auto --pkanet --ph 7.4

  # Use a manually defined docking box
  acd dock --receptor protein.pdb --ligand-file ligand.sdf \
    --center manual --cx 10.5 --cy 22.0 --cz -4.5 \
    --bx 20 --by 20 --bz 20

  # Batch dock a list of ligands
  acd batch --receptor 1M17 --ligands compounds.smi

  # Simultaneous docking: inline SMILES with optional labels
  acd codock --receptor protein.pdb --ligand "CCO ethanol" "c1ccccc1O phenol"

  # Simultaneous docking: compound names (looked up on PubChem)
  acd codock --receptor protein.pdb --ligand aspirin caffeine

  # Simultaneous docking: structure files (.sdf, .mol2, .pdb, .pdbqt)
  acd codock --receptor protein.pdb --ligand ligand1.sdf ligand2.pdbqt

  # Simultaneous docking: a .smi file with at least two SMILES lines
  acd codock --receptor protein.pdb --ligand pair.smi

  # Simultaneous docking: mix input types
  acd codock --receptor protein.pdb --ligand "CCO ethanol" aspirin ligand3.sdf

  # Dock a ligand library against a receptor ensemble
  acd ensemble --receptors "ensemble/*.pdb" --ligands compounds.smi

For help with a command:
  acd <command> --help

Examples:
  acd dock --help
  acd redock --help
  acd batch --help

Documentation:
  https://github.com/nyelidl/anyone-docking

Report problems:
  https://github.com/nyelidl/anyone-docking/issues
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
import textwrap
from pathlib import Path

# ── package version ───────────────────────────────────────────────────────────
try:
    from importlib.metadata import version as _pkg_version
    _VERSION = _pkg_version("anyonecandock")
except Exception:
    _VERSION = "unknown"

# ── optional pretty-print ─────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    _RICH = True
    _con = Console()
except ImportError:
    _RICH = False

    class _FakeConsole:
        def print(self, *a, **kw): print(*a)
        def log(self, *a, **kw): print(*a)
        def rule(self, *a, **kw): print("─" * 60)
    _con = _FakeConsole()


# ── helpers ───────────────────────────────────────────────────────────────────

def _banner():
    txt = textwrap.dedent("""\
        ╔══════════════════════════════════════════════════════╗
        ║  anyone can dock  —  AutoDock Vina 1.2.7 + pKaNET   ║
        ╚══════════════════════════════════════════════════════╝""")
    if _RICH:
        _con.print(f"[bold cyan]{txt}[/bold cyan]")
    else:
        print(txt)


def _ok(msg: str):
    prefix = "[green]✓[/green]" if _RICH else "✓"
    _con.print(f"{prefix}  {msg}")


def _warn(msg: str):
    prefix = "[yellow]⚠[/yellow]" if _RICH else "⚠"
    _con.print(f"{prefix}  {msg}")


def _print_heme_report(result: dict) -> None:
    centers = result.get("prep_meta", {}).get("heme_centers", [])
    for index, center in enumerate(centers, 1):
        _step(f"Heme center {index}")
        print(f"System state       : {center['selected_state']}")
        print(f"Heme               : {center['resname']} {center['chain']} {center['resid']}")
        print(f"Fe                 : {center['fe_name']} atom {center['fe_serial']}")
        if center.get("oxo_serial"):
            print(f"Oxo                : {center['oxo_name']} atom {center['oxo_serial']}")
            print(f"Fe-O               : {center['fe_o_distance']:.2f} A")
            if center.get("oxo_h_serial"):
                print(f"Axial O-H          : removed atom {center['oxo_h_serial']} ({center['oh_distance']:.2f} A)")
            else:
                print("Axial O-H          : none")
        else:
            print("Oxo                : none")
            print("Axial O-H cleanup  : not applicable")
        print(f"Proximal Cys       : {center['cys_resname']} {center['cys_chain']}:{center['cys_resid']} SG")
        print(f"Fe-S               : {center['fe_s_distance']:.2f} A")
        print(f"Charge parameters  : {center['parameter_set']}")
        print(f"Mapped heme atoms  : {center['mapped_heme_atoms']}")
        print(f"Mapped Cys atoms   : {center['mapped_cys_atoms']}")
        print(f"Charge sum         : {center['final_charge']:.4f}")
        print(f"Expected charge    : {center['expected_charge']:.4f}")
        print("PDBQT validation   : PASS")


def _conformer_seed_text(result: dict) -> str:
    seed = result.get("conformer_seed")
    mode = result.get("conformer_seed_mode", "fixed" if seed is not None else "random")
    if mode == "random" and seed is not None:
        return f"random (resolved seed {seed})"
    if mode == "not_used":
        return "not used (input already had 3D coordinates)"
    return str(seed) if seed is not None else "random"


class ACDCLIError(SystemExit):
    def __init__(self, message: str):
        super().__init__(1)
        self.message = message


def _err(msg: str):
    prefix = "[red]✗[/red]" if _RICH else "✗"
    _con.print(f"{prefix}  {msg}")
    raise ACDCLIError(msg)


def _pkanet_kwargs(args) -> dict:
    """Map CLI options to the authoritative core.prepare_ligand interface."""
    return {
        "use_pubchem": not getattr(args, "no_pubchem", False),
        "max_tautomers": getattr(args, "max_tautomers", 8),
        "ph_window": getattr(args, "ph_window", 1.0),
        "pkanet_selection_mode": getattr(args, "pkanet_selection", "auto_recommended"),
        "pkanet_manual_rank": getattr(args, "pkanet_rank", None),
        "conformer_seed": (
            getattr(args, "conformer_seed", None)
            if getattr(args, "conformer_seed", None) not in (None, 0)
            else None
        ),
    }


def _target_ph(value: str) -> float:
    ph = float(value)
    if not 0.0 <= ph <= 14.0:
        raise argparse.ArgumentTypeError("target pH must be between 0 and 14")
    return ph


def _box_size(value: str) -> int:
    size = int(value)
    if size <= 0:
        raise argparse.ArgumentTypeError("box size must be a positive integer")
    return size


def _nonnegative_seed(value: str) -> int:
    seed = int(value)
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be 0 (random) or a positive integer")
    return seed


def _receptor_box_args(core, raw_pdb: str, args) -> tuple:
    if getattr(args, "blind", False):
        center, size = core.calculate_blind_box(raw_pdb, padding=4.0)
        _ok(
            f"Blind box — center ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}); "
            f"size {size[0]} × {size[1]} × {size[2]} Å (4.0 Å padding)"
        )
        return "manual", center, size
    center = (float(args.cx or 0), float(args.cy or 0), float(args.cz or 0))
    return args.center, center, (args.bx, args.by, args.bz)


def _step(msg: str):
    if _RICH:
        _con.rule(f"[bold]{msg}[/bold]")
    else:
        print(f"\n{'─'*60}\n  {msg}\n{'─'*60}")


def _import_core():
    """Import core.py from the same package directory."""
    try:
        from anyonecandock import core
        return core
    except ImportError:
        here = Path(__file__).resolve().parent
        import importlib.util
        spec = importlib.util.spec_from_file_location("core", here / "core.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def _spinner(label: str):
    if _RICH:
        return Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]{label}[/cyan]"),
            TimeElapsedColumn(),
            transient=True,
        )
    import contextlib
    @contextlib.contextmanager
    def _fake():
        print(f"⏳  {label} …")
        yield None
    return _fake()


def _pubchem_smiles(name: str) -> str:
    """Look up a compound by name on PubChem and return its SMILES."""
    import requests
    from urllib.parse import quote

    _ok(f"Searching PubChem for '{name}' …")

    # Step 1: get CID from name
    r = requests.get(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quote(name)}/cids/JSON",
        timeout=10,
    )
    if r.status_code != 200:
        _err(f"Compound '{name}' not found on PubChem (HTTP {r.status_code})")

    cid = r.json()["IdentifierList"]["CID"][0]
    _ok(f"Found CID: {cid}")

    # Step 2: try each property key individually
    for prop in ["IsomericSMILES", "CanonicalSMILES"]:
        try:
            r2 = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                f"{cid}/property/{prop}/JSON",
                timeout=10,
            )
            if r2.status_code == 200:
                props = r2.json().get("PropertyTable", {}).get("Properties", [{}])
                smiles = props[0].get(prop, "") if props else ""
                if smiles:
                    _ok(f"Found SMILES: {smiles}")
                    return smiles
        except Exception:
            continue

    # Step 3: fallback — scrape from full compound JSON
    try:
        r3 = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/JSON",
            timeout=15,
        )
        if r3.status_code == 200:
            compounds = r3.json().get("PC_Compounds", [{}])
            for prop_item in compounds[0].get("props", []):
                label = prop_item.get("urn", {}).get("label", "")
                if "SMILES" in label:
                    val    = prop_item.get("value", {})
                    smiles = val.get("sval") or val.get("string") or ""
                    if smiles:
                        _ok(f"Found SMILES (fallback): {smiles}")
                        return smiles
    except Exception:
        pass

    _err(f"PubChem returned no usable SMILES for '{name}'")


def _resolve_ligand(args, name: str) -> tuple[str, str]:
    """Return (smiles, name) from --smiles or --compound."""
    if getattr(args, "ligand_file", None):
        return "", name  # file path — caller handles prepare_ligand_from_file

    smiles   = getattr(args, "smiles",   None) or ""
    compound = getattr(args, "compound", None) or ""

    if compound and not smiles:
        smiles = _pubchem_smiles(compound)
        if not name or name == "LIG":
            name = compound.replace(" ", "_")

    if not smiles:
        _err("Provide --smiles <SMILES>,  --compound <name>,  or  --ligand-file <path>")

    return smiles, name


def _print_scores(scores: list, ref_score: float | None = None,
                  pose_mols: list | None = None,
                  crystal_pdb: str | None = None,
                  core=None,
                  sort_by_rmsd: bool = False):
    """Print docking score table with RMSD vs crystal when available.

    When ``sort_by_rmsd`` is True (redocking validation), rows are ordered by
    RMSD to the crystal pose (lowest first) instead of by Vina pose number.
    Poses without an RMSD value sink to the bottom.
    """
    if not scores:
        _warn("No scores returned.")
        return

    # compute RMSD vs crystal for each pose
    rmsds: dict = {}
    if core and pose_mols and crystal_pdb and os.path.exists(crystal_pdb):
        for s in scores:
            idx = (s["pose"] - 1) if s["pose"] else 0
            if idx < len(pose_mols):
                try:
                    r = core.calc_rmsd_heavy(pose_mols[idx], crystal_pdb)
                    rmsds[s["pose"]] = round(r, 2) if r is not None else None
                except Exception:
                    rmsds[s["pose"]] = None

    # Redocking view: order rows by RMSD to crystal (lowest = best fit),
    # not by Vina pose number. Poses with no RMSD fall to the bottom.
    if sort_by_rmsd and rmsds:
        rows = sorted(
            scores,
            key=lambda s: (rmsds.get(s["pose"]) is None,
                           rmsds.get(s["pose"]) if rmsds.get(s["pose"]) is not None
                           else float("inf")),
        )
    else:
        rows = scores

    if _RICH:
        t = Table(title="Docking Scores", show_header=True, header_style="bold cyan")
        t.add_column("Pose", style="dim")
        t.add_column("Affinity (kcal/mol)", justify="right")
        if rmsds:
            t.add_column("RMSD vs crystal (Å)", justify="right")
        for s in rows:
            aff   = s["affinity"]
            aclr  = "green" if aff < -8 else "yellow" if aff < -6 else "red"
            row   = [str(s["pose"]), f"[{aclr}]{aff:.2f}[/{aclr}]"]
            if rmsds:
                r = rmsds.get(s["pose"])
                if r is None:
                    row.append("—")
                else:
                    rclr = "green" if r <= 2.0 else "yellow" if r <= 3.0 else "red"
                    row.append(f"[{rclr}]{r:.2f}[/{rclr}]")
            t.add_row(*row)
        _con.print(t)
    else:
        header = f"\n{'Pose':>6}  {'Affinity':>16}"
        if rmsds:
            header += f"  {'RMSD vs crystal':>16}"
        print(header)
        print("─" * (48 + (18 if rmsds else 0)))
        for s in rows:
            line = f"{s['pose']:>6}  {s['affinity']:>16.2f}"
            if rmsds:
                r = rmsds.get(s["pose"])
                line += f"  {str(round(r, 2)) if r is not None else '—':>16}"
            print(line)

    best  = min(s["affinity"] for s in scores)
    label = ("Very strong" if best < -11 else "Strong" if best < -9
             else "Moderate" if best < -7 else "Weak")
    _ok(f"Best pose: {best:.2f} kcal/mol  ({label} predicted binding)")
    if ref_score is not None:
        _ok(f"vs reference: {best - ref_score:+.2f} kcal/mol")
    if rmsds:
        _valid = [v for v in rmsds.values() if v is not None]
        # Redock view headlines the lowest RMSD (row 1 of the sorted table);
        # normal docking keeps reporting the top-affinity pose's RMSD.
        if sort_by_rmsd and _valid:
            r1 = min(_valid)
        else:
            r1 = rmsds.get(scores[0]["pose"])
        if r1 is not None:
            verdict = ("✓ good reproduction" if r1 <= 2.0
                       else "⚠ moderate" if r1 <= 3.0
                       else "✗ differs from crystal")
            label2 = "Best RMSD vs crystal" if sort_by_rmsd else "Best pose RMSD vs crystal"
            _ok(f"{label2}: {r1:.2f} Å  ({verdict})")


def _write_scores_csv(scores: list, path: str, lig_name: str = "",
                      rmsds: dict | None = None,
                      sort_by_rmsd: bool = False):
    rows = scores
    if sort_by_rmsd and rmsds:
        rows = sorted(
            scores,
            key=lambda s: (rmsds.get(s["pose"]) is None,
                           rmsds.get(s["pose"]) if rmsds.get(s["pose"]) is not None
                           else float("inf")),
        )
    with open(path, "w", newline="") as fh:
        fieldnames = ["ligand", "pose", "affinity", "rmsd_vs_crystal"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for s in rows:
            r = (rmsds or {}).get(s["pose"])
            w.writerow({
                "ligand":           lig_name,
                "pose":             s["pose"],
                "affinity":         s["affinity"],
                "rmsd_vs_crystal":  round(r, 2) if r is not None else "",
            })


def _show_and_choose_reference_ligand(core, raw: str, args, *, require_ligand: bool = False,
                                      command_label: str = "dock"):
    """Scan HETATM ligands and choose the reference ligand for receptor prep.

    Behavior:
      - If there is one ligand-like HETATM, use it automatically.
      - If there are multiple ligand-like HETATMs and the user did not give an
        exact selector, pause and ask which ligand index should define
        --center auto.
      - If --resname is given and it matches exactly one ligand, use it.
      - If --resname is given but matches multiple copies, ask which copy.

    Returns: (hetatm_policy, reference_hetatm_key, chosen_row)
      - hetatm_policy/reference_hetatm_key can be passed directly to
        core.prepare_receptor().
      - If no scan/choice is needed, returns (None, "", None).
    """
    center_mode = (getattr(args, "center", "auto") or "auto").lower()
    preferred   = (getattr(args, "resname", "") or "").strip().upper()
    assume_yes  = bool(getattr(args, "yes", False))

    # Only the auto/resname path needs a co-crystal reference. Manual and
    # selection centers can be used without choosing a HETATM reference.
    if center_mode != "auto" and not preferred and not require_ligand:
        return None, "", None

    print("⏳  Scanning HETATM records …")
    try:
        het_rows = core.scan_hetatm_residues(raw)
    except Exception as _se:
        het_rows = []
        _warn(f"HETATM scan error: {_se}")

    lig_rows = [r for r in het_rows if r.get("type_guess") == "ligand"]

    def _example_manual_command() -> str:
        """Return a command example matching the current subcommand label."""
        rec_token = (getattr(args, "receptor", None)
                     or getattr(args, "pdb", None)
                     or getattr(args, "file", None)
                     or "apo_protein.pdb")
        if command_label == "receptor":
            return (
                f"acd receptor --file {rec_token} --center manual "
                "--cx 12.3 --cy 45.6 --cz -7.8 --bx 20 --by 20 --bz 20"
            )
        if command_label == "batch":
            return (
                f"acd batch --receptor {rec_token} --center manual "
                "--cx 12.3 --cy 45.6 --cz -7.8 --bx 20 --by 20 --bz 20 "
                "--ligands compounds.smi"
            )
        if command_label == "codock":
            return (
                f"acd codock --receptor {rec_token} --center manual "
                "--cx 12.3 --cy 45.6 --cz -7.8 --bx 20 --by 20 --bz 20 "
                "--ligands pair.smi"
            )
        return (
            f"acd dock --receptor {rec_token} --center manual "
            "--cx 12.3 --cy 45.6 --cz -7.8 --bx 20 --by 20 --bz 20 "
            "--smiles \"CCO\" --name test"
        )

    def _prompt_manual_center_no_ligand():
        """Ask for manual grid center/box when --center auto has no ligand."""
        _warn("No co-crystal ligand candidate found in this structure.")
        print(
            "\nCannot use --center auto because no ligand-like HETATM residue "
            "is available to define the docking box.\n"
            "Please specify the grid center coordinates and box size.\n\n"
            "Example command:\n"
            f"  {_example_manual_command()}\n"
        )

        # If the user already supplied coordinates while forgetting to switch
        # --center manual, use them safely instead of falling through to 0,0,0.
        have_xyz = (
            getattr(args, "cx", None) is not None and
            getattr(args, "cy", None) is not None and
            getattr(args, "cz", None) is not None
        )
        if have_xyz:
            args.center = "manual"
            _warn(
                "Using the provided --cx/--cy/--cz values and switching "
                "center mode to manual."
            )
            _ok(
                f"Manual center : ({float(args.cx):.3f}, "
                f"{float(args.cy):.3f}, {float(args.cz):.3f})"
            )
            _ok(f"Box size      : ({args.bx}, {args.by}, {args.bz}) Å")
            return

        if assume_yes:
            _err(
                "No co-crystal ligand was found, so --center auto cannot be used.\n\n"
                "  --yes cannot choose a grid center when the structure is apo.\n"
                "  Re-run with explicit manual coordinates, for example:\n"
                f"    {_example_manual_command()}"
            )

        def _read_three_floats(prompt: str, *, default=None):
            while True:
                try:
                    s = input(prompt).strip()
                    if s == "" and default is not None:
                        return default
                    parts = s.replace(",", " ").split()
                    if len(parts) != 3:
                        print("  ✗  Please enter exactly three numbers, separated by spaces.")
                        continue
                    vals = tuple(float(x) for x in parts)
                    return vals
                except ValueError:
                    print("  ✗  Invalid number — example: 12.3 45.6 -7.8")
                except EOFError:
                    _err(
                        "Manual grid input is required but no interactive input is available.\n\n"
                        "  Re-run with explicit coordinates and box size, for example:\n"
                        f"    {_example_manual_command()}"
                    )
                except KeyboardInterrupt:
                    print()
                    _err("Manual center input cancelled by user.")

        xyz = _read_three_floats(
            "  Enter center_x center_y center_z "
            "(example: 12.3 45.6 -7.8): "
        )
        default_box = (float(args.bx), float(args.by), float(args.bz))
        box = _read_three_floats(
            f"  Enter box size_x size_y size_z in Å "
            f"(Enter = {args.bx} {args.by} {args.bz}): ",
            default=default_box,
        )
        if any(v <= 0 for v in box):
            _err("Box sizes must be positive numbers.")

        args.center = "manual"
        args.cx, args.cy, args.cz = xyz
        args.bx, args.by, args.bz = box
        _ok(f"Manual center : ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})")
        _ok(f"Box size      : ({box[0]}, {box[1]}, {box[2]}) Å")

    if not lig_rows:
        if require_ligand:
            _err(
                "No co-crystal ligand was found in this structure.\n\n"
                "  This operation requires a small-molecule ligand (HETATM) "
                "bound in the input structure."
            )
        if center_mode == "auto":
            _prompt_manual_center_no_ligand()
        return None, "", None

    def _print_candidates(rows):
        print(
            f"\n  Found {len(rows)} co-crystal ligand candidate(s) "
            f"in this structure:\n"
        )
        for i, r in enumerate(rows):
            print(
                f"    [{i}]  {r['resname']:>4}  "
                f"chain {r['chain'] or '-':>2}  "
                f"resid {r['resid']:>5}  "
                f"({r['n_atoms']:>3} heavy atoms)"
            )
        print()

    def _choose_interactively(rows, *, prompt_label: str):
        """Ask the user to choose one ligand row by index."""
        if len(rows) == 1:
            return rows[0]

        if assume_yes:
            chosen = rows[0]
            _warn(
                f"Multiple ligands detected; --yes was used, so {command_label} "
                f"will use [0] {chosen['resname']} chain {chosen['chain'] or '-'} "
                f"resid {chosen['resid']} as the --center auto reference."
            )
            return chosen

        _warn(
            f"Multiple ligands detected; please choose which ligand should define "
            f"the --center auto grid for {command_label}."
        )
        while True:
            try:
                raw_sel = input(
                    f"  Select {prompt_label} [0–{len(rows) - 1}] "
                    f"(Enter = 0): "
                ).strip()
                idx = 0 if raw_sel == "" else int(raw_sel)
                if 0 <= idx < len(rows):
                    return rows[idx]
                print(f"  ✗  Please enter a number between 0 and {len(rows) - 1}.")
            except ValueError:
                print("  ✗  Invalid input — please type a number.")
            except EOFError:
                _err(
                    "Ligand confirmation is required but no interactive input is available.\n\n"
                    "  Use one of these instead:\n"
                    "    • --resname <RESNAME>  if the residue name is unique\n"
                    "    • --center selection --sel \"resname XXX and chain A and resid 123\"\n"
                    "    • --yes  to accept the default [0] ligand in non-interactive runs"
                )
            except KeyboardInterrupt:
                print()
                _err("Ligand selection cancelled by user.")

    # Always show the candidate table when there is more than one possible ligand
    # or when the user gave --resname and needs to see what matched.
    if len(lig_rows) > 1 or preferred:
        _print_candidates(lig_rows)

    if preferred:
        matches = [r for r in lig_rows if r["resname"].upper() == preferred]
        if not matches:
            _err(
                f"--resname '{preferred}' was not found among the detected "
                "drug-like ligands.\n\n"
                "  Ligands detected:\n"
                + "\n".join(
                    f"    [{i}]  {r['resname']:>4}  "
                    f"chain {r['chain'] or '-':>2}  "
                    f"resid {r['resid']:>5}  "
                    f"({r['n_atoms']:>3} heavy atoms)"
                    for i, r in enumerate(lig_rows)
                )
            )
        if len(matches) > 1:
            print(
                f"\n  --resname {preferred} matched {len(matches)} ligand copies:\n"
            )
            for i, r in enumerate(matches):
                print(
                    f"    [{i}]  {r['resname']:>4}  "
                    f"chain {r['chain'] or '-':>2}  "
                    f"resid {r['resid']:>5}  "
                    f"({r['n_atoms']:>3} heavy atoms)"
                )
            print()
            chosen_row = _choose_interactively(matches, prompt_label=f"{preferred} copy")
        else:
            chosen_row = matches[0]
            _ok(
                f"Matched --resname {preferred}: "
                f"chain {chosen_row['chain'] or '-'}  "
                f"resid {chosen_row['resid']}"
            )
    else:
        chosen_row = _choose_interactively(lig_rows, prompt_label="ligand")

    _ok(
        f"Reference ligand : {chosen_row['resname']}  "
        f"chain {chosen_row['chain'] or '-'}  "
        f"resid {chosen_row['resid']}  "
        f"({chosen_row['n_atoms']} heavy atoms)"
    )

    chosen_key = chosen_row["key"]
    hetatm_policy = {}
    for hr in het_rows:
        key = hr["key"]
        tg  = hr.get("type_guess", "other")
        if key == chosen_key:
            hetatm_policy[key] = "reference"
        elif tg == "ligand":
            hetatm_policy[key] = "remove"
        elif tg == "water":
            hetatm_policy[key] = "remove"
        elif tg in ("metal", "heme/cofactor", "cofactor"):
            hetatm_policy[key] = "keep"
        else:
            hetatm_policy[key] = "remove"

    return hetatm_policy, chosen_key, chosen_row


# ── sub-command: receptor ─────────────────────────────────────────────────────

def _assess_receptor_pocket(raw, result, folder):
    from anyonecandock.pocket_completeness import assess_missing_residues, save_report
    report = assess_missing_residues(raw, result.get("ligand_pdb_path"), result.get("rec_fh"))
    save_report(report, folder)
    return report


def _check_receptor_pocket(summary, *, enforce=True):
    """Saved summaries retain the gate; legacy summaries are explicitly unknown."""
    report = summary.get("pocket_report") or {
        "status": "unknown", "blocked": False,
        "reason": "Missing-residue status unknown: this saved receptor has no assessment. Re-run receptor preparation to assess it.",
    }
    from anyonecandock.pocket_completeness import normalize_report
    report = normalize_report(report)
    message = report.get("reason", "Missing-residue status unknown.")
    (_ok if report.get("status") == "complete" else _warn)(message)
    for gap in report.get("gaps", []):
        _warn(f"Chain {gap['chain'] or '(blank)'}: {gap['missing_residues']}; "
              f"boundaries {gap['before']} / {gap['after']}; "
              f"distances {gap['before_distance_A']} / {gap['after_distance_A']} Å; {gap['status']}")
    for atom in report.get("missing_atoms", []):
        _warn(f"Annotated missing atom: {atom['chain'] or '(blank)'}:{atom['residue']} {atom['atom']}")
    for warning in report.get("warnings", []):
        _warn(warning)
    if enforce and (report.get("blocked") or report.get("status") == "near"):
        _err(message)
    return report


def cmd_receptor(args):
    core = _import_core()
    _step("Receptor preparation")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.pdb:
        token = args.pdb.strip().upper()
        fmt   = args.fmt.upper()
        ext   = "cif" if fmt == "CIF" else "pdb"
        raw   = str(out / f"raw.{ext}")
        _ok(f"Downloading {token} ({fmt}) from RCSB …")
        rc, _ = core.run_cmd(["curl", "-sf",
                               f"https://files.rcsb.org/download/{token}.{ext}",
                               "-o", raw])
        if rc != 0 or not os.path.exists(raw) or os.path.getsize(raw) < 200:
            if fmt == "PDB":
                raw_cif = str(out / "raw.cif")
                rc2, _ = core.run_cmd(["curl", "-sf",
                                        f"https://files.rcsb.org/download/{token}.cif",
                                        "-o", raw_cif])
                if rc2 == 0 and os.path.exists(raw_cif) and os.path.getsize(raw_cif) > 200:
                    raw = raw_cif
                    _warn("PDB unavailable; fell back to CIF")
                else:
                    _err(f"Download failed for {token}")
            else:
                _err(f"Download failed for {token}")
    elif args.file:
        raw = args.file
        if not os.path.exists(raw):
            _err(f"File not found: {raw}")
    else:
        _err("Provide --pdb <PDB_ID>  or  --file <path>")

    center_mode, manual_xyz, box_size = _receptor_box_args(core, raw, args)

    hetatm_policy, reference_hetatm_key, chosen_row = _show_and_choose_reference_ligand(
        core, raw, args, require_ligand=False, command_label="receptor"
    )

    print("⏳  Preparing receptor …")
    result = core.prepare_receptor(
        raw_pdb              = raw,
        wdir                 = out,
        center_mode          = center_mode,
        manual_xyz           = manual_xyz,
        prody_sel            = args.sel or "",
        box_size             = box_size,
        preferred_ligand     = getattr(args, "resname", "") or "",
        hetatm_policy        = hetatm_policy,
        reference_hetatm_key = reference_hetatm_key,
    )

    if not result["success"]:
        for line in result.get("log", []):
            _warn(line) if line.startswith("⚠") else print(line)
        _err(f"Receptor preparation failed: {result['error']}")

    _print_heme_report(result)
    pocket_report = _assess_receptor_pocket(raw, result, out)
    _check_receptor_pocket({"pocket_report": pocket_report}, enforce=False)

    cx, cy, cz = result["cx"], result["cy"], result["cz"]
    _ok(f"Receptor PDB : {result['rec_fh']}")
    _ok(f"PDBQT        : {result['rec_pdbqt']}")
    _ok(f"Box config   : {result['config_txt']}")
    _ok(f"Grid center  : ({cx:.3f}, {cy:.3f}, {cz:.3f})")
    if result.get("cocrystal_ligand_id"):
        _ok(f"Co-crystal   : {result['cocrystal_ligand_id']}")

    summary = {
        "pocket_report": pocket_report,
        "rec_fh":              result["rec_fh"],
        "rec_pdbqt":           result["rec_pdbqt"],
        "config_txt":          result["config_txt"],
        "cx": cx, "cy": cy, "cz": cz,
        "sx": result["sx"], "sy": result["sy"], "sz": result["sz"],
        "ligand_pdb_path":     result.get("ligand_pdb_path"),
        "cocrystal_ligand_id": result.get("cocrystal_ligand_id", ""),
        "heme_centers":        result.get("prep_meta", {}).get("heme_centers", []),
    }
    summary_path = str(out / "receptor_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    _ok(f"Summary JSON : {summary_path}")
    _ok("Docking       : NOT STARTED (receptor preparation-only command)")


def _validate_cli_ligand_file(core, path):
    try:
        return core.validate_ligand_file(path)["path"]
    except (ValueError, OSError) as exc:
        _err(f"{exc}\nDocking was not started.")


def _log_file_ligand_result(result):
    if result.get("direct_pdbqt") or result.get("atom_mapping_report") or not result["success"]:
        for line in result.get("log", []):
            print(line)


# ── sub-command: ligand ────────────────────────────────────────────────────────

def cmd_ligand(args):
    core = _import_core()
    _step("Ligand preparation")

    out  = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or "LIG"
    ph   = args.ph
    prot_mode = "neutral" if args.neutral else "pkanet"

    print(f"⏳  Preparing {name} at pH {ph} …")

    ligand_file = getattr(args, "file", None) or getattr(args, "ligand_file", None)
    if ligand_file:
        ligand_file = _validate_cli_ligand_file(core, ligand_file)
        result = core.prepare_ligand_from_file(
            ligand_file, name, out,
            add_h = not getattr(args, "no_add_h", False),
            conformer_seed=getattr(args, "conformer_seed", None),
        )
    else:
        smiles, name = _resolve_ligand(args, name)
        result = core.prepare_ligand(
            smiles        = smiles,
            name          = name,
            ph            = ph,
            wdir          = out,
            mode          = prot_mode,
            **_pkanet_kwargs(args),
        )

    _log_file_ligand_result(result)
    if not result["success"]:
        _err(f"Ligand preparation failed: {result['error']}")

    _ok(f"PDBQT         : {result['pdbqt']}")
    _ok(f"SDF (3D)      : {result['sdf']}")
    if not result.get("direct_pdbqt"):
        _ok(f"Formal charge : {int(result.get('net_charge', result.get('charge', 0))):+d}")
    _ok(f"Conformer seed: {_conformer_seed_text(result)}")
    _ok(f"SMILES (final): {result.get('prot_smiles', '')}")
    if result.get("pkanet_ranked_csv"):
        _ok(f"pKaNET CSV    : {result['pkanet_ranked_csv']}")


# ── sub-command: dock ─────────────────────────────────────────────────────────

def cmd_dock(args):
    core = _import_core()
    if getattr(args, "ligand_file", None):
        args.ligand_file = _validate_cli_ligand_file(core, args.ligand_file)
    _banner()
    _step("Single-ligand docking")

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Receptor ──────────────────────────────────────────────────────
    _step("Step 1 — Receptor")

    if args.receptor_json:
        with open(args.receptor_json) as fh:
            rec_summary = json.load(fh)
        _ok(f"Using saved receptor: {rec_summary['rec_fh']}")
    else:
        rec_out = out / "receptor"
        rec_out.mkdir(exist_ok=True)

        if os.path.exists(args.receptor):
            raw = os.path.abspath(args.receptor)
        else:
            token   = args.receptor.strip().upper()
            fmt_ext = "cif" if args.fmt.upper() == "CIF" else "pdb"
            raw     = str(rec_out / f"raw.{fmt_ext}")
            _ok(f"Downloading {token} …")
            rc, _ = core.run_cmd(["curl", "-sf",
                                   f"https://files.rcsb.org/download/{token}.{fmt_ext}",
                                   "-o", raw])
            if rc != 0 or not os.path.exists(raw) or os.path.getsize(raw) < 200:
                raw_cif = str(rec_out / "raw.cif")
                rc2, _ = core.run_cmd(["curl", "-sf",
                                        f"https://files.rcsb.org/download/{token}.cif",
                                        "-o", raw_cif])
                if rc2 == 0 and os.path.exists(raw_cif) and os.path.getsize(raw_cif) > 200:
                    raw = raw_cif
                    _warn("PDB unavailable; fell back to CIF")
                else:
                    _err(f"Download failed for {token}")

        center_mode, manual_xyz, box_size = _receptor_box_args(core, raw, args)

        hetatm_policy, reference_hetatm_key, chosen_row = _show_and_choose_reference_ligand(
            core, raw, args, require_ligand=False, command_label="dock"
        )

        print("⏳  Preparing receptor …")
        rec_result = core.prepare_receptor(
            raw_pdb              = raw,
            wdir                 = rec_out,
            center_mode          = center_mode,
            manual_xyz           = manual_xyz,
            prody_sel            = args.sel or "",
            box_size             = box_size,
            preferred_ligand     = getattr(args, "resname", "") or "",
            hetatm_policy        = hetatm_policy,
            reference_hetatm_key = reference_hetatm_key,
        )
        if not rec_result["success"]:
            for line in rec_result.get("log", []):
                _warn(line) if line.startswith("⚠") else print(line)
            _err(f"Receptor prep failed: {rec_result['error']}")
        _print_heme_report(rec_result)
        pocket_report = _assess_receptor_pocket(raw, rec_result, rec_out)

        rec_summary = {
            "pocket_report": pocket_report,
            "rec_fh":              rec_result["rec_fh"],
            "rec_pdbqt":           rec_result["rec_pdbqt"],
            "config_txt":          rec_result["config_txt"],
            "cx": rec_result["cx"], "cy": rec_result["cy"], "cz": rec_result["cz"],
            "sx": rec_result["sx"], "sy": rec_result["sy"], "sz": rec_result["sz"],
            "ligand_pdb_path":     rec_result.get("ligand_pdb_path"),
            "cocrystal_ligand_id": rec_result.get("cocrystal_ligand_id", ""),
        }
        with open(str(rec_out / "receptor_summary.json"), "w") as fh:
            json.dump(rec_summary, fh, indent=2)

        cx, cy, cz = rec_result["cx"], rec_result["cy"], rec_result["cz"]
        _ok(f"Receptor ready — center ({cx:.2f}, {cy:.2f}, {cz:.2f})")
        if rec_result.get("cocrystal_ligand_id"):
            _ok(f"Co-crystal ligand: {rec_result['cocrystal_ligand_id']}")

    _check_receptor_pocket(rec_summary)
    rec_pdbqt  = os.path.abspath(rec_summary["rec_pdbqt"])
    config_txt = os.path.abspath(rec_summary["config_txt"])

    # ── Step 2: Vina binary ───────────────────────────────────────────────────
    print("⏳  Checking Vina binary …")
    vina_path, vina_err = core.get_vina_binary()
    if vina_path is None:
        _err(f"Could not get Vina binary: {vina_err}")
    _ok(f"Vina: {vina_path}")

    # ── Step 3: Ligand ────────────────────────────────────────────────────────
    _step("Step 2 — Ligand")

    lig_out   = out / "ligand"
    lig_out.mkdir(exist_ok=True)
    name      = args.name or "LIG"
    ph        = args.ph
    prot_mode = "neutral" if args.neutral else "pkanet"

    if getattr(args, "ligand_file", None):
        print(f"⏳  Preparing {name} from file …")
        lig_result = core.prepare_ligand_from_file(
            args.ligand_file, name, lig_out,
            add_h = not getattr(args, "no_add_h", False),
            conformer_seed=getattr(args, "conformer_seed", None),
        )
    else:
        smiles, name = _resolve_ligand(args, name)
        print(f"⏳  Preparing {name} at pH {ph} …")
        lig_result = core.prepare_ligand(
            smiles        = smiles,
            name          = name,
            ph            = ph,
            wdir          = lig_out,
            mode          = prot_mode,
            **_pkanet_kwargs(args),
        )

    _log_file_ligand_result(lig_result)
    if not lig_result["success"]:
        _err(f"Ligand prep failed: {lig_result['error']}")

    prot_smiles = lig_result.get("prot_smiles", "")
    charge      = int(lig_result.get("net_charge", lig_result.get("charge", 0)))
    _ok(f"Ligand SMILES : {prot_smiles}")
    if not lig_result.get("direct_pdbqt"):
        _ok(f"Formal charge : {charge:+d}")
    _ok(f"Conformer seed: {_conformer_seed_text(lig_result)}")

    # ── Step 4: Optional redocking reference ─────────────────────────────────
    ref_score = None
    if args.redock_smiles:
        _step("Step 2b — Redocking reference")
        pts    = args.redock_smiles.strip().split(None, 1)
        rd_smi = pts[0]
        rd_nm  = pts[1].replace(" ", "_") if len(pts) > 1 else "redock"
        print(f"⏳  Docking reference {rd_nm} …")
        rd_prep = core.prepare_ligand(
            rd_smi, "redock_" + rd_nm, ph, lig_out,
            mode=prot_mode, **_pkanet_kwargs(args),
        )
        if rd_prep["success"]:
            rd_dock = core.run_vina(
                receptor_pdbqt = rec_pdbqt,
                ligand_pdbqt   = os.path.abspath(rd_prep["pdbqt"]),
                config_txt     = config_txt,
                vina_path      = vina_path,
                exhaustiveness = args.exhaustiveness,
                n_modes        = args.poses,
                energy_range   = int(args.energy_range),
                wdir           = lig_out,
                out_name       = "redock_" + rd_nm,
                seed           = args.seed,
            )
            if rd_dock["success"] and rd_dock["top_score"] is not None:
                ref_score = rd_dock["top_score"]
                _ok(f"Reference score: {ref_score:.2f} kcal/mol")

    # ── Step 5: Docking ───────────────────────────────────────────────────────
    _step("Step 3 — AutoDock Vina")

    print(f"⏳  Running Vina (exhaustiveness={args.exhaustiveness}) …")
    dock = core.run_vina(
        receptor_pdbqt = rec_pdbqt,
        ligand_pdbqt   = os.path.abspath(lig_result["pdbqt"]),
        **({"ligand_input_sha256": lig_result["input_sha256"]} if lig_result.get("direct_pdbqt") else {}),
        config_txt     = config_txt,
        vina_path      = vina_path,
        exhaustiveness = args.exhaustiveness,
        n_modes        = args.poses,
        energy_range   = int(args.energy_range),
        wdir           = out,
        out_name       = name,
        seed           = args.seed,
    )

    if not dock["success"]:
        _err(f"Vina failed: {dock['error']}\n{dock['log'][:400]}")

    # ── Bond-order fix ────────────────────────────────────────────────────────
    pv_sdf = str(out / f"{name}_pv_ready.sdf")
    if prot_smiles:
        core.fix_sdf_bond_orders(dock["out_sdf"], prot_smiles, pv_sdf)
    if not os.path.exists(pv_sdf) or os.path.getsize(pv_sdf) < 10:
        pv_sdf = dock["out_sdf"]

    # ── Load pose mols for RMSD ───────────────────────────────────────────────
    sdf_src   = pv_sdf if (os.path.exists(pv_sdf) and os.path.getsize(pv_sdf) > 10) \
                else dock["out_sdf"]
    pose_mols = core.load_mols_from_sdf(sdf_src, sanitize=False)
    crystal_pdb = rec_summary.get("ligand_pdb_path") or ""

    # ── Compute RMSD dict for CSV ─────────────────────────────────────────────
    rmsds: dict = {}
    if crystal_pdb and os.path.exists(crystal_pdb):
        for s in dock["scores"]:
            idx = (s["pose"] - 1) if s["pose"] else 0
            if idx < len(pose_mols):
                try:
                    r = core.calc_rmsd_heavy(pose_mols[idx], crystal_pdb)
                    rmsds[s["pose"]] = round(r, 2) if r is not None else None
                except Exception:
                    rmsds[s["pose"]] = None

    # ── Results ───────────────────────────────────────────────────────────────
    _step("Results")
    _print_scores(
        dock["scores"],
        ref_score   = ref_score,
        pose_mols   = pose_mols,
        crystal_pdb = crystal_pdb if crystal_pdb and os.path.exists(crystal_pdb) else None,
        core        = core,
    )

    scores_csv = str(out / f"{name}_scores.csv")
    _write_scores_csv(dock["scores"], scores_csv, lig_name=name, rmsds=rmsds)

    _ok(f"All poses PDBQT : {dock['out_pdbqt']}")
    _ok(f"All poses SDF   : {dock['out_sdf']}")
    if os.path.exists(pv_sdf) and os.path.getsize(pv_sdf) > 10:
        _ok(f"PV-ready SDF    : {pv_sdf}")
    _ok(f"Scores CSV      : {scores_csv}")

    # ── Optional per-pose SDF ─────────────────────────────────────────────────
    if args.save_poses:
        poses_dir = out / "poses"
        poses_dir.mkdir(exist_ok=True)
        for i, mol in enumerate(pose_mols, 1):
            core.write_single_pose(    mol, str(poses_dir / f"{name}_pose{i}.sdf"))
            core.write_single_pose_pdb(mol, str(poses_dir / f"{name}_pose{i}.pdb"))
        _ok(f"Individual poses: {poses_dir}/")

    # ── Optional 2D diagram ───────────────────────────────────────────────────
    if args.diagram:
        _generate_diagram(
            core       = core,
            rec_fh     = rec_summary["rec_fh"],
            pose_sdf   = pv_sdf if os.path.exists(pv_sdf) else dock["out_sdf"],
            smiles     = prot_smiles,
            lig_name   = name,
            best_score = dock["top_score"],
            pose_idx   = 0,
            out        = out,
            cutoff     = args.diagram_cutoff,
        )


# ── sub-command: batch ────────────────────────────────────────────────────────

def _codock_entries(args):
    """Resolve the unified codock input while retaining the older flags."""
    entries = []

    def add_smi_file(path):
        try:
            lines = Path(path).read_text().splitlines()
        except OSError as exc:
            _err(f"Could not read ligand list '{path}': {exc}")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split(None, 1)
                entries.append(("smiles", parts[0],
                                parts[1] if len(parts) > 1 else f"lig_{len(entries)+1}"))

    for value in getattr(args, "ligand", None) or []:
        value = value.strip()
        if not value:
            _err("Empty --ligand input")
        path = Path(value).expanduser()
        if path.is_file():
            if path.suffix.lower() == ".smi":
                add_smi_file(path)
            else:
                entries.append(("file", str(path), path.stem))
            continue
        if path.suffix.lower() in {".smi", ".sdf", ".mol2", ".pdb", ".pdbqt"}:
            _err(f"Ligand file not found: {value}")
        parts = value.split(None, 1)
        from rdkit import Chem, rdBase
        with rdBase.BlockLogs():
            is_smiles = Chem.MolFromSmiles(parts[0]) is not None
        if is_smiles:
            entries.append(("smiles", parts[0],
                            parts[1] if len(parts) > 1 else f"lig_{len(entries)+1}"))
        else:
            if "/" in value or "\\" in value:
                _err(f"Ligand file not found: {value}")
            if args.no_pubchem:
                _err(f"'{value}' is a compound name; remove --no-pubchem to look it up")
            entries.append(("smiles", _pubchem_smiles(value), value))

    if args.ligands:
        add_smi_file(args.ligands)
    for entry in args.smiles_list or []:
        parts = entry.strip().split(None, 1)
        if parts:
            entries.append(("smiles", parts[0],
                            parts[1] if len(parts) > 1 else f"lig_{len(entries)+1}"))
    for path in args.ligand_files or []:
        entries.append(("file", path, Path(path).stem))
    if len(entries) < 2:
        _err("Simultaneous docking requires at least two ligands")
    return entries


def cmd_codock(args):
    """Prepare a ligand set and dock it as one joint Vina system."""
    core = _import_core()
    _banner()
    _step("Simultaneous ligand docking")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    entries = _codock_entries(args)

    if args.receptor_json:
        with open(args.receptor_json) as fh:
            rec_summary = json.load(fh)
    else:
        if not args.receptor:
            _err("Provide --receptor or --receptor-json")
        rec_out = out / "receptor"
        rec_out.mkdir(exist_ok=True)
        if os.path.exists(args.receptor):
            raw = os.path.abspath(args.receptor)
        else:
            token = args.receptor.strip().upper()
            fmt_ext = "cif" if args.fmt.upper() == "CIF" else "pdb"
            raw = str(rec_out / f"raw.{fmt_ext}")
            rc, _ = core.run_cmd(["curl", "-sf",
                                  f"https://files.rcsb.org/download/{token}.{fmt_ext}", "-o", raw])
            if rc != 0 or not os.path.isfile(raw) or os.path.getsize(raw) < 200:
                _err(f"Download failed for {token}")
        policy, reference_key, _ = _show_and_choose_reference_ligand(
            core, raw, args, require_ligand=False, command_label="codock")
        center_mode, manual_xyz, box_size = _receptor_box_args(core, raw, args)
        rec = core.prepare_receptor(
            raw_pdb=raw, wdir=rec_out, center_mode=center_mode,
            manual_xyz=manual_xyz, prody_sel=args.sel or "", box_size=box_size,
            preferred_ligand=getattr(args, "resname", "") or "",
            hetatm_policy=policy, reference_hetatm_key=reference_key,
        )
        if not rec["success"]:
            _err(f"Receptor prep failed: {rec['error']}")
        pocket_report = _assess_receptor_pocket(raw, rec, rec_out)
        rec_summary = {"rec_pdbqt": rec["rec_pdbqt"],
                       "config_txt": rec["config_txt"], "pocket_report": pocket_report}
    _check_receptor_pocket(rec_summary)

    ligand_paths = []
    manifest_ligands = []
    for index, (kind, value, label) in enumerate(entries, 1):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._-") or f"lig_{index}"
        internal_name = f"{index:02d}_{safe}"
        lig_dir = out / "ligands" / internal_name
        lig_dir.mkdir(parents=True, exist_ok=True)
        if kind == "file":
            prep = core.prepare_ligand_from_file(value, internal_name, lig_dir)
        else:
            prep = core.prepare_ligand(
                value, internal_name, args.ph, lig_dir,
                mode="neutral" if args.neutral else "pkanet",
                **_pkanet_kwargs(args),
            )
        if not prep["success"]:
            _err(f"Ligand {label} preparation failed: {prep['error']}")
        ligand_paths.append(os.path.abspath(prep["pdbqt"]))
        manifest_ligands.append({"label": label, "input": value,
                                 "input_kind": kind, "pdbqt": ligand_paths[-1],
                                 "charge": prep.get("charge")})
        _ok(f"Prepared {label}")

    vina_path, vina_err = core.get_vina_binary()
    if vina_path is None:
        _err(f"Could not get Vina: {vina_err}")
    dock = core.run_vina_multi(
        receptor_pdbqt=os.path.abspath(rec_summary["rec_pdbqt"]),
        ligand_pdbqts=ligand_paths,
        config_txt=os.path.abspath(rec_summary["config_txt"]),
        vina_path=vina_path,
        exhaustiveness=args.exhaustiveness,
        n_modes=args.poses,
        energy_range=args.energy_range,
        wdir=out,
        out_name="joint",
        seed=args.seed,
    )
    if not dock["success"]:
        _err(f"Simultaneous docking failed: {dock['error']}\n{dock.get('log', '')}")
    manifest = {"score_scope": "joint_ligand_complex", "ligands": manifest_ligands,
                "scores": dock["scores"], "top_score": dock["top_score"],
                "out_pdbqt": dock["out_pdbqt"], "vina_log": dock["log"]}
    manifest_path = out / "joint_results.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if dock["top_score"] is not None:
        _ok(f"Top joint score: {dock['top_score']:.2f} kcal/mol")
    else:
        _warn("Vina wrote output, but no joint score was parsed; inspect the log")
    _ok(f"Joint poses: {dock['out_pdbqt']}")
    _ok(f"Joint results: {manifest_path}")


def cmd_batch(args):
    core = _import_core()
    _banner()
    _step("Batch docking")

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # ── Receptor ──────────────────────────────────────────────────────────────
    if args.receptor_json:
        with open(args.receptor_json) as fh:
            rec_summary = json.load(fh)
        _ok(f"Loaded receptor: {rec_summary['rec_fh']}")
    else:
        rec_out = out / "receptor"
        rec_out.mkdir(exist_ok=True)

        if os.path.exists(args.receptor):
            raw = os.path.abspath(args.receptor)
        else:
            token = args.receptor.strip().upper()
            fmt_ext = "cif" if args.fmt.upper() == "CIF" else "pdb"
            raw   = str(rec_out / f"raw.{fmt_ext}")
            _ok(f"Downloading {token} …")
            rc, _ = core.run_cmd(["curl", "-sf",
                                   f"https://files.rcsb.org/download/{token}.{fmt_ext}",
                                   "-o", raw])
            if rc != 0 or not os.path.exists(raw) or os.path.getsize(raw) < 200:
                _err(f"Download failed for {token}")

        hetatm_policy, reference_hetatm_key, chosen_row = _show_and_choose_reference_ligand(
            core, raw, args, require_ligand=False, command_label="batch"
        )

        print("⏳  Preparing receptor …")
        center_mode, manual_xyz, box_size = _receptor_box_args(core, raw, args)
        rec_result = core.prepare_receptor(
            raw_pdb              = raw,
            wdir                 = rec_out,
            center_mode          = center_mode,
            manual_xyz           = manual_xyz,
            prody_sel            = args.sel or "",
            box_size             = box_size,
            preferred_ligand     = getattr(args, "resname", "") or "",
            hetatm_policy        = hetatm_policy,
            reference_hetatm_key = reference_hetatm_key,
        )
        if not rec_result["success"]:
            for line in rec_result.get("log", []):
                _warn(line) if line.startswith("⚠") else print(line)
            _err(f"Receptor prep failed: {rec_result['error']}")
        _print_heme_report(rec_result)
        pocket_report = _assess_receptor_pocket(raw, rec_result, rec_out)

        rec_summary = {
            "pocket_report": pocket_report,
            "rec_fh":    rec_result["rec_fh"],
            "rec_pdbqt": rec_result["rec_pdbqt"],
            "config_txt": rec_result["config_txt"],
            "ligand_pdb_path": rec_result.get("ligand_pdb_path"),
        }
        _ok("Receptor ready")

    _check_receptor_pocket(rec_summary)
    rec_pdbqt   = os.path.abspath(rec_summary["rec_pdbqt"])
    config_txt  = os.path.abspath(rec_summary["config_txt"])
    crystal_pdb = rec_summary.get("ligand_pdb_path") or ""

    # ── Read ligand input ─────────────────────────────────────────────────────
    smiles_pairs: list[tuple[str, str]] = []

    if args.ligands:
        with open(args.ligands) as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pts = line.split(None, 1)
                nm  = pts[1].replace(" ", "_") if len(pts) > 1 else f"lig_{i+1}"
                smiles_pairs.append((pts[0], nm))
    elif args.smiles_list:
        for i, entry in enumerate(args.smiles_list):
            pts = entry.strip().split(None, 1)
            nm  = pts[1].replace(" ", "_") if len(pts) > 1 else f"lig_{i+1}"
            smiles_pairs.append((pts[0], nm))
    else:
        _err("Provide --ligands <file.smi>  or  --smiles-list SMILES [SMILES ...]")

    _ok(f"Ligands to dock: {len(smiles_pairs)}")

    # ── Vina binary ───────────────────────────────────────────────────────────
    print("⏳  Checking Vina binary …")
    vina_path, vina_err = core.get_vina_binary()
    if vina_path is None:
        _err(f"Could not get Vina: {vina_err}")

    # ── Optional reference ────────────────────────────────────────────────────
    ref_score = None
    prot_mode = "neutral" if args.neutral else "pkanet"

    if args.redock_smiles:
        pts    = args.redock_smiles.strip().split(None, 1)
        rd_smi = pts[0]
        rd_nm  = pts[1].replace(" ", "_") if len(pts) > 1 else "redock"
        rd_out = out / "redock"
        rd_out.mkdir(exist_ok=True)
        print(f"⏳  Docking reference {rd_nm} …")
        rd_prep = core.prepare_ligand(
            rd_smi, "redock_" + rd_nm, args.ph, rd_out,
            mode=prot_mode, **_pkanet_kwargs(args),
        )
        if rd_prep["success"]:
            rd_dock = core.run_vina(
                receptor_pdbqt = rec_pdbqt,
                ligand_pdbqt   = os.path.abspath(rd_prep["pdbqt"]),
                config_txt     = config_txt,
                vina_path      = vina_path,
                exhaustiveness = args.exhaustiveness,
                n_modes        = args.poses,
                energy_range   = int(args.energy_range),
                wdir           = rd_out,
                out_name       = "redock_" + rd_nm,
                seed           = args.seed,
            )
            if rd_dock["success"] and rd_dock["top_score"] is not None:
                ref_score = rd_dock["top_score"]
                _ok(f"Reference score: {ref_score:.2f} kcal/mol")

    # ── Main loop ─────────────────────────────────────────────────────────────
    results = []
    n = len(smiles_pairs)

    for i, (smi, nm) in enumerate(smiles_pairs, 1):
        print(f"  [{i}/{n}] {nm}")
        lig_wdir = out / nm
        lig_wdir.mkdir(exist_ok=True)
        try:
            prep = core.prepare_ligand(
                smiles = smi, name = nm, ph = args.ph,
                wdir   = lig_wdir, mode = prot_mode,
                **_pkanet_kwargs(args),
            )
            if not prep["success"]:
                results.append({"Name": nm, "SMILES": smi, "Top Score": None,
                                 "Charge": None, "RMSD vs crystal": "",
                                 "Status": f"PREP_FAILED: {prep['error']}"})
                _warn(f"  Prep failed: {prep['error']}")
                continue

            dock = core.run_vina(
                receptor_pdbqt = rec_pdbqt,
                ligand_pdbqt   = os.path.abspath(prep["pdbqt"]),
                config_txt     = config_txt,
                vina_path      = vina_path,
                exhaustiveness = args.exhaustiveness,
                n_modes        = args.poses,
                energy_range   = int(args.energy_range),
                wdir           = lig_wdir,
                out_name       = nm,
                seed           = args.seed,
            )
            if not dock["success"] or dock["top_score"] is None:
                results.append({"Name": nm, "SMILES": smi, "Top Score": None,
                                 "Charge": prep.get("charge"), "RMSD vs crystal": "",
                                 "Status": "DOCK_FAILED"})
                _warn("  Dock failed")
                continue

            pv_sdf = str(lig_wdir / f"{nm}_pv_ready.sdf")
            core.fix_sdf_bond_orders(dock["out_sdf"],
                                      prep.get("prot_smiles", smi), pv_sdf)
            if not os.path.exists(pv_sdf) or os.path.getsize(pv_sdf) < 10:
                pv_sdf = dock["out_sdf"]

            # RMSD vs crystal for best pose
            rmsd_best = ""
            if crystal_pdb and os.path.exists(crystal_pdb):
                mols = core.load_mols_from_sdf(pv_sdf, sanitize=False)
                if mols:
                    try:
                        r = core.calc_rmsd_heavy(mols[0], crystal_pdb)
                        if r is not None:
                            rmsd_best = round(r, 2)
                    except Exception:
                        pass

            top = dock["top_score"]
            results.append({
                "Name":            nm,
                "SMILES":          smi,
                "Prepared SMILES": prep.get("prot_smiles", ""),
                "Top Score":       top,
                "Charge":          prep.get("charge"),
                "RMSD vs crystal": rmsd_best,
                "Status":          "OK",
                "out_sdf":         dock["out_sdf"],
                "pv_sdf":          pv_sdf,
                "out_pdbqt":       dock["out_pdbqt"],
            })
            label = "✓ Strong" if top < -9 else "✓ Moderate" if top < -7 else "  Weak"
            rmsd_txt = f"  RMSD {rmsd_best} Å" if rmsd_best != "" else ""
            _ok(f"  {label}  {top:.2f} kcal/mol{rmsd_txt}")

        except Exception as exc:
            results.append({"Name": nm, "SMILES": smi, "Top Score": None,
                             "Charge": None, "RMSD vs crystal": "",
                             "Status": f"ERROR: {exc}"})
            _warn(f"  Exception: {exc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    _step("Batch Summary")
    ok_results = sorted([r for r in results if r["Status"] == "OK"],
                        key=lambda r: r["Top Score"])

    if _RICH:
        t = Table(
            title = f"Batch Results  (ref = {ref_score:.2f} kcal/mol)" if ref_score
                    else "Batch Results",
            show_header=True, header_style="bold cyan",
        )
        t.add_column("Rank")
        t.add_column("Name")
        t.add_column("Top Score",       justify="right")
        t.add_column("Δ vs ref",        justify="right")
        t.add_column("RMSD vs crystal", justify="right")
        t.add_column("Charge",          justify="right")
        for rank, r in enumerate(ok_results, 1):
            s     = r["Top Score"]
            aclr  = "green" if s < -9 else "yellow" if s < -7 else "red"
            delta = f"{s - ref_score:+.2f}" if ref_score is not None else "—"
            rmsd  = str(r["RMSD vs crystal"]) if r["RMSD vs crystal"] != "" else "—"
            t.add_row(str(rank), r["Name"], f"[{aclr}]{s:.2f}[/{aclr}]",
                      delta, rmsd, str(r.get("Charge", "?")))
        _con.print(t)
    else:
        print(f"\n{'Rank':>5}  {'Name':<28}  {'Score':>8}  {'Δ ref':>7}  "
              f"{'RMSD':>6}  {'Charge':>6}")
        print("─" * 72)
        for rank, r in enumerate(ok_results, 1):
            s     = r["Top Score"]
            delta = f"{s - ref_score:+.2f}" if ref_score is not None else "—"
            rmsd  = str(r["RMSD vs crystal"]) if r["RMSD vs crystal"] != "" else "—"
            print(f"{rank:>5}  {r['Name']:<28}  {s:>8.2f}  {delta:>7}  "
                  f"{rmsd:>6}  {str(r.get('Charge','?')):>6}")

    csv_path = str(out / "batch_scores.csv")
    with open(csv_path, "w", newline="") as fh:
        fieldnames = ["Name", "SMILES", "Prepared SMILES", "Top Score",
                      "Charge", "RMSD vs crystal", "Status", "out_sdf", "pv_sdf"]
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    n_ok = sum(1 for r in results if r["Status"] == "OK")
    _ok(f"Scores CSV  : {csv_path}")
    _ok(f"Completed   : {n_ok}/{n} ligands docked successfully")
    if ref_score is not None and ok_results:
        _ok(f"Better than reference: "
            f"{sum(1 for r in ok_results if r['Top Score'] < ref_score)} ligand(s)")


# ── sub-command: ensemble ────────────────────────────────────────────────────

def _resolve_ensemble_receptors(entries: list[str]) -> list[str]:
    """Expand files, directories, glob patterns, and explicit PDB IDs."""
    receptors = []
    for entry in entries:
        expanded = os.path.expanduser(entry)
        matches = sorted(glob.glob(expanded)) if glob.has_magic(expanded) else []
        if matches:
            candidates = matches
        elif os.path.isdir(expanded):
            candidates = sorted(
                str(path) for path in Path(expanded).iterdir()
                if path.is_file() and path.suffix.lower() in {".pdb", ".cif", ".mmcif"}
            )
        elif os.path.isfile(expanded):
            candidates = [expanded]
        elif len(entry) == 4 and entry.isalnum():
            candidates = [entry.upper()]
        else:
            raise ValueError(f"Receptor input did not match a file, directory, glob, or PDB ID: {entry}")
        receptors.extend(candidates)

    unique = []
    seen = set()
    for receptor in receptors:
        key = os.path.realpath(receptor) if os.path.exists(receptor) else receptor.upper()
        if key not in seen:
            seen.add(key)
            unique.append(receptor)
    return unique


def _ensemble_receptor_name(receptor: str) -> str:
    label = Path(receptor).stem if os.path.exists(receptor) else receptor.upper()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._") or "receptor"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cmd_ensemble(args):
    _banner()
    _step("Receptor-ensemble docking")
    if not args.ligands and not args.smiles_list:
        _err("Provide --ligands <file.smi>  or  --smiles-list SMILES [SMILES ...]")
    if args.ligands and not os.path.isfile(args.ligands):
        _err(f"Ligand list not found: {args.ligands}")
    try:
        receptors = _resolve_ensemble_receptors(args.receptors)
    except ValueError as exc:
        _err(str(exc))
    if not receptors:
        _err("No PDB/mmCIF receptor structures were found")

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    _ok(f"Receptor conformations: {len(receptors)}")
    _ok("Each receptor will be prepared and validated independently")
    if args.center == "manual":
        _warn("The same manual box is applied to every receptor; use only with aligned structures")

    combined_rows = []
    receptor_rows = []
    for index, receptor in enumerate(receptors, 1):
        label = _ensemble_receptor_name(receptor)
        run_dir = out / f"{index:03d}_{label}"
        _step(f"Receptor {index}/{len(receptors)}: {label}")
        batch_args = argparse.Namespace(**vars(args))
        batch_args.command = "batch"
        batch_args.receptor = receptor
        batch_args.receptor_json = None
        batch_args.output = str(run_dir)

        status = "OK"
        error = ""
        try:
            cmd_batch(batch_args)
        except SystemExit as exc:
            if exc.code not in (None, 0):
                status = "FAILED"
                error = getattr(exc, "message", f"batch exited with status {exc.code}")
        except Exception as exc:
            status = "FAILED"
            error = str(exc)

        score_path = run_dir / "batch_scores.csv"
        if status == "OK" and not score_path.exists():
            status = "FAILED"
            error = "batch_scores.csv was not produced"
        if status == "OK":
            with score_path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    combined_rows.append({
                        "Receptor": label,
                        "Receptor Source": receptor,
                        "Receptor Directory": str(run_dir),
                        **row,
                    })
            _ok(f"Ensemble member completed: {label}")
        else:
            _warn(f"Ensemble member failed and was skipped: {label} ({error})")
        receptor_rows.append({
            "Receptor": label,
            "Receptor Source": receptor,
            "Output Directory": str(run_dir),
            "Status": status,
            "Error": error,
        })

    receptor_csv = out / "ensemble_receptors.csv"
    _write_csv(
        receptor_csv, receptor_rows,
        ["Receptor", "Receptor Source", "Output Directory", "Status", "Error"],
    )

    combined_csv = out / "ensemble_scores.csv"
    combined_fields = [
        "Receptor", "Receptor Source", "Receptor Directory", "Name", "SMILES",
        "Prepared SMILES", "Top Score", "Charge", "RMSD vs crystal", "Status",
        "out_sdf", "pv_sdf",
    ]
    _write_csv(combined_csv, combined_rows, combined_fields)

    by_ligand = {}
    for row in combined_rows:
        key = (row.get("Name", ""), row.get("SMILES", ""))
        by_ligand.setdefault(key, []).append(row)
    summary_rows = []
    for (name, smiles), rows in sorted(by_ligand.items()):
        scored = []
        for row in rows:
            try:
                if row.get("Status") == "OK":
                    scored.append((float(row["Top Score"]), row["Receptor"]))
            except (TypeError, ValueError):
                continue
        scores = [score for score, _ in scored]
        best_score, best_receptor = min(scored) if scored else (None, "")
        summary_rows.append({
            "Name": name,
            "SMILES": smiles,
            "Successful Receptors": len(scores),
            "Total Receptors": len(receptors),
            "Best Score": best_score,
            "Best Receptor": best_receptor,
            "Mean Score": statistics.fmean(scores) if scores else None,
            "Median Score": statistics.median(scores) if scores else None,
            "Population SD": statistics.pstdev(scores) if scores else None,
        })
    summary_csv = out / "ensemble_summary.csv"
    _write_csv(
        summary_csv, summary_rows,
        ["Name", "SMILES", "Successful Receptors", "Total Receptors", "Best Score",
         "Best Receptor", "Mean Score", "Median Score", "Population SD"],
    )

    _step("Ensemble Summary")
    succeeded = sum(row["Status"] == "OK" for row in receptor_rows)
    _ok(f"Receptors completed : {succeeded}/{len(receptors)}")
    _ok(f"Receptor status CSV : {receptor_csv}")
    _ok(f"All score rows      : {combined_csv}")
    _ok(f"Ligand summaries    : {summary_csv}")
    if succeeded == 0:
        _err("All receptor ensemble members failed; no docking results were produced")


# ── 2D diagram helper ─────────────────────────────────────────────────────────

def _generate_diagram(core, rec_fh, pose_sdf, smiles, lig_name,
                      best_score, pose_idx, out, cutoff=4.5):
    try:
        title = f"Pose {pose_idx+1}  ·  {lig_name}"
        if best_score is not None:
            title += f"  ·  {best_score:.2f} kcal/mol"
        print("⏳  Generating 2D interaction diagram …")
        svg_bytes = core.draw_interaction_diagram(
            receptor_pdb = rec_fh,
            pose_sdf     = pose_sdf,
            smiles       = smiles,
            title        = title,
            cutoff       = cutoff,
            size         = (800, 759),
            max_residues = 14,
        )
        diag_path = str(out / f"{lig_name}_diagram.svg")
        with open(diag_path, "wb") as fh:
            fh.write(svg_bytes)
        _ok(f"2D diagram SVG: {diag_path}")

        try:
            import cairosvg
            png_bytes = cairosvg.svg2png(bytestring=svg_bytes, scale=2,
                                          background_color="white")
            png_path = str(out / f"{lig_name}_diagram.png")
            with open(png_path, "wb") as fh:
                fh.write(png_bytes)
            _ok(f"2D diagram PNG: {png_path}")
        except ImportError:
            _warn("cairosvg not installed — PNG skipped (pip install cairosvg)")
    except Exception as exc:
        _warn(f"Diagram generation failed: {exc}")


def cmd_diagram(args):
    core = _import_core()
    _step("2D interaction diagram")

    if not os.path.exists(args.receptor):
        _err(f"Receptor PDB not found: {args.receptor}")
    if not os.path.exists(args.pose_sdf):
        _err(f"Pose SDF not found: {args.pose_sdf}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    mols = core.load_mols_from_sdf(args.pose_sdf, sanitize=False)
    if not mols:
        _err("No molecules found in SDF file")
    idx = max(0, min(args.pose - 1, len(mols) - 1))

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tf:
        tmp_sdf = tf.name
    core.write_single_pose(mols[idx], tmp_sdf)

    _generate_diagram(
        core       = core,
        rec_fh     = args.receptor,
        pose_sdf   = tmp_sdf,
        smiles     = args.smiles or "",
        lig_name   = args.name or "ligand",
        best_score = args.score,
        pose_idx   = idx,
        out        = out,
        cutoff     = args.cutoff,
    )
    os.unlink(tmp_sdf)


# ── sub-command: redock ──────────────────────────────────────────────────────

def cmd_redock(args):
    """
    Self-docking validation: automatically re-docks the co-crystal ligand that
    is already bound in the input structure, with no ligand input required.

    Pipeline
    --------
    1. Load structure  (download from RCSB or read from file)
    2. Ligand safety check
          • No drug-like ligand found  → clear error, suggest acd dock
          • Exactly one ligand found   → auto-select and proceed
          • Multiple ligands found     → show numbered list, prompt user to choose
    3. Receptor preparation  (strip chosen ligand, build PDBQT + box)
    4. Acquire SMILES    (RCSB CCD API → CIF _chem_comp.pdbx_smiles → 3D fallback)
    5. Prepare ligand    (protonation, 3-D conformer, PDBQT)
    6. Dock with Vina
    7. Report RMSD vs crystal pose  (PASS ≤ 2.0 Å · BORDERLINE ≤ 3.0 Å · FAIL > 3.0 Å)
    """
    from .redock_workflow import prepare_reference_ligand, run_reference_vina, calculate_pose_rmsds
    core = _import_core()
    _banner()
    _step("Redocking — self-docking validation")

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load structure ────────────────────────────────────────────
    _step("Step 1 — Structure loading")

    raw = ""

    if args.receptor_json:
        # ── Fast path: reuse a previously prepared receptor ───────────────
        with open(args.receptor_json) as fh:
            rec_summary = json.load(fh)
        _ok(f"Using saved receptor : {rec_summary['rec_fh']}")

        # Safety check for the saved-receptor path too
        ligand_pdb_saved   = rec_summary.get("ligand_pdb_path") or ""
        cocrystal_id_saved = rec_summary.get("cocrystal_ligand_id") or ""

        if not ligand_pdb_saved or not os.path.exists(ligand_pdb_saved):
            # The saved receptor has no ligand recorded — inform clearly
            _err(
                "The receptor JSON does not contain a co-crystal ligand.\n\n"
                "  This means the structure was prepared without a bound ligand,\n"
                "  so redocking cannot be performed from this saved receptor.\n\n"
                "  Solutions:\n"
                "    • Re-run without --receptor-json so the structure is\n"
                "      re-scanned and the ligand is extracted fresh.\n"
                "    • Check that the original PDB/CIF actually contained a\n"
                "      co-crystal ligand before preparing the receptor.\n"
                "    • To dock a new ligand instead, use:  acd dock"
            )

        # If --resname is given, update the stored cocrystal_id
        if args.resname:
            parts = cocrystal_id_saved.split("_")
            if len(parts) >= 3:
                rec_summary["cocrystal_ligand_id"] = "_".join(
                    [args.resname.upper()] + parts[1:]
                )
            else:
                rec_summary["cocrystal_ligand_id"] = args.resname.upper()
            _ok(f"Residue name overridden to : {args.resname.upper()}")

        resname = rec_summary["cocrystal_ligand_id"].split("_")[0].upper()
        _ok(f"Co-crystal ligand : {rec_summary['cocrystal_ligand_id']}")
        _ok(f"Extracted PDB     : {ligand_pdb_saved}")

        # Jump straight to SMILES acquisition (skip Steps 2–3 below)
        _do_full_prep = False

    else:
        # ── Full path: download / load raw structure ──────────────────────
        rec_out = out / "receptor"
        rec_out.mkdir(exist_ok=True)

        if os.path.exists(args.receptor):
            raw = os.path.abspath(args.receptor)
            _ok(f"Using local file : {raw}")
        else:
            token   = args.receptor.strip().upper()
            fmt_ext = "cif" if args.fmt.upper() == "CIF" else "pdb"
            raw     = str(rec_out / f"raw.{fmt_ext}")
            _ok(f"Downloading {token} ({fmt_ext.upper()}) from RCSB …")
            rc, _ = core.run_cmd([
                "curl", "-sf",
                f"https://files.rcsb.org/download/{token}.{fmt_ext}",
                "-o", raw,
            ])
            if rc != 0 or not os.path.exists(raw) or os.path.getsize(raw) < 200:
                raw_cif = str(rec_out / "raw.cif")
                rc2, _ = core.run_cmd([
                    "curl", "-sf",
                    f"https://files.rcsb.org/download/{token}.cif",
                    "-o", raw_cif,
                ])
                if rc2 == 0 and os.path.exists(raw_cif) and os.path.getsize(raw_cif) > 200:
                    raw = raw_cif
                    _warn("PDB format unavailable — fell back to CIF")
                else:
                    _err(f"Download failed for {token}")

        _do_full_prep = True

    # ── Step 2: Co-crystal ligand safety check ────────────────────────────
    if _do_full_prep:
        _step("Step 2 — Co-crystal ligand check")

        # Use scan_hetatm_residues (not scan_ligands) so that every "key"
        # value is built by the same _hetatm_key() function that
        # prepare_receptor uses internally — no key-format mismatch possible.
        print("⏳  Scanning HETATM records …")
        try:
            het_rows = core.scan_hetatm_residues(raw)
        except Exception as _se:
            het_rows = []
            _warn(f"HETATM scan error: {_se}")

        # Drug-like ligands only — filter for display and user selection
        lig_rows = [r for r in het_rows if r.get("type_guess") == "ligand"]

        # ── Case A: no ligand found ───────────────────────────────────────
        if not lig_rows:
            _err(
                "No co-crystal ligand was found in this structure.\n\n"
                "  Redocking requires a small-molecule ligand (HETATM) that is\n"
                "  physically bound to the protein in the crystal structure.\n"
                "  This structure appears to contain only protein atoms, water,\n"
                "  ions, or buffer molecules — none qualify as a dockable\n"
                "  co-crystal reference ligand.\n\n"
                "  Common reasons:\n"
                "    • The PDB entry is an apo (unliganded) structure.\n"
                "    • The ligand was removed during structure refinement.\n"
                "    • Only buffer/solvent molecules (GOL, EDO, SO4 …) are present.\n\n"
                "  Suggestions:\n"
                "    • Search RCSB for a holo structure of the same target.\n"
                "    • To dock a new ligand into this apo structure, use:\n"
                "        acd dock --receptor <PDB_ID> --compound <name>"
            )

        # ── Apply --resname filter if the user specified one ──────────────
        if args.resname:
            filtered = [
                r for r in lig_rows
                if r["resname"].upper() == args.resname.strip().upper()
            ]
            if not filtered:
                _err(
                    f"--resname '{args.resname.upper()}' was not found among "
                    f"the detected drug-like ligands.\n\n"
                    "  Ligands detected:\n"
                    + "\n".join(
                        f"    [{i}]  {r['resname']:>4}  "
                        f"chain {r['chain'] or '-':>2}  "
                        f"resid {r['resid']:>5}  "
                        f"({r['n_atoms']:>3} heavy atoms)"
                        for i, r in enumerate(lig_rows)
                    )
                    + "\n\n  Run again without --resname to select interactively."
                )
            lig_rows = filtered

        # ── Case B: one ligand → auto-select ─────────────────────────────
        if len(lig_rows) == 1:
            chosen_row = lig_rows[0]
            _ok(
                f"Co-crystal ligand : {chosen_row['resname']}  "
                f"chain {chosen_row['chain'] or '-'}  "
                f"resid {chosen_row['resid']}  "
                f"({chosen_row['n_atoms']} heavy atoms)"
            )

        # ── Case C: multiple ligands → prompt user ────────────────────────
        else:
            print(
                f"\n  Found {len(lig_rows)} co-crystal ligand candidate(s) "
                f"in this structure:\n"
            )
            for i, r in enumerate(lig_rows):
                print(
                    f"    [{i}]  {r['resname']:>4}  "
                    f"chain {r['chain'] or '-':>2}  "
                    f"resid {r['resid']:>5}  "
                    f"({r['n_atoms']:>3} heavy atoms)"
                )
            print()

            chosen_row = None
            while chosen_row is None:
                try:
                    raw_sel = input(
                        f"  Select ligand to redock [0–{len(lig_rows) - 1}]: "
                    ).strip()
                    idx = int(raw_sel)
                    if 0 <= idx < len(lig_rows):
                        chosen_row = lig_rows[idx]
                    else:
                        print(
                            f"  ✗  Please enter a number between 0 and "
                            f"{len(lig_rows) - 1}."
                        )
                except ValueError:
                    print("  ✗  Invalid input — please type a number.")
                except KeyboardInterrupt:
                    print()
                    _err("Selection cancelled by user.")

            _ok(
                f"Selected : {chosen_row['resname']}  "
                f"chain {chosen_row['chain'] or '-'}  "
                f"resid {chosen_row['resid']}  "
                f"({chosen_row['n_atoms']} heavy atoms)"
            )

        # ── Build hetatm_policy from ALL het_rows ─────────────────────────
        # Keys come directly from scan_hetatm_residues, which uses the same
        # _hetatm_key() function as prepare_receptor.  No manual key
        # construction — guaranteed match every time.
        _chosen_key   = chosen_row["key"]
        hetatm_policy = {}
        for _hr in het_rows:
            _k  = _hr["key"]
            _tg = _hr.get("type_guess", "other")
            if _k == _chosen_key:
                hetatm_policy[_k] = "reference"     # chosen: extract + box centre
            elif _tg == "ligand":
                hetatm_policy[_k] = "remove"        # other drug-like: strip
            elif _tg == "water":
                hetatm_policy[_k] = "remove"        # always strip waters
            elif _tg in ("metal", "heme/cofactor", "cofactor"):
                hetatm_policy[_k] = "keep"          # keep metals / heme / cofactors
            else:
                hetatm_policy[_k] = "remove"        # buffers, ions: strip

        # ── Step 3: Receptor preparation ─────────────────────────────────
        _step("Step 3 — Receptor preparation")

        center_mode, manual_xyz, box_size = _receptor_box_args(core, raw, args)

        print(
            f"⏳  Preparing receptor  "
            f"(reference: {chosen_row['resname']} "
            f"chain {chosen_row['chain'] or '-'} "
            f"resid {chosen_row['resid']}) …"
        )
        rec_result = core.prepare_receptor(
            raw_pdb              = raw,
            wdir                 = rec_out,
            center_mode          = center_mode,
            manual_xyz           = manual_xyz,
            prody_sel            = args.sel or "",
            box_size             = box_size,
            hetatm_policy        = hetatm_policy,
            reference_hetatm_key = _chosen_key,
        )
        if not rec_result["success"]:
            for line in rec_result.get("log", []):
                _warn(line) if line.startswith("⚠") else print(line)
            _err(f"Receptor preparation failed: {rec_result['error']}")
        _print_heme_report(rec_result)
        pocket_report = _assess_receptor_pocket(raw, rec_result, rec_out)

        rec_summary = {
            "pocket_report": pocket_report,
            "rec_fh":              rec_result["rec_fh"],
            "rec_pdbqt":           rec_result["rec_pdbqt"],
            "config_txt":          rec_result["config_txt"],
            "cx": rec_result["cx"], "cy": rec_result["cy"], "cz": rec_result["cz"],
            "sx": rec_result["sx"], "sy": rec_result["sy"], "sz": rec_result["sz"],
            "ligand_pdb_path":     rec_result.get("ligand_pdb_path"),
            "cocrystal_ligand_id": rec_result.get("cocrystal_ligand_id", ""),
        }
        with open(str(rec_out / "receptor_summary.json"), "w") as fh:
            json.dump(rec_summary, fh, indent=2)

        cx, cy, cz = rec_result["cx"], rec_result["cy"], rec_result["cz"]
        _ok(f"Receptor ready — center ({cx:.2f}, {cy:.2f}, {cz:.2f})")

        ligand_pdb_saved   = rec_result.get("ligand_pdb_path") or ""
        cocrystal_id_saved = rec_result.get("cocrystal_ligand_id") or ""

        if not ligand_pdb_saved or not os.path.exists(ligand_pdb_saved):
            _err(
                f"Receptor preparation succeeded but ligand "
                f"'{chosen_row['resname']}' "
                f"(chain {chosen_row['chain'] or '-'}, "
                f"resid {chosen_row['resid']}) was not extracted.\n\n"
                f"  The ligand was detected in the pre-scan but could not be\n"
                f"  isolated — debug info in: {rec_out}\n\n"
                f"  Try running again, or report this as a bug."
            )

        _ok(f"Co-crystal ligand extracted : {cocrystal_id_saved}")
        _ok(f"Ligand PDB                  : {ligand_pdb_saved}")

        if args.resname:
            parts = cocrystal_id_saved.split("_")
            rec_summary["cocrystal_ligand_id"] = "_".join(
                [args.resname.upper()] + (parts[1:] if len(parts) >= 3 else [])
            ) or args.resname.upper()
            _ok(f"Residue name overridden to : {args.resname.upper()}")

    _check_receptor_pocket(rec_summary)

    # ── Resolve final ligand identity (both paths converge here) ─────────
    ligand_pdb   = rec_summary.get("ligand_pdb_path") or ""
    cocrystal_id = rec_summary.get("cocrystal_ligand_id") or ""
    resname      = cocrystal_id.split("_")[0].upper() if cocrystal_id else ""

    # ── Step 4: SMILES acquisition ────────────────────────────────────────
    _step("Step 4 — SMILES acquisition")

    smiles, source, smi_warn = core.get_cocrystal_smiles(
        ligand_pdb_path     = ligand_pdb,
        cocrystal_ligand_id = cocrystal_id,
        raw_pdb             = raw,
    )

    if not smiles:
        _err(
            f"Could not obtain SMILES for '{resname}'.\n"
            f"  Tried: RCSB CCD API  →  CIF _chem_comp.pdbx_smiles  →  "
            f"RDKit/OpenBabel 3D conversion.\n"
            f"  Provide it manually:  acd dock --receptor ... "
            f"--smiles '<SMILES>' --name {resname}"
        )

    _SOURCE_LABELS = {
        "rcsb_ccd":      "RCSB Chemical Component Dictionary (ideal / curated)",
        "cif_block":     "CIF file  _chem_comp.pdbx_smiles",
        "3d_conversion": "3D coordinate conversion — RDKit / OpenBabel  ⚠ verify",
    }
    _ok(f"SMILES source : {_SOURCE_LABELS.get(source, source)}")

    _ok(f"SMILES        : {smiles}")
    if smi_warn:
        _warn(smi_warn)

    # ── Step 5: Vina binary ───────────────────────────────────────────────
    print("⏳  Checking Vina binary …")
    vina_path, vina_err = core.get_vina_binary()
    if vina_path is None:
        _err(f"Could not get Vina binary: {vina_err}")
    _ok(f"Vina : {vina_path}")

    # ── Step 6: Ligand preparation ────────────────────────────────────────
    _step("Step 5 — Ligand preparation")

    name      = args.name or resname or "REDOCK"
    ph        = args.ph
    prot_mode = "neutral" if args.neutral else "pkanet"
    lig_out   = out / "ligand"
    lig_out.mkdir(exist_ok=True)

    print(f"⏳  Preparing {name} at pH {ph} …")
    lig_result = prepare_reference_ligand(core,
        smiles        = smiles,
        name          = name,
        ph            = ph,
        wdir          = lig_out,
        mode          = prot_mode,
        **_pkanet_kwargs(args),
    )

    if not lig_result["success"]:
        _err(f"Ligand preparation failed: {lig_result['error']}")

    prot_smiles = lig_result.get("prot_smiles", "")
    charge      = int(lig_result.get("net_charge", lig_result.get("charge", 0)))
    _ok(f"Prepared SMILES : {prot_smiles}")
    _ok(f"Formal charge   : {charge:+d}")
    _ok(f"Conformer seed : {_conformer_seed_text(lig_result)}")
    if lig_result.get("pkanet_ranked_csv"):
        _ok(f"pKaNET CSV      : {lig_result['pkanet_ranked_csv']}")

    # ── Step 7: AutoDock Vina ─────────────────────────────────────────────
    _step("Step 6 — AutoDock Vina")

    rec_pdbqt  = os.path.abspath(rec_summary["rec_pdbqt"])
    config_txt = os.path.abspath(rec_summary["config_txt"])

    print(f"⏳  Running Vina (exhaustiveness={args.exhaustiveness}) …")
    dock = run_reference_vina(core,
        receptor_pdbqt = rec_pdbqt,
        ligand_pdbqt   = os.path.abspath(lig_result["pdbqt"]),
        config_txt     = config_txt,
        vina_path      = vina_path,
        exhaustiveness = args.exhaustiveness,
        n_modes        = args.poses,
        energy_range   = int(args.energy_range),
        wdir           = out,
        out_name       = name,
        seed           = args.seed,
    )

    if not dock["success"]:
        _err(f"Vina failed: {dock['error']}\n{dock['log'][:400]}")

    # ── Bond-order fix ────────────────────────────────────────────────────
    pv_sdf = str(out / f"{name}_pv_ready.sdf")
    if prot_smiles:
        core.fix_sdf_bond_orders(dock["out_sdf"], prot_smiles, pv_sdf)
    if not os.path.exists(pv_sdf) or os.path.getsize(pv_sdf) < 10:
        pv_sdf = dock["out_sdf"]

    # ── Load pose molecules ───────────────────────────────────────────────
    sdf_src   = (pv_sdf if (os.path.exists(pv_sdf) and
                             os.path.getsize(pv_sdf) > 10)
                 else dock["out_sdf"])
    pose_mols = core.load_mols_from_sdf(sdf_src, sanitize=False)

    # ── RMSD vs crystal pose ──────────────────────────────────────────────
    rmsds = calculate_pose_rmsds(core, dock["scores"], pose_mols, ligand_pdb)

    # ── Score table ───────────────────────────────────────────────────────
    _step("Results")
    _print_scores(
        dock["scores"],
        ref_score    = None,
        pose_mols    = pose_mols,
        crystal_pdb  = ligand_pdb if (ligand_pdb and os.path.exists(ligand_pdb)) else None,
        core         = core,
        sort_by_rmsd = True,
    )

    # ── Redocking verdict ─────────────────────────────────────────────────
    if rmsds:
        _valid = {p: v for p, v in rmsds.items() if v is not None}
        if _valid:
            best_pose = min(_valid, key=_valid.get)   # pose with lowest RMSD
            best_rmsd = _valid[best_pose]
        else:
            best_pose, best_rmsd = None, None
        if best_rmsd is not None:
            if best_rmsd <= 2.0:
                clr     = "green"
                verdict = "✓ PASS"
                detail  = f"{best_rmsd:.2f} Å ≤ 2.0 Å (pose {best_pose}) — docking protocol validated"
            elif best_rmsd <= 3.0:
                clr     = "yellow"
                verdict = "⚠ BORDERLINE"
                detail  = f"{best_rmsd:.2f} Å (pose {best_pose}; 2–3 Å, acceptable for flexible loops)"
            else:
                clr     = "red"
                verdict = "✗ FAIL"
                detail  = f"{best_rmsd:.2f} Å > 3.0 Å (pose {best_pose}) — pose differs from crystal"

            if _RICH:
                _con.print(
                    f"\n  Redocking verdict : "
                    f"[bold {clr}]{verdict}[/bold {clr}]  —  {detail}\n"
                )
            else:
                print(f"\n  Redocking verdict : {verdict}  —  {detail}\n")

    # ── CSV output ────────────────────────────────────────────────────────
    scores_csv = str(out / f"{name}_redock_scores.csv")
    _write_scores_csv(dock["scores"], scores_csv, lig_name=name, rmsds=rmsds,
                      sort_by_rmsd=True)

    # ── File summary ──────────────────────────────────────────────────────
    _ok(f"All poses PDBQT : {dock['out_pdbqt']}")
    _ok(f"All poses SDF   : {dock['out_sdf']}")
    if os.path.exists(pv_sdf) and os.path.getsize(pv_sdf) > 10:
        _ok(f"PV-ready SDF    : {pv_sdf}")
    _ok(f"Crystal ligand  : {ligand_pdb}")
    _ok(f"Scores CSV      : {scores_csv}")
    _ok(f"SMILES source   : {source}")

    # ── Optional per-pose SDF ─────────────────────────────────────────────
    if args.save_poses:
        poses_dir = out / "poses"
        poses_dir.mkdir(exist_ok=True)
        for i, mol in enumerate(pose_mols, 1):
            core.write_single_pose(    mol, str(poses_dir / f"{name}_pose{i}.sdf"))
            core.write_single_pose_pdb(mol, str(poses_dir / f"{name}_pose{i}.pdb"))
        _ok(f"Individual poses : {poses_dir}/")

    # ── Optional 2D diagram ───────────────────────────────────────────────
    if args.diagram:
        _generate_diagram(
            core       = core,
            rec_fh     = rec_summary["rec_fh"],
            pose_sdf   = pv_sdf if os.path.exists(pv_sdf) else dock["out_sdf"],
            smiles     = prot_smiles,
            lig_name   = name,
            best_score = dock["top_score"],
            pose_idx   = 0,
            out        = out,
            cutoff     = args.diagram_cutoff,
        )


# ── argument parser ───────────────────────────────────────────────────────────

def _add_receptor_box_args(p):
    p.add_argument("--fmt",    default="PDB", choices=["PDB", "CIF"],
                   help="Download format when using a PDB ID (default: PDB; use CIF for large/newer entries)")
    p.add_argument("--center", default="auto",
                   choices=["auto", "manual", "selection"],
                   help="Box centre mode: auto=co-crystal ligand centroid, "
                        "manual=--cx/cy/cz, selection=ProDy atom selection (default: auto)")
    p.add_argument("--cx", type=float, help="Box centre X coordinate (--center manual)")
    p.add_argument("--cy", type=float, help="Box centre Y coordinate (--center manual)")
    p.add_argument("--cz", type=float, help="Box centre Z coordinate (--center manual)")
    p.add_argument("--sel", metavar="PRODY_SEL",
                   help="ProDy atom selection string, e.g. 'resid 702 and chain A' (--center selection)")
    p.add_argument("--resname", metavar="3-LETTER-CODE",
                   help="For --center auto, use this ligand residue name as the grid center/reference when multiple ligands exist (e.g. ATP, DC3)")
    p.add_argument("--yes", action="store_true",
                   help="Non-interactive mode: when multiple co-crystal ligands are detected, accept the default [0] ligand without prompting")
    p.add_argument("--blind", action="store_true",
                   help="Cover the whole protein using 4 Å padding; overrides center and box sizes")
    p.add_argument("--bx", type=_box_size, default=18,
                   help="Search box X size in Å (positive integer, default: 18)")
    p.add_argument("--by", type=_box_size, default=18,
                   help="Search box Y size in Å (positive integer, default: 18)")
    p.add_argument("--bz", type=_box_size, default=18,
                   help="Search box Z size in Å (positive integer, default: 18)")


def _add_receptor_args(p):
    p.add_argument("--receptor",      metavar="PDB_ID_or_FILE",
                   help="4-letter PDB ID (auto-downloaded from RCSB) or path to .pdb/.cif file")
    p.add_argument("--receptor-json", metavar="PATH",
                   help="JSON from a previous  acd receptor  run — skips re-preparation entirely")
    _add_receptor_box_args(p)


def _add_ligand_args(p):
    p.add_argument("--smiles",        metavar="SMILES",
                   help="Ligand SMILES string, e.g. \"c1ccc(cc1)O\"")
    p.add_argument("--compound",      metavar="NAME",
                   help="Compound name for PubChem lookup, e.g. erlotinib, baicalein")
    p.add_argument("--ligand-file",   metavar="PATH",
                   help="Ligand file (.sdf / .mol2 / .pdb / .pdbqt); PDBQT is passed unchanged to Vina; PDB uses Meeko with atom mapping")
    p.add_argument("--name",          default="LIG",
                   help="Output name used for all result files (default: LIG)")
    p.add_argument("--ph",            type=_target_ph, default=7.4,
                   help="Target pH for protonation state, 0–14 (default: 7.4)")
    p.add_argument("--pkanet",        action="store_true", default=True,
                   help="Use pKaNET tautomer-aware protonation (default; requires pKaNET.py)")
    p.add_argument("--neutral",       action="store_true",
                   help="Neutral mode: keep input charge state, only add missing hydrogens")
    p.add_argument("--no-pubchem",    action="store_true",
                   help="Skip PubChem experimental pKa lookup (faster, slightly less accurate)")
    p.add_argument("--max-tautomers", type=int, choices=range(1, 21), default=8, metavar="N",
                   help="Max tautomers enumerated by pKaNET (default: 8)")
    p.add_argument("--ph-window", type=float,
                   choices=tuple(round(i / 10, 1) for i in range(2, 21)), default=1.0,
                   help="pH window for protonation enumeration: [pH−window/2, pH+window/2] (default: 1.0)")
    p.add_argument("--pkanet-selection", choices=["auto_recommended", "highest_score", "manual_rank"],
                   default="auto_recommended",
                   help="pKaNET microstate selection policy (default: auto_recommended)")
    p.add_argument("--pkanet-rank", type=int, default=1, metavar="N",
                   help="Microstate rank used with --pkanet-selection manual_rank")
    p.add_argument("--no-add-h",      action="store_true",
                   help="Disable hydrogen addition for structure-file inputs. PDB conversion uses Meeko and validates atom names/mapping; "
                        "PDB keeps existing H atoms (Meeko merging disabled) and restores original atom-name fields. "
                        "Torsion-tree reordering is mapped. PDBQT always bypasses preparation.")
    p.add_argument("--conformer-seed", type=_nonnegative_seed, default=None, metavar="N",
                   help="RDKit 3D conformer seed; omitted or 0 = random (default: random)")


def _add_vina_args(p):
    p.add_argument("-e", "--exhaustiveness", type=int, default=16, metavar="N",
                   help="Search thoroughness — higher is slower but more reliable "
                        "(8=quick screen, 16=standard, 32=publication; default: 16)")
    p.add_argument("-n", "--poses",          type=int, default=10, metavar="N",
                   help="Max poses to output, ranked by affinity (default: 10)")
    p.add_argument("--energy-range",         type=float, default=3.0, metavar="KCAL",
                   help="Max energy gap from best pose in kcal/mol — poses worse than "
                        "(best + range) are discarded (default: 3.0)")
    p.add_argument("--seed",                 type=int, default=None, metavar="N",
                   help="Optional AutoDock Vina random seed for reproducible docking runs")


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="acd",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    root.add_argument("--version", action="version",
                      version=f"%(prog)s {_VERSION} (anyonecandock)")
    sub = root.add_subparsers(dest="command", required=True)

    # ── dock ──────────────────────────────────────────────────────────────────
    p_dock = sub.add_parser(
        "dock",
        help="Dock one molecule into a protein  (the main command)",
        description=textwrap.dedent("""\
            Dock a single molecule into a protein, start to finish.

            Give it a molecule (a drug name, a SMILES string, or a structure
            file) and a protein (a PDB ID or a file) — acd handles the rest.

            What happens under the hood
            ----------------------------
            The receptor is downloaded from RCSB (or loaded from file), cleaned,
            protonated, and converted to PDBQT. SMILES ligands use protonation and
            3D preparation. Structure files use file preparation; PDB atom mapping
            is validated. Prepared PDBQT inputs pass directly to AutoDock Vina
            without ligand preparation, with an input checksum check.
            Bond orders are corrected in the output SDF for visualization.

            Outputs (written to --output directory)
            ----------------------------------------
              receptor/rec.pdb                  cleaned receptor
              receptor/rec.pdbqt                docking-ready receptor
              receptor/rec.box.txt              Vina config (centre + box size)
              receptor/receptor_summary.json    reuse with --receptor-json
              ligand/<name>.pdbqt               prepared ligand (except direct PDBQT)
              ligand/<name>_3d.sdf              3D conformer (except direct PDBQT)
              <name>_out.pdbqt                  all docked poses
              <name>_out.sdf                    all poses (raw SDF)
              <name>_pv_ready.sdf               bond-order-corrected SDF
              <name>_scores.csv                 affinity + RMSD vs crystal per pose
              <name>_diagram.svg                2D interaction diagram (--diagram)

            Examples
            --------
              # Dock by compound name (PubChem lookup)
              acd dock --receptor 1M17 --compound erlotinib

              # Dock by SMILES
              acd dock --receptor 4AGN --smiles "c1ccc(cc1)O" --name phenol

              # Dock from a structure file
              acd dock --receptor receptor.pdb --ligand-file ligand.sdf --name mylig

              # Reuse a prepared receptor (fast — skips re-prep)
              acd dock --receptor-json ./acd_results/receptor/receptor_summary.json \\
                       --compound baicalein

              # With reference docking + 2D diagram + save individual poses
              acd dock --receptor 4AGN --compound gefitinib \\
                       --redock-smiles "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC Erlotinib" \\
                       --diagram --save-poses -e 32

              # Dock from a PDB ligand file without adding H (pure format conversion)
              acd dock --receptor protein.pdb --ligand-file ligand.pdb \\
                       --name mylig --neutral --no-add-h
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_receptor_args(p_dock)
    _add_ligand_args(p_dock)
    _add_vina_args(p_dock)
    p_dock.add_argument("-o", "--output",        default="./acd_results",
                        help="Output directory (default: ./acd_results)")
    p_dock.add_argument("--redock-smiles",        metavar="'SMILES [name]'",
                        help="Dock a reference co-crystal ligand alongside yours. "
                             "Format: SMILES [name]  (space-separated)")
    p_dock.add_argument("--save-poses",           action="store_true",
                        help="Save each docked pose as an individual SDF + PDB file")
    p_dock.add_argument("--diagram",              action="store_true",
                        help="Generate a 2D interaction diagram SVG after docking")
    p_dock.add_argument("--diagram-cutoff",       type=float, default=4.5, metavar="Å",
                        help="Interaction distance cutoff for the 2D diagram (default: 4.5 Å)")

    # ── batch ─────────────────────────────────────────────────────────────────
    p_batch = sub.add_parser(
        "batch",
        help="Dock many molecules into one protein",
        description=textwrap.dedent("""\
            Dock a whole list of molecules independently against one protein,
            then get a ranked table telling you which ones bind best.

            Point --ligands at a .smi file (one molecule per line), or pass them
            inline with --smiles-list.

            .smi file format (one ligand per line)
            ----------------------------------------
              SMILES [name]       # name is optional; defaults to lig_1, lig_2, ...
              CCO ethanol
              c1ccccc1O phenol
              # lines starting with # are ignored

            Outputs (written to --output directory)
            ----------------------------------------
              receptor/receptor_summary.json
              <name>/<name>_out.sdf       per-ligand docked poses
              <name>/<name>_pv_ready.sdf  bond-order-corrected poses
              <name>/<name>_out.pdbqt
              batch_scores.csv            ranked summary of all ligands

            Examples
            --------
              acd batch --receptor 1M17 --ligands compounds.smi
              acd batch --receptor 4AGN --smiles-list "CCO ethanol" "c1ccccc1O phenol"
              acd batch --receptor-json ./rec/receptor_summary.json --ligands hits.smi
              acd batch --receptor 1M17 --ligands flavonoids.smi \\
                        --redock-smiles "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC Erlotinib" \\
                        -e 8
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_receptor_args(p_batch)
    _add_vina_args(p_batch)
    p_batch.add_argument("--ph",           type=_target_ph, default=7.4,
                         help="Target pH for all ligands, 0–14 (default: 7.4)")
    p_batch.add_argument("--pkanet",       action="store_true", default=True,
                         help="Use pKaNET tautomer-aware protonation (default)")
    p_batch.add_argument("--neutral",      action="store_true",
                         help="Neutral mode: keep input charge, add H only")
    p_batch.add_argument("--no-pubchem", action="store_true",
                         help="Skip PubChem experimental pKa lookup")
    p_batch.add_argument("--max-tautomers", type=int, choices=range(1, 21), default=8, metavar="N",
                         help="Max tautomers enumerated by pKaNET (default: 8)")
    p_batch.add_argument("--ph-window", type=float,
                         choices=tuple(round(i / 10, 1) for i in range(2, 21)), default=1.0,
                         help="pH window for protonation enumeration (default: 1.0)")
    p_batch.add_argument("--pkanet-selection",
                         choices=["auto_recommended", "highest_score", "manual_rank"],
                         default="auto_recommended")
    p_batch.add_argument("--pkanet-rank", type=int, default=1, metavar="N")
    p_batch.add_argument("--conformer-seed", type=_nonnegative_seed, default=None, metavar="N",
                         help="RDKit 3D conformer seed; omitted or 0 = random (default: random)")
    p_batch.add_argument("--ligands",      metavar="FILE.smi",
                         help="Path to .smi file — one 'SMILES [name]' per line")
    p_batch.add_argument("--smiles-list",  nargs="+", metavar="'SMILES [name]'",
                         help="Inline SMILES list — each entry as 'SMILES name'")
    p_batch.add_argument("--redock-smiles", metavar="'SMILES [name]'",
                         help="Dock a reference co-crystal ligand for score comparison")
    p_batch.add_argument("-o", "--output", default="./acd_batch",
                         help="Output directory (default: ./acd_batch)")

    # ── codock: one Vina search with multiple ligands ──────────────────────
    p_codock = sub.add_parser(
        "codock",
        help="Dock two or more ligands simultaneously as a joint complex",
        description=("Run one Vina search with multiple ligands. The reported "
                     "affinity belongs to the joint ligand complex, not to each ligand."),
        epilog=textwrap.dedent("""\
            Ligand input examples (at least two ligands after expanding .smi files):
              SMILES:          acd codock --receptor protein.pdb --ligand "CCO ethanol" "c1ccccc1O phenol"
              Compound names:  acd codock --receptor protein.pdb --ligand aspirin caffeine
              Structure files: acd codock --receptor protein.pdb --ligand ligand1.sdf ligand2.pdbqt
              SMILES file:     acd codock --receptor protein.pdb --ligand pair.smi
              Mixed inputs:    acd codock --receptor protein.pdb --ligand "CCO ethanol" aspirin ligand3.sdf

            Quote inputs containing spaces. A .smi file has one 'SMILES [name]'
            per line. Compound names require a PubChem match and network access.
            Supported structure files: .sdf, .mol2, .pdb, .pdbqt.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_receptor_args(p_codock)
    _add_vina_args(p_codock)
    p_codock.add_argument("--ligand", nargs="+", metavar="INPUT",
                           help="Two or more quoted SMILES, compound names, or ligand paths; .smi files expand to their contents")
    p_codock.add_argument("--ligands", metavar="FILE.smi",
                           help=argparse.SUPPRESS)
    p_codock.add_argument("--smiles-list", nargs="+", metavar="'SMILES [label]'",
                           help=argparse.SUPPRESS)
    p_codock.add_argument("--ligand-files", nargs="+", metavar="PATH",
                           help=argparse.SUPPRESS)
    p_codock.add_argument("--ph", type=_target_ph, default=7.4,
                           help="Target pH for SMILES preparation (default: 7.4)")
    p_codock.add_argument("--neutral", action="store_true",
                           help="Keep input SMILES charge state and add hydrogens")
    p_codock.add_argument("--no-pubchem", action="store_true")
    p_codock.add_argument("--max-tautomers", type=int, default=8)
    p_codock.add_argument("--ph-window", type=float, default=1.0)
    p_codock.add_argument("--conformer-seed", type=_nonnegative_seed, default=None)
    p_codock.add_argument("-o", "--output", default="./acd_codock",
                           help="Output directory (default: ./acd_codock)")

    # ── ensemble ──────────────────────────────────────────────────────────────
    p_ensemble = sub.add_parser(
        "ensemble",
        help="Dock many molecules against multiple receptor conformations",
        description=textwrap.dedent("""\
            Dock a ligand library independently against every member of a
            receptor ensemble. Inputs may be files, directories, glob patterns,
            or four-character PDB IDs. Each receptor uses the existing ACD batch
            workflow, including independent receptor and heme-state validation.

            Use a shared manual box only when all structures are aligned.
            Automatic and selection-based centers are evaluated per receptor.

            Outputs
            -------
              001_<receptor>/batch_scores.csv  per-receptor docking results
              ensemble_receptors.csv           receptor preparation/run status
              ensemble_scores.csv              all receptor-ligand score rows
              ensemble_summary.csv             best/mean/median/SD per ligand

            Examples
            --------
              acd ensemble --receptors "ensemble/*.pdb" --ligands compounds.smi

              acd ensemble --receptors model1.pdb model2.cif 1M17 \\
                           --smiles-list "CCO ethanol" "c1ccccc1O phenol" \\
                           --center manual --cx 10 --cy 12 --cz -4 \\
                           --bx 20 --by 20 --bz 20 --seed 42
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_ensemble.add_argument(
        "--receptors", nargs="+", required=True, metavar="PDB_ID_FILE_DIR_OR_GLOB",
        help="Receptor files, directories, glob patterns, or four-character PDB IDs",
    )
    _add_receptor_box_args(p_ensemble)
    _add_vina_args(p_ensemble)
    p_ensemble.add_argument("--ph", type=_target_ph, default=7.4,
                            help="Target pH for all ligands, 0–14 (default: 7.4)")
    p_ensemble.add_argument("--pkanet", action="store_true", default=True,
                            help="Use pKaNET tautomer-aware protonation (default)")
    p_ensemble.add_argument("--neutral", action="store_true",
                            help="Neutral mode: keep input charge, add H only")
    p_ensemble.add_argument("--no-pubchem", action="store_true",
                            help="Skip PubChem experimental pKa lookup")
    p_ensemble.add_argument("--max-tautomers", type=int, choices=range(1, 21), default=8, metavar="N",
                            help="Max tautomers enumerated by pKaNET (default: 8)")
    p_ensemble.add_argument("--ph-window", type=float,
                            choices=tuple(round(i / 10, 1) for i in range(2, 21)), default=1.0,
                            help="pH window for protonation enumeration (default: 1.0)")
    p_ensemble.add_argument("--pkanet-selection",
                            choices=["auto_recommended", "highest_score", "manual_rank"],
                            default="auto_recommended")
    p_ensemble.add_argument("--pkanet-rank", type=int, default=1, metavar="N")
    p_ensemble.add_argument("--conformer-seed", type=_nonnegative_seed, default=None, metavar="N",
                            help="RDKit 3D conformer seed; omitted or 0 = random")
    p_ensemble.add_argument("--ligands", metavar="FILE.smi",
                            help="Path to .smi file — one 'SMILES [name]' per line")
    p_ensemble.add_argument("--smiles-list", nargs="+", metavar="'SMILES [name]'",
                            help="Inline SMILES list — each entry as 'SMILES name'")
    p_ensemble.add_argument("--redock-smiles", metavar="'SMILES [name]'",
                            help="Dock a reference co-crystal ligand for each receptor")
    p_ensemble.add_argument("-o", "--output", default="./acd_ensemble",
                            help="Output directory (default: ./acd_ensemble)")

    # ── receptor ──────────────────────────────────────────────────────────────
    p_rec = sub.add_parser(
        "receptor",
        help="Prepare a protein only  (.pdb + .pdbqt + docking box)",
        description=textwrap.dedent("""\
            Download or load a protein structure, clean it, add hydrogens,
            and write the PDBQT + Vina box config.  Saves receptor_summary.json
            so subsequent  acd dock / redock / batch  runs can pass
            --receptor-json and skip re-preparation entirely.

            Examples
            --------
              acd receptor --pdb 4AGN
              acd receptor --pdb 4AGN --fmt CIF
              acd receptor --file structure.pdb --center auto
              acd receptor --file structure.pdb --center manual --cx 10.5 --cy -3.2 --cz 8.0
              acd receptor --pdb 1M17 --center selection --sel "resid 820 and chain A"
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_rec.add_argument("--pdb",    metavar="PDB_ID",
                       help="4-letter PDB ID to download from RCSB")
    p_rec.add_argument("--file",   metavar="PATH",
                       help="Path to local .pdb or .cif file")
    p_rec.add_argument("--fmt",    default="PDB", choices=["PDB", "CIF"],
                       help="Download format when using --pdb (default: PDB)")
    p_rec.add_argument("--center", default="auto",
                       choices=["auto", "manual", "selection"],
                       help="Box centre mode: auto=co-crystal centroid, "
                            "manual=--cx/cy/cz, selection=ProDy string (default: auto)")
    p_rec.add_argument("--cx", type=float, help="Box centre X (--center manual)")
    p_rec.add_argument("--cy", type=float, help="Box centre Y (--center manual)")
    p_rec.add_argument("--cz", type=float, help="Box centre Z (--center manual)")
    p_rec.add_argument("--sel", metavar="PRODY_SEL",
                       help="ProDy selection string, e.g. 'resid 702 and chain A'")
    p_rec.add_argument("--resname", metavar="3-LETTER-CODE",
                       help="For --center auto, use this ligand residue name as the grid center/reference when multiple ligands exist (e.g. ATP, DC3)")
    p_rec.add_argument("--yes", action="store_true",
                       help="Non-interactive mode: when multiple co-crystal ligands are detected, accept the default [0] ligand without prompting")
    p_rec.add_argument("--blind", action="store_true",
                       help="Cover the whole protein using 4 Å padding")
    p_rec.add_argument("--bx", type=_box_size, default=18,
                       help="Search box X size in Å (positive integer, default: 18)")
    p_rec.add_argument("--by", type=_box_size, default=18,
                       help="Search box Y size in Å (positive integer, default: 18)")
    p_rec.add_argument("--bz", type=_box_size, default=18,
                       help="Search box Z size in Å (positive integer, default: 18)")
    p_rec.add_argument("-o", "--output", default="./acd_receptor",
                       help="Output directory (default: ./acd_receptor)")

    # ── ligand ────────────────────────────────────────────────────────────────
    p_lig = sub.add_parser(
        "ligand",
        help="Prepare a molecule only  (.pdbqt + 3D .sdf)",
        description=textwrap.dedent("""\
            Protonate a ligand at the target pH via pKaNET, generate a 3D
            conformer, and write PDBQT + SDF.  No receptor or docking needed.

            Examples
            --------
              acd ligand --smiles "c1ccccc1O" --name phenol
              acd ligand --compound erlotinib --ph 7.4
              acd ligand --file ligand.sdf --name mylig
              acd ligand --file ligand.pdb --name mylig --no-add-h
              acd ligand --smiles "CCN" --name ethylamine --ph 7.4 --no-pubchem
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_ligand_args(p_lig)
    p_lig.add_argument("--file", metavar="PATH",
                       help="Ligand file (.sdf / .mol2 / .pdb / .pdbqt); PDBQT is passed unchanged; structure files skip protonation")
    p_lig.add_argument("-o", "--output", default="./acd_ligand",
                       help="Output directory (default: ./acd_ligand)")

    # ── diagram ───────────────────────────────────────────────────────────────
    p_diag = sub.add_parser(
        "diagram",
        help="Draw a 2D interaction map from a finished docking run",
        description=textwrap.dedent("""\
            Generate a publication-ready 2D ligand-receptor interaction diagram
            from an existing receptor PDB and docked pose SDF.

            8 interaction types detected: H-bond, hydrophobic, π-π stacking,
            cation-π, ionic, metal/heme coordination, halogen bond, H···halogen.

            Examples
            --------
              acd diagram --receptor rec.pdb --pose-sdf pose1.sdf
              acd diagram --receptor rec.pdb --pose-sdf poses.sdf --pose 2 \\
                          --smiles "c1ccc(cc1)O" --name phenol --score -8.5
              acd diagram --receptor rec.pdb --pose-sdf pose1.sdf --cutoff 4.0
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_diag.add_argument("--receptor", required=True, metavar="PATH",
                        help="Path to the cleaned receptor PDB (rec.pdb from acd receptor)")
    p_diag.add_argument("--pose-sdf", required=True, metavar="PATH",
                        help="Path to the docked pose SDF (single pose or multi-model)")
    p_diag.add_argument("--smiles",   metavar="SMILES",
                        help="Ligand SMILES — used to fix bond orders in the diagram")
    p_diag.add_argument("--name",     default="ligand",
                        help="Ligand label shown in the diagram title (default: ligand)")
    p_diag.add_argument("--pose",     type=int, default=1,
                        help="Pose index to diagram when SDF has multiple models (default: 1)")
    p_diag.add_argument("--score",    type=float,
                        help="Binding affinity in kcal/mol to display in the diagram title")
    p_diag.add_argument("--cutoff",   type=float, default=4.5,
                        help="Interaction distance cutoff in Å (default: 4.5)")
    p_diag.add_argument("-o", "--output", default="./acd_diagram",
                        help="Output directory (default: ./acd_diagram)")

    # ── redock ────────────────────────────────────────────────────────────────
    p_redock = sub.add_parser(
        "redock",
        help="Check your setup: re-dock the protein's own ligand (no SMILES needed)",
        description=textwrap.dedent("""\
            The best way to sanity-check your docking setup before trusting it.

            Re-docks the ligand that is already in the input PDB/CIF and measures
            how close the result lands to the real crystal pose (RMSD). If it
            lands close, your box, settings, and pipeline are trustworthy.
            No SMILES or compound name needed — acd figures the molecule out.

            SMILES acquisition (tried in order)
            ------------------------------------
              1. RCSB Chemical Component Dictionary (CCD) REST API
                     → ideal, stereo-correct, curated SMILES
              2. _chem_comp.pdbx_smiles field in the CIF file
                     → no network call; works offline for CIF inputs
              3. 3D → SMILES via OpenBabel + RDKit (H-aware bond perception)
                     → last resort; a warning is printed; verify the result

            RMSD verdict (printed after docking)
            --------------------------------------
              ✓ PASS        ≤ 2.0 Å  — docking protocol validated
              ⚠ BORDERLINE  2–3 Å    — acceptable for flexible loops
              ✗ FAIL        > 3.0 Å  — pose differs from crystal;
                                        check box placement and ligand SMILES

            Outputs (written to --output directory)
            ----------------------------------------
              receptor/receptor_summary.json
              ligand/<name>.pdbqt / <name>_3d.sdf
              <name>_out.pdbqt / _out.sdf / _pv_ready.sdf
              <name>_redock_scores.csv    affinity + RMSD vs crystal per pose
              <name>_diagram.svg          (--diagram only)

            Examples
            --------
              acd redock --receptor 4AGN
              acd redock --receptor structure.cif --fmt CIF
              acd redock --receptor 4AGN --resname DC3 --diagram
              acd redock --receptor 4AGN --resname ATP --ph 6.5 -e 32
              acd redock --receptor-json ./rec/receptor_summary.json
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_redock.add_argument("--receptor",      metavar="PDB_ID_or_FILE",
                          help="4-letter PDB ID (auto-downloaded from RCSB) or path to .pdb/.cif")
    p_redock.add_argument("--receptor-json", metavar="PATH",
                          help="JSON from a previous  acd receptor  run — skips re-preparation")
    p_redock.add_argument("--fmt",    default="PDB", choices=["PDB", "CIF"],
                          help="Download format when using a PDB ID (default: PDB)")
    p_redock.add_argument("--center", default="auto",
                          choices=["auto", "manual", "selection"],
                          help="Box centre mode (default: auto)")
    p_redock.add_argument("--cx", type=float, help="Box centre X (--center manual)")
    p_redock.add_argument("--cy", type=float, help="Box centre Y (--center manual)")
    p_redock.add_argument("--cz", type=float, help="Box centre Z (--center manual)")
    p_redock.add_argument("--sel", metavar="PRODY_SEL",
                          help="ProDy selection string (--center selection)")
    p_redock.add_argument("--blind", action="store_true",
                          help="Cover the whole protein using 4 Å padding")
    p_redock.add_argument("--bx", type=_box_size, default=18,
                          help="Search box X size in Å (positive integer, default: 18)")
    p_redock.add_argument("--by", type=_box_size, default=18,
                          help="Search box Y size in Å (positive integer, default: 18)")
    p_redock.add_argument("--bz", type=_box_size, default=18,
                          help="Search box Z size in Å (positive integer, default: 18)")
    p_redock.add_argument("--resname",  metavar="3-LETTER-CODE",
                          help="Override the auto-detected ligand residue name "
                               "(e.g. DC3, ATP) — useful when multiple HETATMs exist "
                               "or when using --receptor-json")
    p_redock.add_argument("--name",          default="",
                          help="Output name for the re-docked ligand (default: residue code)")
    p_redock.add_argument("--ph",            type=_target_ph, default=7.4,
                          help="Target pH for protonation, 0–14 (default: 7.4)")
    p_redock.add_argument("--neutral",       action="store_true",
                          help="Neutral mode: keep input charge, add H only")
    p_redock.add_argument("--no-pubchem",    action="store_true",
                          help="Skip PubChem pKa lookup")
    p_redock.add_argument("--max-tautomers", type=int, choices=range(1, 21), default=8,
                          help="Max tautomers for pKaNET (default: 8)")
    p_redock.add_argument("--ph-window", type=float,
                          choices=tuple(round(i / 10, 1) for i in range(2, 21)), default=1.0,
                          help="pH window for protonation enumeration (default: 1.0)")
    p_redock.add_argument("--pkanet-selection",
                          choices=["auto_recommended", "highest_score", "manual_rank"],
                          default="auto_recommended")
    p_redock.add_argument("--pkanet-rank", type=int, default=1, metavar="N")
    p_redock.add_argument("--conformer-seed", type=_nonnegative_seed, default=None, metavar="N",
                          help="RDKit 3D conformer seed; omitted or 0 = random (default: random)")
    _add_vina_args(p_redock)
    p_redock.add_argument("-o", "--output",      default="./acd_redock",
                          help="Output directory (default: ./acd_redock)")
    p_redock.add_argument("--save-poses",        action="store_true",
                          help="Save each docked pose as an individual SDF + PDB file")
    p_redock.add_argument("--diagram",           action="store_true",
                          help="Generate a 2D interaction diagram SVG after docking")
    p_redock.add_argument("--diagram-cutoff",    type=float, default=4.5, metavar="Å",
                          help="Interaction distance cutoff for the 2D diagram (default: 4.5 Å)")

    return root


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "dock":     cmd_dock,
        "batch":    cmd_batch,
        "codock":   cmd_codock,
        "ensemble": cmd_ensemble,
        "receptor": cmd_receptor,
        "ligand":   cmd_ligand,
        "diagram":  cmd_diagram,
        "redock":   cmd_redock,
    }
    fn = dispatch.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)

    try:
        fn(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
