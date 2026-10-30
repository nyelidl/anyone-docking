# MCP ion and crystallization-additive fixes

Branch: `fix/mcp-ion-and-lcp-additives`. No push or PyPI upload was performed.

## Changes

- `st/pose_access.py`: case-sensitive ion mappings cover all 27 METAL_RESNAMES. `Na` means sodium; `NA` remains nitrogen acceptor. No retained metal is unmapped. Canonical and uppercase forms are accepted except uppercase NA, which must retain its AutoDock meaning.
- `core.py`, `st/core.py`, `anyonecandock/anyonecandock/core.py`: add exactly the requested lipid/additive and buffer codes, exclude buffers from the legacy removable-ligand candidate list, and preserve the six specified genuine-ligand codes. If auto mode has no genuine reference, choose the largest eligible additive/buffer with a warning; when there are no eligible HETATMs, use the protein geometric center with a warning. Explicit selections retain their existing precedence. Additive reference warnings also appear in the preparation manifest.
- `st/api.py`: preserves reference_warning in saved receptor metadata and both compact response forms, alongside the exact cocrystal_ligand_id.
- `tests/`, `tools/check_core_sync.py`, `pytest.ini`: ion, classification, source-sync, warning, fallback and real-PDB regression checks.
- `anyonecandock/pyproject.toml`: version 1.3.12. The clean upstream checkout did not contain CLI source, so the user's local 1.3.11 source from Downloads/ACD_API/anyonecandock was imported into `anyonecandock/` before patching. The other imported CLI modules are unchanged from that local source.
- `st/Dockerfile`, `st/Dockerfile.txt`, `st/README.md`: use the bundled 1.3.12 wheel. Replaces the bundled 1.3.1 wheel.

No changes to Streamlit app-1.py through app-24.py or index.html. Optional chemistry-similarity ranking (2b) is not implemented; candidate ordering among remaining genuine ligands stays unchanged.

## Validation

- Offline suite against installed replacement wheel: 44 passed, 2 network tests deselected.
- Both network tests passed against the installed wheel: real preparation selected CFF_A_2401 for 5MZP and AQ4_A_999 for 1M17.
- Both no-reference fallback branches passed offline, with the conversion layer isolated.
- All changed Python files pass py_compile. The three classification constant sets match exactly.
- Wheel build succeeded (wheel and source archive); wheel metadata is 1.3.12 and its core.py is byte-identical to the patched package source.
- Docker is not installed on this machine. `docker build -t acd-gpt ./st` could not be executed and is NOT claimed to pass. Run it before deployment in a Docker-capable environment. No end-to-end remote MCP docking/viewer test was run here; pose parsing and real receptor preparation were tested directly.

## Exact Render redeploy procedure

1. Review this branch locally. If approved, push `fix/mcp-ion-and-lcp-additives` to GitHub yourself; no push was made by the agent. Alternatively merge an approved PR through your normal process. Do not upload either wheel to PyPI for this deployment.
2. Confirm the deployed branch contains updated st/api.py, st/pose_access.py, st/core.py, Dockerfile, Dockerfile.txt, README.md, and anyonecandock-1.3.12-py3-none-any.whl; remove the old 1.3.1 wheel. Preserve st/requirements-api.txt, Aptfile and viewer3d.html.
3. In a Docker-capable checkout, run `docker build -t acd-gpt ./st` from the repository root. Resolve any environment/package issue before redeployment.
4. In Render, open the existing ACD API web service. In Settings, select the reviewed deployment branch. Preserve its environment variables, authentication key, resource limits and persistent storage configuration.
5. Keep the Docker build context equivalent to this repository's st/ directory: for a Render Root Directory of st, use Dockerfile path ./Dockerfile and build context . relative to that Root Directory. The required invariant is that Docker COPY reads st/anyonecandock-1.3.12-py3-none-any.whl, not a root-level file. With repository-root build settings, use Dockerfile path ./st/Dockerfile and build context ./st.
6. Choose Manual Deploy → Clear build cache & deploy. Check the build log installs the bundled 1.3.12 wheel and startup succeeds.
7. Check /ping. Submit a fresh caffeine/5MZP job in auto mode and confirm receptor.cocrystal_ligand_id is CFF_A_2401. Also submit with reference_hetatm_key=CFF|A|2401, then verify the returned 3D viewer and pose-interactions URLs succeed with the retained sodium present. Recheck a fresh 1M17 job. Old jobs keep their original box/reference; rerun jobs that used the crystallization lipid.

The Cloudflare MCP connector needs no route or schema change for these fixes.
