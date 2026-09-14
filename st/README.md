# Anyone Can Dock GPT API

**API 0.3.0 adds exact 3D pose access.** See [POSE_ACCESS.md](POSE_ACCESS.md)
for endpoints and deployment, [gpt-action-schema.json](gpt-action-schema.json)
for the updated Actions, and [GPT_INSTRUCTIONS.md](GPT_INSTRUCTIONS.md) for
coordinate-grounded GPT behavior.

This folder is self-contained and can be uploaded as `st/` in the GitHub
repository. The deployable CLI package is `anyonecandock-1.3.1-py3-none-any.whl`, including
`ligand_validation.py` and heme data. Upload this wheel beside api.py and Dockerfile
in st/. The vendor/ directory is retained as source but is not needed by Docker.
The root-level `api.py` outside this folder is a separate older copy.

## Deployment

Use this folder as the Docker **build context**. In the local workspace:

```bash
docker build -t acd-gpt ./ACD_gpt
```

After uploading its contents into GitHub's `st/` folder:

```bash
docker build -t acd-gpt ./st
docker run --rm -p 10000:10000 \
  -e DOCKGPT_API_KEY -e PUBLIC_BASE_URL \
  acd-gpt
```

For Render, set **Root Directory: st**, **Dockerfile Path: ./Dockerfile**,
and **Docker Build Context Directory: .** (relative to that root).
Alternatively, with repository root left unset, use Dockerfile Path
`./st/Dockerfile` and Docker Build Context Directory `./st`.
The Docker COPY paths resolve within st, so no ACD_gpt or ACD_API directory
is required in the deployed repository.

The build installs the bundled CLI wheel, not the published PyPI package.
It installs the diagram extra; this API does not use ADMET.
`vendor/source-sha256.json` records the bundled source hashes. When the CLI is
updated later, rebuild the wheel from the updated source before redeployment.

`requirements-api.txt` mirrors the GPT dependency list at
[st/requirements-api.txt](https://github.com/nyelidl/anyone-docking/blob/main/st/requirements-api.txt).
The updated shared CLI package is installed separately alongside these dependencies.

`Aptfile` is the system package list used by the Docker build. `Aptfile.txt` and
`Dockerfile.txt` are synchronized copies for workflows that previously used the
text files; canonical deployment filenames have no `.txt` suffix. Open Babel is
still needed for receptor and docked-output conversion, but is not used for
PDB/PDBQT ligand input preparation.

Run one worker: the job registry and daily quota lock are process-local. Jobs
still use background tasks rather than a durable queue; restarting interrupts
running jobs. Configure `ACD_API_WORKDIR` on persistent storage if completed
results must survive replacement of a container. Existing resource-limit
environment variables remain supported.

For a local development environment, from the repository root:

```bash
python -m pip install './ACD_gpt/vendor/anyonecandock[diagram]' -r ACD_gpt/requirements-api.txt
python -m uvicorn api:app --app-dir ACD_gpt --host 0.0.0.0 --port 10000 --workers 1
```

## Ligand input

Existing requests remain valid:

```json
{"pdb_id":"1M17","ligands":[{"name":"ethanol","smiles":"CCO"}]}
```

Each ligand must contain **either** nonempty `smiles` **or**:

- `file_name`: filename ending in `.pdb`, `.sdf`, `.mol2`, or `.pdbqt`;
- exactly one of `file_content` (text) or `file_content_base64` (original bytes);
- optional `no_add_h` (default false).

Server-side paths and remote URLs are not accepted as ligand files. The client
must send the file contents. Base64 is recommended to preserve exact original
bytes, including CRLF line endings. `name` controls output filenames only.

Example client (submits an actual job when executed):

```python
import base64
import os
from pathlib import Path
import requests

ligand = Path('ligand.pdb')  # Or ligand.pdbqt, ligand.sdf, ligand.mol2
payload = {
    'pdb_id': '1M17',
    'ligands': [{
        'name': 'my_ligand',
        'file_name': ligand.name,
        'file_content_base64': base64.b64encode(ligand.read_bytes()).decode('ascii'),
        'no_add_h': True,
    }],
    'seed': 42,             # Optional Vina seed
    'conformer_seed': 123, # Optional; used only when a conformer is generated
}
response = requests.post(
    os.environ['PUBLIC_BASE_URL'].rstrip('/') + '/dock', json=payload,
    headers={'X-API-Key': os.environ['DOCKGPT_API_KEY']}, timeout=60,
)
response.raise_for_status()
print(response.json())
```

### Behavior shared with CLI

```text
PDBQT → read-only validation → uploaded original bytes → SHA256 guard → Vina
PDB → RDKit → Meeko → restore original atom-name fields → validate mapping → Vina
SDF/MOL2 → existing shared file preparation → Vina
SMILES → existing configured protonation and preparation → Vina
```

PDB `no_add_h: true` disables addition and Meeko merging of existing hydrogens.
Normal PDB mode adds missing hydrogens and reports default Meeko merging.
PDB atom names are restored by explicit Meeko index mapping; torsion reordering
is traceable and charges/types remain Meeko's. Ambiguous mappings fail before
that ligand is docked. The shared CLI's limitations for PDB chemistry,
alternate locations, disconnected fragments, and pseudoatom mapping remain.

Direct PDBQT does not call RDKit, Meeko, protonation, or input conversion.
The original uploaded file is passed to Vina; `input_sha256` and
`vina_input_sha256` are exposed in successful results. Docked-output conversion
may still use Open Babel. Without a ligand SMILES, SMILES-based output bond-order
correction is skipped. This does not alter the original PDBQT input.

Missing sources, conflicting sources, unsupported formats, empty files,
invalid base64, and malformed PDBQT return HTTP 422 before queueing. Chemistry
or mapping failures are per-ligand failures: inspect each result's `status` and
`error`, even when the overall batch status is `completed`.

## Results and GPT integration

Poll `/jobs/{job_id}` with the API key. Results include `ligand_mode`,
`protonation_mode_used` (`not_applied` for file input), checksum fields, and
`atom_mapping_url`/`atom_mapping` when available. A mapping failure report is
exposed when the shared validator generated one. Each ligand has a separate
preparation directory, avoiding batch report collisions.

Download mapping TSVs using `atom_mapping_url` and the same authentication.
The result ZIP also contains uploaded files, prepared PDBQT, reports, metadata,
and workflow logs. Use final docked PDBQT for atom-label based pose analysis;
downstream SDF or third-party exports may have different label semantics.

The updated schema is served at `/openapi.json`; a checked-in `openapi.json`
snapshot is included here. Update the GPT Action schema from the deployed API
and configure its authentication as API key header `X-API-Key`. Set the Action
server URL to the real deployment URL. The supplied schema does not embed a
made-up deployment URL or credentials. The server changes alone do not update
an existing GPT Action configuration.

GPT instructions should select exactly one ligand source, upload file bytes
without rewriting atom names, use `no_add_h` only for structure files, poll until
completion, inspect per-ligand errors, and report checksum/mapping evidence
without claiming these are docking accuracy validations.

## Validation

```bash
python -m pip install pytest httpx
PYTHONPATH=ACD_gpt/vendor/anyonecandock python -m pytest ACD_gpt/tests ACD_API/tests -q
```

Tests use real RDKit/Meeko with mocked receptor preparation, Vina, and external
services. No production docking is required. The Docker image must additionally
be built and checked on a Docker-enabled host before deployment.

Job file paths validate the job ID. Deleting a queued/running job returns HTTP
409 so its files cannot be removed underneath the background task. See
[REVISION_REPORT.md](REVISION_REPORT.md) for the final test results and limits.

## Fix for missing vendor during deployment

If Docker reports `/vendor/anyonecandock: not found`, the old Dockerfile is still
in use. Upload the new Dockerfile and `anyonecandock-1.3.1-py3-none-any.whl`
together into st/. The new Dockerfile copies the wheel directly and has no
COPY vendor instruction. Required build files are Dockerfile, Aptfile, api.py, pose_access.py,
viewer3d.html, requirements-api.txt, and the wheel. Use st/ as the build context.
