# Exact 3D pose access (API 0.3.0)

This revision adds read-only retrieval of saved docking poses and their viewer.
Updating only the GPT schema cannot create these endpoints: deploy the backend
files first, then replace the GPT Action schema and instructions.

Upload these changed/new files into GitHub `st/`:

- `api.py`
- `pose_access.py`
- `viewer3d.html`
- `Dockerfile` (and `Dockerfile.txt` if retaining its mirror)
- `openapi.json`
- `gpt-action-schema.json`

Keep the existing Aptfile, requirements-api.txt, and bundled CLI wheel in st/.
Docker now copies pose_access.py and viewer3d.html alongside api.py. No new Python
dependencies are needed. The viewer loads the official 3Dmol.js browser library
from https://3Dmol.org/build/3Dmol-min.js and needs WebGL/internet access.

## Actions

| Endpoint | Operation |
|---|---|
| GET /jobs/{job_id}/selected-pose | getSelectedDockingPose |
| GET /jobs/{job_id}/selected-pose/3d | getDocking3DViewer |
| GET /jobs/{job_id}/poses/{pose_rank}/structure | getDockingPoseStructure |
| GET /jobs/{job_id}/poses/{pose_rank}/interactions | getDockingPoseInteractions |
| GET /jobs/{job_id}/files/{filename}/url | getDockingJobFile |

All structure/interaction endpoints accept `ligand_name` and `pocket_distance`
(default 4.5 Å; range 2–8 Å). Multi-ligand jobs require an explicit ligand_name.
Use filenames and URLs returned by the API: generated structure filenames use
an isolated namespace rather than guessing ligand output paths.

```bash
curl 'https://anyone-docking-api-docker.onrender.com/jobs/dock_6b613a5ef20a/selected-pose?ligand_name=lapatinib'
```

This example succeeds only if that completed job's coordinate files still exist
on the deployed server. No new docking is started by these endpoints.

## Coordinate provenance

- Parse each saved Vina MODEL rank and its REMARK VINA RESULT score. Cross-check
  the existing score table; disagreement fails with HTTP 409.
- Copy physical atom coordinates from that exact MODEL into pose PDBQT/PDB.
  No embedding, optimization, coordinate fitting, or independent conformation.
- Export the saved prepared receptor PDBQT as PDB (or saved receptor PDB when
  PDBQT was not recorded). No RCSB refetch and no receptor coordinate transform.
- Combine receptor and ligand atoms in one PDB; ligand records are HETATM.
  Renumber serials to avoid duplicates. If residue identity collides with the
  receptor, move only the ligand chain label and return complex_ligand_chain_map.
- Keep existing SDF chemistry by extracting the matching saved SDF record.
  Validate one-to-one element/coordinate correspondence for every heavy atom
  against the exact PDBQT pose within 0.0011 Å. If it cannot be validated, do not
  publish an SDF URL; return a warning and exact PDB/PDBQT links instead.
- Return hashes of receptor PDB, extracted pose PDBQT, complex PDB, source files,
  and optional SDF. `hash_definitions` describes the three primary hashes.
- Retain PDB ID, ligand name, prepared SMILES, charge, protonation mode, rank,
  affinity, grid center/size, selection method, and fallback reason.

The PDB format does not encode full ligand bond-order chemistry; viewer bonds
are a rendering interpretation of the saved coordinates, not an interaction
analysis. Nonphysical Meeko macrocycle glue atoms are excluded from PDB exports;
original PDBQT contains all records. Unsupported atom types or PDB capacity
limits fail explicitly rather than inventing elements or overflowing fields.

Normal selection uses the lowest Vina score. Redocking uses the existing
co-crystal/reference identification and lowest finite heavy-atom RMSD. New job
RMSD selection validates SDF record identities against Vina ranks first, avoiding
rank shifts when a record is unreadable. Failed RMSD falls back to lowest score
with an explicit warning. Older job metadata can supply an existing validated
selection/RMSD; missing metadata cannot be recreated by guessing.

## Interactions and visualization limits

Chemical classification (PLIP or equivalent) is **not configured** in this
revision. `interactions` is empty with a clear source/warning. Separately,
`proximity_contacts` reports the nearest ligand heavy atom for each receptor
heavy atom within the requested cutoff, using Euclidean distances. These are
not hydrogen bonds or other classified interactions. Contacts and pocket-residue
lists are capped at 200 each with total counts and truncation flags.

The browser viewer supports cartoon/surface/stick protein styles, element-colored
ligand sticks/ball-and-stick, configurable pocket residues, residue labels,
optional distance lines, rotation/zoom/pan, and browser PNG export with the score
caption. Classified-interaction controls stay disabled when no analysis exists.
There is no server PNG renderer yet; render_3d_png_url remains empty. The viewer
and coordinate links are the supported initial rendering workflow.

The viewer HTML shell contains no coordinates or keys. Protected coordinate
fetches still require X-API-Key. For a protected deployment, enter the key into
the viewer's in-memory field; it is not placed in a URL or browser storage.

Selected-pose artifacts are generated before new-job ZIP creation. Other poses
are extracted on demand. Older completed jobs can be visualized if their saved
files remain on disk; on-demand artifacts are directly downloadable but are not
retroactively added to an already-created ZIP. Existing API routes remain.

## Verification

Automated tests cover exact pose/score identity, untransformed coordinates,
complex contents and chain collisions, SDF mismatch rejection, RMSD selection
and fallback, distance-contact provenance, errors for absent/running jobs,
batch ligand disambiguation, file hashes/downloads, authentication, and schema
operations. Tests use saved synthetic Vina outputs; no production docking runs.

The viewer was loaded in a browser against local synthetic fixtures. Molecular
rendering, proximity toggle, responsive layout, and on-canvas score caption were
visually checked. The PNG export control was exercised with no browser errors.
This test fixture is not an actual lapatinib/1M17 scientific docking result.
Docker and live Render deployment remain untested from this environment.

Final validation command:

```bash
PYTHONPATH=ACD_gpt:/tmp/acd-gpt-bundle-installed /tmp/acd-cli-ligand-tests/bin/python -m pytest ACD_gpt/tests ACD_API/tests -q --tb=short
```

Result: **100 passed, 1 warning in 2.10s**. The warning is Starlette's test-client
httpx deprecation. All 15 Action operation IDs are unique and local schema
references resolve. Docker COPY inputs exist and the Dockerfiles match.

`deploy-st.zip` contains the current deployable st/ folder, including the CLI
wheel and both new runtime files. Extract it before uploading the st/ contents;
uploading only the ZIP file will not provide Docker's required input paths.
