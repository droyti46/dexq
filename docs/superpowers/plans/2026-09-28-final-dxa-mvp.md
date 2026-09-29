# DEXQ Final DXA MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the exact reference `dxa_qc_with_models.zip` QC inference locally behind DEXQ, offer honest image overlays and manual spine-axis adjustment, provide batch CSV, metrics, local samples and offline submission packaging.

**Architecture:** Preserve the reference `combined_qc` inference source byte-for-byte. A local extraction script materializes verified ONNX/NPZ assets outside Git (with a manifest covering their hashes); `Analyzer` calls a single `QCPipeline` per process through a temporary original-input file and maps its result into an explicit API DTO/`QualityCheck` results. The browser draws native-coordinate geometry without mutating machine decisions.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, pydicom 3.0.2, numpy 2.3.5, Pillow 12.3.0, scipy 1.18.1, onnxruntime 1.30.0 CPUExecutionProvider, React 18, TypeScript 5.8, Vite 8, Docker Compose.

**Spec:** [2026-09-28-dexq-final-mvp-design.md](../specs/2026-09-28-dexq-final-mvp-design.md)

## Global Constraints

- `dxa_qc_with_models.zip` SHA-256 `2cd75f3777a4a7735d8e36cca371d4637b54053fb9283863c64c7214535a8867` is the ONLY model source; never mix standalone hip V13.
- Python 3.12, public type hints and Russian Google-style docstrings; no silent fallback to the existing heuristics. The existing `QualityCheck` contract remains the outward neutral per-check contract; model and preprocessing stay inside its module/adapter.
- No DICOM, derivatives, ONNX/NPZ, per-image metric records, secrets or local sample output in Git or GitHub. Source ZIP and medical images never go to external services. Audit `git diff --cached --name-only` before every push.
- Automatic hip output has **two** flags: `hip_positioning_rotation` and `hip_roi`; do not invent individual rotation/positioning or a femur ROI segmentation.
- Projection is `unknown`/`not_determined` for the supplied corpus; neither area nor visual hip side proves AP/PA. A known unsupported projection fails safely.
- For `Success`, `quality_class` is 0 or 1; incomplete decision is `Failure` with empty CSV class. One row per input DICOM; after three images per Study UID, every later image receives its own Failure row.
- Reference CPUExecutionProvider parity: exact routing, laterality, labels and OR; scores `atol=1e-5`, `rtol=1e-4`. Acceptance time ≤180 seconds per study (≤3 images) on declared warm configuration, measured from reading/decode through postprocessing; cold start separate.
- Original model source unchanged. Keep Evolventa typography and existing UI idiom. No new state manager, UI kit, repository/UoW or unsolicited clinical claims.

## File Structure

- `backend/reference/combined_qc/` — all 69 reference `.py` files verbatim, preserving original import closure without an inferred rewrite (no training data/weights); all copied files hashed against ZIP.
- `backend/app/model_runtime.py` — verified asset paths, one persistent reference pipeline, safe temporary source and translation to a *restricted* runtime result (no raw DICOM metadata). Imports `combined_qc` by adding `backend/reference/` to `PYTHONPATH` at backend startup, not copying it into app.
- `backend/app/checks/reference.py` — explicit five reference `QualityCheck` descriptors and result conversion, replacing placeholder checks in registration only.
- `backend/app/analyzer.py`, `imaging.py`, `schemas.py` — JSON DTO assembly, safe UID extraction, projection and quality status.
- `backend/app/batch.py`, `api.py`, `cli.py` — grouped processing and strict CSV semantics.
- `scripts/prepare_reference.py`, `scripts/prepare_samples.py`, `scripts/metrics.py` — safe local model installation, private manual samples, aggregate metrics/CI.
- `frontend/src/pages/AnalyzePage.tsx`, `ResultPage.tsx`, `frontend/src/components/StudyViewer.tsx`, `frontend/src/geometry.ts`, `frontend/src/api.ts`, `types.ts`, `styles.css` — UI workflow, native-coordinate SVG overlay and separate manual angle.
- `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`, `scripts/run.sh`, `.gitignore`, `.dockerignore` — pinned offline deployment and separate model submission bundle.
- `README.md`, `SPEC.md`, `docs/deployment.md`, `docs/metrics.md`, `docs/user-guide.md`, `docs/model.md` — reality-checked handoff and warnings.

## Review Focus

1. Uploaded DICOM with a plausible filename containing a patient name → no filename/path, raw metadata or exception leaks in error/JSON beyond a sanitized operator-facing name; pin in Task 3 tests.
2. Two studies each with three files plus an extra file for the first UID → four rows for the first study, the fourth Failure, other study unaffected; pin in Task 4 tests.
3. Different image aspect ratios on mobile → draggable SVG points align with actual image native coordinates and stay in bounds; pin in Task 5 tests/manual acceptance.
4. Valid pixels but routing uncertainty or absent spine axis → cannot return `Success` with null class or allow phantom editing; pin in Tasks 3/5 tests.
5. Altered/missing model file or archive symlink/traversal entry → installation/startup fails closed with clear error and no silent heuristic fallback; pin in Tasks 1/2 tests.

---

### Task 1: Reproducible Reference Source and Offline Model Install

**Files:** Create `scripts/prepare_reference.py`, `backend/tests/test_reference_setup.py`; modify `.gitignore`, `backend/.dockerignore`, `backend/pyproject.toml`, `backend/requirements.lock`; generate `backend/reference/combined_qc/` from the exact reference ZIP (do not reformat it).

**Interfaces:** `prepare_reference(zip_path: Path, source_root: Path, models_root: Path, *, expected_zip_sha: str = REFERENCE_SHA) -> dict[str, str]` copies only allowlisted runtime source/config/assets, validates SHA, paths and file sizes, returns a manifest map. `verify_models(models_root: Path) -> None` checks every installed model on startup. Default local model root `backend/models/reference/`, source package `backend/reference/combined_qc/`. Use `Path` joins on fixed allowlisted relative names only and reject symlinks/duplicates/traversal. Mark `backend/models/`, `local-test-images/`, `submission/`, `*.onnx`, `*.npz`, `*.dcm`, `*.dicom`, medical PNG paths ignored; tracked artifacts already present require explicit audit, not an ignore-rule promise.

- [ ] **Step 1: Write failing tests** in `backend/tests/test_reference_setup.py` with a tiny synthetic ZIP carrying a fake `combined_qc/common.py` and model manifest. Assert `prepare_reference()` rejects bad ZIP SHA, `../escape`, duplicate paths, symlink entry, wrong model checksum, and passes verified file bytes. Keep tests tiny; inject expected hash/prefix into the setup function for fixtures, while production pins the constant above.
  ```python
  def test_rejects_traversal(tmp_path: Path) -> None:
      with ZipFile(tmp_path / "input.zip", "w") as archive:
          archive.writestr("dxa_qc_code/combined_qc/../escape.py", b"bad")
      with pytest.raises(ValueError, match="path"):
          prepare_reference(tmp_path / "input.zip", tmp_path / "src", tmp_path / "models", expected_zip_sha=sha256(tmp_path / "input.zip"))
  ```
- [ ] **Step 2: Run** `cd backend && .venv/Scripts/python -m pytest tests/test_reference_setup.py -q`; expect import/contract failure.
- [ ] **Step 3: Implement** streaming SHA-256 ZIP verification before extraction, exact prefix `dxa_qc_code/combined_qc/`, all 69 original `combined_qc/**/*.py` files plus runtime manifest/config/asset files (source-only `__init__` import dependencies retained verbatim), `ZipInfo.external_attr` symlink check, size cap and per-file SHA. Preserve source bytes. Copy runtime assets to `backend/models/reference/{router,hip,spine}/checkpoints/runtime/` (exactly these roots passed via `QCPipeline(checkpoints_dir=models_root / "router" / "checkpoints", hip_checkpoints_dir=models_root / "hip" / "checkpoints", spine_checkpoints_dir=models_root / "spine" / "checkpoints")`). Do not extract `source_data`, training/checkpoint source, manifests with personal identifiers or outputs into Git. Pin compatible dependencies in both lock and pyproject; regenerate lock deterministically using Python 3.12 and verify an import-only cold load. Include explicit type hints and Russian docstrings.
- [ ] **Step 4: Verify** tests, `ruff check .` and `git diff --cached --name-only`; compare every added reference source SHA to its ZIP member; keep generated weights and image data ignored.
- [ ] **Step 5: Commit** only script, tests, lock/config and verified source with `feat(backend): install pinned reference inference locally` plus required co-author trailer. Do not push until branch payload audit.

### Task 2: One Persistent Pipeline with Parity Fixtures

**Files:** Create `backend/app/model_runtime.py`, `backend/tests/test_model_runtime.py`, `backend/tests/test_reference_parity.py`; modify `backend/app/main.py`.

**Interfaces:** `ReferenceRuntime(models_root: Path)` owns one `QCPipeline(..., device="cpu")`; `infer(content: bytes, suffix: str, region: AnatomicalRegion) -> dict` calls `QCPipeline.infer(path)` for auto; explicit region invokes original `combined_qc.{spine,hip}.pipeline.infer(path, model=loaded_handle, device="cpu")`, marking operator route. `_safe_result` retains only labels, scores, status, region, laterality, needs_review, geometry/image payload and per-check criteria needed by DTO (never `raw`, source path, patient metadata). `ready() -> bool` validates model inventory and tries a cold load; absent models fail health readiness, not module import. Use `TemporaryDirectory` per input and fixed `input.dcm`/`input.png` basename, never input filename. Always remove temp files and no logging of payload.

- [ ] **Step 1: Test** missing model root → explicit readiness error, and fake `QCPipeline` → called once for auto, temp path deleted, source payload not returned, operator path does not invoke classifier QC; check invalid model hash fails closed.
- [ ] **Step 2: Run** `cd backend && .venv/Scripts/python -m pytest tests/test_model_runtime.py -q`; expect failure.
- [ ] **Step 3: Implement** runtime with a single reusable model handle and original import package on PYTHONPATH without changing its files. Set `device="cpu"`; assert `CPUExecutionProvider` exists. Do not eagerly instantiate at import: app startup/health can report model unavailable without fatal import to preserve testability.
- [ ] **Step 4: Generate parity test** by reading three original DICOM cases **in memory from the private ZIP** (one spine, both hip sides), running the untouched `QCPipeline` directly and `ReferenceRuntime` on identical bytes, checking route, side, labels, OR exactly and scores via `np.testing.assert_allclose(..., atol=1e-5, rtol=1e-4)`. Do not commit fixtures/data/paths. Mark real-model test `pytest -m model` to skip with an explicit reason only when the local bundle is absent; run it locally with installed assets.
- [ ] **Step 5: Verify** runtime tests and cold-load once; commit source/tests only with `feat(backend): connect reference qc pipeline` plus co-author trailer.

### Task 3: Safe JSON Contract and Explicit Quality Checks

**Files:** Create `backend/app/checks/reference.py`, `backend/tests/test_analysis_contract.py`; modify `backend/app/analyzer.py`, `backend/app/schemas.py`, `backend/app/imaging.py`, `backend/tests/test_api.py` and `backend/app/api.py` for readiness + predictable errors. Keep old heuristic modules only if clearly labelled legacy/unreachable; no hidden fallback.

**Interfaces:** `AnalysisResult` adds `projection: Literal["unknown"]`, `projection_source: Literal["not_determined"]`; a nonempty DICOM `ViewPosition` other than `unknown` produces `Failure` (no supported AP/PA claim), `geometry: SpineGeometry | HipGeometry | None`, `annotated_data_url: str | None`, `processing_status: Literal["Success","Failure"]`, `quality_class: int | None`; old IDs replaced with exact five reference IDs. `Analyzer(runtime: ReferenceRuntime)` uses `runtime.infer(content, suffix, region)` once, maps known check IDs into `QualityCheck` descriptors and result objects and verifies the OR against original. A simple failure factory exposes sanitized cause and no traceback. A success validator forbids null class and undefined required labels. Extract Study/SOP UID with `pydicom.dcmread(..., stop_before_pixels=True, specific_tags=[...])`, validate UID format/length, and discard all other tags. `preview_data_url` comes from *native* reference image payload, not existing percentile normalization. Limit output geometry to native width/height, spine axis/candidates/gaps or hip brightness bbox. Do not echo paths or raw criteria provenance.

- [ ] **Step 1: Write failing contract tests:** synthetic reference output for spine, hip, uncertain route, absent axis; success class exactly 0/1, unknown decision Failure + null; `projection=unknown`; stable `hip_positioning_rotation` code; disabled editing for absent points; malicious original filename/UID/metadata does not reach error/JSON except sanitized basename. Test unparseable DICOM single endpoint returns 422, batch Failure.
- [ ] **Step 2: Run** `cd backend && .venv/Scripts/python -m pytest tests/test_analysis_contract.py tests/test_api.py -q`; expect failures.
- [ ] **Step 3: Implement** explicit five `QualityCheck` adapters consuming restricted inference result, `AnalysisResult` shape/validators and safe uid/preview conversion. Refuse unsupported explicit projection if DICOM provides one. Healthcheck distinguishes `{status:"ready"}` from unavailable model with 503 and safe detail. Keep labels exactly as reference and expose `needs_review` separately without making an otherwise complete binary decision null. Preserve a single `analyze` call signature for existing batch/CLI.
- [ ] **Step 4: Verify** contract tests, `ruff check .`, JSON has no temp path or DICOM private tags; commit `feat(api): expose reference qc results safely` plus co-author trailer.

### Task 4: Study-Aware Batch and CSV

**Files:** Modify `backend/app/batch.py`, `backend/app/api.py`, `backend/app/cli.py`, `backend/tests/test_api.py`; add `backend/tests/test_batch.py`.

**Interfaces:** `analyze_items(analyzer: Analyzer, items: list[tuple[str, bytes]], region: AnatomicalRegion) -> BatchResult` preserves original order and every input, computes UID via safe metadata-only parse, counts per UID, processes first three and emits separate Failure rows for later entries. `render_csv(batch: BatchResult) -> str` emits exactly the eight mandated columns with stable sorted `;` codes and blank class on Failure; `path_to_study` uses input-relative path or sanitized basename, never temp path. `read_archive(content, max_file_bytes)` keeps existing bomb limits and rejects unsafe entries/ambiguous duplicates; returns all DICOM in deterministic order including malformed ones. `time_of_processing` measured per image from upload/ZIP member read and DICOM decode to completed row, including inference and postprocessing; measure at API/CLI batch boundary and pass elapsed seconds into the per-file result rather than excluding archive read.

- [ ] **Step 1: Add tests** for two interleaved Study UIDs with four and three images respectively, one invalid DICOM with empty UID, duplicate and traversal ZIP paths, valid one/two/three uploads, fourth file rejected by browser/API without truncation, rows count and order; use synthetic DICOM from `tests/test_api.py` with a controlled UID. Assert CSV blank class on failure, eight exact headers, semicolon codes, relative `path_to_study`, per-file seconds.
- [ ] **Step 2: Run** `cd backend && .venv/Scripts/python -m pytest tests/test_batch.py tests/test_api.py -q`; expect failures.
- [ ] **Step 3: Implement** UID grouping without assuming absence means same study; cap human batch to three but archive emits Failure row for fourth under same UID; reject unsafe archive metadata and sanitize filename before error output. Continue other files after a per-file exception, never swallow an entire archive silently. Preserve success and failure counts.
- [ ] **Step 4: Verify** tests + CLI CSV generated privately from two local DICOM; commit `feat(backend): make study batches and csv deterministic` plus co-author trailer.

### Task 5: Visual Review, Manual Spine Axis and Partial Failures

**Files:** Create `frontend/src/components/StudyViewer.tsx`, `frontend/src/geometry.ts`, `frontend/src/geometry.test.ts` (run with `node --experimental-strip-types --test src/geometry.test.ts` on Node 22.23.2; use extension-bearing import supported by TypeScript with `allowImportingTsExtensions` for test-only module); modify `frontend/src/types.ts`, `frontend/src/pages/ResultPage.tsx`, `frontend/src/pages/AnalyzePage.tsx`, `frontend/src/api.ts`, `frontend/src/styles.css`.

**Interfaces:** `measureManualAxis(top: Point, bottom: Point, width: number, height: number): {angleDeg:number; violation:boolean}` mirrors reference `atan2(dx,dy)*180/π`, `>5` except `abs(angle-5)<=1e-10`; rejects non-finite, out-of-bounds and top below/equal bottom. `StudyViewer({result, manualAxis, onAxisChange})` renders native grayscale `<img>` plus same-size viewBox SVG; editable axis endpoints only if both exist. Pointer capture and keyboard arrow controls update coordinates in native units; other overlays never pretend to be anatomical masks.

- [ ] **Step 1: Write geometry test cases** for 0°, exactly 5°, >5°, invalid inverted/outside/NaN points, uneven aspect ratio mapping. Add contract mock for two images and one Failure so UI does not discard successes. Run `npm run build` and `node --experimental-strip-types --test src/geometry.test.ts`; expect failures before implementation.
- [ ] **Step 2: Implement** typed DTO; change upload selection from `.slice(0,3)` to explicit error for >3. `api.ts` returns `BatchResult` including failures so successful tabs remain visible. Show error rows in ResultPage and empty-state warning if all failed. Render original PNG behind SVG with `preserveAspectRatio="xMidYMid meet"`, matching displayed image dimensions; separate original/annotated view toggle and remove decorative `.image-viewer__axis`. For spine two colored handles; badges separate automatic and manual OR, reset, download manual CSV JSON-safe escaping. For hip draw brightness field/proxies with disclaimer; no handles. Never mutate the original result array or machine checks.
- [ ] **Step 3: Run** geometry tests and `npm run build`; inspect layout at 375px and desktop, drag endpoint on aspect-ratio image and verify it aligns after resize; check keyboard/focus and missing-points state. Commit `feat(frontend): add honest qc overlays and manual axis` plus co-author trailer.

### Task 6: Private Sample Catalog and Aggregate Metrics

**Files:** Create `scripts/prepare_samples.py`, `scripts/metrics.py`, `backend/tests/test_samples_metrics.py`, `docs/metrics.md`; update `.gitignore` for `local-test-images/`.

**Interfaces:** `prepare_samples(zip_path: Path, out_dir: Path) -> dict[str,int]` reads corrected spine truth and hip truth from archive; maps native source DICOM by verified ID/hash without leaking source paths; writes neutral identifiers under `normal/`, `spine_positioning/`, `spine_axis/`, `spine_artifacts/`, `hip_positioning_rotation/`, `hip_roi/`, `multiple/`, `unlabeled/`. Some cases intentionally belong to multiple groups. `aggregate_metrics(zip_path: Path, seed: int=20260928, repeats: int=2000) -> dict` reads private `per_image`, study-cluster bootstrap with replacement, computes F1/recall/specificity/balanced accuracy/defined AUC, macro-F1, valid replicate counts; exports aggregates only, not per-image data.

- [ ] **Step 1: Add tiny fixture tests**: multi-label case belongs to both type folders plus `multiple`; no raw patient filename in output; deterministic rerun; cluster bootstrap repeats a whole study, same seed reproduces CI; one-class replicate excluded from AUC with reported count. Run pytest; expect import failure.
- [ ] **Step 2: Implement** safe extraction by manifest `canonical_source_path` after validating inside ZIP and source SHA; avoid exporting metadata/person names in index or README. Avoid copying whole 2.67GB ZIP; cap output, neutral filenames, idempotent overwrite only of files created by the script after hash comparison. Write a local `local-test-images/README.md` with purpose, class overlap, privacy and development-data warning. For bootstrap, sample unique `study_id`s, include all images for sampled studies, compute percentile CI and count undefined replicates. Aggregate table includes positive counts, per-region/per-task and OR metrics; do not invent AUC for boolean OR or Dice/IoU.
- [ ] **Step 3: Run** tests, full sample generation and metrics privately; inspect category counts and `git status --short` to ensure no data tracked; compare headline F1 with archived values. Commit script/tests and aggregate-only `docs/metrics.md` with `feat(metrics): document study-level qc uncertainty` plus co-author trailer.

### Task 7: Offline Docker and Submission Bundle

**Files:** Modify `backend/Dockerfile`, `frontend/Dockerfile`, `backend/.dockerignore`, `frontend/.dockerignore`, `docker-compose.yml`, `scripts/run.sh`; create `docs/deployment.md`, `backend/tests/test_deployment.py`.

**Interfaces:** `scripts/run.sh` checks a local `models/reference` directory populated by `prepare_reference.py`, validates SHA, and passes its **absolute** path as `DEXQ_MODELS_DIR` to Docker Compose; mount `${DEXQ_MODELS_DIR}:/models/reference:ro`. The final submission is an offline archive containing source + separate `models/` with all required ONNX/NPZ + manifest + run script, assembled locally under ignored `submission/`, not Git. Pin Python, Node and nginx base images by digest, lock all Python/JS dependencies and record digests/provider in docs.

- [ ] **Step 1: Add tests** for missing model directory, hash mismatch, read-only mount declaration and positive preflight; make the script exit nonzero before `docker compose` for missing files. Run tests; expect fail.
- [ ] **Step 2: Implement** local preflight, Linux/Unix path handling, non-root backend and readiness healthcheck that loads models. Obtain real immutable base image digests via Docker inspection; fail rather than invent digests if network unavailable. Do not `COPY` models into image or download at runtime. Validate `docker compose config` and `docker compose up --build` offline with local cache. State actual minimum and recommended CPU/RAM/GPU/disk from benchmark, with explicitly measured configuration/provider and separate cold-start; update `docs/deployment.md` with submission-bundle layout and shell commands.
- [ ] **Step 3: Verify** tests, Docker health, container inference/CSV with local DICOM, no outgoing network and model mount read-only. Commit `chore(deploy): package offline model submission` plus co-author trailer.

### Task 8: End-to-End Acceptance, Documentation and Safe Push

**Files:** Modify `README.md`, `SPEC.md`, `docs/user-guide.md`, `docs/model.md`, `frontend/src/pages/AboutPage.tsx` to replace stale prototype claims; update contract tests with final outputs; add benchmark script `scripts/benchmark.py`.

**Interfaces:** `scripts/benchmark.py --source <private-dir> --models <local-dir>` records per-study warm wall-clock including decode/infer/postprocess, cold load separately and success fraction; writes aggregate JSON under ignored local directory. It must fail CI/local acceptance if any studied group of ≤3 exceeds 180 s on declared configuration; if current machine cannot meet target, report failure, not success.

- [ ] **Step 1: Add final contract assertions** for Success binary, unknown projection warning, model unavailable 503, all archive rows, patient metadata scrubbing, deterministic repeated inference. Run them to see failures before changing remaining wiring.
- [ ] **Step 2: Run model parity and benchmark** on local etalon cases; record real target CPU/RAM/GPU/disk, provider and exact wall-time boundary; run full `cd backend && .venv/Scripts/python -m pytest` (including local model marker), `ruff check .`, `cd frontend && npm run build`, Docker health + CSV CLI + browser smoke (desktop/375px, drag/reset, partial errors, manual export). Document every skipped check or unmet criterion explicitly.
- [ ] **Step 3: Reconcile docs** with shipped behavior: projection unknown limitation, combined hip flag, development-only metrics/CI, original preprocessing/threshold/hash, training procedure, API inputs/outputs, error handling, offline Linux/Unix launch, weights delivery, user steps and demo case narratives. Remove outdated claims about pending hip models or clinically validated outputs.
- [ ] **Step 4: Run `git diff --check`, `git status`, `git diff --cached --name-only` and a tracked-file audit for `.dcm/.png/.onnx/.npz/.pt/.pth/.env`, plus archive path leak scan. Commit `docs: describe offline dxq mvp limitations` with co-author trailer. Push reviewed small commits only if no new medical data/weights are included. Existing tracked archive history remains a separately disclosed issue; do not force-push or rewrite it.
- [ ] **Step 5: Request independent whole-branch review** and verify any finding with failing test before fix. Report measured functionality and explicit noncompliance (projection classifier and separate hip rotation) without calling the system clinically validated.
