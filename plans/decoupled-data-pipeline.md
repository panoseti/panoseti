# Decoupled Data Pipeline Architecture

## Context

Our observation-stop critical path currently blocks for hours on multi-TB `rsync` pulls from DAQ nodes to the head node. `stop.py::stop_run()` calls `collect.collect_data()` synchronously at `control/stop.py:510`, then `_cleanup_daq_grpc()` at `control/stop.py:536` — **both execute under the advisory lock held from `control/stop.py:432` to `:563`**. Consequences:

- Any concurrent `start.py` is blocked for the full rsync duration.
- The ledger sits in `STOPPING` with no operator-visible transfer progress.
- The `cancel_event` signal path (`stop.py:423`) only fires between phases; rsync itself is not interruptible.
- A single packet-loss flurry or head-node disk blip aborts the whole transfer with no resume contract.
- `CleanupData` (proto at `grpc/protos/daq_control.proto:68-78`) unconditionally `rmtree`s run dirs (`server.py:_cleanup_dir` at `grpc/src/panoseti_grpc/daq_control/server.py:143`), destroying JSON metadata/logs that are useful as a permanent on-DAQ catalog.
- There is **no integrity check** today — we trust rsync exit codes and nothing else.

Target outcome: `stop.py` returns in seconds. A dedicated Transfer Daemon owns all bulk I/O, driven by a durable worklist, with cryptographic manifests and selective cleanup that preserves metadata.

---

## Step 1 — Current Coupling: Evaluation

### Blocking touchpoints in `stop.py::stop_run`

| Line | Call | Blocking? | Notes |
|------|------|-----------|-------|
| `stop.py:432` | `state_mgr.acquire_lock()` | — | Held through `:563` |
| `stop.py:468` | `stop_recording()` (gRPC `StopDaq`) | seconds | Acceptable — must stay synchronous |
| `stop.py:471-493` | `kill_hv_updater`, `kill_hk_recorder`, `kill_module_temp_monitor`, `stop_data_flow` | seconds | Must stay synchronous |
| `stop.py:510` | **`collect.collect_data()`** | **hours** | ❌ Main offender; synchronous rsync |
| `stop.py:536` | **`_cleanup_daq_grpc()`** | minutes | ❌ Runs only after rsync, so inherits the delay |
| `stop.py:541` | `make_links()` | ms | Must run after collection, so inherits delay |
| `stop.py:542` | `write_complete_file(run_complete)` | ms | Semantically means "everything done" — must be deferred to daemon |

### Deprecation path for `collect.py`

- **Keep** `utils/file_xfer.copy_dir_from_node` — this is the low-level rsync primitive the daemon will reuse.
- **Delete** `utils/collect.cleanup_daq` — already superseded by `_cleanup_daq_grpc` (`control/stop.py:340`).
- **Retire** `utils/collect.collect_data` from the `stop_run` critical path. Keep the function body callable by the daemon as `_rsync_one_node` (renamed + moved to `utils/transfer/rsync_worker.py`).
- **Preserve** the `collect.py` CLI (`--run_dir X`) as a manual recovery tool; have it print a deprecation banner and enqueue a daemon job rather than execute inline.

---

## Step 2 — gRPC Co-Design

### 2a. Selective Cleanup

Extend `CleanupDataRequest` in `grpc/protos/daq_control.proto`:

```proto
enum CleanupMode {
    CLEANUP_FULL      = 0;   // legacy: rmtree the run dir (current behavior)
    CLEANUP_SELECTIVE = 1;   // delete only files matching delete_patterns
}

message CleanupDataRequest {
    string data_dir = 1;
    string run_dir = 2;
    repeated uint32 module_id = 3 [packed=true];
    bool   force = 4;
    CleanupMode mode = 5;                        // NEW; default CLEANUP_FULL (wire-compat)
    repeated string delete_patterns  = 6;        // e.g. ["*.pff"]
    repeated string preserve_patterns = 7;       // safety allowlist; takes precedence
}

message CleanupDataResponse {
    bool   success = 1;
    string message = 2;
    uint32 deleted_count = 3;    // NEW — observability
    uint64 freed_bytes   = 4;    // NEW
    repeated string preserved_paths = 5;  // NEW — audit trail for the catalog
}
```

Server rule (to implement in `DaqControlServicer._cleanup_dir_selective`):

> If `mode == CLEANUP_SELECTIVE`, walk the run dir and only `os.unlink` files matching any `delete_patterns` AND not matching any `preserve_patterns`. Empty directories are left in place. `*.pff` is never matched by a preserve pattern by default.

Proto default `CLEANUP_FULL` keeps wire-compatibility with any caller that has not been rebuilt.

### 2b. Manifest Generation

Add a new RPC:

```proto
service DaqControl {
    // ... existing RPCs ...
    rpc GenerateManifest (GenerateManifestRequest) returns (GenerateManifestResponse) {}
    rpc GetManifest      (GetManifestRequest)      returns (stream ManifestEntry)      {}
}

message GenerateManifestRequest {
    string data_dir = 1;
    string run_dir  = 2;
    repeated uint32 module_id = 3 [packed=true];
    string algorithm = 4;                 // "blake3" (default) | "xxh3_128"
    repeated string include_patterns = 5; // default ["*.pff"]
}

message GenerateManifestResponse {
    bool   success = 1;
    string message = 2;
    string manifest_path = 3;             // {data_dir}/module_{id}/{run}/manifest.{algo}
    uint32 file_count = 4;
    uint64 total_bytes = 5;
    double elapsed_seconds = 6;
    string algorithm = 7;
}

message GetManifestRequest {
    string data_dir = 1;
    string run_dir  = 2;
    uint32 module_id = 3;
}

message ManifestEntry {
    string relative_path = 1;
    string digest_hex    = 2;
    uint64 size_bytes    = 3;
    int64  mtime_ns      = 4;
}
```

Chosen algorithm: **blake3** (fast, keyed, native `pip install blake3`). `xxh3_128` retained as an option for nodes without blake3 wheels.

Manifest file format: newline-delimited `{digest}  {size}  {relpath}` (compatible with `b3sum --check`), written atomically via `tempfile + os.replace` (mirror `run_state.py:save_state`).

### 2c. `Pydantic` models — add to `grpc/src/panoseti_grpc/daq_control/config.py`

- `GenerateManifestModel` — enum-validated `algorithm`; non-empty `include_patterns`.
- Extend `CleanupDataModel` with `mode`, `delete_patterns`, `preserve_patterns`. Reject `CLEANUP_SELECTIVE` with empty `delete_patterns`.

---

## Step 3 — State Ledger Extensions

### New status vocabulary (in `RunStateLedger.status`)

Current set (from `stop.py` writes): `STARTING | ACTIVE | STOPPING | COMPLETED | STOPPED_WITH_ERRORS`.

Proposed superset — `stop.py` transitions to `RECORDING_ENDED` in **seconds**, the daemon owns the rest:

```
STARTING → ACTIVE → STOPPING → RECORDING_ENDED
                                  │
                                  ▼
                          MANIFEST_PENDING → MANIFEST_GENERATING → MANIFEST_READY
                                                                         │
                                                                         ▼
                                                                 TRANSFER_PENDING
                                                                         │
                                                                         ▼
                                                                  TRANSFERRING
                                                                         │
                              ┌──────────────────────────────────────────┤
                              ▼                                          ▼
                       TRANSFER_FAILED (retry)                      VERIFYING
                                                                         │
                                                    ┌────────────────────┤
                                                    ▼                    ▼
                                              VERIFY_FAILED         CLEANUP_PENDING
                                                                         │
                                                                         ▼
                                                                      CLEANING
                                                                         │
                                                                         ▼
                                                                     ARCHIVED
```

### New/changed fields on `RunStateLedger`

```python
class RunStateLedger(BaseModel):
    # existing fields...
    status: Literal[
        "STARTING","ACTIVE","STOPPING",
        "RECORDING_ENDED",
        "MANIFEST_PENDING","MANIFEST_GENERATING","MANIFEST_READY",
        "TRANSFER_PENDING","TRANSFERRING","TRANSFER_FAILED",
        "VERIFYING","VERIFY_FAILED",
        "CLEANUP_PENDING","CLEANING",
        "ARCHIVED","COMPLETED","STOPPED_WITH_ERRORS",
    ]
    transfer_attempts: int = 0
    last_transfer_error: str | None = None
    manifest_algorithm: str | None = None
    next_action_not_before: datetime | None = None   # backoff
```

### New per-node state on `NodeReceipt`

```python
class NodeReceipt(BaseModel):
    # existing fields...
    manifest_path: str | None = None
    manifest_bytes: int | None = None
    rsync_bytes_transferred: int | None = None
    rsync_last_progress_at: datetime | None = None
    verify_ok: bool | None = None
    cleanup_ok: bool | None = None
```

### Durable worklist (daemon-owned)

Separate from `run_state.toml` — a job queue directory:

```
tmp/transfer_queue/
  pending/     {run_name}.job.toml
  active/      {run_name}.job.toml
  failed/      {run_name}.job.toml
  completed/   {run_name}.job.toml
```

Each `*.job.toml` carries `{run_name, head_data_dir, daq_nodes[...], created_at, attempts}`. Atomic state transitions use `os.rename`. `stop.py` only writes into `pending/`; the daemon moves jobs through the lifecycle. This lets `stop.py` release the advisory lock immediately after creating the pending job.

### Lock separation

- `tmp/panoseti_control.lock` — stays as-is for `start.py`/`stop.py` mutual exclusion (sub-second holds only).
- `tmp/panoseti_transfer.lock` — new advisory lock held by the daemon while it owns a job. Multiple `stop.py` invocations never contend with the daemon.

---

## Step 4 — Execution Plan

Three TDD phases. Each phase: **(a) write failing tests first**, **(b) implement**, **(c) make tests green**, **(d) integration gate**. Do not advance to the next phase until the prior phase's full CI (`python ci/qa.py lint` + `unit` + `integration` + `python tests/qa.py all` in `grpc/`) is green.

### Phase 1 — Protobuf & gRPC Server

**Goal:** extend `daq_control` service with selective cleanup + manifest RPCs. No control-plane changes yet.

1. **Red: proto + generated-bindings tests**
   - `grpc/tests/daq_control/unit/test_proto_schema.py` — assert new fields/enum values exist in `daq_control_pb2`.
   - `grpc/tests/daq_control/unit/test_cleanup_model.py` — Pydantic rejects `SELECTIVE` + empty `delete_patterns`; accepts `FULL` wire-compat default.
   - `grpc/tests/daq_control/unit/test_manifest_model.py` — algorithm enum guard; default patterns.
2. **Impl:**
   - Edit `grpc/protos/daq_control.proto` per Step 2.
   - `python scripts/compile_protos.py`.
   - Extend `grpc/src/panoseti_grpc/daq_control/config.py` with `GenerateManifestModel`, extended `CleanupDataModel`.
   - New file: `grpc/src/panoseti_grpc/daq_control/manifest.py` with `async def compute_manifest(run_dir: Path, patterns, algo) -> ManifestResult`. Use `blake3` (fall back to `xxhash`) with `asyncio.to_thread` for blocking I/O.
   - Extend `DaqControlServicer`:
     - New `_cleanup_dir_selective(run_dir, delete_patterns, preserve_patterns)` method.
     - Route `CleanupData` to selective-vs-full based on `request.mode`.
     - New `GenerateManifest` handler → writes `manifest.{algo}` atomically, records in response.
     - New `GetManifest` streaming handler → `yield ManifestEntry` per line.
   - Extend client `grpc/src/panoseti_grpc/daq_control/client.py` with `GenerateManifest`, `GetManifest`, updated `CleanupData` kwargs.
3. **Green: integration tests** (`grpc/tests/daq_control/integration/`)
   - `test_cleanup_selective.py` — spin up servicer, populate fake run dir with `.pff` + `.json` + `.log`, call `CleanupData(mode=SELECTIVE, delete_patterns=["*.pff"])`, assert only pff gone; assert `preserved_paths` list echoes `.json`/`.log`.
   - `test_manifest_roundtrip.py` — generate manifest on fixture run; re-read via `GetManifest` stream; assert digest count + sizes match `hashlib` reference.
   - Backwards-compat: rerun existing `CleanupData` tests with no changes — `CLEANUP_FULL` default must behave identically.
4. **Integration gate:** `bash grpc/scripts/run-ci-tests/run-daq-control-test.sh` green; `control/` suite untouched and still green.

**Critical files:** `grpc/protos/daq_control.proto`, `grpc/src/panoseti_grpc/daq_control/{server.py,client.py,config.py,manifest.py}`, plus generated `daq_control_pb2{,_grpc}.py`.

### Phase 2 — State Ledger & `stop.py` Decoupling

**Goal:** `stop.py` completes in seconds. All collect/cleanup logic leaves the critical section and becomes a pending job. No daemon yet — jobs accumulate in `tmp/transfer_queue/pending/`.

1. **Red: unit tests for new ledger states** (`control/ci/unit/test_run_state_extended.py`)
   - Accepts the full status `Literal` set.
   - `NodeReceipt` round-trips new fields through `save_state`/`load_state` (exercise `_escape_toml_str` via manifest path with spaces).
   - Legacy `run_state.toml` files (only pre-existing statuses) still load.
2. **Red: unit tests for transfer-queue writer** (`control/ci/unit/test_transfer_queue.py`)
   - `TransferQueue.enqueue(run_name, ...)` creates exactly one `pending/*.job.toml` atomically.
   - Double-enqueue of the same run is idempotent (no duplicate file).
   - `TransferQueue.claim()` / `.complete()` / `.fail()` move files between dirs via `os.rename`.
3. **Red: integration test that `stop.py` is fast** (`control/ci/integration/test_stop_fast_path.py`)
   - Mock DAQ nodes that sleep 30 s in `StopDaq` are unacceptable; assert end-to-end `stop_run()` returns in < 5 s with the transfer queue populated.
   - Assert ledger status is `RECORDING_ENDED` (not `COMPLETED`).
   - Assert `run_complete` marker is **not** yet written (moved to daemon).
4. **Impl:**
   - New module `control/utils/transfer/queue.py` implementing `TransferQueue` (enqueue/claim/complete/fail + `list_pending`).
   - Extend `control/utils/run_state.py`:
     - Expand status `Literal` and `NodeReceipt` (and `save_state` formatter for new keys).
     - Add `RunStateManager.transition(status, **fields)` helper — single call site for all writes going forward.
   - Edit `control/stop.py`:
     - Replace `collect.collect_data` block (`:503-523`) and `_cleanup_daq_grpc` block (`:529-540`) with: `TransferQueue().enqueue(run_name=run, ...)` and `state_mgr.transition("RECORDING_ENDED")`.
     - Still write `recording_ended` marker (`stop.py:497`).
     - Remove `write_complete_file(run_complete_filename)` from `stop.py` — daemon owns it.
     - Keep `make_links()` **only after** `RECORDING_ENDED` succeeds; leaves broken links until daemon finishes, which is acceptable because `make_links` reads from the run dir that still exists on the head node (empty until transfer).
     - Adjust `--no_collect` / `--no_cleanup` to mean "enqueue with these flags" rather than "skip inline"; the daemon respects them.
     - Keep `--force-cleanup` to mean "enqueue bypassing manifest-verify gate".
   - Deprecate `control/utils/collect.py`:
     - Move `collect_data` body into `control/utils/transfer/rsync_worker.py::rsync_one_node`.
     - Replace `collect.py` top-of-file and `__main__` CLI with a deprecation banner + enqueue call.
     - Delete `collect.cleanup_daq`.
5. **Green:** Phase 2 tests pass; existing `control/ci/integration/` suite continues to pass, with the single change that `run_complete` assertions move to the daemon test suite (Phase 3). Any test that asserts on-disk data presence post-`stop.py` is converted to drain the queue synchronously via a test helper (`TransferQueue.run_once_inline()`).
6. **Integration gate:** `python ci/qa.py unit integration` green; `python ci/qa.py chaos -k "SC002 or SC010"` shows no regressions versus master baseline.

**Critical files:** `control/stop.py`, `control/utils/run_state.py`, `control/utils/transfer/queue.py` (new), `control/utils/transfer/rsync_worker.py` (new), `control/utils/collect.py` (shrunk).

### Phase 3 — The Transfer Daemon

**Goal:** a long-running async process drains the queue: manifest → rsync → verify → selective cleanup → archive.

1. **Red: unit tests** (`control/ci/unit/test_transfer_daemon.py`)
   - State-machine tests driving a daemon instance with a fake gRPC client + fake filesystem, asserting exact ledger transitions per scenario:
     - Happy path → `ARCHIVED`.
     - Manifest RPC fails → `MANIFEST_PENDING` retry with exponential backoff in `next_action_not_before`.
     - Rsync partial → `TRANSFER_FAILED` → retry resumes from rsync's own partial state (no re-manifest).
     - Head-node digest mismatch for one module → node flagged `verify_ok=False`, others proceed to cleanup, overall run → `STOPPED_WITH_ERRORS`.
     - `--force-cleanup` flag on job skips verify.
2. **Red: integration tests** (`control/ci/integration/test_transfer_daemon_e2e.py`)
   - Two-node topology (existing `daqnode` + `daqnode-2`); run a tiny synthetic job (few MB of fake `.pff` + `.json`); assert:
     - DAQ side after archive: `.pff` gone, `.json` + `hp_stdout.log` + `manifest.blake3` preserved.
     - Head side: all `.pff` copied; head-node manifest matches; ledger `ARCHIVED`; `run_complete` marker present.
   - Chaos test: kill the daemon mid-rsync; restart; assert resumes and completes.
3. **Impl:**
   - New file `control/utils/transfer/daemon.py` — `async def run_daemon(poll_interval=5.0)`:
     - Acquire `tmp/panoseti_transfer.lock` (`fcntl.LOCK_EX | LOCK_NB`); exit 0 if another daemon already running.
     - Loop: `TransferQueue.claim_next()` → `process_job(job)` → `.complete()` or `.fail()`.
     - Per-job state machine driving `RunStateManager.transition(...)`:
       1. `MANIFEST_GENERATING` — concurrent `GenerateManifest` RPCs (one per DAQ node via `asyncio.gather`).
       2. `TRANSFERRING` — `rsync_one_node` per node, with a throttled progress callback writing `rsync_bytes_transferred` + `rsync_last_progress_at` onto `NodeReceipt` every ~5 s.
       3. `VERIFYING` — stream `GetManifest` from each DAQ node; recompute head-side digest using the same algo; compare.
       4. `CLEANING` — `CleanupData(mode=SELECTIVE, delete_patterns=["*.pff"], preserve_patterns=["*.json","*.log","*.manifest.*"])` per node; record `deleted_count` + `freed_bytes`.
       5. `ARCHIVED` + write `run_complete` marker + `make_links()`.
     - Graceful SIGTERM: finish current step, mark job back to the pending status that precedes the current active status (e.g. `TRANSFERRING` → `TRANSFER_PENDING`), release lock.
   - New CLI entry: `python -m utils.transfer.daemon` and a daemon entry under `control/daemons/transfer_daemon.py` so `session_start.py` launches it via `util.start_daemon()` (matching existing patterns in `daemons.json`).
   - New `control/status.py` additions: pretty-print current daemon ledger status + per-node progress.
4. **Green:** Phase 3 tests pass; full suite (`unit`, `integration`, `chaos`) shows no net regressions; `python tests/qa.py all` in `grpc/` still green.
5. **Integration gate:** On a Palomar staging night or synthetic replay, run a full `start → stop → daemon drain` with ≥1 TB of fake data, confirm: `stop.py` completed in < 10 s; daemon finished within 2× of legacy rsync time; ledger trail matches the diagram; `.pff` gone from DAQ, catalog intact.

**Critical files (new):** `control/utils/transfer/{daemon.py,queue.py,rsync_worker.py,verify.py}`, `control/daemons/transfer_daemon.py`, daemon entry in `control/configs/*/daemons.json`.

---

## Verification Summary

| Gate | Command | Expected |
|------|---------|----------|
| Phase 1 | `bash grpc/scripts/run-ci-tests/run-daq-control-test.sh` | green incl. new selective/manifest tests |
| Phase 1 | `python tests/qa.py all` (in `grpc/`) | green |
| Phase 2 | `python ci/qa.py lint unit integration` (in `control/`) | green; `stop.py` wall-clock < 5 s in `test_stop_fast_path.py` |
| Phase 2 | `python ci/qa.py chaos -k "SC002 or SC010"` | no new failures vs. master |
| Phase 3 | `python ci/qa.py integration -k "transfer_daemon"` | green |
| Phase 3 | Staging night dry-run | `stop.py` < 10 s; daemon archives full run; `.pff` gone, metadata preserved |

---

## Risks & Mitigations

- **Ledger schema break** — `RunStateLedger.status` becomes a `Literal` superset. Pydantic will reject legacy values only if we remove them; we don't. Write a one-shot migration in `load_state` that maps unknown strings → `STOPPED_WITH_ERRORS`.
- **Orphaned daemon** — a daemon crash mid-`CLEANING` could leave partial deletion. Mitigation: `_cleanup_dir_selective` is idempotent (missing files are not errors); daemon re-runs cleanup on restart if status is `CLEANING`.
- **Clock for `next_action_not_before`** — always UTC via `datetime.now(UTC)`, consistent with `stop.py:_ut_human_timestamp`.
- **Disk pressure on DAQ during retention** — preserved `.json` + logs are small (MB/run); document a separate catalog-retention policy, not this project's scope.
- **Wire compatibility** — all new proto fields use new tag numbers; `CleanupMode` default is the legacy behavior. Old clients keep working until rebuilt.
