# Transfer Queue & State Management — Architectural Blueprint

## Context

The HW-SW pipeline is now stable end-to-end for Start/Stop transactions, but the post-recording transfer path is unreliable:

1. **Port-forwarding is dropped at the queue boundary.** `stop.py::StopTransaction.__aexit__` enqueues a job with only `{ip_addr, data_dir, module_ids}` — `node.port_forwarding` is silently discarded. The downstream `rsync_worker.rsync_one_node` already supports gateways (`-e "ssh -p ..."` + `gw_ip`), but it never receives the data. Rsync over the physical router fails.
2. **The transfer daemon is orphaned.** Its docstring claims launch by `session_start.py`, but no `start_daemon('transfer_daemon.py')` callsite exists. `pseti stop` blindly enqueues into a queue with no consumer.
3. **`tmp/` is a junk drawer.** Locks, run state, manifests, transfer queue, topology PNGs, run-name files, ph baselines, and stray test artifacts (`sc039_run_name.txt`) all live side-by-side. There is no separation between *machine state* (recoverable, owned by transactions) and *operator artifacts* (snapshots, debug dumps).
4. **`CleanupData` is invoked without `manifest_digest`.** The integrity precondition the proto/server enforces is therefore not exercised — a regression against the documented invariant in `TRANSACTIONS.md`.
5. **`pseti stop` has no transfer-related UX.** No `--no-transfer`/`--skip-verify`, no warning if the daemon isn't running, no follow-up command to inspect or retry.
6. **No JobModel.** Queue jobs are hand-serialized dicts. There is no Pydantic schema, no version field, no migration story.

This blueprint addresses all six. **No code is written yet** — this is the design contract for the next session.

---

## Phase 1 — State Management Refactor

### 1.1 New top-level `state/` directory

Replace `tmp/` (overloaded) with a typed, role-segregated tree under a single override root. Default: `control/state/` (override `PSETI_STATE`):

```
control/state/
├── locks/                    # Advisory locks (control, transfer, calibration)
│   ├── control.lock
│   └── transfer.lock
├── runs/
│   ├── current               # Plain text: name of the current run (replaces tmp/current_run)
│   └── ledger.toml           # RunStateLedger (replaces tmp/run_state.toml)
├── transfer/
│   ├── queue/
│   │   ├── pending/          # {run_name}.job.toml
│   │   ├── active/
│   │   ├── completed/
│   │   └── failed/
│   ├── manifests/            # Cached manifest copies pulled from DAQ nodes (head-side)
│   │   └── {run_name}/{module_id}/manifest.{blake3,xxh3_128,sha256}
│   └── daemon.pid            # Single-instance pid file (companion to flock)
├── calibration/              # Outputs of pseti obs config calibrate-*
│   ├── quabo_ph_baseline.json
│   └── quabo_uids.json
├── snapshots/                # Per-run artifact snapshots (read-only after creation)
│   └── {run_name}/
│       ├── sw_info.json
│       └── topology.{json,html,png}
└── logs/                     # Daemon stdout/stderr (separate from control/logs/ structured logs)
    └── transfer_daemon/
        ├── current.log
        └── archive/{ISO8601}.log
```

**Path resolution layer.** Add `PanoPaths.state_dir()` and seven typed accessors:

| Accessor | Path |
|---|---|
| `PanoPaths.state_dir()` | `control/state/` |
| `PanoPaths.locks_dir()` | `state/locks/` |
| `PanoPaths.runs_dir()` | `state/runs/` |
| `PanoPaths.transfer_queue_dir()` | `state/transfer/queue/` |
| `PanoPaths.transfer_manifests_dir()` | `state/transfer/manifests/` |
| `PanoPaths.calibration_dir()` | `state/calibration/` |
| `PanoPaths.snapshots_dir(run_name)` | `state/snapshots/{run_name}/` |
| `PanoPaths.daemon_logs_dir(name)` | `state/logs/{name}/` |

Each accessor honors a corresponding env var (`PSETI_LOCKS_DIR`, `PSETI_TRANSFER_QUEUE_DIR`, etc.) — preserves the existing override pattern.

**Migration.** A one-shot `pseti state migrate` command (idempotent) that:
- creates the new tree (`PanoPaths.ensure_state_dirs()`),
- moves locks, `current_run`, `run_state.toml`, `quabo_*.json`, `transfer_queue/` into their new homes,
- leaves a `tmp/MIGRATED` marker so legacy callsites can fail loud.

### 1.2 Calibration output standardization

**Root cause confirmed.** The bug is read-side, not write-side:

- `config.py::do_calibrate_ph` (line 692) **correctly** writes to `PanoPaths.tmp_dir() / quabo_ph_baseline_filename`.
- `start.py::ph_baseline_file_ok` (line 248) calls `os.path.exists(filename)` where `filename` defaults to the bare string `'quabo_ph_baseline.json'` — **no directory prefix**. It probes CWD, not `tmp/`. So `pseti obs start --no_hv --no_redis` reports "not found" even though the file exists in `tmp/`.

The state refactor eliminates this entire class of bug. After migration:

- All calibration I/O routes through one typed accessor: `PanoPaths.calibration_file("quabo_ph_baseline.json")` returning a fully-qualified `Path`. There is no API that accepts a bare filename.
- `ph_baseline_file_ok` becomes `ph_baseline_file_ok(path: Path = PanoPaths.calibration_file("quabo_ph_baseline.json"))`. Callers cannot pass a string.
- `configs/` becomes **read-only at runtime** — anything written there is a bug. CI guard test (`test_no_runtime_writes_to_configs`) greps for `Path("configs"...)` write-mode opens in non-test code.
- `config_file.get_quabo_ph_baselines()` reads from `state/calibration/` first, falling back to `configs/` for legacy site bundles (committed checked-in baselines still work).

This generalizes: **every** mutable runtime artifact has exactly one place to live, determined by its lifecycle (lock, ledger, queue, calibration, snapshot, log). String-based filename plumbing is removed throughout.

---

## Phase 2 — Transfer Architecture & CLI UX

### 2.1 New `src/control/transfer/` package

Move and split the current `utils/transfer/` (which mixes daemon, queue, rsync, verify under `utils/`) into a first-class package. The daemon process itself moves here too (decision: package-owned daemon, not `control/daemons/`):

```
src/control/transfer/
├── __init__.py
├── __main__.py        # Allows `python -m control.transfer` to launch the daemon
├── models.py          # Pydantic: TransferJob, TransferNodeSpec, TransferStatus
├── queue.py           # TransferQueue (atomic rename, idempotent enqueue)
├── daemon.py          # run_daemon(), main loop, signal handling, pid file, heartbeat
├── rsync.py           # build_rsync_cmd(node: TransferNodeSpec) -> list[str]
├── verify.py          # verify_manifest() (moved verbatim)
├── manifest.py        # Head-side manifest IO + sha256 digest helpers
├── lifecycle.py       # State machine: stages, transitions, retry policy
├── progress.py        # Progress reporter (parses rsync --info=progress2 stderr)
├── service.py         # Public API used by stop.py and CLI: enqueue(), status(), retry()
└── cli.py             # Typer app for `pseti obs transfer`
```

**Hard cut migration** (no shim layer): all imports update atomically in one PR. `control/utils/transfer/` is deleted. `control/daemons/transfer_daemon.py` is deleted — `start_daemon` is updated to accept any module-resolvable target via `python -m control.transfer`, or to invoke a small launcher in `daemons/` that just imports and calls `control.transfer.daemon.run_daemon`. Decision: extend `start_daemon(prog)` to accept a `["python", "-m", "control.transfer"]`-style command, eliminating the `daemons_dir()` dependency for the transfer process specifically. Other daemons keep their existing layout.

### 2.2 `TransferJob` Pydantic model (the missing schema)

```python
class TransferNodeSpec(BaseStrictModel):
    ip_addr: IPvAnyAddress
    username: str
    data_dir: str
    module_ids: list[int]
    port_forwarding: PortForwarding | None = None    # ← THE FIX

class TransferJob(BaseStrictModel):
    schema_version: Literal[1] = 1
    run_name: str
    head_data_dir: str
    head_node_username: str
    created_at: datetime
    attempts: int = 0
    no_cleanup: bool = False
    no_collect: bool = False
    skip_verify: bool = False
    daq_nodes: list[TransferNodeSpec]
```

`stop.py` constructs a `TransferJob` from the validated `DaqConfig` (port_forwarding already attached by `attach_daq_config`) and serializes via `model_dump_toml()`. The daemon parses with `TransferJob.model_validate(toml.load(f))`. **This single change resolves the port-forwarding bug** — it propagates as a typed field instead of a dict that loses keys.

### 2.3 Port-forwarding-aware rsync (architecture)

`rsync.build_rsync_cmd(node: TransferNodeSpec, run_name: str, head_run_dir: str) -> list[str]` is the single chokepoint:

```
pf = node.port_forwarding
use_pf = pf is not None and pf.status

cmd = ["rsync", "-aP", "--info=progress2", "--partial-dir=.rsync-partial"]
if use_pf:
    ssh_opts = ["-p", str(pf.port), *util.ssh_options]
    cmd += ["-e", f"ssh {' '.join(ssh_opts)}"]
    host = f"{node.username}@{pf.gw_ip}"
else:
    cmd += ["-e", f"ssh {' '.join(util.ssh_options)}"]
    host = f"{node.username}@{node.ip_addr}"

cmd += [
    f"{host}:{node.data_dir}/{run_name}/{hp_stdout_prefix}*",
    f"{host}:{node.data_dir}/{run_name}/{pss_prefix}*",
    *[f"{host}:{node.data_dir}/module_{m}/{run_name}/" for m in node.module_ids],
    head_run_dir,
]
return cmd
```

**Edge cases handled:**

- **PF status flips mid-job (router reboot).** Each retry attempt re-resolves PF from the in-memory `TransferJob` (immutable for the job) but logs a warning if the live `network_config.json` now differs — operator decides whether to abort.
- **PF port collision.** `network_config.json` validation already catches duplicate `port` fields at startup; we add a runtime sanity check in `build_rsync_cmd` that `pf.port` is in `[1024, 65535]`.
- **Bare IP fallback when `pf.status=False`.** Already covered. We additionally treat `pf` whose `gw_ip` resolves to the same subnet as `head_node_ip_addr` as PF=False (LAN shortcut) — saves an SSH hop in lab setups.
- **`--partial-dir`** preserves byte progress across rsync retries (5s/30s/exhausted) without reverifying the whole tree.
- **`rsync --rsync-path` for non-standard remote rsync.** Reserved as a future hook; not needed for the v1 lab fleet.
- **Bandwidth limit.** Optional `--bwlimit` from `daq_config.transfer.bwlimit_kbps` — disabled by default.
- **Stalled rsync detection.** If `rsync_last_progress_at` (already on `NodeReceipt`) goes >10 min without updates, the worker SIGKILLs rsync and counts the attempt as a transient failure.
- **Disk full on head node.** `shutil.disk_usage(head_data_dir)` checked pre-flight against the manifest's `total_bytes`. Insufficient space → `TRANSFER_FAILED` with `last_transfer_error="ENOSPC"`, no retry, ledger flips to `STOPPED_WITH_ERRORS`.

### 2.4 Daemon lifecycle (it must actually start)

- `session_start.py` calls `start_daemon(PanoPaths.daemons_dir() / "transfer_daemon.py")` after redis daemons. Pid lands in `state/transfer/daemon.pid`.
- `session_stop.py` calls a graceful shutdown: SIGTERM → wait for in-flight job to reach a stable stage → exit. If the daemon is mid-rsync and SIGTERM arrives, it lets the current attempt finish, then exits with the job back in `pending/`.
- Health check: the daemon writes a heartbeat to `state/transfer/daemon.heartbeat` every 5 s. CLI uses heartbeat staleness (>30 s) to declare the daemon dead.

### 2.5 gRPC upgrades to `daq_control.proto`

Three new RPCs, one extension:

| RPC | Purpose | Why |
|---|---|---|
| `GetTransferStatus(GetTransferStatusRequest) → GetTransferStatusResponse` | Per-DAQ-node: hashpipe state, run dirs present, free disk, manifest existence. | Consumed by `pseti obs transfer status`. Replaces ad-hoc SSH probes. |
| `GetManifestDigest(GetManifestDigestRequest) → GetManifestDigestResponse` | Returns SHA-256 of the manifest file content for `(data_dir, run_dir, module_id)`. | Required so the head node can pass `manifest_digest` to `CleanupData(CLEANUP_SELECTIVE)` (closes the documented invariant). |
| `RetryFailedTransfer(RetryFailedTransferRequest) → RetryFailedTransferResponse` | DAQ-side: re-emit a missing file by absolute path, returns size+digest. | Lets the head node patch a single corrupt file without re-rsyncing the whole tree. |

Extension to `CleanupDataRequest`: add `bool dry_run = 11;` returning the `preserved_paths[]` and `deleted_count` it *would* produce. Used by `pseti obs transfer plan-cleanup`.

`StatusDaq` is left alone — `GetTransferStatus` is the new aggregate, `StatusDaq` remains the legacy single-purpose probe.

### 2.6 `pseti obs transfer` CLI sub-app

Register in `tools/obs_cli.py` `lazy_mapping`:

```python
"transfer": ("control.transfer.cli", "app", "Inspect and manage the transfer queue.")
```

Subcommands:

| Command | Behavior |
|---|---|
| `pseti obs transfer status [run]` | Daemon health (heartbeat age, current job, pid). If `[run]` given: per-run progress — bytes transferred / total, manifest digest match, per-module rsync state. Reads queue + heartbeat + ledger. Pretty-printed via `rich.Table`. |
| `pseti obs transfer tail [-f] [-n N]` | Tails `state/logs/transfer_daemon/current.log`. `-f` follows. |
| `pseti obs transfer start` | Idempotent: refuses if heartbeat fresh; otherwise `start_daemon`. |
| `pseti obs transfer stop` | SIGTERM, wait up to 60s for graceful exit, then SIGKILL. |
| `pseti obs transfer restart` | stop + start. |
| `pseti obs transfer queue [pending\|active\|completed\|failed]` | Lists jobs in the named bucket. Default: all four, summary counts only. |
| `pseti obs transfer retry <run_name>` | Moves `failed/{run}.job.toml` → `pending/` after resetting `attempts=0`. |
| `pseti obs transfer cancel <run_name>` | Moves `pending/{run}.job.toml` → `failed/` with `last_transfer_error="CANCELLED_BY_OPERATOR"`. Refuses if job is `active/`. |
| `pseti obs transfer run <run_name> [--no-cleanup] [--skip-verify]` | Manual one-shot: builds a `TransferJob` from current configs and runs the state machine inline (no daemon). For ad-hoc rescue. |
| `pseti obs transfer manifest <run_name> [--module M]` | Pretty-prints the head-side manifest, flags any digest mismatches against re-hashed local files. |
| `pseti obs transfer verify <run_name>` | Runs `verify_manifest()` standalone, no state transitions. |

### 2.7 `pseti stop` UX changes

New flags (replacing the current `--no_cleanup`/`--no_collect`/`--force-cleanup`):

| Flag | Behavior |
|---|---|
| `--no-transfer` | Skip enqueueing entirely. Ledger stays at `STOPPED_WITH_ERRORS` with reason `OPERATOR_SKIPPED_TRANSFER`. Forces `--keep-daq-data`. |
| `--keep-daq-data` | Sets `no_cleanup=True` on the job (DAQ `.pff` files retained after archive). |
| `--skip-verify` | Sets `skip_verify=True` (still rsyncs, still cleans, but no manifest re-hash). Discouraged; CLI prints a warning. |
| `--force-cleanup` | Existing semantics (cleanup even on partial verify failure). |
| `--yes/-y` | Auto-confirm safety prompts. |

**Daemon-down warning.** Before enqueue, `stop.py` checks heartbeat freshness:

- **Daemon healthy** → enqueue silently, transition to `RECORDING_ENDED`.
- **Daemon stale** (`>30s` since heartbeat) and stdin is a TTY → interactive prompt:
  > Transfer daemon appears down. Job will be queued but no transfer will occur until you run `pseti obs transfer start`. Continue? [y/N]
- **Daemon stale** and `--yes` → enqueue, but emit a `WARNING` log + write a sentinel file `state/transfer/queue/pending/{run}.WARN_DAEMON_DOWN` that `pseti obs transfer status` surfaces in red.
- **Daemon stale** and not a TTY (CI, headless) → enqueue, emit warning, exit 0 (so cron-driven stops don't fail).

**Where disk-fill prevention actually belongs: `pseti start` pre-flight, not `pseti stop`.**

`pseti stop` cannot meaningfully gate on DAQ disk space — by then the recording has already happened. The risk the user flagged (suppressed pending jobs causing DAQ disks to silently fill across runs) belongs upstream:

- Extend `utils/global_validator.py` with a new check `validate_daq_disk_headroom(daq_config, data_config)`:
  - Compute estimated bytes/sec/module via existing `util.daq_bytes_per_sec_per_module(data_config)`.
  - Multiply by configured run duration × number of modules per node → projected bytes per node.
  - Probe each DAQ node via `GetTransferStatus.disk_usage` (new RPC, see §2.5) for free bytes.
  - Sum projected bytes for *all pending and active runs* in the transfer queue (jobs not yet ARCHIVED still own DAQ data) plus the new run's projection.
  - If projected total > free × `0.9` (10% safety margin) → fail validation; operator must run `pseti obs transfer status` and clear stuck jobs before retrying.
- This makes `pseti start` the chokepoint for storage accountability. `pseti stop` only emits the warning; `pseti start` enforces.
- Override flag: `pseti start --skip-disk-check` for emergencies (logs WARNING with daemon-down sentinel parallel).

### 2.8 `TRANSACTIONS.md` updates

- Replace all `tmp/...` paths with `state/...`.
- Document the `TransferJob` schema as the contract between `pseti stop` and the daemon.
- Add a "Daemon Lifecycle" section: how `session_start`/`session_stop` own the daemon process, heartbeat semantics, what happens on `pseti stop` with a dead daemon.
- Document the new RPCs and the `manifest_digest` precondition (currently violated by the daemon).
- Add an "Operator Recovery" section listing the `pseti obs transfer` workflows for stuck runs.

---

## Phase 3 — Testing Strategy

### 3.1 Software simulation: mock fleet (`ci/integration/transfer/`)

Spin up a Docker fleet using the existing `daqnode_net` / `headnode_net` topology. Three test modules:

#### `test_transfer_basic.py` — Standard transfer (no PF)

1. Fixture: 1 head + 2 daqnodes on shared subnet, no port_forwarding.
2. Use `panoseti_grpc.daq_data.simulate` to trickle synthetic `.pff` files into each `module_*/{run}/` on the daqnodes (no real hashpipe).
3. Invoke `pseti stop` against a synthetic active run.
4. Assert: ledger reaches `ARCHIVED`, manifest digests match on head, `.pff` removed from daqnodes, `.json/.log/.toml` preserved.
5. Sub-cases: 1 node, 2 nodes, 4 nodes; module count 1, 4, 16.

#### `test_transfer_port_forwarding.py` — PF over socat

1. Fixture: head on `headnode_net`, daqnodes on `daqnode_net`, gateway container running socat to forward port 22→2200/2201 per daqnode.
2. `network_config.json` injected with `port_forwarding.status=True, gw_ip=10.0.1.254, port=2200/2201`.
3. Same trickle as above. Assert rsync command (captured via `rsync` shim) contains `-e "ssh -p 2200"` and connects to `10.0.1.254`.
4. **Critical regression test:** the job TOML on disk must contain a `port_forwarding` block. Reading the TOML and re-validating with `TransferJob.model_validate` must succeed and round-trip the PF.

#### `test_transfer_chaos.py` — Failure injection

| Scenario | Injection | Expected |
|---|---|---|
| Node drops mid-rsync | `iptables -A OUTPUT -d <node> -j DROP` after first 100KB | Backoff 5s/30s, ledger transitions to `TRANSFER_FAILED` after 3 attempts, no cleanup. |
| Manifest corruption | flip 1 byte in head-side manifest before VERIFYING | `VERIFY_FAILED`, cleanup skipped, DAQ data preserved. |
| Disk full on head | `dd if=/dev/zero` to fill `head_data_dir` to <100MB free | Pre-flight rejects job, ledger `STOPPED_WITH_ERRORS`, daqnode untouched. |
| Daemon killed mid-job | `kill -9 <daemon_pid>` during TRANSFERRING | Restart daemon → `active/` job swept back to `pending/` → transfer resumes via `--partial-dir` (no re-transfer of completed bytes). |
| Stale rsync | `pkill -STOP rsync` on daemon side | After 10 min progress timeout, daemon SIGKILLs rsync, retries. |
| Concurrent stops | Two `pseti stop` invocations against same run | Second stop hits `panoseti_control.lock` → fails fast. Idempotent enqueue prevents duplicate jobs. |
| `manifest_digest` mismatch | Daemon hand-edits a byte of head manifest after compute, before CleanupData | DAQ server returns `FAILED_PRECONDITION`, ledger transitions to `VERIFY_FAILED`, files preserved. |
| Daemon crashes between cleanup-success and ARCHIVED marker | SIGKILL between `CleanupData` ack and `run_complete` write | On restart, ledger shows `CLEANING`, `run_complete` absent → daemon re-runs `CleanupData(dry_run=True)` to confirm idempotency, then writes marker. |

All fixtures parameterize **only** on `daq_config.json`, `obs_config.json`, `network_config.json` — no IPs in test code (guardrail).

### 3.2 HW-SW happy path: `ci/hardware-software/test_07_transfer_pipeline.py`

1 head node + 1 DAQ node (real hardware, real Quabos). Zero hardcoded IPs.

```python
def test_07_transfer_pipeline(panoseti_env):
    daq_config = config_file.get_daq_config()
    obs_config = config_file.get_obs_config()
    network_config = config_file.get_network_config()
    util.attach_daq_config(daq_config, network_config)

    # Phase 1: short observing run
    run_name = pseti("start", "--run-type=hwsw_transfer", "--duration=30s")
    pseti("status", expect_status="ACTIVE")
    wait_until(lambda: ledger_status(run_name) == "ACTIVE", timeout=60)

    # Phase 2: stop & enqueue
    pseti("stop", "--yes")
    assert ledger_status(run_name) == "RECORDING_ENDED"
    assert (PanoPaths.transfer_queue_dir() / "pending" / f"{run_name}.job.toml").exists()
    job = TransferJob.model_validate(toml.load(...))
    # Regression assertion — port_forwarding must round-trip
    for node, expected in zip(job.daq_nodes, daq_config.daq_nodes):
        assert (node.port_forwarding is None) == (expected.port_forwarding is None)
        if node.port_forwarding:
            assert node.port_forwarding.gw_ip == expected.port_forwarding.gw_ip
            assert node.port_forwarding.port == expected.port_forwarding.port

    # Phase 3: drive transfer to completion
    wait_until(lambda: ledger_status(run_name) == "ARCHIVED", timeout=600)

    # Phase 4: integrity proofs
    head_run = Path(daq_config.head_node_data_dir) / run_name
    assert (head_run / "run_complete").exists()
    for algo in ("blake3", "xxh3_128", "sha256"):
        if (head_run / f"manifest.{algo}").exists():
            ok, errs = verify_manifest(head_run / f"manifest.{algo}", head_run)
            assert ok, errs

    # Phase 5: selective cleanup proof
    for daq_node in daq_config.daq_nodes:
        remote_pff = ssh_ls(daq_node, f"{daq_node.data_dir}/module_*/{run_name}/*.pff")
        remote_meta = ssh_ls(daq_node, f"{daq_node.data_dir}/module_*/{run_name}/*.json")
        assert remote_pff == []          # .pff deleted
        assert remote_meta != []         # .json/.log/.toml preserved
```

`ssh_ls` uses `util.attach_daq_config`-resolved `port_forwarding` — same code path as production, so this test is a true end-to-end proof of the PF fix.

---

## Files To Modify

| Path | Role |
|---|---|
| `control/src/control/utils/paths.py` | Add `state_dir()` + 7 typed accessors, env overrides. |
| `control/src/control/utils/pydantic_config_models.py` | Add `TransferJob`, `TransferNodeSpec`, `TransferStatus`. |
| `control/src/control/transfer/` (new package) | Move + split `utils/transfer/*`; new `cli.py`, `service.py`, `models.py`, `progress.py`, `lifecycle.py`. |
| `control/src/control/utils/transfer/` | Re-export shims for one release, then delete. |
| `control/src/control/stop.py` | Build `TransferJob` (with PF), new flags, daemon-down warning. |
| `control/src/control/start.py` & `session_start.py` | Launch transfer daemon via `start_daemon()`. |
| `control/src/control/tools/obs_cli.py` | Register `transfer` sub-app. |
| `control/src/control/config.py::do_calibrate_ph` | Route writes through `PanoPaths.calibration_dir()` (verify with user first). |
| `grpc/protos/daq_control.proto` | Add `GetTransferStatus`, `GetManifestDigest`, `RetryFailedTransfer`; extend `CleanupDataRequest.dry_run`. |
| `grpc/src/panoseti_grpc/daq_control/server.py` | Implement new RPCs. |
| `grpc/src/panoseti_grpc/daq_control/client.py` | Wrap new RPCs in `AsyncDaqControlClient`. |
| `control/TRANSACTIONS.md` | Update paths, document daemon lifecycle, document new RPCs and CLI. |
| `control/ci/integration/transfer/test_transfer_basic.py` (new) | Standard transfer suite. |
| `control/ci/integration/transfer/test_transfer_port_forwarding.py` (new) | PF over socat. |
| `control/ci/integration/transfer/test_transfer_chaos.py` (new) | Failure injection matrix. |
| `control/ci/hardware-software/test_07_transfer_pipeline.py` (new) | HW-SW E2E happy path. |

---

## Verification

1. **State migration:** `pseti state migrate` → `state/` populated, `tmp/MIGRATED` present, all paths accessible via `PanoPaths`.
2. **PF round-trip:** unit test loads a `daq_config.json` with `port_forwarding.status=True`, builds a `TransferJob`, serializes to TOML, reloads, validates — `port_forwarding` round-trips exactly.
3. **Daemon launch:** `pseti session-start` → `state/transfer/daemon.pid` exists, heartbeat fresh within 5s.
4. **Daemon-down warning:** `kill $(cat state/transfer/daemon.pid); pseti stop` → interactive prompt fires.
5. **CLI smoke:** `pseti obs transfer --help`, `pseti obs transfer status`, `pseti obs transfer queue`, `pseti obs transfer tail -n 20`.
6. **Integration:** `pseti test sw integration -k transfer` → 100% pass.
7. **HW-SW:** `pseti test hw run -k HW_07` → ARCHIVED with manifest match and PF round-trip assertion green.
8. **Lint/types:** `pseti test lint` clean.

## Resolved Decisions

1. **Calibrate-ph "not found" error** is a read-side defect at `start.py:248` — `os.path.exists(bare_filename)` probes CWD. The state-dir refactor mandates typed-Path-only APIs (`PanoPaths.calibration_file(...)`), eliminating the bug class entirely.
2. **Migration strategy:** hard cut in a single PR. `utils/transfer/` deleted, all imports updated atomically. No deprecation shim.
3. **Daemon location:** `control/transfer/daemon.py` (package-owned). `start_daemon()` extended to accept `python -m control.transfer`-style invocation. `control/daemons/transfer_daemon.py` deleted.
4. **Headless `pseti stop`:** enqueue + warn + exit 0 (matches user direction). Disk-fill safety lives in `pseti start` pre-flight via an extended `global_validator.py` that aggregates projected data rates against live DAQ free space and pending/active queue obligations.
