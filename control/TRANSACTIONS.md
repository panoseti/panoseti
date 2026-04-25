# Observing Run Transactions

This document describes the transactional integrity and rollback mechanisms implemented in the PANOSETI observatory control plane (`pseti start` and `pseti stop`) using the **Context Manager Architecture**, and the decoupled Transfer Daemon that owns all bulk I/O.

## Overview

The observatory control plane manages a distributed system (Head node, DAQ nodes, Quabo detectors). Starting or stopping an observation is handled atomically by `StartTransaction` and `StopTransaction` context managers — implemented in `control/src/control/start.py` (class `StartTransaction`, line ~77) and `control/src/control/stop.py` (class `StopTransaction`, line ~62) respectively. Since the `pseti stop` refactor, bulk data transfer (rsync, manifest generation, selective cleanup) is decoupled from the advisory lock and executed by the Transfer Daemon (`control/src/control/transfer/daemon.py`).

## State Management & Locking

### Advisory Lock Hierarchy

Two separate advisory locks prevent concurrent operations at different granularities:

| Lock file | Mechanism | Held by | Duration |
|---|---|---|---|
| `state/locks/panoseti_control.lock` | `os.O_EXCL` + stale-PID healing | `pseti start` / `pseti stop` | Seconds (hardware I/O only) |
| `state/locks/panoseti_transfer.lock` | `fcntl.LOCK_EX \| LOCK_NB` | Transfer Daemon | Full job duration (minutes to hours) |

The control lock uses atomic file creation (`O_CREAT | O_EXCL`). If acquisition fails, the PID inside the file is checked — a dead PID causes a self-healing delete and retry (SC-015/SC-021). The transfer lock uses `flock`, which the kernel releases automatically on process exit.

## High-Performance Orchestration

Starting with Phase 4 of the architectural modernization, the control plane uses native asynchronous gRPC for all distributed operations:

- **Async-Native gRPC**: Utilizes `AsyncDaqControlClient` (built on `grpc.aio`) for non-blocking coordination of the DAQ fleet.
- **Strict Parameter Validation**: All gRPC requests are validated client-side via dedicated Pydantic `client_models.py` before hitting the network.
- **Concurrent Execution**: Multi-node operations (Start/Stop/Status) are executed in parallel using `asyncio.TaskGroup`, ensuring the head node can scale to large observatory topologies without thread-pool bottlenecks.
- **Authoritative Resolution**: `util.get_quabo_ip_port()` is the single source of truth for resolving effective Quabo addresses, correctly handling Gateway port forwarding for both TCP (gRPC) and UDP (Command) traffic.

### Distributed Ledger

The system state is persisted in a TOML-based ledger (`state/runs/ledger.toml`).

**Full status vocabulary:**

| Status | Owner | Meaning |
|---|---|---|
| `STARTING` | pseti start | Hardware bring-up in progress |
| `ACTIVE` | pseti start | Observation recording |
| `ABORTED` | pseti start | Rollback completed after startup failure |
| `STOPPING` | pseti stop | Hardware teardown in progress |
| `RECORDING_ENDED` | pseti stop | Hardware stopped; transfer job enqueued |
| `MANIFEST_GENERATING` | daemon | DAQ nodes computing checksums |
| `MANIFEST_READY` | daemon | All manifests generated |
| `TRANSFER_PENDING` | daemon | Awaiting rsync start |
| `TRANSFERRING` | daemon | rsync in progress |
| `TRANSFER_FAILED` | daemon | rsync failed; will retry |
| `VERIFYING` | daemon | Checking transferred files |
| `VERIFY_FAILED` | daemon | Digest mismatch detected |
| `CLEANUP_PENDING` | daemon | Awaiting selective cleanup |
| `CLEANING` | daemon | Removing .pff files from DAQ nodes |
| `ARCHIVED` | daemon | run_complete marker written |
| `COMPLETED` | (legacy) | Synonymous with ARCHIVED in old code |
| `STOPPED_WITH_ERRORS` | daemon | Archive complete but with errors |

**Node receipts** (`NodeReceipt` in `pydantic_config_models.py`) track per-DAQ-node state including `manifest_path`, `manifest_bytes`, `rsync_bytes_transferred`, `rsync_last_progress_at`, `verify_ok`, and `cleanup_ok`.

---

## Start Transaction (`StartTransaction`)

Managed via `async with StartTransaction(...) as tx:`.

### 1. `__aenter__`
- Acquires the control advisory lock.
- Returns the transaction context.

### 2. Execution Phase (in `start_run`)
- **Pre-flight**: Validates configs, checks Quabo reachability.
- **Initialize**: Writes `STARTING` status and node receipts.
- **Start**: Launches local daemons, concurrent `StartDaq` RPCs, and heartbeat probes.

### 3. `__aexit__` (Rollback Ladder)
If an exception occurs (e.g., node timeout), `__aexit__` triggers the rollback:
1. **Stop Attempted DAQs**: Concurrent `StopDaq` for nodes with receipts.
2. **Stop Quabo Flow**: Halt data transmission.
3. **Kill Local Daemons**: Cleanup HK/HV/Temp processes.
4. **Archive artifacts**: Move partial run data to `_aborted/` with a context dump.
5. **Update Ledger**: Mark as `ABORTED`.
6. **Release Lock**.

### Start Flow Diagram
```mermaid
flowchart TD
    A[Start Request] --> B[tx.__aenter__: Acquire Control Lock]
    B --> C[Validate & Initialize Ledger: STARTING]
    C --> D[TaskGroup: StartDaq & Heartbeat]
    D -- Success --> E[Update Ledger: ACTIVE]
    E --> F[tx.__aexit__: Release Control Lock]
    D -- Failure / Exception --> G[tx.__aexit__: Rollback Ladder]
    G --> H[Stop DAQs & Quabo Flow]
    H --> I[Kill Daemons & Archive Data]
    I --> J[Update Ledger: ABORTED]
    J --> F
```

---

## Stop Transaction (`StopTransaction`)

Managed via `async with StopTransaction(...) as tx:`.

### 1. `__aenter__`
- Acquires the control advisory lock.

### 2. `__aexit__` (Teardown sequence)
Ensures **resilient best-effort hardware shutdown**. All steps execute even if previous ones fail. Bulk I/O is NOT in this sequence:

1. **Stop DAQs**: Concurrent `StopDaq` RPCs to all DAQ nodes.
2. **Kill Daemons**: Terminate local control processes (HV updater, HK recorder, etc.).
3. **Stop Quabos**: Signal hardware to halt data generation.
4. **Enqueue transfer job**: Build a `TransferJob` (see schema below) and write `{run_name}.job.toml` to `state/transfer/queue/pending/`. Skipped if `--no-transfer`.
5. **Update Ledger**: Transition to `RECORDING_ENDED`.
6. **Release Control Lock** — happens in seconds, not hours.

### TransferJob schema — the stop→daemon contract

`pseti stop` constructs a `TransferJob` (defined in `control/src/control/utils/pydantic_config_models.py`) and serializes it as TOML into the pending queue. The daemon parses it with `TransferJob.model_validate(toml.load(f))`. All fields round-trip exactly, including `port_forwarding`.

```toml
schema_version = 1
run_name = "start_2024-01-01T00:00:00Z.sci"
head_data_dir = "/data/panoseti"
head_node_username = "panoseti"
created_at = "2024-01-01T00:00:00+00:00"
attempts = 0
no_cleanup = false
no_collect = false
skip_verify = false

[[daq_nodes]]
ip_addr = "192.168.0.10"
username = "panoseti"
data_dir = "/data"
module_ids = [250, 251]

[daq_nodes.port_forwarding]
status = true
gw_ip = "10.0.1.254"
port = 2200
```

**Key invariant**: the `port_forwarding` block is preserved from `daq_config.json` through `TransferNodeSpec` into the TOML. Prior to this refactor, the field was silently dropped, causing rsync to fail over the physical router when the observatory uses a VPN gateway.

### Stop Flow Diagram
```mermaid
flowchart TD
    A[Stop Request] --> B[tx.__aenter__: Acquire Control Lock]
    B --> C[Set Ledger: STOPPING]
    C --> D[tx.__aexit__: Resilient Hardware Teardown]
    D --> E[Stop DAQs via gRPC + Kill Daemons + Stop Quabos]
    E --> F{--no-transfer?}
    F -- No --> G[Build TransferJob + enqueue to state/transfer/queue/pending/]
    F -- Yes --> H[Skip enqueue — log WARNING]
    G --> I[Set Ledger: RECORDING_ENDED]
    H --> I
    I --> J[Release Control Lock — seconds elapsed]
    J --> K[Transfer Daemon picks up job asynchronously]
```

### pseti stop flags

| Flag | Effect |
|---|---|
| `--no-transfer` | Skip enqueue entirely; data stays on DAQ nodes. |
| `--keep-daq-data` | Sets `no_cleanup=True` on the job (.pff files preserved after archive). |
| `--skip-verify` | Sets `skip_verify=True` on the job (manifest re-hash skipped). Discouraged — CLI prints a warning. |
| `--force-cleanup` | Force cleanup even if hashpipe liveness is uncertain. |
| `--yes / -y` | Auto-confirm safety prompts including daemon-down warning. |

**Daemon-down warning**: before enqueuing, `pseti stop` checks the heartbeat at `state/transfer/daemon.heartbeat`. If the daemon is stale (>30 s since last write):
- Interactive TTY: prompts the operator to confirm before enqueuing.
- Non-interactive / `--yes`: enqueues and emits a WARNING log.

---

## Transfer Daemon (`control/transfer/daemon.py`)

### Lifecycle

The daemon is started by `session_start.py` via `util.start_daemon(["python", "-m", "control.transfer"])` after Redis daemons. It writes its PID to `state/transfer/daemon.pid` and updates a heartbeat at `state/transfer/daemon.heartbeat` every 5 seconds.

`session_stop.py` sends SIGTERM and waits up to 30 seconds for the daemon to finish its current processing step before escalating to SIGKILL. An in-flight rsync is always allowed to complete its current attempt before the daemon exits.

On receipt of SIGTERM/SIGINT the daemon: finishes the current stage, moves any active job back to `pending/` (crash recovery), then releases the flock and exits cleanly.

### Queue Layout

```
state/transfer/queue/
  pending/    {run_name}.job.toml   ← pseti stop writes here
  active/     {run_name}.job.toml   ← daemon moves here on claim()
  failed/     {run_name}.job.toml   ← daemon moves here after MAX_ATTEMPTS
  completed/  {run_name}.job.toml   ← daemon moves here on success
```

All transitions use `os.rename` (POSIX-atomic). Double-enqueue of the same run is idempotent.

### Job Lifecycle

```
pending/ → daemon calls claim() → active/
active/  → success              → completed/
active/  → failure MAX_ATTEMPTS → failed/
```

### State Machine per Job

```mermaid
flowchart TD
    RE[RECORDING_ENDED] --> MG[MANIFEST_GENERATING]
    MG -- all manifests OK --> TR[TRANSFERRING]
    MG -- partial failure --> TR
    TR -- rsync OK --> VY[VERIFYING]
    TR -- rsync error --> TFail[TRANSFER_FAILED]
    TFail -- retry < MAX_ATTEMPTS\nbackoff 5s / 30s --> TR
    TFail -- exhausted --> SE[STOPPED_WITH_ERRORS]
    VY -- all digests match --> CL[CLEANING]
    VY -- any mismatch --> VF[VERIFY_FAILED]
    VF --> SE
    CL -- manifest_digest accepted --> AR[ARCHIVED]
    CL -- FAILED_PRECONDITION --> VF
```

### Integrity Invariant — No Deletion Without Verified Integrity

The VERIFYING stage calls `verify_manifest()` (`transfer/verify.py`) on every manifest file found on the head node (`manifest.blake3`, `manifest.xxh3_128`, `manifest.sha256`).  Each file listed in the manifest is re-hashed and compared against the recorded digest.  On any mismatch the daemon transitions to `VERIFY_FAILED`, logs the exact file paths, and **skips cleanup** — DAQ-side `.pff` files are preserved for manual recovery.

The CLEANING stage passes `manifest_digest` (SHA-256 of the manifest file content) to `CleanupData(mode=CLEANUP_SELECTIVE)`.  The DAQ Control server recomputes the digest of its local manifest and refuses the RPC with `FAILED_PRECONDITION` if the values differ.  This closes the loop: the head node must prove it verified the same manifest the DAQ node generated, or deletion is impossible.

### Selective Cleanup

The daemon calls `CleanupData(mode=CLEANUP_SELECTIVE, manifest_digest=<sha256_of_manifest>)` with:
- `delete_patterns = ["*.pff"]` — science files removed from DAQ nodes
- `preserve_patterns = ["*.json", "*.log", "*.toml"]` — metadata retained on-DAQ as a permanent catalog

### Retry Ladder

Transfer failures use exponential backoff before re-queuing to `pending/`:

| Attempt | Backoff before next attempt |
|---|---|
| 1 → 2 | 5 s |
| 2 → 3 | 30 s |
| 3 (MAX) | → `failed/` queue, no retry |

### Daemon Crash Recovery

On startup the daemon sweeps `active/` for jobs left behind by a prior crash.  Each stranded job is renamed back to `pending/` (POSIX-atomic) before the main poll loop begins, ensuring no run is silently dropped (SC-TX-005).

---

## Operator Recovery (`pseti obs transfer`)

Use the `pseti obs transfer` sub-commands to inspect the queue and recover from failures.

| Command | Purpose |
|---|---|
| `pseti obs transfer status` | Daemon health (heartbeat age, pid) + per-bucket job counts. |
| `pseti obs transfer status <run>` | Show which bucket a specific run is in. |
| `pseti obs transfer queue [pending\|active\|completed\|failed]` | List jobs in a bucket. |
| `pseti obs transfer retry <run>` | Move a failed job back to pending/ (resets attempts). |
| `pseti obs transfer start` | Start the daemon (idempotent; no-op if already running). |
| `pseti obs transfer stop` | SIGTERM the daemon; wait up to 60 s for graceful exit. |
| `pseti obs transfer tail [-f] [-n N]` | Tail `state/logs/transfer_daemon/current.log`. |
| `pseti obs transfer verify <run>` | Run manifest verification standalone (no state transitions). |

**Common recovery flows:**

*Transfer daemon was down when `pseti stop` ran:*
```bash
pseti obs transfer status          # confirm daemon is down
pseti obs transfer start           # restart it
# daemon auto-picks up the pending job
pseti obs transfer status <run>    # confirm it moved to active/
```

*rsync failed and exhausted retries:*
```bash
pseti obs transfer queue failed    # confirm run is in failed/
# investigate root cause (disk space, network, SSH keys)
pseti obs transfer retry <run>     # move back to pending/
pseti obs transfer start           # ensure daemon is running
```

*Manifest digest mismatch (VERIFY_FAILED):*
```bash
pseti obs transfer verify <run>    # re-run verification to confirm which files differ
# DAQ data is preserved — do NOT run CleanupData manually
# Fix the head-side issue (re-rsync the specific file), then:
pseti obs transfer retry <run>
```

---

## Network Interaction

```mermaid
sequenceDiagram
    participant Head as Head Node (pseti stop)
    participant Daemon as Transfer Daemon
    participant DAQ as DAQ Nodes
    participant Quabo as Quabo Boards

    Head->>Head: tx.__aenter__ (Control Lock)
    Head->>DAQ: StopDaq (Concurrent gRPC)
    Head->>Quabo: Stop Data Flow
    Head->>Head: Enqueue TransferJob → state/transfer/queue/pending/
    Head->>Head: Ledger: RECORDING_ENDED
    Head->>Head: tx.__aexit__ (Release Control Lock)

    Note over Head,Quabo: Control lock released in seconds

    Daemon->>Daemon: Acquire Transfer Lock
    Daemon->>Daemon: Recover stranded active/ jobs → pending/
    Daemon->>DAQ: GenerateManifest (per module, blake3)
    Daemon->>DAQ: rsync run directories (up to 3 attempts, backoff 5s/30s)
    Daemon->>Head: verify_manifest() — re-hash every file in manifest
    Note over Daemon,Head: VERIFY_FAILED → skip cleanup, preserve DAQ data
    Daemon->>DAQ: CleanupData SELECTIVE (*.pff only, manifest_digest required)
    Daemon->>Head: Write run_complete marker
    Daemon->>Head: Ledger: ARCHIVED
```
