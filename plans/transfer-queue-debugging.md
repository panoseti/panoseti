# EXECUTION_PLAN.md — Transfer-Queue & Start/Stop Stabilization

**Author:** Principal architect (planning role)
**Audience:** Coding agent (Sonnet)
**Scope:** Diagnose-and-fix package targeting the bugs surfaced in `control/ci/transfer-queue-debugging-notes.md`. No new features beyond what's listed; no refactors beyond what each bug requires.

---

## Context

The hardware-software (HITL) pipeline is now stable for `pseti start`/`pseti stop`, but manual operator testing has uncovered five categories of defects that block the next milestone (automated HW-SW transfer tests):

1. **Silent transfer-daemon crashes**, plus an "infinite bounce" of jobs between `active/` and `pending/` with no retry counter increment and no log evidence.
2. **`pseti start` ignores Quabo unreachability** (always lenient in container mode), so the start transaction succeeds with no real data flow — and HW-SW failure modes are masked.
3. **Race condition / non-idempotent `pseti start`**: first invocation reports `ABORTED` while a hashpipe child is still running on the DAQ node; subsequent invocations succeed against the orphaned state.
4. **CLI observability blindspots**: `pseti obs transfer tail` looks at a path that the daemon never writes to; ledger inspection is awkward; remote DAQ status requires hand-resolving IPs and port-forwarding.
5. **gRPC server singleton not enforced** — multiple `panoseti-server` processes can attach to the same node config, racing for UDP packets.

This plan fixes #1–#5 with minimal scope, adds tests that will fail on the current `master`/`test-refactor` branch, and gives a manual HITL verification checklist for the work that requires real Quabos.

---

## Root-cause diagnosis (read this before touching code)

### D-1. Why the daemon dies silently

Two cooperating defects:

- `control/src/control/utils/util.py:265-269` — `start_daemon()` redirects **stdout and stderr to `subprocess.DEVNULL`**.
- `control/src/control/transfer/__main__.py:7-10` — the daemon entrypoint calls `logging.basicConfig(level=INFO, format=…)`, which installs a single `StreamHandler` writing to **stderr**, i.e. `/dev/null`.

Net result: any `Exception`, `ExceptionGroup`, traceback, or print emitted by the daemon is lost. The "daemon randomly crashed and there are no logs" observation is the literal expected behavior of this configuration.

### D-2. Why jobs bounce between `active/` and `pending/` without `attempts` incrementing

In `control/src/control/transfer/daemon.py:376-407`:

```python
try:
    success = await _process_job(job)
    if success:
        tq.complete(run_name)
    elif attempts >= MAX_ATTEMPTS:
        tq.fail(run_name)
    else:
        # increments attempts, writes back to active/, renames to pending/
        ...
except Exception:
    logger.exception(...)
    tq.fail(run_name)
```

`attempts` is **only incremented when `_process_job` returns**. If the daemon **process** dies before that (uncaught exception escaping the `try`, OOM, `SIGKILL`, asyncio loop crash, an `OSError` in `tq._write_job` or `os.rename`, an unhandled `ExceptionGroup` from the inner `TaskGroup`s, a network partition that makes `subprocess.run(rsync)` hang past container restart), the `finally` block releases the file lock and unlinks the pid file — but **leaves the job in `active/` with its original attempts value**.

On the next daemon start, `_sweep_stranded_jobs()` (`daemon.py:307-324`) renames the active job back to `pending/`. **It does not increment `attempts`.** The next claim picks the same job, the same crash recurs, the bounce is infinite, and because of D-1 it is invisible.

There are also at least two specific in-process exceptions that escape the local try/except in `_process_job`:

- `state_mgr.transition()` — if the ledger is missing or in a state that disallows the requested transition (the `RECORDING_ENDED → MANIFEST_GENERATING` transition for example), this raises and is **not** caught inside `_process_job` (it sits inside `try:` but the `if manifest_errors: state_mgr.transition("TRANSFER_FAILED")` line sits outside the `try`). The exception bubbles to the outer `except Exception` — which calls `tq.fail()` correctly. **But** the ExceptionGroup from `asyncio.TaskGroup` (`gen_manifest`, `cleanup_node`) is *not* caught at all and will propagate out of `_process_job`, exit the loop's `try`, hit `finally`, and tear the daemon down. That is the exact crash signature we expect from a single Quabo gRPC unavailability.
- The retry write-back branch at `daemon.py:398-403` writes to `active/` and renames to `pending/`. If `_write_job` raises mid-rename (e.g. transient `ENOSPC`), the daemon dies with the job stranded and `attempts` not yet persisted.

### D-3. Why `pseti start` is too lenient on Quabo reachability

`control/src/control/start.py:935-937`:

```python
await _check_quabo_reachability(
    quabo_uids, network_config, lenient=bool(daq_config.head_node_container)
)
```

The single signal `daq_config.head_node_container` is overloaded — it means both "we're in CI" and "skip every hardware check". In the HITL container this flag is `True`, so Quabo timeouts are downgraded to `WARNING` and the start transaction proceeds, configures hashpipe, and reports `ACTIVE` against silent Quabos.

The same overloaded flag is checked in five places (`start.py:840, 848, 884, 891, 899`) — every hardware precondition silently degrades to a warning. There is no way for HW-SW to say "I am in a container *and* I require real hardware".

### D-4. Why multiple `pseti start` invocations behave inconsistently

`start_run` does not pre-flight check for an already-running remote hashpipe before issuing UDP reconfiguration to Quabos at `start.py:944` (`start_data_flow`). The order is:

1. Acquire control lock, write `STARTING` ledger.
2. `make_run_dirs` (creates dirs on DAQ nodes via SSH).
3. Lenient Quabo reachability sweep.
4. **`start_data_flow`** — sends DAQ-mode UDP commands to every Quabo. Irreversible.
5. `start_recording` — `StartDaq` gRPC. If a hashpipe is already alive on the DAQ side the server's pid check accepts it and returns `success=True`, OR a TimeoutError is raised, triggering rollback.
6. On rollback the ladder calls `StopDaq` per node. But Step 4 has already mutated Quabo state.

So when the first invocation fails partway through Step 5, Quabo state is already changed (Step 4 completed), and the rollback fires `StopDaq` — which kills hashpipe but does not undo Quabo data-flow direction. The second invocation then succeeds because `start_data_flow` is *re-runnable* (Step 4 is idempotent against current Quabo state) and the DAQ node is now empty.

**Critical constraint on rollback:** The `stop_data_flow` call in `StartTransaction.__aexit__` (line 172-175) is **wrong to call unconditionally**. If a pre-existing valid run is already active and another operator accidentally calls `pseti start` — which fails at some pre-flight check — the rollback ladder must NOT call `stop_data_flow`, because that would halt data transmission for the running observation. Only the *same transaction that called `start_data_flow`* may undo it.

Two things must change:
- `start_data_flow` must come **after** a remote-hashpipe-running pre-flight check (a `StatusDaq` probe with `check_hashpipe_running=True`); refuse without `--force-restart`.
- Add a `_data_flow_started: bool = False` flag to `StartTransaction`. Set it to `True` immediately before `start_data_flow` is called (`start.py:944`). In `__aexit__`, call `stop_data_flow` **only if `self._data_flow_started is True`**. Failures that abort before reaching `start_data_flow` leave Quabo state untouched.

### D-5. Why `pseti obs transfer tail` looks at the wrong path

`cli.py:165-169` reads `PanoPaths.daemon_logs_dir("transfer_daemon") / "current.log"` → `state/logs/transfer_daemon/current.log`. The daemon **never writes that file** (see D-1). Two possible fixes; we choose: **make the daemon actually write to that path** (using the unified `panoseti_grpc.telemetry.logger.get_logger`), keeping the tail command unchanged. This makes the daemon's logs show up in Loki for free.

### D-6. Multiple gRPC server processes

The unified `panoseti-server` does not enforce a process-level singleton on a node. Two servers on different ports can coexist; two servers on the **same** port will fail to bind, but two servers on different ports both attempting to drive the same hashpipe are possible. The fix is to use the same atomic-lock pattern as the transfer daemon: a `state/locks/grpc_server.lock` file with PID and `O_EXCL` semantics. Out of scope for this plan unless you can scope it cheaply — see Phase 5.

---

## Phase 1 — Daemon Stabilization

### 1.1 Make the daemon log to disk and Loki

**File:** `control/src/control/transfer/__main__.py`

Replace the `logging.basicConfig` call with a single call to the project's structured logger so logs land at the path that the CLI tail command already reads:

```python
from panoseti_grpc.telemetry.logger import get_logger
from control.utils.paths import PanoPaths
log_dir = PanoPaths.daemon_logs_dir("transfer_daemon")
log_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger("transfer_daemon", log_dir=log_dir, grpc_enabled=False)
```

The logger writes:
- `{log_dir}/transfer_daemon.log` (plain text)
- `{log_dir}/transfer_daemon.jsonl` (Alloy → Loki)
- console (for ad-hoc foreground runs)

**Rename target:** the existing CLI tail at `cli.py:166` references `current.log`. Either (a) update `cli.py:166` to use `transfer_daemon.log` (the actual filename produced), or (b) symlink/use `current.log` as the canonical name in `get_logger`. Choose (a) — it's a one-line CLI fix and keeps logger behavior unchanged.

Also: replace the module-level `logger = logging.getLogger("panoseti.transfer_daemon")` in `daemon.py:21` with `logger = get_logger("transfer_daemon", log_dir=PanoPaths.daemon_logs_dir("transfer_daemon"))` so `_process_job` logs reach the same handlers without reconfiguring on every call.

### 1.2 Capture daemon stdout/stderr as a backstop

**File:** `control/src/control/utils/util.py:246-273`

`start_daemon()` currently routes stdout/stderr to `/dev/null`. Change so daemons launched via this helper redirect to `state/logs/<daemon-name>/stdout.log` and `stderr.log`. This protects us from any uncaught exception that escapes the structured logger (e.g. the interpreter dying before a handler is installed).

Implementation sketch:
- Accept an optional `name` kwarg; default to a sanitized form of `prog_label`.
- `log_dir = PanoPaths.daemon_logs_dir(name); log_dir.mkdir(parents=True, exist_ok=True)`
- Open `stdout.log` and `stderr.log` in append mode with `os.O_APPEND | O_CREAT`, pass the fds to `Popen`.
- Update the `session_start.py:99-100` callsite to pass `name="transfer_daemon"`.

This is cheap (≤25 LOC) and solves "the daemon disappeared and we have no idea why" forever.

### 1.3 Make every job processing failure visible **and** countable

**File:** `control/src/control/transfer/daemon.py`

The current `_process_job` returns `False` on logical failure but lets exceptions escape. The outer loop catches them and calls `tq.fail()` — but only when control reaches the loop. If the exception escapes the loop's `try` (e.g. a write-back error in the retry branch), the daemon dies.

Required edits:

**(a)** Wrap the entire body of `_process_job` in `try/except Exception`. On any exception, log via `logger.exception(...)` (full traceback), transition the ledger to `TRANSFER_FAILED`, and **return `False`**. Never let an exception escape `_process_job`.

**(b)** In the daemon main loop (`daemon.py:367-407`), increment the persisted attempt count **at claim time**, not at retry time:

```python
job = tq.claim()
if job is None: ...
attempts = job.attempts + 1
# Persist the bumped count BEFORE processing, so a daemon crash leaves
# the bumped count on disk in active/.
incremented = job.model_copy(update={"attempts": attempts})
tq._write_job(tq._queue / "active" / f"{job.run_name}.job.toml", incremented)
```

**(c)** Update `_sweep_stranded_jobs` (`daemon.py:307-324`) to **not** ressurrect jobs whose `attempts` already meets `MAX_ATTEMPTS`. Move those directly to `failed/` with a sentinel `last_transfer_error="DAEMON_CRASHED_DURING_PROCESSING"`. This combined with (b) breaks the infinite-bounce loop deterministically.

**(d)** Wrap the retry write-back at `daemon.py:398-403` in `try/except OSError` so a transient filesystem error doesn't kill the daemon.

**(e)** Wrap the `asyncio.TaskGroup` block in `_process_job` (`gen_manifest`, `cleanup_node`) so that an `ExceptionGroup` becomes a list of error strings and a single `state_mgr.transition("TRANSFER_FAILED")`. Don't let it propagate.

### 1.4 Make `_process_job` honour shutdown signals cleanly

**File:** `control/src/control/transfer/daemon.py`

Pass the `shutdown` `asyncio.Event` from `run_daemon` into `_process_job`. After each stage, check `if shutdown.is_set(): return False`. The state machine should transition to `TRANSFER_FAILED` (with reason `"DAEMON_SHUTDOWN"`) and rely on the on-restart sweep to re-enqueue. This prevents partial-writes on SIGTERM.

### 1.5 Vet the start/stop transactions for the same crash class

**File:** `control/src/control/start.py:103-229` (`StartTransaction.__aexit__`)
**File:** `control/src/control/stop.py:124-240` (`StopTransaction.__aexit__`)

**StartTransaction fixes:**
- Add `self._data_flow_started: bool = False` to `__init__`. Set it to `True` in `start_run` immediately before the `start_data_flow(...)` call. In `__aexit__` Ladder Step 3 (`stop_data_flow`), guard with `if self._data_flow_started:` — do NOT stop data flow if this transaction never started it. Calling `stop_data_flow` on a rollback from a pre-flight failure would silently kill an active co-existing run.
- `__aexit__` re-loads the ledger twice (lines 124, 132). If `state_mgr.load_state()` raises (partial TOML write during concurrent access), the rollback dies. Wrap both loads in `try/except`; treat `None` as "no node receipts to roll back."
- Verify that nothing in the rollback ladder can `raise` past the surrounding `try/finally` (which releases the lock). Each of the 5 ladder steps already has its own `except`; double-check Step 5 (archive partial artifacts) which calls `shutil.move` and can raise on permission errors.

**StopTransaction fixes:**
- `__aexit__` line 138-141: if `exc_type is not None`, it transitions to `STOPPED_WITH_ERRORS` and returns `False` (re-raises immediately), **skipping steps 1-4** (stop recording, kill daemons, stop quabos, enqueue transfer). That means a pre-flight exception in the `with` block leaves DAQs and Quabos running. Change the early-return to log-and-fall-through so ladder steps always execute regardless of `exc_type`.
- `load_state()` call in the `with` block (stop.py:622): wrap in `try/except` so a missing/corrupt ledger doesn't abort the stop entirely.

### 1.6 Add structured failure context to the job TOML

**File:** `control/src/control/transfer/models.py` (read this file before editing; it should already have the `TransferJob` model)

Add two fields:
- `last_error: str | None = None`
- `last_error_at: datetime | None = None`

Update `_process_job` failure paths to set these via `model_copy` and `_write_job` before transitioning. Without this, `pseti obs transfer queue failed` is uninformative.

---

## Phase 2 — Strictness Modes (`pseti start`)

### 2.1 New flag: `--strict` / `--no-strict`

**File:** `control/src/control/start.py`

Add `strict: bool = typer.Option(None, "--strict/--no-strict", help="...")` to the Typer command and propagate through `start_run`.

Resolution order for the effective `strict` value:
1. CLI flag (highest).
2. Env var `PSETI_STRICT={1|0}`.
3. **Default**: `True` unless `daq_config.head_node_container is True` AND `os.environ.get("PSETI_TEST_TIER")` is one of `tier3_fleet`, `tier4_chaos`, `tier5_integration`. (Tier-aware default lets pure software CI stay lenient while HW-SW defaults to strict.)

### 2.2 Replace overloaded `head_node_container` checks

In `start.py` lines 840, 848, 884, 891, 899, 935-937: replace `if daq_config.head_node_container:` with `if not strict:`. The behavior in lenient mode is unchanged (warning + continue); in strict mode we raise `ValidationError`.

### 2.3 Add a hashpipe-already-running pre-flight

**File:** `control/src/control/start.py` — new helper `_check_no_remote_hashpipe(daq_config)`.

For each DAQ node with `module_ids`, call `StatusDaq({"check_hashpipe_running": True, ...})`. If any node reports `hashpipe_running=True`, raise `ValidationError("Hashpipe already running on {ip}; run `pseti stop` first or use --force-restart")`.

Call it **before** `start_data_flow` at `start.py:944` and after `_check_quabo_reachability`. Add `--force-restart` flag for operators who really mean it (calls `StopDaq` on each offending node first).

### 2.4 Document strictness in `TRANSACTIONS.md`

Single section under "Pre-flight": list the seven preconditions (config validation, head-node identity, ledger freshness, HK recorder absence, Redis daemons, PH baseline, Quabo reachability, remote hashpipe absence) and note that strict mode requires all; lenient mode warns and proceeds.

---

## Phase 3 — CLI Observability

### 3.1 Fix `pseti obs transfer tail`

**File:** `control/src/control/transfer/cli.py:165-171`

Update to read `transfer_daemon.log` instead of `current.log` (matches the file produced by 1.1). Also update the error message to print *both* `state/logs/transfer_daemon/transfer_daemon.log` and `state/logs/transfer_daemon/stderr.log` so operators don't miss the backstop.

### 3.2 New command: `pseti obs status` with auto-resolution

**File:** `control/src/control/status.py` (read first; if it's currently a thin wrapper, extend it)

Add commands so operators don't hand-resolve IPs:

| Command | Behavior |
|---|---|
| `pseti obs status` | Local head-node summary: ledger status, run name, transfer queue counts, daemon health, Redis status, disk free. |
| `pseti obs status --remote` | Iterate `daq_config.daq_nodes` (use `util.daq_grpc_endpoint` + `attach_daq_config` so port-forwarding is transparent); per-node show: gRPC reachable, hashpipe_pid, run_dir present, free disk. |
| `pseti obs status --watch [--interval N]` | Re-render every N seconds (default 5) using `rich.live.Live`. Supports `--remote` modifier. |
| `pseti obs status sweep` | Full network sweep: Quabo ping, gRPC reachability, port-forwarding checks. Read-only; no state changes. |

The implementation reuses `_check_daq_reachability()` and `_check_quabo_reachability(lenient=True)` already in `start.py`. Move them into `control/utils/preflight.py` and import from both places — small refactor, keeps blast radius low.

### 3.3 New command: `pseti obs ledger [<run_name>]`

**File:** `control/src/control/tools/ledger_cli.py` (new) — register in `obs_cli.py:14-25` lazy_mapping.

| Subcommand | Behavior |
|---|---|
| `pseti obs ledger` | Show current ledger TOML pretty-printed (rich Syntax) with full path: `state/runs/ledger.toml`. |
| `pseti obs ledger path` | Print just the absolute path (for shell composition: `vim "$(pseti obs ledger path)"`). |
| `pseti obs ledger history` | List archived ledgers under `head_node_data_dir/_aborted/*/stale_run_state.toml`. |
| `pseti obs ledger show <run_name>` | Read-only view of a specific run's ledger snapshot. |

**Hard rule (per user request):** no `edit`/`set` subcommand. Inspection only.

### 3.4 Make `pseti grpc status` config-aware

**File:** `panoseti_grpc/cli.py` (in the grpc submodule). Extend the existing `status` command:

- New flag `--from-config`: load `daq_config.json`, iterate nodes, print one row per node.
- Use `util.daq_grpc_endpoint(node)` so port-forwarding is honored.
- Output columns: `node | grpc_endpoint | health | hashpipe_pid | last_seen`.
- `--watch`/`--interval` parity with 3.2.

Note: this lives in the gRPC submodule; the coding agent must update the submodule pin after merging.

---

## Phase 4 — Verification Tests

Each test below must **fail** on the current `test-refactor` branch and **pass** after Phases 1–3.

### Test 4.1 — Tier 4 (chaos): infinite-bounce regression

**Path:** `control/ci/tier4_chaos/test_transfer_daemon_crash_recovery.py` (new)

**Scenario:** Enqueue a job whose `_process_job` is monkeypatched to raise `RuntimeError("boom")` on the first attempt. Start the daemon. Assert that within 5 seconds:
- Job moves through `active/` to `failed/` (not back to `pending/`).
- `attempts == 1` in the failed job TOML.
- `state/logs/transfer_daemon/transfer_daemon.log` contains the traceback string `"boom"`.

**Stronger variant:** SIGKILL the daemon mid-`_process_job` (use a `MockTransferStage` that sleeps forever). Restart. Assert the job is in `pending/` with `attempts == 1` (because we now persist on claim, see Phase 1.3.b). Restart with the same monkeypatch, assert it reaches `failed/` after `MAX_ATTEMPTS=3` total bumps.

### Test 4.2 — Tier 2 (logic): strict mode aborts on Quabo unreachable

**Path:** `control/ci/tier2_logic/test_start_strict_mode.py` (new)

**Scenario:** Mock `_check_reachability` to return `(False, "timeout")` for one Quabo. Call `start_run(...)` with `strict=True`. Assert that:
- `ValidationError` is raised before `start_data_flow` is ever called (use a spy).
- Ledger is not written.
- Lock is released.

Then `strict=False` against same mock: assert `start_data_flow` **is** called and a warning is logged.

### Test 4.3 — Tier 4 (chaos): start.py refuses if remote hashpipe already running

**Path:** `control/ci/tier4_chaos/test_start_remote_hashpipe_guard.py` (new)

**Scenario:** Spin up a fake `daq_control` server that returns `hashpipe_pid=999, hashpipe_running=True` for `StatusDaq`. Call `start_run(strict=True)`. Assert `ValidationError` is raised, no UDP commands are issued to Quabos (spy `start_data_flow`), ledger is not in `STARTING`. Repeat with `--force-restart=True`: assert `StopDaq` is called first, then start succeeds.

### Test 4.4 — Tier 5 (integration): tail produces real output and ledger CLI works end-to-end

**Path:** `control/ci/tier5_integration/test_transfer_observability.py` (new)

**Scenario:** Start the real transfer daemon via `pseti obs transfer start` inside the integration container. Wait for heartbeat. Run `pseti obs transfer tail -n 5`; assert non-empty output containing `"Transfer daemon started"`. Run `pseti obs ledger path`; assert it prints an existing path. Stop the daemon; assert `pseti obs transfer status` reports STALE within `>30s` heartbeat age.

---

## Phase 5 — Optional / Stretch

These are scoped but **only commit if Phases 1-4 pass clean** in CI:

- **5.1** Singleton `panoseti-server` enforcement via lock file under `state/locks/grpc_server.lock` (mirrors transfer-daemon pattern). One file edit in `panoseti_grpc/server/main.py`.
- **5.2** `pseti obs transfer tail -f` should also tail `stderr.log` and `stdout.log` interleaved so the daemon-died-before-logging case is still visible. Implement via `tail -F log1 log2 log3` exec.
- **5.3** Decrement noisy `quabo_driver` `WARNING` to `DEBUG` during reboot windows; out-of-scope for this plan (note in TRANSACTIONS.md follow-up section).

---

## Files To Modify (summary)

| Path | Phase | What |
|---|---|---|
| `control/src/control/transfer/__main__.py` | 1.1 | Switch to `panoseti_grpc.telemetry.logger.get_logger`. |
| `control/src/control/transfer/daemon.py` | 1.1 / 1.3 / 1.4 | Use shared logger; persist `attempts` at claim; sweep guard against `attempts >= MAX_ATTEMPTS`; pass shutdown event into `_process_job`; wrap inner `TaskGroup`s. |
| `control/src/control/transfer/models.py` | 1.6 | Add `last_error` / `last_error_at`. |
| `control/src/control/transfer/cli.py` | 3.1 | Tail `transfer_daemon.log`; show stderr/stdout paths in error. |
| `control/src/control/utils/util.py` | 1.2 | `start_daemon()` writes stdout/stderr to per-daemon log files. |
| `control/src/control/start.py` | 1.5 / 2.* | Strict mode, remote-hashpipe pre-flight, robust rollback. |
| `control/src/control/stop.py` | 1.5 | Always run ladder steps, robust against `load_state()` exceptions. |
| `control/src/control/utils/preflight.py` | 3.2 | Move `_check_*_reachability` helpers here for reuse. |
| `control/src/control/status.py` | 3.2 | `--remote`, `--watch`, `sweep` subcommands. |
| `control/src/control/tools/ledger_cli.py` | 3.3 | New, read-only ledger CLI. |
| `control/src/control/tools/obs_cli.py` | 3.3 | Register `ledger` lazy mapping. |
| `panoseti_grpc/.../cli.py` | 3.4 | `--from-config`, `--watch`. (submodule) |
| `control/TRANSACTIONS.md` | 2.4 | Document strictness modes. |
| `control/ci/tier2_logic/test_start_strict_mode.py` | 4.2 | New. |
| `control/ci/tier4_chaos/test_transfer_daemon_crash_recovery.py` | 4.1 | New. |
| `control/ci/tier4_chaos/test_start_remote_hashpipe_guard.py` | 4.3 | New. |
| `control/ci/tier5_integration/test_transfer_observability.py` | 4.4 | New. |

---

## Verification — automated

Run in this order; each must pass before the next:

```bash
pseti test lint                                     # ruff + mypy
pseti test sw unit                                  # tier 1
pseti test sw logic   -k strict_mode                # tier 2 — new test 4.2
pseti test sw chaos   -k 'crash_recovery or hashpipe_guard'   # tier 4 — new tests 4.1, 4.3
pseti test sw integration -k transfer_observability # tier 5 — new test 4.4
pseti test sw chaos                                 # full chaos sweep — must not regress
pseti test sw integration                           # full integration — must not regress
```

---

## Verification — manual HW-SW (Nico runs this; agent does not have hardware)

Pre-condition: HITL container running, real Quabos powered, DAQ node reachable. Use `pseti obs status sweep` to confirm baseline.

### M-1. Daemon-crash visibility (validates Phase 1.1, 1.2)

```bash
pseti session-start
pseti obs transfer status                  # daemon RUNNING, fresh heartbeat
sudo kill -9 $(cat $PSETI_STATE/transfer/daemon.pid)
sleep 2
ls -la $PSETI_STATE/logs/transfer_daemon/  # expect: transfer_daemon.log, transfer_daemon.jsonl, stderr.log, stdout.log
pseti obs transfer tail -n 20              # must show last logs incl. shutdown reason if any
pseti obs transfer start                   # restart
pseti obs transfer tail -f                 # continues to stream
```

**Pass criteria:** all four files exist and contain content; tail shows real log lines (not "file not found"); restart produces a new "Transfer daemon started" entry.

**Automated analog:** Test 4.4 covers the file existence and tail behavior under a non-killed daemon. To cover the SIGKILL case automatically, extend Test 4.1's stronger variant.

### M-2. Infinite-bounce extinction (validates Phase 1.3)

```bash
pseti start --strict --nsecs=20            # 20-second run
pseti stop                                 # enqueues transfer
# Now break the network mid-transfer:
sudo iptables -A OUTPUT -d <daqnode_ip> -j DROP
pseti obs transfer status                  # watch for 60s
sudo iptables -D OUTPUT -d <daqnode_ip> -j DROP
pseti obs transfer queue failed            # job must appear here within ~3 attempts × backoff
pseti obs transfer tail -n 100             # must contain 3 retry log lines + 1 "Marking failed" line
```

**Pass criteria:** within `MAX_ATTEMPTS × max_backoff = 3 × 30s = ~90s` after the partition heals, the job lands in `failed/`. The `attempts` field in the failed TOML is exactly `3`. No infinite bounce.

**Automated analog:** Test 4.1 simulates this with monkey-patched failures; for true network-level fidelity, plan a future Tier 6 HW-SW chaos test under `control/ci/hardware-software/test_transfer_partition.py` using the iptables fixture.

### M-3. Strict-mode Quabo-down abort (validates Phase 2)

```bash
# Power off ONE Quabo:
pseti power off --quabo <ip>
pseti start --strict                       # MUST refuse with ValidationError listing the unreachable Quabo
pseti start --no-strict                    # warns and proceeds
pseti stop --yes
pseti power on --quabo <ip>
```

**Pass criteria:** strict mode aborts before `start_data_flow`; ledger is not written; no UDP commands reach the surviving Quabos. `--no-strict` proceeds with warnings.

**Automated analog:** Test 4.2 covers this with mocked reachability. HW-SW analog: `control/ci/hardware-software/test_strict_mode.py` cycling Quabo power via the WPS gRPC.

### M-4. Remote-hashpipe guard (validates Phase 2.3)

```bash
pseti start --strict                       # success
# Force-orphan the hashpipe by SIGKILLing the head-side python process:
sudo kill -9 $(pgrep -f 'pseti start')
pseti obs ledger                           # ledger is in STARTING/ACTIVE
pseti start --strict                       # MUST refuse: "Hashpipe already running on {ip}"
pseti start --strict --force-restart       # stops the orphan, then succeeds
pseti stop --yes
```

**Pass criteria:** the second `pseti start` refuses without `--force-restart`. Ledger does not transition to `STARTING` during the refused attempt. `start_data_flow` is not called (verify via Quabo HK packet timestamps if needed).

**Automated analog:** Test 4.3.

### M-5. Observability commands (validates Phase 3)

```bash
pseti obs status                           # local summary
pseti obs status --remote                  # remote DAQ summary, no manual --host needed
pseti obs status --remote --watch          # Live view, Ctrl-C to exit
pseti obs status sweep                     # full reachability matrix
pseti obs ledger                           # current ledger
pseti obs ledger path                      # absolute path
vim "$(pseti obs ledger path)"             # opens the actual file
pseti obs ledger history                   # archived ledgers
```

**Pass criteria:** every command exits 0, prints content matching `daq_config.json` topology, and resolves port-forwarding transparently.

---

## Out of scope (explicitly)

- The full `state/` migration is already done; do not reorganize paths.
- `tools/interleave.py` daemon refactor (per memory: deferred).
- Renaming `head_node_container` config field (overloaded but widely used; replacing the **checks** is the surgical fix).
- New gRPC RPCs (`GetTransferStatus`, `RetryFailedTransfer`, etc.) from the original transfer-queue blueprint — defer until Phase 1-4 stable.
- Disk-fill prevention in `pseti start`. Defer.

---

## Definition of done

1. All seven `pseti test ...` commands above are green on a clean checkout of the resulting branch.
2. Manual checklist M-1 through M-5 passes on the HITL fixture (operator-driven).
3. `pseti obs transfer tail` returns real content within 10 seconds of `pseti session-start`.
4. A deliberately-injected daemon crash converges on `failed/` within `MAX_ATTEMPTS` and produces a stack trace in the daemon log.
5. `TRANSACTIONS.md` documents strictness modes and the new CLI surface.
