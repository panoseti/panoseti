# EXECUTION_PLAN.md — Day-2 Operability Fixes

**Author:** Principal architect (planning role)
**Audience:** Coding agent (Sonnet)
**Scope:** Surgical fixes for the four bug clusters discovered during HITL operator testing of the now-stable Phase-1 stack: (a) async exception swallowing & stop/ledger desync, (b) transfer daemon → ledger desync, (c) logging hygiene & status false positives, (d) CLI UX & infra friction. No new features beyond what is listed.

The previous plan ("Transfer-Queue & Start/Stop Stabilization") is complete and verified. This plan supersedes it.

---

## Context

After the Phase-1 stabilization landed, manual HITL testing surfaced six recurring operator pain points:

1. `pseti start` aborts correctly when Quabos are down, but the operator-visible message is `unhandled errors in a TaskGroup (2 sub-exceptions)` — the underlying tracebacks are lost.
2. `pseti stop` runs the full hardware ladder regardless of ledger state, even when the ledger marks the run as already finished.
3. The transfer daemon correctly moves failed jobs to `failed/`, but **the run ledger** never sees `transfer_attempts` increment or `last_transfer_error` populated. Operators cannot tell from the ledger why a transfer failed.
4. `pseti status` reports Quabos as `OK` even when they are physically off — lenient mode is misused as a status report.
5. Console output is **doubled** for `start/stop/status` (but not for `config` or `quabo_driver`); console lines also lack a service tag, so operators cannot tell which subsystem produced each line.
6. Operator friction: `pseti ledger` is too long; `pseti cfg` doesn't exist; transfer status has no `--watch` or progress bar; `pseti test hw clean` always wipes Docker volumes (no non-destructive sibling).

This plan delivers each fix with minimal scope, proves each one with a Tier-1/Tier-2 test that fails on the current branch, and leaves Tier-3+ regressions for the dedicated verification phase.

---

## Diagnostic Summary (read before touching code)

### D-1. Why `start` swallows the real tracebacks

**File:** `control/src/control/start.py`

`asyncio.TaskGroup` re-raises sibling failures as `ExceptionGroup` (3.11+). Two paths matter:

- `start.py:753-760` `_check_quabo_reachability` uses a bare `async with TaskGroup()` with **no `except*` / `ExceptionGroup` unwrap**. Sub-tasks raising `ValidationError("Quabo at … is UNREACHABLE")` get bundled and propagate as a group.
- `start.py:120` `StartTransaction.__aexit__` logs `f"[CRITICAL FAILURE] Start process aborted: {exc_val}"`. When `exc_val` is an `ExceptionGroup`, `str(exc_val)` is exactly the message the operator sees. The JSON dump at `start.py:217-221` uses `traceback.format_tb(exc_tb)` — the outer frame only, not the sub-exception tracebacks.
- A working unwrap template already exists at `start.py:777-785` (`_check_daq_reachability`): catch `ExceptionGroup`, walk `eg.exceptions`, format each with `traceback.format_exception(type(exc), exc, exc.__traceback__)`, then re-raise as a single `ValidationError`.

**Fix shape:** clone the `_check_daq_reachability` pattern into `_check_quabo_reachability`; in `__aexit__`, when `isinstance(exc_val, BaseExceptionGroup)`, use `traceback.format_exception(type(exc_val), exc_val, exc_tb)` (Python 3.11+ formats nested groups automatically).

### D-2. Why `pseti stop` ignores the ledger

**File:** `control/src/control/stop.py`

`StopTransaction` pre-flight (`stop.py:609-644`) calls `state_mgr.load_state()` only to resolve the run name and for the `--force-cleanup` mismatch check. It then **unconditionally overwrites** `ledger.status = "STOPPING"` at line 641 and runs the full ladder. The "stoppable" set is already encoded canonically at `run_state.py:256` — `["STARTING", "ACTIVE", "STOPPING"]`.

**Fix shape:** before the unconditional overwrite, refuse with `ValidationError("Ledger says run is already in <status>; nothing to stop. Use --force-cleanup to override.")` if `ledger.status` is not in `{STARTING, ACTIVE, STOPPING}`. `--force-cleanup` should still bypass.

### D-3. Why the transfer daemon does not update the run ledger

**Files:** `control/src/control/transfer/daemon.py`, `control/src/control/utils/run_state.py`, `control/src/control/utils/pydantic_config_models.py`

The schema is **already ready**: `RunStateLedger` carries `transfer_attempts: int = 0`, `last_transfer_error: str | None`, `next_action_not_before: datetime | None` (`pydantic_config_models.py:488-491`). `state_mgr.transition(status, **fields)` at `run_state.py:217-229` already accepts arbitrary kwargs and `setattr`s them onto the ledger. **No model or persistence work is needed.**

What's missing:
- `daemon.py:419-423` bumps `attempts` on the **TransferJob** TOML in `active/` but never writes the same number to the ledger.
- `daemon.py:444-449` (MAX_ATTEMPTS branch) calls `tq.fail(run_name)` and logs `error_msg`, but never calls `state_mgr.transition("TRANSFER_FAILED", transfer_attempts=…, last_transfer_error=error_msg)`.
- `daemon.py:450-454` (retry branch) does not write the in-flight `transfer_attempts` or `next_action_not_before` to the ledger.
- Inside `_process_job`, `state_mgr.transition("TRANSFER_FAILED")` calls at lines 189, 213, 326 (and `VERIFY_FAILED` at 239, 306) all pass no extra fields.

**Fix shape:** at every transfer-attempts mutation in the daemon and `_process_job`, mirror the value to the ledger via `state_mgr.transition(...)` kwargs. Wrap each ledger write in `try/except Exception` and log-but-continue, since a missing/corrupt ledger must never crash the daemon (Phase-1 invariant).

### D-4. Why status reports Quabos UP when they are off

**Files:** `control/src/control/start.py`, `control/src/control/status.py`

`_check_quabo_reachability(..., lenient=True)` at `start.py:746-751` does:
```
if lenient:
    logger.warning(f"{msg} (Non-fatal in container/CI environment)")
    return                          # silently returns, no exception
raise ValidationError(msg)
```
Then `_sweep_summary` at `status.py:153-160` does:
```
try:
    await _check_quabo_reachability(..., lenient=True)
    lines.append("Quabos:    OK — all configured Quabos reachable")
except ValidationError as e:
    lines.append(f"Quabos:    WARNING — {e}")
```
Lenient mode never raises, so the `except` arm is unreachable; status always prints `OK`.

**Fix shape:** introduce `_quabo_reachability_report(...) -> list[QuaboProbeResult]` that returns structured per-Quabo results (no exceptions, no lenient flag). Refactor `_check_quabo_reachability` to call the report helper and decide raise/warn from the result. `status._sweep_summary` calls the report helper directly and renders `OK` / `DEGRADED (N/M reachable)` / `DOWN (0/M reachable)`.

### D-5. Why output is doubled

**Files:** `grpc/src/panoseti_grpc/telemetry/logger.py`, `control/src/control/tools/interleave.py`, `control/src/control/utils/panoseti_interface.py`, `control/src/control/daemons/storeInfluxDB.py`

Two cooperating defects:

1. `PanosetiLogFactory.configure_logger` (`logger.py:225-279`) attaches a `RichHandler` to a named logger but **never sets `logger.propagate = False`**. Records propagate to the root logger.
2. `interleave.py:37` calls `logging.basicConfig(level=INFO, handlers=[RichHandler(...)])` at **module import time**, installing a `RichHandler` on the **root** logger. `stop.py:39` does `from control.tools.interleave import PID_FILE` at module top, so importing `stop` triggers it; `start.py:48` imports `stop`, so importing `start` triggers it transitively. (Same pattern in `panoseti_interface.py:27` and `daemons/storeInfluxDB.py:78`.)

Net result: every record from `PSETI.Start` / `PSETI.Stop` fires on its named RichHandler **and** propagates to the root RichHandler installed by `interleave.basicConfig`. Two prints. `pseti config` and `quabo_driver` do not import `interleave` (or `stop`/`start`), so they only have one handler — matching the operator's observation.

Console format: `RichHandler` is constructed without a custom formatter, so the logger name (`%(name)s`) is not surfaced; lines have no `[start]`/`[stop]`/`[transfer_daemon]` tag.

**Fix shape:**
- (a) Submodule edit: `logger.py:226-228` set `logger.propagate = False` and make `configure_logger` idempotent (no-op if a `RichHandler` is already attached). Console handler: build a `logging.Formatter(fmt="[%(name)s] %(message)s")` and attach it via `handler.setFormatter(...)` so the service tag appears.
- (b) Remove the module-level `logging.basicConfig` calls from `interleave.py`, `panoseti_interface.py`, `storeInfluxDB.py`. Any logger needs in those files use `get_logger(...)`.
- (c) Pytest impact: `caplog` requires propagation to capture. Add a `conftest.py` autouse fixture that flips `logger.propagate = True` for the duration of any test that uses `caplog`, then restores. (Or: tests use `propagate=True` only when needed; default stays off.)

### D-6. Why the CLI feels heavy / aliases are missing

**File:** `control/src/control/pseti.py`, `control/src/control/tools/obs_cli.py`, `control/src/control/config.py`, `control/src/control/transfer/cli.py`, `control/src/ci/test_cli.py`

- Boot path is already lazy via `BaseLazyGroup` (`grpc/src/panoseti_grpc/util/cli.py:106-120`) — `--help` returns a stub `click.Command` without importing the subcommand. Confirmed: no top-level `get_*_config()` calls leak.
- One leak remains: `config.py:47-49` runs `PanoPaths.logs_dir().mkdir()` and `get_logger(..., grpc_enabled=True)` at **import** time. Move to a Typer callback so they only fire on actual `pseti config` / `pseti cfg` invocation.
- Aliases: `pseti.py` already aliases `start/stop/status` at line 28-30. Add `cfg`. `obs_cli.py` already lists `ledger`; add a short alias under the same `lazy_mapping`.
- `transfer/cli.py` has no `--watch` flag. Pattern exists in `status.py:193-194,218-219`. `transfer/progress.py` already has `parse_rsync_progress(line)` for `rsync --info=progress2` output — exactly what a progress bar needs. `rich` is already a project dep.
- `pseti test hw clean` (`ci/test_cli.py:495-525`) shells out `compose … down -v` (destroys volumes). A non-destructive sibling `hw down` is the same code path minus `-v`.

---

## Phase 1 — Transactions: tracebacks & ledger truth

### 1.1 Unwrap `ExceptionGroup` at every `TaskGroup` site in `start.py`

**File:** `control/src/control/start.py`

(a) `_check_quabo_reachability` (`start.py:729-760`): mirror `_check_daq_reachability` (`start.py:777-785`). Catch `ExceptionGroup`, iterate `eg.exceptions`, render each with `traceback.format_exception(type(exc), exc, exc.__traceback__)`, log every sub-exception at ERROR, then raise a single `ValidationError` whose message is a multi-line summary listing each failing Quabo. Edge cases:
- Use `BaseExceptionGroup` (not `ExceptionGroup`) so `KeyboardInterrupt` / `asyncio.CancelledError` in the group are surfaced and re-raised, never suppressed.
- Nested groups: prefer `traceback.format_exception(...)` which handles arbitrarily nested groups in 3.11+. Do **not** hand-roll a recursive walker.

(b) `_check_no_remote_hashpipe` (`start.py:872-879`): currently re-raises only the **first** `ValidationError` from `eg.exceptions`. Change to log every sub-exception with the same template, then re-raise.

(c) `StartTransaction.__aexit__` (`start.py:115-120` and `start.py:217-221`):
- The console line that operators see (`logger.error(f"[CRITICAL FAILURE] Start process aborted: {exc_val}")`) must use `"\n".join(traceback.format_exception(type(exc_val), exc_val, exc_tb))` so the operator gets the whole tree.
- The JSON dump at `start.py:217-221`: replace `traceback.format_tb(exc_tb)` with `traceback.format_exception(type(exc_val), exc_val, exc_tb)` so the failure context file contains the sub-exception tracebacks.

**Edge cases to avoid:**
- Do not blanket-catch and re-raise `BaseExceptionGroup` everywhere. Only at the `TaskGroup` boundaries that actually need it. Other call sites must continue to propagate normally.
- Do not collapse multiple distinct sub-exceptions into a single string before logging — log each on its own ERROR line so they show up as separate Loki events.

### 1.2 Stop respects the ledger

**File:** `control/src/control/stop.py`

In the StopTransaction pre-flight (around `stop.py:624-641`):

```
ledger = state_mgr.load_state()
if ledger is None:
    if not force_cleanup:
        raise ValidationError("No active ledger; nothing to stop.")
    # --force-cleanup → continue and try best-effort hardware teardown
elif ledger.status not in {"STARTING", "ACTIVE", "STOPPING"}:
    if not force_cleanup:
        raise ValidationError(
            f"Ledger says run '{ledger.run_name}' is in '{ledger.status}'; "
            "nothing to stop. Use --force-cleanup to run the full ladder anyway."
        )
```

Replace the manual `load → mutate → save_state` at `stop.py:640-642` with `state_mgr.transition("STOPPING")`.

**Edge cases:**
- Idempotency: `STOPPING` must remain in the stoppable set so a re-entered stop can complete cleanup.
- `--force-cleanup` path must work even if `ledger is None` (e.g., ledger file deleted manually).
- Do **not** use the legacy `get_current_run_name()` helper here — call `load_state()` directly so we have access to `ledger.status`.

---

## Phase 2 — Transfer daemon → ledger sync

**File:** `control/src/control/transfer/daemon.py`

### 2.1 Mirror queue-job mutations onto the ledger

Every site that mutates the TransferJob attempts/error must also call `state_mgr.transition(...)` with the same fields. Use the existing kwargs API at `run_state.py:217-229` — no schema change.

(a) Claim-time bump (`daemon.py:419-423`): immediately after `_write_job(active_job_path, bumped_job)`, call:
```
_safe_ledger_update(state_mgr, status="TRANSFERRING", transfer_attempts=bumped_attempts)
```

(b) Retry branch (`daemon.py:450-454`):
```
_safe_ledger_update(
    state_mgr,
    status="TRANSFERRING",                # still in flight
    transfer_attempts=bumped_attempts,
    last_transfer_error=error_msg,
    next_action_not_before=now + RETRY_DELAYS[bumped_attempts - 1],
)
```

(c) MAX_ATTEMPTS terminal (`daemon.py:444-449`):
```
_safe_ledger_update(
    state_mgr,
    status="TRANSFER_FAILED",
    transfer_attempts=bumped_attempts,
    last_transfer_error=error_msg,
)
```

(d) Inside `_process_job`, every existing `state_mgr.transition("TRANSFER_FAILED")` and `state_mgr.transition("VERIFY_FAILED")` (`daemon.py:189, 213, 239, 306, 326`) must pass `last_transfer_error=<reason>` so the ledger carries the actionable string.

### 2.2 Add a tiny safe wrapper

Define `_safe_ledger_update(state_mgr, *, status, **fields)` near the top of `daemon.py`:
```
def _safe_ledger_update(state_mgr, *, status, **fields):
    try:
        state_mgr.transition(status, **fields)
    except Exception as exc:
        logger.warning("Ledger update failed (non-fatal): %s", exc)
```
**Why:** the daemon's Phase-1 invariant is "no exception escapes the loop." A partial-write or missing ledger must not crash the daemon mid-job. Log and continue.

### 2.3 Manifest failure visibility

The user reports manifest generation is failing without context. The `_process_job` `MANIFEST_GENERATING` branch at `daemon.py:172-213` already builds an `ExceptionGroup` from the per-node `TaskGroup`. Mirror the unwrap pattern from Phase 1.1.a here so every per-node manifest failure (with traceback) is logged at ERROR and joined into the `last_transfer_error` string written to the ledger and the queue TOML. Do not attempt to *fix* the underlying manifest bug in this pass — surfacing it is the deliverable.

**Edge cases:**
- `state_mgr.transition` returns `None` if the ledger is missing; treat as success in `_safe_ledger_update`.
- Do not increment `transfer_attempts` more than once per claim. The bump happens at claim time; retry/terminal sites read the bumped value, not increment further.

---

## Phase 3 — Logging hygiene

### 3.1 Stop the propagation cascade

**File:** `panoseti_grpc` submodule — `grpc/src/panoseti_grpc/telemetry/logger.py:225-279`

In `PanosetiLogFactory.configure_logger`:
- Set `logger.propagate = False` after handlers are attached.
- Make idempotent: at function entry, check `any(isinstance(h, RichHandler) for h in logger.handlers)` — if true, return the existing logger unmodified. Prevents duplicate handlers when `get_logger(name)` is called more than once for the same name.

**Submodule pin:** the coding agent must bump `panoseti_grpc` after this change lands and update the `grpc/` submodule pin in `panoseti-software/control`.

### 3.2 Remove offending root-handler installations

Delete the module-level `logging.basicConfig(...)` calls in:
- `control/src/control/tools/interleave.py:37`
- `control/src/control/utils/panoseti_interface.py:27`
- `control/src/control/daemons/storeInfluxDB.py:78`

Replace with `logger = get_logger("<service>", log_dir=PanoPaths.logs_dir())` if the file actually emits logs (each does — keep the logger, just route through `get_logger`).

**Edge case:** `interleave.py` and `panoseti_interface.py` are imported transitively by many CLI commands. Confirm via `pytest -k logging` that no test was relying on the rogue root handler.

### 3.3 Add service tag to console output

In `logger.py` (same edit as 3.1), set a formatter on the `RichHandler`:
```
handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
```
RichHandler still renders timestamp/level/markup; the formatter only controls the message body. Operator console becomes:
```
[14:02:11] INFO  [PSETI.Start] Acquiring control lock...
```

### 3.4 Pytest caplog compatibility

**File (new or extended):** `control/src/ci/conftest.py`

Add an autouse fixture **scoped to tests that request `caplog`**:
```
@pytest.fixture
def _enable_log_propagation_for_caplog(caplog):
    """caplog attaches to root; flip propagation for the test only."""
    saved = {}
    for name in ("PSETI.Start", "PSETI.Stop", "PSETI.Status",
                 "transfer_daemon", "PSETI.Config"):
        lg = logging.getLogger(name)
        saved[name] = lg.propagate
        lg.propagate = True
    yield
    for name, p in saved.items():
        logging.getLogger(name).propagate = p
```
Bind it via the `caplog` fixture so existing tests keep working without per-test changes.

**Edge case:** the fixture must restore propagation in `finally` even if the test raises.

---

## Phase 4 — Status truthfulness

### 4.1 Refactor reachability into a structured report

**File:** `control/src/control/start.py` (or `control/src/control/utils/preflight.py` if it exists; if not, leave the helper in `start.py`).

Introduce a typed result:
```
@dataclass(frozen=True)
class QuaboProbeResult:
    uid: str
    ip: str
    port: int
    reachable: bool
    error: str | None
```

New helper `async def _quabo_reachability_report(quabo_uids, network_config) -> list[QuaboProbeResult]` — performs the same `TaskGroup` fan-out as today but **does not raise** on per-Quabo failure; collects results.

Refactor `_check_quabo_reachability(quabo_uids, network_config, *, lenient)` to call `_quabo_reachability_report` and:
- if all reachable → return.
- else if lenient → log one WARNING per unreachable Quabo, return.
- else → raise `ValidationError` whose message lists every unreachable Quabo (multi-line).

### 4.2 Wire status to the report helper

**File:** `control/src/control/status.py`

In `_sweep_summary` (`status.py:153-160`), call `_quabo_reachability_report(...)` directly. Render:
- All reachable → `Quabos:    OK    — N/N reachable`.
- Some reachable → `Quabos:    DEGRADED — M/N reachable; down: <list>`.
- None reachable → `Quabos:    DOWN  — 0/N reachable`.

Same change applies anywhere else that misuses `lenient=True` to mean "tell me the truth": grep `status.py` for `lenient=True` and replace.

**Edge cases:**
- A Quabo that responds *partially* (one of cmd/HK ports up) should be reported with the failing port in the error string. Use the existing `_check_reachability` semantics.
- Do **not** change the `_check_*` callsites in `start.py` — strict/lenient semantics there are correct for pre-flight.

---

## Phase 5 — CLI UX & infra

### 5.1 `pseti cfg` alias

**File:** `control/src/control/pseti.py`

Add to `PanoLazyGroup.lazy_mapping` (around line 28-30):
```
"cfg": ("control.config", "app", "Alias for 'pseti config'."),
```
Append `"cfg"` to `command_order` (line 39).

### 5.2 Short alias for `pseti ledger`

**File:** `control/src/control/tools/obs_cli.py`

Add a sibling key in `lazy_mapping` pointing at the same target as `ledger`:
```
"led": ("control.tools.ledger_cli", "app", "Short alias for 'ledger'."),
```
Add to `command_order`. Keep `ledger` for discoverability; document both in `CLI.md`.

### 5.3 Defer logger init in `config.py`

**File:** `control/src/control/config.py:47-49`

Move the `PanoPaths.logs_dir().mkdir(...)` and `get_logger("PSETI.Config", ..., grpc_enabled=True)` calls out of module scope into a Typer callback (`@app.callback()`). Use a module-level `logger: logging.Logger | None = None` that the callback initializes. This is the only remaining import-time side effect on the hot path; tightening it makes `pseti --help` cold-cache faster.

**Edge case:** any function in `config.py` that uses `logger` must defensively `logger = logger or logging.getLogger(__name__)` for direct invocation paths (tests).

### 5.4 Transfer `--watch` and progress bar

**Files:** `control/src/control/transfer/cli.py`, `control/src/control/transfer/daemon.py`

(a) `--watch` flag on the existing `status` command — clone the loop pattern from `status.py:193-194,218-219` (`while True: render; os.system("clear"); time.sleep(interval)`). Default interval 5 s; flag `--interval`.

(b) Progress bar: use `rich.progress.Progress` (`rich` already a dep). The daemon's `transfer/daemon.py` already shells `rsync --info=progress2`; `transfer/progress.py::parse_rsync_progress` already parses it. Wire the daemon to:
1. Read rsync stdout line-by-line via `asyncio.create_subprocess_exec` and `async for line in proc.stdout`.
2. For each parsed `{bytes, pct, speed, eta}` dict, write the latest snapshot to a per-job sidecar file `state/transfer/queue/active/{run_name}.progress.json` (write to `tmp` and `os.replace`).
3. The CLI `pseti transfer status --watch` reads each active job's sidecar and renders a `Progress` task per active job (one row per node × run). Use `Progress.update(task_id, completed=bytes)` and a `BarColumn`, `TransferSpeedColumn`, `TimeRemainingColumn`.

**Edge cases:**
- Sidecar must tolerate partial writes (atomic `os.replace` from `tmp_path`).
- If the sidecar is missing (job hasn't reached rsync yet), show an indeterminate spinner, not an error.
- Don't block the daemon main loop on stdout reads — async streaming is mandatory.

### 5.5 `pseti test hw down` — non-destructive teardown

**File:** `control/src/ci/test_cli.py`

Clone `hw clean` (lines 495-525) into a new `hw down` command. Keep all SSH-tunnel / podman-context plumbing identical. The only diffs are:
- Drop `-v` from the three `compose … down -v` invocations (lines 504, 515, 523).
- Drop the data-wipe placeholder line 525.
- Help text: `"Stop containers but preserve volumes (use 'clean' for full wipe)."`

**Edge case:** the `compose … down` exit code is 0 on already-stopped stacks; preserve that — do not raise on non-zero unless the docker context itself is unreachable.

---

## Phase 6 — Happy-path automated verification

Each test below must **fail** on the current `test-refactor` branch before the corresponding fix lands and **pass** after. Co-locate tests with the existing tier structure.

### 6.1 Tier 2 — `start.py` ExceptionGroup unwrap

**Path:** `control/src/ci/tier2_logic/test_start_exceptiongroup_unwrap.py`

- Mock `_check_reachability` so two Quabos raise distinct exceptions (`ConnectionRefusedError("port closed")`, `TimeoutError("UDP")`).
- Call `_check_quabo_reachability(..., lenient=False)`; assert `ValidationError` is raised whose message contains **both** `"port closed"` and `"UDP"`.
- Use `caplog` to assert two ERROR records were emitted, each containing a traceback (search for `"Traceback (most recent call last):"`).
- Second test: drive `StartTransaction.__aexit__` with an `ExceptionGroup` and assert the JSON failure dump contains `"port closed"` and `"UDP"` strings.

### 6.2 Tier 2 — `stop.py` ledger guard

**Path:** `control/src/ci/tier2_logic/test_stop_ledger_guard.py`

- Pre-write a ledger with `status="RECORDING_ENDED"` and `run_name="r"`.
- Call `stop_run(force_cleanup=False)`; assert `ValidationError` raised with message matching `"is in 'RECORDING_ENDED'"`.
- Call `stop_run(force_cleanup=True)` against the same ledger; assert it proceeds (mock the hardware ladder steps to no-op).
- Pre-write `status="ACTIVE"`; assert `stop_run(force_cleanup=False)` proceeds normally.
- `ledger=None` case: `stop_run(force_cleanup=False)` must raise; `force_cleanup=True` must proceed.

### 6.3 Tier 2 — Transfer daemon → ledger sync

**Path:** `control/src/ci/tier2_logic/test_transfer_daemon_ledger_sync.py`

Reuse the in-process daemon harness from `tier4_chaos/test_transfer_daemon_crash_recovery.py`.

- Pre-write a `RunStateLedger` with `status="RECORDING_ENDED", run_name="r1"`. Enqueue a job for `r1`.
- Patch `_process_job` to return `(False, "rsync_blackbox_error")`. Patch `RETRY_DELAYS=[0.01,0.01]`.
- Run the daemon until the job lands in `failed/`.
- Assert the on-disk ledger has:
  - `status == "TRANSFER_FAILED"`
  - `transfer_attempts == MAX_ATTEMPTS`
  - `last_transfer_error == "rsync_blackbox_error"`
- Variant: succeed-on-second-try; assert intermediate ledger snapshot has `transfer_attempts==1, last_transfer_error="<first error>"` before final success transitions ledger to `ARCHIVED`.

### 6.4 Tier 1 — Logger propagation & idempotency

**Path:** `control/src/ci/tier1_unit/test_logger_propagation.py`

- `get_logger("X.Y")`; assert `logger.propagate is False` and exactly one `RichHandler`.
- Call `get_logger("X.Y")` again; assert handler count is **still 1** (idempotent).
- Assert formatter on the RichHandler emits `"[X.Y] hello"` for a `logger.info("hello")` call (use `caplog` after enabling propagation).
- Assert `logging.getLogger().handlers` contains no `RichHandler` (no root handler installed by importing `start`/`stop`).

### 6.5 Tier 2 — Status reports DEGRADED when Quabos down

**Path:** `control/src/ci/tier2_logic/test_status_quabo_report.py`

- Mock `_check_reachability`: 2/3 Quabos reachable, 1 unreachable.
- Run `_sweep_summary()` synchronously (or via `asyncio.run`).
- Assert returned lines contain `"DEGRADED"` and `"2/3 reachable"`.
- Variant: 0/3 → `"DOWN"`; 3/3 → `"OK"`.

### 6.6 Tier 1 — CLI alias loading

**Path:** `control/src/ci/tier1_unit/test_cli_aliases.py`

- Use Typer's `runner.invoke(app, ["cfg", "--help"])`; assert exit code 0 and the output contains the same help text as `["obs", "config", "--help"]`.
- Same for `["obs", "led", "--help"]` vs `["obs", "ledger", "--help"]`.
- Boot-time check: import `control.pseti` and assert `logging.getLogger().handlers` does not include a `RichHandler` (regression guard for D-5).

### 6.7 Tier 5 — Transfer `--watch` smoke

**Path:** `control/src/ci/tier5_integration/test_transfer_watch.py`

- Start the daemon; enqueue a job that succeeds via mocked rsync writing a fake progress sidecar.
- Run `pseti transfer status --watch --interval 0.5` as a subprocess for 2 s; capture stdout; assert at least 2 frames were rendered and at least one frame contains a `%` progress token.
- Skip when `RUN_REAL_DATA_TESTS` is unset (consistent with other Tier-5 conventions).

### 6.8 Verification command sequence

After implementation, the agent runs in order; each must pass before the next:

```
pseti test lint                                                  # ruff + mypy
pseti test sw unit -k 'logger_propagation or cli_aliases'        # 6.4, 6.6
pseti test sw logic -k 'exceptiongroup_unwrap or stop_ledger_guard or daemon_ledger_sync or status_quabo_report'  # 6.1-6.3, 6.5
pseti test sw chaos                                              # full chaos (regression guard)
pseti test sw integration -k 'transfer_watch or transfer_observability'  # 6.7 + Phase-1 regression
pseti test sw integration                                        # full integration
```

---

## Files To Modify (summary)

| Path | Phase | What |
|---|---|---|
| `control/src/control/start.py` | 1.1, 4.1 | Unwrap ExceptionGroup at TaskGroup sites; structured Quabo report helper. |
| `control/src/control/stop.py` | 1.2 | Pre-flight ledger-status guard; switch to `state_mgr.transition`. |
| `control/src/control/transfer/daemon.py` | 2.1, 2.2, 2.3, 5.4 | Ledger mirroring via `_safe_ledger_update`; manifest TaskGroup unwrap; rsync progress sidecar. |
| `grpc/src/panoseti_grpc/telemetry/logger.py` | 3.1, 3.3 | `propagate=False`, idempotent configure, `[%(name)s]` formatter. (Submodule.) |
| `control/src/control/tools/interleave.py` | 3.2 | Drop `logging.basicConfig`; use `get_logger`. |
| `control/src/control/utils/panoseti_interface.py` | 3.2 | Same. |
| `control/src/control/daemons/storeInfluxDB.py` | 3.2 | Same. |
| `control/src/ci/conftest.py` | 3.4 | caplog-propagation fixture. |
| `control/src/control/status.py` | 4.2 | Wire `_sweep_summary` to report helper; OK / DEGRADED / DOWN rendering. |
| `control/src/control/pseti.py` | 5.1 | Add `cfg` alias. |
| `control/src/control/tools/obs_cli.py` | 5.2 | Add `led` alias for `ledger`. |
| `control/src/control/config.py` | 5.3 | Defer logger init from import scope to Typer callback. |
| `control/src/control/transfer/cli.py` | 5.4 | `--watch` and `rich.progress` rendering from sidecar. |
| `control/src/ci/test_cli.py` | 5.5 | New `hw down` (non-destructive teardown). |
| `control/src/ci/tier1_unit/test_logger_propagation.py` | 6.4 | New. |
| `control/src/ci/tier1_unit/test_cli_aliases.py` | 6.6 | New. |
| `control/src/ci/tier2_logic/test_start_exceptiongroup_unwrap.py` | 6.1 | New. |
| `control/src/ci/tier2_logic/test_stop_ledger_guard.py` | 6.2 | New. |
| `control/src/ci/tier2_logic/test_transfer_daemon_ledger_sync.py` | 6.3 | New. |
| `control/src/ci/tier2_logic/test_status_quabo_report.py` | 6.5 | New. |
| `control/src/ci/tier5_integration/test_transfer_watch.py` | 6.7 | New. |
| `control/CLI.md` | 5.1, 5.2, 5.5 | Document `cfg`, `led`, `hw down`. |
| `control/TRANSACTIONS.md` | 1.2, 2.1 | Document stop/ledger guard and ledger-attempts mirroring. |

---

## Out of scope (explicitly)

- Fixing the underlying manifest-generation bug observed in HITL — Phase 2.3 surfaces it but does not fix the rsync/path issue. File a follow-up ticket once the traceback is visible.
- A general "pseti ledger edit" surface (per prior plan: inspection only).
- Renaming the overloaded `head_node_container` config field.
- Replacing the legacy gRPC `Log` RPC during the Alloy shadow period.
- Disk-fill prevention in `pseti start`.
- A native Typer alias mechanism (continue using the lazy-mapping duplicate-key pattern).

---

## Definition of done

1. All commands in §6.8 are green on a clean checkout of the resulting branch.
2. `pseti start` against unreachable Quabos prints **every** sub-exception's traceback to console and writes them to the failure-context JSON.
3. `pseti stop` against a `RECORDING_ENDED` ledger refuses without `--force-cleanup`.
4. After a transfer-daemon failure burst, `cat state/runs/ledger.toml` shows `transfer_attempts == MAX_ATTEMPTS` and a non-empty `last_transfer_error`.
5. `pseti status sweep` against one powered-off Quabo prints `Quabos: DEGRADED — N/M reachable`.
6. Every console line carries a `[service]` tag; no double output for any command.
7. `pseti cfg --help` and `pseti led --help` both work; `pseti test hw down` stops containers but `docker volume ls` still shows the data volumes.
