# PANOSETI Control Plane — Refactor Audit & Execution Plan

**Status:** Planning phase — no code to be written by the reviewer.
**Role:** Principal Software Architect / QA Lead review of the recent transactional refactor.
**Inputs:** `start.py`, `stop.py`, `utils/config_file.py`, `utils/pydantic_config_models.py`, `utils/run_state.py`, `utils/collect.py`, `utils/util.py`, `ci/integration/scenarios/`, `docs/plans/refactor-control-plane-critical-path.md`.

---

## Context

A previous execution agent migrated the PANOSETI control plane from dict-typed configs and non-transactional startup/teardown to a Pydantic-typed, rollback-capable system with a TOML state ledger and advisory locking. The work is nominally complete but was not verified end-to-end. This plan audits the delivery, identifies the residual gaps that threaten the transactional guarantees, and hands a downstream execution pipeline to coding agents. The chaos test suite in `ci/integration/scenarios/` is the acceptance harness — many scenarios are currently stubbed or skipped and must be wired up to drive the remaining production fixes via TDD.

---

## Findings summary (one line each)

1. **Pydantic migration is partial.** Models are built at the entry point but immediately degraded to `dict[str, Any]` at every external call boundary via `.model_dump()`, defeating the point of the migration.
2. **State ledger writes are non-atomic.** `RunStateManager.save_state` uses a plain `open(...).write()`; a SIGKILL mid-write corrupts the ledger. There is no TOML string-escaping, so any metadata with quotes/newlines breaks the file.
3. **Advisory locking is correct at the kernel layer but brittle at the application layer.** Ledger can be left in `ACTIVE` after a SIGKILL; next `start.py` refuses to run with no self-heal path.
4. **Rollback ladder misses three critical cases.** `SIGINT` is not trapped; the `_aborted/` archive path is not collision-safe; node-level `start_recording` receipts are only written on heartbeat success, so nodes that accepted StartDaq but failed heartbeat leave no trail.
5. **`collect_data` has no retry** — contradicts §4 of the refactor plan. A single transient rsync failure aborts collection, prevents `collect_complete`, and blocks cleanup.
6. **Shell injection in `collect.py` and `start.py`.** `os.system(f'mv ... {run_dir} ...')` with string-interpolated paths; `ssh` commands use f-string interpolation of node fields.
7. **Scenario suite has ~47 stubs / skips.** The tests that exist for the six TDD exemplars fail red for the right reasons, but the distributed, data-integrity, and telemetry matrices are stubs.

---

## Execution pipeline (dependency graph)

Tasks are grouped into four waves. Each wave must be green before the next begins. Agent assignments follow the user's model roster (Gemini Pro 3.1 / Flash 3.1 for large-context mechanical work; Sonnet 4.6 for subtle transactional and test logic).

### Wave 1 — Typing & API boundary hygiene (unblocks everything else)

| # | Task | Critical files | Agent | Depends on |
|---|---|---|---|---|
| 1.1 | Change `util.daq_grpc_endpoint` to accept `DaqNodeValidator`; remove every `node.model_dump()` at call sites in `start.py`, `stop.py`, `collect.py`. | `utils/util.py:136`, `start.py:384,416,548`, `stop.py:250,375` | Gemini Pro 3.1 | — |
| 1.2 | Change `util.write_run_name`, `util.start_hk_recorder`, `file_xfer.copy_config_files`, `file_xfer.copy_dir_from_node` to accept validators, not dicts. | `utils/util.py:416`, `utils/file_xfer.py`, `start.py:348,372,592` | Gemini Pro 3.1 | — |
| 1.3 | Change `stop.py::stop_data_flow` and `util.stop_data_flow` to accept `QuaboUidsValidator` only (not dict); remove `dome['modules']` indexing at `stop.py:211-228`. | `stop.py:200`, `utils/util.py:707` | Gemini Pro 3.1 | — |
| 1.4 | Remove `DaqConfigValidator \| dict[str, Any]` unions from signatures. Validation happens once at `config_file.get_*`; every downstream function is strict. | `utils/collect.py:22,69`, `utils/config_file.py:157,190`, `utils/util.py:735,760` | Sonnet 4.6 | 1.1–1.3 |
| 1.5 | Treat `PortForwarding` as a first-class attribute (already in `DaqNodeValidator`). Delete `model_dump() + 'port_forwarding' in node_dict` checks in `start.py:334-340` and `utils/collect.py:99-103`. | `utils/pydantic_config_models.py:276`, `start.py:334`, `utils/collect.py:99` | Gemini Flash 3.1 | 1.4 |
| 1.6 | Run `python ci/qa.py lint`; resolve every MyPy error. No new `type: ignore`. | repo-wide | Sonnet 4.6 | 1.1–1.5 |

### Wave 2 — Transactional hardening

| # | Task | Critical files | Agent | Depends on |
|---|---|---|---|---|
| 2.1 | **Atomic ledger writes.** Replace `open(path, "w").write()` with `tempfile.NamedTemporaryFile + os.replace()`. Escape string values or adopt a vetted TOML writer (e.g. `tomli_w`). | `utils/run_state.py:63-97` | Sonnet 4.6 | Wave 1 |
| 2.2 | **Stale ledger self-heal.** Add `pid` and `host` to `RunStateLedger`. On `start.py` entry, if status is `STARTING/ACTIVE/STOPPING` but `pid` is dead on this host, archive the stale ledger to `_aborted/` and proceed with a fresh run. Expose `--force-reset`. | `utils/pydantic_config_models.py:424`, `utils/run_state.py`, `start.py:482-486` | Sonnet 4.6 | 2.1 |
| 2.3 | **SIGINT / SIGTERM handler.** Install a handler in `start.py` and `stop.py` that sets a cancellation flag; the main task awaits the flag inside the rollback `try` and triggers the ladder. Guarantee reentrant-safety and default-disposition restore. | `start.py:465`, `stop.py:497` | Sonnet 4.6 | 2.1 |
| 2.4 | **Heartbeat probe loop.** Replace `asyncio.sleep(2.0)` + single StatusDaq with a retry loop (≤ 5 attempts × 1 s back-off). Distinguish "hashpipe never started" from "hashpipe crashed after start." | `start.py:408-441` | Sonnet 4.6 | 2.1 |
| 2.5 | **Per-node receipts on StartDaq success.** Write a `STARTING` receipt immediately after a successful `StartDaq` RPC (before heartbeat); upgrade to `START_SUCCESS` after heartbeat. Rollback consults the ledger to know which nodes need StopDaq. | `start.py:381-406`, `utils/run_state.py:104` | Sonnet 4.6 | 2.4 |
| 2.6 | **Collision-safe `_aborted/` archive.** If `<head_node_data_dir>/_aborted/<run_name>/` exists, append a monotonic suffix. Never `shutil.move` into an existing directory. | `start.py:567-572` | Gemini Flash 3.1 | 2.1 |
| 2.7 | **Retry `collect_data`.** Transient rsync failures (exit codes 12, 23, 30, 255) retry ≤ 3× with 5 s back-off. Distinguish transient from terminal (disk full, permission). Return a structured `CollectResult` (Pydantic) instead of a concatenated error string. | `utils/collect.py:22`, `stop.py:469` | Sonnet 4.6 | Wave 1 |
| 2.8 | **Eliminate `os.system` shell interpolation.** Replace `os.system(f'mv ... {run_dir}/*')` and `os.system(f'ssh {user}@{ip} "{rcmd}"')` with `subprocess.run([...])` list form. | `utils/collect.py:49,52,91,101-106`, `start.py:312,343` | Gemini Pro 3.1 | Wave 1 |
| 2.9 | **Concurrency-safe `update_node_receipt`.** Serialize through an `asyncio.Lock` or a single writer task consuming a queue. Current load→mutate→save races under concurrent `probe_node` calls. | `utils/run_state.py:104`, `start.py:437` | Sonnet 4.6 | 2.1 |

### Wave 3 — Chaos test implementation (TDD harness)

Each ticket below corresponds to a skipped or stub test. Every test must fail red on `master` for a documented reason and go green after the Wave 2 fix. Agent choice: Sonnet 4.6 for logic, Gemini Flash 3.1 for fixture plumbing.

| # | Ticket | Test file | Production code it forces |
|---|---|---|---|
| 3.1 | SC-010 family (a, b, c) — orphaned hashpipe, live-hashpipe force-rejection, forced-cleanup incident key. | `test_sc_grpc_failures.py` | server-side liveness check; `force` proto field; Redis incident key |
| 3.2 | SC-002 — partial StartDaq rollback with `_aborted/` snapshot. | `test_sc_transactional_state.py` | Wave 2.5, 2.6 |
| 3.3 | SC-006 — StopDaq continues after first-node failure. | `test_sc_grpc_failures.py` | pin current behavior in `stop.py::stop_recording` |
| 3.4 | SC-024 — concurrent start advisory lock. | `test_sc_transactional_state.py` | Wave 2.1 + existing `acquire_lock` |
| 3.5 | SC-027 — `stop.py --run X` with mismatching ledger refuses unless `--force-cleanup`. | `test_sc_transactional_state.py` | `stop.py:433-436` |
| 3.6 | SC-031 — PH baseline 24-hour vs 24-day. | `test_sc_transactional_state.py` | pin `start.py:132` |
| 3.7 | SC-033 / SC-034 — stale interleave PID; SIGKILL escalation + MAROC reset. | `test_sc_transactional_state.py` | pin `stop.py:57-125` |
| 3.8 | SC-069 / SC-070 / SC-071 — 3-node partial start/stop; sequential vs parallel latency. | `test_sc_distributed.py` | requires `daqnode_fleet(n=3)` fixture |
| 3.9 | SC-041 / SC-042 / SC-043 / SC-049b — PFF gap detection, OOO packet tolerance, truncated header recovery, fixed-frame-size invariant across rollover. | `test_sc_data_integrity.py` | PFF writer audit |
| 3.10 | SC-056 / SC-057 / SC-067 — Loki down, Redis `maxmemory`, RedisBatcher flush-loss. | `test_sc_telemetry.py` | telemetry spool/backpressure |
| 3.11 | SC-012 — full-disk head node; collect fails, cleanup refused. | `test_sc_grpc_failures.py` | Wave 2.7 |
| 3.12 | SC-015 — daqnode reboots during recording, ledger self-heals. | new test | Wave 2.2 |
| 3.13 | SC-N003 — 4-node start, kill node 2 mid-flight; rollback covers nodes 0-1, node 3 never started. | `test_sc_distributed.py` | Wave 2.5 + fleet fixture |

Full specifications are in **Appendix: Test Design Logic**.

### Wave 4 — Regression pinning & docs

| # | Task | Agent |
|---|---|---|
| 4.1 | Pin every now-green SC-### test in `ci/qa.py chaos`. Promote chaos to the default merge gate once Wave 3 ships. | Gemini Flash 3.1 |
| 4.2 | Document ledger lifecycle and rollback semantics in `control/CLAUDE.md`. | Gemini Flash 3.1 |
| 4.3 | Operator runbook: "What to do when `tmp/run_state.toml` says ACTIVE but nothing is running." | Gemini Flash 3.1 |

---

## Critical files (reference list)

- `control/start.py` — transactional start coordinator
- `control/stop.py` — best-effort shutdown
- `control/utils/run_state.py` — state ledger + advisory lock
- `control/utils/collect.py` — data collection
- `control/utils/util.py` — mixed-typed helpers (main source of dict leaks)
- `control/utils/pydantic_config_models.py` — model definitions (needs `pid/host` on ledger)
- `control/utils/config_file.py` — loader entry points
- `control/ci/integration/scenarios/` — chaos test suite (47 stubs to close)
- `control/ci/integration/scenarios/conftest.py` — `_start/_stop/_cleanup` normalizers already in place

---

## Appendix: Test Design Logic for Skipped Scenarios

### SC-069 — 3-node partial start, rollback on node 2 failure

- **Setup:** `daqnode_fleet(n=3)`; mock-quabo spans 3 modules.
- **Stimulus:** Inject UNAVAILABLE on node 2's `StartDaq` via `grpc_proxy`. Call `start_run`.
- **Assert:** `start_run` returns `False`; ledger `status=ABORTED`; no hashpipe on any node; every quabo's `data_packet_destination` cleared; `_aborted/<run_name>/start_failure_context.json` references node 2; ledger `nodes[]` records node 0 transitioning through `STARTING → rolled back` (snapshots during rollback, not just final).

### SC-070 — 3-node partial stop

- **Setup:** Fleet of 3; start successfully; inject timeout on `StopDaq` to node 1.
- **Stimulus:** `stop_run`.
- **Assert:** returns without raising; nodes 0 and 2 clean; ledger ends `COMPLETED` or `STOPPING` (partial); node 1 error recorded in `stop_errors`; nodes 0/2 were still contacted.

### SC-071 — Sequential vs parallel StartDaq latency

- **Setup:** Fleet of 6 (`RUN_LARGE_FLEET=1`); `netem delay 200ms` on node 5.
- **Stimulus:** time `start_recording`.
- **Assert:** `elapsed ≈ max(per_node_latency)`; sequential regressions fail the test.

### SC-072 — Rolling DAQ-node restart during active run

- **Setup:** 2-node fleet, active run, `docker restart daqnode-1` at t=5 s.
- **Stimulus:** wait 30 s, `stop_run`.
- **Assert:** `stop_run` completes without hang; rejoin logic (if Wave 5+) produces `RESUMED` receipt; minimum contract is "no deadlock on a dropped node."

### SC-073 — Socat gateway crash during port-forwarded command

- **Setup:** Active port-forwarded quabo command; `process_chaos.kill` on `ci-gateway`.
- **Assert:** typed exception names the offending quabo; not a generic timeout.

### SC-074 — Module moved between DAQ nodes between runs

- **Setup:** Run 1 with module 128 on daqnode-1; swap config; Run 2 with module 128 on daqnode-2.
- **Assert:** after Run 2, quabo 128.0's `data_packet_destination` points at daqnode-2; no stale destination from Run 1.

### SC-077 — BOARDLOC collision across domes

- **Setup:** `obs_config.json` with two domes sharing module IDs.
- **Stimulus:** `global_validator.validate_all()`.
- **Assert:** raises `DuplicateBoardlocError` naming the offending quabo pairs.

### SC-078 — Mixed port-forward + direct topology

- **Setup:** 3-node fleet; node 0 direct; nodes 1–2 via gateway.
- **Assert:** all 3 `StartDaq` RPCs succeed using the correct endpoints; no gateway RPC attempted for node 0.

### SC-080 — panoseti-server SIGHUP reload

- **Setup:** running server with active run; SIGHUP.
- **Assert:** streams not dropped; a non-runtime config field reloads; hashpipe not restarted.

### SC-041 / SC-042 — Packet loss / OOO injection

- **Setup:** `tcpreplay --loss 0.05` or `--shuffle`.
- **Assert:** PFF fixed-frame invariant preserved; per-module `pkt_num` gaps detected (metric or log). Red today: gaps swallowed silently.

### SC-049b — Fixed-frame-size invariant across rollover

- **Setup:** `max_file_size_mb=1`; write ~2 MB; stop.
- **Assert:** within a file, every JSON header length equals frame 0; between files, frame 0 lengths may differ but intra-file padding is consistent (mmap-stride read).

### SC-056 — Loki down during run

- **Setup:** stop `loki` mid-run for 30 s; restart.
- **Assert:** logs buffered and flushed OR `storeLoki.py` exits loudly; no silent loss.

### SC-067 — RedisBatcher flush-loss

- **Setup:** monkey-patch `flush` to crash after 50/100 messages.
- **Assert:** remaining 50 retried; queue depth returns to zero.

### SC-015 — Stale ledger self-heal

- **Setup:** Start a run, `docker restart` the head-node process.
- **Stimulus:** second `start.py`.
- **Assert:** stale ACTIVE ledger detected; archived to `_aborted/<run_name>/stale_ledger.toml`; new run proceeds. Forces Wave 2.2.

### SC-N003 — 4-node fleet, kill node 2 mid-start

- **Setup:** `daqnode_fleet(n=4)`; hook `kill_hashpipe` 500 ms after node 2's `StartDaq` returns.
- **Assert:** heartbeat on node 2 fails; nodes 0–1 rolled back via StopDaq; node 3 **never** issued `StartDaq`. Requires `start.py:406` to cancel in-flight gather tasks.

---

## Verification strategy

1. **Wave 1 gate:** `python ci/qa.py lint` passes with zero MyPy errors, zero new `type: ignore`.
2. **Wave 2 gate:** the six TDD exemplars (SC-010, SC-002, SC-006, SC-024, SC-031, SC-033) transition from red → green without touching tests.
3. **Wave 3 gate:** `python ci/qa.py chaos` passes for all non-stubbed scenarios; new scenarios land one PR per ticket with the red→green transition captured in CI logs.
4. **Wave 4 gate:** chaos suite enters the default merge gate; operator runbook merged; `control/CLAUDE.md` updated.
5. **Out of scope:** HITL (Pillar 2) and dynamic fleet (Pillar 3) stay deferred — this plan keeps mock-quabo + fixed 2-node topology as the CI substrate.
