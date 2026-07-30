# v2 Test Suite — Cleanup, Parity Completion, and v1 Sunset

## Context

A coding agent was handed off the task of porting v1 software-only tests
(`control/src/ci/software_only/`) into the new v2 architecture
(`control/src/ci/software_only_v2/`) which uses consolidated, robust fixtures:
`pseti_workspace`, `session_fleet`, `chaos_fleet`, and the `FleetSpec` DSL. The
intent was that test files declare topology and scenarios; they should NOT
re-implement workspace setup, env-var redirection, container boot, or fault
injection.

The agent's commit (`8d06b62`) made significant progress on Tier 2 (~11/12
files ported) but landed several anti-patterns: tests reimplement fixtures,
monkey-patch internals instead of using chaos handles, blanket-disable mypy,
and use vague names (`test_SC001_*`). Tier 3/4/5 still have large parity gaps
vs v1. Meanwhile, v1 itself is no longer stable due to docker-env changes, so
keeping it on life support is dragging the team down.

This plan delivers three things:
1. Remove anti-patterns and re-route tests to the canonical fixtures.
2. Fill the most important parity gaps and stabilize the failing tests.
3. Pragmatically sunset v1 (skip the 7-day soak; v1 is already broken).

It also includes guidance for Sonnet on naming/docstrings and a separate
follow-up section for the HW-SW entrypoint scripts (out of scope for this pass
since HW-SW tests are currently green).

---

## 1. Anti-patterns to fix in existing v2 tests

### 1a. Stop reimplementing fixtures
- `control/src/ci/software_only_v2/tier4_chaos/test_chaos_smoke.py:54-99` —
  defines a private `chaos_workspace` and `chaos_fleet` with manual
  `os.environ[...] = ...` and `importlib.reload(_cfm)`. **Replace with the
  canonical `chaos_fleet` fixture from `fixtures/chaos.py:325` and
  `pseti_workspace` from `fixtures/workspace.py`.** Delete the local fixtures.
- `tier2_logic/test_start_strict_mode.py:27-44, 96-188`,
  `test_start_exceptiongroup_unwrap.py:26-39, 87-114`,
  `test_stop_ledger_guard.py:21-27` — build `MagicMock` daq_configs with
  per-test `tmp_path` plumbing. **Replace with `pseti_workspace` parametrized
  on `FleetSpec.minimal_fleet()` (or `minimal_unit()`); read the real
  validated config from `pseti_workspace.topology`.**
- All newly-ported tier3 files
  (`test_two_node_direct.py:59,73`, `test_data_collection.py:66,91`,
  `test_concurrent_daq_operations.py:70,75-76,97`,
  `test_gateway_topology.py:86`) call `docker.from_env().containers.get(...)
  .exec_run(...)` and construct `DaqControlClient(host=..., port=...)`
  directly. **Replace with `fleet.daq_control_client(idx)` and add a single
  `fleet.exec_in_node(idx, cmd)` helper to `orchestrator/fleet.py` if
  unavoidable.**

### 1b. Replace monkey-patch chains with chaos handles
- `tier4_chaos/test_sc_grpc_failures.py:86-95, 144` patches 8+ attributes
  inside `control.start.*`. **Use `chaos_fleet.chaos.grpc.proxy(...)
  .set_mode(...)` (see `chaos.grpc` API) for fault injection. Keep mocks only
  where the goal is to prove that `start_run` rolls back on a *non-RPC*
  failure (e.g. validator); for RPC faults, drive the real client through the
  chaos proxy.**
- The `test_SC001_startdaq_timeout_triggers_rollback` failure is partly a
  symptom of this: the test patches `control.start.AsyncDaqControlClient`
  with a `side_effect` factory but the real call site uses
  `async with AsyncDaqControlClient(...)`, so `__aenter__` is never satisfied.
  Switching to `chaos.grpc.proxy(client).set_mode("StartDaq", "timeout", ...)`
  removes the need to mock the async context manager at all.

### 1c. Drop blanket type/lint suppression
- `# mypy: ignore-errors` headers in `test_two_node_direct.py:1`,
  `test_data_collection.py:1`, `test_concurrent_daq_operations.py:1`,
  `test_daq_lifecycle.py:1`, `test_transfer_basic.py:1`,
  `test_pseti_commands.py:1`, `test_grpc_cli.py:1`, `test_telemetry.py:1`.
  **Remove these. Fix typing properly. The project mandate
  (`control/CLAUDE.md`) bans blanket ignores.** Use `# type: ignore[code]` on
  specific lines only when justified with an inline comment.

### 1d. Extract duplicated helpers
- `_docker_available()` / `requires_docker` is duplicated across 9 test
  files. **Move to `tier3_fleet/conftest.py` and `tier4_chaos/conftest.py`
  (currently 1-byte stubs).**

### 1e. Demote tier1-grade tests
- `tier2_logic/test_telemetry.py::test_when_redis_full_then_backpressure_logged`
  only patches `redis.Redis.rpush` — pure unit test. **Move to `tier1_unit/`.**
- `tier2_logic/test_grpc_cli.py` — entirely CliRunner + mocks. **Move to
  `tier1_unit/` next to `test_pseti_cli.py`/`test_cli_aliases.py`.**
- `tier3_fleet/test_smoke.py::test_validate_all_rules_pass` (line 122) and
  `test_n_nodes_matches_topology` (line 145) duplicate tier1 validator/fleet
  spec coverage. **Delete; tier3 should only smoke-test container boot.**
- `tier2_logic/test_config_validation.py::test_minimal_unit_workspace_configs_validate`
  is also redundant with tier1 `test_fleet_spec.py` / `test_config_validator.py`.

### 1f. Fix the failing Tier 2 / Tier 3 tests (root causes already known)
- `tier2_logic/test_pseti_commands.py::test_pseti_val_graph` (lines 32-42):
  asserts on `pseti_workspace.root / "tmp"` but `pseti val graph` writes to
  `os.environ['PSETI_TMP']` (set by the fixture in
  `fixtures/workspace.py:124`). **Fix: assert against
  `Path(os.environ['PSETI_TMP']) / "topology.json"`.**
- `test_pseti_show_paths` (lines 57-65): Rich line-wrapping breaks
  substring match on `pseti_workspace.root.name`. **Fix: strip ANSI and
  collapse whitespace before assertion, or match a stable substring like
  `"PSETI_CONFIG"` only.**
- `test_pseti_show_config` (lines 67-75): hardcoded `"module ID 0"` —
  `FleetSpec.minimal_unit` does not guarantee module_id 0. **Fix: derive
  expected from `pseti_workspace.topology.daq_config.daq_nodes[0].module_ids[0]`.**
- `tier3_fleet/test_smoke.py::test_headnode_container_is_running`
  (lines 107-112): function-scoped `pseti_workspace` reused across class
  methods produces colliding `tc_id` in `Fleet.from_topology`
  (`orchestrator/fleet.py:136`). **Fix: append a per-test uuid suffix to
  `tc_id` in `Fleet.from_topology`, OR move these to `session_fleet`.**

### 1g. Wire the 3 unwired parity scenarios
`infra/parity.py` has 14 registered scenarios; 11 are called via
`run_scenario(...)` from test bodies. **Add `run_scenario("two_node_start_stop")`
to `tier3_fleet/test_two_node_direct.py`,
`run_scenario("grpc_inject_unavailable")` to one of the
`test_sc_grpc_failures.py` cases, and
`run_scenario("process_kill_and_restart")` to `tier4_chaos/test_chaos_smoke.py`.**

---

## 2. Missing work from the handoff to complete

### Tier 2 — 1 file remaining
- Port `software_only/tier2_logic/test_stale_ledger_healing.py` →
  `software_only_v2/tier2_logic/test_stale_ledger_healing.py`. Use
  `pseti_workspace` and the existing ledger primitives in `infra/workspace.py`.

### Tier 3 — 6 transfer files + start_collision_safety
Port these from `software_only/tier3_fleet/`, all using `session_fleet`:
- `test_transfer_daemon_e2e.py`
- `test_transfer_manifest_edge_cases.py`
- `test_transfer_manifest_integrity.py`
- `test_transfer_port_forwarding.py`
- `test_transfer_robustness.py`
- `test_start_collision_safety.py`

`test_transfer_queue_validity.py` was already absorbed into tier5; leave it
there.

### Tier 4 — large gap (focus on what stays in tier4)
**Decision (per user):** Tier 4 stays sim-only. Anything requiring a real
hashpipe PID moves to Tier 5.

Port these to v2 tier4 (RPC/process/network fault injection only):
- `test_sc_grpc_failures_{1,2,3}.py` — the existing v2 `test_sc_grpc_failures.py`
  has only SC001+SC006 of ~21 cases. Expand using `chaos.grpc.proxy`.
- `test_sc_config_validation.py` — pure validator chaos; no hashpipe needed.
- `test_lifecycle_chaos.py` and `test_start_remote_hashpipe_guard.py` — port
  the cases that don't depend on a real hashpipe PID.

Move these to v2 tier5 (require real hashpipe):
- `test_sc_transactional_state_{1-6}.py`
- `test_sc_data_integrity.py`
- `test_sc_distributed.py`
- `test_sc_transfer_daemon.py`
- `test_transfer_chaos.py`
- `test_transfer_daemon_crash_recovery.py`

The compose stack already has the real hashpipe binary
(`integration-daqnode`); use the existing `tier5_integration` conftest
patterns. Add a `chaos.proc` handle that operates against compose containers
if not already supported there.

### Tier 5 — 6 files missing
Port:
- `test_integration_loki_pipeline.py` (SC056-SC068)
- `test_integration_telemetry.py`
- `test_integration_real_data.py` (gated by `RUN_REAL_DATA_TESTS=1`)
- `test_integration_transfer_advanced.py`
- `test_transfer_watch.py`
- `test_transfer_observability.py`

---

## 3. Sunset v1 (pragmatic — delete now)

The user has explicitly chosen the pragmatic path: skip the 7-day soak, skip
the parity-coverage requirement. Once sections 1+2 above are done and v2 Tier
1+2 are green, execute SUNSET.md:

1. `git rm -r control/src/ci/software_only/`
2. `git rm` the 11 v1-only fixture modules listed in
   `software_only_v2/SUNSET.md` lines 18-67
   (`factories.py`, `rsync_fixtures.py`, `data_fixtures.py`,
   `chaos_fixtures.py`, `workspace_fixtures.py`, `transfer_fixtures.py`,
   `state_probe.py`, `client_fixtures.py`, `mocks.py`, `fleet.py`,
   `network_fixtures.py`)
3. `git rm -r control/src/ci/fixtures/chaos/` (v1 chaos package)
4. `git rm control/src/ci/qa.sw.toml`
5. Strip `pseti test sw <suite>` commands from
   `control/src/ci/test_cli.py:213-297` (sw_unit, sw_logic, sw_fleet,
   sw_chaos, sw_integration, sw_all, sw_build, sw_cleanup). Keep `pseti test
   sw2` as the only software-only entrypoint and remove the `sw v2` legacy
   alias too.
6. Update `control/TEST.md` and `control/CLAUDE.md` to drop v1 references.
7. Verify nothing else imports from `software_only/`:
   `grep -r "software_only/" control/ --include='*.py' --include='*.toml'`
8. **Keep forever** (per SUNSET.md lines 70-77):
   `topology_fixtures.py`, `fixtures/__init__.py`, `hardware_software/`,
   `shared/`, `fixtures/configs/`.

After deletion, run `pseti test sw2 unit logic fleet chaos integration` to
confirm v2 still green.

---

## 4. Guidance for Sonnet on test naming and docstrings

**Naming convention (apply to ported and existing v2 tests):**

- Use the `test_when_<setup>_then_<expected>` shape for behavioral tests:
  - Good: `test_when_logger_called_then_jsonl_output_is_valid`,
    `test_when_redis_full_then_backpressure_logged`,
    `test_concurrent_start_only_one_wins`
  - Bad (rename these): `test_pseti_val_basic`
    (`test_pseti_commands.py:24` → `test_when_pseti_val_runs_then_succeeds`),
    `test_pseti_grpc_help` (`test_grpc_cli.py:37`),
    `test_fleet_starts_and_is_healthy` (too generic),
    `test_collection_happy_path` (`test_data_collection.py:45`)

- Replace opaque SC codes with descriptive names; keep the SC code in the
  docstring for traceability:
  - Bad: `test_SC001_startdaq_timeout_triggers_rollback`
  - Good:
    ```python
    def test_startdaq_timeout_triggers_full_rollback(...):
        """SC001: when StartDaq RPC times out mid-startup, the rollback ladder
        unwinds quabos, daemons, and ledger state cleanly."""
    ```

- Add a one-line docstring to every test. Format:
  `"""<scenario>: <assertion>."""`. The 4 missing docstrings to fix first:
  `test_transfer_basic.py:134, 140, 149`.

- Class names should describe the subsystem under test:
  `TestFleetSmoke`, `TestStartTransactionRollback`, `TestTransferQueueState`.

- Avoid "happy_path" as a test name suffix; describe what makes the path
  happy. v1 used `happy_path` extensively; do not carry that over.

- Test bodies: use Arrange / Act / Assert with blank-line separators. Don't
  comment WHAT — comment WHY when non-obvious.

---

## 5. Out-of-scope follow-ups (HW-SW entrypoints)

Document these in a tracking issue; do **not** touch in this implementation
pass since HW-SW tests are passing.

- `control/src/ci/scripts/entrypoint.sh:13,17,21` — `2>/dev/null || true`
  silently swallows chown/usermod failures despite `set -e`.
- `entrypoint.sh:21` — blanket `chown -R panoseti:panoseti /app /grpc
  /opt/venv` on every container start. `Dockerfile.ci:42` already chowns at
  build via `--chown=panoseti:panoseti`, so the runtime recursion is only
  needed when `LOCAL_UID != 1000`. Narrow it to `/app /grpc` (skip
  `/opt/venv`).
- `entrypoint-daqnode.sh:12` — `rm -f "${DATA_DIR}/module.config"` runs
  unconditionally and can race with another daqnode on a shared volume
  (CLAUDE.md `/home/panoseti/panoseti/CLAUDE.md:324` warns about this).
  Add a per-run guard or move the cleanup into the daq_control server's
  StartDaq path.
- All entrypoints: add `set -uo pipefail` (currently only `set -e`).
- Fundamental fix (defer): replace runtime `usermod`/`gosu` juggling with
  `--build-arg UID=...` + `USER panoseti` directly in `Dockerfile.ci:35-36,
  44`. Out of scope for this pass.

---

## Critical files

**Anti-pattern fixes:**
- `control/src/ci/software_only_v2/tier4_chaos/test_chaos_smoke.py`
- `control/src/ci/software_only_v2/tier4_chaos/test_sc_grpc_failures.py`
- `control/src/ci/software_only_v2/tier2_logic/test_start_strict_mode.py`
- `control/src/ci/software_only_v2/tier2_logic/test_start_exceptiongroup_unwrap.py`
- `control/src/ci/software_only_v2/tier2_logic/test_stop_ledger_guard.py`
- `control/src/ci/software_only_v2/tier2_logic/test_pseti_commands.py`
- `control/src/ci/software_only_v2/tier3_fleet/test_two_node_direct.py`
- `control/src/ci/software_only_v2/tier3_fleet/test_data_collection.py`
- `control/src/ci/software_only_v2/tier3_fleet/test_concurrent_daq_operations.py`
- `control/src/ci/software_only_v2/tier3_fleet/test_gateway_topology.py`
- `control/src/ci/software_only_v2/tier3_fleet/test_smoke.py`
- `control/src/ci/software_only_v2/tier3_fleet/conftest.py`
- `control/src/ci/software_only_v2/tier4_chaos/conftest.py`
- `control/src/ci/software_only_v2/orchestrator/fleet.py` (add
  `exec_in_node` helper, fix `tc_id` collision)
- `control/src/ci/software_only_v2/infra/parity.py` (no changes;
  scenarios just need to be wired from test bodies)

**Fixtures to reuse (do NOT reimplement):**
- `control/src/ci/software_only_v2/fixtures/workspace.py` — `pseti_workspace`
- `control/src/ci/software_only_v2/fixtures/fleet.py` — `session_fleet`
- `control/src/ci/software_only_v2/fixtures/chaos.py:325` — `chaos_fleet`
- `control/src/ci/software_only_v2/infra/spec.py` — `FleetSpec` DSL
- `control/src/ci/software_only_v2/infra/parity.py` — `@parity_test`,
  `run_scenario`

**Sunset deletions:**
- `control/src/ci/software_only/` (entire tree)
- `control/src/ci/qa.sw.toml`
- `control/src/ci/fixtures/chaos/` (entire tree)
- 11 v1 fixture modules listed in SUNSET.md
- v1 commands in `control/src/ci/test_cli.py:213-297`

---

## Verification

After implementation:

1. **Anti-patterns gone:**
   - `grep -rn "# mypy: ignore-errors" control/src/ci/software_only_v2/` →
     zero results
   - `grep -rn "docker.from_env" control/src/ci/software_only_v2/tier3_fleet
     control/src/ci/software_only_v2/tier4_chaos` → zero results outside
     `orchestrator/`
   - `grep -rn "os.environ\[" control/src/ci/software_only_v2/tier4_chaos/
     test_chaos_smoke.py` → zero results
   - `pseti test lint` passes (Ruff + MyPy)

2. **Failing tests pass:** `pseti test sw2 unit logic` exits 0 with
   `74 passed, 0 failed` (or higher after new tests).

3. **Tier 3/4/5 parity additions land green:**
   - `pseti test sw2 fleet` — all transfer files + collision safety pass
   - `pseti test sw2 chaos` — expanded SC grpc + config validation +
     lifecycle pass
   - `pseti test sw2 integration` — Loki, telemetry, real_data
     (with `RUN_REAL_DATA_TESTS=1`), advanced transfer, transfer_watch,
     observability all pass; transactional_state, data_integrity,
     distributed, transfer_chaos, transfer_daemon_crash_recovery pass
     against the compose stack with real hashpipe

4. **Parity scenarios wired:** `python -c "from
   control.src.ci.software_only_v2.infra.parity import
   parity_coverage_report; print(parity_coverage_report())"` shows
   `two_node_start_stop`, `grpc_inject_unavailable`, and
   `process_kill_and_restart` as called (not stub).

5. **v1 deleted:**
   - `find control/src/ci/software_only -type f` → no such directory
   - `pseti test sw unit` → command not found / removed from test_cli
   - `pseti test sw2 unit logic fleet chaos integration` still green
   - `grep -rn "software_only/" control/ --include='*.py' --include='*.toml'
     --include='*.yml'` → zero non-historical references

6. **HW-SW unaffected:** `pseti test hw run` still passes (we did not
   touch entrypoints).
