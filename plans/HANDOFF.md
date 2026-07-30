# HANDOFF.md — Tier 3/4 Test Infrastructure: Testcontainers Migration

**Branch:** `test-refactor`
**Status:** Partially implemented. Two bugs remain that cause 61 skips and 2 failures in `pseti test sw fleet`.
**Author:** Previous engineer session
**Recipient:** Next engineer

---

## 1. The Strict Mandate

**Tier 3 and Tier 4 tests MUST use testcontainers and dynamic fleets. There is no fallback to `docker-compose` static
infrastructure or `DAQNODE_DIRECT_HOST` environment variables.**

The previous architecture connected tier3/4 tests to pre-wired Docker Compose containers using env vars
(`DAQNODE_DIRECT_HOST=192.168.0.10`, `DAQNODE2_HOST=192.168.0.20`, etc.). This is now **deprecated for tier 3/4**. Those env vars
remain in `docker-compose.integration.yml` for the Docker Compose stack's own internal networking but must not be referenced by
test fixtures in `ci/tier3_fleet/` or `ci/tier4_chaos/`.

The `session_fleet` fixture in `ci/tier3_fleet/conftest.py` is the single source of truth for DAQ node connectivity. It starts a
dynamic testcontainers fleet and yields `(fleet, daq_cfg)`. All tier3/4 client fixtures (`daq_control_direct`, `daq_control_node2`,
etc.) derive from it. No test fixture is permitted to fall back to `DAQNODE_DIRECT_HOST` or any other static env var.

---

## 2. Bug 1 — The Ryuk TC_SESSION_ID 409 Conflict (Root Cause of 61 Skips)

### What Ryuk Is

Testcontainers-python spawns a sidecar container called **Ryuk** (`testcontainers-ryuk`) at the start of any testcontainers
session. Ryuk is a garbage-collector: it watches for the test process to exit (or crash) and then removes all containers created
during that session. There is one Ryuk per session.

### How Ryuk Is Named

Ryuk's Docker container name is derived from `TC_SESSION_ID`:

testcontainers-ryuk-{TC_SESSION_ID}

If `TC_SESSION_ID` is not set in the environment, testcontainers generates a random UUID at startup. If it IS set, that value is
used verbatim.

### How the 409 Happens

**Scenario: stale Ryuk from a previous run.**

When a test run crashes or the Docker Compose teardown races with container cleanup, Ryuk may be left running. On the next run, if
`TC_SESSION_ID` resolves to the same value (because it was deterministically set or stored in a testcontainers properties file),
testcontainers tries to create `testcontainers-ryuk-{same-ID}` again. Docker returns:

409 Client Error: Conflict
"The container name /testcontainers-ryuk-a63f2e11-... is already in use"

**Scenario: xdist workers sharing the same TC_SESSION_ID.**

When `pytest-xdist` spawns parallel workers, each worker is a subprocess that inherits the parent's environment. If `TC_SESSION_ID`
is set (or unset) identically for all workers, every worker tries to create `testcontainers-ryuk-{same-ID}`. The first worker
wins; every subsequent worker gets a 409.

**Why `session_fleet` cascades into 61 skips.**

`session_fleet` is `scope="session"`. When it fails with the 409 `APIError`, it calls `pytest.skip(...)`. A session-scoped fixture
skip propagates to every test that depends on it. Since `daq_control_direct`, `daq_control_node2`, `run_params`, and
`daqnode_container` all depend on `session_fleet`, the entire `test_daq_lifecycle.py`, `test_concurrent_daq_operations.py`, and
`test_data_collection.py` modules skip completely.

### Current State of the Fix (Incomplete)

`ci/conftest.py::pytest_configure` contains a partial fix:

```python
if "TC_SESSION_ID" not in os.environ and hasattr(config, "workerinput"):
  wid = config.workerinput.get("workerid", "")
  if wid:
      os.environ["TC_SESSION_ID"] = f"tc-session-{wid}"

This only fires for xdist workers (when workerinput is present) and is a no-op in the single-process fleet runner (no xdist). It
also does not address the stale-Ryuk-from-previous-run scenario because "tc-session-master" is deterministic and persists across
runs.

ci/fixtures/conftest.py::auto_isolate also sets TC_SESSION_ID unconditionally per test function, which is the wrong scope — it runs
after session_fleet has already been requested and doesn't actually prevent the collision.

---
3. Bug 2 — test_distributed_flows.py Inline Fleet Collision

What the File Does Wrong

ci/tier3_fleet/test_distributed_flows.py currently calls make_fleet(n=2).start() inside each test function body:

async def test_when_distributed_run_started_then_all_nodes_recording(tmp_path):
  fleet = make_fleet(n=2)
  fleet.start()          # <-- spawns a SECOND testcontainers session
  try:
      fleet.wait_healthy()
      ...
  finally:
      fleet.tear_down()

This creates a second concurrent testcontainers session within the same pytest session that already has session_fleet running. Both
sessions share the same TC_SESSION_ID (since none of the current fixes apply cleanly at test-function scope), so the second
fleet.start() tries to create a Ryuk container with the same name as the one already running → 409. This is why both
test_distributed_flows.py tests show as FAILED rather than skipped (the error occurs inside the test body, not inside a fixture).

The Fix

test_distributed_flows.py must be refactored to accept session_fleet as a fixture parameter instead of instantiating its own fleet.
The fleet is already running for the session — there is no reason to create a second one. The test's purpose is to call
start_run() and stop_run() against real gRPC servers; the session_fleet fixture provides exactly those servers.

---
4. Required Fixes — Implementation Strategy

Fix 1: Unique TC_SESSION_ID Per Session in ci/conftest.py::pytest_configure

The fix must produce a unique, non-deterministic TC_SESSION_ID that:
- Is unique per pytest session (prevents stale-Ryuk collisions across runs)
- Is unique per xdist worker (prevents cross-worker Ryuk collisions)
- Is set once, early, before any testcontainers code runs (i.e., in pytest_configure, not in a fixture)

Strategy: generate a UUID at pytest_configure time and combine it with the worker ID.

# In pytest_configure(config):
import uuid as _uuid

run_uuid = _uuid.uuid4().hex[:8]  # short enough to fit in container names

if hasattr(config, "workerinput"):
  # xdist worker process
  worker_id = config.workerinput.get("workerid", "master")
else:
  # single-process run (fleet/chaos suites run without xdist)
  worker_id = "solo"

os.environ["TC_SESSION_ID"] = f"tc-{worker_id}-{run_uuid}"

Critical note: pytest_configure runs once per process (including each xdist worker subprocess). The run_uuid must be shared across
all workers in a parallel run so that inter-worker coordination still works correctly. The correct pattern is:

- In the controller (root) process: generate the UUID and write it to a temp file or pass it via pytest_configure's config object
before workers are forked.
- In each worker process: read the UUID from workerinput (which is populated by the pytest_runtest_protocol hook in the controller
before workers start).

The xdist-aware pattern uses pytest_configure_node (controller hook) to inject the run UUID into each worker's workerinput, then
reads it back in pytest_configure on the worker side:

# Controller-side hook (still in ci/conftest.py):
def pytest_configure_node(node):
  """Called by xdist controller to configure each worker before it starts."""
  if not hasattr(node.config, "_tc_run_uuid"):
      import uuid as _uuid
      node.config._tc_run_uuid = _uuid.uuid4().hex[:8]
  node.workerinput["tc_run_uuid"] = node.config._tc_run_uuid

# In pytest_configure:
if hasattr(config, "workerinput"):
  worker_id = config.workerinput.get("workerid", "master")
  run_uuid  = config.workerinput.get("tc_run_uuid", _uuid.uuid4().hex[:8])
else:
  worker_id = "solo"
  run_uuid  = _uuid.uuid4().hex[:8]

os.environ["TC_SESSION_ID"] = f"tc-{worker_id}-{run_uuid}"

For the non-xdist fleet/chaos runner (single process), workerinput is absent and worker_id = "solo". A fresh UUID is generated per
session, preventing stale-Ryuk collisions across consecutive runs.

Fix 2: session_fleet Must Be Unconditionally Testcontainers-Based

Remove the Docker-daemon availability guard that calls pytest.skip() when Docker is unreachable. Instead, let the fleet.start()
failure surface as a hard error so it is clearly diagnosed rather than silently skipping 61 tests. Alternatively, keep the skip but
ensure it only fires for environments that genuinely cannot reach Docker (e.g., unit test runners), not the integration runner
where Docker is always available.

The fixture must not check for DAQNODE_DIRECT_HOST or any other env var to decide whether to use testcontainers. It must always use
testcontainers.

Fix 3: Refactor test_distributed_flows.py to Use session_fleet

The two test functions in test_distributed_flows.py must accept session_fleet (and/or daq_control_direct, daq_control_node2) as
fixtures instead of creating a fleet inline.

The test's core logic (calling start_run() / stop_run() and asserting gRPC state) does not require a freshly-created fleet per
test. The fleet is already running and healthy when these tests execute. Use it.

The _docker_available() guard at the top of that file can be removed entirely — if the fleet isn't running, session_fleet will have
already skipped (or failed) long before these tests run.

---
5. Files to Modify

┌──────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────┐
│                   File                   │                                  Change Required                                   │
├──────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
│ ci/conftest.py                           │ Replace partial pytest_configure TC_SESSION_ID logic with the full xdist-aware     │
│                                          │ UUID strategy described above                                                      │
├──────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
│ ci/fixtures/conftest.py                  │ Remove os.environ["TC_SESSION_ID"] = ... from auto_isolate — it runs at function   │
│                                          │ scope which is too late and wrong                                                  │
├──────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
│ ci/tier3_fleet/conftest.py               │ session_fleet fixture: remove Docker-availability guard that silently skips;       │
│                                          │ ensure it is unconditionally testcontainers-based                                  │
├──────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
│ ci/tier3_fleet/test_distributed_flows.py │ Remove inline make_fleet(n=2).start() / fleet.tear_down() from both test bodies;   │
│                                          │ accept session_fleet (or derived client fixtures) as fixture parameters instead    │
└──────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────┘

---
6. Verification

After implementing the fixes, run:

pseti test sw fleet   # target: 0 failed, 0 skipped (from Ryuk), 2 distributed_flows PASSED
pseti test sw chaos   # target: no regressions from SC004 fix

The specific tests that must flip from SKIPPED → PASSED:

- test_daq_lifecycle.py — all 18 parametrized cases (direct + gateway)
- test_concurrent_daq_operations.py — all 4 cases
- test_data_collection.py — all 9 cases

The specific tests that must flip from FAILED → PASSED:

- test_distributed_flows.py::test_when_distributed_run_started_then_all_nodes_recording
- test_distributed_flows.py::test_when_distributed_run_stopped_then_all_nodes_halted

---
7. Known Good State

The following tests are currently passing and must not regress:

- test_transfer_basic.py — 7 tests
- test_transfer_daemon_e2e.py — 2 tests
- test_transfer_port_forwarding.py — 6 tests
- ci/tier4_chaos/test_sc_grpc_failures.py::test_SC004_... — 1 test (fixed this session)

The SC004 fix (in test_sc_grpc_failures.py) cleared module_ids on all DAQ nodes beyond node 0 before the StartDaq retry test,
ensuring exactly 2 calls instead of 3. Do not revert this.
