# Debugging Guide — PANOSETI Control Plane

This document captures hard-won debugging strategies for the control plane CI stack.
It is not a tutorial — it assumes familiarity with the architecture described in `CLAUDE.md`.

---

## 1. State Leak Identification

State leaks are the most common cause of test failures that reproduce intermittently or only in full-suite runs.

### Advisory lock stale on entry

`tmp/panoseti_control.lock` is held by `RunStateManager.__enter__`. If a previous process crashed without releasing it:

```bash
# Inside the container
flock --timeout 1 /app/tmp/panoseti_control.lock echo ok || echo "LOCK HELD"
lsof /app/tmp/panoseti_control.lock          # shows which PID holds it
```

If the holding process is gone, the lock is automatically released by the kernel — but a stale `.lock` file with no holder can still confuse manual inspection.

### Ledger left in ACTIVE state

`tmp/run_state.toml` persisting between tests causes `start.py` to refuse with "Another PANOSETI control process is already running." The `fresh_run_state` fixture covers daqnode-1 only. Check for leaks:

```bash
cat /app/tmp/run_state.toml                  # status must not be ACTIVE after test teardown
python -c "import tomllib; import sys; d=tomllib.load(open('tmp/run_state.toml','rb')); print(d)"
```

Manually clear a stuck ledger (tests only — never in production):

```bash
python -c "
from utils.run_state import RunStateManager
with RunStateManager() as mgr:
    mgr.clear()
"
```

### Hashpipe process left running after a test

```bash
pgrep -a hashpipe                            # on the daqnode container
ps aux | grep hashpipe
```

The DAQ Control server tracks only the PID it spawned. If the PID is alive but not tracked (e.g., after a container restart), `StartDaq` will find it via `psutil` scan and refuse. Kill it manually:

```bash
pkill -INT hashpipe                          # polite; matches how StopDaq works
pkill -KILL hashpipe                         # force-kill if SIGINT ignored
```

### Run directory residue

Per-test run dirs (`chaos_run_<hex>.pffd`) are cleaned by the `fresh_run_state` teardown fixture. If teardown itself crashed, dirs accumulate:

```bash
ls /data/*.pffd                              # should be empty between tests
ls /data/module_*/                           # per-module subdirs
```

---

## 2. Container Log Inspection

### Structured log tailing

```bash
# All services, last 100 lines each
python ci/qa.py logs

# Single container, follow
docker logs -f ctl-int-daqnode-1

# Since the last test run started
docker logs --since 5m ctl-int-daqnode-1 2>&1 | grep -E "ERROR|WARN|traceback|Exception"
```

### gRPC server startup confirmation

The DAQ Control server prints a startup banner. If it's absent, the server crashed on init:

```bash
docker logs ctl-int-daqnode-1 2>&1 | grep "Listening on"
```

### Loki log query (after `ENABLE_TELEMETRY_TESTS=1` runs)

```bash
# From inside the test-runner container or headnode_net
curl -s "http://10.0.1.21:3100/loki/api/v1/query_range?query={job%3D\"panoseti\"}&limit=20" \
  | python -m json.tool | grep message
```

### Redis key inspection

```bash
docker exec ctl-int-redis-1 redis-cli KEYS "logs:*"
docker exec ctl-int-redis-1 redis-cli LLEN logs:ingress
docker exec ctl-int-redis-1 redis-cli LRANGE logs:ingress 0 4
```

### DAQ Control server internal log

The server writes to `/var/log/panoseti/daq_control_server.log` inside the daqnode container:

```bash
docker exec ctl-int-daqnode-1 tail -50 /var/log/panoseti/daq_control_server.log
```

Per-run Hashpipe logs land in `{data_dir}/{run_dir}/`:

```bash
docker exec ctl-int-daqnode-1 cat /data/chaos_run_<hex>.pffd/hp_stdout.log
docker exec ctl-int-daqnode-1 cat /data/chaos_run_<hex>.pffd/hp_stderr.log
```

---

## 3. Isolating Hangs

### Identify which test is blocking

pytest-timeout is installed. If a test hangs beyond its `@pytest.mark.timeout(N)` it is killed and marked FAILED. Without a timeout decorator, the whole suite blocks.

Run a single test with a wall-clock timeout from the shell:

```bash
timeout 120 python ci/qa.py chaos -k "SC006" -v
```

### gRPC deadline not set → infinite block

The most common hang source is a gRPC call with no deadline on a frozen server or frozen subprocess. Symptoms: the test runner appears completely idle; `docker stats` shows the daqnode container CPU near zero.

Verify deadlines are set in `grpc/src/panoseti_grpc/daq_control/client.py`:

- `StartDaq`: no timeout (server returns immediately after subprocess spawn — safe)
- `StopDaq`: `timeout=30.0` — **critical**; a hashpipe that ignores SIGINT will block `p.wait()` indefinitely without this
- `StatusDaq`: no timeout (fast filesystem check)
- `CleanupData`: no timeout (fast directory deletion)

If you add a new RPC that waits on a subprocess or external resource, always pass `timeout=`.

### asyncio event loop starvation

If `start.py` hangs mid-run, the event loop may be blocked by a synchronous call. Symptoms: no log output for >5 s; the hang is inside an `async def` function.

Check for blocking calls inside `async` functions:

```bash
grep -rn "subprocess.run\|time.sleep\|os.path\|open(" control/utils/ control/start.py \
  | grep -v "asyncio.to_thread"
```

Any blocking I/O not wrapped in `asyncio.to_thread(...)` can stall the loop.

Get a stack trace of a hung process:

```bash
# Find the PID inside the container
docker exec ctl-int-daqnode-1 pgrep -f start.py

# Attach with py-spy (if installed) or use SIGUSR1 for asyncio debug dump
kill -USR1 <pid>   # prints all tasks to stderr if PYTHONASYNCIODEBUG=1
```

Enable asyncio debug mode for a test run:

```bash
PYTHONASYNCIODEBUG=1 python ci/qa.py chaos -k "SC021" -v 2>&1 | grep -A5 "blocking call"
```

### StopDaq blocks when hashpipe ignores SIGINT

Symptom: `StopDaq` returns after exactly 30 s with a gRPC DEADLINE_EXCEEDED error.
Cause: the hashpipe wrapper process catches or ignores SIGINT.
Fix path: the server should escalate to SIGKILL after the SIGINT timeout — tracked as a known limitation.

---

## 4. Using CLI Tools (qa.py / qa.toml)

### qa.py command reference

```bash
python ci/qa.py up            # start persistent background containers (idempotent)
python ci/qa.py down          # tear down all containers and volumes
python ci/qa.py build         # rebuild Docker images (needed after grpc/ submodule changes)
python ci/qa.py unit          # pytest ci/unit/ — hardware-agnostic, ~3 s
python ci/qa.py integration   # pytest ci/integration/ — E2E with real hashpipe, ~75 s
python ci/qa.py chaos         # pytest ci/integration/scenarios/ — TDD-forcing, ~10 min
python ci/qa.py lint          # ruff + mypy concurrently
python ci/qa.py logs          # tail all container logs
```

Additional pytest args pass through after the command:

```bash
python ci/qa.py chaos -k "SC010 or SC002" -v --tb=short
python ci/qa.py integration -k "TestDaqLifecycle" --no-header
python ci/qa.py unit -x --pdb   # stop on first failure, drop into debugger
```

### qa.toml environment overrides

`ci/qa.toml` controls which environment variables are injected into each test command. Key levers:

```toml
[commands.chaos]
env = { ENABLE_TELEMETRY_TESTS = "1" }   # enable telemetry scenario tests
```

```toml
[commands.integration]
env = { RUN_REAL_DATA_TESTS = "1" }      # enable tcpreplay / real hashpipe data tests
```

When modifying `qa.toml`, the change takes effect immediately (no rebuild needed) because `ci/` is live-mounted.

### Updating the gRPC submodule in a running container

The `grpc/` directory is **baked into the Docker image**, not live-mounted. After changing Python files in `grpc/src/`:

```bash
# Fast path — copy changed file directly into the running container
docker cp grpc/src/panoseti_grpc/daq_control/client.py \
  ctl-int-daqnode-1:/opt/venv/lib/python3.12/site-packages/panoseti_grpc/daq_control/client.py

# Slow path — full rebuild (required after .proto changes or new files)
python ci/qa.py build
```

The fast path survives container restart only if the volume is not wiped. Always use `python ci/qa.py build` before committing to ensure the image reflects the source.

### Checking what's inside the running container

```bash
docker exec -it ctl-int-daqnode-1 bash

# From inside:
python -c "import panoseti_grpc; print(panoseti_grpc.__file__)"
pip show panoseti-grpc
cat /opt/venv/lib/python*/site-packages/panoseti_grpc/daq_control/client.py | grep timeout
```

---

## 5. Advanced Insights

### asyncio cancellation and rollback ordering

`start.py::start_run()` is an `async` function that fans out `StartDaq` RPCs to multiple DAQ nodes. If a `CancelledError` propagates before the fan-out completes, only the nodes that received a receipt in the `RunStateLedger` will be rolled back. Nodes that got a receipt but had their RPC cancelled mid-flight (after the server started hashpipe but before the client got the response) will be left with a live hashpipe but no local receipt — the rollback ladder will miss them.

Mitigation: in `start.py`, always write the ledger receipt **before** awaiting the RPC, not after. The extra `StopDaq` for an already-stopped node is idempotent.

### TOML ledger and atomic writes

`tmp/run_state.toml` is written via a write-then-rename pattern to prevent partial reads. Never write it directly:

```python
# Wrong — partial write visible to concurrent readers
with open("tmp/run_state.toml", "w") as f:
    tomllib_w.dump(state, f)

# Right — atomic via RunStateManager
with RunStateManager() as mgr:
    mgr.write(state)
```

If a test mocks `RunStateManager` or injects a pre-written TOML file, ensure the file is valid TOML before the test runs — `tomllib.load()` raises `TOMLDecodeError` on partial writes, which surfaces as an unrelated-looking crash inside the production code under test.

### gRPC status codes vs. application errors

The DAQ Control service uses two error channels:

1. **gRPC status codes** (e.g., `UNAVAILABLE`, `DEADLINE_EXCEEDED`): transport-layer failures. The client wraps these in `ConnectionError`.
2. **`success=False` + `message`** in the response proto: application-layer rejections (e.g., "hashpipe already running"). The client raises `ValueError`.

Tests that expect a server rejection must catch `ValueError`, not `ConnectionError`. A common mistake:

```python
# Wrong — catches the wrong exception type
with pytest.raises(ConnectionError):
    client.StartDaq(params)

# Right
with pytest.raises(ValueError, match="HASHPIPE instances running"):
    client.StartDaq(params)
```

---

## 6. Effective Debugging Patterns

### Chaos / TDD Scenario Reproduction
The fastest way to debug distributed race conditions or rollback failures is via `qa.py chaos`. These tests are designed to be "TDD-forcing" (failing on current master). Use them to pin a bug before fixing:
```bash
python ci/qa.py chaos -k SCN003 -vv  # -vv shows real-time print() debugs
```

### Mocking gRPC with IP Tracking
When testing the orchestration ladder (e.g. `start.py`), you often need to mock `DaqControlClient` while still verifying *which* node was called. Use this pattern in your tests:
```python
def mocked_client_init(self, host, port):
    self._mock_host = host

def mocked_start_daq(self, params):
    if self._mock_host == "192.168.0.32":
        raise RuntimeError("Simulated Node 2 Failure")
    return True
```

### Fail-Fast Orchestration
The control scripts now use `asyncio.TaskGroup` for concurrent RPCs. This makes debugging easier: if any task in the group fails, all others are immediately cancelled. This prevents "phantom" processes from starting on some nodes while you are trying to diagnose a failure on another.

### Rsync Transient vs. Fundamental Failure
If data collection fails, check the rsync exit code in the logs.
- **Transient (12, 23, 30, 35, 255):** Network drops or SSH timeouts. The system will auto-retry with exponential backoff (5s, 10s, 20s).
- **Fundamental (1, 3, 5, etc.):** Permissions, disk full, or missing directories. These fail fast and skip the `CleanupData` step to prevent data loss.

### Inspecting the Live Ledger
If a run hangs or crashes, the "black box" is `tmp/run_state.toml`. Inspect it *before* running `stop.py`, as `stop.py` will transition it to `STOPPING` or `ABORTED`.
```bash
# See which nodes successfully returned a receipt before the crash
cat tmp/run_state.toml
```
