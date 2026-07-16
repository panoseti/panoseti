# Plan: Unblock `test-pseti` deployment — fix grpc syntax break, hw-sw failures, admin CLI, and Loki/Alloy

## Context

You are on the verge of deploying to real hardware. All `software_only` and `hardware_software` tests
*were* green, but on `feature/admin-cli-deploy` six hw-sw happy-path tests now fail (`pseti start -y --no-hv`
exit 1), the DaqData service shows `UNIMPLEMENTED` on the remote DAQ node, the Loki container is broken, and
the new `pseti admin` deploy tooling is untested. The goal is to make the system deployable with the DAQ-node
stack (grpc server + hashpipe + Grafana Alloy) running **in one container per node** (your choice — matches the
test arch, isolation is a non-concern, prioritise reproducibility), while preserving near-bare-metal hashpipe
performance.

### Root cause (verified this session)

A grpc-submodule commit titled **"chore: fix linting errors" (`9a2c812`)** stripped the parentheses from
`except (A, B):` → `except A, B:`, which is a hard **`SyntaxError` in Python 3**. Confirmed:
`import panoseti_grpc.daq_data.server` fails; `python -m compileall` fails on **5 source files** and the
pattern also appears in **14 test files**:

- `grpc/src/panoseti_grpc/daq_data/server.py:364`
- `grpc/src/panoseti_grpc/daq_data/data_sources.py:230`
- `grpc/src/panoseti_grpc/daq_data/simulate.py:156`
- `grpc/src/panoseti_grpc/daq_control/server.py:148, 593, 790`
- `grpc/src/panoseti_grpc/telemetry/resources.py:50`
- (+ 14 `tests/**` files — see `grep -rn "except [A-Za-z_.]*, [A-Za-z_.]*:" grpc/src grpc/tests`)

Because `daq_data` is first in the server's `INIT_ORDER` and there is **no try/except around the
per-service registration loop** (`grpc/src/panoseti_grpc/server.py:360-373`), the failed import aborts the
entire `--profile daq_node` startup → the client sees `UNIMPLEMENTED`/`UNAVAILABLE`. And
`control/pyproject.toml:130` installs grpc as an **editable local path** (`panoseti_grpc = { path = "../grpc",
editable = true }`), so **every container built from this tree ships the broken source.** This is the P0
blocker: the moment any DAQ-node image is rebuilt from this branch, the whole grpc server dies.

> The 18 currently-passing tests only pass because the *remote* `pseti-daqnode:hitl` image predates the
> 2026-05-18 break. **Fix grpc before the next `pseti test hw build`, or all tests will fail.**

---

## Phase 0 — P0: Repair the grpc syntax break (blocks everything)

Work happens in the `grpc/` submodule (also on branch `feature/admin-cli-deploy`).

1. Re-parenthesize every multi-type `except` across `grpc/src` **and** `grpc/tests`. The correct form is the
   original (see `git show 9a2c812` — e.g. `except (KeyboardInterrupt, asyncio.CancelledError):`).
   Apply mechanically, then verify — do **not** trust the regex blindly:
   ```bash
   cd grpc
   # candidate fix, then VERIFY:
   python -m compileall -q src/panoseti_grpc            # must report 0 errors
   grep -rn "except [A-Za-z_][A-Za-z0-9_.]*, [A-Za-z_][A-Za-z0-9_.]*:" src tests   # must be empty
   python -c "import panoseti_grpc.daq_data.server, panoseti_grpc.daq_control.server, panoseti_grpc.telemetry.resources"
   ```
2. **Audit `9a2c812` for other bad autofixes.** That commit ran a broken linter fix pass; the `except`
   breakage may not be the only regression. Review `git show 9a2c812 --stat` and diff for other suspicious
   rewrites (bare `raise`, comprehension scoping, walrus, etc.). Compile-scan is the safety net.
2b. Run the grpc unit suite (`cd grpc && pip install -e ".[dev]" && pytest tests/ -x -q` or
    `pseti test grpc all`) to confirm the test files also import/collect cleanly after re-parenthesizing.
3. Add a guard so this class of error can't ship again: ensure `ruff`'s `E` rules + a `python -m compileall`
   step run in the grpc CI (and in `pseti test lint`). Optionally wrap the `server.py:360-373` service loop
   in a `try/except` that logs and continues so one broken service can't silently kill the whole server.
4. Commit on the grpc branch; bump the submodule pointer in the parent repo (the parent already shows `grpc`
   as modified). Keep the version bump (`chore: bump version to 0.4.12`) so a fixed wheel is publishable.

**Critical files:** the 5 source files above; `grpc/src/panoseti_grpc/server.py`; `grpc/pyproject.toml` (CI).

---

## Phase 1 — Diagnose & fix the 6 hw-sw happy-path failures

The failing params are exactly the complex data modes (2-pixel/3-pixel trigger, no-anytrig PH grouping,
img8-interleave); the basic modes pass. The full `pseti start` output is **already embedded** in the test's
`AssertionError` (`test_core_happy_path.py::_invoke`, lines 55-60) — you just need to surface it.

1. **Capture the real error.** Re-run one failing case with the rebuilt (fixed-grpc) daqnode image and read
   the assertion output + DAQ-node hashpipe logs:
   ```bash
   pseti test hw run -k "ph_two-pix_stim-q1" -vv          # ~ per-case, not the full 20 min
   # then on the DAQ node / in the daqnode container:
   #   {DAQ_DATA_DIR}/{run_dir}/hp_stderr.log , hp_stdout.log
   #   head-node PSETI.Start log for START_FAILED / "liveness check failed"
   ```
2. Two hypotheses, disambiguated by that output:
   - **(a) hashpipe mode instability** — `start_recording`'s heartbeat / Phase-5 liveness
     (`control/src/control/start.py:638-720`) raises `RuntimeError` because hashpipe exits under the
     2pix/3pix/no-anytrig/img8 acquisition byte built by `driver/quabo_driver.py:get_daq_params` /
     `send_daq_params`. Fix in the driver/plugin or the mode config.
   - **(b) strict-mode teardown cascade** — `_check_no_remote_hashpipe` (`start.py:845-917`) hard-fails in
     strict mode if a prior parametrization left hashpipe alive (the symlink-swap fixture iterates variants).
     Fix by making the happy-path fixture teardown call `StopDaq` unconditionally between variants (mirrors
     the documented CI rule: "Cleanup fixtures must call StopDaq unconditionally").
3. Apply the fix indicated by the evidence and re-run the affected subset, then the full suite.

**Critical files:** `control/src/ci/hardware_software/suites/happy_path/{test_core_happy_path.py,conftest.py}`,
`control/src/control/start.py`, `control/src/control/driver/quabo_driver.py`,
`control/src/ci/hardware_software/core_obs_configs/data_config_*.json`.

---

## Phase 2 — DAQ-node deployment: grpc + hashpipe in one container (near bare-metal)

**Your performance question, answered:** hashpipe in a container will run at **effectively bare-metal speed**
provided the datapath escapes Docker's abstractions — which it does here:

- **`network_mode: host`** → no bridge/NAT/veth; hashpipe's `net_thread` reads the NIC directly (this is the
  only path that would otherwise add real overhead, and it's eliminated).
- **`IPC_LOCK` + `ulimits: memlock=-1`** → databufs are `mlock`ed/pinned, no paging.
- **`SYS_NICE` + `ulimits: rtprio=99`** → real-time thread priority + CPU affinity from inside the container.
- **`shm_size: 2gb`** (or `ipc: host`) → hashpipe shared-memory databufs.
- A container is just cgroups + namespaces around a **natively-executing** process — CPU and memory access are
  not virtualised, so compute/output threads see no measurable overhead.

The one thing the container **cannot** do is own a hashpipe that lives outside it: `StartDaq` uses
`asyncio.create_subprocess_exec` and later finds/kills hashpipe by PID via `psutil.process_iter`
(`grpc/src/panoseti_grpc/daq_control/server.py:299-430, 432-510`) — grpc and hashpipe **must share a PID
namespace**. Hence "both in one container" is the correct (and only clean) containerized shape.

### Changes

1. **Harden `grpc/deploy/docker-compose.daqnode.yml`** (already has `network_mode: host`,
   `cap_add: [NET_RAW,NET_ADMIN,IPC_LOCK,SYS_NICE]`, `shm_size: 2gb`). Add:
   ```yaml
   ulimits:
     memlock: -1          # unlimited pinned memory (pairs with IPC_LOCK)
     rtprio: 99           # allow SCHED_FIFO/RR (pairs with SYS_NICE)
     nofile: 1048576
   # optional, once host isolcpus is set:
   # cpuset: "2-7"        # pin hashpipe threads to isolated cores
   ```
2. **Host-level tuning (bare-metal, applies regardless of container):** document/set `isolcpus`+`nohz_full`
   for hashpipe cores, CPU governor = `performance`, NIC ring buffers (`ethtool -G`), and IRQ affinity off the
   hashpipe cores. These are what actually matter for line-rate capture; the container inherits them.
3. **Provide an evaluation path** so you can prove perf before committing: run the tier5 real-data / tcpreplay
   integration (`RUN_REAL_DATA_TESTS=1`) or a bare-metal-vs-container A/B with `tcpreplay` at target rate and
   compare dropped-packet counters. If overhead is unacceptable, the bare-metal fallback already exists
   (`grpc/scripts/setup_panoseti_grpc.sh` installs `panoseti_grpc.service`) — a low-risk switch later.
4. **Production build path:** `grpc/deploy/Dockerfile.daqnode` installs `panoseti-grpc` **from PyPI**, not
   local source. To deploy the fixed code you must either (a) **publish a fixed wheel** (recommended for
   reproducibility — bump + `python scripts/compile_protos.py` + build + upload), or (b) temporarily point the
   Dockerfile at local `../grpc`. Note this divergence from the HITL image (editable local source).

**Critical files:** `grpc/deploy/docker-compose.daqnode.yml`, `grpc/deploy/Dockerfile.daqnode`.

---

## Phase 3 — Fix & test the `pseti admin` CLI

`control/src/control/admin/cli.py` (`deploy`, `status`; docker via per-node `docker context`, resolved from
`DaqNode.docker_context` in `pydantic_config_models.py:294`). Bugs found:

1. **systemd unit-name mismatch (bare-metal path).** cli.py targets `panoseti_grpc_daemon`, but the installer
   (`grpc/scripts/setup_panoseti_grpc.sh`) creates `panoseti_grpc.service`. `deploy --mode bare-metal` and
   `status --mode bare-metal` will both fail. → use `panoseti_grpc` consistently. (Lower priority since you
   chose container mode, but fix it so `status` bare-metal doesn't lie.)
2. **`nodes == "all"` is mocked** to `["daq01","daq02"]` (cli.py:92-94). → resolve from `get_daq_config()`
   node list (use `DaqConfig` + `get_node_by_ip`).
3. Bare-metal deploy hardcodes conda env `grpc-py314` and `echo panoseti | sudo -S`. → make env/sudo
   configurable or document as an assumption; not needed for the container path.
4. **`admin deploy` does not deploy Alloy** — only `status` references it. Add Alloy to the deployed stack
   (Phase 4).

**Test the docker path end-to-end (your chosen arch):**
```bash
# one-time per node, from the headnode:
docker context create pseti-daq-ucb1 --docker "host=ssh://panoseti@192.168.88.152"
pseti admin status 192.168.88.152 --mode docker      # docker --context ... compose ps
pseti admin deploy 192.168.88.152 --mode docker      # build+up the daqnode stack on the remote daemon
```
Verify the remote container comes up and, from the headnode, that DaqControl **and** DaqData both respond
(health probe / `pseti val`) — i.e. the Phase-0 fix resolved the `UNIMPLEMENTED`.

**Critical files:** `control/src/control/admin/cli.py`,
`control/src/control/utils/pydantic_config_models.py`, `control/configs/ucb/daq_config*.json`.

---

## Phase 4 — Fix Loki + ship logs via containerized Alloy (host network)

Log path: services write `{service}.jsonl` under `/var/log/panoseti` → **Alloy** tails them → **Loki** on the
headnode → Grafana. Bugs found:

1. **Alloy → Loki URL is wrong across namespaces.** `grpc/deploy/alloy/config.alloy` pushes to
   `http://loki:3100`, but Alloy runs on `network_mode: host` where the bridge DNS name `loki` does not
   resolve — and from a DAQ node Loki is on the *headnode*. → push to `http://<HEADNODE_IP>:3100`
   (parameterise via env). This is the concrete "Loki is broken" fix for log delivery.
2. **Grafana compose hardcodes a macOS path** (`/Users/nico/...:/etc/grafana/provisioning`) →
   Grafana provisioning mount fails on Linux. Point it at the repo's provisioning dir.
   (`grpc/src/panoseti_grpc/telemetry/docker-compose.loki.yml`.)
3. Loki compose has `user:`/`restart:` commented out and writes `./loki-data` as root → check ownership of
   the data dir so Loki can start/write; set `restart: unless-stopped`.
4. **Deploy Alloy on each DAQ node as a host-network container** (your choice) via
   `grpc/deploy/alloy/docker-compose.alloy.yml` (already `network_mode: host`, mounts `/var/log/panoseti:ro`).
   Fold this into `pseti admin deploy --mode docker` so a node deploy brings up grpc **and** Alloy.

**Critical files:** `grpc/deploy/alloy/config.alloy`, `grpc/deploy/alloy/docker-compose.alloy.yml`,
`grpc/src/panoseti_grpc/telemetry/docker-compose.loki.yml`, `grpc/deploy/docker-compose.daqnode.yml`
(or a combined compose the admin CLI brings up).

---

## Verification (end-to-end)

1. **grpc integrity:** `cd grpc && python -m compileall -q src/panoseti_grpc` (0 errors) + import smoke test +
   `pseti test grpc all`.
2. **Rebuild & redeploy DAQ node:** `pseti test hw build` → `pseti test hw deploy` (now safe post-fix); or for
   production `pseti admin deploy <node> --mode docker`. Confirm the daq_node-profile server starts and
   **both** DaqControl + DaqData respond (no `UNIMPLEMENTED`).
3. **hw-sw suite green:** `pseti test hw run` → expect 24 passed (was 18/6). Spot-check a previously-failing
   param first (`-k ph_two-pix_stim-q1`).
4. **CLI smoke:** `pseti -h`, `pseti val`, `pseti admin status <node> --mode docker`.
5. **Logs flowing:** trigger a run, confirm `.jsonl` under `/var/log/panoseti`, Alloy container up, and lines
   visible in Loki/Grafana on the headnode.
6. **Perf check (optional gate before trusting containerized hashpipe):** tcpreplay at target rate,
   compare hashpipe drop counters container-vs-baremetal; if unacceptable, fall back to the existing
   `panoseti_grpc.service` bare-metal install.

## Sequencing

Phase 0 first (unblocks all builds) → Phase 1 (green tests) → Phases 2-4 can proceed in parallel, then the
full verification pass. Do **not** run `pseti test hw build` until Phase 0 lands.
