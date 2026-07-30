# PANOSETI HITL Test Suite — Bug Fix Plan

## Context

Running `pseti test hw run` against the UCB rig (headnode `192.168.88.103`, gateway `192.168.88.152`, quabos `192.168.3.248–251`) yielded **47 failed / 10 passed / 35 skipped in 18 min**. The framework runs end-to-end, but several distinct categories of bugs cascade across the suite. Most damning: `pseti stat sweep` reports **Quabos: DOWN — 0/4 reachable** *outside* of pytest, which means production code itself can't talk to the quabos via the gateway during HITL runs. This pulls the floor out from under the rest of the suite.

The user's two key observations:

1. **Quabos may not be getting rebooted to a clean state.** The `wps_power_on` primitive only flips the WPS outlet — it does not load firmware, wait for boot, or verify reachability. Tests that nominally require `BOOTED` proceed against unbooted/unreachable hardware.
2. **Sockets may be leaking.** The `quabo` and `hk_socket` fixtures construct sockets without finalizers, and `hk_socket` collides with `capture_hk.py` on UDP/60002.

Reading the failure logs carefully against `wiki_docs/Sessions-and-configuration.md`, `pydantic_config_models.py`, `paths.py`, and `TEST-HW-SW.md`, I traced the 47 failures down to **11 distinct root causes**:

| # | Root cause | Failures unblocked |
|---|---|---|
| 1 | `wps_power_on` doesn't reboot quabos / wait for them — they stay unreachable | most of `hw0`, `hw3`, `hw4` |
| 2 | `quabo` fixture builds `QUABO(first.ip)` with raw `192.168.3.x` and no port — bypasses gateway port-forwarding | every test that uses the `quabo` fixture |
| 3 | `QuaboAddr` doesn't carry `real_ip`/`cmd_port`/`reboot_port` — fixtures can't see port-forwarding overrides | structural; blocks #2 |
| 4 | 5 sites still treat Pydantic `ObsConfig`/`DaqConfig` as dicts (`.items()`, `.get(...)`) | `hw3_lifecycle`, `hw4_telemetry` |
| 5 | `test_hk_udp.py:111` calls `get_boardloc(a.module_id, a.quadrant)` — wrong type and redundant (boardloc already on `QuaboAddr`) | `hw4_telemetry` HK low-level |
| 6 | `test_maroc_roundtrip` builds `dict[str,int]` but `make_maroc_cmd` calls `val.split(",")` (needs `dict[str,str]`) | `hw0_driver_protocol::test_maroc_roundtrip` |
| 7 | 11 test sites call `pseti start`/`pseti stop` without `--yes`, hanging on confirmation prompt → 30 s pytest-timeout | `hw2_observing`, one in `hw4_telemetry` |
| 8 | `capture_hk.py` is **never** started in the HITL container — Redis `QUABO_<boardloc>` HASH never populates, InfluxDB `metadata` DB never created | all of `hw4_telemetry` mid/high-level |
| 9 | `hk_socket` fixture binds UDP/60002 without `SO_REUSEPORT` — collides with `capture_hk.py` once #8 is fixed | `hw4_telemetry` low-level |
| 10 | `quabo` and `hk_socket` fixtures lack teardown finalizers — sockets leak across tests | resource exhaustion under repeated runs |
| 11 | Global pytest-timeout of 30 s is too tight for power-on / boot / observing tests | `hw3_lifecycle::test_full_power_cycle`, `hw2_observing::test_full_run_to_archive` |

Fixing these in two phases (bring-up + fixture/contract bugs first, telemetry pipeline second) should drop the failure count from 47 → near 0 and remove all cascade noise so any residual hardware quirks become individually visible.

This plan is **design + scope only**. Implementation waits for approval; no edits are made now.

---

## Phase A — Bring-up correctness, fixture wiring, API contracts

Goal: make every test that *should* be able to talk to a quabo actually able to talk to a quabo, with clean teardown.

### A1. State machine: real `BOOTED` semantics
**Problem:** `hw_state_machine.toml::wps_power_on` claims to drive `UNPOWERED → BOOTED` in 120 s typical, but its entrypoint `control.power:quabo_power(on=True)` only toggles the WPS outlet. It never:
- waits for the quabo CPU to boot,
- loads firmware via TFTP,
- verifies UDP reachability through the port-forwarding gateway.

**Fix:**
1. Insert a new state `POWERED` between `UNPOWERED` and `BOOTED` in `hw_state_machine.toml`. `wps_power_on` retargets to `POWERED` (`budget_s = { typical = 5, max = 15 }`, `safety = "high"`).
2. Add a new primitive `boot_verify` (`POWERED → BOOTED`, `budget_s = { typical = 120, max = 300 }`, `safety = "low"`) whose entrypoint is a new helper `hw_utils.driver_ops:boot_verify` that:
   - Polls each quabo's `cmd_port` (resolved through `util.get_quabo_ip_port(...)` — i.e. through the gateway when `network_config.json` says so) until UDP echo responds, with a per-quabo timeout drawn from `budget_s.max`.
   - Optionally calls `tftp_reboot` for each quabo first if `network_config.modules[*].port_forwarding.reboot_port` is set; otherwise just waits.
   - Logs every quabo's reachability state via the unified Rich logger so failures are easy to diff against `pseti stat sweep`.
3. The scheduler will now plan `wps_power_on → boot_verify` automatically when a test requires `BOOTED`.

This single change unblocks ~70% of failures because every downstream test currently hits an unreachable quabo.

### A2. `QuaboAddr` carries port-forwarding info
**Problem:** `topology.py::QuaboAddr` only has `ip`, `module_id`, `quadrant`, `boardloc`. Tests then pass `QuaboAddr.ip` to `QUABO(...)`, getting the raw `192.168.3.x` IP instead of the gateway IP+port. Production code (`start.py:401`) uses `util.get_quabo_ip_port(module_ip, quadrant, network_config)` for exactly this reason.

**Fix:** Extend `QuaboAddr`:
```python
@dataclass
class QuaboAddr:
    ip: str           # raw quabo IP (kept for backward compat / display)
    module_id: int
    quadrant: int
    boardloc: int
    real_ip: str      # gateway IP if forwarded, else == ip
    cmd_port: int     # 60000+quadrant or forwarded port
    reboot_port: int  # 60004+quadrant or forwarded port
```

In `HwTopology.quabo_ips()`, populate the three new fields by calling `util.get_quabo_ip_port(module.ip_addr, q, self._net)`. No test should ever construct gateway IPs by hand again.

### A3. `quabo` fixture uses the production lookup
**Problem:** `fixtures/quabo_fixtures.py:22-31` does `QUABO(first.ip)`. No port; no gateway awareness.

**Fix:** Mirror `start.py:401-412`:
```python
@pytest.fixture
def quabo(topology):
    a = next(x for x in topology.quabo_ips() if x.quadrant == 0)
    q = QUABO(a.real_ip, a.cmd_port)
    yield q
    q.close()    # see A8
```

Same fix for any other fixture (`quabos_all`, `quabo_q1` if added later) — always go through `QuaboAddr.real_ip`/`cmd_port`.

### A4. Pydantic config access (5 sites)
**Problem:** Config loaders return Pydantic models, but five test sites still treat them as dicts:

| File:line | Current | Fix |
|---|---|---|
| `hw3_lifecycle/test_lifecycle.py:59` | `obs_config.items()` | iterate `obs_config.domes` / `model_extra` as appropriate |
| `hw4_telemetry/test_hk_influx.py:122` | `config_file.get_obs_config().get("name", "")` | `obs_config.name` (or `model_extra.get("name", "")` if not in schema) |
| `hw4_telemetry/test_hk_influx.py:279` | `obs.items()` | same as :122 — iterate model attributes |
| `hw4_telemetry/test_hk_udp.py:213` | `topology._obs.get("domes", [])` | `topology._obs.domes` |
| `hw4_telemetry/test_hk_udp.py:249` | same | `topology._obs.domes` |

Per CLAUDE.md mandate: prefer model attribute access over dict indexing.

### A5. `get_boardloc` API correction
**Problem:** `hw4_telemetry/test_hk_udp.py:111` calls `get_boardloc(a.module_id, a.quadrant)`. Signature is `get_boardloc(module_ip_addr: str, quadrant: int)` — passing an `int` raises. And it's redundant: `a.boardloc` is already computed in `QuaboAddr`.

**Fix:** Replace with `a.boardloc`. Sweep the rest of `suites/` for the same antipattern.

### A6. `make_maroc_cmd` data type
**Problem:** `hw0_driver_protocol/test_driver_protocol.py:127-161` constructs the MAROC config dict with **integer** values:
```python
cfg = {"GAIN_CHANNEL_0": 0, ...}
quabo.send_maroc_params(cfg)
```
But `quabo_driver.make_maroc_cmd` (line 668) does `val.split(",")` — it expects strings of comma-joined 4-tuples like `"0,0,0,0"`. The current test crashes inside `make_maroc_cmd`.

**Fix:** Use the canonical string form. Reference an existing fixture or helper if one exists; otherwise add a small helper in `fixtures/quabo_fixtures.py` that returns a known-good MAROC config dict so future tests don't repeat the mistake.

### A7. `--yes` flag on `pseti start` / `pseti stop` (11 sites)
**Problem:** `pseti start` and `pseti stop` prompt for confirmation; the test just hangs until the 30 s pytest-timeout fires.

**Fix sites** (add `--yes` or use the typer-runner equivalent):
- `hw2_observing/test_observing.py` lines 70, 96, 118, 140, 169, 204, 237, 275, 322, 361
- `hw4_telemetry/test_hk_influx.py:242`

Centralize this in a helper in `fixtures/run_helpers.py` (`start_run(...)`, `stop_run(...)`) so future tests can't forget. The helper should also stream `--yes` and any other unattended flags.

### A8. Socket lifecycle / teardown finalizers
**Problem:** `quabo` fixture and `hk_socket` fixture hand out raw sockets / `QUABO` objects with no finalizer. After ~50 tests, Linux file-descriptor pressure or stuck UDP binds become a confounding variable.

**Fix:**
1. Add `QUABO.close()` (or wrap an existing teardown if one exists) and call it in the fixture finalizer.
2. `hk_socket` becomes a `yield` fixture with `sock.close()` on teardown.
3. Same audit for `wps_outlet`, `redis_client`, `influx_client` — every fixture must close its own resources.

---

## Phase B — Telemetry pipeline + polish

Goal: bring `hw4_telemetry` online, eliminate the remaining timeouts, and harden test isolation.

### B1. Bootstrap `capture_hk.py` in the HITL container
**Problem:** `docker-compose.hw-sw.yml:69-73` starts `storeInfluxDB.py` and `storeLoki.py` but **not** `capture_hk.py`. So no quabo HK packets ever reach Redis; the `metadata` DB in InfluxDB stays empty; every mid- and high-level HK test fails.

**Fix:** Two options, listed in order of preference:

**Option 1 (preferred):** Add a `capture_hk` autouse fixture in `hw4_telemetry/conftest.py` that spawns `capture_hk.py` as a subprocess scoped to that suite (`session` or `module` scope), with a finalizer that SIGTERM's it. This keeps the docker-compose config stable and lets non-telemetry suites avoid the daemon entirely (which simplifies the `hk_socket` SO_REUSEPORT story).

**Option 2:** Add `python src/control/daemons/capture_hk.py &` to the `headnode-server` command in `docker-compose.hw-sw.yml`. Simpler but global — affects every HITL run.

Either way, the fixture/daemon must:
- Wait for `capture_hk.py` to bind UDP/60002 (poll `lsof` or watch logs).
- Verify the daemon is healthy (one `QUABO_<boardloc>` HASH appears within 10 s of quabos being `BOOTED`).

### B2. `hk_socket` SO_REUSEPORT
**Problem:** Once `capture_hk.py` is bound to UDP/60002, low-level HK tests can't also bind there.

**Fix:** In `fixtures/telemetry_fixtures.py:13-23`, set `SO_REUSEPORT` and `SO_REUSEADDR` before bind. Both peers must set both options, so `capture_hk.py` would also need to set them — verify and patch if missing. (Per Linux semantics, `SO_REUSEPORT` is what allows two processes to receive the same UDP datagrams cooperatively; without it the test socket steals packets from the daemon.)

Document the pairing requirement in `TEST-HW-SW.md` Troubleshooting.

### B3. Per-test `@pytest.mark.timeout(N)` overrides
**Problem:** `pyproject.toml` (or pytest config) sets `timeout = 30`. Some tests inherently exceed this:

| Test | Realistic budget |
|---|---|
| `hw3_lifecycle::test_full_power_cycle` | 360 s (max boot budget + slack) |
| `hw3_lifecycle::test_tftp_reboot` | 180 s |
| `hw2_observing::test_full_run_to_archive` | 180 s (30 s data + transfer + verify) |
| `hw2_observing::test_multi_run_drain` | 360 s |
| `hw4_telemetry::test_redis_populated_for_all_quabos` | 60 s |

**Fix:** Apply `@pytest.mark.timeout(N)` per test. Don't raise the global default — that just hides hangs in fast tests. Where appropriate, add a `slow_hw` marker so `pytest -m "not slow_hw"` gives a developer-fast subset.

### B4. `test_07` cascade robustness
The summary mentions `test_07_*` tests cascading into failures when an upstream `BOOTED` test fails. Pytest already supports `pytest.fail()` early, but the cascade is loud and obscures the root cause. Two complementary fixes:

1. The state machine should mark any test in a batch whose **transition into the batch failed** as `skip("transition X failed: <reason>")` instead of letting them run and produce confusing assertion failures.
2. Add a session-scoped `topology_reachable` fixture that runs a quick UDP echo to every quabo at session start; on failure, every test that requires `BOOTED` is skipped with the actual reachability error. (Belt and braces with A1's `boot_verify`.)

### B5. `pseti test hw check-env` extensions
The user runs `check-env` before `run`, but it doesn't currently catch the issues Phase A fixes. Extend it to verify:

- WPS outlet reachable and toggleable.
- Each quabo reachable through the gateway via `cmd_port` AND `reboot_port` (this would have caught the "DOWN — 0/4 reachable" before pytest started).
- `capture_hk.py` is configured to start (Option 1 fixture present, or Option 2 compose command updated).
- `network_config.json` matches what the test fixtures will compute.
- `obs_config.json` and `daq_config.json` validate cleanly.

This is the single most-leveraged change for developer ergonomics: a 10-second `check-env` that fails loudly beats an 18-minute `run` that fails silently.

---

## Critical files to modify

**Phase A:**
- `control/src/ci/hardware_software/hw_state_machine.toml` — add `POWERED` state + `boot_verify` primitive.
- `control/src/ci/hardware_software/hw_utils/driver_ops.py` — implement `boot_verify` entrypoint.
- `control/src/ci/hardware_software/hw_utils/topology.py` — extend `QuaboAddr`, populate via `util.get_quabo_ip_port`.
- `control/src/ci/hardware_software/fixtures/quabo_fixtures.py` — fix `quabo` fixture, add finalizer, add MAROC helper.
- `control/src/ci/hardware_software/fixtures/run_helpers.py` (NEW) — `start_run(--yes)` / `stop_run(--yes)` helpers.
- `control/src/ci/hardware_software/suites/hw0_driver_protocol/test_driver_protocol.py` — fix MAROC dict types (line 127-161).
- `control/src/ci/hardware_software/suites/hw2_observing/test_observing.py` — replace direct `pseti start`/`stop` with helper (10 sites).
- `control/src/ci/hardware_software/suites/hw3_lifecycle/test_lifecycle.py` — Pydantic access fix (line 59).
- `control/src/ci/hardware_software/suites/hw4_telemetry/test_hk_influx.py` — Pydantic access (lines 122, 279), `--yes` (line 242).
- `control/src/ci/hardware_software/suites/hw4_telemetry/test_hk_udp.py` — Pydantic access (lines 213, 249), `get_boardloc` → `a.boardloc` (line 111).

**Phase B:**
- `control/src/ci/hardware_software/suites/hw4_telemetry/conftest.py` (NEW) — `capture_hk` autouse fixture (Option 1).
- `control/src/ci/hardware_software/fixtures/telemetry_fixtures.py` — `SO_REUSEPORT`/`SO_REUSEADDR` on `hk_socket`, finalizer.
- `control/src/control/daemons/capture_hk.py` — verify `SO_REUSEPORT` set on its bind; patch if not.
- Per-test `@pytest.mark.timeout(N)` annotations across `hw2_observing/`, `hw3_lifecycle/`, `hw4_telemetry/`.
- `control/src/ci/hardware_software/conftest.py` — `topology_reachable` session fixture; scheduler skip-on-failed-transition.
- `control/src/ci/hardware_software/hw_utils/cli.py` — extend `check-env` (reachability via gateway, daemon configuration, config validation).
- `control/TEST-HW-SW.md` — document `boot_verify`, SO_REUSEPORT pairing, `slow_hw` marker, new `check-env` checks.

**Reuse (no changes):**
- `control.utils.util.get_quabo_ip_port` — canonical port-forwarded address resolution.
- `control.utils.config_file` (`get_obs_config`, `get_daq_config`, `get_network_config`) — canonical config loaders.
- `control.driver.quabo_driver.QUABO` + `make_maroc_cmd` — driver API (we conform to it, don't change it).
- `control.daemons.capture_hk.{storeInRedis, get_true_detector_current}` — reused for HK assertions.
- `control.power.quabo_power` — `wps_power_on` / `wps_power_off` entrypoints (unchanged; only state graph changes).

---

## Verification

After Phase A lands:

1. `pseti test hw check-env` (Phase A version: at minimum the existing checks) → should pass.
2. `pseti test hw plan` → batch plan now shows `wps_power_on (5 s) → boot_verify (120 s) → ...` between `UNPOWERED` and `BOOTED` batches.
3. `pseti test hw run --class driver_protocol` → all driver-protocol tests should pass (covers A1, A2, A3, A6, A8).
4. `pseti test hw run --class observing` → observing tests should no longer time out (covers A7).
5. `pseti test hw run --class lifecycle -k test_full_power_cycle` → hits the new `boot_verify`; should complete and validate quabo reachability (covers A1).

After Phase B lands:

6. `pseti test hw run --class telemetry` → all three layers (UDP, Redis, InfluxDB) green (covers B1, B2).
7. `pseti test hw run` (full suite, no filter) → expected drop from 47 failed → 0 (modulo any genuinely flaky hardware on the rig).
8. `pseti test hw check-env` (Phase B version) → catches a deliberately-broken `network_config.json` port-forwarding entry without running pytest.
9. `pseti test sw all` → unchanged; conftest.py changes are HITL-scoped, no software-only regression.
10. `pseti test lint` → clean (Ruff + MyPy).

If any test still fails after this plan lands, it should now be a true single-cause failure — no more cascade noise — and individually addressable.

---

## Out of scope (intentionally deferred)

- The HITL framework architectural plan from the prior session (state-aware scheduler refinements, software-only/HITL conftest split, Pydantic-typed `qa.toml`). That work is broader and not blocking the rig coming back online; this plan is strictly bug-fix + minimal hardening.
- Refactoring `make_maroc_cmd` to accept `dict[str, list[int] | str]` — would be cleaner but changes a production API for one test's convenience.
- Migrating `hw_safety_net` to use the state machine's emergency teardown rather than direct CLI calls. Worth doing eventually; not in this plan.
