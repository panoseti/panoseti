# PANOSETI HITL Testing Framework — Design Plan

## Context

Hardware-software (HITL) testing today is a single-suite, sequence-coupled scaffold under `control/ci/hardware-software/` (one active test, six skipped). The QA harness (`qa.toml` / `qa_utils.py` / `test_cli.py`) was designed for Docker-based software CI: it conflates suite definition with `compose up/down` lifecycle, generates random subnets, and forces `PSETI_*` paths to `tmp_path`. Bringing dozens more HITL tests online in this scaffold would (a) waste 5–7 min on every test that doesn't need a power cycle, (b) leak software-CI assumptions onto real hardware, (c) make heterogeneity (driver protocol, GNSS, focus, HK pipeline) painful to express, and (d) duplicate a `FakeSocket` packet-assertion pattern already invented four times in `tier1_unit/`.

Goal: an extensible, TOML-driven, **state-aware** HITL test framework that batches tests by physical prerequisites, derives topology dynamically from the active observatory configs, and protects hardware via ironclad teardown. The framework must scale from a single-module UCB lab rig to a full Palomar production deployment using the same TOML, with no code changes.

This plan is design-only. Implementation waits for approval.

---

## 1. Repository Layout (refactor)

```
control/ci/
├── software-only/                # NEW — was the body of ci/conftest.py
│   ├── conftest.py               # auto_isolate, session_fleet, generate_ci_topology, fakeredis
│   ├── tier1_unit/               # moved from ci/tier1_unit
│   ├── tier2_logic/
│   ├── tier3_fleet/
│   ├── tier4_chaos/
│   ├── tier5_integration/
│   ├── qa.sw.toml                # renamed from qa.toml — software-only suites
│   └── qa_utils.py               # Docker-CI orchestrator (unchanged behavior, scope narrowed)
│
├── hardware-software/            # EXPANDED
│   ├── conftest.py               # HITL-only fixtures; never rewrites PSETI_* paths
│   ├── hw_tests.toml             # NEW — state-aware test classification (see §3)
│   ├── hw_state_machine.toml     # NEW — declarative state DAG + transition primitives (see §2)
│   ├── hw_utils.py               # NEW — HardwareStateMachine, StateAwareScheduler, SafetyManager
│   ├── hw_assertions.py          # NEW — HK packet parsing, Redis/InfluxDB query helpers
│   ├── hw_topology.py            # NEW — dynamic topology adapter over obs/daq/network configs
│   ├── configs/                  # unchanged — UCB single-module gold-standard
│   ├── fixtures/                 # NEW — shared fixtures (quabo, wps, hk_socket, …)
│   │   ├── packet_capture.py     # promoted from tier1_unit/ FakeSocket duplication
│   │   ├── quabo_fixtures.py
│   │   ├── wps_fixtures.py
│   │   └── telemetry_fixtures.py
│   ├── suites/                   # NEW — test suites organized by tier (see §6)
│   │   ├── hw0_driver_protocol/  # Low-level packet protocol against real quabos
│   │   ├── hw1_reconfig/         # Fast: configure + observe (no power cycle)
│   │   ├── hw2_observing/        # Real DAQ flow, transfer queue, science previews
│   │   ├── hw3_lifecycle/        # Power cycle, firmware reboot, calibration
│   │   └── hw4_telemetry/        # HK pipeline (UDP → Redis → InfluxDB)
│   └── preflight/                # NEW — symlinks/markers for pre-observation subset
│
└── shared/                       # NEW — used by both software-only and HITL
    ├── qa_models.py              # QAConfig, SuiteConfig, EnvironmentConfig (extracted)
    └── stream.py                 # _stream + TEST_METRICS_JSON parser
```

**Why split**: today's root `conftest.py` autouse-rewrites `PSETI_CONFIG/STATE/CONTROL/TMP/QUABOS` to `tmp_path`. On real hardware those paths must persist (`quabo_uids.json`, `firmware/`, `/mnt/panoseti-test/`). Today the split is a fragile `is_hw_sw_test` branch; making the directory split physical removes the foot-gun entirely. The current `ci/qa.toml` `test-hw` suite definition becomes a thin `[suites.test-hw]` entry in `qa.sw.toml` that delegates to the new HITL runner, so `pseti test hw` keeps its current entrypoint.

---

## 2. Hardware State Machine (declarative, not hardcoded)

A separate `hw_state_machine.toml` defines the state vocabulary and transition primitives. Both are loaded into Python `enum.Enum` and a directed graph at startup; no test or runner code references state names as strings. Adding a new state or primitive (e.g., `WR_SYNCED`, `tftp_load_firmware`) is a TOML edit only.

```toml
# hw_state_machine.toml
[state_machine]
initial = "UNPOWERED"
safe = "UNPOWERED"   # state to leave hardware in on framework exit

# States ordered loosely from "cold" → "warm" → "hot"
[[states]]
name = "UNPOWERED"
desc = "WPS outlet off."

[[states]]
name = "BOOTED"
desc = "Quabo CPU up, FPGA loaded, ping responsive, registers at firmware default."

[[states]]
name = "MAROC_LOADED"
desc = "MAROC chip registers programmed."

[[states]]
name = "ACQ_CONFIGURED"
desc = "ACQ params + trigger masks + GOE masks + dest IPs set."

[[states]]
name = "HV_ON"
desc = "HV ramped to setpoint; detector live."

[[states]]
name = "ACQUIRING"
desc = "DAQ mode bits set; quabo emitting science packets to DAQ node."

# Primitives — atomic state transitions.  Each primitive declares (from_states → to_state),
# wall-clock budget (typical/max), safety class, and the python entrypoint.
[[primitives]]
name = "wps_power_on"
from_states = ["UNPOWERED"]
to_state = "BOOTED"
budget_s = { typical = 120, max = 300 }
safety = "high"          # tagged for the SafetyManager dry-run preview
entrypoint = "control.power:quabo_power"
kwargs = { on = true }

[[primitives]]
name = "wps_power_off"
from_states = ["*"]      # wildcard — allowed from any state (e.g., emergency teardown)
to_state = "UNPOWERED"
budget_s = { typical = 5, max = 10 }
safety = "high"
entrypoint = "control.power:quabo_power"
kwargs = { on = false }

[[primitives]]
name = "tftp_reboot"
from_states = ["BOOTED", "MAROC_LOADED", "ACQ_CONFIGURED", "HV_ON"]
to_state = "BOOTED"
budget_s = { typical = 60, max = 120 }
safety = "medium"
entrypoint = "control.driver.quabo_tftp:tftpw.reboot"

[[primitives]]
name = "soft_reset"      # cmd 0x04 — logic reset, no firmware reload
from_states = ["MAROC_LOADED", "ACQ_CONFIGURED", "HV_ON", "ACQUIRING"]
to_state = "BOOTED"
budget_s = { typical = 1, max = 2 }
safety = "low"
entrypoint = "control.driver.quabo_driver:QUABO.reset"

[[primitives]]
name = "configure_maroc"
from_states = ["BOOTED"]
to_state = "MAROC_LOADED"
budget_s = { typical = 1, max = 5 }
safety = "low"
entrypoint = "hw_utils.driver_ops:configure_maroc"

[[primitives]]
name = "configure_acq"
from_states = ["MAROC_LOADED"]
to_state = "ACQ_CONFIGURED"
budget_s = { typical = 1, max = 5 }
safety = "low"
entrypoint = "hw_utils.driver_ops:configure_acq"

[[primitives]]
name = "hv_on"
from_states = ["ACQ_CONFIGURED"]
to_state = "HV_ON"
budget_s = { typical = 2, max = 10 }
safety = "high"          # detector damage risk
entrypoint = "hw_utils.driver_ops:hv_set_from_config"
guards = ["shutter_must_be_closed", "light_sensor_dark"]   # named guards in hw_utils.guards

[[primitives]]
name = "hv_off"
from_states = ["*"]
to_state = "ACQ_CONFIGURED"   # downgrade — never leave HV on after teardown
budget_s = { typical = 2, max = 5 }
safety = "high"
entrypoint = "hw_utils.driver_ops:hv_zero"

[[primitives]]
name = "start_acq"
from_states = ["HV_ON", "ACQ_CONFIGURED"]
to_state = "ACQUIRING"
budget_s = { typical = 5, max = 15 }
safety = "low"
entrypoint = "hw_utils.run_ops:start_run_via_cli"

[[primitives]]
name = "stop_acq"
from_states = ["ACQUIRING"]
to_state = "ACQ_CONFIGURED"
budget_s = { typical = 60, max = 90 }   # 60s graceful flush per TEST.md
safety = "low"
entrypoint = "hw_utils.run_ops:stop_run_via_cli"
```

Loaded into:

```python
class HwState(str, Enum): ...                         # generated from [[states]]
class Primitive(BaseModel): name, from_states, to_state, budget_s, safety, ...
class HardwareStateMachine:
    states: dict[str, HwState]
    primitives: dict[str, Primitive]
    graph: networkx.DiGraph                           # for shortest-path planning
    def plan(self, current: HwState, target: HwState) -> list[Primitive]: ...
    def cost(self, plan: list[Primitive]) -> float: ...    # sum of budget_s
```

Tests never call primitives directly. They declare a required state via a pytest marker; the scheduler calls the planner to compute the cheapest path. The shortest-path search uses `budget_s.typical` as edge weight, so `soft_reset → configure_maroc → ...` (~3 s) is strongly preferred over `wps_power_off → wps_power_on → ...` (~6 min).

---

## 3. `hw_tests.toml` — Test Classification Schema

Tests are tagged in TOML, not in code, so the classification can evolve without editing pytest collection. The tagging is keyed by **test node id glob**, so `pytest --collect-only` → toml lookup → state requirement.

```toml
# hw_tests.toml
[settings]
default_safety_check = "before_each_batch"
emergency_teardown_state = "UNPOWERED"
preflight_marker = "preflight"

# ── Test classes ──
# Each class declares the minimum state a test in it requires, and an optional
# guarantee of the state the test leaves the hardware in (default: same as required).

[classes.driver_protocol]
description = "Low-level UDP/TFTP packet protocol verification against real quabos."
required_state = "BOOTED"
leaves_state = "BOOTED"
batch_priority = 0
preflight = false
parallel = false                  # one quabo at a time
description_long = """
HW-equivalent of unit tests: send a known driver call, sniff the wire, assert byte layout
matches Quabo-packet-interface.md.  Reuses ci/tier1_unit FakeSocket assertions but the
counterparty is a real FPGA.
"""

[classes.fast_reconfig]
description = "Configuration changes + short observation, no power cycle."
required_state = "ACQ_CONFIGURED"
leaves_state = "ACQ_CONFIGURED"
batch_priority = 1
preflight = true
parallel = false

[classes.observing]
description = "Real DAQ flow: start.py → record → stop.py → transfer → verify."
required_state = "ACQUIRING"
leaves_state = "ACQ_CONFIGURED"   # stop_acq runs at end
batch_priority = 2
preflight = true
parallel = false

[classes.lifecycle]
description = "Power cycle, firmware reboot, baseline calibration."
required_state = "UNPOWERED"
leaves_state = "BOOTED"
batch_priority = 3
preflight = false
parallel = false

[classes.telemetry]
description = "HK packet pipeline: UDP → capture_hk.py → Redis → InfluxDB."
required_state = "BOOTED"          # quabo emits HK ~3s once booted
leaves_state = "BOOTED"
batch_priority = 0                 # piggy-back on driver_protocol batch
preflight = true
parallel = false                   # shares HK socket on UDP/60002

# ── Test → class mapping (node-id glob) ──
[[mapping]]
glob = "suites/hw0_driver_protocol/**::*"
class = "driver_protocol"

[[mapping]]
glob = "suites/hw1_reconfig/test_interleave_state_switch.py::*"
class = "fast_reconfig"

[[mapping]]
glob = "suites/hw2_observing/test_transfer_queue_e2e.py::*"
class = "observing"

[[mapping]]
glob = "suites/hw3_lifecycle/test_firmware_reboot.py::*"
class = "lifecycle"

[[mapping]]
glob = "suites/hw4_telemetry/**::*"
class = "telemetry"

# ── Topology requirements (config-driven gating, see §5) ──
# Tests that need hardware features absent in the active observatory config
# are skipped (not failed) at collection time.

[[requirements]]
glob = "suites/hw2_observing/test_transfer_queue_multinode.py::*"
requires_min_modules = 2
requires_min_daq_nodes = 2

[[requirements]]
glob = "suites/hw3_lifecycle/test_wr_sync.py::*"
requires_capability = "white_rabbit"

[[requirements]]
glob = "suites/hw3_lifecycle/test_gnss_lock.py::*"
requires_capability = "gnss"

[[requirements]]
glob = "suites/hw0_driver_protocol/test_port_forwarding.py::*"
requires_capability = "port_forwarding"
```

Equivalent in-code marker (sugar; the TOML mapping is authoritative):

```python
@pytest.mark.hw_class("driver_protocol")
def test_hv_set_packet_layout(quabo, hk_socket): ...
```

A pytest plugin (`hw_utils.pytest_plugin`) reads `hw_tests.toml` at collection time and:
1. Attaches a `required_state` marker to each test node from the mapping table.
2. Calls `HwTopology.gate(test_id)` against `[[requirements]]` and skips tests whose preconditions don't match the loaded `obs_config.json` / `daq_config.json` / `network_config.json`.
3. Re-orders tests by `batch_priority` then by `required_state`, grouping into batches.
4. Inserts state-transition primitives between batches as **virtual nodes** in the test report so the developer sees `→ wps_power_on (120 s) → tftp_reboot (60 s) → [batch: lifecycle (4 tests)]`.

---

## 4. Architecture & Key Components

```
hw_utils/
├── state_machine.py     # HardwareStateMachine, Primitive, HwState (loaded from TOML)
├── scheduler.py         # StateAwareScheduler — collects tests, plans batches, runs transitions
├── safety.py            # SafetyManager — atexit hook, signal handlers, emergency_teardown
├── topology.py          # HwTopology — adapts obs/daq/network configs, gates tests
├── driver_ops.py        # configure_maroc, configure_acq, hv_set_from_config, hv_zero (primitive bodies)
├── run_ops.py           # start_run_via_cli, stop_run_via_cli (uses pseti CliRunner)
├── guards.py            # shutter_must_be_closed, light_sensor_dark, ...
├── pytest_plugin.py     # collection hook + state-batch reporter
└── stream.py            # reused from shared/
```

### 4.1 `HardwareStateMachine`
- Loads `hw_state_machine.toml`, builds `nx.DiGraph` of states/primitives.
- `plan(current, target)`: shortest-path (cost = `budget_s.typical`).
- `execute(plan, dry_run=False)`: invokes each primitive's `entrypoint` via `importlib`; checks named `guards` before each; logs each transition with rich.
- Tracks live state in a `state/hw_runtime_state.json` file (so a crashed run can resume without redoing transitions when `--assume-state` is honest).

### 4.2 `StateAwareScheduler`
- Takes the pytest collection items (post-plugin annotation), groups by `(class, batch_priority)`, computes the minimum-cost transition plan between batches.
- Emits a **batch plan** before any test runs:
  ```
  Batch 1 [driver_protocol + telemetry]  (12 tests, target=BOOTED)
    → wps_power_on (120 s)
  Batch 2 [fast_reconfig]                (8 tests, target=ACQ_CONFIGURED)
    → configure_maroc (1 s) → configure_acq (1 s)
  Batch 3 [observing]                    (6 tests, target=ACQUIRING)
    → start_acq (5 s)
  Batch 4 [lifecycle]                    (4 tests, target=UNPOWERED)
    → stop_acq (60 s) → hv_off (2 s) → wps_power_off (5 s)
  Estimated wall clock: 14 min 32 s
  ```
- Honors `--dev` / `--assume-state` / `--no-power-cycle` to skip transitions.

### 4.3 `SafetyManager`
- Registered via `pytest_sessionstart`; installs `atexit` + `signal.SIGTERM/SIGINT` handlers.
- On any exit path (clean, panic, KeyboardInterrupt, OOM), drives the state machine to `safe = "UNPOWERED"`. Reads the live state file to know the starting point.
- Independent of pytest fixtures (which can be skipped in panics) — uses raw `subprocess.run` to invoke the WPS curl directly. Idempotent: safe to invoke twice.
- `--keep-running` opt-in disables the safety stop *and* prints a loud red banner: hardware was not returned to safe state.

### 4.4 `HwTopology`
- Loads active `obs_config.json`, `daq_config.json`, `network_config.json` once per session (uses `control.utils.config_file` — the existing canonical loader).
- Exposes:
  - `quabo_ips() -> list[QuaboAddr]` — flattens all modules × quabos.
  - `daq_nodes() -> list[DaqNode]`
  - `wps_outlets() -> list[WpsOutlet]` — every `wps*` key in obs_config.
  - `capabilities() -> set[str]` — derived: `"white_rabbit"` if any module has `timing_mode == "wr"`, `"gnss"` if any GNSS port configured, `"port_forwarding"` if `network_config.json::port_forwarding` non-empty, etc.
  - `gate(test_id, requirements) -> bool | str` — returns True or a skip reason.
- **Single source of truth**: tests *never* hardcode IPs; they receive a `topology` fixture and iterate.
- Same code drives single-module UCB lab and full Palomar; the only thing that changes is which JSON the symlink points at.

### 4.5 Pytest plugin
- `pytest_collection_modifyitems`: for each item, look up its class in `hw_tests.toml`, attach `item.user_properties['required_state']` and `item.user_properties['leaves_state']`. Apply `pytest.mark.skip` for unmet topology requirements.
- `pytest_runtest_protocol`: intercepted to ensure the scheduler's batch ordering is honored (overrides default file order). Inserts a "transition step" pseudo-test before the first item of each batch so the wall-clock cost shows up in `--durations`.
- `pytest_terminal_summary`: prints the actual batch plan vs estimated, plus per-state time consumed.

---

## 5. CLI Expansion (`pseti test hw`)

All existing subcommands (`build`, `check-env`, `deploy`, `clean`, `down`, `attach`) keep their current behavior. New / changed:

| Subcommand | Purpose |
|---|---|
| `pseti test hw plan` | Dry-run: print the batch plan + estimated wall clock for the current TOML / topology / filter. **No hardware touched.** |
| `pseti test hw run [-k EXPR] [--class CLASS] [--state STATE]` | Existing `run` augmented: `--class driver_protocol` filters to one TOML class; `--state BOOTED` filters to tests requiring at most BOOTED state (lets you say "give me everything that doesn't need a power cycle"). |
| `pseti test hw run --dev` | **Dev mode**: `--no-power-cycle --assume-state=ACQ_CONFIGURED --keep-running`. For tight iteration loops once hardware is warm. Prints loud banner: "DEV MODE — hardware will NOT be returned to safe state". |
| `pseti test hw run --assume-state=STATE` | Skip the implicit "drive to UNPOWERED first" startup; trust the user that hardware is already in `STATE`. |
| `pseti test hw run --no-power-cycle` | Refuse to invoke any primitive tagged `safety="high"`; tests whose required state can only be reached via a high-safety primitive are skipped (not failed) with a clear reason. |
| `pseti test hw run --explain TEST_ID` | Print the state plan a single test would trigger. |
| `pseti test hw preflight` | Run only tests with `preflight = true` in their TOML class. Intended for the pre-observation calibration suite. |
| `pseti test hw status` | Read `state/hw_runtime_state.json` + WPS query, report current believed state + reachability. |
| `pseti test hw safe-down` | Manually invoke the SafetyManager's emergency teardown (driving to `safe` state). |
| `pseti test hw list-classes` | Print TOML classes + how many tests each contains in current collection. |

Default behavior (`pseti test hw run` with no flags): full state-aware batching, drives to `UNPOWERED` at start *and* end, runs all classes in `batch_priority` order. Estimated wall clock printed before the first transition; `--yes` skips the y/n prompt for unattended CI.

---

## 6. Sample Test Suite Designs (no implementation)

For each suite I list: representative tests, fixtures used, what's asserted, what software-only test (if any) it adapts, and the TOML class it lives in.

### 6.1 `hw0_driver_protocol/` — class `driver_protocol`
Adapts the duplicated `FakeSocket` pattern from `tier1_unit/test_quabo_driver_protocol_*.py` to assert against **real FPGA echo responses** instead of a fake socket.

| Test | Fixtures | Assertion |
|---|---|---|
| `test_hv_set_packet_layout` | `quabo`, `hk_socket` | `quabo.hv_set([0,0,0,0])`; sniff next echo packet on UDP/60000; bytes[0]==0x82, [2:10]==0, length==64 |
| `test_acq_params_packet_layout` | `quabo` | Send `send_daq_params(...)`; assert echo at [0]==0x83, mode/interval/hold layout matches wiki |
| `test_maroc_roundtrip` | `quabo` | `send_maroc_params()` twice; on second call, echo's 829-bit shift register matches input |
| `test_calibrate_baseline_returns_256_coeffs` | `quabo` | `calibrate_ph_baseline()`; assert reply is 64 bytes × 8 fragments containing 256 little-endian uint16 |
| `test_data_packet_destination_returns_macs` | `quabo`, `topology` | `data_packet_destination(daq_ip)`; assert 12-byte reply (PH MAC + IM MAC) |
| `test_software_pps_resets_nanosec` | `quabo`, `hk_socket` | Capture HK packet `NANOSEC`; send cmd 0x8f; capture again; assert wraparound |
| `test_software_pps_only_q0` | `quabo`, `topology` | Send 0x8f to Q1/Q2/Q3; assert no NANOSEC effect (per wiki: must go to Q0) |
| `test_port_forwarding_command_path` | `quabo`, `topology` | (Skipped if no port_forwarding capability) Send any command via the gateway IP+port from `network_config.json` and verify the echo arrives — proves the deploy command's port-forwarding rules survive on real network |

### 6.2 `hw1_reconfig/` — class `fast_reconfig`
Adapts `tools/interleave.py` and the `data_config.json` interleave logic. Hardware stays in `ACQ_CONFIGURED`; tests do `configure_acq` + `configure_maroc` reconfigs only.

| Test | Adapts | Assertion |
|---|---|---|
| `test_interleave_state_switch` | `wiki_docs/Interleaving-Observing-Mode-and-Configuration-Validation.md` | Configure `image` mode → switch to `image_8bit` via reconfig → verify HK reflects new ACQMODE within ~3 s |
| `test_trigger_mask_per_channel` | `quabo_config.txt` CHANMASK_* | Set CHANMASK to disable channels {0,1,2,3}; with internal STIM enabled, assert no PH packets emitted from those channels |
| `test_goe_mask_modes` | wiki packet 0x8e | Cycle GOE mask through 0x1, 0x2, 0x3; assert PH trigger threshold semantics (3-pixel, 2-pixel, 1-pixel) using internal stim |
| `test_hv_setpoint_step_response` | (none — new) | Set HV to 30000; wait 5 s; assert HVMON converges within 1 V; set to 0; assert ramp-down |

### 6.3 `hw2_observing/` — class `observing`
Ports the highest-value software-only integration tests to real hardware. Reuses the existing `verify_manifest` / `RunStateManager` / ledger-poll machinery.

| Test | Adapts | Assertion |
|---|---|---|
| `test_full_run_to_archive` | `tier5_integration/test_integration_transfer_queue_validity.py::test_integration_transfer_queue_lifecycle` | Run `pseti start --nsecs 30 --no-hv`; wait for ledger `ARCHIVED`; `verify_integration_transfer_accuracy` byte-checks PFF files on head node match what the daq node generated |
| `test_multi_run_drain` | `test_integration_transfer_advanced.py::test_integration_transfer_queue_drain` | 3 short runs back-to-back; assert all reach `ARCHIVED`, no orphans in `state/transfer/queue/active/` |
| `test_active_daemon_race` | `test_integration_transfer_advanced.py::test_integration_transfer_queue_active_daemon` | Daemon already running; enqueue while polling; assert clean handoff |
| `test_distributed_run_started` | `test_integration_distributed_flows.py::test_when_distributed_run_started_then_all_nodes_recording` | (Skipped if <2 daq nodes per `[[requirements]]`) |
| `test_grpc_streams_real_quabo_data` | `daq_data_hashpipe/integration/test_real_data_flow.py::test_grpc_server_streams_real_hashpipe_data` | DaqDataClient → assert ≥10 PanoImage frames in 15 s, shape matches data product |
| `test_frame_header_fields` | `test_real_data_validation.py::test_frame_header_has_required_fields` | Same assertions as software-only — passes byte-for-byte against real quabos |
| `test_module_id_consistency` | `test_real_data_validation.py::test_module_id_is_consistent_across_frames` | `module_ids` from streamed frames `== set(topology.module_ids())` |
| `test_concurrent_clients_receive_same_frames` | `test_snapshot_grpc_robustness.py::test_concurrent_clients_receive_same_frames` | Two DaqDataClients see identical sequence of frame timestamps |
| `test_cleanup_precondition_enforced` | `test_integration_real_data.py::test_data_collectible_after_stop` | StopDaq → CleanupData with wrong manifest_digest → assert FAILED_PRECONDITION; correct digest → succeeds |
| `test_no_hv_safety_during_run` | (new) | Assert that `--no-hv` flag actually keeps `HVMON*` near zero throughout the run (cross-checks that the test framework's safety contract holds) |

### 6.4 `hw3_lifecycle/` — class `lifecycle`
The expensive tests. Power cycles, firmware reboots, calibration. Run only in nightly CI.

| Test | Adapts | Assertion |
|---|---|---|
| `test_full_power_cycle` | (new) | `wps_power_off → wps_power_on`; assert ping responsive within 300 s; first HK packet has `bootbyte == 0xaa`, subsequent have `bootbyte == 0x00` |
| `test_tftp_reboot` | `quabo_tftp.tftpw.reboot` | `tftp_reboot`; assert ping back within 90 s; FWVER unchanged |
| `test_firmware_load_and_reboot` | `quabo_tftp.put_bin_file` | (Manual / opt-in only — actually flashes firmware) Load known-good firmware bin from `firmware.json`; reboot; assert FWVER and FWTIME match expected |
| `test_baseline_calibration` | `wiki_docs/Quabo-command-line-interface.md` `B` cmd | After boot, run `calibrate_ph_baseline` for each quabo; assert all 256 coefficients in plausible range (e.g., 0–4095) |
| `test_uid_stability_across_reboot` | (new) | Capture UID before reboot; reboot; assert UID identical (DS18B20 chip ID) |
| `test_session_start_full` | `session_start.py` | Full `pseti session-start --no-hv`; assert all daemons running; assert HK Redis HASH populated for every quabo |

### 6.5 `hw4_telemetry/` — class `telemetry`
HK packet validation at three layers (low / Redis / InfluxDB). Designed to be many small assertions, not few large ones.

#### Low-level (raw UDP/60002 binding — uses `hw_assertions.HKPacketParser`):
| Test | Assertion |
|---|---|
| `test_hk_magic_byte` | bytes[0] == 0x20 |
| `test_hk_bootbyte_first_after_boot` | After `wps_power_on`, exactly one packet has bytes[1] == 0xaa, all subsequent have 0x00 |
| `test_hk_boardloc_matches_obs_config` | bytes[2:4] LE == `(aperture_id << 2) | quadrant` for every quabo in topology |
| `test_hk_inter_packet_interval` | Sliding-window over 30 s of packets; mean delta == 3 s ± 0.5 s |
| `test_hk_packet_length_exact` | `recvfrom(64)` returns exactly 64 bytes; `recvfrom(128)` returns 64 |
| `test_hk_uid_matches_quabo_uids_json` | Decoded UID hex == `quabo_uids.json` entry |
| `test_hk_pcb_revision` | byte[53] bit 0 == `obs_config` hardware revision field |
| `test_hk_fwver_matches_firmware_json` | Decoded FWVER ASCII == `firmware.json` expected for that hardware version |
| `test_hk_ext_status_consistency` | If EXT_10MHz_STATUS == 0, EXT_1PPS_STATUS must == 0 (per capture_hk.py:151 invariant) |
| `test_hk_packet_count_per_quabo` | Over 30 s window, each quabo emits 9–11 packets (3 s cadence × 4 quabos × 1 module) |

#### Mid-level (Redis HASH lookup — runs `capture_hk.py` daemon, queries `QUABO_<boardloc>`):
| Test | Assertion |
|---|---|
| `test_redis_populated_for_all_quabos` | Every quabo in topology has a `QUABO_<boardloc>` HASH within 10 s of daemon start |
| `test_redis_voltage_rails_in_spec` | V12MON ≈ 1.20 ± 0.06 V; V18MON ≈ 1.80 ± 0.09 V; V33MON ≈ 3.30 ± 0.17 V; V37MON ≈ 3.70 ± 0.19 V |
| `test_redis_currents_in_spec` | I10MON, I18MON, I33MON within board-typical ranges |
| `test_redis_temperatures_plausible` | TEMP1 ∈ [-10, 60] °C; TEMP2 ∈ [20, 85] °C; VCCINT ≈ 1.0 V; VCCAUX ≈ 1.8 V |
| `test_redis_hv_off_state` | With HV not commanded, |HVMON0..3| < 1 V |
| `test_redis_hv_on_state` | After `hv_set([V,V,V,V])`, |HVMON_x − V_in_volts| < 1 V within 5 s (parametrized over channels and setpoints) |
| `test_redis_detector_current_offset_corrected` | DETR0..3_CURR (the post-`get_true_detector_current` value) ≈ 0 with HV off (proves the offset math from capture_hk.py:56) |
| `test_redis_startup_flag_once` | After power cycle, exactly one Redis update has `StartUp == 1`; subsequent updates have `StartUp == 0` |
| `test_redis_computer_utc_monotonic` | Successive `Computer_UTC` for a given quabo strictly increase |
| `test_redis_shutter_state_reflects_command` | Send `shutter_new(closed=True)`; within 5 s, Redis `SHUTTER_STATUS == 0` (matches command) |
| `test_redis_uid_stable_across_packets` | Over 60 s, UID for each quabo is constant |

#### High-level (end-to-end InfluxDB — requires `storeInfluxDB.py` daemon, queries `metadata` DB):
| Test | Assertion |
|---|---|
| `test_influx_measurement_per_quabo` | `SHOW MEASUREMENTS` includes `QUABO_<boardloc>` for every quabo |
| `test_influx_point_count_matches_cadence` | Over 60 s, count(points) per measurement ≈ 20 ± 2 (3 s cadence) |
| `test_influx_dedup_correct` | No two points with identical `Computer_UTC` (proves capture_hk's `key_timestamps` dedup) |
| `test_influx_tags_correct` | Every point tagged `observatory=<obs_name>`, `datatype=housekeeping` |
| `test_influx_field_types` | Numeric HK fields are float in Influx; UID/FWVER/FWTIME are strings (proves `get_casted_redis_value` correctness) |
| `test_influx_hv_step_response_visible` | Step HV from 0 → 30000 → 0; assert the resulting HVMON time series shows the step within 2 polling intervals |
| `test_influx_continues_through_run_lifecycle` | HK time series uninterrupted across `pseti start` / `pseti stop` (HK is independent of DAQ run state) |
| `test_influx_stops_on_power_off` | After `wps_power_off`, last HK point is within 5 s of the power-off command |

---

## 7. `TEST-HW-SW.md` — Documentation Outline

Complete rewrite, structured as:

1. **Overview** — what HITL means for PANOSETI; the state-machine philosophy (cold/warm/hot, primitive cost, batching wins).
2. **The Hardware State Machine** — diagram, every state defined, every primitive table (from `hw_state_machine.toml`).
3. **The Test TOML** — class definitions, mapping rules, requirements, parallel constraints. Example: a complete TOML walk-through.
4. **CLI Reference** — `pseti test hw plan/run/preflight/status/safe-down/list-classes/explain`. Examples for each.
5. **Dynamic Topology** — how `HwTopology` reads `obs/daq/network_config.json` and what gates a test. Same TOML drives single-module UCB lab and full Palomar — this section is the proof.
6. **Hardware Safety** — `SafetyManager`, the atexit/signal contract, `--keep-running` opt-out, what to do if you trip the safety net.
7. **Dev/Debug Workflows** — the `--dev`, `--assume-state`, `--no-power-cycle` recipes; how to iterate on a single test in seconds once hardware is warm.
8. **Adding a New Test** — the 4-step recipe:
   1. Pick a class (or define one in `hw_tests.toml`).
   2. Place the test under `suites/hwN_<class>/`.
   3. Use existing fixtures (`quabo`, `topology`, `hk_socket`, `wps_outlet`, `redis_client`, `influx_client`) — never hardcode an IP.
   4. (Optional) add `[[requirements]]` if the test needs hardware features not present everywhere.
9. **Pre-flight Suite** — what the `preflight = true` subset is for, how it fits into the observing-night workflow.
10. **Troubleshooting** — common failure modes (stale state file, WPS unreachable, port-forwarding misconfig), how to diagnose with `pseti test hw status`.
11. **Migration Notes** — what moved from `ci/conftest.py` to `ci/software-only/conftest.py`, why the split.

---

## 8. Critical Files

**To create**:
- `control/ci/hardware-software/hw_state_machine.toml`
- `control/ci/hardware-software/hw_tests.toml`
- `control/ci/hardware-software/hw_utils/{state_machine,scheduler,safety,topology,driver_ops,run_ops,guards,pytest_plugin,stream}.py`
- `control/ci/hardware-software/hw_assertions.py`
- `control/ci/hardware-software/fixtures/{packet_capture,quabo_fixtures,wps_fixtures,telemetry_fixtures}.py`
- `control/ci/hardware-software/suites/hw{0..4}_*/test_*.py`
- `control/ci/shared/{qa_models,stream}.py`

**To modify**:
- `control/ci/test_cli.py:325-589` — replace inline `hw_*` Typer commands with delegations to `hw_utils.cli` (keeps the public CLI surface).
- `control/ci/qa.toml` — narrow scope; the `[suites.test-hw]` block becomes a thin pointer to the new HITL runner.
- `control/ci/qa_utils.py` — extract `QAConfig`/`SuiteConfig`/`EnvironmentConfig` and `_stream` to `control/ci/shared/`; leave Docker orchestration in place.
- `control/ci/conftest.py` — split: HITL-incompatible logic (`auto_isolate`'s PSETI_* rewrites, `session_fleet`, `_generate_dynamic_env`) moves to `control/ci/software-only/conftest.py`; root keeps only the truly shared bits.
- `control/TEST-HW-SW.md` — full rewrite per §7.
- `control/CLI.md` — update the `pseti test hw` block to list the new subcommands.

**To reuse (no changes)**:
- `control.power.quabo_power` (and `do_all`) — primitive entrypoints.
- `control.driver.quabo_driver.QUABO` — primitive entrypoints + driver_protocol assertions.
- `control.driver.quabo_tftp.tftpw` — `tftp_reboot` / `tftp_load_firmware` primitives.
- `control.utils.config_file` — single canonical config loader for `HwTopology`.
- `control.daemons.capture_hk.{storeInRedis,get_true_detector_current,signed}` — referenced by `hw_assertions.HKPacketParser` (do not re-implement).
- `control.daemons.capture_hk.panosetiSIconvert.HKconvert` — re-used for raw→SI conversion in low-level assertions.
- `panoseti_grpc.daq_data.client.DaqDataClient` — used by `hw2_observing` tests.
- `panoseti_grpc.daq_control.client` — used by `hw2_observing` lifecycle tests.
- `control.utils.transfer.verify.verify_manifest` — used by `hw2_observing` integrity tests.
- `tier5_integration/transfer_integration_utils.{generate_integration_run, verify_integration_transfer_accuracy}` — promoted to `ci/shared/transfer_helpers.py` (used by both software-only and HITL).
- `tier1_unit/test_quabo_driver_protocol_*.py` `FakeSocket` pattern — promoted to `fixtures/packet_capture.py` so both software-only and HITL tests share assertion helpers.

---

## 9. Verification Plan (when implementation lands)

1. **Schema sanity**: `pseti test hw plan` with no filter on the UCB lab config; expected output is the batch plan with 4 batches and a wall-clock estimate of ~10–15 min.
2. **Topology adapter**: temporarily symlink `configs/` to a synthetic 4-module config; run `pseti test hw plan`; verify multi-module tests are no longer skipped and quabo_ips() returns 16 entries.
3. **Safety**: `pseti test hw run --class lifecycle`; mid-run, `kill -9` the pytest process; verify `state/hw_runtime_state.json` reflects last known state and the `SafetyManager`'s atexit hook fires `wps_power_off`. Confirm via WPS query.
4. **Dev mode wall-clock**: with hardware warm in `ACQ_CONFIGURED`, time `pseti test hw run --dev --class fast_reconfig -k test_interleave_state_switch`. Should complete in <30 s end-to-end.
5. **Cold-start wall-clock**: `pseti test hw run` from `UNPOWERED`. Expected ≈ estimated ± 10%.
6. **HK assertions vs UCB rig**: `pseti test hw run --class telemetry`. Verify all three layers (raw UDP, Redis, InfluxDB) pass on the lab rig.
7. **Software-only regression**: `pseti test sw all` after the conftest split. All 538 unit tests + integration suite continue to pass with no HITL bleed-over.
8. **Lint**: `pseti test lint` clean (Ruff + MyPy).
9. **Docs**: `pseti test hw --help` matches `TEST-HW-SW.md` examples.

---

## 10. User-Confirmed Decisions

1. **Refactor scope**: Atomic — conftest split lands together with the HITL framework in one PR. Implies extra verification load (run full `pseti test sw all` before merge) but no half-migrated state.
2. **CLI surface**: Flag form with short aliases. `pseti test hw run -c|--class CLASS` and `-s|--state STATE`. Pytest-flag collision avoidance: pytest uses `-k` (keyword), `-m` (marker), `-x` (exitfirst), `--co` (collect-only) — `-c`/`-s` are free in pytest, but to be safe the Typer CLI will use `--hw-class` / `--hw-state` as the long forms when extra args (`ctx.args`) are passed through to pytest, with `-c` / `-s` reserved as the Typer short aliases that are stripped before delegation. Class names stay in TOML only (no subcommand-per-class).
3. **Firmware test**: `test_firmware_load_and_reboot` is included but gated behind `--allow-firmware-flash` at the CLI level; without the flag, the test is collected and explicitly skipped with a clear reason. The TOML `[[requirements]]` block gets a new `requires_flag = "allow-firmware-flash"` field for general extensibility.

### Still TBD (flag for the upcoming hardware-lead meeting, not blocking)
- Whether `WR_SYNCED` and `GNSS_LOCKED` should be first-class states or attributes of `BOOTED`.
- Whether `QUABO.reset()` (cmd 0x04) is safe to chain after `HV_ON` without an explicit `hv_off` first.
- Whether `preflight` should graduate to `preflight_basic` / `preflight_full`.
