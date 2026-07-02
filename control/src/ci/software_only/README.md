# software_only — PANOSETI v2 Test Platform

This tree replaces `ci/software_only/` (v1). It provides a 5-tier test suite driven by a declarative `FleetSpec` DSL, testcontainers for tier 3/4, and a first-class `Chaos` accessor for fault injection. gRPC is never mocked — always real.

See [SUNSET.md](SUNSET.md) for the v1 deletion timeline and checklist.

---

## Quick Start

```bash
pseti test sw v2 unit        # Tier 1: fast logic (no Docker)
pseti test sw v2 logic       # Tier 2: isolated workspace (no Docker)
pseti test sw v2 fleet       # Tier 3: testcontainers fleet
pseti test sw v2 chaos       # Tier 4: fault injection
pseti test sw v2 integration # Tier 5: real Hashpipe + tcpreplay (needs compose stack)
```

---

## Tier table

| Tier | Dir | Docker? | Key fixture |
|---|---|---|---|
| 1 | `tier1_unit/` | No | `pseti_workspace` |
| 2 | `tier2_logic/` | No | `pseti_workspace` |
| 3 | `tier3_fleet/` | Yes (testcontainers) | `session_fleet` |
| 4 | `tier4_chaos/` | Yes (testcontainers) | `chaos_fleet` + `chaos` |
| 5 | `tier5_integration/` | Yes (static compose) | `session_fleet` (compose-backed) |

---

## Directory map

```
software_only/
├── conftest.py              # tier autouse isolation
├── pytest.ini               # markers: tier1..tier5
├── qa.toml                  # suite runners and compose config
├── SUNSET.md                # v1 deletion checklist
│
├── infra/                   # importable (non-fixture) building blocks
│   ├── spec.py              # FleetSpec DSL + Topology
│   ├── synth.py             # FleetSpec → 7 Pydantic configs + GlobalConfigValidator
│   ├── materialize.py       # Pydantic configs → JSON files in PSETI_CONFIG
│   ├── workspace.py         # Workspace dataclass + StateProbe
│   ├── corpus.py            # PFFCorpus — PFF test data discovery + synthesis
│   └── parity.py            # v1↔v2 equivalence registry (for sunset gate)
│
├── containers/              # testcontainers blueprints (DockerContainer subclasses)
│   ├── base.py              # PsetiContainer base (env / volume / healthcheck DSL)
│   ├── headnode.py          # HeadnodeContainer
│   ├── daqnode_sim.py       # DaqNodeSimContainer (pseti-grpc server --profile daq_node)
│   ├── mock_quabo.py        # MockQuaboContainer
│   ├── gateway.py           # socat port-forward container
│   └── telemetry.py         # Redis + InfluxDB + Loki sidecar
│
├── orchestrator/            # fleet lifecycle
│   ├── fleet.py             # Fleet: FleetSpec → live containers + typed handles + Chaos
│   ├── network.py           # SharedNetwork + per-worker subnet shifting
│   └── lifecycle.py         # start / wait_healthy / tear_down
│
└── fixtures/                # one pytest fixture per concern
    ├── workspace.py         # pseti_workspace, pseti_workspace_session
    ├── fleet.py             # session_fleet, daq_control_client, daq_data_client
    ├── chaos.py             # (shadowed by chaos/ package; see chaos/__init__.py)
    └── chaos/               # fault-injection sub-modules + Chaos accessor
        ├── __init__.py      # Chaos, NetemHandle, IptablesHandle, DiskHandle,
        │                    #   ProcessHandle, GrpcHandle — import from here
        ├── netem.py         # tc-netem helpers
        ├── iptables.py      # iptables blackhole helpers
        ├── disk_chaos.py    # ENOSPC simulation
        ├── process_chaos.py # kill / freeze / wait helpers
        ├── grpc_proxy.py    # GrpcChaosProxy — in-process fault injection
        └── clock_chaos.py   # CLOCK_REALTIME manipulation
```

---

## Fixture catalog

### `pseti_workspace` (function-scoped)

Redirects all `PSETI_*` env vars to a tmp dir, materializes all 7 config files from a `FleetSpec`, and runs `GlobalConfigValidator.validate_all()` at setup. No raw-dict configs, no bypassed validators.

```python
# Default (minimal_unit spec)
def test_something(pseti_workspace: Workspace) -> None: ...

# Parametrized
@pytest.mark.parametrize("pseti_workspace", [FleetSpec.minimal_fleet()], indirect=True)
def test_fleet(pseti_workspace: Workspace) -> None: ...
```

### `session_fleet` (module-scoped, Tier 3/4)

Boots a `Fleet` of `HeadnodeContainer` + N×`DaqNodeSimContainer`, waits for gRPC health (`grpc_health_probe`), yields the `Fleet` handle, then tears down. Requires Docker.

```python
def test_status(session_fleet: Fleet) -> None:
    client = session_fleet.daq_control_client(0)
    resp = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True,
                              "check_disk_usage": False, "check_run_dirs": False})
```

### `chaos` / `chaos_fleet` (Tier 4)

`chaos_fleet` is the same as `session_fleet` but module-scoped for the chaos tier. Access fault injection via `fleet.chaos.<handle>`:

```python
def test_kill(chaos_fleet: Fleet) -> None:
    node = chaos_fleet.daq_nodes[0]
    chaos_fleet.chaos.proc.kill(node, "pseti-grpc", sig="TERM")
    assert chaos_fleet.chaos.proc.wait_dead(node, "pseti-grpc", timeout=10)
```

Sub-handles: `chaos.net` (tc-netem), `chaos.iptables`, `chaos.disk`, `chaos.proc`, `chaos.grpc`.

---

## v1 → v2 fixture mapping

| v1 fixture | v2 replacement |
|---|---|
| `auto_isolate` | `pseti_workspace` |
| `mock_workspace` | `pseti_workspace` |
| `isolated_transfer_env` | `pseti_workspace` |
| `chaos_headnode_workspace` | `pseti_workspace` + `FleetSpec.minimal_fleet()` |
| `session_fleet` (v1) | `session_fleet` (v2, module-scoped) |
| `daq_client` / `daq_control_direct` | `fleet.daq_control_client(0)` |
| `daq_data_client` | `fleet.daq_data_client(0)` |
| `make_mock_daq_config` | `FleetSpec.build().topology.daq_config` |
| `make_transfer_job` | `FleetSpec.build()` + materialize |
| `RsyncMock` | real rsync against container FS; `chaos.disk` for negative tests |
| `dummy_data_generator` | `PFFCorpus.make_synthetic()` |

---

## Adding new tests

1. Put new tests in `tier{N}/` under this directory, not under `software_only/` (v1 is in sunset).
2. Use `pseti_workspace` for any test that touches configs — do not call `os.environ` directly.
3. Tier 3/4 tests need Docker available; mark them `@pytest.mark.tier3` / `@pytest.mark.tier4`.
4. Keep module-scoped fixtures as `scope="module"` — they boot real containers and are expensive.
5. Register a `@parity_test` entry in `infra/parity.py` for every test that replaces a v1 test. The sunset gate checks this registry.
