# Plan: software_only_v2 — modernized control-plane test infrastructure

## Context

The current `control/src/ci/software_only/` test tree (tier1_unit … tier5_integration) is brittle. Survey of `ci/fixtures/` and `ci/shared/` identified concrete debt:

- **4 redundant DaqConfig builders**: `factories.make_mock_daq_config` (Pydantic), `chaos_fixtures.chaos_headnode_workspace` (raw dict), `transfer_fixtures.transfer_job_factory`, `fleet.make_fleet`.
- **3 rsync-mock paths**: `rsync_fixtures.RsyncMock`, `mocks._mock_subprocess_ok/_fail`, `workspace_fixtures.copy_run_dir`.
- **3 isolated-env helpers**: `workspace_fixtures.mock_env`, `transfer_fixtures.isolated_transfer_env`, `shared.transfer_helpers.setup_isolated_integration_transfer_env`.
- **Hand-rolled raw-dict configs** in `chaos_fixtures` and parts of `mock_workspace` bypass Pydantic; one signature change cascades into hand-edits across many tests.
- **Pure-alias fixtures** (`daq_control_direct` = `daq_client`) and **redundant data generators** (`simulate_daq_filesystem` vs `dummy_data_generator`).
- **Path discovery duplication**: `state_probe.py` re-implements env-var lookup instead of using `PanoPaths`.
- The current `ci/fixtures/build/fake_hashpipe.py` is a stub that writes 16 bytes and sleeps — useless for realistic data-plane simulation. Meanwhile `panoseti_grpc.daq_data.simulate.UdsStrategy` is a production-quality fake hashpipe replaying real Lick PFF frames at 10 Hz.
- **Real-world consequence**: tests run with a *minimal* environment that diverges from production. When a function signature changes or a new config field is added, the minimal mock context is incomplete and tests break in surprising ways, requiring hand-edits across dozens of files. Tests test the mocks, not the system.

The HITL suite at `control/src/ci/hardware_software/` is now passing and must stay untouched.

**Goal**: a v2 platform where (a) tests declare a topology in code, (b) realistic configs and per-node testcontainers are generated automatically, (c) gRPC code runs for real (never mocked), (d) only Quabos / hashpipe / network primitives are simulated, and (e) fault injection is a first-class accessor on the fleet handle. Outcome: a tool that helps developers and AI agents probe edge cases instead of fighting boilerplate.

User-confirmed decisions:
1. **Phased sunset** of v1: v2 lives alongside v1 at `control/src/ci/software_only_v2/`. Once every v1 tier has a passing v2 parity test in CI for ~7 days, delete v1.
2. **PFF corpus**: reuse the corpus bundled inside `panoseti_grpc` (its `pyproject.toml` line 86 ships `simulated_data_dir/**/*` as package data). Access via `importlib.resources.files("panoseti_grpc.daq_data") / "simulated_data_dir"`. Allow override through a global test-config TOML or fixture parameter.
3. **Tier 5**: keep the hand-written `docker-compose.integration.yml`. Defer a FleetSpec→compose emitter.

---

## Architecture

### 1. Directory layout

```
control/src/ci/software_only_v2/
├── conftest.py                      # tier-aware autouse isolation; loads pytest_plugins
├── README.md
├── pytest.ini                       # markers: tier1, tier2, tier3, tier4, tier5
├── qa.toml                          # global v2 test config (corpus path override, defaults)
│
├── infra/                           # importable from any tier — NOT pytest fixtures
│   ├── spec.py                      # FleetSpec DSL (declarative topology)
│   ├── synth.py                     # FleetSpec → 7 Pydantic configs (+ runs GlobalConfigValidator)
│   ├── materialize.py               # Pydantic configs → JSON files in PSETI_CONFIG; ensure_state_dirs()
│   ├── workspace.py                 # Workspace dataclass; rebuilt StateProbe on PanoPaths
│   ├── corpus.py                    # PFFCorpus — wraps importlib.resources lookup + synthetic generator
│   └── parity.py                    # v1↔v2 equivalence harness used during sunset
│
├── containers/                      # testcontainers blueprints
│   ├── base.py                      # PsetiContainer base (env/volumes/healthcheck DSL)
│   ├── headnode.py                  # HeadnodeContainer
│   ├── daqnode_sim.py               # DaqNode using panoseti_grpc.daq_data.simulate.UdsStrategy
│   ├── daqnode_real.py              # DaqNode with real hashpipe (Tier 5 only)
│   ├── module_sim.py                # CollapsedModuleContainer (Tier 3-lite)
│   ├── mock_quabo.py                # wraps existing ci/mock_quabo/ (lifted as-is)
│   ├── gateway.py                   # alpine/socat router from NetworkConfig.daq_nodes[*].port_forwarding
│   └── telemetry.py                 # Redis + InfluxDB + Loki triplet
│
├── orchestrator/
│   ├── fleet.py                     # Fleet: FleetSpec → live containers + typed handles + Chaos accessor
│   ├── network.py                   # SharedNetwork + per-xdist-worker subnet shifting
│   └── lifecycle.py                 # start / wait_healthy (grpc_health_probe) / tear_down
│
├── fixtures/                        # ONE pytest fixture per concern — no duplication
│   ├── workspace.py                 # pseti_workspace (replaces 4 v1 fixtures)
│   ├── fleet.py                     # session_fleet
│   ├── chaos.py                     # exposes existing fixtures/chaos/* via Chaos accessor
│   ├── corpus.py                    # pff_corpus
│   ├── clients.py                   # real DaqControl/DaqData clients (NOT mocked)
│   └── state_probe.py               # rebuilt on PanoPaths
│
├── tier1_unit/                      # zero containers; in-process Pydantic + GlobalConfigValidator
├── tier2_logic/                     # state-machine logic; isolated workspace; no containers
├── tier3_fleet/                     # Fleet of sim daqnodes; UdsStrategy
├── tier4_chaos/                     # Tier 3 + chaos toolkit
└── tier5_integration/               # real hashpipe + tcpreplay (drives docker-compose.integration.yml)
```

Production code touched (small extensions):
- `control/src/control/topology/fleet.py`: add `seed: int | None` parameter (eliminate unseeded `random` in `generate_fleet_configs`); add `generate_data_config(...)`, `generate_firmware_config(...)`, `generate_daemons_config(...)` so all 7 configs are emitted from one place.

### 2. The Topology DSL (`infra/spec.py`)

Tests declare topology in Python; `FleetSpec.build()` produces a frozen `Topology` containing 7 Pydantic models + an `nx.DiGraph` from `GraphBuilder.build_from_configs`. `GlobalConfigValidator.validate_all()` runs at `build()` time so invalid topologies fail before any container starts.

```python
spec = (
    FleetSpec(seed=42, name="two_dome_mixed_timing", tier="tier3")
        .with_headnode(ip="10.0.1.5", data_dir="/data/head")
        .add_dome(name="dome0", lat=37.342, lon=-121.637, alt=1283.0)
            .add_module(id=200, version="qfp", timing="wr",   ip="192.168.3.32")
            .add_module(id=201, version="bga", timing="gnss", ip="192.168.3.36")
        .add_dome(name="dome1", lat=37.343, lon=-121.638, alt=1283.0)
            .add_module(id=202, version="qfp", timing="wr",   ip="192.168.3.40")
        .add_daq_node(ip="192.168.0.10", modules=[200, 201], gateway=None)
        .add_daq_node(ip="192.168.0.20", modules=[202],
                      gateway=Gateway(ip="10.200.146.13", grpc_port=50051))
        .with_data(run_type="sci-data", overvoltage=2,
                   image=ImageMode(integration_time_usec=200, quabo_sample_size=16))
        .with_firmware(qfp="qfp_v3.bin", bga="bga_v2.bin", gold="gold.bin")
        .build()
)
```

Knobs (every one drives a `GlobalConfigValidator._check_*` rule):
- `seed: int` — deterministic randomization.
- `quabo_version_mix={"qfp": 0.5, "bga": 0.5}` → drives `_check_hardware_firmware`.
- `timing_mix={"wr": 0.7, "gnss": 0.3}` → drives `_check_timing_port_collisions`.
- `gateway_probability=0.5` → drives `_check_daq_module_subnet_coherence`, `_check_port_collisions`.
- `multi_daq_modules=False` → flip to True to provoke `_check_daq_assignment_overlap`.
- `wps_units={"wps0": "http://...", ...}` → drives `_check_wps_references`.
- `tier ∈ {tier1, tier2, tier3, tier3-lite, tier4, tier5}` → selects collapsed vs separated container shape.
- Convenience: `FleetSpec.minimal_unit()`, `.minimal_fleet()`, `.from_palomar()` factory methods.

### 3. Container blueprint catalog

All blueprints inherit `PsetiContainer(testcontainers.core.container.DockerContainer)` standardizing env, volumes, and healthcheck via `grpc_health_probe -addr :50051 -service panoseti.daq_control` (per `grpc/CLAUDE.md`).

| Blueprint | Image (Dockerfile.ci stage) | Process | Knobs |
|---|---|---|---|
| `MockQuaboContainer` | existing `ci/mock_quabo/Dockerfile` (lift as-is) | asyncio UDP 60000–60003 + HK on 60002 | `module_id`, `hk_dest_ip`, `science_emission ∈ {silent, idle, replay-pcap}`, `pcap_path`, `ip_alias_count=4` |
| `DaqNodeContainer(sim)` | `headnode` stage | `panoseti-server --profile daq_node` + `SimulationManager` driving `UdsStrategy` | `pff_corpus`, `replay_rate_hz=10.0`, `module_ids`, `frame_limit=-1` |
| `DaqNodeContainer(real)` | `integration-daqnode` stage | real `hashpipe.so` via `pseti-grpc server` | Tier 5 only; `shm_size=2GB`, `cap_add=[NET_RAW,NET_ADMIN,IPC_LOCK,SYS_NICE]` |
| `CollapsedModuleContainer` | `headnode` stage | 4 in-process MockQuabos + sim daqnode; one unified gRPC server | `module_ids`, replay rate. Tier 3-lite default for cheap fleets. |
| `HeadnodeContainer` | `headnode` stage | `panoseti-server --profile headnode` + transfer daemon | `enable_transfer_daemon`, `enable_loki` |
| `GatewayContainer` | `alpine/socat` | port-forward map computed from `NetworkConfig.daq_nodes[*].port_forwarding` | mirrors `ci/fixtures/build/gateway_setup.sh` |
| `TelemetryContainer` | redis:alpine + grafana/loki + influxdb:1.8 | sidecar triplet | optional per-tier |

Replaces the v1 stub `ci/fixtures/build/fake_hashpipe.py` entirely — `daqnode_sim` uses `panoseti_grpc.daq_data.simulate.UdsStrategy` instead.

### 4. Fleet orchestrator (`orchestrator/fleet.py`)

Replaces `control/src/ci/fixtures/fleet.py`. **Keep**: `SharedNetwork`, `setup_docker_host()`, the boot-and-discover port-mapping pattern (`port_forwarding.gw_ip = container_host_ip`), two-phase TCP+grpc healthcheck. **Rewrite**: spec construction (now `FleetSpec`-driven), typed handles, lifecycle.

```python
class Fleet:
    def __init__(self, topology: Topology, workspace: Workspace, *,
                 tier: str, telemetry: bool = False): ...
    def start(self) -> None: ...
    def wait_healthy(self, timeout: float = 90) -> None: ...
    def tear_down(self) -> None: ...

    @property
    def headnode(self) -> HeadnodeContainer: ...
    @property
    def daq_nodes(self) -> list[DaqNodeContainer]: ...
    @property
    def modules(self) -> dict[int, MockQuaboContainer | CollapsedModuleContainer]: ...
    @property
    def gateways(self) -> list[GatewayContainer]: ...
    @property
    def telemetry(self) -> TelemetryContainer | None: ...
    @property
    def chaos(self) -> Chaos: ...
```

Per-xdist-worker subnet shifting in `orchestrator/network.py` derives a `/24` shift from `TC_SESSION_ID` (already populated by `software_only/conftest.py:65`); `synth.py` consumes it so two parallel workers don't fight over `192.168.3.x`.

testcontainers for tiers 3/4 (dynamic per-test fleets); `docker-compose.integration.yml` for tier 5 (real hashpipe — too expensive to spin per test).

### 5. Unified workspace fixture (`fixtures/workspace.py`)

Replaces all of: `mock_env`, `mock_workspace`, `isolated_transfer_env`, `chaos_headnode_workspace`, `setup_isolated_integration_transfer_env`.

```python
@pytest.fixture
def pseti_workspace(request, tmp_path, monkeypatch) -> Workspace:
    spec: FleetSpec = getattr(request, "param", FleetSpec.minimal_unit())
    topology = synth.realize(spec)            # validates via GlobalConfigValidator

    for key, sub in [("PSETI_CONFIG", "configs"), ("PSETI_STATE", "state"),
                     ("PSETI_TMP", "tmp"), ("PSETI_LOGS", "state/logs"),
                     ("PSETI_QUABOS", "quabos"), ("PSETI_FIRMWARE", "firmware"),
                     ("HEAD_DATA_DIR", "head_data"), ("DAQ_DATA_DIR", "daq_data")]:
        path = tmp_path / sub
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(key, str(path))

    materialize.write_all(topology, PanoPaths.config_dir())   # all 7 JSON files
    PanoPaths.ensure_state_dirs()
    importlib.reload(control.utils.config_file)               # pick up new env

    return Workspace(
        root=tmp_path, topology=topology, paths=PanoPaths,
        state_probe=StateProbe(paths=PanoPaths),
    )
```

Always writes all 7 configs (`obs_config.json`, `daq_config.json`, `network_config.json`, `data_config.json`, `firmware.json`, `quabo_uids.json`, `daemons.json`). No raw-dict path. Realism gate: a fresh workspace must satisfy `GlobalConfigValidator.validate_all()`.

`StateProbe` is rewritten on `PanoPaths` — eliminates `state_probe.py:19-20` hardcoded `HEAD_DATA_DIR`/`DAQ_DATA_DIR`.

### 6. Fixture catalog & cleanup (delete plan)

| v2 fixture | Scope | Returns | Replaces (v1) |
|---|---|---|---|
| `pseti_workspace` | function | `Workspace` | `mock_env`, `mock_workspace`, `isolated_transfer_env`, `chaos_headnode_workspace` |
| `pseti_workspace_session` | session | shared read-only `Workspace` | `auto_isolate` |
| `topology` | function | `Topology` | `topology_fixtures.topology` (now Pydantic-derived) |
| `pff_corpus` | session | `PFFCorpus` | n/a (new) |
| `session_fleet` | session | `Fleet` | `session_fleet`, `Fleet`, `make_fleet`, `DaqnodeSpec` |
| `daq_control_client` | function | real `DaqControlClient` | `daq_client`, `daq_control_direct`, `daq_control_node2` (alias-only fixtures gone) |
| `daq_data_client` | session | real `DaqDataClient` | `data_client`, `daq_data_client` |
| `state_probe` | function | `StateProbe` | rewritten `state_probe.py` |
| `chaos` | function | `Chaos(fleet)` | `chaos_fixtures.*` (low-level kept; surface unified) |

**Deleted at sunset**: `factories.make_mock_daq_config`, `factories.make_transfer_job`, `factories.simulate_daq_filesystem`, `data_fixtures.dummy_data_generator`, `data_fixtures.mock_daq_filesystem`, `rsync_fixtures.RsyncMock`, `mocks._mock_subprocess_ok/_fail`, `workspace_fixtures.copy_run_dir`, `transfer_helpers.setup_isolated_integration_transfer_env`, `client_fixtures.daq_control_direct`/`daq_control_node2` aliases.

Three rsync-mock paths collapse to **zero** — Tier 3+ runs real rsync against real container fs. A single `chaos.rsync` injector is used only for negative tests.

### 7. Fault-injection toolkit

Keep `control/src/ci/fixtures/chaos/{netem,iptables,disk_chaos,clock_chaos,grpc_proxy,process_chaos}.py` intact. Wrap in `Chaos` accessor:

```python
class Chaos:
    def __init__(self, fleet: Fleet): self._fleet = fleet
    @property
    def net(self) -> NetemChaos: ...
    @property
    def iptables(self) -> IptablesChaos: ...
    @property
    def disk(self) -> DiskChaos: ...
    @property
    def clock(self) -> ClockChaos: ...
    @property
    def proc(self) -> ProcessChaos: ...
    @property
    def grpc(self) -> GrpcProxyChaos: ...

# usage
fleet.chaos.net.add_latency(fleet.daq_nodes[0], "200ms")
fleet.chaos.proc.kill(fleet.headnode, "transfer_daemon")
```

Each chaos module already takes a container name + command; the wrapper just feeds in the right typed handle from the Fleet.

### 8. PFF corpus & generator (`infra/corpus.py`)

```python
class PFFCorpus:
    """Lick observatory PFF corpus, sourced from panoseti_grpc package data."""
    def __init__(self, root: pathlib.Path | None = None):
        if root is None:
            cfg = load_v2_test_config()                 # qa.toml override
            root = cfg.pff_corpus_path or (
                importlib.resources.files("panoseti_grpc.daq_data") / "simulated_data_dir"
            )
        self.root = pathlib.Path(root)

    def for_module(self, module_id: int) -> ModuleCorpus: ...
    def make_synthetic(self, *, n_frames: int, run_name: str,
                       module_ids: list[int], dest: pathlib.Path) -> None: ...
```

The corpus is shipped as package data by `panoseti_grpc` (`grpc/pyproject.toml:86` includes `simulated_data_dir/**/*`). `qa.toml` exposes `[corpus] path = "..."` for dev override. Synthetic generator builds on `panoseti_grpc.panoseti_util.pff.write_image_1D/2D` (same module `simulate.py:53` uses).

`DaqNodeContainer(sim)` feeds `PFFCorpus.for_module(...)` paths into `UdsStrategy.common_config.source_data` via `SimulateDaqConfig`.

### 9. Tier mapping

| Tier | Containers | gRPC | Workspace | Fleet? |
|---|---|---|---|---|
| **1** | none | none | `pseti_workspace` | no — pure Pydantic + `validate_all()` |
| **2** | none | none (boundary tests use real client wired to in-process server fixtures from grpc package) | `pseti_workspace` | no |
| **3** | `Headnode` + N×`DaqNode(sim)` (or `CollapsedModule` in tier3-lite) | real `panoseti-server` everywhere; `UdsStrategy` replaces hashpipe | session | yes |
| **4** | Tier 3 + `Telemetry` + `Gateway` | real | session | yes + `chaos` accessor active |
| **5** | `Headnode` + N×`DaqNode(real)` + `MockQuabo`s + tcpreplay | real, with real `hashpipe.so` | session, compose-managed | yes (compose-driven) |

Per the user directive: **gRPC is never mocked**.

### 10. Migration & rollout (phased sunset)

- **Phase 0 (skeleton, ~1 day)**: create `software_only_v2/` tree empty, add `pseti test sw v2 {unit,logic,fleet,chaos,integration}` CLI route in `control/src/ci/test_cli.py`, add `pytest.ini` markers, add `qa.sw.v2.toml`. v1 still default. CI matrix: `v2-tier1` empty job validates plumbing.
- **Phase 1 (DSL + Tier 1)**: implement `infra/{spec,synth,materialize,workspace,corpus}.py`, `fixtures/workspace.py`, extend `control/topology/fleet.py` with seeded RNG and missing generators. Port Tier 1 unit tests. Parity gate: every Tier 1 test passes under both v1 and v2.
- **Phase 2 (Tier 2)**: port `tier2_logic/test_config_logic.py`, `test_config_validation.py`, `test_ledger.py`. Confirm `pseti_workspace` carries them.
- **Phase 3 (containers)**: implement `containers/{base,daqnode_sim,headnode,mock_quabo,gateway,telemetry}.py` and `orchestrator/fleet.py`. Smoke test: 1 headnode + 2 daqnodes boots and `wait_healthy()` passes.
- **Phase 4 (Tier 3)**: port `tier3_fleet/test_two_node_direct.py`, `test_data_collection.py`, `test_transfer_*`. PFF corpus integrated. UdsStrategy wired.
- **Phase 5 (Tier 4)**: expose `Chaos`; port `tier4_chaos/test_sc_*`, `test_lifecycle_chaos.py`. `fixtures/chaos/*` modules untouched.
- **Phase 6 (Tier 5)**: port `tier5_integration/test_integration_*` to drive the existing `docker-compose.integration.yml`.
- **Phase 7 (sunset)**: 7-day soak in CI with both v1 and v2 green. Then delete `control/src/ci/software_only/`, delete redundant fixture modules: `factories.py`, `rsync_fixtures.py`, `data_fixtures.py`, `chaos_fixtures.py`, `workspace_fixtures.py`, `transfer_fixtures.py`, `state_probe.py`, `client_fixtures.py`, `topology_fixtures.py`, `mocks.py`, the old `fleet.py`. Move `ci/fixtures/chaos/` → `ci/software_only_v2/fixtures/chaos/`. `ci/hardware_software/` is untouched throughout.

### 11. Critical files to create / modify

**Phase 1 (must exist before any test ports)**:
- `control/src/ci/software_only_v2/infra/spec.py` — FleetSpec DSL.
- `control/src/ci/software_only_v2/infra/synth.py` — calls `control.topology.fleet.generate_fleet_configs` + `GlobalConfigValidator`.
- `control/src/ci/software_only_v2/infra/materialize.py` — Pydantic → 7 JSON files in `PSETI_CONFIG`.
- `control/src/ci/software_only_v2/infra/workspace.py` — Workspace + StateProbe-on-PanoPaths.
- `control/src/ci/software_only_v2/infra/corpus.py` — PFFCorpus via `importlib.resources`.
- `control/src/ci/software_only_v2/fixtures/workspace.py` — `pseti_workspace`.
- `control/src/ci/software_only_v2/conftest.py` — autouse env isolation, plugin loader.
- `control/src/ci/software_only_v2/qa.toml` — corpus path override + defaults.
- `control/src/control/topology/fleet.py` — extend with seeded RNG + `generate_data_config` / `generate_firmware_config` / `generate_daemons_config`.

**Phase 3 (containers)**:
- `control/src/ci/software_only_v2/containers/{base,daqnode_sim,headnode,mock_quabo,gateway,telemetry}.py`.
- `control/src/ci/software_only_v2/orchestrator/{fleet,network,lifecycle}.py`.

**Phase 5 (chaos)**:
- `control/src/ci/software_only_v2/fixtures/chaos.py` — Chaos accessor (no changes to `fixtures/chaos/*`).

**Phase 6 (Tier 5)**:
- Wire `tier5_integration/conftest.py` to drive existing `control/src/ci/docker-compose.integration.yml`. No new emitter.

### 12. Verification

**Parity harness** (`infra/parity.py`): for each ported test, register a `@parity_test` entry that runs the same scenario under both v1 and v2 fixtures and asserts identical observable behavior:

```python
@parity_test(
    v1_fixtures=("mock_workspace", "session_fleet"),
    v2_fixtures=("pseti_workspace", "session_fleet_v2"),
    scenario="two_node_start_stop",
)
def assertions(probe: StateProbe, run_name: str):
    assert probe.ledger_status() == "ARCHIVED"
    assert probe.any_pff_files(run_name, head=True)
```

Sunset gate: v1 cannot be deleted until every test in `software_only/tier{1..5}_*/` has a passing v2 parity entry, plus 7-day soak with zero v2-only flakes.

**End-to-end smoke test** (`tier3_fleet/test_smoke_v2.py`): boot a 2-daqnode + 4-module fleet via `FleetSpec`, run a 30-second observation against `PFFCorpus`, then assert:
1. `workspace.state_probe.ledger_status()` reaches `ARCHIVED` within 90 s.
2. Manifest files exist in `workspace.paths.transfer_manifests_dir()` and `verify_manifest()` returns OK.
3. `workspace.paths.runs_dir() / "current"` is gone; ledger run name matches `state_probe.current_run_name()`.
4. Each daqnode's `/data/{run}/module_{id}/data_seq*.pff` files were rsync'd to `HEAD_DATA_DIR` with matching digests.
5. `GlobalConfigValidator.validate_all()` returns OK over the workspace's 7 configs (proves no validator was bypassed mid-test).

**Realism gate**: every workspace writes all 7 configs and runs `validate_all()` at fixture setup. A test wanting less coverage opts out explicitly via `FleetSpec.minimal_unit()` — there is no path to a workspace with raw dicts.

**CI integration**:
- `pseti test sw v2 {unit,logic,fleet,chaos,integration}` CLI routes (one per tier).
- GitHub Actions matrix runs both `v1-tier{N}` and `v2-tier{N}` until v1 is deleted; "release blocker" job is `v2-tier{1..5}` only.
- Beelink runner runs Tier 5 nightly.
