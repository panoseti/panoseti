# PanoSETI gRPC Modernization — Execution Plan

## Context

`plans/modernize-grpc.md` lays out a broad blueprint. This plan executes the concrete subset the user asked for, in **phased, independently-mergeable PRs**:

1. Fix the stale systemd path (`grpc/scripts/start_grpc.sh` launches only legacy `python -m daq_data.server` under a non-existent `grpc-py39` env), launch the **unified `panoseti-server`** instead, add a **Grafana Alloy** systemd unit + config, and make the unified logger default its `.jsonl`/`.log` output to a **per-host** path `/var/log/panoseti/<host>/...` that Alloy globs.
2. Wire **`grpc.health.v1`** into the active services (daq_data, daq_control, telemetry; **exclude ublox_control**), add per-service `SERVING`/`NOT_SERVING` toggling during reconfiguration, and migrate liveness probes off the legacy `daq_data.Ping` (kept, deprecated).
3. Re-architect daq_data into a **headnode aggregator gateway**: a single daq_data server on the headnode connects server→server to each per-node daq_data edge server and re-exposes one `StreamImages`/`Status`. Consumers connect to **one** endpoint (kills M×N connection scaling). **Auto-init** `HpIoManager` on edge-server startup defaulting to the real UDS→hashpipe path (no simulation); manual client `InitHpIo` becomes optional reconfiguration only.
4. Refactor daq_data, daq_control, telemetry clients onto `grpc_utils` (`@grpc_call`, `AsyncChannelManager`, typed `PanosetiRpcError`, `HealthClient`, `build_retry_service_config`) so all three match the clean single-target `daq_control` client shape.
5. Migrate the affected `grpc/` and `control/` tests, add Alloy to the test Docker/compose stacks, and add a daqnode status command that reports Alloy health.

### Resolved design decisions (from user)

- **Topology:** headnode aggregator gateway. Clients talk only to the headnode; the gateway talks to the DAQ nodes.
- **Client API:** clean break — single-target `DaqDataClient(host, port)` like `DaqControlClient`; update all callers (orchestrator, fleet/HW tests).
- **Auto-init:** server auto-inits real UDS path; keep `InitHpIo` + `daq_data.Ping` RPCs, deprecate gradually (`DeprecationWarning`, removed a release later).
- **RabbitMQ?** No. The data plane is high-rate science-image frames with **lossy-overwrite preview semantics** and effectively **zero production consumers** today (`grep` shows only tests/CLI/notebook). A broker (RabbitMQ/Kafka) adds a service to operate, durable persistence we don't need, and per-frame (de)serialization cost, for a problem we don't have. This matches `plans/modernize-grpc.md` §2.1 ("KEEP the current design"; brokers "justified only if we hit high-fanout (>100 subscribers) or durable replay"). The aggregator is **not** a message queue — it is a stateless gRPC fan-in that preserves the existing per-reader cursor + rate-shaping model (`server.py:_get_fresh_images_for_client`, `state.ReaderState`). If durable replay is ever needed, the §2.1 evolution path is **Redis Streams** (Redis already deployed), not RabbitMQ.
- **`control/utils/paths.py` config concern:** today `DaqDataClient(daq_config, network_config)` consumes `daq_config.json`/`network_config.json` (port-forwarding, dynamic dir overrides) on the *client* side. Resolution: **move that consumption to the aggregator server**. The aggregator reads `daq_config`/`network_config` (via its own server config / env, resolvable through `PanoPaths`) to learn which edge nodes to dial and how (port-forwarding/gateway). Consumers only need the single headnode `host:port`. `paths.py` dynamic overrides keep working because they feed the aggregator's server-side config exactly as they feed the orchestrator today — no capability is lost, it just moves one hop.

---

## Phase 1 — Unified logger per-host path + systemd + Alloy

**Goal:** correct deployment story; logs land where Alloy can ship them.

### 1a. Per-host log directory (single chokepoint)
- `grpc/src/panoseti_grpc/telemetry/logger.py::get_logger` (lines 326–329): when `log_dir` is set, append a host segment: `log_dir_path / socket.gethostname()` (fall back to `os.getenv("HOSTNAME")`). This is the only edit needed — all three services route through `get_logger`. `JsonlFormatter._hostname` (line 62) already records hostname as a field, so labels are unaffected; this only changes file layout.
- `telemetry/server.py` uses `make_rich_logger` (line 37) and emits **no** `.jsonl`. Switch it to `get_logger("telemetry_server", log_dir=..., grpc_enabled=False)` so telemetry logs are also shipped (its own Log-RPC ingress stays separate).
- Update the inert top-level `[server] log_dir` TOML keys to be consumed, or document them as per-service; set per-service `log_dir = "/var/log/panoseti"` in `config/server.toml`, `server_daq_node.toml` (the host segment is added by `get_logger`, not the TOML).

### 1b. systemd
- Rewrite `grpc/scripts/start_grpc.sh`: drop `conda activate grpc-py39` + `python -m daq_data.server`; instead `exec panoseti-server --config <profile.toml>` (profile chosen by env, default `server_daq_node.toml` on DAQ nodes, `server_headnode.toml` on headnode). Ensure `/var/log/panoseti` is writable (the `FileLogConfig.validate_directory` fallback at logger.py:143–149 already degrades gracefully).
- Update `grpc/scripts/setup_panoseti_grpc.sh`: keep the generated unit, but install **two** units — `panoseti_grpc.service` (wraps the new `start_grpc.sh`) and `panoseti_alloy.service` (`ExecStart=/usr/bin/alloy run /etc/alloy/config.alloy`), both `WantedBy=multi-user.target`, `Restart=on-failure`. Add an `--alloy`/`--no-alloy` flag.

### 1c. Alloy config
- Promote the repo-root `alloy/config.alloy` into the grpc package as the canonical artifact (e.g. `grpc/deploy/alloy/config.alloy`). Change the glob `local.file_match "panoseti"` from `/var/log/panoseti/*.jsonl` to `/var/log/panoseti/*/*.jsonl` to match the new per-host subdirectory. Keep the `loki.process` json/labels stages as-is.
- Add `grpc/deploy/alloy/docker-compose.alloy.yml` (referenced by `control/CLAUDE.md` but missing) for container deployment on the headnode.

---

## Phase 2 — Health checks (exclude ublox_control)

`register_health` already exists (`grpc_utils/health.py`) and is already called by `PanosetiServer.run()` (server.py:350–360) for every reflected active service. Remaining work:

- **Per-service liveness transitions:** expose `register_health` so the servicers can flip a service to `NOT_SERVING` during disruptive reconfiguration and back to `SERVING` after. Wire:
  - daq_data: `NOT_SERVING` while the writer lock is held / `HpIoTaskManager.start()` is restarting the task (`managers.py:50-92`, `server.py:InitHpIo`).
  - daq_control: `NOT_SERVING` mid-`StartDaq`/`StopDaq` launch window.
  - Pass a small health-toggle handle (set/clear by service name) from `register_health` into each servicer.
- **Ensure standalone entrypoints register health too** (today only the unified server does): add a `register_health` call in `daq_data/server.py::serve` and `telemetry/server.py::serve` guarded by the optional `grpcio-health-checking` import (mirror server.py:356–360).
- **Client migration:** add `HealthClient` use to `_cli/root.py` `status` probe and `_cli/daq_data.py ping` (keep `Ping` working, emit `DeprecationWarning`). Add `grpcio-health-checking` to `pyproject.toml` if not already a hard dep.
- ublox_control: untouched (disabled by default; out of scope).

---

## Phase 3 — grpc_utils refactor of the three clients

`daq_control/client.py` is the reference shape: `self.target`, `AsyncChannelManager`-style channel, every method `@grpc_call`, Pydantic params in `client_models.py`. Bring the others in line.

- **daq_data client** (`daq_data/client.py`, ~1036 LOC, the biggest win): rewrite `DaqDataClient`/`AioDaqDataClient` to single-target (`host, port`) using `AsyncChannelManager` + `keepalive_options` + `build_retry_service_config`. Replace the hand-rolled `self.daq_nodes`/`valid_daq_hosts` multi-host fan-out, `_attach_port_forwarding`, `validate_daq_hosts`, and the `asyncio.gather(..., return_exceptions=True)` channel-cleanup swallower (lines 677, 913). Decorate methods with `@grpc_call`; drop the bespoke `except grpc.RpcError`/`ConnectionError` blocks and the `__aexit__` exception-suppression ladder. Replace `ping()` with `HealthClient`. (Multi-host fan-out moves server-side into the Phase-4 aggregator.)
- **telemetry client** (`telemetry/client.py`): out of deep scope here, but route `grpc.RpcError → PanosetiRpcError` via `@grpc_call` on its RPC methods for consistency; do not yet touch the `threading.Thread`/`queue.Queue` model (gated on the separate Alloy log-path decision).
- **daq_control client:** already on `@grpc_call`; minor — adopt `AsyncChannelManager` + retry service config for channel parity.

---

## Phase 4 — daq_data aggregator gateway + auto-init

### 4a. Auto-init (small, ship first within this phase)
- `DaqDataServerConfig.init_from_default` already exists and `DaqDataServicer.start_initial_task()` already starts an initial `HpIoManager` from `default_hp_io_config_file`. Currently the default points at `hp_io_config_simulate.json`. Change the **edge server** profile so it auto-starts the **real UDS** path (`simulate_daq=False`) by default: set `init_from_default = true` + a new `hp_io_config_default.json` (real UDS, no sim) in `config/server_daq_node.toml`'s daq_data section.
- `InitHpIo` (server.py:199) stays as an **optional reconfiguration** RPC (writer-lock semantics unchanged). Mark client `init_hp_io`/`init_sim` with `DeprecationWarning` for the "must call before streaming" usage; tests/sim pass `simulate_daq=True` explicitly as an override.

### 4b. Aggregator gateway (the large piece)
- New module `grpc/src/panoseti_grpc/daq_data/aggregator.py` (+ wire into `server.py` as a selectable role, e.g. `daq_data.role = "gateway" | "edge"` in server config; default `edge` on DAQ nodes, `gateway` on headnode).
- The gateway implements the same `DaqData` servicer interface (no proto change). On start it reads `daq_config`/`network_config` (server-side, via its config / `PanoPaths`-resolvable paths) to enumerate edge nodes + port-forwarding. It holds **one** upstream `AsyncDaqDataClient` (Phase-3 single-target) per edge node (server→server). `StreamImages` on the gateway fans in from all edges using the **outcome-collection under `TaskGroup`** pattern from `grpc_utils/README.md` (best-effort: a down node must not cancel the merged stream); `Status`/health aggregate per-node state. Reuse the existing per-reader cursor merge logic conceptually but server-side.
- Consumers now construct `DaqDataClient(headnode_host, port)` — single channel, single stream. M×N connections collapse to M (clients→gateway) + N (gateway→edges).

---

## Phase 5 — Test migration

Note (per user): **5 daq_control integration tests are currently failing due to shoddy test implementation** (likely fixed-`time.sleep`/race/hardcoded-PID issues in `tests/daq_control/integration/test_concurrent_requests.py` and `test_process_edge_cases.py`). Not in scope to fix here — **flag them** in the test-migration PR so they aren't confused with regressions from this work.

- **grpc/tests/daq_data:** `conftest.py` and `integration/` build `AioDaqDataClient/DaqDataClient` with the old multi-host signature and call `init_sim`/`Ping`. Update to single-target client; replace `Ping` assertions with `HealthClient.check`; sim tests pass explicit `simulate_daq=True`. Add gateway tests (gateway in front of ≥2 in-process edge servers).
- **control/src/ci/fixtures/client_fixtures.py:** `data_client` builds `DaqDataClient(daq_config.model_dump(), network_config=...)` — change to single-target pointed at the aggregator endpoint resolved from topology. Audit every caller from the earlier grep (`tier3_fleet/*`, `tier5_integration/*`, `hardware_software/core/{daq_status,stream}.py`, `software_only_v2/orchestrator/fleet.py`).
- **Orchestrator/control:** any production `DaqDataClient` construction in `control/src/control` updated to the single headnode endpoint; aggregator gets `daq_config`/`network_config` server-side.
- Keep `control/` sw2 + hardware-software suites green (they currently pass). Coverage/count gates per `plans/modernize-grpc.md` §4.5.
- Consider giving `grpc/tests` access to the `control` testcontainer fleet pattern (`software_only_v2/containers/daqnode_sim.py` runs `panoseti-server --profile daq_node`) for true gateway↔edge E2E instead of only in-process servers.

---

## Phase 6 — Docker/compose + Alloy CI + daqnode status command

- Add an `alloy` service (and `loki` where commented out) to `grpc/tests/telemetry/docker-compose.test.yml` and the integration composes; mount the per-host `/var/log/panoseti` volume and `config.alloy`. Add an Alloy stage/volume to `grpc/Dockerfile.ci` and `control/src/ci/Dockerfile.ci` so the jsonl→Alloy→Loki path is exercised in CI.
- **daqnode status command:** extend `pseti-grpc` (and/or daq_control) with `daqnode status` that reports: unified-server health (`HealthClient` per service), Alloy systemd/process liveness (or `:12345/-/ready` Alloy endpoint), and disk for `/var/log/panoseti`. This doubles as the §3.4 HW env check.

---

## Critical files

New:
- `grpc/src/panoseti_grpc/daq_data/aggregator.py`
- `grpc/deploy/alloy/config.alloy`, `grpc/deploy/alloy/docker-compose.alloy.yml`
- `grpc/src/panoseti_grpc/daq_data/config/hp_io_config_default.json` (real-UDS auto-init)

Modified:
- `grpc/src/panoseti_grpc/telemetry/logger.py` (`get_logger` per-host segment)
- `grpc/src/panoseti_grpc/telemetry/server.py` (use `get_logger`)
- `grpc/scripts/start_grpc.sh`, `grpc/scripts/setup_panoseti_grpc.sh`
- `grpc/src/panoseti_grpc/grpc_utils/health.py` (return a per-service toggle handle)
- `grpc/src/panoseti_grpc/server.py` (role select; pass health toggles to servicers)
- `grpc/src/panoseti_grpc/daq_data/server.py` (auto-init real UDS; health transitions; standalone `register_health`)
- `grpc/src/panoseti_grpc/daq_data/client.py` (single-target rewrite on `grpc_utils`)
- `grpc/src/panoseti_grpc/daq_data/config.py`, `config/server*.toml`
- `grpc/src/panoseti_grpc/daq_control/server.py` (health transitions)
- `grpc/src/panoseti_grpc/_cli/{root,daq_data}.py` (HealthClient; Ping DeprecationWarning)
- `grpc/tests/daq_data/conftest.py` + `integration/*`
- `control/src/ci/fixtures/client_fixtures.py` + fleet/HW callers
- `grpc/tests/telemetry/docker-compose.test.yml`, `grpc/Dockerfile.ci`, `control/src/ci/Dockerfile.ci`
- `grpc/pyproject.toml` (`grpcio-health-checking` hard dep)

Reuse (do not reinvent): `grpc_utils.{grpc_call, AsyncChannelManager, keepalive_options, build_retry_service_config, register_health, HealthClient, from_rpc_error}`; existing `init_from_default`/`start_initial_task`; `client_models` Pydantic params; outcome-collection pattern from `grpc_utils/README.md`.

---

## Verification

Per phase, local + Docker:

```bash
# Phase 1: logger path
python -c "from panoseti_grpc.telemetry.logger import get_logger; get_logger('t', log_dir='/tmp/p'); import socket,os; print(os.listdir(f'/tmp/p/{socket.gethostname()}'))"
cd grpc && python tests/qa.py lint

# Phase 2: health
panoseti-server --profile daq_node & sleep 3
grpc_health_probe -addr=localhost:50051 -service=daqdata.DaqData
grpc_health_probe -addr=localhost:50051 -service=daqcontrol.DaqControl

# Phase 3/4: clients + aggregator + auto-init
cd grpc && python tests/qa.py daq_data
pytest tests/daq_data/integration -v          # single-target + gateway E2E
# auto-init: start edge server, stream WITHOUT calling init_hp_io → frames arrive

# Phase 5: regression
cd control && pseti test sw2 unit && pseti test sw2 fleet
pseti test grpc all
# (note: 5 pre-existing daq_control integration failures are NOT caused by this work)

# Phase 6: Alloy path
docker compose -f grpc/tests/telemetry/docker-compose.test.yml up --build --exit-code-from test_runner
pseti-grpc daqnode status                     # health + Alloy liveness + log disk
logcli query '{service="daq_data.server"}' --since=2m   # jsonl → Alloy → Loki
```

Success: unified server runs under the new systemd unit with Alloy shipping per-host `.jsonl` to Loki; `grpc_health_probe` succeeds for all 3 active services; consumers use a single `DaqDataClient(host, port)` against the headnode gateway; streaming works without a manual `InitHpIo`; `control/` sw2 + HW suites stay green (minus the 5 known-bad daq_control tests); `tokei` shows net LOC reduction in the three clients.

---

## Sequencing (independently mergeable)

1. **Phase 1** logging per-host + systemd + Alloy config — no API change, lowest risk.
2. **Phase 2** health transitions + standalone register_health + CLI HealthClient (Ping kept).
3. **Phase 3** grpc_utils refactor of daq_data client to single-target (behind no behavior change yet — orchestrator still per-node).
4. **Phase 4a** auto-init real UDS; **4b** aggregator gateway + flip consumers to single endpoint.
5. **Phase 5** test migration (grpc + control) — lands with/after Phase 4; flags the 5 pre-existing daq_control failures.
6. **Phase 6** Alloy in CI Docker/compose + `daqnode status` command.

Risk: Phases 1–2 pure addition. Phase 3 high LOC churn, low behavior risk. Phase 4b is the highest-risk change (data-plane topology) — gate on new gateway E2E tests + green `control/` fleet before merge.
