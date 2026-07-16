# Plan: Env-driven gRPC endpoint resolution + `pseti admin` orchestration hardening

## Context

`test-pseti` is nearly ready for the 4-telescope production deploy, but two deployment
mechanisms are unreliable:

1. **gRPC port reconfiguration does not propagate.** Changing a port in the `.env` dotfile
   moves *some* clients but neither the unified gRPC server nor the control-plane client
   resolver, forcing operators to hand-edit TOML profiles and "inexplicably" breaking client
   code. Root cause: **there is no single source of truth** — four independent, inconsistent
   port-resolution mechanisms exist (below).
2. **`pseti admin` fails to propagate the environment** to compose invocations consistently
   (only `deploy`/`build` pass `env=`; `down`/`status`/`attach`/`logs` don't), and never uses
   `--env-file`, so co-located deployments (Lick single-machine) hit port / container-name /
   data-dir contention.

Goal: **one canonical, env-driven source of truth for every gRPC endpoint** (server bind port
AND every client connect port) that flows cleanly `.env` → `pseti` process → docker compose →
containers (and → systemd bare-metal), with **per-node overrides via config** for heterogeneous
fleets, plus rigorous tests as the success metric. Scope this round: **fix + solidify the current
docker and bare-metal modes** (no per-component mixing yet). Deliver on `test-pseti` (control) +
a matching `grpc/` submodule feature branch, with PRs. Never touch `main`/`dev`/`develop`.

## Root-cause diagnosis (confirmed by exploration)

The four disconnected port mechanisms:

1. **Unified server bind port** = static TOML `[server].port` only. `grpc/.../server.py`
   (`PanosetiServerConfig.port`, line 143; bound at `server.add_insecure_port(f"[::]:{cfg.port}")`)
   and `grpc/.../unified_main.py` read **no env var**. Profiles: `server.toml`/`server_daq_node.toml`/
   `server_gateway.toml` = 50051; **`server_headnode.toml` = 50052** (outlier). The container
   entrypoint is `pseti-grpc server --config …`, so the container's `GRPC_PORT` env is **dead** for
   the unified server — it binds the TOML value.
2. **Legacy `GRPC_PORT` override** exists only in `control/.../daemons/capture_telemetry_service.py:56`
   and the standalone (non-unified) servers' `__main__`. **Nothing in `session_start.py`/launchers
   actually sets `GRPC_PORT`** — only CI does.
3. **Control-plane client resolver** `control/.../utils/util.py:114 daq_grpc_endpoint()`
   **hardcodes 50051** (branches 1 & 3); only honors `port_forwarding.grpc_port`. Ignores the env
   vars. Used by start/stop/status/transfer-daemon/health.
4. **daq_data + grpc CLI clients** read env (`DAQ_DATA_GATEWAY_PORT`, `HEADNODE_GRPC_PORT`, default
   50051): `start_preflight.py:293`, `tools/show_cli.py:287`, `grpc/.../cli.py:56`. `health.py:151,270`
   hardcodes `localhost:50051`.

Two corrections found during design (verified):
- `control/src/ci/software_only_v2/` is **empty** on this branch; real tier1 tests live in
  `control/src/ci/software_only/tier1_unit/` (has `test_admin_cli.py`, `test_env_loader.py`,
  `test_grpc_cli.py`, `daq_config_fixtures.py`). New tests go **there**.
- `PortForwarding.grpc_port` defaults to **50051**, not `None` (`pydantic_config_models.py:276`,
  comment self-contradicts), so `daq_grpc_endpoint()`'s `is not None` "explicit override" branch can
  never distinguish "operator set it" from "defaulted". Must fix as part of the precedence rule.

## Architecture: canonical endpoint resolution

**Env-var vocabulary (single source of truth, documented in `.env.example`):**
- `HEADNODE_GRPC_PORT` — head/gateway/telemetry server bind **and** every client connecting to the
  head. Default 50051.
- `DAQNODE_GRPC_PORT` — each daq_node server bind **and** every control-plane client connecting to a
  DAQ node. Default 50051.
- `GRPC_PORT` — kept as a **legacy low-priority fallback** only (keeps CI's `GRPC_PORT=50051` a no-op).
- `DAQ_DATA_GATEWAY_PORT` — **collapsed into** `HEADNODE_GRPC_PORT` (the daq_data gateway *is* the
  head unified server); kept as a deprecated read-fallback for one release.

**Per-node overrides (answers the "per-node ports in config?" question):** per-node ports are already
implicitly supported via `network_config.json`'s `port_forwarding.grpc_port` (defined on the node
models — confirmed). The env vars are the **fleet-wide role default**; config provides per-node
overrides for heterogeneous fleets. Precedence (defined in exactly one helper):

```
explicit per-node config  >  role env var  >  legacy env var  >  50051
```
- **Forwarded node** (behind a gateway/router, e.g. UCB): `network_config.json`
  `port_forwarding.grpc_port` — the existing, canonical per-node explicit port.
- **Direct node on a custom port** (no forwarding): add a small optional additive field
  `grpc_port: int | None = None` to the `DaqNode` model in `daq_config.json` for this case.
- Fix `PortForwarding.grpc_port` to `int | None = Field(None, …)` so "operator set it" is detectable
  and matches the documented `None = not forwarded` semantics.

Wiki docs (`wiki_docs/Deploying-the-Modernized-Control-System.md`, `wiki_docs/PSETI-CLI-Reference.md`)
to be updated to document this precedence and the per-node fields.

## File-by-file changes

### control/ (branch `test-pseti`)
- **`utils/util.py`** — add `resolve_grpc_port(role, explicit=None)` (the *one* place precedence
  lives). Rewrite `daq_grpc_endpoint()` (line 114): branches 1 & 3 return
  `(ip, resolve_grpc_port("daqnode", explicit=<node direct grpc_port>))`; branch 2 returns the
  forwarded `port_forwarding.grpc_port` only when it's a real (non-`None`) value.
- **`utils/pydantic_config_models.py`** — `PortForwarding.grpc_port` → `int | None = None` (fix
  comment); add optional `grpc_port: int | None = None` to `DaqNode` (direct-node override).
- **`health.py`** — replace hardcoded `localhost:50051` (lines 151, 270) with `resolve_grpc_port("headnode")`.
  Add two preflight checks: (a) **desync probe** — compute the endpoint the client *will* use via
  `resolve_grpc_port` for each role and `HealthClient`-probe it (green ⇒ server/client in sync); (b)
  **co-location collision check** (pure) — if head and a daq node are `is_local()` to each other
  (reuse `util.is_local`) and `HEADNODE_GRPC_PORT == DAQNODE_GRPC_PORT`, or `DAQ_DATA_DIR` overlaps
  `PSETI_DATA_DIR`, fail loudly with the resolved values.
- **`start_preflight.py:293`, `tools/show_cli.py:287`** — replace `DAQ_DATA_GATEWAY_PORT` reads with
  `resolve_grpc_port("headnode")`.
- **`utils/env_loader.py`** — add `HEADNODE_GRPC_PORT`, `DAQNODE_GRPC_PORT` to `get_env_info()`'s
  `known_runtime` so `pseti env` shows them.
- **`admin/cli.py`** — env-propagation hardening:
  - Add `_write_compose_env_file()` that materializes the pseti-resolved env subset (current
    `_PRINTABLE_ENV_KEYS` **+** `HEADNODE_GRPC_PORT`, `DAQNODE_GRPC_PORT`, `GRPC_PORT`,
    `DAQ_DATA_GATEWAY_HOST`, `REDIS_HOST`, `LOKI_URL`) to a temp file under `PanoPaths.tmp_dir()`;
    pass `--env-file <path>` on **every** compose invocation (deterministic interpolation, honors
    `PSETI_ENV_FILE` transitively).
  - Pass `env=` on the `down`/`status`/`attach`/`logs` DAQ paths that currently omit it
    (cli.py:287, 299, 388, 397 + alloy variants).
  - Remove the `os.environ.update(env)` mutation in `_get_compose_cmd_base` in favor of `--env-file`.
  - Extend `_PRINTABLE_ENV_KEYS` (line 24) with the port vars.
  - Bare-metal/SSH path: prefix remote commands with inline `HEADNODE_GRPC_PORT=… DAQNODE_GRPC_PORT=…`
    (scalable; avoids per-node sshd `AcceptEnv` config).
- **`deploy/docker-compose.headnode.yml`** — change the `headnode-server` `command` to append
  `--port-env HEADNODE_GRPC_PORT`; keep `GRPC_PORT: ${HEADNODE_GRPC_PORT:-50051}`.

### grpc/ (matching feature branch, e.g. `feature/grpc-port-env`)
- **`src/panoseti_grpc/unified_main.py`** — add the env seam. New args `--port` and `--port-env`.
  Factor a pure, unit-testable `resolve_bind_port(args, cfg) -> int` with order: `--port` >
  `os.getenv(args.port_env)` > `os.getenv("GRPC_PORT")` (legacy) > `cfg.port` (TOML). Set `cfg.port`
  from it before `PanosetiServer.run(cfg)`. `server.py` (config class, `add_insecure_port`) untouched.
- **`src/panoseti_grpc/config/server_headnode.toml`** — change `port = 50052` → `50051` (co-location
  distinctness now comes from `.env`, not a baked TOML value). Grep-verify nothing else hardcodes 50052.
- **`deploy/docker-compose.daqnode.yml`** — line 31 `HEADNODE_GRPC_PORT: "50051"` →
  `"${HEADNODE_GRPC_PORT:-50051}"`; add `command`/entrypoint arg `--port-env DAQNODE_GRPC_PORT` (keep
  `GRPC_PORT: ${DAQNODE_GRPC_PORT:-50051}`).
- **`deploy/alloy/docker-compose.alloy.yml`** — remove hardcoded `container_name: panoseti_alloy`
  (line 61) so it derives from the compose project name (`pseti-daqnode-<host>` vs `pseti-headnode`),
  preventing co-located collision. (`admin status --mode bare-metal`'s `systemctl is-active
  panoseti_alloy` is a systemd unit name — unrelated, safe.)

## Testing & verification (success metric)

- **(a) Tier1 unit — new `control/src/ci/software_only/tier1_unit/test_grpc_endpoint.py`**: precedence
  matrix for `resolve_grpc_port` (explicit > role env > legacy env > 50051 via `monkeypatch.setenv`);
  `daq_grpc_endpoint` for local (branch 1 honors `DAQNODE_GRPC_PORT`), explicit forwarded port,
  `grpc_port=None` fallthrough, direct node; co-location collision assertion. Reuse
  `daq_config_fixtures.py`.
- **(b) grpc unit — extend `grpc/tests/unified_server/unit/test_config.py`**: test `resolve_bind_port`
  (`--port-env DAQNODE_GRPC_PORT` + `DAQNODE_GRPC_PORT=50055` ⇒ 50055; `GRPC_PORT` lower-priority;
  unset ⇒ TOML default). Add `test_headnode_profile_port_is_50051` to lock the 50052→50051 change.
- **(c) Tier3 fleet — `control/src/ci/software_only/tier3_fleet/`** (testcontainers; model on existing
  `test_single_node_headnode.py`): bring the unified server up on a **non-default** port purely via
  env (`DAQNODE_GRPC_PORT=50055`) and assert a client resolved through `daq_grpc_endpoint()` connects
  and `HealthClient.check` passes — the end-to-end proof of one source of truth.
- **(d) Live HITL smoke runbook** (UCB `pseti-daq-ucb1`, daqnode `panoseti@192.168.88.152` /
  `192.168.0.228`; **no `pseti test hw`**): set `DAQNODE_GRPC_PORT=50055` in the site `.env` →
  `pseti admin down all` → `deploy all` → `status all` (printed invocation now shows the port vars +
  `--env-file`) → `pseti health` (probes resolve to 50055, green) → `pseti grpc stat` / `pseti stat` →
  full `pseti start`/`pseti stop` lifecycle **without hand-editing any TOML** → revert `.env`, confirm
  clean return to 50051.
- Gate: `pseti test lint` + `pseti test sw unit` + `pseti test grpc lint` green before PRs.

## Backward-compat / risk register

- **CI `GRPC_PORT=50051`**: safe — becomes a fallback equal to default. Verify `.env.ci`, Dockerfiles,
  `docker-compose.integration.yml`, `docker-compose.hw-sw.yml` don't rely on old (ignored) behavior.
- **`PortForwarding.grpc_port` 50051→None**: the one real behavior change — a node with
  `port_forwarding.status=true` and no explicit `grpc_port` previously forwarded on 50051; now falls
  to direct-IP + env. **Audit** all `daq_config.json`/`network_config.json` under `configs/` and
  `control/src/ci/hardware_software/configs` (the live `.env` points `PSETI_CONFIG` here).
- **`server_headnode.toml` 50052→50051**: any head relying on baked 50052 must set
  `HEADNODE_GRPC_PORT=50052`. Release-note it.
- **`DAQ_DATA_GATEWAY_PORT` deprecation**: read-fallback for one release; log when it's the source.
- **Alloy `container_name` removal**: grep confirmed the literal only in the compose file.

## Delivery
- control/ changes on `test-pseti`; grpc/ changes on a new `feature/grpc-port-env` submodule branch.
- Open a PR for each. Update `wiki_docs/Deploying-the-Modernized-Control-System.md` +
  `wiki_docs/PSETI-CLI-Reference.md` for the env-var precedence and per-node port fields.
- Do **not** commit to `main`/`dev`/`develop`.
