# PSETI Control CLI (`pseti`)

The `pseti` command is the unified entry point for the PSETI observatory control plane. It uses a **lazy-loading proxy architecture** to provide high-performance, snappy access to commands while keeping the code DRY (Don't Repeat Yourself).

## Global Options

- `-h`, `--help`: Show the help message and exit.
- `-t`, `--tree`: Display the command tree for the current hierarchy and exit.
- `--no-env`: Disable automatic loading of `.env` files.
- `--env-template`: Copy the packaged `.env.example` to `./.env_pseti_<timestamp>` and exit.

---

## Primary Commands

### `pseti start`
Start a new recording run.
- `--no-hv`: Take data without high voltage.
- `--no-redis`: OK if redis daemons not running.
- `--no-data`: Set up to record, but don't start data flow or record.
- `--nsecs N`: Record for N seconds, then stop.
- `--stop-session`: Stop session at end of run.

### `pseti stat`
Show observatory health, acquisition status, and transactional ledger.
- `pseti stat`: (Default) Summary of head node and remote DAQ node status.
- `pseti stat ledger`: Inspect the run state ledger (read-only). Lazy-loaded from `control.tools.ledger_cli`.
- `pseti stat sweep`: Full network reachability sweep (Quabo ping + gRPC).
- Options: `--watch` (interactive), `--interval`, `--no-remote`.

### `pseti stop`
Stop and finish the current recording run.
- `--no-cleanup`: Keep .pff files on DAQ nodes after transfer.
- `--no-collect`: Skip rsync to head node.

### `pseti cfg`
Configure observatory hardware and daemons. Subcommands:
`ping`, `reboot`, `reboot-single`, `loads`, `init-daq-nodes`, `hk-dest`,
`redis-daemons`, `stop-redis-daemons`, `permanent-daemons`, `stop-permanent-daemons`,
`show-permanent-daemons`, `hv-on`, `hv-off`, `maroc-config`, `mask-config`,
`calibrate-ph`, `show-ph-baselines`, `shutter-open`, `shutter-close`, `disk-space`,
`start-interleave`, `stop-interleave`, `dry-run-interleave`.

### `pseti val`
Configuration and topology validation tools. Subcommands: `all`, `network`, `graph`, `debug`.

### `pseti power`
Control Quabo power via Web Power Switches (WPS). By default, queries the status of all switches.
- `on`: Turn all Quabo power switches ON.
- `off`: Turn all Quabo power switches OFF.

### `pseti uids`
Fetch quabo IP addrs based on the current `obs_config.json` to get UIDs.

### `pseti xfr`
Manage the background file transfer queue. Supports `stat`, `queue`, `retry`, `clean`, `tail`, `verify`, and `start`/`stop` for the daemon.
- `clean <run_name>`: Remove a job from `pending/` so the daemon skips it — no data is copied from the DAQ node for that run.

### `pseti session-start`
Initialize hardware, power, and calibration for an observing session.
- `--no-hv`: Turn off HV when running `start.py`.

### `pseti session-stop`
Gracefully terminate a session. Powers off all modules and stops background Redis daemons.

---

## System Commands

### `pseti show`
Inspect and visualize PSETI system state. Subcommands:
- `sci`: Live-updating text visualization of the science data stream (requires running gRPC).
- `pff`: Inspect PFF data files on disk.

### `pseti paths`
Display the current resolved paths for all key directories and environment variable overrides. This is a **top-level** command (`pseti paths`), not a subcommand of `pseti show`.

### `pseti test`
Quality Assurance and Testing Suite. All Docker-backed suites accept shared flags at the `pseti test` level (before the subcommand):
- `--dev`: Add `.dev.yml` overlay — hot-mounts source into containers (fast iteration without rebuild).
- `--no-build`: Skip image build; use existing cached image.
- `--tool docker|podman`: Container runtime (default: `docker`).
- `--debug` / `--no-teardown`: Skip container teardown on completion (preserves state for debugging).

#### `pseti test sw2` — v2 test suite (current)
Primary software QA suite. Run `pseti test sw2 -h` for all options.
| Tier | Command | Docker | Description |
|------|---------|--------|-------------|
| 1 | `pseti test sw2 unit` | No | Pure logic, Pydantic, driver parsing |
| 2 | `pseti test sw2 logic` | No | State-machine logic with isolated workspace |
| 3 | `pseti test sw2 fleet` | Yes (testcontainers) | Multi-node E2E |
| 4 | `pseti test sw2 chaos` | Yes (testcontainers) | Fault injection |
| 5 | `pseti test sw2 integration` | Yes (static compose) | Real Hashpipe + tcpreplay |
| — | `pseti test sw2 all` | Yes | lint + all five tiers |
| — | `pseti test sw2 build` | Yes | Build CI images only |
| — | `pseti test sw2 cleanup` | Yes | Tear down compose stacks |

`pseti test sw v2 <tier>` is a valid legacy alias (e.g. `pseti test sw v2 unit`).

#### `pseti test sw` — v1 test suite (sunset in progress)
| Subcommand | Notes |
|------------|-------|
| `logic` | v1 logic tests |
| `fleet` | v1 fleet tests |
| `integration` | v1 integration tests |
| `all` | All v1 tiers |
| `build` / `cleanup` | Build / teardown |

#### Other test suites
| Command | Description |
|---------|-------------|
| `pseti test lint [ruff\|mypy\|all]` | Static analysis: Ruff and/or MyPy. |
| `pseti test grpc [all\|lint\|daq-control\|daq-data\|telemetry]` | gRPC service tests (Docker). |
| `pseti test pff` | PFF file format tests. |
| `pseti test prune` | Prune stale Docker resources. |
| `pseti test hw check-env` | Verify hardware connectivity (real Quabos + DAQ node required). |
| `pseti test hw run [-k SCENARIO]` | Full HW-in-the-loop suite or single scenario. |

### `pseti grpc`
PSETI unified gRPC CLI. Connects to the unified server and issues RPCs.

### `pseti admin`
Admin/deployment tools for remote DAQ nodes and the head node — manages the containerized
(or bare-metal) gRPC server + Hashpipe stack and Grafana Alloy log shipping, from one place.

- **`pseti admin deploy <nodes> [--mode docker|bare-metal]`**: Deploy the DAQ node stack
  and/or the head node stack. `<nodes>` is a comma-separated list of IPs/hostnames,
  `headnode`, or `all` (every DAQ node in `daq_config.json`'s `daq_nodes` list, plus the
  head node). **Runs every node's job concurrently** (`asyncio.TaskGroup`), not one at a
  time — output is multiplexed with a `[host]` prefix per line (same convention as
  docker compose/ansible/pm2) so N nodes' interleaved build/deploy output stays readable,
  and a pass/fail summary table prints at the end. One node's failure (bad SSH host,
  network blip) doesn't cancel the others still in flight; the command exits non-zero if
  any node failed.
  - `--mode docker` (default): builds and starts the gRPC server **and** Grafana Alloy
    containers on the node via `docker --context <ctx> compose -f grpc/deploy/…yml up -d
    --build`. The docker context comes from the node's `docker_context` field in
    `daq_config.json` (falls back to `pseti-daq-<ip-with-dashes>` if unset) and must
    already exist — create it once per node with:
    ```bash
    docker context create <ctx> --docker "host=ssh://<user>@<node-ip>"
    ```
    The DAQ node image (`grpc/deploy/Dockerfile.daqnode`) builds `panoseti-grpc` from the
    **local checked-out source** (the `grpc/` submodule), not PyPI — the image always
    matches exactly the commit this checkout has pinned, with no publish step in between.
  - `--mode bare-metal`: SSHes into the node, activates the `grpc-py314` conda env,
    installs `panoseti-grpc` via `pip install git+https://github.com/panoseti/panoseti_grpc.git@<sha>`
    pinned to the exact commit this checkout's `grpc/` submodule has checked out (not
    PyPI, same "always matches what's actually committed" guarantee as `--mode docker`),
    writes/updates `/etc/panoseti/grpc.env`, and restarts the `panoseti_grpc`/`panoseti_alloy`
    systemd services (installed by `grpc/scripts/setup_panoseti_grpc.sh`). Assumes that
    conda env and a `panoseti` sudo password already exist on the target node.
- **`pseti admin build <nodes> [--mode docker]`**: Build images without starting
  containers. Same concurrent multi-node execution and summary as `deploy`.
- **`pseti admin status <nodes> [--mode docker|bare-metal]`**: Report whether the gRPC
  server and Alloy are running on each node (`docker compose ps`, or
  `systemctl is-active panoseti_grpc panoseti_alloy` in bare-metal mode).
- **`pseti admin down <nodes> [--mode docker|bare-metal]`**: Tear down the stack on each node.
- **`pseti admin attach <node> [service]`**: Open an interactive shell in a running
  service container (default `headnode-server`/`daqnode-server`). Execs in as
  `HOST_UID:HOST_GID`, not root — the image's baked-in default user for a fresh `exec`
  session is root (the main process itself drops to a non-root `panoseti` user at startup
  via `gosu`, but a new `exec` session doesn't inherit that), so without this, anything
  written through a bind mount (e.g. editing configs with `pseti cfg edit`) would land
  root-owned on the host.
- **`pseti admin logs <node> [service] [--follow/-f]`**: Tail a service's container logs.

---

## Developer Guide: Adding & Editing Commands

The `pseti` CLI implements a **Lazy Proxy Pattern** via the `BaseLazyGroup` class in `panoseti_grpc.util.cli`. This means you **never** need to duplicate command signatures or help text in `pseti.py`.

### 1. Modifying Existing Commands
To change options or behavior for a command like `start`, edit the corresponding implementation file directly (e.g., `control/src/control/start.py`). `pseti` will automatically reflect these changes in its `--help` output.

### 2. Adding a New Command
1.  **Implement the command** in its own module using `typer.Typer`.
    ```python
    # src/control/new_tool.py
    import typer
    app = typer.Typer()
    @app.command()
    def main(name: str):
        print(f"Hello {name}")
    ```
2.  **Register it** in `control/src/control/pseti.py` inside the `lazy_mapping` dictionary.
3. **Order it**: the `command_order` argument allows you to specify an explicit command ordering.

### 3. The "Unwrap" Pattern
If a module's Typer app contains only one command (usually named `main` or `@app.command()`), the lazy loader automatically "unwraps" it. This allows `pseti start --option` instead of forcing `pseti start main --option`.

### 4. Discovery
Use `pseti -t` / `pseti --tree` for a full top-level command tree, or `pseti <cmd> -h` for per-command help. There is no `pseti show commands` subcommand.
