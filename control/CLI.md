# PSETI Control CLI (`pseti`)

The `pseti` command is the unified entry point for the PSETI observatory control plane. It uses a **lazy-loading proxy architecture** to provide high-performance, snappy access to commands while keeping the code DRY (Don't Repeat Yourself).

## Global Options

- `-h`, `--help`: Show the help message and exit.
- `-t`, `--tree`: Display the command tree for the current hierarchy and exit.

---

## Top-Level Commands

### `pseti start` (Alias for `pseti obs start`)
Start a new recording run.

### `pseti stop` (Alias for `pseti obs stop`)
Stop and finish the current recording run.

### `pseti status` (Alias for `pseti obs status`)
Show control plane status. Checks the transactional ledger, local markers, and probes remote DAQ nodes via gRPC/SSH to report on Hashpipe liveness and disk usage.

### `pseti cfg` (Alias for `pseti obs config`)
Configure observatory hardware and daemons.

---

## Sub-App Commands

### `pseti obs`
Observatory operations (Start/Stop, Power, Config, Validation).

**Subcommands:**
- `power`: Control Quabo power via Web Power Switches (WPS) (`on`, `off`, `status`).
- `get-uids`: Fetch quabo IP addrs based on the current obs_config.json to get UIDs.
- `config`: Configure observatory hardware and daemons (e.g., `ping`, `reboot`, `hv-on`, `maroc-config`).
- `val`: Configuration and topology validation tools (`all`, `network`, `graph`, `debug`).
- `start`: Start a new recording run.
- `status`: Show observatory health and acquisition status.
- `stop`: Stop and finish the current recording run.
- `transfer`: Manage the file transfer daemon. Supports `--watch` for real-time progress.
- `ledger`: Inspect the run state ledger (read-only).
- `led`: Short alias for `ledger`.
- `session-start`: Initialize hardware, power, and calibration for an observing session.
- `session-stop`: Gracefully terminate a session. Powers off all modules and stops background Redis daemons.

---

### `pseti test`
Quality Assurance and Testing Suite.

**Subcommands:**
- `lint`: Static analysis and linting (Ruff, MyPy).
- `sw`: Software QA tests (Docker-based CI simulations).
  - `unit`: Run parallel unit tests.
  - `logic`: Run mocked grpc tests.
  - `fleet`: Run mocked distributed system nodes.
  - `chaos`: Run TDD-forcing chaos/scenario tests.
  - `integration`: Run structural/topology tests.
  - `all`: Run the full software testing suite.
  - `build`: Rebuild the testing Docker images.
  - `cleanup`: Tear down all test containers and volumes.
- `hw`: Hardware-in-the-Loop (HITL) physical lab tests.
  - `build`: Build HITL images.
  - `check-env`: Verify physical lab environment.
  - `deploy`: Deploy stack to physical nodes.
  - `down`: Stop containers but preserve volumes.
  - `clean`: Tear down containers and wipe volumes.
  - `run`: Run HITL test suite.
- `grpc`: gRPC service layer tests (`all`, `lint`, `daq_data`, `daq_control`, `telemetry`, etc.).

---

### `pseti show`
Inspect and visualize PSETI system state.

**Subcommands:**
- `commands`: Display a tree-like view of all available PSETI commands.
- `paths`: Display the current resolved paths for all key directories and environment variable overrides.

---

### `pseti grpc`
PSETI unified gRPC CLI. Connects to the unified server and issues RPCs.

**Subcommands:**
- `status`: Probe all services and print a summary.
- `reflect`: List all services via gRPC reflection.
- `telemetry`: Telemetry service operations.
- `daq-data`: DAQ Data service operations.
- `daq-control`: DAQ Control service operations.
- `server`: Manage and run the unified gRPC server.

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
2.  **Register it** in `control/src/control/pseti.py` (or the relevant lazy group file) inside the `lazy_mapping` dictionary:
    ```python
    lazy_mapping = {
        ...,
        "new-tool": ("control.new_tool", "app", "Description of tool."),
    }
    ```
3. **Order it**: the `command_order` argument allows you to specify an explicit command ordering to ensure consistent and intuitive UX.

### 3. The "Unwrap" Pattern
If a module's Typer app contains only one command (usually named `main` or `@app.command()`), the lazy loader automatically "unwraps" it. This allows `pseti my-tool --option` instead of forcing `pseti my-tool main --option`.
