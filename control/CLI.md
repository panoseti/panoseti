# PSETI Control CLI (`pseti`)

The `pseti` command is the unified entry point for the PSETI observatory control plane. It uses a **lazy-loading proxy architecture** to provide high-performance, snappy access to commands while keeping the code DRY (Don't Repeat Yourself).

## Global Options

- `-h`, `--help`: Show the help message and exit.
- `-t`, `--tree`: Display the command tree for the current hierarchy and exit.

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
- `pseti stat ledger`: Inspect the run state ledger (read-only).
- `pseti stat remote`: Query each DAQ node via gRPC.
- `pseti stat sweep`: Full network reachability sweep.

### `pseti stop`
Stop and finish the current recording run.
- `--no-cleanup`: Keep .pff files on DAQ nodes after transfer.
- `--no-collect`: Skip rsync to head node.

### `pseti cfg`
Configure observatory hardware and daemons (e.g., `ping`, `reboot`, `hv-on`, `maroc-config`).

### `pseti val`
Configuration and topology validation tools (`all`, `network`, `graph`, `debug`).

### `pseti power`
Control Quabo power via Web Power Switches (WPS). By default, queries the status of all switches.
- `on`: Turn all Quabo power switches ON.
- `off`: Turn all Quabo power switches OFF.

### `pseti uids`
Fetch quabo IP addrs based on the current `obs_config.json` to get UIDs.

### `pseti xfr`
Manage the background file transfer queue. Supports `stat`, `queue`, `retry`, `tail`, `verify`, and `start`/`stop` for the daemon.

### `pseti session-start`
Initialize hardware, power, and calibration for an observing session.
- `--no-hv`: Turn off HV when running `start.py`.

### `pseti session-stop`
Gracefully terminate a session. Powers off all modules and stops background Redis daemons.

---

## System Commands

### `pseti show`
Inspect and visualize PSETI system state.
- `commands`: Display a tree-like view of all available PSETI commands.
- `paths`: Display the current resolved paths for all key directories and environment variable overrides.

### `pseti test`
Quality Assurance and Testing Suite.
- `lint`: Static analysis and linting (Ruff, MyPy).
- `sw`: Software QA tests (Docker-based CI simulations).
- `hw`: Hardware-in-the-Loop (HITL) physical lab tests.

### `pseti grpc`
PSETI unified gRPC CLI. Connects to the unified server and issues RPCs.

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
