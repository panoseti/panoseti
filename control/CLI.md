# PANOSETI Control CLI (`pseti`)

The `pseti` command is the unified entry point for the PANOSETI observatory control plane. It provides high-performance, snappy access to commands for starting/stopping runs, checking status, configuring hardware, and running tests.

## Global Options

- `-h`, `--help`: Show the help message and exit.

---

## Top-Level Commands

### `pseti start`
Start a new recording run.

**Options:**
- `--no-hv`: Take data without high voltage.
- `--no-redis`: Skip checking if redis daemons are running.
- `--no-data`: Set up the run but do not start data flow or recording.
- `--nsecs INTEGER`: Record for N seconds, then stop the run automatically.
- `--stop-session`: Stop the session at the end of the run (used with `--nsecs`).
- `--verbose`: Print detailed command output.
- `--force-reset`: Force reset the state ledger if it appears stale.
- `-y`, `--yes`: Confirm the action without an interactive prompt.

---

### `pseti stop`
Stop and finish the current recording run.

**Options:**
- `--no-cleanup`: Do not delete data files from the DAQ nodes.
- `--no-collect`: Do not collect data files to the head node.
- `--run TEXT`: Stop/Cleanup a specific run name (defaults to the current run in the ledger).
- `--force-cleanup`: Force cleanup on DAQ nodes even if hashpipe liveness is uncertain.
- `--verbose`: Print details.
- `-y`, `--yes`: Confirm the action without an interactive prompt.

---

### `pseti status`
Show control plane status. Checks the transactional ledger, local markers, and probes remote DAQ nodes via gRPC/SSH to report on Hashpipe liveness and disk usage.

---

### `pseti session-start`
Start a complete observing session (power, UIDs, HV, calibration). Orchestrates module power-on, UID scanning, reboots, and daemon initialization.

**Options:**
- `--no-hv`: Turn off High Voltage (HV) when running the session.
- `--stage TEXT`: Start the session from a specific stage (e.g., `poweron`, `get_uids`, `reboot`, etc.).

---

### `pseti session-stop`
Gracefully terminate a session. Powers off all modules and stops background Redis daemons.

---

## Sub-App Commands

### `pseti validate`
Configuration and topology validation tools.

**Subcommands:**
- `network`: Validate configs and perform network ping sweep.
- `graph`: Validate configs and display topology graph.
- `debug`: Validate configs with verbose debug output.
- `all`: Run all validation checks (Schema, Global, Network, Graph).
- **Default**: Running `pseti validate` without a subcommand performs standard schema and global checks.

---

### `pseti power`
Control Quabo power via Web Power Switches (WPS).

**Subcommands:**
- `on`: Turn all configured Quabo power switches ON.
- `off`: Turn all configured Quabo power switches OFF.
- `status`: Query the power state of all configured switches.

---

### `pseti path`
Manage and visualize PANOSETI directory paths.

**Subcommands:**
- `show`: Display all resolved paths and environment variable overrides.
- `init`: Create standard workspace directories if they do not exist.
- `clean`: Remove transient/log directories (requires confirmation).

---

### `pseti config`
Configure observatory hardware and daemons.

**Subcommands:**
- `show`: Show list of domes/modules/quabos and redis daemon status.
- `ping`: Ping all configured quabos.
- `reboot`: Reboot all configured quabos.
- `reboot-single <IP>`: Reboot a single quabo.
- `loads`: Load silver firmware into quabos.
- `init-daq-nodes`: Copy software and configs to remote DAQ nodes.
- `hk-dest`: Set the destination IP for Housekeeping (HK) packets on quabos.
- `redis-daemons`: Start daemons for Redis population (HK/GPS/WR).
- `stop-redis-daemons`: Stop the Redis population daemons.
- `permanent-daemons`: Start permanent system daemons.
- `stop-permanent-daemons`: Stop permanent system daemons.
- `show-permanent-daemons`: Show the status of permanent daemons.
- `hv-on`: Enable detectors (High Voltage ON).
- `hv-off`: Disable detectors (High Voltage OFF).
- `maroc-config`: Configure MAROC registers based on calibration files.
- `mask-config`: Configure pixel masks.
- `calibrate-ph`: Run Pulse Height (PH) baseline calibration.
- `show-ph-baselines`: Show summary statistics for PH baselines.
- `shutter-open`: Open all module shutters.
- `shutter-close`: Close all module shutters.
- `disk-space`: Check disk space on head and DAQ nodes.

---

### `pseti test`
Quality Assurance and Testing Suite.

**Subcommands:**
- `unit`: Run parallel unit tests.
- `integration`: Run end-to-end integration tests.
- `chaos`: Run TDD-forcing chaos/scenario tests.
- `lint`: Run Ruff and MyPy static analysis.
- `all`: Run the full testing suite.
- `build`: Rebuild the testing Docker images.
