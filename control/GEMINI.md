# GEMINI.md — PANOSETI Control Mandates

This file serves as a foundational mandate for Gemini CLI and other AI agents working within the `control/` directory. It defines the architectural invariants and transactional standards of the PANOSETI control plane.

## 🚀 Architectural Invariants

### 1. Transactional Integrity (Context Managers)
Every observing run lifecycle event (Start/Stop) MUST be managed by a context manager defined in `control/utils/run_state.py`.
- **`StartTransaction`**: Implements a strict, ordered rollback ladder. Any exception within the `with` block triggers a hardware-wide reset and state archival.
- **`StopTransaction`**: Implements a resilient teardown sequence. Steps are executed best-effort; failures in one step (e.g., rsync timeout) MUST NOT block subsequent cleanup steps.
- **Mandate**: NEVER implement procedural rollback logic. Use the context managers.

### 2. Atomic Advisory Locking
- **Standard**: Mutual exclusion is enforced via low-level `os.O_EXCL` file creation on `tmp/panoseti_control.lock`.
- **Self-Healing**: Lock acquisition MUST check for stale PIDs. If the PID file exists but the process is dead, the lock is cleared automatically.
- **Mandate**: NEVER use standard `flock` or `open(..., "w")` for locking as they are unreliable on Docker volumes.

### 3. Non-Blocking Telemetry
- **Standard**: All scripts MUST use the asynchronous `panoseti_grpc.telemetry` client via `panoseti_grpc.telemetry.logger.get_logger`.
- **Initialization**: Loggers MUST be initialized at the module level using `PanoPaths.logs_dir()` for directory resolution.
- **Logging**: Logs are shipped via gRPC to Loki. `builtins.print` should only be used for strictly interactive CLI output; all system events MUST use `logger.info`.
- **Mandate**: NEVER use blocking file I/O or standard `logging.getLogger` without the gRPC handler. Use the unified factory.

---

## 🛠️ Development Mandates

### Pydantic Authority
- The source of truth for configuration is the set of instantiated Pydantic models from `utils/pydantic_config_models.py`.
- **Mandate**: Pass models across function boundaries. Validate dictionaries into models at the first possible entry point.

### Distributed Rollback Contract
- **Receipts**: Node receipts MUST be written to `tmp/run_state.toml` **BEFORE** issuing a `StartDaq` gRPC call (WAL pattern).
- **Concurrency**: Use `asyncio.TaskGroup` for fail-fast parallel RPCs. If one node fails, the group cancels all others, triggering the `StartTransaction` rollback.

---

## 🧪 Testing & Validation

### Chaos-Forced Green
- All transaction-related changes MUST be verified via the chaos suite: `pseti test sw chaos`.
- **Mandate**: A change is considered broken if it passes on localhost but fails in the 4-node Docker fleet simulation.

### CI Environment Detection
- Scripts MUST detect CI environments via `daq_config.head_node_container: true`.
- **Validation**: If this flag is set, pre-flight checks SHOULD be lenient regarding missing local binary files to allow logic-only tests to pass.

---

## 📁 Critical Documentation
- **Transactions**: Read [TRANSACTIONS.md](TRANSACTIONS.md) for rollback ladder sequence.
- **Debugging**: Read [DEBUGGING.md](DEBUGGING.md) for lock and Loki pipeline troubleshooting.
- **CI Architecture**: Read [ci/README.md](ci/README.md) for network and isolation details.
