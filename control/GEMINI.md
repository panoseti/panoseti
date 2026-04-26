# GEMINI.md — PANOSETI Control Mandates

This file serves as a foundational mandate for Gemini CLI and other AI agents working within the `control/` directory. It defines the architectural invariants and transactional standards of the PANOSETI control plane.

## 🚀 Architectural Invariants

### 1. Transactional Integrity (Context Managers)
Every observing run lifecycle event (Start/Stop) MUST be managed by a context manager defined in `control/utils/run_state.py`.
- **`StartTransaction`**: Implements a strict, ordered rollback ladder. Any exception within the `with` block triggers a hardware-wide reset and state archival.
- **`StopTransaction`**: Implements a "Fast-Path" teardown. It performs minimal hardware stop commands, enqueues a background job in the `TransferQueue`, and transitions the ledger to `RECORDING_ENDED`.
- **Mandate**: NEVER implement procedural rollback or collection logic. Use the context managers and let the `TransferWorker` daemon handle post-run processing.

### 2. Atomic Advisory Locking
- **Standard**: Mutual exclusion is enforced via low-level `os.O_EXCL` file creation on `state/locks/control.lock` (and `state/locks/transfer.lock` for the transfer daemon).
- **Self-Healing**: Lock acquisition MUST check for stale PIDs. If the PID file exists but the process is dead, the lock is cleared automatically.
- **Mandate**: NEVER use standard `flock` or `open(..., "w")` for locking as they are unreliable on Docker volumes.

### 3. Non-Blocking Telemetry & Control
- **Standard**: All scripts MUST use the asynchronous `panoseti_grpc.telemetry` client via `panoseti_grpc.telemetry.logger.get_logger`.
- **Async gRPC**: Always prefer `AsyncDaqControlClient` with `asyncio.TaskGroup` for concurrent multi-node operations (Start/Stop/Status).
- **Mandate**: NEVER use blocking `DaqControlClient` or `asyncio.to_thread` wrappers for gRPC calls in the control plane. Use native async context managers.

---

## 🛠️ Development Mandates

### Pydantic Authority
- The source of truth for configuration is the set of instantiated Pydantic models from `utils/pydantic_config_models.py`.
- **Network Resolution**: Always use `util.get_quabo_ip_port()` to resolve effective IPs and ports. It is the single source of truth for handling Gateway port forwarding.
- **Mandate**: Pass models across function boundaries. Signatures MUST use strictly-typed Pydantic models (e.g., `IPvAnyAddress`, `QuaboIpPorts`). Avoid `dict[str, Any]` fallbacks.

### Distributed Rollback Contract
- **Receipts**: Node receipts MUST be written to `state/runs/ledger.toml` **BEFORE** issuing a `StartDaq` gRPC call (WAL pattern).
- **Fail-Fast**: Use `asyncio.TaskGroup` for parallel RPCs. If one node fails, the group cancels all others, triggering the `StartTransaction` rollback.

---
## 🧪 Testing & Validation

### Tiered Validation
- **Tier 1 (Unit)**: Pure logic, no external dependencies. `ci/tier1_unit/`.
- **Tier 2 (Logic)**: State logic with mocked gRPC. `ci/tier2_logic/`.
- **Tier 3 (Fleet)**: Dynamic multi-node with testcontainers. `ci/tier3_fleet/`.
- **Tier 4 (Chaos)**: Fail-fast and resilience tests. `ci/tier4_chaos/`.
- **Tier 5 (Integration)**: Static stack HW-SW simulation. `ci/tier5_integration/`.

### 🛡️ Resilience Mandates
- **Idempotent Cleanup**: `CleanupData` MUST be idempotent. Requests to clean missing/already-deleted directories must return `ok=True`.
- **Split-Brain Isolation**: Host-side test preparation MUST use isolated `tmp_path` directories. Every dynamic container MUST receive a unique host directory for its `/data` volume.
- **Permission Boundary**: Host-side prepared directories MUST be recursively `chmod 777` to allow container root access.

### Chaos-Forced Green
...
- All transaction-related changes MUST be verified via the chaos suite: `pseti test sw chaos`.
- **Mandate**: A change is considered broken if it passes on localhost but fails in the 4-node Docker fleet simulation.

### CI Environment Detection
- Scripts MUST detect CI environments via `daq_config.head_node_container: true`.
- **Validation**: If this flag is set, pre-flight checks SHOULD be lenient regarding missing local binary files to allow logic-only tests to pass.

---

## 📁 Critical Documentation Routing

| Document | Description |
|---|---|
| [TRANSACTIONS.md](TRANSACTIONS.md) | Rollback ladder sequence, atomic locking, and run state transitions. |
| [DEBUGGING.md](DEBUGGING.md) | Core debugging principles, lock and Loki pipeline troubleshooting, state isolation. |
| [TEST.md](TEST.md) | Test suite architecture, Docker runner usage, and isolation mandates (`PSETI_STATE`). |
| [ci/README.md](ci/README.md) | CI network topology and isolation details. |

