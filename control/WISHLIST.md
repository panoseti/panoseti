# PANOSETI Testing & Infrastructure Wishlist

This document identifies friction points and architectural gaps discovered during the Phase 1 State Management refactor. It serves as a roadmap for the upcoming "Major Test Refactor."

## 🚀 Top 5 Great Features
1. **Transactional Rollback Ladder**: The context manager architecture (`StartTransaction`) makes hardware state recovery predictable and atomic.
2. **Pydantic Authority**: Strict schema validation at the gRPC and Config boundaries catches 90% of integration errors before they hit the hardware.
3. **Atomic Advisory Locking**: The `os.O_EXCL` implementation is robust against container restarts and volume-mount edge cases.
4. **Chaos Suite**: The ability to simulate network drops and process crashes in a 4-node Docker fleet is a high-signal TDD tool.
5. **High-Velocity Linting**: `uv` + `ruff` + `mypy` provide near-instant feedback, keeping the codebase clean of low-level bugs.

## 🛠️ Top 5 Frustrations & Friction Points
1. **State Pollution (Collisions)**: Tests running in parallel via `xdist` frequently collide on `/app/state/locks/` or `/data/head`. Isolation currently requires manual `monkeypatch` calls in every test.
2. **Manual Path Construction**: We still have `pathlib.Path("configs")` and `tmp/` string-slurping in the tests. The codebase needs a "Path Totality" mandate where *no* script constructs a path outside of `PanoPaths`.
3. **Async Mocking Footguns**: Patching the `asyncio` module is dangerous. We need standardized `AsyncMock` utilities for common patterns like `run_daemon` loops and `TaskGroup` errors.
4. **Service vs. Volume Disconnect**: The `int-tester` container and `daqnode` share `/data`, but the gRPC server performs internal Pydantic validation on those paths. Tests must "reach across" to pre-create directories for Stage 1 (Manifest) to pass.
5. **Traceback Obscurity**: When a `TaskGroup` fails in the background (e.g., in `_process_job`), the error can be swallowed or lead to confusing state transitions (like reaching `ARCHIVED` when a node cleanup actually failed).

## 📋 Strategic Recommendations for Test Refactor

### A. Grouping by Dependency
Refactor the current `unit/` and `integration/` folders into three clearer tiers:
1. **`ci/unit/`**: Zero external dependencies (no Docker, no gRPC, no filesystem I/O). Pure logic/math/parsing.
2. **`ci/distributed/`**: Tests the *Control Plane logic* using mocked gRPC/Hardware but real Filesystem state. Requires `PSETI_STATE` isolation.
3. **`ci/fleet/`**: Full Docker stack tests. Verifies the actual gRPC server/client handshake and high-throughput data paths.

### B. Automated Isolation Fixture
Create a session-scoped `panoseti_isolation` fixture that:
- Automatically sets `PSETI_STATE` and `PSETI_CONTROL` to a unique `tmp_path` per test.
- Pre-populates a minimal `configs/` directory in that temp path so `config_file.get_*` works instantly.
- Ensures `PanoPaths.ensure_state_dirs()` is called before the test body.

### C. Daemon Lifecycle Testing
Standardize how we test infinite loops like `run_daemon`. Every daemon should accept an `asyncio.Event` for shutdown, allowing chaos tests to drive the loop for a specific duration or until a condition is met, then exit cleanly.

### D. Docker Volume Mapping
The editable install of `panoseti_grpc` is fragile in CI. We should move the `grpc/` mount to match the build-time path or use a `PYTHONPATH` that includes `/grpc/src` to ensure code changes on the host are instantly reflected on the `daqnode` server.
