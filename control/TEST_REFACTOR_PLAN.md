# Comprehensive Test Refactor Plan: PANOSETI Control Plane

This document serves as the architectural blueprint for the Phase 2 major test refactoring effort. It outlines the strategy to transition our organically grown test suite into a highly scalable, modular, and maintainable framework designed for high-concurrency execution and extensibility.

---

## 1. High-Level Goals & Philosophy

- **Extreme Isolation**: Eliminate all cross-test state bleed. A test failure must never be caused by a residual lock, file, network bind, or database state from a prior test.
- **Domain-Driven Grouping**: Move away from arbitrary "integration" or "chaos" buckets towards functional domains.
- **DRY Boilerplate**: Consolidate repetitive setups into robust, typed fixtures and factory functions.
- **Semantic Naming**: Tests must describe *behavior*, not just component names.
- **Focus Level**: Software-only tests focus strictly on the **observatory logic layer** (above the board level). They simulate systems logic without needing physical Quabos. Dedicated hardware-software tests handle full-system, board-level integration and the software-hardware interface.

---

## 2. Tiered Dependency Matrix & CI Pipeline Optimization

We will restructure the `ci/` directory into four tiers based on their execution environment and dependency requirements. This structure is designed to optimize CI pipelines (e.g., GitHub Actions): **Tiers 1 & 2 run on every commit**, while the heavier **Tiers 3 & 4 are reserved for PR merges or nightly builds**.

| Tier | Directory | Scope | Dependencies | Parallelism | CI Cadence |
|---|---|---|---|---|---|
| **Tier 1** | `ci/tier1_unit/` | Pure logic, math, Pydantic validation. | None (No I/O) | High (Process-level) | Every Commit |
| **Tier 2** | `ci/tier2_logic/` | Subsystem logic, state transitions. | Isolated FS, Mocked gRPC, Contract Tests | High (Isolated State) | Every Commit |
| **Tier 3** | `ci/tier3_fleet/` | End-to-end distributed flows. | Docker Containers (Testcontainers), Real gRPC | Medium (Isolated Net) | PRs / Nightly |
| **Tier 4** | `ci/tier4_chaos/` | Fault tolerance & recovery. | Tier 3 + Fault Injection | Low (Subnet Isolation) | PRs / Nightly |

---

## 3. Mandatory State Isolation

The most common friction point is state collision. We will enforce complete isolation via the `auto_isolate` fixture, extending beyond the filesystem to include databases and telemetry streams.

### The `auto_isolate` Fixture
Applied automatically to all Tier 2+ tests:
```python
@pytest.fixture(autouse=True)
def auto_isolate(tmp_path, monkeypatch, worker_id):
    """Guarantees 100% isolated state for every test."""
    # Redirect state hierarchy to ephemeral test directory
    monkeypatch.setenv("PSETI_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("PSETI_CONTROL", str(tmp_path / "control"))
    
    # Telemetry and Database Isolation
    # Assign unique Redis DBs and Loki Tenant IDs based on xdist worker_id
    db_index = int(worker_id.replace("gw", "")) if worker_id != "master" else 0
    monkeypatch.setenv("REDIS_DB", str(db_index))
    monkeypatch.setenv("LOKI_TENANT_ID", f"test_tenant_{db_index}")
    
    # Ensure role-segregated tree exists (locks/, runs/, transfer/, etc.)
    from control.utils.paths import PanoPaths
    PanoPaths.ensure_state_dirs()
    
    yield tmp_path
```

---

## 4. Incorporating Existing Test Utilities & Large-Scale Topologies

We will leverage and extend existing helpers to provide a high-level API for test authors, scaling them to handle realistic network conditions and large configurations.

### A. Dynamic Fleets & Large-Scale Topologies
The existing `ci/integration/fleet.py` and utilities in `ci/test/topologies` will be refined to support large-scale, valid networks mimicking real sites (e.g., Lick and Palomar).
- **Commit Fully to `testcontainers-python`**: We will discard static `docker-compose` files for Tier 3/4 fleet tests. `fleet.py` will exclusively use `testcontainers-python` to spin up ephemeral `daqnode` containers. This binds the container lifecycle directly to the Pytest garbage collector, preventing zombie containers on test crashes and allowing completely parallelized execution without port collisions.
- **Topology Generation**: Use the `ci/test/topologies` utilities to generate valid `daq_config.json`, `obs_config.json`, and `network_config.json` matrices. This ensures tests validate realistic environments involving multiple Domes, Modules, and complex Network Routing (e.g., emulating Palomar's port forwarding and distinct subnets).

### B. State Inspection (`state_probe.py`)
`StateProbe` will be the primary tool for asserting distributed state.
- **Pattern**: `assert probe.current_run() == run_name` or `assert probe.any_pff_files(run_name)`
- **Extension**: Add `probe.ledger_status()` to wrap `RunStateManager` checks across isolated state directories, and robust log parsing isolated by `LOKI_TENANT_ID`.

### C. Mock Observatory Objects & Preventing "Mock Drift"
We will introduce high-level software mocks to decouple Tier 2 logic tests from hardware requirements:
- **`MockHeadNode`**: Central orchestrator. Manages configuration generation and verifies control plane command sequencing.
- **`MockDaqNode`**: Simulates a data recorder. Encapsulates gRPC mock responses, shared volume management, and `hashpipe` lifecycle simulation.
- **`MockModule`**: A logical grouping of 4 independent `MockQuabos`.
- **Contract Tests Mandate**: To prevent "Mock Drift", we will implement strict Contract Tests in Tier 3 that verify `MockHeadNode` and `MockDaqNode` behave exactly identically to the physical hardware responses and actual gRPC schemas. These contract tests ensure our high-speed Tier 2 mocks never diverge from production reality.

---

## 5. Domain-Driven Organization

Tests within each tier will be grouped by the domain they protect.

| Domain | Filename | Examples |
|---|---|---|
| **Configuration** | `test_config.py` | Range validation, topology consistency, port-forwarding logic. |
| **Ledger** | `test_ledger.py` | Transactional status transitions, lock acquisition. |
| **Transfer** | `test_transfer.py` | Daemon poll loops, manifest generation, cleanup validation. |
| **Lifecycle** | `test_lifecycle.py` | `StartTransaction` rollbacks, `StopTransaction` teardown. |
| **Telemetry** | `test_telemetry.py` | Log shipping, JSONL formatting, Redis backpressure. |

---

## 6. Descriptive Naming & Standards

### BDD-Style Naming
`test_when_[state_or_action]_then_[expectation]`
*Example*: `test_when_cleanup_fails_on_one_node_then_archiving_is_blocked`

### High-Density Docstrings
Every test class or function must strictly document:
1. **Intent**: The architectural invariant or requirement being validated.
2. **Scenario**: The exact inputs, mock topology, or faults being injected.
3. **Assertion**: The precise state, data path, or gRPC response verified.

---

## 7. Infrastructure & Runner Modernization

### Runner CLI (`test_cli.py`)
- **Dynamic `.env` Templating**: Deprecate monolithic `.env` definitions in `qa.toml`. The runner will dynamically template `.env.{suite_id}` files per execution to assign random, non-overlapping IPv4 subnets (e.g., `10.x.y.0/24`) and mapped host ports, providing a final layer of network isolation.
- **Domain Slice Execution**: Allow developers to filter runs by domain, enabling commands like `pseti test sw logic transfer`.

### Build Efficiency & Multi-Stage Docker
- **Multi-Stage `Dockerfile.ci`**: Use a dedicated `builder` stage to compile C/C++ extensions (e.g., `hashpipe.so`) to keep the final runner image extremely lean (< 200MB).
- **uv Caching & Layering**: Maximize the speed of Python builds using `uv` lockfiles and `--mount=type=cache,target=/root/.cache/uv`. Structure `COPY` layers so codebase modifications do not invalidate dependency caches.

---

## 8. Implementation Roadmap (Iterative)

1. **Phase 1 (COMPLETED)**: Implement the expanded `auto_isolate` fixture.
   - **Status**: Core logic isolation is active via `control/ci/fixtures/conftest.py`.
   - **Infrastructure**: Introduced `MockDaqNode`, `make_transfer_job`, and `simulate_daq_fs` factories in `control/ci/fixtures/`.
   - **Contract Verification**: Tier 2 logic tests now include `test_contract_mocks.py` to prevent drift against production gRPC models.
2. **Phase 2**: Refine the topology generation utilities (`ci/test/topologies`) to output multi-site mock configurations matching the Palomar architecture.
3. **Phase 3**: Migrate remaining legacy integration/chaos scenarios into the Domain-Driven Tier 2 and Tier 4 structures.
4. **Phase 4**: Add `testcontainers-python` to `pyproject.toml` and migrate `fleet.py` to support dynamic Tier 3 fleet orchestration.
