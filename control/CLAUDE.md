# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file supplements the root-level `../CLAUDE.md`, which covers the full repo architecture, hardware topology, config system, and observing run lifecycle. Read that first for context.

---

## Corrections to root CLAUDE.md

The root CLAUDE.md has some stale entries for the `control/` package:

- **Python version**: `requires-python = ">=3.14"` (not 3.9)
- **CI runner**: `python ci/qa.py <cmd>`
- **Integration test count**: 65 passing
- **Unit test count**: 460 passing
- **Chaos/scenario test count**: 114 tests (91 active, 23 stubs) in `ci/integration/scenarios/`

---

## Verification & Quality Standards

### Linting and Type Safety
The project enforces strict linting via Ruff and type checking via MyPy. All new code must pass `python ci/qa.py lint`.

- **Pydantic Model Authority**: Instantiated models from `utils/pydantic_config_models.py` must be passed across call boundaries. Polymorphic functions must validate dictionaries into models at the entry point.
- **Attribute Access**: Always prefer model attribute access (`config.daq_nodes`) over dictionary indexing (`config['daq_nodes']`).
- **MyPy Strictness**: Avoid `type: ignore` whenever possible. If required, use it on a specific line with a comment explaining why. Ensure `unused-ignore` rules pass.

### Documentation (Google Style Docstrings)
All functions and methods must have high-quality docstrings. Preserving legacy comments (prefixed with `#`) by transforming them into formal docstrings is mandatory.

---

## Transaction Logic
The observatory uses a **Context Manager Architecture** to manage the lifecycle of an observing run.
- **StartTransaction**: Handles atomic locking and a multi-step rollback ladder. If any startup step fails, it guarantees all hardware and remote processes are restored to a safe state.
- **StopTransaction**: Implements a resilient teardown sequence, ensuring that all shutdown tasks (collection, cleanup) execute even if individual steps fail.
- **Distributed Ledger**: State is persisted in `tmp/run_state.toml`.

Read [TRANSACTIONS.md](TRANSACTIONS.md) for detailed diagrams and rollback rules.

---

## Testing and Debugging
- **Unit Tests**: Add new cases to `ci/unit/` for every utility function. No hardware or network access is allowed.
- **Integration Tests**: Verify end-to-end flows in `ci/integration/`. Use `-k` to isolate failures.
- **Chaos Tests**: Verifies transaction integrity under failure conditions in `ci/integration/scenarios/`. Run via `python ci/qa.py chaos`.
- **Atomic Locking**: Locks are managed via `os.O_EXCL` file creation with stale PID detection. Orphaned locks from crashed runs are self-healing.
- **Telemetry Integration**: Logs are shipped via non-blocking gRPC handlers to a central Loki instance.

Read [DEBUGGING.md](DEBUGGING.md) for advanced troubleshooting techniques and [ci/README.md](ci/README.md) for test architecture details.

---

## Run tests

```bash
# Docker-based (preferred — matches CI exactly)
python ci/qa.py up           # start persistent background containers once
python ci/qa.py unit         # Parallel unit tests
python ci/qa.py integration  # E2E with real hashpipe
python ci/qa.py chaos        # Chaos/TDD-forcing scenarios
python ci/qa.py lint         # ruff + mypy concurrently
python ci/qa.py down         # tear down

# Targeted test runs
python ci/qa.py chaos -k SCN003 -vv    # Verbose scenario debugging
python ci/qa.py integration -k "real_data"

# Native (no Docker, unit tests only)
uv sync --all-extras
uv run pytest ci/unit/
```

The `chaos` command runs `ci/integration/scenarios/`.

---

## CI Architecture Notes
- **Persistent containers**: `python ci/qa.py up` starts containers that are reused across runs to minimize overhead.
- **Live mount**: `control/` is volume-mounted into containers; source edits are visible instantly.
- **Validation Leniency**: In CI, we bypass strict hardware checks if `daq_config.json` has `head_node_container: true`.
- **Networking**: `headnode_net` (10.0.1.0/24) hosts telemetry and Loki; `daqnode_net` (192.168.0.0/24) hosts the DAQ fleet.
- **Loki Pipeline**: Logs are queued in Redis (`logs:ingress`) and processed by `storeLoki.py` with non-blocking resilience.

Read [ci/README.md](ci/README.md) for the full network topology and test isolation strategy.
