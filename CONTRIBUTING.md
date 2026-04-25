# Contributing to PANOSETI Software

This guide outlines the engineering standards, TDD philosophy, and architectural invariants for PANOSETI. These mandates apply to both human developers and AI coding agents.

## 🧪 TDD & Testing Philosophy
We prioritize empirical verification and transactional integrity.

1. **Reproduction First**: For bug fixes, you MUST reproduce the failure with a new test case before applying the fix.
2. **Tiered Validation**:
   - **Tier 1 (Unit)**: Zero-dependency logic and parsing.
   - **Tier 2 (Logic)**: Mocked gRPC with real filesystem state. MUST use `PSETI_STATE` isolation.
   - **Tier 3 (Fleet)**: Full E2E Docker fleet simulation.
3. **Chaos-Forced Green**: Transactional changes are considered broken if they pass on localhost but fail in the chaos suite (`pseti test sw chaos`).
4. **State Isolation**: ALL integration tests must isolate their state by redirecting `PSETI_STATE` to a unique temporary directory.

## 📏 Code Style & Formatting
We use **Python 3.14+**, **uv**, **Ruff**, and **MyPy**.

1. **Pydantic Authority**: Pass instantiated models across function boundaries. Signatures MUST use strictly-typed models. Avoid `dict[str, Any]` fallbacks.
2. **Attribute Access**: Always prefer `model.field` over dictionary-style `model["field"]`.
3. **Path Totality**: Never construct paths via string concatenation. Use `control.utils.paths.PanoPaths`.
4. **Async-First**: The control plane is asynchronous. Never use blocking I/O or `asyncio.to_thread` wrappers for gRPC. Use native async context managers and `asyncio.TaskGroup`.

## 🧠 Architectural Invariants
1. **Atomic Receipt First (WAL)**: Always write node receipts to the state ledger *before* issuing gRPC control calls.
2. **Transaction Managers**: Observing run lifecycles MUST be managed by the `StartTransaction` and `StopTransaction` context managers to ensure the rollback ladder is respected.
3. **Atomic Locking**: Mutual exclusion is enforced via low-level `os.O_EXCL` on `state/locks/`. standard `flock` is prohibited on Docker volumes.

## 🤖 Guidance for AI Agents
1. **Context Efficiency**: Combine search and read operations. Use `grep_search` to identify targets before surgical `read_file` calls.
2. **Explain Before Acting**: Provide a concise one-sentence technical rationale immediately before executing tool calls.
3. **Validation is Finality**: A task is incomplete until behavioral correctness is verified via passing tests and linting.

## 📁 Documentation Routing
For detailed technical guides, refer to:
- **`control/GEMINI.md`**: Foundational architectural mandates.
- **`control/TRANSACTIONS.md`**: Rollback ladder and run state logic.
- **`control/DEBUGGING.md`**: Core principles and async pitfalls.
- **`control/TEST.md`**: Test suite usage and isolation standard.
