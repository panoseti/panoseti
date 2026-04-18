# Observing Run Transactions

This document describes the transactional integrity and rollback mechanisms implemented in the PANOSETI observatory control plane (`start.py` and `stop.py`).

## Overview

The observatory control plane manages a distributed system consisting of a head node, multiple DAQ nodes (running Hashpipe), and numerous Quabo detector boards. Starting or stopping an observation is a complex operation that must be handled atomically to avoid leaving the system in an inconsistent or "stuck" state.

## State Management & Locking

### Advisory Locking
To prevent concurrent or conflicting operations, all control processes must acquire an exclusive advisory lock on `tmp/panoseti_control.lock`. This ensures that only one `start.py`, `stop.py`, or `config.py` process is active at a time.

### Distributed Ledger
The system state is persisted in a TOML-based ledger located at `tmp/run_state.toml`. This ledger tracks the lifecycle of an observing run:
- **`STARTING`**: Initializing hardware and remote nodes.
- **`ACTIVE`**: Observation is currently running and recording.
- **`STOPPING`**: Graceful shutdown and data collection in progress.
- **`COMPLETED`**: Run finished successfully; all data centralized.
- **`ABORTED`**: Run failed during startup or was manually cancelled; rollback performed.

The ledger also contains **Node Receipts**, which track the status of each remote DAQ node (e.g., Hashpipe PID, local data directory).

## Start Transaction (`start.py`)

The startup sequence is handled as a single transaction with a "Rollback Ladder" for error recovery.

### 1. Pre-flight Checks
- Acquire advisory lock.
- Validate all configuration files (Tier-1 and Tier-2).
- Ensure no other run is active in the ledger.

### 2. Initialization
- Record `STARTING` state in the ledger.
- Create local run directories on the head node.
- Concurrent remote initialization:
    - Create run directories on remote DAQ nodes via gRPC/SSH.
    - Copy configuration files to remote nodes.

### 3. Execution
- **Start DAQ**: Concurrent gRPC calls to all DAQ nodes to launch the Hashpipe pipeline.
- **Liveness Probe**: 2-second pause followed by a status check to verify Hashpipe is `ALIVE`.
- **Start Data Flow**: Configure Quabos to point their UDP streams to the assigned DAQ nodes.
- **Persistence**: Update ledger to `ACTIVE`.

### 4. Rollback Ladder
If any step fails, the transaction is aborted and a rollback is triggered:
1. **Stop remote DAQs**: Signal remote Hashpipe processes to exit.
2. **Stop Quabo flow**: Send zeroed parameters to all Quabos to halt data transmission.
3. **Kill local daemons**: Terminate any background monitoring processes.
4. **Archive artifacts**: Move partial run data to `_aborted/` with a context dump for debugging.
5. **Update Ledger**: Mark as `ABORTED`.

## Stop Transaction (`stop.py`)

The shutdown sequence prioritizes resilience and best-effort cleanup.

### 1. Resilient Shutdown
- Acquire advisory lock.
- **Stop Recording**: Concurrent gRPC calls to stop remote Hashpipes. Failures on individual nodes do not halt the shutdown of others.
- **Stop Interleave**: Gracefully stop the interleave daemon with SIGTERM, escalating to SIGKILL if necessary.
- **MAROC Restoration**: Synchronously restore MAROC registers to default values.

### 2. Data Collection
- Point ledger to `STOPPING`.
- **Collect Data**: rsync PFF artifacts from remote DAQ nodes to the head node.
- **Verify Integrity**: Ensure all expected files arrived.

### 3. Cleanup
- If data collection succeeded, remove artifacts from remote DAQ nodes.
- Remove the transactional ledger (`tmp/run_state.toml`) and release the lock.
- Mark state as `COMPLETED`.

## Summary of Constraints
- **Atomicity**: The system should never be left with Quabos sending data to a dead Hashpipe.
- **Identity**: Processes verify remote PIDs via `/proc/pid/cmdline` before signaling.
- **Non-blocking**: gRPC and rsync operations are performed concurrently where possible.
