"""
StartTransaction — context manager for a transactional observing run startup.

Split out of start.py so the rollback ladder (lock management, DAQ-node
StopDaq escalation, Quabo data-flow teardown, daemon kill, partial-artifact
archival) can be read, tested, and reasoned about independently of the CLI
and pre-flight validation code.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import traceback
from typing import Any

from panoseti_grpc.daq_control.client import AsyncDaqControlClient
from panoseti_grpc.telemetry.logger import get_logger

from control.utils import util
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import DaqConfig, DaqNode, NetworkConfig, QuaboUids, RunStatus
from control.utils.run_state import RunStateManager, ValidationError

log_dir = PanoPaths.logs_dir()
log_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger(
    "PSETI.Start",
    log_dir=log_dir,
    grpc_enabled=True,
    reset=True
)


class StartTransaction:
    """
    Context manager for a transactional observing run startup.
    Implements a robust rollback ladder and lock management.
    """
    def __init__(
        self,
        state_mgr: RunStateManager,
        run_name: str,
        daq_config: DaqConfig,
        quabo_uids: QuaboUids,
        network_config: NetworkConfig,
        process_mgr: Any = None,
        net_client: Any = None,
        fs_mgr: Any = None,
    ) -> None:
        self.state_mgr = state_mgr
        self.run_name = run_name
        self.daq_config = daq_config
        self.quabo_uids = quabo_uids
        self.network_config = network_config
        self.process_mgr = process_mgr
        self.net_client = net_client
        self.fs_mgr = fs_mgr
        self.success = False
        # Set to True in start_run immediately before start_data_flow() is called.
        # Only this transaction may undo data flow that it started — calling
        # stop_data_flow() on a rollback from a pre-flight failure would silently
        # halt an already-running valid observation on the same Quabos.
        self.data_flow_started: bool = False
        # Set to True in start_run after the ledger is initialized for THIS run.
        self.ledger_initialized: bool = False
        # Set of IP addresses of nodes that this transaction attempted to start.
        self.nodes_attempted: set[str] = set()

    async def __aenter__(self) -> StartTransaction:
        await asyncio.to_thread(self.state_mgr.acquire_lock)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        try:
            if exc_type is not None:
                # Ladder Step 0: Identify the failure
                if exc_type is ValidationError:
                    logger.warning(f"Aborting start due to validation failure: {exc_val}")
                else:
                    # Use traceback.format_exception to handle ExceptionGroups (Python 3.11+)
                    # This automatically renders the nested tree of sub-exceptions.
                    full_tb = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
                    summary = str(exc_val)
                    if isinstance(exc_val, ExceptionGroup):
                        sub_errs = ", ".join(type(e).__name__ + ": " + str(e) for e in exc_val.exceptions)
                        summary = f"{summary} -> {sub_errs}"
                    logger.error(f"[FAILURE] Start process aborted: {summary}\n{full_tb}")

                logger.info("Triggering Rollback Ladder...")

                # Wait briefly for cancelled tasks to finish their synchronous I/O
                await asyncio.sleep(0.2)

                # Ladder Step 1: Update ledger to ABORTED immediately (WAL pattern)
                # We re-load to ensure we have any node receipts written just before cancellation
                if self.ledger_initialized:
                    try:
                        ledger = await asyncio.to_thread(self.state_mgr.load_state)
                        if ledger and ledger.run_name == self.run_name:
                            ledger.status = RunStatus.ABORTED
                            await asyncio.to_thread(self.state_mgr.save_state, ledger)
                    except Exception as e_led:
                        logger.error(f"Failed to update ledger to ABORTED: {e_led}")
                else:
                    logger.info("Skipping ledger status update — this transaction never initialized the ledger.")

                # Ladder Step 2: Stop remote DAQ nodes (Any that were attempted)
                logger.info("Stopping remote DAQ nodes...")
                # Re-load again to be absolutely sure we have all concurrent updates
                try:
                    ledger = await asyncio.to_thread(self.state_mgr.load_state)
                except Exception:
                    ledger = None

                async def rollback_node(node: DaqNode) -> None:
                    # Rollback logic MUST only apply to nodes that THIS transaction
                    # attempted to start. Rolling back nodes from a pre-existing
                    # conflicting run would violate the non-interference invariant.
                    if str(node.ip_addr) not in self.nodes_attempted:
                        return

                    receipt = next((n for n in ledger.nodes if str(n.ip_addr) == str(node.ip_addr)), None) if ledger else None
                    if not receipt:
                        return

                    logger.info(f"Rolling back node {node.ip_addr} (Status: {receipt.status})...")
                    try:
                        grpc_host, grpc_port = util.daq_grpc_endpoint(node, self.daq_config)
                        async with AsyncDaqControlClient(host=grpc_host, port=grpc_port) as client:
                            await client.StopDaq({'data_dir': node.data_dir, 'run_dir': self.run_name}, timeout=15.0)
                    except Exception as stop_err:
                        logger.error(f"StopDaq RPC failed for {node.ip_addr} during rollback ({stop_err}). Escalating to SSH pkill...")
                        try:
                            ssh_args = ["ssh", *util.ssh_options]
                            if node.port_forwarding and node.port_forwarding.status:
                                real_ip = str(node.port_forwarding.gw_ip)
                                port = str(node.port_forwarding.port)
                                ssh_args.extend(["-p", port, f"{node.username}@{real_ip}"])
                            else:
                                ssh_args.append(f"{node.username}@{node.ip_addr}")

                            ssh_args.append("pkill -9 hashpipe")
                            res = await asyncio.create_subprocess_exec(*ssh_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            await res.wait()
                            if res.returncode in [0, 1]:
                                logger.info(f"Hard-kill escalation succeeded for node {node.ip_addr}")
                            else:
                                logger.critical(f"Hard-kill SSH escalation failed for node {node.ip_addr} (rc={res.returncode})")
                        except Exception as ssh_err:
                            logger.critical(f"Failed to stop node {node.ip_addr} even with SSH escalation: {ssh_err}")
                            logger.exception(ssh_err)

                async with asyncio.TaskGroup() as tg:
                    for node in self.daq_config.daq_nodes:
                        if node.module_ids:
                            tg.create_task(rollback_node(node))

                # Ladder Step 3: Stop Quabo data flow — ONLY if this transaction
                # was the one that started it. Calling stop_data_flow without
                # having called start_data_flow would interrupt a pre-existing
                # valid observation on the same Quabos.
                if self.data_flow_started:
                    logger.info("Stopping Quabo data flow (started by this transaction)...")
                    try:
                        await asyncio.to_thread(util.stop_data_flow, self.quabo_uids, self.network_config)
                    except Exception as e2:
                        logger.error(f"Failed to stop Quabo data flow: {e2}")
                else:
                    logger.info("Skipping stop_data_flow — this transaction never called start_data_flow.")

                # Ladder Step 4: Kill local daemons
                logger.info("Stopping local daemons...")
                try:
                    if self.process_mgr:
                        await asyncio.to_thread(self.process_mgr.kill, util.hk_recorder_name)
                        await asyncio.to_thread(self.process_mgr.kill, util.hv_updater_name)
                        await asyncio.to_thread(self.process_mgr.kill, util.module_temp_monitor_name)
                    else:
                        await asyncio.to_thread(util.kill_hk_recorder)
                        await asyncio.to_thread(util.kill_hv_updater)
                        await asyncio.to_thread(util.kill_module_temp_monitor)
                except Exception as e3:
                    logger.error(f"Failed to kill local daemons: {e3}")

                # Ladder Step 5: Archive partial artifacts
                try:
                    aborted_base = f"{self.daq_config.head_node_data_dir}/_aborted/{self.run_name}"
                    suffix = 1
                    aborted_dir = aborted_base
                    while await asyncio.to_thread(os.path.exists, aborted_dir):
                        aborted_dir = f"{aborted_base}_{suffix}"
                        suffix += 1

                    logger.info(f"Archiving partial artifacts to {aborted_dir}")
                    await asyncio.to_thread(os.makedirs, aborted_dir, exist_ok=True)

                    # Write failure context
                    err_msg = str(exc_val)
                    full_tb = "".join(traceback.format_exception(exc_type, exc_val, exc_tb)) if exc_tb else ""
                    def dump_context(msg: str, tb: str) -> None:
                        with open(f"{aborted_dir}/start_failure_context.json", "w") as f:
                            json.dump({"error": msg, "traceback": tb}, f, indent=4)
                    await asyncio.to_thread(dump_context, err_msg, full_tb)

                    local_run_dir = f"{self.daq_config.head_node_data_dir}/{self.run_name}"
                    if await asyncio.to_thread(os.path.exists, local_run_dir):
                        items = os.listdir(local_run_dir)
                        logger.info(f"Found {len(items)} items in {local_run_dir} to archive: {items}")
                        for item in items:
                            s = os.path.join(local_run_dir, item)
                            d = os.path.join(aborted_dir, item)
                            await asyncio.to_thread(shutil.move, s, d)
                        await asyncio.to_thread(os.rmdir, local_run_dir)
                    else:
                        logger.warning(f"local_run_dir {local_run_dir} does not exist; nothing to archive.")
                except Exception:
                    logger.exception("Failed to archive partial artifacts (non-fatal)")

            if exc_type is ValidationError:
                return True # Suppress validation errors for a clean exit

            if exc_type is not None:
                return not issubclass(exc_type, (KeyboardInterrupt, SystemExit, asyncio.CancelledError))

            return False

        finally:
            await asyncio.to_thread(self.state_mgr.release_lock)
