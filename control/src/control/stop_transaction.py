"""
StopTransaction — context manager for a transactional observing run shutdown.

Split out of stop.py so the teardown ladder (StopDaq fan-out, daemon kill,
Quabo data-flow stop, transfer-job enqueue, ledger finalization) can be read,
tested, and reasoned about independently of the CLI.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
from datetime import UTC, datetime
from typing import Any

from panoseti_grpc.telemetry.logger import get_logger

from control.interfaces import FileSystemManager, NetworkClient, ProcessManager
from control.transfer.models import TransferJob, TransferNodeSpec, TransferStatus
from control.transfer.queue import TransferQueue
from control.utils import util
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import DaqConfig, DaqNode, DataConfig, NetworkConfig, QuaboUids, RunStatus
from control.utils.run_state import RunStateManager, ValidationError
from control.utils.util import now_str, recording_ended_filename

log_dir = PanoPaths.logs_dir()
log_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger("PSETI.Stop", log_dir=str(log_dir), grpc_enabled=True)


def write_complete_file(run_dir: str, filename: str) -> None:
    """Write an empty marker file to indicate recording finished."""
    with open(f"{run_dir}/{filename}", "w") as f:
        f.write(now_str())


def complete_file_exists(run_dir: str, filename: str) -> bool:
    """Check if the completion marker exists."""
    return os.path.exists(f"{run_dir}/{filename}")


class StopTransaction:
    """
    Context manager for a transactional observing run shutdown.
    Ensures that all teardown steps execute even if one fails.
    """
    def __init__(
        self,
        state_mgr: RunStateManager,
        daq_config: DaqConfig,
        network_config: NetworkConfig,
        quabo_uids: QuaboUids,
        data_config: DataConfig,
        run: str | None,
        no_collect: bool,
        no_cleanup: bool,
        no_transfer: bool,
        skip_verify: bool,
        force_stop: bool,
        cancel_event: asyncio.Event,
        process_mgr: ProcessManager,
        net_client: NetworkClient,
        fs_mgr: FileSystemManager,
    ) -> None:
        self.state_mgr = state_mgr
        self.daq_config = daq_config
        self.network_config = network_config
        self.quabo_uids = quabo_uids
        self.data_config = data_config
        self.run = run
        self.no_collect = no_collect
        self.no_cleanup = no_cleanup
        self.no_transfer = no_transfer
        self.skip_verify = skip_verify
        self.force_stop = force_stop
        self.cancel_event = cancel_event
        self.process_mgr = process_mgr
        self.net_client = net_client
        self.fs_mgr = fs_mgr
        self.all_errors: list[str] = []
        self.success = False

    async def __aenter__(self) -> StopTransaction:
        await asyncio.to_thread(self.state_mgr.acquire_lock)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        try:
            if exc_type is ValidationError:
                logger.warning(f"Aborting stop due to validation failure: {exc_val}")
                self.success = True
                return True

            # Ladder Step 0: Log fundamental failures from the 'with' block, then
            # fall through so the teardown ladder always executes.  An early return
            # here would leave DAQ nodes and Quabos running.
            if exc_type is not None:
                self.last_exception = str(exc_val)
                logger.error(f"[CRITICAL FAILURE] Stop transaction entered with exception: {exc_val}. "
                             "Continuing teardown ladder to avoid leaving hardware in a running state.")

            if not self.run:
                if exc_type is None:
                    self.success = True
                return False

            # Ensure all shutdown steps execute even if one fails
            logger.info(f"Initiating teardown ladder for run {self.run}")

            # 1. Stop recording on DAQ nodes
            try:
                async def stop_node_task(node: DaqNode) -> None:
                    try:
                        ok = await self.net_client.stop_daq_node(node, timeout_s=20.0, retries=2)
                        if not ok:
                            self.all_errors.append(f"StopDaq failed for {node.ip_addr}")
                    except Exception as e:
                        self.all_errors.append(f"StopDaq error for {node.ip_addr}: {e}")

                async with asyncio.TaskGroup() as tg:
                    for node in self.daq_config.daq_nodes:
                        if node.module_ids:
                            tg.create_task(stop_node_task(node))
            except Exception as e:
                self.all_errors.append(f"stop_recording (parallel) failed: {e}")

            # 2. Kill local daemons
            for daemon_name, process_id in [
                ("HV updater", util.hv_updater_name),
                ("HK recording", util.hk_recorder_name),
                ("Temperature monitor", util.module_temp_monitor_name)
            ]:
                try:
                    logger.info(f"stopping {daemon_name}")
                    await asyncio.to_thread(self.process_mgr.kill, process_id)
                except Exception as e:
                    self.all_errors.append(f"{daemon_name} shutdown failed: {e}")

            # 3. Stop Quabo data flow
            try:
                logger.info("stopping data generation from quabos")
                await asyncio.to_thread(util.stop_data_flow, self.quabo_uids, self.network_config)
            except Exception as e:
                self.all_errors.append(f"stop_data_flow failed: {e}")

            # 4. Enqueue for background transfer and transition ledger
            data_dir = self.daq_config.head_node_data_dir
            run_dir = f'{data_dir}/{self.run}'
            run_dir_exists = await asyncio.to_thread(os.path.exists, run_dir)

            if not run_dir_exists:
                msg = f"Run dir {data_dir}/{self.run} not found; recorded artifacts may be missing."
                logger.error(msg)
                self.all_errors.append(msg)

            # --- DECISION: Should we enqueue for transfer? ---
            # We only enqueue if the teardown ladder started cleanly (exc_type is None)
            # AND the local run directory exists.
            can_enqueue = (exc_type is None) and run_dir_exists

            if can_enqueue:
                if not await asyncio.to_thread(complete_file_exists, run_dir, recording_ended_filename):
                    await asyncio.to_thread(write_complete_file, run_dir, recording_ended_filename)

                if self.no_transfer:
                    logger.info(f"Skipping transfer enqueue for run {self.run} (--no-transfer)")
                else:
                    # Construct job
                    job = TransferJob(
                        run_name=self.run,
                        head_data_dir=data_dir,
                        head_node_username=__import__('getpass').getuser(),
                        created_at=datetime.now(UTC),
                        no_cleanup=self.no_cleanup,
                        no_collect=self.no_collect,
                        skip_verify=self.skip_verify,
                        daq_nodes=[
                            TransferNodeSpec(
                                ip_addr=n.ip_addr,
                                username=n.username,
                                data_dir=n.data_dir,
                                module_ids=n.module_ids,
                                port_forwarding=n.port_forwarding,
                                grpc_port=n.grpc_port
                            )
                            for n in self.daq_config.daq_nodes if n.module_ids
                        ]
                    )

                    # Enqueue
                    tq = TransferQueue()
                    await asyncio.to_thread(tq.enqueue, job)
                    logger.info(f"Enqueued run {self.run} for transfer")

                    # Snapshot the transfer job and the finalized ledger into the run directory.
                    # This ensures the .pffd contains a complete record of the run lifecycle.
                    with contextlib.suppress(Exception):
                        job_path = tq._job_path(TransferStatus.PENDING, self.run)
                        if job_path.exists():
                            shutil.copy2(job_path, f"{run_dir}/transfer_job.toml")

                        ledger_path = PanoPaths.runs_dir() / "ledger.toml"
                        if ledger_path.exists():
                            shutil.copy2(ledger_path, f"{run_dir}/run_ledger.toml")

            # Finalize ledger
            final_status = RunStatus.STOPPED_WITH_ERRORS if exc_type is not None else RunStatus.RECORDING_ENDED
            extra_fields = {}
            if exc_type is not None and getattr(self, 'last_exception', None):
                extra_fields["last_transfer_error"] = self.last_exception
            self.state_mgr.transition(final_status, **extra_fields)
            self.success = True
            return True

        except Exception as e:
            self.all_errors.append(f"StopTransaction cleanup failed: {e}")
            return False
        finally:
            await asyncio.to_thread(self.state_mgr.release_lock)
