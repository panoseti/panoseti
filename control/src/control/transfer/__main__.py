"""Launch the transfer daemon via `python -m control.transfer`."""
import asyncio

from panoseti_grpc.telemetry.logger import get_logger

from control.transfer.daemon import run_daemon
from control.utils.paths import PanoPaths

log_dir = PanoPaths.daemon_logs_dir("transfer_daemon")
log_dir.mkdir(parents=True, exist_ok=True)
# Initialise the shared logger before run_daemon imports it at module level.
# This ensures the FileHandler is attached before any log record is emitted.
get_logger("transfer_daemon", log_dir=log_dir, grpc_enabled=False)

if __name__ == "__main__":
    asyncio.run(run_daemon())
