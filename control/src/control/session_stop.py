#! /usr/bin/env python3

import contextlib
import os
import signal
import time

import typer
from panoseti_grpc.telemetry.logger import get_logger

import control.power as power
from control.utils import config_file, util
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import ObsConfig

log_dir = PanoPaths.logs_dir()
log_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger("PSETI.SessionStop", log_dir=str(log_dir), grpc_enabled=True)


def _stop_transfer_daemon(timeout: float = 30.0) -> None:
    """Send SIGTERM to the transfer daemon and wait for it to exit gracefully.

    Reads the PID from ``state/transfer/daemon.pid``. If the daemon does not
    exit within *timeout* seconds, escalates to SIGKILL. Silently does nothing
    if the pid file is absent or the process is already gone.

    Args:
        timeout: Seconds to wait for graceful shutdown before escalating.
    """
    pid_path = PanoPaths.state_dir() / "transfer" / "daemon.pid"
    if not pid_path.exists():
        return
    try:
        pid = int(pid_path.read_text().strip())
    except (ValueError, OSError):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    logger.info("Waiting for transfer daemon (pid=%d) to finish current job...", pid)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)  # probe: raises ProcessLookupError if dead
        except ProcessLookupError:
            logger.info("Transfer daemon exited cleanly.")
            return
        time.sleep(1.0)
    logger.warning("Transfer daemon did not exit after %.0fs; sending SIGKILL.", timeout)
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def session_stop(obs_config: ObsConfig) -> None:
    """Gracefully terminate an observing session.

    Stops the transfer daemon (SIGTERM, up to 30 s), powers off all modules,
    and stops background Redis daemons.

    Args:
        obs_config: Validated observatory configuration.
    """
    _stop_transfer_daemon()
    power.do_all(obs_config, 'off')
    try:
        util.stop_redis_daemons()
    except PermissionError:
        logger.error("You don't have permission to stop the redis daemons. "
                     "Try running 'sudo ./config.py --stop_redis_daemons'.")

app = typer.Typer(help="Gracefully terminate an observing session.", no_args_is_help=False, context_settings={"help_option_names": ["-h", "--help"]})

@app.command()
def main() -> None:
    """
    Gracefully terminate an observing session.
    
    Powers off all modules and stops background Redis daemons.
    """
    obs_config = config_file.get_obs_config()
    session_stop(obs_config)


if __name__ == "__main__":
    app()
    

