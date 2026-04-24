#! /usr/bin/env python3

# show the status of a recording run

from datetime import UTC, datetime

import typer
from panoseti_grpc.telemetry.logger import get_logger

from control.utils import config_file, util
from control.utils.paths import PanoPaths
from control.utils.run_state import RunStateManager


# ---------- logging setup ----------
def ut_now_str() -> str:
    """Return the current time as a formatted UTC string."""
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

def ut_date_str() -> str:
    """Return the current date as a YYYYMMDD string."""
    return datetime.now(UTC).strftime("%Y%m%d")

log_dir = PanoPaths.logs_dir()
log_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger("PSETI.Status", log_dir=str(log_dir), grpc_enabled=True)

# ---------- main logic ----------
def status() -> None:
    """Query and display the current status of the observatory control plane.
    
    Checks the transactional ledger, local markers, and probes remote DAQ 
    nodes via gRPC/SSH to report on Hashpipe liveness and disk usage.
    """
    state_mgr = RunStateManager()
    ledger = state_mgr.load_state()
    
    if ledger:
        logger.info(f'Run in ledger: {ledger.run_name} (Status: {ledger.status}, Started: {ledger.start_time})')
    else:
        run_name = util.read_run_name()
        if run_name:
            logger.info(f'Run in legacy marker: {run_name}')
        else:
            logger.info("No run is in progress")

    if util.is_hk_recorder_running():
        logger.info('HK recorder is running')
    else:
        logger.info('HK recorder is not running')

    # in theory should use config files in run dir
    config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    config_file.get_data_config()
    config_file.associate(daq_config, quabo_uids)

    for node in daq_config.daq_nodes:
        if not node.module_ids:
            continue
        ip_addr = str(node.ip_addr)
        logger.info(f'status on DAQ node {ip_addr}:')
        j = util.get_daq_node_status(node)

        if j['hashpipe_running']:
            logger.info('   hashpipe is running')
        else:
            logger.info('   hashpipe is not running')

        if 'current_run' in j:
            logger.info(f'   current run: {j["current_run"]}')
            if 'current_run_disk' in j:
                logger.info(f'   disk usage: {j["current_run_disk"]}')
            else:
                logger.info("   run dir doesn't exist")
        else:
            logger.info('   no current run')

        vols = j['vols']
        logger.info('   volumes:')
        for name in vols:
            vol = vols[name]
            logger.info(f'      name: {name}')
            logger.info('         free space: %.2fGB' % (vol['free'] / 1e9))
            logger.info(f'         modules: {vol["modules"]}')


app = typer.Typer(help="Show the status of a PSETI recording run.", no_args_is_help=False)

@app.command()
def main() -> None:
    """
    Query and display the current status of the observatory control plane.
    
    Checks the transactional ledger, local markers, and probes remote DAQ 
    nodes via gRPC/SSH to report on Hashpipe liveness and disk usage.
    """
    status()

if __name__ == "__main__":
    app()

