#! /usr/bin/env python3

# start an "observing session":
# - open domes (TBD)
# - power on relevant modules
# - wait for quabos to come up
# - get quabo UIDs
# - reboot quabos
# - turn on HV (using levels from quabo config files)
# - set gain params of Marocs
# - do PH baseline calibration
# - start the Redis daemons
# - copy software to DAQ nodes

import sys
import time
from typing import Any

import typer
from panoseti_grpc.telemetry.logger import get_logger

import control.config as config
import control.get_uids as get_uids
import control.power as power
from control.utils import config_file, util
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import (
    DaqConfig,
    DataConfig,
    NetworkConfig,
    ObsConfig,
)

log_dir = PanoPaths.logs_dir()
log_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger("PSETI.SessionStart", log_dir=str(log_dir), grpc_enabled=True)

def session_start(
    obs_config: ObsConfig,
    quabo_info: dict[str, Any],
    data_config: DataConfig,
    daq_config: DaqConfig,
    network_config: NetworkConfig,
    no_hv: bool,
    stage: str
) -> None:
    """Orchestrate the initialization of a complete PANOSETI observing session.
    
    Performs a multi-stage startup sequence:
    1. poweron: Power on all modules via WPS and wait for boot.
    2. get_uids: Scan and cache hardware UIDs.
    3. reboot: Software reboot to ensure clean state and timing mode.
    4. hk_dest: Point Quabo telemetry to the head node.
    5. start_redis: Launch local redis daemons.
    6. maroc_config: Load ASIC gains and thresholds from calibration.
    7. mask_config: Configure trigger masks.
    8. calibrate_ph: Run baseline calibration for Pulse Height mode.

    Args:
        obs_config: Validated observatory physical configuration.
        quabo_info: Map of Quabo UIDs to metadata.
        data_config: Validated science observing parameters.
        daq_config: Validated DAQ node configuration.
        network_config: Network routing and port forwarding configuration.
        no_hv: If True, do not enable detector high voltage.
        stage: The starting phase of the sequence (e.g., 'poweron').
    """

    modules = config_file.get_modules(obs_config)
    # power on the telescopes
    if stage == 'poweron':
        stage = 'get_uids'
        power.do_all(obs_config, 'on')
        logger.info('waiting 60 secs for quabos to boot up')
        time.sleep(60)

    if stage == 'get_uids':
        stage = 'reboot'
        logger.info('getting quabo UIDs')
        quabo_uids = get_uids.get_uids(obs_config, network_config)
        if not quabo_uids:
            raise RuntimeError("Failed to get quabo UIDs")

    if stage == 'reboot':
        stage = 'hk_dest'
        modules = config_file.get_modules(obs_config)
        logger.info('rebooting quabos')
        quabo_uids = config_file.get_quabo_uids() # type: ignore[assignment]
        if not quabo_uids:
            raise RuntimeError("Missing quabo_uids.json")
        config.do_reboot(modules, quabo_uids, network_config)
        logger.info('Reboot Successfully.')

    if stage == 'hk_dest':
        stage = 'start_redis'
        logger.info('setting hk dest to this computer')
        quabo_uids = config_file.get_quabo_uids() # type: ignore[assignment]
        if not quabo_uids:
            raise RuntimeError("Missing quabo_uids.json")
        config.do_hk_dest(modules, quabo_uids, daq_config, network_config)

    if stage == 'start_redis':
        stage = 'maroc_config'
        logger.info('starting Redis daemons')
        util.start_redis_daemons()
        logger.info('starting transfer daemon')
        util.start_daemon([sys.executable, "-m", "control.transfer"], name="transfer_daemon")
    
    if stage == 'maroc_config':
        stage = 'mask_config'
        logger.info('configuring Marocs')
        quabo_uids = config_file.get_quabo_uids() # type: ignore[assignment]
        if not quabo_uids:
            raise RuntimeError("Missing quabo_uids.json")
        config.do_maroc_config(modules, quabo_uids, quabo_info, data_config, obs_config, daq_config, network_config, True)

    if stage == 'mask_config':
        stage = 'calibrate_ph'
        logger.info('configuring Masks')
        quabo_uids = config_file.get_quabo_uids() # type: ignore[assignment]
        if not quabo_uids:
            raise RuntimeError("Missing quabo_uids.json")
        config.do_mask_config(modules, data_config, network_config, quabo_uids, True)
    
    if stage == 'calibrate_ph':
        stage = 'open_shutters'
        logger.info('calibrating PH')
        quabo_uids = config_file.get_quabo_uids() # type: ignore[assignment]
        if not quabo_uids:
            raise RuntimeError("Missing quabo_uids.json")
        config.do_calibrate_ph(modules, quabo_uids, network_config)
        config.do_show_ph_baselines(quabo_uids)

    # TODO: we need more tests for do_shutter
    # if stage == 'open_shutters':
    #     print('opening shutters')
    #     config.do_shutter("open")


app = typer.Typer(help="Start an observing session.", no_args_is_help=False, context_settings={"help_option_names": ["-h", "--help"]})

@app.command()
def main(
    no_hv: bool = typer.Option(False, "--no-hv", help="Turn off HV when running `start.py`."),
    stage: str = typer.Option("poweron", help="The session will start from this stage: poweron, get_uids, reboot, hk_dest, start_redis, maroc_config, mask_config, calibrate_ph, show_ph_baselines.")
) -> None:
    """Initialize hardware, power, and calibration for an observing session."""
    # session start
    session_start(
        config_file.get_obs_config(),
        config_file.get_quabo_info(),
        config_file.get_data_config(),
        config_file.get_daq_config(),
        config_file.get_network_config(),
        no_hv,
        stage
    )


if __name__ == "__main__":
    app()
