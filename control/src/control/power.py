#! /usr/bin/env python3

# power.py [wps1] [on|off]
#
# on/off: turn a web power switch (WPS) on or off
# neither: query quabo power
# wps1 (or other name):
#   use the "wps1" element from obs_config.json
#   default is "wps"

import os
from typing import Any

import typer
from panoseti_grpc.telemetry.logger import get_logger

from control.utils import config_file
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import ObsConfigValidator, WpsConfig

log_dir = PanoPaths.logs_dir()
log_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger("PSETI.Power", log_dir=str(log_dir), grpc_enabled=True)


# turn power on or off
#
def quabo_power(wps: WpsConfig | dict[str, Any], on: bool) -> None:
    """Turn Quabo power on or off via a Web Power Switch.

    Args:
        wps: Configuration for the WPS unit (url, quabo_socket).
        on: True to turn power on, False to turn power off.
    """
    if isinstance(wps, WpsConfig):
        url = wps.url
        socket = wps.quabo_socket
    else:
        url = wps['url']
        socket = wps['quabo_socket']
    
    value = 'ON' if on else 'OFF'
    cmd = f'curl -s {url}/outlet?{socket}={value} > /dev/null'
    ret = os.system(cmd)
    if ret:
        raise Exception(f'{cmd} returned {ret}')


# return True if power is on
#
def quabo_power_query(wps: WpsConfig | dict[str, Any]) -> str | None:
    """Query the power state of a Quabo socket.

    Args:
        wps: Configuration for the WPS unit.

    Returns:
        The state string from the WPS response if successful, otherwise None.
    """
    if isinstance(wps, WpsConfig):
        url = wps.url
        socket = wps.quabo_socket
    else:
        url = wps['url']
        socket = wps['quabo_socket']
    
    cmd = f'curl -s {url}/status'
    out = os.popen(cmd).read()
    off = out.find('state">')
    off += len('state">')
    y = out[off:off+2]
    status = int(y, 16)
    if status & (1 << (socket - 1)):
        return 'true'
    return None


def do_wps(name: str, obs_config: ObsConfigValidator, op: str) -> None:
    """Perform a power operation (on/off/query) on a named WPS unit.

    Args:
        name: The key name of the WPS unit in the configuration.
        obs_config: Validated observatory configuration.
        op: The operation to perform ('on', 'off', or 'query').
    """
    extra = obs_config.model_extra or {}
    if name not in extra:
        print(f"Error: {name} not found in obs_config.")
        return
    
    # extra[name] might be a dict if it was just loaded
    wps_data = extra[name]
    wps = WpsConfig(**wps_data) if isinstance(wps_data, dict) else wps_data

    if op == 'query':
        if quabo_power_query(wps):
            logger.info(f"{name}: power is on")
        else:
            logger.info(f"{name}: power is off")
    elif op == 'on':
        quabo_power(wps, True)
        logger.info(f"{name}: turned power on")
    elif op == 'off':
        quabo_power(wps, False)
        logger.info(f"{name}: turned power off")


def do_all(obs_config: ObsConfigValidator, op: str) -> None:
    """Perform a power operation on all WPS units defined in the configuration.

    Args:
        obs_config: Validated observatory configuration.
        op: The operation to perform.
    """
    extra = obs_config.model_extra or {}
    for key in [k for k in extra if 'wps' in k.lower()]:
        do_wps(key, obs_config, op)




app = typer.Typer(help="Control Quabo power via Web Power Switches (WPS).", no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]})

@app.command()
def on():
    """Turn all Quabo power switches ON."""
    c = config_file.get_obs_config()
    do_all(c, 'on')

@app.command()
def off():
    """Turn all Quabo power switches OFF."""
    c = config_file.get_obs_config()
    do_all(c, 'off')

@app.command()
def status():
    """Query the power state of all Quabo switches."""
    c = config_file.get_obs_config()
    do_all(c, 'query')


if __name__ == "__main__":
    app()
