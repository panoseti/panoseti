#! /usr/bin/env python3

# power.py [wps1] [on|off]
#
# on/off: turn a web power switch (WPS) on or off
# neither: query quabo power
# wps1 (or other name):
#   use the "wps1" element from obs_config.json
#   default is "wps"

import os
import sys
from datetime import UTC, datetime
from typing import Any

from control.utils import config_file
from control.utils.pydantic_config_models import ObsConfigValidator, WpsConfig

LOG_DIR_ROOT = "/mnt/data11"

# ---------- logging helper (UTC) ----------
def log_print(msg: str) -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    ts = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    yyyymmdd = now.strftime("%Y%m%d")

    log_dir = f"{LOG_DIR_ROOT}/data/palomar/L0/{yyyymmdd}/obslogs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"datarec_{yyyymmdd}.log")

    line = f"{ts} {msg}"
    print(line)

    # Prepend the new line at the beginning of the log file
    old = ""
    if os.path.exists(log_file):
        with open(log_file) as f:
            old = f.read()

    tmp_file = log_file + ".tmp"
    with open(tmp_file, "w") as f:
        f.write(line + "\n")
        if old:
            f.write(old)

    os.replace(tmp_file, log_file)
# -----------------------------------------



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
            log_print(f"{name}: power is on")
        else:
            log_print(f"{name}: power is off")
    elif op == 'on':
        quabo_power(wps, True)
        log_print(f"{name}: turned power on")
    elif op == 'off':
        quabo_power(wps, False)
        log_print(f"{name}: turned power off")


def do_all(obs_config: ObsConfigValidator, op: str) -> None:
    """Perform a power operation on all WPS units defined in the configuration.

    Args:
        obs_config: Validated observatory configuration.
        op: The operation to perform.
    """
    extra = obs_config.model_extra or {}
    for key in [k for k in extra if 'wps' in k.lower()]:
        do_wps(key, obs_config, op)


if __name__ == "__main__":
    op = 'query'
    wps_name = 'wps'
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == 'on':
            op = 'on'
        elif sys.argv[i] == 'off':
            op = 'off'
        else:
            raise Exception('usage: power.py [on|off]')
        i += 1

    c = config_file.get_obs_config()
    do_all(c, op)
