#! /usr/bin/env python3

# scan possible quabo IP addrs.
# If they respond to ping, get their UID
# write these to quabo_uids.json
#
# --exclude N    exclude quabo N (0..3) from each module

import json
import os
import struct
from typing import Any

import typer
from panoseti_grpc.telemetry.logger import get_logger

from control.driver.quabo_tftp import tftpw
from control.utils import config_file, util
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import NetworkConfigValidator, ObsConfigValidator

log_dir = PanoPaths.logs_dir()
log_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger("PSETI.GetUIDs", log_dir=str(log_dir), grpc_enabled=True)

# return quabo UID as hex string
#
def get_uid(ip_addr: str, port: int) -> str:
    """Retrieve the unique hardware ID (UID) from a Quabo via TFTP.

    Args:
        ip_addr: The IP address of the Quabo.
        port: The TFTP port to use for the request.

    Returns:
        The hardware UID as a hex string, or an empty string if retrieval fails.
    """
    x = tftpw(ip_addr, port)
    try:
        x.get_flashuid()
        with open('flashuid', 'rb') as f:
            i = struct.unpack('q', f.read(8))
            return f"{i[0]:x}"
    except Exception:
        return ""


def get_uids(obs_config: ObsConfigValidator | dict[str, Any], network_config: NetworkConfigValidator | dict[str, Any], exclude: list[int] | None = None) -> None:
    """Scan the observatory for Quabos and cache their unique hardware IDs.
    
    Iterates through all modules defined in the configuration, attempts to 
    contact each Quabo, and writes the resulting UID map to quabo_uids.json.

    Args:
        obs_config: Physical observatory configuration model or dict.
        network_config: Network routing configuration model or dict.
        exclude: Optional list of Quabo indices (0-3) to skip in every module.
    """
    if isinstance(obs_config, dict):
        obs_config = ObsConfigValidator(**obs_config)
    if isinstance(network_config, dict):
        network_config = NetworkConfigValidator(**network_config)

    if exclude is None:
        exclude = []

    quabo_uids: dict[str, Any] = {}
    quabo_uids['domes'] = []
    for d in obs_config.domes:
        dome: dict[str, Any] = {}
        dome['modules'] = []
        for m in d.modules:
            module: dict[str, Any] = {}
            m_ip = str(m.ip_addr)
            module['ip_addr'] = m_ip
            module['quabos'] = []
            for i in range(4):
                quabo: dict[str, Any] = {}
                if i not in exclude:
                    ip_ports = util.get_quabo_ip_port(m_ip, i, network_config)
                    ip_addr = config_file.quabo_ip_addr(m_ip, i)
                    real_ip = ip_ports['ip_addr']
                    port = ip_ports['reboot_port']
                    logger.info(f"get uid {ip_addr}")
                    # TODO: we need to ping the board before get_uid
                    uid = get_uid(real_ip, port)
                    if len(uid):
                        logger.info(f"{ip_addr} has UID {uid}")
                    else:
                        logger.info(f"{ip_addr} is offline")
                    quabo['uid'] = uid
                else:
                    quabo['uid'] = ''
                module['quabos'].append(quabo)

            dome['modules'].append(module)
        quabo_uids['domes'].append(dome)
    quabo_uids_path = PanoPaths.tmp_dir() / config_file.quabo_uids_filename
    with open(quabo_uids_path, "w", encoding="utf-8") as f:
        json.dump(quabo_uids, f, ensure_ascii=False, indent=4)


app = typer.Typer(help="Scan and cache Quabo hardware UIDs.", no_args_is_help=False)

@app.command()
def main(
    exclude: list[int] = typer.Option(None, "--exclude", "-e", help="Quabo indices (0-3) to skip in every module.")
):
    """
    Scan possible quabo IP addrs.
    If they respond to ping, get their UID
    write these to quabo_uids.json
    """
    obs_config = config_file.get_obs_config()
    network_config = config_file.get_network_config()
    get_uids(obs_config, network_config, exclude)
    if os.path.exists('flashuid'):
        os.remove('flashuid')

if __name__ == "__main__":
    app()
