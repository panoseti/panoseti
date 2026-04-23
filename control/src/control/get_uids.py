#! /usr/bin/env python3

# scan possible quabo IP addrs.
# If they respond to ping, get their UID
# write these to quabo_uids.json
#
# --exclude N    exclude quabo N (0..3) from each module

import os
import struct

import typer
from panoseti_grpc.telemetry.logger import get_logger

from control.driver.quabo_tftp import tftpw
from control.utils import config_file, util
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import (
    NetworkConfig,
    ObsConfig,
    QuaboUidDome,
    QuaboUidEntry,
    QuaboUidModule,
    QuaboUids,
)

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


def get_uids(obs_config: ObsConfig, network_config: NetworkConfig, exclude: list[int] | None = None) -> None:
    """Scan the observatory for Quabos and cache their unique hardware IDs."""
    if exclude is None:
        exclude = []

    dome_uids: list[QuaboUidDome] = []
    
    for d in obs_config.domes:
        module_uids: list[QuaboUidModule] = []
        for m in d.modules:
            m_ip = str(m.ip_addr)
            quabo_entries: list[QuaboUidEntry] = []
            
            for i in range(4):
                if i not in exclude:
                    ip_ports = util.get_quabo_ip_port(m_ip, i, network_config)
                    real_ip = ip_ports['ip_addr']
                    port = ip_ports['reboot_port']
                    
                    logger.info(f"get uid for {m_ip}:{i} (real_ip: {real_ip})")
                    uid = get_uid(real_ip, port)
                    quabo_entries.append(QuaboUidEntry(uid=uid))
                else:
                    quabo_entries.append(QuaboUidEntry(uid=''))
            
            module_uids.append(QuaboUidModule(
                ip_addr=m.ip_addr,
                quabos=quabo_entries,
                id=m.id,
                daq_node=m.daq_node
            ))
        dome_uids.append(QuaboUidDome(modules=module_uids, num=d.num))

    quabo_uids = QuaboUids(domes=dome_uids)
    
    quabo_uids_path = PanoPaths.tmp_dir() / config_file.quabo_uids_filename
    with open(quabo_uids_path, "w", encoding="utf-8") as f:
        f.write(quabo_uids.model_dump_json(indent=4))


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
