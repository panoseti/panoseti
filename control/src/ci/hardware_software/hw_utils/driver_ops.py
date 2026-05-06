"""
Low-level driver operations — primitive entrypoints for the state machine.

MVP scope: only the primitives used by the boot_sequence and SafetyManager
are retained.  Archived primitives (soft_reset, tftp_reboot, hv_set, acq
start/stop) have been removed because cmd 0x04 does not re-enter the TFTP
bootloader — the only reliable reset is a WPS power cycle.
"""

from __future__ import annotations

import ipaddress
import logging

from ci.fixtures.topology_fixtures import ObservatoryTopology
from control.utils import util

logger = logging.getLogger(__name__)


def wps_power_on(**kwargs) -> None:
    """Power on all WPS outlets in obs_config. Transitions: UNPOWERED → POWERED."""
    _wps_toggle(on=True)


def wps_power_off(**kwargs) -> None:
    """Power off all WPS outlets in obs_config. Transitions: any → UNPOWERED."""
    _wps_toggle(on=False)


def _wps_toggle(on: bool) -> None:
    from control.power import quabo_power
    from control.utils import config_file
    obs = config_file.get_obs_config()
    extra = obs.model_extra or {}
    wps_entries = {k: v for k, v in extra.items() if k.startswith("wps") and isinstance(v, dict)}
    if not wps_entries:
        raise RuntimeError("No WPS entries found in obs_config (looking for keys starting with 'wps')")
    for name, wps_cfg in wps_entries.items():
        logger.info("_wps_toggle: %s on=%s", name, on)
        quabo_power(wps_cfg, on=on)


def boot_verify(quabo_ip: str | None = None, **kwargs) -> None:
    """
    Discover and cache hardware UIDs, then reboot all quabos via TFTP.
    Mirrors the session-start golden path: sleep → get_uids → do_reboot.
    Transitions: POWERED → BOOTED.
    """
    import os
    import time

    import control.config as config
    import control.get_uids as get_uids
    from control.utils import config_file

    obs_config = config_file.get_obs_config()
    network_config = config_file.get_network_config()
    modules = config_file.get_modules(obs_config)

    boot_wait = int(os.environ.get("HW_TEST_QUABO_BOOT_WAIT", 60))
    logger.info("boot_verify: waiting %ds for quabos to boot", boot_wait)
    time.sleep(boot_wait)

    get_uids.get_uids(obs_config, network_config)
    quabo_uids = config_file.get_quabo_uids()
    config.do_reboot(modules, quabo_uids, network_config)


def route_hk(**kwargs) -> None:
    """Configure HK packet destination. Transitions: BOOTED → HK_ROUTED."""
    import control.config as config
    from control.utils import config_file

    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    network_config = config_file.get_network_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()

    config.do_hk_dest(modules, quabo_uids, daq_config, network_config)


def configure_maroc(**kwargs) -> None:
    """Program MAROC registers. Transitions: HK_ROUTED → MAROC_CONFIGURED."""
    import control.config as config
    from control.utils import config_file

    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    data_config = config_file.get_data_config()
    network_config = config_file.get_network_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()
    quabo_info = config_file.get_quabo_info()

    config.do_maroc_config(modules, quabo_uids, quabo_info, data_config, obs_config, daq_config, network_config)


def configure_masks(**kwargs) -> None:
    """Write trigger masks. Transitions: MAROC_CONFIGURED → MASKS_CONFIGURED."""
    import control.config as config
    from control.utils import config_file

    obs_config = config_file.get_obs_config()
    data_config = config_file.get_data_config()
    network_config = config_file.get_network_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()

    config.do_mask_config(modules, data_config, network_config, quabo_uids)


def calibrate_ph(**kwargs) -> None:
    """Run PH baseline calibration. Transitions: MASKS_CONFIGURED → PH_CALIBRATED."""
    import control.config as config
    from control.utils import config_file

    obs_config = config_file.get_obs_config()
    network_config = config_file.get_network_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()

    config.do_calibrate_ph(modules, quabo_uids, network_config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_all_reachable(topo: ObservatoryTopology) -> list[str]:
    """Return error strings for each quabo that fails util.ping (command port).
    
    Performs parallel pings across the topology.
    """
    from concurrent.futures import ThreadPoolExecutor
    quabo_addrs = list(topo.quabo_ips())
    if not quabo_addrs:
        return []

    errors = []
    with ThreadPoolExecutor(max_workers=len(quabo_addrs)) as executor:
        futures = {
            executor.submit(util.ping, ipaddress.ip_address(a.real_ip), a.cmd_port): a 
            for a in quabo_addrs
        }
        
        for future in futures:
            a = futures[future]
            try:
                if not future.result():
                    errors.append(f"{a.ip} (loc={a.boardloc}, real={a.real_ip}:{a.cmd_port}) not reachable")
            except Exception as exc:
                errors.append(f"{a.ip} (loc={a.boardloc}) ping error: {exc}")
    return errors

