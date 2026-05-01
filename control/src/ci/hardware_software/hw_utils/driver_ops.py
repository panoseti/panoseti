"""
Low-level driver operations — primitive entrypoints.
Each function is the body of a state-machine primitive and is called by
HardwareStateMachine.execute() via importlib.
"""

from __future__ import annotations

import logging

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
    Reboot all quabos via TFTP (Q0→Q1→Q2→Q3 within each module, parallel
    across modules), then discover and cache hardware UIDs.
    Mirrors the session-start golden path: do_reboot → get_uids.
    Transitions: POWERED → BOOTED.
    """
    import control.config as config
    from control.utils import config_file

    obs_config = config_file.get_obs_config()
    network_config = config_file.get_network_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()

    config.do_reboot(modules, quabo_uids, network_config)


def route_hk(quabo_ip: str | None = None, **kwargs) -> None:
    """
    Configure the housekeeping packet destination for Quabos.
    Transitions: BOOTED → HK_ROUTED.
    """
    import control.config as config
    from control.utils import config_file

    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    network_config = config_file.get_network_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()

    logger.info("route_hk: Setting HK destination to this node")
    config.do_hk_dest(modules, quabo_uids, daq_config, network_config)


def configure_maroc(quabo_ip: str | None = None, **kwargs) -> None:
    """
    Program MAROC registers on all quabos via config.do_maroc_config,
    which applies per-quabo calibration data from quabo_info.
    Transitions: HK_ROUTED → MAROC_CONFIGURED.
    """
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


def configure_masks(quabo_ip: str | None = None, **kwargs) -> None:
    """
    Write trigger masks and GOE masks to all quabos via config.do_mask_config.
    Transitions: MAROC_CONFIGURED → MASKS_CONFIGURED.
    """
    import control.config as config
    from control.utils import config_file

    obs_config = config_file.get_obs_config()
    data_config = config_file.get_data_config()
    network_config = config_file.get_network_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()

    config.do_mask_config(modules, data_config, network_config, quabo_uids)


def calibrate_ph(quabo_ip: str | None = None, **kwargs) -> None:
    """
    Run the PH baseline calibration.
    Transitions: ACQ_CONFIGURED → PH_CALIBRATED.
    """
    import control.config as config
    from control.utils import config_file

    obs_config = config_file.get_obs_config()
    network_config = config_file.get_network_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()

    logger.info("calibrate_ph: Running PH baseline calibration")
    config.do_calibrate_ph(modules, quabo_uids, network_config)


def hv_set_from_config(quabo_ip: str | None = None, **kwargs) -> None:
    """
    Ramp HV to the setpoint specified in obs_config detector_overvoltage.

    Guards (shutter_must_be_closed, light_sensor_dark) run before this call.
    Transitions: PH_CALIBRATED → HV_ON.
    """
    from control.utils import config_file

    obs = config_file.get_obs_config()
    overvoltage = obs.detector_overvoltage or 0

    quabos = _all_quabos(quabo_ip)

    for q in quabos:
        logger.info("hv_set_from_config: setting HV overvoltage=%.2f on %s:%d", overvoltage, q.ip_addr, q.port)
        q.hv_set_from_config(overvoltage)


def hv_zero(quabo_ip: str | None = None, **kwargs) -> None:
    """
    Zero the HV on all (or one) quabo(s).

    Transitions: any → ACQ_CONFIGURED.
    """

    quabos = _all_quabos(quabo_ip)

    for q in quabos:
        logger.info("hv_zero: zeroing HV on %s:%d", q.ip_addr, q.port)
        q.hv_set([0, 0, 0, 0])


def soft_reset_all(quabo_ip: str | None = None, **kwargs) -> None:
    """
    Send cmd 0x04 (logic reset, no firmware reload) to all quabos.
    Transitions: any configured state → BOOTED.
    """
    quabos = _all_quabos(quabo_ip)
    for q in quabos:
        logger.info("soft_reset_all: resetting %s:%d", q.ip_addr, q.port)
        q.reset()


def tftp_reboot_all(quabo_ip: str | None = None, **kwargs) -> None:
    """
    TFTP firmware reload + reboot on all quabos (same as boot_verify but
    used when already above POWERED, e.g. from a configured state).
    Delegates to config.do_reboot for correct Q0→Q1→Q2→Q3 ordering.
    Transitions: any state above POWERED → BOOTED.
    """
    import control.config as config
    from control.utils import config_file

    obs_config = config_file.get_obs_config()
    network_config = config_file.get_network_config()
    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()

    config.do_reboot(modules, quabo_uids, network_config)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_quabos(quabo_ip: str | None = None) -> list:
    """Instantiate QUABO objects for every quabo in the observatory layout, respecting port forwarding."""
    from ci.hardware_software.hw_utils.topology import HwTopology
    from control.driver.quabo_driver import QUABO
    
    topo = HwTopology()
    addrs = topo.quabo_ips()
    
    if quabo_ip:
        addrs = [a for a in addrs if a.ip == quabo_ip or a.real_ip == quabo_ip]
        
    return [QUABO(a.real_ip, port=a.cmd_port) for a in addrs]
