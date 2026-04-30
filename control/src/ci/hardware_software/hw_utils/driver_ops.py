"""
Low-level driver operations — primitive entrypoints.
Each function is the body of a state-machine primitive and is called by
HardwareStateMachine.execute() via importlib.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


def boot_verify(quabo_ip: str | None = None, **kwargs) -> None:
    """
    Poll each quabo's cmd_port until UDP echo responds.
    Transitions: POWERED → BOOTED.
    """
    from ci.hardware_software.hw_utils.topology import HwTopology
    
    topo = HwTopology()
    addrs = topo.quabo_ips()
    
    if quabo_ip:
        # Filter to just this raw IP (or it might be passed as real_ip, 
        # but usually state machine works on the topology's identity)
        addrs = [a for a in addrs if a.ip == quabo_ip or a.real_ip == quabo_ip]

    if not addrs:
        logger.warning("boot_verify: no quabos found to verify")
        return

    timeout_s = kwargs.get("budget_s", {}).get("max", 300)
    start_time = time.time()
    reachable = set()
    
    logger.info("boot_verify: polling %d quabos for UDP reachability (timeout=%ds)", len(addrs), timeout_s)
    
    # Optional: trigger TFTP reboot for each quabo first if reboot_port is defined
    from control.driver.quabo_tftp import tftpw
    for a in addrs:
        if a.reboot_port != 69:  # Non-default or explicitly forwarded port
            logger.info("boot_verify: triggering TFTP reboot for %s via port %d", a.real_ip, a.reboot_port)
            try:
                t = tftpw(a.real_ip, port=a.reboot_port)
                t.reboot()
            except Exception as exc:
                logger.warning("boot_verify: TFTP reboot trigger failed for %s: %s", a.real_ip, exc)

    while len(reachable) < len(addrs) and (time.time() - start_time) < timeout_s:
        for a in addrs:
            if a.boardloc in reachable:
                continue
            
            from control.utils import util
            if util.ping(a.real_ip, a.cmd_port):
                logger.info("boot_verify: quabo at %s (loc=%d) is REACHABLE", a.real_ip, a.boardloc)
                reachable.add(a.boardloc)
        
        if len(reachable) < len(addrs):
            time.sleep(2)

    if len(reachable) < len(addrs):
        missing = [a.ip for a in addrs if a.boardloc not in reachable]
        raise RuntimeError(f"boot_verify: timeout reached. Unreachable quabos: {missing}")

    logger.info("boot_verify: all quabos are BOOTED and reachable")


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
    Program MAROC registers on all (or one) quabo(s).

    Loads the config from quabo_config.txt and sends the 829-byte serial command.
    Transitions: HK_ROUTED → MAROC_LOADED.
    """
    quabos = _all_quabos(quabo_ip)

    for q in quabos:
        logger.info("configure_maroc: sending MAROC params to %s:%d", q.ip_addr, q.port)
        q.send_maroc_params()


def configure_acq(quabo_ip: str | None = None, **kwargs) -> None:
    """
    Write ACQ params, trigger masks, GOE masks, and destination IPs to quabo(s).

    Transitions: MAROC_LOADED → ACQ_CONFIGURED.
    """
    from control.driver.quabo_driver import DAQ_PARAMS
    from control.utils import config_file

    data = config_file.get_data_config()

    quabos = _all_quabos(quabo_ip)

    for q in quabos:
        logger.info("configure_acq: setting DAQ params on %s:%d", q.ip_addr, q.port)
        params = DAQ_PARAMS(data)
        q.send_daq_params(params)


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
    overvoltage = obs.get("detector_overvoltage", 0)

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
