"""
Low-level driver operations — primitive entrypoints.
Each function is the body of a state-machine primitive and is called by
HardwareStateMachine.execute() via importlib.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def configure_maroc(quabo_ip: str | None = None, **kwargs) -> None:
    """
    Program MAROC registers on all (or one) quabo(s).

    Loads the config from quabo_config.txt and sends the 829-byte serial command.
    Transitions: BOOTED → MAROC_LOADED.
    """
    from control.driver.quabo_driver import QUABO
    from control.utils import config_file

    if quabo_ip:
        quabos = [QUABO(quabo_ip)]
    else:
        obs = config_file.get_obs_config()
        quabos = _all_quabos(obs)

    for q in quabos:
        logger.info("configure_maroc: sending MAROC params to %s", q.ip_addr)
        q.send_maroc_params()


def configure_acq(quabo_ip: str | None = None, **kwargs) -> None:
    """
    Write ACQ params, trigger masks, GOE masks, and destination IPs to quabo(s).

    Transitions: MAROC_LOADED → ACQ_CONFIGURED.
    """
    from control.driver.quabo_driver import DAQ_PARAMS, QUABO
    from control.utils import config_file

    obs = config_file.get_obs_config()
    data = config_file.get_data_config()

    quabos = [QUABO(quabo_ip)] if quabo_ip else _all_quabos(obs)

    for q in quabos:
        logger.info("configure_acq: setting DAQ params on %s", q.ip_addr)
        params = DAQ_PARAMS(data)
        q.send_daq_params(params)


def hv_set_from_config(quabo_ip: str | None = None, **kwargs) -> None:
    """
    Ramp HV to the setpoint specified in obs_config detector_overvoltage.

    Guards (shutter_must_be_closed, light_sensor_dark) run before this call.
    Transitions: ACQ_CONFIGURED → HV_ON.
    """
    from control.driver.quabo_driver import QUABO
    from control.utils import config_file

    obs = config_file.get_obs_config()
    overvoltage = obs.get("detector_overvoltage", 0)

    quabos = [QUABO(quabo_ip)] if quabo_ip else _all_quabos(obs)

    for q in quabos:
        logger.info("hv_set_from_config: setting HV overvoltage=%.2f on %s", overvoltage, q.ip_addr)
        q.hv_set_from_config(overvoltage)


def hv_zero(quabo_ip: str | None = None, **kwargs) -> None:
    """
    Zero the HV on all (or one) quabo(s).

    Transitions: any → ACQ_CONFIGURED.
    """
    from control.driver.quabo_driver import QUABO
    from control.utils import config_file

    if quabo_ip:
        quabos = [QUABO(quabo_ip)]
    else:
        obs = config_file.get_obs_config()
        quabos = _all_quabos(obs)

    for q in quabos:
        logger.info("hv_zero: zeroing HV on %s", q.ip_addr)
        q.hv_set([0, 0, 0, 0])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_quabos(obs_config: dict) -> list:
    """Instantiate QUABO objects for every quabo in the observatory layout."""
    from control.driver.quabo_driver import QUABO
    result = []
    for dome in obs_config.get("domes", []):
        for module in dome.get("modules", []):
            base_ip: str = module["ip"]
            parts = base_ip.split(".")
            for q in range(4):
                ip = f"{parts[0]}.{parts[1]}.{parts[2]}.{int(parts[3]) + q}"
                result.append(QUABO(ip))
    return result
