"""Pytest fixtures for the tier 1 unit tests"""

import os
import struct
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from ci.fixtures.packet_capture import FakeSocket
from control.driver.quabo_driver import QUABO
from control.utils.global_validator import GlobalConfigValidator
from control.utils.pydantic_config_models import (
    DaqConfig,
    DataConfig,
    FirmwareConfig,
    NetworkConfig,
    ObsConfig,
)

# Quabo driver fixtures

@pytest.fixture
def quabo_and_sock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mock_network: Any) -> tuple[QUABO, Any]:
    """Yield (quabo, fake_sock).  All socket I/O is captured in fake_sock."""
    from control import driver
    from control.driver.quabo_driver import QUABO_CONFIG_FILE
    
    # mock_network already monkeypatches socket.socket
    fake_sock = mock_network
    monkeypatch.setattr("socket.gethostbyname", lambda x: x)

    # Suppress log-file creation — tests don't need a real log file
    monkeypatch.setattr("control.driver.quabo_driver.get_logger", lambda *a, **kw: MagicMock())

    tmp_quabo_cfg_path = tmp_path / QUABO_CONFIG_FILE
    # Copy the real quabo_config.txt into tmp so send_maroc_params_file() works
    real_cfg = Path(driver.__file__) / QUABO_CONFIG_FILE
    if os.path.exists(real_cfg):
        with open(real_cfg) as f:
            tmp_quabo_cfg_path.write_text(f.read())
    else:
        tmp_quabo_cfg_path.write_text("* minimal stub\n")

    q = QUABO(
        "192.168.3.100",
        config_file_path=str(tmp_quabo_cfg_path),
    )
    return q, fake_sock

def _make_hk_packet(
    boardloc: int = 0,
    hvmon0: int = 0,
    temp1_raw: int = 0,
    temp2_raw: int = 0,
    uid: tuple = (0, 0, 0, 0),
    status_pcbrev: int = 0,
) -> bytes:
    """Build a synthetic 64-byte HK packet with specified field values.

    All uint16 LE pairs start at byte 2.  Array index mapping:
      array[ 0] bytes[ 2: 4] — BOARDLOC (unsigned)
      array[ 1] bytes[ 4: 6] — HVMON0   (unsigned)
      array[17] bytes[36:38] — TEMP1    (signed)
      array[18] bytes[38:40] — TEMP2    (unsigned)
      array[21] bytes[44:46] — UID[0]   (unsigned)
      array[22] bytes[46:48] — UID[1]
      array[23] bytes[48:50] — UID[2]
      array[24] bytes[50:52] — UID[3]
      array[25] bytes[52:54] — status (low byte) | PCBrev (high byte)
    """
    pkt = bytearray(64)
    pkt[0] = 0x20   # PANOSETI HK packet magic
    pkt[1] = 0x00   # not a startup packet
    struct.pack_into("<H", pkt, 2, boardloc & 0xFFFF)
    struct.pack_into("<H", pkt, 4, hvmon0 & 0xFFFF)
    struct.pack_into("<h", pkt, 36, temp1_raw)          # signed int16
    struct.pack_into("<H", pkt, 38, temp2_raw & 0xFFFF)
    for i, u in enumerate(uid):
        struct.pack_into("<H", pkt, 44 + i * 2, u & 0xFFFF)
    struct.pack_into("<H", pkt, 52, status_pcbrev & 0xFFFF)
    return bytes(pkt)


def _parse_hk_field(packet: bytes, array_index: int, signed: bool = False) -> int:
    """Read a 16-bit LE field at byte offset 2 + array_index * 2."""
    offset = 2 + array_index * 2
    fmt = "<h" if signed else "<H"
    return struct.unpack_from(fmt, packet, offset)[0]


def _minimal_maroc_config() -> dict:
    """All-zero MAROC config dict with every required key present."""
    scalar_keys = [
        "OTABG_ON", "DAC_ON", "SMALL_DAC", "ENB_OUT_ADC", "INV_START_GRAY",
        "RAMP8B", "RAMP10B", "CMD_CK_MUX", "D1_D2", "INV_DISCR_ADC",
        "POLAR_DISCRI", "ENB3ST", "VAL_DC_FSB2", "SW_FSB2_50F", "SW_FSB2_100F",
        "SW_FSB2_100K", "SW_FSB2_50K", "VALID_DC_FS", "CMD_FSB_FSU",
        "SW_FSB1_50F", "SW_FSB1_100F", "SW_FSB1_100K", "SW_FSB1_50k",
        "SW_FSU_100K", "SW_FSU_50K", "SW_FSU_25K", "SW_FSU_40F", "SW_FSU_20F",
        "H1H2_CHOICE", "EN_ADC", "SW_SS_1200F", "SW_SS_600F", "SW_SS_300F",
        "ON_OFF_SS", "SWB_BUF_2P", "SWB_BUF_1P", "SWB_BUF_500F", "SWB_BUF_250F",
        "CMD_FSB", "CMD_SS", "CMD_FSU",
    ]
    config = {k: "0,0,0,0" for k in scalar_keys}
    config["DAC2"] = "0,0,0,0"
    config["DAC1"] = "0,0,0,0"
    for i in range(64):
        config[f"GAIN{i}"] = "0,0,0,0"
        config[f"CTEST_{i}"] = "0,0,0,0"
        config[f"MASKOR1_{i}"] = "0,0,0,0"
        config[f"MASKOR2_{i}"] = "0,0,0,0"
    return config



# Global Validator Fixtures

def _make_validator(
    obs: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    daq: dict[str, Any] | None = None,
    net: dict[str, Any] | None = None,
    firmware: dict[str, Any] | None = None
) -> GlobalConfigValidator:
    """Build a GlobalConfigValidator with sensible defaults, overrideable per-test."""
    # Build minimal valid dictionaries to satisfy model requirements
    obs_dict: dict[str, Any] = {"name": "test", "domes": []}
    if obs:
        obs_dict.update(obs)
        # Ensure domes have required fields if provided
        domes = cast(list[dict[str, Any]], obs_dict.get("domes", []))
        for dome in domes:
            if "obsalt" not in dome:
                dome["obsalt"] = 0.0
            if "modules" not in dome:
                dome["modules"] = []

    data_dict: dict[str, Any] = {"run_type": "sci"}
    if data:
        data_dict.update(data)
        # Ensure image mode has pe_threshold if provided
        if data_dict.get("image"):
            image_conf = cast(dict[str, Any], data_dict["image"])
            if "pe_threshold" not in image_conf:
                image_conf["pe_threshold"] = 1.0

    daq_dict: dict[str, Any] = {"head_node_data_dir": "/data", "head_node_ip_addr": "10.0.0.1", "daq_nodes": []}
    if daq:
        daq_dict.update(daq)
    
    net_dict: dict[str, Any] = {"modules": [], "daq_nodes": []}
    if net:
        net_dict.update(net)
    
    fw_dict: dict[str, Any] = firmware or {}

    return GlobalConfigValidator({
        "obs":      ObsConfig(**obs_dict),
        "data":     DataConfig(**data_dict),
        "daq":      DaqConfig(**daq_dict),
        "network":  NetworkConfig(**net_dict),
        "firmware": FirmwareConfig(**fw_dict),
    })


def _run_check(validator: GlobalConfigValidator, method_name: str) -> tuple[bool, Any]:
    """Call a single _check_* method and return (passed, report)."""
    getattr(validator, method_name)()
    return not validator.report.has_errors, validator.report


def _check_passes(validator: GlobalConfigValidator, method_name: str) -> bool:
    passed, _ = _run_check(validator, method_name)
    return passed


def _check_fails(validator: GlobalConfigValidator, method_name: str) -> bool:
    passed, _ = _run_check(validator, method_name)
    return not passed