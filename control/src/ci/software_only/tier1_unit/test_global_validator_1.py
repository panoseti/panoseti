"""
test_global_validator_1.py

Unit tests for control/utils/global_validator.py.
Tests each _check_* rule of GlobalConfigValidator in isolation by constructing
minimal config dicts and verifying the ValidationReport outcome.
No hardware or network access required.
"""

from typing import Any, cast

import pytest

from control.utils.global_validator import GlobalConfigValidator
from control.utils.pydantic_config_models import (
    DaqConfig,
    DataConfig,
    FirmwareConfig,
    NetworkConfig,
    ObsConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ===========================================================================
# _check_science_guardrails
# WARN when run_type != 'eng*' and flash/stim params are present
# ===========================================================================

class TestScienceGuardrails:
    def test_sci_run_with_no_stim_passes(self) -> None:
        v = _make_validator(data={"run_type": "sci"})
        v._check_science_guardrails()
        assert not v.report.has_errors

    def test_eng_run_with_flash_passes(self) -> None:
        v = _make_validator(data={"run_type": "eng", "flash_params": {"rate": 3, "level": 15, "width": 7}})
        v._check_science_guardrails()
        assert not v.report.has_errors

    def test_sci_run_with_flash_warns(self) -> None:
        v = _make_validator(data={"run_type": "sci", "flash_params": {"rate": 3, "level": 15, "width": 7}})
        v._check_science_guardrails()
        # WARN is not an ERROR; has_errors stays False
        assert not v.report.has_errors
        statuses = [t["status"] for t in v.report.tests]
        assert "WARN" in statuses

    def test_sci_run_with_stim_warns(self) -> None:
        v = _make_validator(data={"run_type": "sci", "stim_params": {"rate": 1, "level": 128, "mask": [True]*4}})
        v._check_science_guardrails()
        assert not v.report.has_errors
        assert any(t["status"] == "WARN" for t in v.report.tests)

    def test_empty_data_config_passes(self) -> None:
        v = _make_validator(data={})
        v._check_science_guardrails()
        assert not v.report.has_errors


# ===========================================================================
# _check_geospatial_coherence
# ERROR when domes > 2 km apart; PASS when ≤ 2 km or only one dome
# ===========================================================================

class TestGeospatialCoherence:
    PALOMAR_LAT = 33.357
    PALOMAR_LON = -116.865

    def _nearby(self):
        """Two domes ~111 m apart (safe)."""
        return [
            {"name": "D0", "obslat": self.PALOMAR_LAT, "obslon": self.PALOMAR_LON},
            {"name": "D1", "obslat": self.PALOMAR_LAT + 0.001, "obslon": self.PALOMAR_LON},
        ]

    def _far_apart(self):
        """Two domes ~111 km apart (violates 2 km limit)."""
        return [
            {"name": "D0", "obslat": self.PALOMAR_LAT, "obslon": self.PALOMAR_LON},
            {"name": "D1", "obslat": self.PALOMAR_LAT + 1.0, "obslon": self.PALOMAR_LON},
        ]

    def test_single_dome_passes(self) -> None:
        v = _make_validator(obs={"name": "test", "domes": [{"name": "D0", "obslat": 33.0, "obslon": -116.0}]})
        v._check_geospatial_coherence()
        assert not v.report.has_errors

    def test_no_domes_passes(self) -> None:
        v = _make_validator(obs={"name": "test", "domes": []})
        v._check_geospatial_coherence()
        assert not v.report.has_errors

    def test_two_nearby_domes_pass(self) -> None:
        v = _make_validator(obs={"name": "test", "domes": self._nearby()})
        v._check_geospatial_coherence()
        assert not v.report.has_errors

    def test_two_far_domes_error(self) -> None:
        v = _make_validator(obs={"name": "test", "domes": self._far_apart()})
        v._check_geospatial_coherence()
        assert v.report.has_errors
        assert any("ERROR" in t["status"] for t in v.report.tests)

    def test_decimal_typo_triggers_error(self) -> None:
        """Simulates a common typo: Palomar vs. Palomar-far where decimal is off."""
        domes = [
            {"name": "D0", "obslat": 33.357, "obslon": -116.865},
            {"name": "D1", "obslat": 43.357, "obslon": -116.865},  # 10° off = ~1100 km
        ]
        v = _make_validator(obs={"name": "test", "domes": domes})
        v._check_geospatial_coherence()
        assert v.report.has_errors


# ===========================================================================
# _check_hardware_firmware
# ERROR when a module uses a hardware type not listed in firmware_config
# ===========================================================================

class TestHardwareFirmware:
    def test_matching_firmware_passes(self) -> None:
        obs = {"name": "test", "domes": [{"name": "d", "obslat": 0, "obslon": 0, "obsalt": 0, "modules": [{"mobo_serialno": "s", "quabo_version": "bga", "ip_addr": "192.168.3.200"}]}]}
        v = _make_validator(obs=obs, firmware={"bga": "fw_bga.bin"})
        v._check_hardware_firmware()
        assert not v.report.has_errors

    def test_missing_firmware_errors(self) -> None:
        obs = {"name": "test", "domes": [{"name": "d", "obslat": 0, "obslon": 0, "obsalt": 0, "modules": [{"mobo_serialno": "s", "quabo_version": "bga", "ip_addr": "192.168.3.200"}]}]}
        v = _make_validator(obs=obs, firmware={"qfp": "fw_qfp.bin"})  # bga is missing
        v._check_hardware_firmware()
        assert v.report.has_errors

    def test_multiple_hw_types_all_covered(self) -> None:
        obs = {"name": "test", "domes": [{"name": "d", "obslat": 0, "obslon": 0, "obsalt": 0, "modules": [
            {"mobo_serialno": "s1", "quabo_version": "bga", "ip_addr": "192.168.3.200"},
            {"mobo_serialno": "s2", "quabo_version": "qfp", "ip_addr": "192.168.3.204"},
        ]}]}
        fw = {"bga": "fw_bga.bin", "qfp": "fw_qfp.bin"}
        v = _make_validator(obs=obs, firmware=fw)
        v._check_hardware_firmware()
        assert not v.report.has_errors

    def test_per_quabo_version_list_checked(self) -> None:
        """quabo_version can be a list of per-quabo versions."""
        obs = {"name": "test", "domes": [{"name": "d", "obslat": 0, "obslon": 0, "obsalt": 0, "modules": [{"mobo_serialno": "s", "quabo_version": ["bga", "bga", "qfp", "bga"], "ip_addr": "192.168.3.200"}]}]}
        fw = {"bga": "fw_bga.bin"}  # qfp missing
        v = _make_validator(obs=obs, firmware=fw)
        v._check_hardware_firmware()
        assert v.report.has_errors

    def test_no_modules_passes(self) -> None:
        v = _make_validator(obs={"name": "test", "domes": [{"name": "d", "obslat": 0, "obslon": 0, "obsalt": 0, "modules": []}]}, firmware={"bga": "fw.bin"})
        v._check_hardware_firmware()
        assert not v.report.has_errors

    def test_empty_obs_passes(self) -> None:
        v = _make_validator(firmware={"bga": "fw.bin"})
        v._check_hardware_firmware()
        assert not v.report.has_errors

    def test_empty_firmware_dict_errors(self) -> None:
        obs = {"name": "test", "domes": [{"name": "d", "obslat": 0, "obslon": 0, "obsalt": 0, "modules": [{"mobo_serialno": "s", "quabo_version": "bga", "ip_addr": "192.168.3.200"}]}]}
        v = _make_validator(obs=obs, firmware={})
        v._check_hardware_firmware()
        assert v.report.has_errors


# ===========================================================================
# _check_overvoltage_consensus
# ERROR when obs_config and data_config have different detector_overvoltage
# ===========================================================================

class TestOvervoltageConsensus:
    def test_matching_voltages_passes(self) -> None:
        v = _make_validator(
            obs={"name": "test", "domes": [], "detector_overvoltage": 3},
            data={"run_type": "sci", "detector_overvoltage": 3},
        )
        v._check_overvoltage_consensus()
        assert not v.report.has_errors

    def test_mismatched_voltages_errors(self) -> None:
        v = _make_validator(
            obs={"name": "test", "domes": [], "detector_overvoltage": 3},
            data={"run_type": "sci", "detector_overvoltage": 2},
        )
        v._check_overvoltage_consensus()
        assert v.report.has_errors

    def test_obs_overvoltage_none_passes(self) -> None:
        """If obs doesn't specify overvoltage, no constraint."""
        v = _make_validator(
            obs={"name": "test", "domes": []},
            data={"run_type": "sci", "detector_overvoltage": 3},
        )
        v._check_overvoltage_consensus()
        assert not v.report.has_errors

    def test_data_overvoltage_none_passes(self) -> None:
        v = _make_validator(
            obs={"name": "test", "domes": [], "detector_overvoltage": 3},
            data={"run_type": "sci"},
        )
        v._check_overvoltage_consensus()
        assert not v.report.has_errors

    def test_both_none_passes(self) -> None:
        v = _make_validator()
        v._check_overvoltage_consensus()
        assert not v.report.has_errors

    @pytest.mark.parametrize("obs_v, data_v, should_error", [
        (2, 2, False),
        (3, 3, False),
        (2, 3, True),
        (3, 2, True),
    ])
    def test_voltage_pairs(self, obs_v, data_v, should_error) -> None:
        v = _make_validator(
            obs={"name": "test", "domes": [], "detector_overvoltage": obs_v},
            data={"run_type": "sci", "detector_overvoltage": data_v},
        )
        v._check_overvoltage_consensus()
        assert v.report.has_errors == should_error
