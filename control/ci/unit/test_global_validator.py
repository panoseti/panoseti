"""
test_global_validator.py

Unit tests for control/utils/global_validator.py.
Tests each _check_* rule of GlobalConfigValidator in isolation by constructing
minimal config dicts and verifying the ValidationReport outcome.
No hardware or network access required.
"""

from typing import Any, cast
from unittest.mock import patch

import pytest

from control.utils.global_validator import GlobalConfigValidator
from control.utils.pydantic_config_models import (
    DaqConfigValidator,
    DataConfigValidator,
    FirmwareConfigValidator,
    NetworkConfigValidator,
    ObsConfigValidator,
    QuaboUidsValidator,
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
        "obs":      ObsConfigValidator(**obs_dict),
        "data":     DataConfigValidator(**data_dict),
        "daq":      DaqConfigValidator(**daq_dict),
        "network":  NetworkConfigValidator(**net_dict),
        "firmware": FirmwareConfigValidator(**fw_dict),
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


# ===========================================================================
# _check_port_collisions
# ERROR when two modules on the same gateway use overlapping forwarded ports
# ===========================================================================

class TestPortCollisions:
    def test_no_network_config_passes(self) -> None:
        v = _make_validator(net={"modules": [], "daq_nodes": []})
        v._check_port_collisions()
        assert not v.report.has_errors

    def test_no_active_forwarding_passes(self) -> None:
        net = {"modules": [
            {"ip_addr": "192.168.3.200", "port_forwarding": {"status": False, "gw_ip": "1.2.3.4",
                                                              "cmd_port": [60000, 60001, 60002, 60003]}}
        ], "daq_nodes": []}
        v = _make_validator(net=net)
        v._check_port_collisions()
        assert not v.report.has_errors

    def test_non_overlapping_ports_on_same_gateway_pass(self) -> None:
        net = {"modules": [
            {"ip_addr": "192.168.3.200",
             "port_forwarding": {"status": True, "gw_ip": "1.2.3.4",
                                 "cmd_port": [60000, 60001, 60002, 60003]}},
            {"ip_addr": "192.168.3.204",
             "port_forwarding": {"status": True, "gw_ip": "1.2.3.4",
                                 "cmd_port": [61000, 61001, 61002, 61003]}},
        ], "daq_nodes": []}
        v = _make_validator(net=net)
        v._check_port_collisions()
        assert not v.report.has_errors

    def test_overlapping_ports_on_same_gateway_errors(self) -> None:
        net = {"modules": [
            {"ip_addr": "192.168.3.200",
             "port_forwarding": {"status": True, "gw_ip": "1.2.3.4",
                                 "cmd_port": [60000, 60001, 60002, 60003]}},
            {"ip_addr": "192.168.3.204",
             "port_forwarding": {"status": True, "gw_ip": "1.2.3.4",
                                 "cmd_port": [60000, 61001, 61002, 61003]}},  # 60000 collision!
        ], "daq_nodes": []}
        v = _make_validator(net=net)
        v._check_port_collisions()
        assert v.report.has_errors

    def test_same_ports_on_different_gateways_pass(self) -> None:
        net = {"modules": [
            {"ip_addr": "192.168.3.200",
             "port_forwarding": {"status": True, "gw_ip": "1.2.3.4",
                                 "cmd_port": [60000, 60001, 60002, 60003]}},
            {"ip_addr": "192.168.3.204",
             "port_forwarding": {"status": True, "gw_ip": "5.6.7.8",  # different gateway
                                 "cmd_port": [60000, 60001, 60002, 60003]}},
        ], "daq_nodes": []}
        v = _make_validator(net=net)
        v._check_port_collisions()
        assert not v.report.has_errors


# ===========================================================================
# _check_daq_assignment_overlap
# ERROR when a single module ID is assigned to multiple DAQ nodes
# ===========================================================================

class TestDaqAssignmentOverlap:
    def test_no_overlap_passes(self) -> None:
        daq = {
            "head_node_data_dir": "/data",
            "head_node_ip_addr": "10.0.0.1",
            "daq_nodes": [
                {"username": "p", "data_dir": "/data", "ip_addr": "10.0.0.2", "module_ids": "0-127"},
                {"username": "p", "data_dir": "/data", "ip_addr": "10.0.0.3", "module_ids": "128-255"},
            ]
        }
        v = _make_validator(daq=daq)
        v._check_daq_assignment_overlap()
        assert not v.report.has_errors

    def test_overlap_errors(self) -> None:
        daq = {
            "head_node_data_dir": "/data",
            "head_node_ip_addr": "10.0.0.1",
            "daq_nodes": [
                {"username": "p", "data_dir": "/data", "ip_addr": "10.0.0.2", "module_ids": "0-10"},
                {"username": "p", "data_dir": "/data", "ip_addr": "10.0.0.3", "module_ids": "5-15"},  # 5-10 overlaps
            ]
        }
        v = _make_validator(daq=daq)
        v._check_daq_assignment_overlap()
        assert v.report.has_errors

    def test_single_node_passes(self) -> None:
        daq = {
            "head_node_data_dir": "/data",
            "head_node_ip_addr": "10.0.0.1",
            "daq_nodes": [{"username": "p", "data_dir": "/data", "ip_addr": "10.0.0.2", "module_ids": "224-231"}]
        }
        v = _make_validator(daq=daq)
        v._check_daq_assignment_overlap()
        assert not v.report.has_errors

    def test_empty_nodes_passes(self) -> None:
        v = _make_validator(daq={"head_node_data_dir": "/data", "head_node_ip_addr": "10.0.0.1", "daq_nodes": []})
        v._check_daq_assignment_overlap()
        assert not v.report.has_errors

    def test_adjacent_ranges_do_not_overlap(self) -> None:
        daq = {
            "head_node_data_dir": "/data",
            "head_node_ip_addr": "10.0.0.1",
            "daq_nodes": [
                {"username": "p", "data_dir": "/data", "ip_addr": "10.0.0.2", "module_ids": "0-63"},
                {"username": "p", "data_dir": "/data", "ip_addr": "10.0.0.3", "module_ids": "64-127"},
                {"username": "p", "data_dir": "/data", "ip_addr": "10.0.0.4", "module_ids": "128-191"},
                {"username": "p", "data_dir": "/data", "ip_addr": "10.0.0.5", "module_ids": "192-255"},
            ]
        }
        v = _make_validator(daq=daq)
        v._check_daq_assignment_overlap()
        assert not v.report.has_errors


# ===========================================================================
# _check_wps_references
# ERROR when a module references a WPS name not defined in obs_config
# ===========================================================================

class TestWpsReferences:
    def test_wps_defined_passes(self) -> None:
        obs = {
            "name": "test",
            "domes": [{"name": "d", "obslat": 0, "obslon": 0, "obsalt": 0, "modules": [{"mobo_serialno": "s", "quabo_version": "bga", "ip_addr": "192.168.3.200", "wps": "wps"}]}],
            "wps": {"url": "http://x", "quabo_socket": 1},
        }
        v = _make_validator(obs=obs)
        v._check_wps_references()
        assert not v.report.has_errors

    def test_wps_undefined_errors(self) -> None:
        obs = {
            "name": "test",
            "domes": [{"name": "d", "obslat": 0, "obslon": 0, "obsalt": 0, "modules": [{"mobo_serialno": "s", "quabo_version": "bga", "ip_addr": "192.168.3.200", "wps": "wps2"}]}],
            "wps": {"url": "http://x", "quabo_socket": 1},
            # wps2 is NOT defined
        }
        v = _make_validator(obs=obs)
        v._check_wps_references()
        assert v.report.has_errors

    def test_default_wps_reference_passes(self) -> None:
        """Module with no explicit 'wps' key defaults to 'wps'."""
        obs = {
            "name": "test",
            "domes": [{"name": "d", "obslat": 0, "obslon": 0, "obsalt": 0, "modules": [{"mobo_serialno": "s", "quabo_version": "bga", "ip_addr": "192.168.3.200"}]}],  # no 'wps' key → defaults to 'wps'
            "wps": {"url": "http://x", "quabo_socket": 1},
        }
        v = _make_validator(obs=obs)
        v._check_wps_references()
        assert not v.report.has_errors

    def test_multiple_wps_references_all_defined_passes(self) -> None:
        obs = {
            "name": "test",
            "domes": [{"name": "d", "obslat": 0, "obslon": 0, "obsalt": 0, "modules": [
                {"mobo_serialno": "s1", "quabo_version": "bga", "ip_addr": "192.168.3.200", "wps": "wps"},
                {"mobo_serialno": "s2", "quabo_version": "bga", "ip_addr": "192.168.3.204", "wps": "wps1"},
            ]}],
            "wps":  {"url": "http://x", "quabo_socket": 1},
            "wps1": {"url": "http://y", "quabo_socket": 2},
        }
        v = _make_validator(obs=obs)
        v._check_wps_references()
        assert not v.report.has_errors

    def test_empty_obs_skips_check(self) -> None:
        """If obs_conf is empty, the check should pass silently."""
        v = _make_validator(obs={"name": "test", "domes": []})
        v._check_wps_references()
        assert not v.report.has_errors


# ===========================================================================
# _check_headnode_disk_space
# ERROR when data dir doesn't exist; PASS/WARN based on available space
# ===========================================================================

class TestHeadnodeDiskSpace:
    def test_missing_data_dir_errors(self) -> None:
        daq = {
            "head_node_data_dir": "/nonexistent_path_xyz_test",
            "head_node_ip_addr": "10.0.0.1",
            "daq_nodes": [],
        }
        v = _make_validator(daq=daq)
        v._check_headnode_disk_space()
        assert v.report.has_errors

    def test_existing_dir_passes(self, tmp_path) -> None:
        daq = {"head_node_data_dir": str(tmp_path), "head_node_ip_addr": "10.0.0.1", "daq_nodes": []}
        obs: dict[str, Any] = {"name": "test", "domes": [{"name": "d", "obslat": 0, "obslon": 0, "obsalt": 0, "modules": []}]}
        data = {"run_type": "sci", "image": {"integration_time_usec": 100_000, "quabo_sample_size": 16}}
        v = _make_validator(obs=obs, data=data, daq=daq)
        v._check_headnode_disk_space()
        # Should not error (tmp_path exists, system has disk space)
        # It may WARN if space is low, but should not ERROR unless truly full
        [t for t in v.report.tests if t["status"] == "ERROR" and "Path" not in t["info"]]
        # Only error is disk-full scenario, not path-not-found
        assert not any("missing or unreachable" in t["info"] for t in v.report.tests)


# ===========================================================================
# validate_all_rules — smoke test that all rules run without crashing
# ===========================================================================

class TestValidateAllRules:
    @pytest.fixture(autouse=True)
    def mock_quabo_uids(self):
        """Mock get_quabo_uids to avoid requiring the physical JSON file in CI."""
        with patch("control.utils.config_file.get_quabo_uids") as mock_get:
            mock_get.return_value = QuaboUidsValidator(domes=[])
            yield mock_get

    def test_runs_without_error_on_minimal_config(self, tmp_path) -> None:
        """All _check_* methods can run on a minimal config without raising."""
        daq = {"head_node_data_dir": str(tmp_path), "head_node_ip_addr": "10.0.0.1", "daq_nodes": []}
        obs = {
            "name": "test",
            "domes": [{"name": "d", "obslat": 0, "obslon": 0, "obsalt": 0, "modules": [{"mobo_serialno": "s", "quabo_version": "bga", "ip_addr": "192.168.3.200", "wps": "wps"}]}],
            "wps": {"url": "http://x", "quabo_socket": 1},
        }
        v = _make_validator(obs=obs, daq=daq, firmware={"bga": "fw.bin"})
        try:
            v.validate_all_rules()  # prints the report; should not raise
        except Exception as e:
            pytest.fail(f"validate_all_rules() raised an exception: {e}")

    def test_count_of_test_results_matches_check_methods(self, tmp_path) -> None:
        """Each _check_* method contributes at least one row to the report."""
        daq = {"head_node_data_dir": str(tmp_path), "head_node_ip_addr": "10.0.0.1", "daq_nodes": []}
        obs = {
            "name": "test",
            "domes": [{"name": "d", "obslat": 0, "obslon": 0, "obsalt": 0, "modules": [{"mobo_serialno": "s", "quabo_version": "bga", "ip_addr": "192.168.3.200", "wps": "wps"}]}],
            "wps": {"url": "http://x", "quabo_socket": 1},
        }
        v = _make_validator(obs=obs, daq=daq, firmware={"bga": "fw.bin"})
        check_methods = [m for m in dir(v) if m.startswith("_check_")]
        v.validate_all_rules()
        # At minimum, each check method should have added something
        assert len(v.report.tests) >= len(check_methods)


    def test_reports_errors_with_valid_config(self, tmp_path) -> None:
        """Verify that validation fails when there are overlapping module IDs."""
        daq = {
            "head_node_data_dir": str(tmp_path), 
            "head_node_ip_addr": "10.0.0.1", 
            "daq_nodes": [
                {
                    "ip_addr": "192.168.0.10",
                    "data_dir": "/data",
                    "username": "root",
                    "module_ids": "1",
                    "bindhost": "lo"
                },
                {
                    "ip_addr": "192.168.0.11",
                    "data_dir": "/data",
                    "username": "root",
                    "module_ids": "1",
                    "bindhost": "lo"
                },
            ]
        }
        obs = {
            "name": "test",
            "domes": [{"name": "d", "obslat": 0, "obslon": 0, "obsalt": 0, "modules": [{"mobo_serialno": "s", "quabo_version": "bga", "ip_addr": "192.168.3.200", "wps": "wps"}]}],
            "wps": {"url": "http://x", "quabo_socket": 1},
        }
        v = _make_validator(obs=obs, daq=daq, firmware={"bga": "fw.bin"})
        v.validate_all_rules()
        assert v.report.has_errors

    def test_reports_no_errors_with_invalid_config(self, tmp_path) -> None:
        """Verify that a valid minimal config passes all rules."""
        daq = {
            "head_node_data_dir": str(tmp_path), 
            "head_node_ip_addr": "10.0.0.1", 
            "daq_nodes": [
                {
                    "ip_addr": "192.168.0.10",
                    "data_dir": "/data",
                    "username": "root",
                    "module_ids": "1",
                    "bindhost": "lo"
                },
            ]
        }
        obs = {
            "name": "test",
            "domes": [{"name": "d", "obslat": 0, "obslon": 0, "obsalt": 0, "modules": [{"mobo_serialno": "s", "quabo_version": "bga", "ip_addr": "192.168.3.200", "wps": "wps"}]}],
            "wps": {"url": "http://x", "quabo_socket": 1},
        }
        v = _make_validator(obs=obs, daq=daq, firmware={"bga": "fw.bin"})
        v.validate_all_rules()
        assert not v.report.has_errors
