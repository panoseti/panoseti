"""
test_global_validator_2.py

Unit tests for control/utils/global_validator.py.
Tests each _check_* rule of GlobalConfigValidator in isolation by constructing
minimal config dicts and verifying the ValidationReport outcome.
No hardware or network access required.
"""

from typing import Any
from unittest.mock import patch

import pytest

from ci.software_only.tier1_unit.conftest import _make_validator
from control.utils.pydantic_config_models import QuaboUids

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
            mock_get.return_value = QuaboUids(domes=[])
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
