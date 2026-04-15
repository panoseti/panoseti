"""
test_global_validator.py

Unit tests for control/utils/global_validator.py.
Tests each _check_* rule of GlobalConfigValidator in isolation by constructing
minimal config dicts and verifying the ValidationReport outcome.
No hardware or network access required.
"""

import pytest

from utils.global_validator import GlobalConfigValidator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_validator(obs=None, data=None, daq=None, net=None, firmware=None):
    """Build a GlobalConfigValidator with sensible defaults, overrideable per-test."""
    return GlobalConfigValidator({
        "obs":      obs      or {},
        "data":     data     or {},
        "daq":      daq      or {},
        "network":  net      or {},
        "firmware": firmware or {},
    })


def _run_check(validator, method_name):
    """Call a single _check_* method and return (passed, report)."""
    getattr(validator, method_name)()
    return not validator.report.has_errors, validator.report


def _check_passes(validator, method_name):
    passed, _ = _run_check(validator, method_name)
    return passed


def _check_fails(validator, method_name):
    passed, _ = _run_check(validator, method_name)
    return not passed


# ===========================================================================
# _check_science_guardrails
# WARN when run_type != 'eng*' and flash/stim params are present
# ===========================================================================

class TestScienceGuardrails:
    def test_sci_run_with_no_stim_passes(self):
        v = _make_validator(data={"run_type": "sci"})
        v._check_science_guardrails()
        assert not v.report.has_errors

    def test_eng_run_with_flash_passes(self):
        v = _make_validator(data={"run_type": "eng", "flash_params": {"rate": 3}})
        v._check_science_guardrails()
        assert not v.report.has_errors

    def test_sci_run_with_flash_warns(self):
        v = _make_validator(data={"run_type": "sci", "flash_params": {"rate": 3}})
        v._check_science_guardrails()
        # WARN is not an ERROR; has_errors stays False
        assert not v.report.has_errors
        statuses = [t["status"] for t in v.report.tests]
        assert "WARN" in statuses

    def test_sci_run_with_stim_warns(self):
        v = _make_validator(data={"run_type": "sci", "stim_params": {"rate": 1}})
        v._check_science_guardrails()
        assert not v.report.has_errors
        assert any(t["status"] == "WARN" for t in v.report.tests)

    def test_empty_data_config_passes(self):
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

    def test_single_dome_passes(self):
        v = _make_validator(obs={"domes": [{"name": "D0", "obslat": 33.0, "obslon": -116.0}]})
        v._check_geospatial_coherence()
        assert not v.report.has_errors

    def test_no_domes_passes(self):
        v = _make_validator(obs={"domes": []})
        v._check_geospatial_coherence()
        assert not v.report.has_errors

    def test_two_nearby_domes_pass(self):
        v = _make_validator(obs={"domes": self._nearby()})
        v._check_geospatial_coherence()
        assert not v.report.has_errors

    def test_two_far_domes_error(self):
        v = _make_validator(obs={"domes": self._far_apart()})
        v._check_geospatial_coherence()
        assert v.report.has_errors
        assert any("ERROR" in t["status"] for t in v.report.tests)

    def test_decimal_typo_triggers_error(self):
        """Simulates a common typo: Palomar vs. Palomar-far where decimal is off."""
        domes = [
            {"name": "D0", "obslat": 33.357, "obslon": -116.865},
            {"name": "D1", "obslat": 43.357, "obslon": -116.865},  # 10° off = ~1100 km
        ]
        v = _make_validator(obs={"domes": domes})
        v._check_geospatial_coherence()
        assert v.report.has_errors


# ===========================================================================
# _check_hardware_firmware
# ERROR when a module uses a hardware type not listed in firmware_config
# ===========================================================================

class TestHardwareFirmware:
    def test_matching_firmware_passes(self):
        obs = {"domes": [{"modules": [{"quabo_version": "bga"}]}]}
        v = _make_validator(obs=obs, firmware={"bga": "fw_bga.bin"})
        v._check_hardware_firmware()
        assert not v.report.has_errors

    def test_missing_firmware_errors(self):
        obs = {"domes": [{"modules": [{"quabo_version": "bga"}]}]}
        v = _make_validator(obs=obs, firmware={"qfp": "fw_qfp.bin"})  # bga is missing
        v._check_hardware_firmware()
        assert v.report.has_errors

    def test_multiple_hw_types_all_covered(self):
        obs = {"domes": [{"modules": [
            {"quabo_version": "bga"},
            {"quabo_version": "qfp"},
        ]}]}
        fw = {"bga": "fw_bga.bin", "qfp": "fw_qfp.bin"}
        v = _make_validator(obs=obs, firmware=fw)
        v._check_hardware_firmware()
        assert not v.report.has_errors

    def test_per_quabo_version_list_checked(self):
        """quabo_version can be a list of per-quabo versions."""
        obs = {"domes": [{"modules": [{"quabo_version": ["bga", "bga", "qfp", "bga"]}]}]}
        fw = {"bga": "fw_bga.bin"}  # qfp missing
        v = _make_validator(obs=obs, firmware=fw)
        v._check_hardware_firmware()
        assert v.report.has_errors

    def test_no_modules_passes(self):
        v = _make_validator(obs={"domes": [{"modules": []}]}, firmware={"bga": "fw.bin"})
        v._check_hardware_firmware()
        assert not v.report.has_errors

    def test_empty_obs_passes(self):
        v = _make_validator(firmware={"bga": "fw.bin"})
        v._check_hardware_firmware()
        assert not v.report.has_errors


# ===========================================================================
# _check_overvoltage_consensus
# ERROR when obs_config and data_config have different detector_overvoltage
# ===========================================================================

class TestOvervoltageConsensus:
    def test_matching_voltages_passes(self):
        v = _make_validator(
            obs={"detector_overvoltage": 3},
            data={"detector_overvoltage": 3},
        )
        v._check_overvoltage_consensus()
        assert not v.report.has_errors

    def test_mismatched_voltages_errors(self):
        v = _make_validator(
            obs={"detector_overvoltage": 3},
            data={"detector_overvoltage": 2},
        )
        v._check_overvoltage_consensus()
        assert v.report.has_errors

    def test_obs_overvoltage_none_passes(self):
        """If obs doesn't specify overvoltage, no constraint."""
        v = _make_validator(
            obs={},
            data={"detector_overvoltage": 3},
        )
        v._check_overvoltage_consensus()
        assert not v.report.has_errors

    def test_data_overvoltage_none_passes(self):
        v = _make_validator(
            obs={"detector_overvoltage": 3},
            data={},
        )
        v._check_overvoltage_consensus()
        assert not v.report.has_errors

    def test_both_none_passes(self):
        v = _make_validator()
        v._check_overvoltage_consensus()
        assert not v.report.has_errors

    @pytest.mark.parametrize("obs_v, data_v, should_error", [
        (2, 2, False),
        (3, 3, False),
        (2, 3, True),
        (3, 2, True),
    ])
    def test_voltage_pairs(self, obs_v, data_v, should_error):
        v = _make_validator(
            obs={"detector_overvoltage": obs_v},
            data={"detector_overvoltage": data_v},
        )
        v._check_overvoltage_consensus()
        assert v.report.has_errors == should_error


# ===========================================================================
# _check_port_collisions
# ERROR when two modules on the same gateway use overlapping forwarded ports
# ===========================================================================

class TestPortCollisions:
    def test_no_network_config_passes(self):
        v = _make_validator(net={})
        v._check_port_collisions()
        assert not v.report.has_errors

    def test_no_active_forwarding_passes(self):
        net = {"modules": [
            {"ip_addr": "192.168.3.200", "port_forwarding": {"status": False, "gw_ip": "1.2.3.4",
                                                              "cmd_port": [60000, 60001, 60002, 60003]}}
        ]}
        v = _make_validator(net=net)
        v._check_port_collisions()
        assert not v.report.has_errors

    def test_non_overlapping_ports_on_same_gateway_pass(self):
        net = {"modules": [
            {"ip_addr": "192.168.3.200",
             "port_forwarding": {"status": True, "gw_ip": "1.2.3.4",
                                 "cmd_port": [60000, 60001, 60002, 60003]}},
            {"ip_addr": "192.168.3.204",
             "port_forwarding": {"status": True, "gw_ip": "1.2.3.4",
                                 "cmd_port": [61000, 61001, 61002, 61003]}},
        ]}
        v = _make_validator(net=net)
        v._check_port_collisions()
        assert not v.report.has_errors

    def test_overlapping_ports_on_same_gateway_errors(self):
        net = {"modules": [
            {"ip_addr": "192.168.3.200",
             "port_forwarding": {"status": True, "gw_ip": "1.2.3.4",
                                 "cmd_port": [60000, 60001, 60002, 60003]}},
            {"ip_addr": "192.168.3.204",
             "port_forwarding": {"status": True, "gw_ip": "1.2.3.4",
                                 "cmd_port": [60000, 61001, 61002, 61003]}},  # 60000 collision!
        ]}
        v = _make_validator(net=net)
        v._check_port_collisions()
        assert v.report.has_errors

    def test_same_ports_on_different_gateways_pass(self):
        net = {"modules": [
            {"ip_addr": "192.168.3.200",
             "port_forwarding": {"status": True, "gw_ip": "1.2.3.4",
                                 "cmd_port": [60000, 60001, 60002, 60003]}},
            {"ip_addr": "192.168.3.204",
             "port_forwarding": {"status": True, "gw_ip": "5.6.7.8",  # different gateway
                                 "cmd_port": [60000, 60001, 60002, 60003]}},
        ]}
        v = _make_validator(net=net)
        v._check_port_collisions()
        assert not v.report.has_errors


# ===========================================================================
# _check_daq_assignment_overlap
# ERROR when a single module ID is assigned to multiple DAQ nodes
# ===========================================================================

class TestDaqAssignmentOverlap:
    def test_no_overlap_passes(self):
        daq = {"daq_nodes": [
            {"module_ids": "0-127"},
            {"module_ids": "128-255"},
        ]}
        v = _make_validator(daq=daq)
        v._check_daq_assignment_overlap()
        assert not v.report.has_errors

    def test_overlap_errors(self):
        daq = {"daq_nodes": [
            {"module_ids": "0-10"},
            {"module_ids": "5-15"},  # 5-10 overlaps
        ]}
        v = _make_validator(daq=daq)
        v._check_daq_assignment_overlap()
        assert v.report.has_errors

    def test_single_node_passes(self):
        daq = {"daq_nodes": [{"module_ids": "224-231"}]}
        v = _make_validator(daq=daq)
        v._check_daq_assignment_overlap()
        assert not v.report.has_errors

    def test_empty_nodes_passes(self):
        v = _make_validator(daq={"daq_nodes": []})
        v._check_daq_assignment_overlap()
        assert not v.report.has_errors

    def test_adjacent_ranges_do_not_overlap(self):
        daq = {"daq_nodes": [
            {"module_ids": "0-63"},
            {"module_ids": "64-127"},
            {"module_ids": "128-191"},
            {"module_ids": "192-255"},
        ]}
        v = _make_validator(daq=daq)
        v._check_daq_assignment_overlap()
        assert not v.report.has_errors


# ===========================================================================
# _check_wps_references
# ERROR when a module references a WPS name not defined in obs_config
# ===========================================================================

class TestWpsReferences:
    def test_wps_defined_passes(self):
        obs = {
            "domes": [{"modules": [{"wps": "wps"}]}],
            "wps": {"url": "http://x", "quabo_socket": 1},
        }
        v = _make_validator(obs=obs)
        v._check_wps_references()
        assert not v.report.has_errors

    def test_wps_undefined_errors(self):
        obs = {
            "domes": [{"modules": [{"wps": "wps2"}]}],
            "wps": {"url": "http://x", "quabo_socket": 1},
            # wps2 is NOT defined
        }
        v = _make_validator(obs=obs)
        v._check_wps_references()
        assert v.report.has_errors

    def test_default_wps_reference_passes(self):
        """Module with no explicit 'wps' key defaults to 'wps'."""
        obs = {
            "domes": [{"modules": [{}]}],  # no 'wps' key → defaults to 'wps'
            "wps": {"url": "http://x", "quabo_socket": 1},
        }
        v = _make_validator(obs=obs)
        v._check_wps_references()
        assert not v.report.has_errors

    def test_multiple_wps_references_all_defined_passes(self):
        obs = {
            "domes": [{"modules": [
                {"wps": "wps"},
                {"wps": "wps1"},
            ]}],
            "wps":  {"url": "http://x", "quabo_socket": 1},
            "wps1": {"url": "http://y", "quabo_socket": 2},
        }
        v = _make_validator(obs=obs)
        v._check_wps_references()
        assert not v.report.has_errors

    def test_empty_obs_skips_check(self):
        """If obs_conf is empty, the check should pass silently."""
        v = _make_validator(obs={})
        v._check_wps_references()
        assert not v.report.has_errors


# ===========================================================================
# _check_headnode_disk_space
# ERROR when data dir doesn't exist; PASS/WARN based on available space
# ===========================================================================

class TestHeadnodeDiskSpace:
    def test_missing_data_dir_errors(self):
        daq = {
            "head_node_data_dir": "/nonexistent_path_xyz_test",
            "daq_nodes": [],
        }
        v = _make_validator(daq=daq)
        v._check_headnode_disk_space()
        assert v.report.has_errors

    def test_existing_dir_passes(self, tmp_path):
        daq = {"head_node_data_dir": str(tmp_path), "daq_nodes": []}
        obs = {"domes": [{"modules": []}]}
        data = {"image": {"integration_time_usec": 100_000, "quabo_sample_size": 16}}
        v = _make_validator(obs=obs, data=data, daq=daq)
        v._check_headnode_disk_space()
        # Should not error (tmp_path exists, system has disk space)
        # It may WARN if space is low, but should not ERROR unless truly full
        errors = [t for t in v.report.tests if t["status"] == "ERROR" and "Path" not in t["info"]]
        # Only error is disk-full scenario, not path-not-found
        assert not any("missing or unreachable" in t["info"] for t in v.report.tests)


# ===========================================================================
# validate_all_rules — smoke test that all rules run without crashing
# ===========================================================================

class TestValidateAllRules:
    def test_runs_without_error_on_minimal_config(self, tmp_path):
        """All _check_* methods can run on a minimal config without raising."""
        daq = {"head_node_data_dir": str(tmp_path), "daq_nodes": []}
        obs = {
            "domes": [{"modules": [{"quabo_version": "bga", "wps": "wps"}]}],
            "wps": {"url": "http://x", "quabo_socket": 1},
        }
        v = _make_validator(obs=obs, daq=daq, firmware={"bga": "fw.bin"})
        try:
            v.validate_all_rules()  # prints the report; should not raise
        except Exception as e:
            pytest.fail(f"validate_all_rules() raised an exception: {e}")

    def test_count_of_test_results_matches_check_methods(self, tmp_path):
        """Each _check_* method contributes at least one row to the report."""
        daq = {"head_node_data_dir": str(tmp_path), "daq_nodes": []}
        obs = {
            "domes": [{"modules": [{"quabo_version": "bga", "wps": "wps"}]}],
            "wps": {"url": "http://x", "quabo_socket": 1},
        }
        v = _make_validator(obs=obs, daq=daq, firmware={"bga": "fw.bin"})
        check_methods = [m for m in dir(v) if m.startswith("_check_")]
        v.validate_all_rules()
        # At minimum, each check method should have added something
        assert len(v.report.tests) >= len(check_methods)
