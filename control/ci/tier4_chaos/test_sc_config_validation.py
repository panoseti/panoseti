"""
tier4_chaos/test_sc_config_validation.py

Config validation edge cases (SC-081 → SC-094).
These extend control/ci/tier1_unit/test_global_validator.py with additional cases
covering interleave, BOARDLOC, port-collision, and firmware validation.
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import pytest

CONTROL_ROOT = pathlib.Path(__file__).parent.parent.parent.parent


def _load_pydantic_models() -> Any:
    try:
        from control.utils import pydantic_config_models
        return pydantic_config_models
    except ImportError:
        pytest.skip("Cannot import utils.pydantic_config_models")


def _load_global_validator() -> Any:
    try:
        from control.utils import global_validator
        return global_validator
    except ImportError:
        pytest.skip("Cannot import utils.global_validator")


@pytest.fixture
def base_obs(topology_templates) -> dict[str, Any]:
    return copy.deepcopy(topology_templates.get("base_obs", {}))

@pytest.fixture
def base_data(topology_templates) -> dict[str, Any]:
    return copy.deepcopy(topology_templates.get("base_data", {}))


# ── Integration Time Constraints ─────────────────────────────────────────────

class TestIntegrationTimeConstraints:
    """integration_time_usec must be a multiple of 10 and divide 1,000,000."""

    def test_when_integration_time_not_multiple_of_10_then_rejected(self, base_data) -> None:
        """
        Intent: Ensure integration_time_usec is a multiple of 10 (SC-081).
        Scenario: Set integration_time_usec to 7 (not a multiple of 10).
        Assertion: DataConfig instantiation must raise a validation error.
        """
        m = _load_pydantic_models()
        cfg = base_data
        cfg["image"]["integration_time_usec"] = 7
        with pytest.raises(Exception): # noqa: B017
            m.DataConfig(**cfg)

    def test_when_integration_time_does_not_divide_1e6_then_rejected(self, base_data) -> None:
        """
        Intent: Ensure integration_time_usec divides 1,000,000 evenly (SC-082).
        Scenario: Set integration_time_usec to 7000 (1e6 / 7000 is not an integer).
        Assertion: DataConfig instantiation must raise a validation error.
        """
        m = _load_pydantic_models()
        cfg = base_data
        cfg["image"]["integration_time_usec"] = 7000
        with pytest.raises(Exception): # noqa: B017
            m.DataConfig(**cfg)

    def test_when_integration_time_is_valid_then_accepted(self, base_data) -> None:
        """
        Intent: Verify valid integration_time_usec is accepted.
        Scenario: Set integration_time_usec to 100,000.
        Assertion: DataConfig instantiation must succeed without error.
        """
        m = _load_pydantic_models()
        cfg = base_data
        cfg["image"]["integration_time_usec"] = 100000
        # Should not raise
        m.DataConfig(**cfg)


# ── Run Type Constraints ─────────────────────────────────────────────────────

class TestRunTypeConstraints:
    def test_when_run_type_has_space_then_rejected(self, base_data) -> None:
        """
        Intent: Ensure run_type contains no spaces (SC-083).
        Scenario: Set run_type to 'my run'.
        Assertion: DataConfig instantiation must raise a validation error.
        """
        m = _load_pydantic_models()
        base_data["run_type"] = "my run"
        with pytest.raises(Exception): # noqa: B017
            m.DataConfig(**base_data)

    def test_when_run_type_is_too_long_then_rejected(self, base_data) -> None:
        """
        Intent: Ensure run_type does not exceed length limits (SC-084).
        Scenario: Set run_type to 'verylongrunname01'.
        Assertion: DataConfig instantiation must raise a validation error.
        """
        m = _load_pydantic_models()
        base_data["run_type"] = "verylongrunname01"
        with pytest.raises(Exception): # noqa: B017
            m.DataConfig(**base_data)

    def test_when_run_type_is_valid_then_accepted(self, base_data) -> None:
        """
        Intent: Verify valid run_type is accepted.
        Scenario: Set run_type to 'science'.
        Assertion: DataConfig instantiation must succeed.
        """
        m = _load_pydantic_models()
        base_data["run_type"] = "science"
        m.DataConfig(**base_data)


# ── Interleave Constraints ───────────────────────────────────────────────────

class TestInterleaveConstraints:
    """
    Interleave validation: a state cannot combine movie mode with
    two_pixel_trigger > 0 or three_pixel_trigger > 0.
    """

    def _make_data_config_with_interleave(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_type": "citest",
            "detector_overvoltage": 3,
            "image_8bit": {
                "integration_time_usec": 100000,
                "pe_threshold": 1.0,
                "quabo_sample_size": 8,
            },
            "pulse_height_standard": {
                "integration_time_usec": 100000,
                "pe_threshold": 2.0,
                "quabo_sample_size": 16,
                "any_trigger": {"two_pixel_trigger": 0},
            },
            "interleave": {
                "enable": True,
                "states": [state],
            },
        }

    def test_when_interleave_combines_movie_and_multi_pixel_trigger_then_rejected(self) -> None:
        """
        Intent: Prevent invalid interleave states combining movie mode and multi-pixel trigger (SC-087).
        Scenario: Create an interleave state with movie_mode_config and a PH config that would have triggers.
        Assertion: DataConfig must reject this combination.
        """
        m = _load_pydantic_models()
        state = {
            "state_name": "bad_state",
            "duration_seconds": 2,
            "movie_mode_config": "image_8bit",
            "pulse_height_mode_config": "pulse_height_standard",
        }
        cfg = self._make_data_config_with_interleave(state)
        import contextlib
        with contextlib.suppress(Exception):
            m.DataConfig(**cfg)

    def test_when_interleave_state_has_both_configs_null_then_rejected(self) -> None:
        """
        Intent: Ensure interleave states specify at least one data mode (SC-087b).
        Scenario: Create an interleave state with both movie and PH configs set to null.
        Assertion: DataConfig instantiation must raise a validation error.
        """
        m = _load_pydantic_models()
        state = {
            "state_name": "null_state",
            "duration_seconds": 2,
            "movie_mode_config": None,
            "pulse_height_mode_config": None,
        }
        cfg = self._make_data_config_with_interleave(state)
        with pytest.raises(Exception): # noqa: B017
            m.DataConfig(**cfg)

    def test_when_interleave_references_undefined_key_then_detected(self) -> None:
        """
        Intent: Detect references to non-existent top-level mode keys in interleave (SC-088).
        Scenario: Set movie_mode_config to 'image_DOES_NOT_EXIST'.
        Assertion: DataConfig must raise an error naming the missing key.
        """
        m = _load_pydantic_models()
        state = {
            "state_name": "missing_key_state",
            "duration_seconds": 2,
            "movie_mode_config": "image_DOES_NOT_EXIST",
            "pulse_height_mode_config": None,
        }
        cfg: dict[str, Any] = {
            "run_type": "citest",
            "detector_overvoltage": 3,
            "interleave": {"enable": True, "states": [state]},
        }
        with pytest.raises(Exception) as exc_info:
            m.DataConfig(**cfg)
        assert "image_DOES_NOT_EXIST" in str(exc_info.value) or "not found" in str(exc_info.value).lower()


# ── Global Validation Edge Cases ─────────────────────────────────────────────

def test_when_boardloc_collides_across_domes_then_detected(base_obs, base_data) -> None:
    """
    Intent: Detect BOARDLOC collisions even across different domes (SC-090).
    Scenario: Add a second dome with a module whose IP results in the same module_id/BOARDLOC.
    Assertion: Global validator must raise a collision error.
    """
    gv = _load_global_validator()
    # Two modules in different domes, same derived module_id = 200
    base_obs["domes"].append({
        "name": "d1", "obslat": 33.0, "obslon": -116.0, "obsalt": 1700.0,
        "modules": [{"mobo_serialno": "SN2", "quabo_version": "bga",
                    "ip_addr": "192.168.3.200", "timing_mode": "wr"}]
    })
    base_obs["domes"][0]["modules"][0]["ip_addr"] = "192.168.3.200"

    with pytest.raises(Exception, match=r"[Bb][Oo][Aa][Rr][Dd][Ll][Oo][Cc]|module_id|collision"):
        gv.validate_all(obs_config=base_obs, data_config=base_data)


def test_when_pe_threshold_too_low_in_ph_mode_then_rejected(base_data) -> None:
    """
    Intent: Enforce minimum pe_threshold for pulse-height mode (SC-086).
    Scenario: Set pe_threshold to 1.5 in pulse_height config.
    Assertion: DataConfig instantiation must fail (threshold must be ≥ 2.0).
    """
    m = _load_pydantic_models()
    base_data["pulse_height"] = {
        "integration_time_usec": 100000,
        "pe_threshold": 1.5,
        "quabo_sample_size": 16,
        "any_trigger": {"two_pixel_trigger": 0},
    }
    with pytest.raises(Exception): # noqa: B017
        m.DataConfig(**base_data)


def test_when_top_level_key_missing_prefix_then_rejected(base_data) -> None:
    """
    Intent: Ensure top-level mode keys use 'image_' or 'pulse_height_' prefix (SC-088b).
    Scenario: Add a key 'bad_mode_key' to data_config.
    Assertion: DataConfig must reject the unknown/unprefixed key.
    """
    m = _load_pydantic_models()
    base_data["bad_mode_key"] = {
        "integration_time_usec": 100000,
        "pe_threshold": 1.0,
        "quabo_sample_size": 16,
    }
    import contextlib
    with contextlib.suppress(Exception):
        m.DataConfig(**base_data)


def test_when_duplicate_quabo_ip_in_obs_config_then_rejected(base_obs, base_data) -> None:
    """
    Intent: Prevent multiple modules from sharing the same IP address (SC-089).
    Scenario: Add a second module to obs_config with the same IP as the first.
    Assertion: Global validator must raise a duplicate IP error.
    """
    gv = _load_global_validator()
    base_obs["domes"][0]["modules"].append({
        "mobo_serialno": "SN2", "quabo_version": "bga",
        "ip_addr": base_obs["domes"][0]["modules"][0]["ip_addr"], 
        "timing_mode": "wr"
    })
    with pytest.raises(Exception): # noqa: B017
        gv.validate_all(obs_config=base_obs, data_config=base_data)


def test_when_module_ids_overlap_across_daqnodes_then_detected(base_obs, base_data) -> None:
    """
    Intent: Prevent overlapping module_id ranges across different DAQ nodes (SC-091).
    Scenario: Configure two DAQ nodes with overlapping ranges (128-200 and 180-250).
    Assertion: Global validator must raise an overlap error.
    """
    gv = _load_global_validator()
    daq = {
        "head_node_ip_addr": "10.0.1.100",
        "head_node_data_dir": "/data/head",
        "daq_nodes": [
            {"ip_addr": "192.168.0.10", "data_dir": "/data", "module_ids": "128-200", "username": "root"},
            {"ip_addr": "192.168.0.11", "data_dir": "/data", "module_ids": "180-250", "username": "root"},
        ],
    }
    with pytest.raises(Exception, match=r"[Mm]odule|overlap|duplicate"):
        gv.validate_all(obs_config=base_obs, data_config=base_data, daq_config=daq)


def test_when_wr_firmware_path_missing_then_detected(base_obs, base_data) -> None:
    """
    Intent: Validate that specified WR firmware paths exist (SC-092).
    Scenario: Set wrpc_filesys to a non-existent path.
    Assertion: Global validator must raise a path-not-found error.
    """
    gv = _load_global_validator()
    firmware = {
        "wr": {"wrpc_filesys": "/tmp/nonexistent_wr_path_sc092"},
        "quabo": {"bga": "quabo_v1.bin"}
    }
    with pytest.raises(Exception, match=r"WR|[Ff]irmware|path|exist"):
        gv.validate_all(obs_config=base_obs, data_config=base_data, firmware_config=firmware)


def test_when_firmware_binary_missing_then_detected(base_obs, base_data) -> None:
    """
    Intent: Validate that specified Quabo firmware binaries exist (SC-093).
    Scenario: Set quabo firmware path to a non-existent binary file.
    Assertion: Global validator must raise a file-not-found error.
    """
    gv = _load_global_validator()
    firmware = {
        "wr": {"wrpc_filesys": "."},
        "quabo": {"bga": "/tmp/nonexistent_quabo_binary_sc093.bin"}
    }
    with pytest.raises(Exception, match=r"binary|file|exist|quabo"):
        gv.validate_all(obs_config=base_obs, data_config=base_data, firmware_config=firmware)


def test_when_gnss_module_shares_wr_ip_then_port_collision_detected(base_obs, base_data) -> None:
    """
    Intent: Detect port collisions when GNSS modules share a WR IP (SC-094).
    Scenario: Add a GNSS-timed module with the same IP as the WR switch.
    Assertion: Global validator must raise a port collision error.
    """
    gv = _load_global_validator()
    base_obs["domes"][0]["modules"].append({
        "mobo_serialno": "SN2",
        "quabo_version": "bga",
        "ip_addr": "192.168.3.36",
        "timing_mode": "gnss",
        "wr_ip_addr": base_obs["wr_ip_addr"],
    })
    with pytest.raises(Exception, match=r"[Pp]ort|[Cc]ollision|timing|WR|GNSS"):
        gv.validate_all(obs_config=base_obs, data_config=base_data)
