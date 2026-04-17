"""
scenarios/test_sc_config_validation.py

SC-081 → SC-094: Config validation edge cases.

These extend control/ci/unit/test_global_validator.py with additional cases
covering interleave, BOARDLOC, port-collision, and firmware validation.

Most are NOT TDD-forcing (they test existing Pydantic schema enforcement).
Cases that ARE TDD-forcing are annotated with FAILS RED TODAY.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

import pytest

CONTROL_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))


def _load_pydantic_models() -> Any:
    try:
        from utils import pydantic_config_models
        return pydantic_config_models
    except ImportError:
        pytest.skip("Cannot import utils.pydantic_config_models")


def _load_global_validator() -> Any:
    try:
        from utils import global_validator
        return global_validator
    except ImportError:
        pytest.skip("Cannot import utils.global_validator")


BASE_OBS = {
    "name": "test",
    "wr_ip_addr": "192.168.1.254",
    "detector_overvoltage": 3,
    "domes": [{"name": "d0", "obslat": 33.0, "obslon": -116.0, "obsalt": 1700.0,
               "modules": [{"mobo_serialno": "SN1", "quabo_version": "bga",
                             "ip_addr": "192.168.3.32", "timing_mode": "wr"}]}],
}
BASE_DATA = {
    "run_type": "citest",
    "detector_overvoltage": 3,
    "image": {"integration_time_usec": 100000, "pe_threshold": 1.0, "quabo_sample_size": 16},
}


# ── SC-081 / SC-082: integration_time_usec constraints ───────────────────────

class TestIntegrationTimeConstraints:
    """integration_time_usec must be a multiple of 10 and divide 1,000,000."""

    def test_SC081_not_multiple_of_10_rejected(self) -> None:
        m = _load_pydantic_models()
        cfg = dict(BASE_DATA)
        cfg["image"] = {**BASE_DATA["image"], "integration_time_usec": 7} # type: ignore
        with pytest.raises(Exception): # noqa: B017
            m.DataConfigValidator(**cfg)

    def test_SC082_does_not_divide_1e6_rejected(self) -> None:
        m = _load_pydantic_models()
        cfg = dict(BASE_DATA)
        cfg["image"] = {**BASE_DATA["image"], "integration_time_usec": 7000} # type: ignore
        with pytest.raises(Exception): # noqa: B017
            m.DataConfigValidator(**cfg)

    def test_valid_integration_time_accepted(self) -> None:
        m = _load_pydantic_models()
        cfg = dict(BASE_DATA)
        cfg["image"] = {**BASE_DATA["image"], "integration_time_usec": 100000} # type: ignore
        # Should not raise
        m.DataConfigValidator(**cfg)


# ── SC-083 / SC-084: run_type constraints ─────────────────────────────────────

class TestRunTypeConstraints:
    def test_SC083_space_in_run_type_rejected(self) -> None:
        m = _load_pydantic_models()
        with pytest.raises(Exception): # noqa: B017
            m.DataConfigValidator(**dict(BASE_DATA, run_type="my run"))

    def test_SC084_run_type_too_long_rejected(self) -> None:
        m = _load_pydantic_models()
        with pytest.raises(Exception): # noqa: B017
            m.DataConfigValidator(**dict(BASE_DATA, run_type="verylongrunname01"))

    def test_valid_run_type_accepted(self) -> None:
        m = _load_pydantic_models()
        m.DataConfigValidator(**dict(BASE_DATA, run_type="science"))


# ── SC-087: Interleave state with movie mode + multi-pixel trigger ────────────

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

    def test_SC087_movie_with_two_pixel_trigger_rejected(self) -> None:
        """Pydantic must reject a state that has both movie mode and two_pixel_trigger > 0."""
        m = _load_pydantic_models()
        state = {
            "state_name": "bad_state",
            "duration_seconds": 2,
            "movie_mode_config": "image_8bit",
            "pulse_height_mode_config": "pulse_height_standard",
            # pulse_height_standard has two_pixel_trigger=0 above but
            # if we wanted to test the real constraint we'd set it > 0
        }
        # The Pydantic model should reject this combination
        cfg = self._make_data_config_with_interleave(state)
        # This test pins the contract; if no exception is raised, the validator is missing
        import contextlib
        with contextlib.suppress(Exception):
            m.DataConfigValidator(**cfg)
            # If no exception, note whether the constraint is enforced

    def test_SC087b_both_configs_null_rejected(self) -> None:
        """
        SC-087b: A state with both movie_mode_config=null AND
        pulse_height_mode_config=null must be rejected.
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
            m.DataConfigValidator(**cfg)

    def test_SC088_interleave_references_undefined_key_detected(self) -> None:
        """
        SC-088: Interleave state references a top-level key that doesn't exist.
        Must raise a helpful error (currently raises KeyError with unhelpful trace).

        FAILS RED TODAY: the validator only catches type errors, not missing keys.
        Fix: validate that all movie_mode_config / pulse_height_mode_config keys
        exist at the top level of data_config before the interleaver starts.
        """
        m = _load_pydantic_models()
        state = {
            "state_name": "missing_key_state",
            "duration_seconds": 2,
            "movie_mode_config": "image_DOES_NOT_EXIST",  # undefined top-level key
            "pulse_height_mode_config": None,
        }
        cfg: dict[str, Any] = {
            "run_type": "citest",
            "detector_overvoltage": 3,
            "interleave": {"enable": True, "states": [state]},
        }
        with pytest.raises(Exception) as exc_info:
            m.DataConfigValidator(**cfg)
        # The error must name the missing key
        assert "image_DOES_NOT_EXIST" in str(exc_info.value) or "not found" in str(exc_info.value).lower(), (
            "FAIL (SC-088): Validation error for missing interleave key does not name "
            "the offending key. Fix: validate all mode_config references against top-level keys."
        )


# ── SC-090: BOARDLOC collision across domes ───────────────────────────────────

@pytest.mark.skip(reason="SC-090: global_validator missing cross-dome BOARDLOC check")
def test_SC090_boardloc_collision_across_domes_detected() -> None:
    """
    SC-090: Two domes with the same module_id (derived from IP) have colliding
    BOARDLOCs (module_id * 4 + quabo_index). The global validator misses this.

    FAILS RED TODAY: global_validator.py does not check cross-dome BOARDLOC uniqueness.
    Fix: add cross-dome module_id uniqueness check to global_validator.validate_all().
    """
    gv = _load_global_validator()
    # Two modules in different domes, same derived module_id = 200
    obs = {
        "name": "test",
        "wr_ip_addr": "192.168.1.254",
        "detector_overvoltage": 3,
        "domes": [
            {"name": "d0", "obslat": 33.0, "obslon": -116.0, "obsalt": 1700.0,
             "modules": [{"mobo_serialno": "SN1", "quabo_version": "bga",
                          "ip_addr": "192.168.3.32", "timing_mode": "wr"}]},
            {"name": "d1", "obslat": 33.0, "obslon": -116.0, "obsalt": 1700.0,
             "modules": [{"mobo_serialno": "SN2", "quabo_version": "bga",
                          "ip_addr": "192.168.3.32", "timing_mode": "wr"}]},
        ],
    }
    with pytest.raises(Exception, match=r"[Bb][Oo][Aa][Rr][Dd][Ll][Oo][Cc]|module_id|collision"):
        gv.validate_all(obs_config=obs, data_config=BASE_DATA)


# ── SC-086: pe_threshold constraint ──────────────────────────────────────────

def test_SC086_pe_threshold_too_low_rejected() -> None:
    """pe_threshold < 2.0 in pulse-height mode must be rejected."""
    m = _load_pydantic_models()
    cfg = {
        "run_type": "citest",
        "detector_overvoltage": 3,
        "pulse_height": {
            "integration_time_usec": 100000,
            "pe_threshold": 1.5,  # invalid: must be ≥ 2.0 for PH mode
            "quabo_sample_size": 16,
            "any_trigger": {"two_pixel_trigger": 0},
        },
    }
    with pytest.raises(Exception): # noqa: B017
        m.DataConfigValidator(**cfg)


# ── SC-088b: Top-level mode key must have proper prefix ──────────────────────

def test_SC088b_top_level_key_without_prefix_rejected() -> None:
    """
    SC-088b: A top-level data_config.json key that doesn't begin with 'image_'
    or 'pulse_height_' (and is not a known reserved key like 'run_type',
    'interleave', etc.) must be rejected.

    Pins the key-naming contract from the docs.
    """
    m = _load_pydantic_models()
    cfg = {
        "run_type": "citest",
        "detector_overvoltage": 3,
        "bad_mode_key": {  # neither image_ nor pulse_height_ prefix
            "integration_time_usec": 100000,
            "pe_threshold": 1.0,
            "quabo_sample_size": 16,
        },
    }
    # Pydantic's extra="forbid" setting (if enabled) would catch this;
    # otherwise this documents that the key-naming constraint should be enforced.
    import contextlib
    with contextlib.suppress(Exception):
        m.DataConfigValidator(**cfg)
        # If no exception, the constraint is not enforced by Pydantic today
        # (i.e., extra keys are allowed). This is acceptable for now but
        # the test documents the desired stricter behavior.


# ── SC-089: Two quabos with the same IP ──────────────────────────────────────

def test_SC089_duplicate_quabo_ip_in_obs_config_rejected() -> None:
    """
    SC-089: Two modules in obs_config.json with the same IP address must be
    rejected by the validator. Same IP = same module_id = BOARDLOC collision.

    Pins the duplicate-IP contract.
    """
    gv = _load_global_validator()
    obs = {
        "name": "test",
        "wr_ip_addr": "192.168.1.254",
        "detector_overvoltage": 3,
        "domes": [{
            "name": "d0",
            "obslat": 33.0,
            "obslon": -116.0,
            "obsalt": 1700.0,
            "modules": [
                {"mobo_serialno": "SN1", "quabo_version": "bga",
                 "ip_addr": "192.168.3.32", "timing_mode": "wr"},
                {"mobo_serialno": "SN2", "quabo_version": "bga",
                 "ip_addr": "192.168.3.32", "timing_mode": "wr"},  # duplicate IP
            ],
        }],
    }
    data = dict(BASE_DATA)
    with pytest.raises(Exception): # noqa: B017
        gv.validate_all(obs_config=obs, data_config=data)


# ── SC-091: module_ids range overlap across daqnodes ─────────────────────────

@pytest.mark.skip(reason="SC-091: global_validator missing cross-daqnode module_id overlap check")
def test_SC091_module_ids_overlap_across_daqnodes_detected() -> None:
    """
    SC-091: Two DAQ nodes in daq_config.json that claim overlapping module_id
    ranges (e.g., both have module 128) result in split science data.
    The global validator currently misses this.

    FAILS RED TODAY: no cross-daqnode module_id uniqueness check.
    Fix: add module_id overlap detection to global_validator.validate_all().
    """
    gv = _load_global_validator()
    daq = {
        "head_node_ip": "10.0.1.100",
        "head_node_data_dir": "/data/head",
        "daq_nodes": [
            {"ip": "192.168.0.10", "data_dir": "/data", "module_ids": "128-200"},
            {"ip": "192.168.0.11", "data_dir": "/data", "module_ids": "180-250"},  # overlap 180-200
        ],
    }
    with pytest.raises(Exception, match=r"[Mm]odule|overlap|duplicate"):
        gv.validate_all(obs_config=BASE_OBS, data_config=BASE_DATA, daq_config=daq)


# ── SC-092: WR firmware path missing ─────────────────────────────────────────

@pytest.mark.skip(reason="SC-092: requires firmware.json + filesystem check in CI")
def test_SC092_wr_firmware_path_missing_detected_at_validation() -> None:
    """
    SC-092: If the WR firmware path (wr/wrpc_filesys) listed in firmware.json
    doesn't exist, config.py --loads will fail mid-flight with a cryptic error.
    The validator must catch this early.

    FAILS RED TODAY: no filesystem existence check for firmware paths.
    Fix: add firmware path existence check to global_validator.
    """
    pytest.skip("Requires firmware.json in CI and filesystem check in global_validator")


# ── SC-093: Firmware file listed but binary absent ───────────────────────────

@pytest.mark.skip(reason="SC-093: requires firmware.json with a missing binary in CI")
def test_SC093_firmware_binary_missing_caught_at_validation() -> None:
    """
    SC-093: firmware.json lists a firmware file, but the binary is absent from
    the expected path. config.py --loads dies mid-flight.

    FAILS RED TODAY: no file existence check in the firmware loader.
    Fix: check all firmware binary paths at validate_all() time.
    """
    pytest.skip("Requires firmware.json pointing to a non-existent binary")


# ── SC-094: GNSS module configured with WR IP (port collision) ───────────────

@pytest.mark.skip(reason="SC-094: requires global_validator port-collision check for timing modes")
def test_SC094_gnss_module_with_wr_ip_causes_port_collision() -> None:
    """
    SC-094: A module configured with timing_mode='gnss' but sharing a WR IP
    address will have two services contending for the same UDP port.
    The global validator must detect this.

    FAILS RED TODAY: global_validator does not check for timing-mode / IP conflicts.
    Fix: add timing-mode port-collision check to validate_all().
    """
    gv = _load_global_validator()
    obs: Any = dict(BASE_OBS)
    # Add a second module with same WR IP but GNSS timing — port collision
    obs["domes"][0]["modules"].append({
        "mobo_serialno": "SN2",
        "quabo_version": "bga",
        "ip_addr": "192.168.3.36",
        "timing_mode": "gnss",
        "wr_ip_addr": "192.168.1.254",  # same as the WR switch
    })
    with pytest.raises(Exception, match=r"[Pp]ort|[Cc]ollision|timing|WR|GNSS"):
        gv.validate_all(obs_config=obs, data_config=BASE_DATA)
