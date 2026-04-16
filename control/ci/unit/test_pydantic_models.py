# mypy: ignore-errors
"""
test_pydantic_models.py

Comprehensive unit tests for control/utils/pydantic_config_models.py.
Covers every field boundary, cross-field validator, and interleave constraint.
No hardware required.
"""


from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from utils.pydantic_config_models import (
    AnyTriggerConfig,
    DaqConfigValidator,
    DaqNodeValidator,
    DataConfigValidator,
    FirmwareConfigValidator,
    FlashParams,
    ImageMode,
    InterleaveState,
    ObsConfigValidator,
    ObsDomeConfig,
    ObsModuleConfig,
    PortForwarding,
    PulseHeightMode,
    QuaboUidsValidator,
    StimParams,
)

# ===========================================================================
# ImageMode
# ===========================================================================

class TestImageMode:
    def test_valid_image_mode(self):
        m = ImageMode(integration_time_usec=100_000, pe_threshold=1.0, quabo_sample_size=16)
        assert m.integration_time_usec == 100_000
        assert m.pe_threshold == 1.0

    @pytest.mark.parametrize("usec", [20, 100, 1000, 10_000, 100_000, 200_000, 500_000, 1_000_000])
    def test_valid_integration_times(self, usec):
        """All values that evenly divide 1,000,000 and are >= 20 are valid."""
        ImageMode(integration_time_usec=usec, pe_threshold=1.0, quabo_sample_size=16)

    @pytest.mark.parametrize("usec", [21, 30, 99_999, 333_333, 700_000])
    def test_invalid_integration_time_non_divisor(self, usec):
        """Values >= 20 that don't evenly divide 1,000,000 must be rejected."""
        with pytest.raises(ValidationError, match="must evenly divide"):
            ImageMode(integration_time_usec=usec, pe_threshold=1.0, quabo_sample_size=16)

    def test_integration_time_below_minimum(self):
        with pytest.raises(ValidationError):
            ImageMode(integration_time_usec=10, pe_threshold=1.0, quabo_sample_size=16)

    def test_pe_threshold_at_minimum(self):
        """pe_threshold == 1.0 is the minimum for image mode."""
        m = ImageMode(integration_time_usec=100_000, pe_threshold=1.0, quabo_sample_size=16)
        assert m.pe_threshold == 1.0

    def test_pe_threshold_below_minimum(self):
        with pytest.raises(ValidationError):
            ImageMode(integration_time_usec=100_000, pe_threshold=0.9, quabo_sample_size=16)

    @pytest.mark.parametrize("size", [8, 16])
    def test_valid_quabo_sample_sizes(self, size):
        ImageMode(integration_time_usec=100_000, pe_threshold=1.0, quabo_sample_size=size)

    def test_invalid_quabo_sample_size(self):
        with pytest.raises(ValidationError):
            ImageMode(integration_time_usec=100_000, pe_threshold=1.0, quabo_sample_size=4)  # type: ignore

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ImageMode(integration_time_usec=100_000, pe_threshold=1.0,
                      quabo_sample_size=16, unexpected_key=True)


# ===========================================================================
# PulseHeightMode
# ===========================================================================

class TestPulseHeightMode:
    def test_valid_pulse_height_mode(self):
        m = PulseHeightMode(pe_threshold=3.0, two_pixel_trigger=0, three_pixel_trigger=0)
        assert m.pe_threshold == 3.0
        assert m.two_pixel_trigger == 0
        assert m.three_pixel_trigger == 0

    def test_pe_threshold_at_minimum(self):
        PulseHeightMode(pe_threshold=2.0, two_pixel_trigger=0, three_pixel_trigger=0)

    def test_pe_threshold_below_minimum(self):
        with pytest.raises(ValidationError):
            PulseHeightMode(pe_threshold=1.9, two_pixel_trigger=0, three_pixel_trigger=0)

    def test_two_pixel_trigger(self):
        m = PulseHeightMode(pe_threshold=3.0, two_pixel_trigger=1, three_pixel_trigger=0)
        assert m.two_pixel_trigger == 1

    def test_three_pixel_trigger(self):
        m = PulseHeightMode(pe_threshold=3.0, two_pixel_trigger=0, three_pixel_trigger=1)
        assert m.three_pixel_trigger == 1

    def test_any_trigger_config(self):
        m = PulseHeightMode(
            pe_threshold=3.0,
            two_pixel_trigger=0,
            three_pixel_trigger=0,
            any_trigger=AnyTriggerConfig(group_ph_frames=1)
        )
        assert m.any_trigger is not None
        assert m.any_trigger.group_ph_frames == 1

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            PulseHeightMode(pe_threshold=3.0, two_pixel_trigger=0, three_pixel_trigger=0, unexpected=True) # type: ignore[call-arg]


# ===========================================================================
# DataConfigValidator
# ===========================================================================

class TestDataConfigValidator:
    def test_valid_minimal_data_config(self, minimal_data_config):
        v = DataConfigValidator(**minimal_data_config)
        assert v.run_type == "sci"

    def test_run_type_max_length(self):
        """run_type must be <= 14 characters."""
        DataConfigValidator(run_type="a" * 14)

    def test_run_type_too_long(self):
        with pytest.raises(ValidationError):
            DataConfigValidator(run_type="a" * 15)

    @pytest.mark.parametrize("bad_char", [".", "_", " "])
    def test_run_type_invalid_chars(self, bad_char):
        with pytest.raises(ValidationError, match="invalid character"):
            DataConfigValidator(run_type=f"sci{bad_char}run")

    def test_run_type_valid_alphanumeric(self):
        DataConfigValidator(run_type="scicalib01")

    def test_detector_overvoltage_valid(self):
        """Only 2 and 3 are valid overvoltage values."""
        DataConfigValidator(run_type="sci", detector_overvoltage=2)
        DataConfigValidator(run_type="sci", detector_overvoltage=3)

    def test_detector_overvoltage_invalid(self):
        with pytest.raises(ValidationError):
            DataConfigValidator(run_type="sci", detector_overvoltage=5)  # type: ignore

    def test_image_and_ph_without_trigger_ok(self):
        """Image + PH with no multi-pixel trigger is valid."""
        DataConfigValidator(**{  # type: ignore
            "run_type": "sci",
            "image": {"integration_time_usec": 100_000, "pe_threshold": 1.0, "quabo_sample_size": 16},
            "pulse_height": {"pe_threshold": 3.0},
        })

    def test_image_and_ph_with_two_pixel_trigger_raises(self):
        """image + PH with two_pixel_trigger violates hardware constraint."""
        with pytest.raises(ValidationError, match="Hardware Constraint"):
            DataConfigValidator(**{  # type: ignore
                "run_type": "sci",
                "image": {"integration_time_usec": 100_000, "pe_threshold": 1.0, "quabo_sample_size": 16},
                "pulse_height": {"pe_threshold": 3.0, "two_pixel_trigger": 1},
            })

    def test_image_and_ph_with_three_pixel_trigger_raises(self):
        with pytest.raises(ValidationError, match="Hardware Constraint"):
            DataConfigValidator(**{  # type: ignore
                "run_type": "sci",
                "image": {"integration_time_usec": 100_000, "pe_threshold": 1.0, "quabo_sample_size": 16},
                "pulse_height": {"pe_threshold": 3.0, "three_pixel_trigger": 1},
            })

    def test_dynamic_image_key_valid(self):
        """Keys prefixed 'image_' must be valid ImageMode objects."""
        DataConfigValidator(**{  # type: ignore
            "run_type": "sci",
            "image_8bit": {"integration_time_usec": 200_000, "pe_threshold": 1.0, "quabo_sample_size": 8},
        })

    def test_dynamic_image_key_invalid_fields(self):
        with pytest.raises(ValidationError, match="Invalid fields in dynamic mode"):
            DataConfigValidator(**{  # type: ignore
                "run_type": "sci",
                "image_8bit": {"integration_time_usec": 7, "pe_threshold": 1.0, "quabo_sample_size": 8},
            })

    def test_dynamic_ph_key_valid(self):
        """Keys prefixed 'pulse_height_' must be valid PulseHeightMode objects."""
        DataConfigValidator(**{  # type: ignore
            "run_type": "sci",
            "pulse_height_uhe": {"pe_threshold": 5.0},
        })

    def test_unrecognized_top_level_key_raises(self):
        """Keys that are neither image_* nor pulse_height_* must be rejected."""
        with pytest.raises(ValidationError, match="Unrecognized configuration key"):
            DataConfigValidator(**{"run_type": "sci", "foobar": {"x": 1}})  # type: ignore

    def test_max_file_size_must_be_positive(self):
        with pytest.raises(ValidationError):
            DataConfigValidator(run_type="sci", max_file_size_mb=0)

    def test_max_file_size_valid(self):
        DataConfigValidator(run_type="sci", max_file_size_mb=500)


# ===========================================================================
# InterleaveState & InterleaveConfig
# ===========================================================================

class TestInterleaveState:
    def test_valid_movie_only_state(self):
        s = InterleaveState(state_name="movie", duration_seconds=30.0, movie_mode_config="image")
        assert s.movie_mode_config == "image"
        assert s.pulse_height_mode_config is None

    def test_valid_ph_only_state(self):
        InterleaveState(state_name="ph", duration_seconds=10.0, pulse_height_mode_config="pulse_height")

    def test_valid_both_modes_state(self):
        InterleaveState(state_name="both", duration_seconds=10.0,
                        movie_mode_config="image", pulse_height_mode_config="pulse_height")

    def test_both_null_raises(self):
        """At least one mode must be active."""
        with pytest.raises(ValidationError, match="at least one valid mode"):
            InterleaveState(state_name="empty", duration_seconds=10.0)

    def test_duration_must_be_positive(self):
        with pytest.raises(ValidationError):
            InterleaveState(state_name="s", duration_seconds=0.0, movie_mode_config="image")

    def test_duration_minimum_threshold(self):
        """Duration > 0.01 is the minimum."""
        with pytest.raises(ValidationError):
            InterleaveState(state_name="s", duration_seconds=0.001, movie_mode_config="image")


class TestInterleaveValidation:
    """Integration: DataConfigValidator with interleave block."""

    def _make_config(self, states, extra_modes=None):
        cfg = {
            "run_type": "sci",
            "image": {"integration_time_usec": 100_000, "pe_threshold": 1.0, "quabo_sample_size": 16},
            "pulse_height": {"pe_threshold": 3.0},
            "interleave": {"enable": True, "states": states},
        }
        if extra_modes:
            cfg.update(extra_modes)
        return cfg

    def test_valid_interleave_config(self):
        cfg = self._make_config([
            {"state_name": "movie", "duration_seconds": 30.0, "movie_mode_config": "image"},
            {"state_name": "ph", "duration_seconds": 10.0, "pulse_height_mode_config": "pulse_height"},
        ])
        DataConfigValidator(**cfg)

    def test_interleave_references_missing_image_mode(self):
        cfg = self._make_config([
            {"state_name": "movie", "duration_seconds": 30.0, "movie_mode_config": "image_nonexistent"},
        ])
        with pytest.raises(ValidationError, match="references missing movie mode"):
            DataConfigValidator(**cfg)

    def test_interleave_references_missing_ph_mode(self):
        cfg = self._make_config([
            {"state_name": "ph", "duration_seconds": 10.0,
             "pulse_height_mode_config": "pulse_height_nonexistent"},
        ])
        with pytest.raises(ValidationError, match="references missing pulse height mode"):
            DataConfigValidator(**cfg)

    def test_interleave_state_movie_plus_two_pixel_trigger_raises(self):
        """Hardware constraint: movie + two_pixel_trigger PH in same state."""
        cfg = self._make_config(
            states=[
                {
                    "state_name": "bad",
                    "duration_seconds": 10.0,
                    "movie_mode_config": "image",
                    "pulse_height_mode_config": "pulse_height_multitrig",
                }
            ],
            extra_modes={
                "pulse_height_multitrig": {"pe_threshold": 3.0, "two_pixel_trigger": 1}
            },
        )
        with pytest.raises(ValidationError, match="Hardware Constraint Violation"):
            DataConfigValidator(**cfg)

    def test_interleave_state_movie_plus_three_pixel_trigger_raises(self):
        cfg = self._make_config(
            states=[
                {
                    "state_name": "bad",
                    "duration_seconds": 10.0,
                    "movie_mode_config": "image",
                    "pulse_height_mode_config": "pulse_height_3trig",
                }
            ],
            extra_modes={
                "pulse_height_3trig": {"pe_threshold": 3.0, "three_pixel_trigger": 1}
            },
        )
        with pytest.raises(ValidationError, match="Hardware Constraint Violation"):
            DataConfigValidator(**cfg)

    def test_interleave_references_dynamic_image_mode(self):
        """States can reference dynamic image_* modes."""
        cfg = self._make_config(
            states=[
                {"state_name": "8bit", "duration_seconds": 30.0, "movie_mode_config": "image_8bit"},
            ],
            extra_modes={
                "image_8bit": {"integration_time_usec": 200_000, "pe_threshold": 1.0, "quabo_sample_size": 8}
            },
        )
        DataConfigValidator(**cfg)

    def test_interleave_disabled_with_no_states(self):
        """Disabled interleave with empty states is valid."""
        cfg = {
            "run_type": "sci",
            "interleave": {"enable": False, "states": []},
        }
        DataConfigValidator(**cfg)


# ===========================================================================
# ObsConfigValidator
# ===========================================================================

class TestObsConfigValidator:
    def test_valid_obs_config(self, minimal_obs_config):
        v = ObsConfigValidator(**minimal_obs_config)
        assert v.name == "test_obs"

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            ObsConfigValidator(domes=[])  # type: ignore

    def test_invalid_wr_ip_addr(self):
        with pytest.raises(ValidationError):
            ObsConfigValidator(name="x", domes=[], wr_ip_addr="not.an.ip")  # type: ignore

    def test_invalid_wps_extra_key(self, minimal_obs_config):
        """Only 'wps' prefixed extra keys are allowed."""
        cfg = dict(minimal_obs_config)
        cfg["notawps_key"] = {"url": "x", "quabo_socket": 1}
        with pytest.raises(ValidationError, match="not allowed"):
            ObsConfigValidator(**cfg)  # type: ignore

    def test_valid_wps_extra_key(self, minimal_obs_config):
        """Keys starting with 'wps' (e.g. wps1) are allowed."""
        cfg = dict(minimal_obs_config)
        cfg["wps1"] = {"url": "http://192.168.1.3", "quabo_socket": 2}
        ObsConfigValidator(**cfg)  # type: ignore

    def test_invalid_wps_format_raises(self, minimal_obs_config):
        cfg = dict(minimal_obs_config)
        cfg["wps1"] = {"url": "http://192.168.1.3"}  # missing quabo_socket
        with pytest.raises(ValidationError, match="Invalid format"):
            ObsConfigValidator(**cfg)  # type: ignore


class TestObsDomeConfig:
    def test_valid_dome(self):
        ObsDomeConfig(
            name="dome0", obslat=33.357, obslon=-116.865, obsalt=1706.0,
            modules=[{"mobo_serialno": "SN1", "quabo_version": "bga", "ip_addr": "192.168.3.200"}]
        )

    def test_invalid_latitude(self):
        with pytest.raises(ValidationError):
            ObsDomeConfig(name="d", obslat=91.0, obslon=0.0, obsalt=0.0, modules=[])

    def test_invalid_longitude(self):
        with pytest.raises(ValidationError):
            ObsDomeConfig(name="d", obslat=0.0, obslon=181.0, obsalt=0.0, modules=[])


class TestObsModuleConfig:
    def test_valid_module(self):
        ObsModuleConfig(mobo_serialno="SN1", quabo_version="bga", ip_addr="192.168.3.200")

    def test_invalid_ip_addr(self):
        with pytest.raises(ValidationError):
            ObsModuleConfig(mobo_serialno="SN1", quabo_version="bga", ip_addr="999.0.0.1")  # type: ignore

    @pytest.mark.parametrize("mode", ["wr", "gnss"])
    def test_valid_timing_modes(self, mode):
        ObsModuleConfig(mobo_serialno="SN1", quabo_version="bga",
                        ip_addr="192.168.3.200", timing_mode=mode)

    def test_invalid_timing_mode(self):
        with pytest.raises(ValidationError):
            ObsModuleConfig(mobo_serialno="SN1", quabo_version="bga",
                            ip_addr="192.168.3.200", timing_mode="ntp")

    def test_azimuth_boundaries(self):
        ObsModuleConfig(mobo_serialno="SN1", quabo_version="bga",
                        ip_addr="192.168.3.200", azimuth=0.0)
        ObsModuleConfig(mobo_serialno="SN1", quabo_version="bga",
                        ip_addr="192.168.3.200", azimuth=360.0)
        with pytest.raises(ValidationError):
            ObsModuleConfig(mobo_serialno="SN1", quabo_version="bga",
                            ip_addr="192.168.3.200", azimuth=361.0)

    def test_elevation_boundaries(self):
        ObsModuleConfig(mobo_serialno="SN1", quabo_version="bga",
                        ip_addr="192.168.3.200", elevation=-90.0)
        ObsModuleConfig(mobo_serialno="SN1", quabo_version="bga",
                        ip_addr="192.168.3.200", elevation=90.0)
        with pytest.raises(ValidationError):
            ObsModuleConfig(mobo_serialno="SN1", quabo_version="bga",
                            ip_addr="192.168.3.200", elevation=91.0)

    def test_quabo_version_as_list(self):
        """quabo_version can be a list of per-quabo version strings."""
        ObsModuleConfig(mobo_serialno="SN1",
                        quabo_version=["bga", "bga", "qfp", "bga"],
                        ip_addr="192.168.3.200")


# ===========================================================================
# DaqNodeValidator & DaqConfigValidator
# ===========================================================================

class TestDaqNodeValidator:
    def test_valid_range_string(self):
        n = DaqNodeValidator(username="p", data_dir="/d", ip_addr="10.0.0.2", module_ids="128-255")
        assert n.module_ids == "128-255"

    def test_valid_list_of_ints(self):
        n = DaqNodeValidator(username="p", data_dir="/d", ip_addr="10.0.0.2",
                             module_ids=[0, 1, 2, 5])
        assert n.module_ids == [0, 1, 2, 5]

    def test_range_string_start_greater_than_end_raises(self):
        with pytest.raises(ValidationError, match="must be <="):
            DaqNodeValidator(username="p", data_dir="/d", ip_addr="10.0.0.2", module_ids="10-5")  # type: ignore

    def test_module_ids_empty_list_raises(self):
        with pytest.raises(ValidationError, match="non-empty"):
            DaqNodeValidator(username="p", data_dir="/d", ip_addr="10.0.0.2", module_ids=[])  # type: ignore

    def test_module_ids_negative_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            DaqNodeValidator(username="p", data_dir="/d", ip_addr="10.0.0.2", module_ids=[-1, 0])  # type: ignore

    def test_module_ids_duplicates_raise(self):
        with pytest.raises(ValidationError, match="unique"):
            DaqNodeValidator(username="p", data_dir="/d", ip_addr="10.0.0.2",
                             module_ids=[1, 2, 1])

    def test_invalid_module_ids_format(self):
        with pytest.raises(ValidationError, match="format"):
            DaqNodeValidator(username="p", data_dir="/d", ip_addr="10.0.0.2",
                             module_ids="notarange")

    def test_invalid_ip_addr(self):
        with pytest.raises(ValidationError):
            DaqNodeValidator(username="p", data_dir="/d", ip_addr="999.0.0.1", module_ids="0-10")  # type: ignore


class TestDaqConfigValidator:
    def test_valid_daq_config(self, minimal_daq_config):
        v = DaqConfigValidator(**minimal_daq_config)
        assert v.head_node_ip_addr is not None

    def test_head_node_data_dir_required(self):
        with pytest.raises(ValidationError):
            DaqConfigValidator(head_node_ip_addr="10.0.0.1", daq_nodes=[])

    def test_head_node_matches_daq_node_data_dir_must_match(self):
        """When DAQ node IP == head node IP, data_dirs must match."""
        with pytest.raises(ValidationError, match="does not match"):
            DaqConfigValidator(
                head_node_data_dir="/data",
                head_node_ip_addr="10.0.0.1",
                daq_nodes=[{
                    "username": "p", "data_dir": "/other",
                    "ip_addr": "10.0.0.1", "module_ids": "0-10",
                }],
            )

    def test_head_node_same_ip_matching_data_dir_ok(self):
        DaqConfigValidator(
            head_node_data_dir="/data",
            head_node_ip_addr="10.0.0.1",
            daq_nodes=[{
                "username": "p", "data_dir": "/data",
                "ip_addr": "10.0.0.1", "module_ids": "0-10",
            }],
        )

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            DaqConfigValidator(
                head_node_data_dir="/data",
                head_node_ip_addr="10.0.0.1",
                daq_nodes=[],
                unexpected=True,
            )


# ===========================================================================
# Flash / Stim / LongPulse
# ===========================================================================

class TestFlashParams:
    def test_valid_flash_params(self):
        FlashParams(rate=3, level=15, width=7)

    @pytest.mark.parametrize("rate", [-1, 8])
    def test_rate_out_of_range(self, rate):
        with pytest.raises(ValidationError):
            FlashParams(rate=rate, level=15, width=7)

    @pytest.mark.parametrize("level", [-1, 32])
    def test_level_out_of_range(self, level):
        with pytest.raises(ValidationError):
            FlashParams(rate=3, level=level, width=7)

    @pytest.mark.parametrize("width", [-1, 16])
    def test_width_out_of_range(self, width):
        with pytest.raises(ValidationError):
            FlashParams(rate=3, level=15, width=width)


class TestStimParams:
    def test_valid_stim_params(self):
        StimParams(rate=3, level=128, mask=[True, False, True, False])

    def test_mask_must_have_4_elements(self):
        with pytest.raises(ValidationError):
            StimParams(rate=3, level=128, mask=[True, False, True])

    def test_mask_cannot_exceed_4(self):
        with pytest.raises(ValidationError):
            StimParams(rate=3, level=128, mask=[True, True, True, True, True])


# ===========================================================================
# QuaboUids validators
# ===========================================================================

class TestQuaboUidsValidator:
    def _make_module(self):
        return {"quabos": [{"uid": "AABB"}, {"uid": "CCDD"}, {"uid": "EEFF"}, {"uid": ""}]}

    def test_valid_quabo_uids(self):
        QuaboUidsValidator(domes=[{"modules": [self._make_module()]}])

    def test_module_must_have_four_quabos(self):
        with pytest.raises(ValidationError):
            QuaboUidsValidator(domes=[{"modules": [{"quabos": [{"uid": "X"}]}]}])

    def test_module_cannot_have_three_quabos(self):
        with pytest.raises(ValidationError):
            QuaboUidsValidator(domes=[{"modules": [{"quabos": [
                {"uid": "A"}, {"uid": "B"}, {"uid": "C"}
            ]}]}])

    def test_empty_uid_is_valid(self):
        """Empty string UID means quabo is offline."""
        mod = {"quabos": [{"uid": ""}, {"uid": ""}, {"uid": ""}, {"uid": ""}]}
        QuaboUidsValidator(domes=[{"modules": [mod]}])


# ===========================================================================
# FirmwareConfigValidator (extra='allow')
# ===========================================================================

class TestFirmwareConfigValidator:
    def test_arbitrary_hw_keys_allowed(self):
        """Firmware config accepts any hardware variant keys."""
        FirmwareConfigValidator(bga="fw_bga_v2.bin", qfp="fw_qfp_v1.bin")  # type: ignore

    def test_empty_firmware_config_ok(self):
        FirmwareConfigValidator()  # type: ignore


# ===========================================================================
# PortForwarding — grpc_port field
# ===========================================================================

class TestPortForwarding:
    _BASE: ClassVar[dict[str, Any]] = {"status": True, "gw_ip": "203.0.113.1"}

    def test_valid_without_grpc_port(self):
        """grpc_port is optional; omitting it is valid."""
        pf = PortForwarding(**self._BASE)
        assert pf.grpc_port is None

    def test_valid_grpc_port(self):
        """grpc_port in 1-65535 is accepted."""
        pf = PortForwarding(**self._BASE, grpc_port=50051)
        assert pf.grpc_port == 50051

    def test_grpc_port_zero_rejected(self):
        """grpc_port=0 is below the valid range."""
        with pytest.raises(ValidationError):
            PortForwarding(**self._BASE, grpc_port=0)  # type: ignore

    def test_grpc_port_too_large_rejected(self):
        """grpc_port above 65535 is rejected."""
        with pytest.raises(ValidationError):
            PortForwarding(**self._BASE, grpc_port=65536)  # type: ignore

    def test_grpc_port_max_valid(self):
        """grpc_port=65535 is the highest valid value."""
        pf = PortForwarding(**self._BASE, grpc_port=65535)  # type: ignore
        assert pf.grpc_port == 65535
