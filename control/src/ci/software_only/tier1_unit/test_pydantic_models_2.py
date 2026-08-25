# mypy: ignore-errors
"""
test_pydantic_models_2.py

Comprehensive unit tests for control/utils/pydantic_config_models.py.
Covers every field boundary, cross-field validator, and interleave constraint.
No hardware required.
"""


from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from control.utils.pydantic_config_models import (
    DaqConfig,
    DaqNode,
    DataConfig,
    FirmwareConfig,
    FlashParams,
    InterleaveState,
    NetworkConfig,
    NetworkDaqNode,
    NetworkHeadnode,
    ObsConfig,
    ObsDomeConfig,
    ObsModuleConfig,
    PortForwarding,
    QuaboUids,
    StimParams,
)

# ===========================================================================
# InterleaveState & InterleaveConfig
# ===========================================================================

class TestInterleaveState:
    def test_valid_movie_only_state(self) -> None:
        s = InterleaveState(state_name="movie", duration_seconds=30.0, movie_mode_config="image")
        assert s.movie_mode_config == "image"
        assert s.pulse_height_mode_config is None

    def test_valid_ph_only_state(self) -> None:
        InterleaveState(state_name="ph", duration_seconds=10.0, pulse_height_mode_config="pulse_height")

    def test_valid_both_modes_state(self) -> None:
        InterleaveState(state_name="both", duration_seconds=10.0,
                        movie_mode_config="image", pulse_height_mode_config="pulse_height")

    def test_both_null_raises(self) -> None:
        """At least one mode must be active."""
        with pytest.raises(ValidationError, match="at least one valid mode"):
            InterleaveState(state_name="empty", duration_seconds=10.0)

    def test_duration_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            InterleaveState(state_name="s", duration_seconds=0.0, movie_mode_config="image")

    def test_duration_minimum_threshold(self) -> None:
        """Duration > 0.01 is the minimum."""
        with pytest.raises(ValidationError):
            InterleaveState(state_name="s", duration_seconds=0.001, movie_mode_config="image")


class TestInterleaveValidation:
    """Integration: DataConfig with interleave block."""

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

    def test_valid_interleave_config(self) -> None:
        cfg = self._make_config([
            {"state_name": "movie", "duration_seconds": 30.0, "movie_mode_config": "image"},
            {"state_name": "ph", "duration_seconds": 10.0, "pulse_height_mode_config": "pulse_height"},
        ])
        DataConfig(**cfg)

    def test_interleave_references_missing_image_mode(self) -> None:
        cfg = self._make_config([
            {"state_name": "movie", "duration_seconds": 30.0, "movie_mode_config": "image_nonexistent"},
        ])
        with pytest.raises(ValidationError, match="references missing movie mode"):
            DataConfig(**cfg)

    def test_interleave_references_missing_ph_mode(self) -> None:
        cfg = self._make_config([
            {"state_name": "ph", "duration_seconds": 10.0,
             "pulse_height_mode_config": "pulse_height_nonexistent"},
        ])
        with pytest.raises(ValidationError, match="references missing pulse height mode"):
            DataConfig(**cfg)

    def test_interleave_state_movie_plus_two_pixel_trigger_raises(self) -> None:
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
            DataConfig(**cfg)

    def test_interleave_state_movie_plus_three_pixel_trigger_raises(self) -> None:
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
            DataConfig(**cfg)

    def test_interleave_references_dynamic_image_mode(self) -> None:
        """States can reference dynamic image_* modes."""
        cfg = self._make_config(
            states=[
                {"state_name": "8bit", "duration_seconds": 30.0, "movie_mode_config": "image_8bit"},
            ],
            extra_modes={
                "image_8bit": {"integration_time_usec": 200_000, "pe_threshold": 1.0, "quabo_sample_size": 8}
            },
        )
        DataConfig(**cfg)

    def test_interleave_disabled_with_no_states(self) -> None:
        """Disabled interleave with empty states is valid."""
        cfg = {
            "run_type": "sci",
            "interleave": {"enable": False, "states": []},
        }
        DataConfig(**cfg)


# ===========================================================================
# ObsConfig
# ===========================================================================

class TestObsConfig:
    def test_valid_obs_config(self, minimal_obs_config) -> None:
        v = ObsConfig(**minimal_obs_config)
        assert v.name == "test_obs"

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            ObsConfig(domes=[])  # type: ignore

    def test_invalid_wr_ip_addr(self) -> None:
        with pytest.raises(ValidationError):
            ObsConfig(name="x", domes=[], wr_ip_addr="not.an.ip")  # type: ignore

    def test_invalid_wps_extra_key(self, minimal_obs_config) -> None:
        """Only 'wps' prefixed extra keys are allowed."""
        cfg = dict(minimal_obs_config)
        cfg["notawps_key"] = {"url": "x", "quabo_socket": 1}
        with pytest.raises(ValidationError, match="not allowed"):
            ObsConfig(**cfg)  # type: ignore

    def test_valid_wps_extra_key(self, minimal_obs_config) -> None:
        """Keys starting with 'wps' (e.g. wps1) are allowed."""
        cfg = dict(minimal_obs_config)
        cfg["wps1"] = {"url": "http://192.168.1.3", "quabo_socket": 2}
        ObsConfig(**cfg)  # type: ignore

    def test_invalid_wps_format_raises(self, minimal_obs_config) -> None:
        cfg = dict(minimal_obs_config)
        cfg["wps1"] = {"url": "http://192.168.1.3"}  # missing quabo_socket
        with pytest.raises(ValidationError, match="Invalid format"):
            ObsConfig(**cfg)  # type: ignore


class TestObsDomeConfig:
    def test_valid_dome(self) -> None:
        ObsDomeConfig(
            name="dome0", obslat=33.357, obslon=-116.865, obsalt=1706.0,
            modules=[{"mobo_serialno": "SN1", "quabo_version": "bga", "ip_addr": "192.168.3.200"}]
        )

    def test_invalid_latitude(self) -> None:
        with pytest.raises(ValidationError):
            ObsDomeConfig(name="d", obslat=91.0, obslon=0.0, obsalt=0.0, modules=[])

    def test_invalid_longitude(self) -> None:
        with pytest.raises(ValidationError):
            ObsDomeConfig(name="d", obslat=0.0, obslon=181.0, obsalt=0.0, modules=[])


class TestObsModuleConfig:
    def test_valid_module(self) -> None:
        ObsModuleConfig(
            mobo_serialno="SN1", quabo_version="bga", ip_addr="192.168.3.200", 
            timing_mode="wr", azimuth=0.0, elevation=0.0, id=1
        )
    def test_invalid_ip_addr(self) -> None:
        with pytest.raises(ValidationError):
            ObsModuleConfig(mobo_serialno="SN1", quabo_version="bga", ip_addr="999.0.0.1")  # type: ignore

    @pytest.mark.parametrize("mode", ["wr", "gnss"])
    def test_valid_timing_modes(self, mode) -> None:
        ObsModuleConfig(mobo_serialno="SN1", quabo_version="bga",
                        ip_addr="192.168.3.200", timing_mode=mode, azimuth=0.0, elevation=0.0, id=1)

    def test_invalid_timing_mode(self) -> None:
        with pytest.raises(ValidationError):
            ObsModuleConfig(mobo_serialno="SN1", quabo_version="bga",
                            ip_addr="192.168.3.200", timing_mode="ntp", azimuth=0.0, elevation=0.0, id=1)

    def test_azimuth_boundaries(self) -> None:
        ObsModuleConfig(
            mobo_serialno="SN1", quabo_version="bga", ip_addr="192.168.3.200", 
            timing_mode="wr", azimuth=0.0, elevation=0.0, id=1
        )
        ObsModuleConfig(
            mobo_serialno="SN1", quabo_version="bga", ip_addr="192.168.3.200", 
            timing_mode="wr", azimuth=360.0, elevation=0.0, id=1
        )
        with pytest.raises(ValidationError):
            ObsModuleConfig(
                mobo_serialno="SN1", quabo_version="bga", ip_addr="192.168.3.200", 
                timing_mode="wr", azimuth=361.0, elevation=0.0, id=1
            )

    def test_elevation_boundaries(self) -> None:
        ObsModuleConfig(mobo_serialno="SN1", quabo_version="bga",
                        ip_addr="192.168.3.200", timing_mode="wr", azimuth=0.0, elevation=-90.0, id=1)
        ObsModuleConfig(
            mobo_serialno="SN1", quabo_version="bga", ip_addr="192.168.3.200", 
            timing_mode="wr", azimuth=0.0, elevation=90.0, id=1
        )
        with pytest.raises(ValidationError):
            ObsModuleConfig(
                mobo_serialno="SN1", quabo_version="bga", ip_addr="192.168.3.200", 
                timing_mode="wr", azimuth=0.0, elevation=91.0, id=1
            )

    def test_quabo_version_as_list(self) -> None:
        """quabo_version can be a list of per-quabo version strings."""
        ObsModuleConfig(mobo_serialno="SN1",
                        quabo_version=["bga", "bga", "qfp", "bga"],
                        ip_addr="192.168.3.200", timing_mode="wr", azimuth=0.0, elevation=0.0, id=1)


# ===========================================================================
# DaqNode & DaqConfig
# ===========================================================================

class TestDaqNode:
    def test_valid_range_string(self) -> None:
        n = DaqNode(username="p", data_dir="/d", ip_addr="10.0.0.2", module_ids="128-255")
        assert n.module_ids == list(range(128, 256))

    def test_valid_list_of_ints(self) -> None:
        n = DaqNode(username="p", data_dir="/d", ip_addr="10.0.0.2",
                             module_ids=[0, 1, 2, 5])
        assert n.module_ids == [0, 1, 2, 5]

    def test_range_string_start_greater_than_end_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be <="):
            DaqNode(username="p", data_dir="/d", ip_addr="10.0.0.2", module_ids="10-5")  # type: ignore

    def test_module_ids_empty_list_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            DaqNode(username="p", data_dir="/d", ip_addr="10.0.0.2", module_ids=[])  # type: ignore

    def test_module_ids_negative_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            DaqNode(username="p", data_dir="/d", ip_addr="10.0.0.2", module_ids=[-1, 0])  # type: ignore

    def test_module_ids_duplicates_raise(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            DaqNode(username="p", data_dir="/d", ip_addr="10.0.0.2",
                             module_ids=[1, 2, 1])

    def test_invalid_module_ids_format(self) -> None:
        with pytest.raises(ValidationError, match="format"):
            DaqNode(username="p", data_dir="/d", ip_addr="10.0.0.2",
                             module_ids="notarange")

    def test_invalid_ip_addr(self) -> None:
        with pytest.raises(ValidationError):
            DaqNode(username="p", data_dir="/d", ip_addr="999.0.0.1", module_ids="0-10")  # type: ignore


class TestDaqConfig:
    def test_valid_daq_config(self, minimal_daq_config) -> None:
        v = DaqConfig(**minimal_daq_config)
        assert v.head_node_ip_addr is not None

    def test_head_node_data_dir_required(self) -> None:
        with pytest.raises(ValidationError):
            DaqConfig(head_node_ip_addr="10.0.0.1", daq_nodes=[])

    def test_head_node_matches_daq_node_data_dir_must_not_match(self) -> None:
        """When DAQ node IP == head node IP, data_dirs must NOT match."""
        with pytest.raises(ValidationError, match="MUST be different"):
            DaqConfig(
                head_node_data_dir="/data",
                head_node_ip_addr="10.0.0.1",
                daq_nodes=[{
                    "username": "p", "data_dir": "/data",
                    "ip_addr": "10.0.0.1", "module_ids": "0-10",
                }],
            )

    def test_head_node_same_ip_different_data_dir_ok(self) -> None:
        DaqConfig(
            head_node_data_dir="/data",
            head_node_ip_addr="10.0.0.1",
            daq_nodes=[{
                "username": "p", "data_dir": "/other",
                "ip_addr": "10.0.0.1", "module_ids": "0-10",
            }],
        )

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            DaqConfig(
                head_node_data_dir="/data",
                head_node_ip_addr="10.0.0.1",
                daq_nodes=[],
                unexpected=True,
            )


# ===========================================================================
# Flash / Stim / LongPulse
# ===========================================================================

class TestFlashParams:
    def test_valid_flash_params(self) -> None:
        FlashParams(rate=3, level=15, width=7)

    @pytest.mark.parametrize("rate", [-1, 8])
    def test_rate_out_of_range(self, rate) -> None:
        with pytest.raises(ValidationError):
            FlashParams(rate=rate, level=15, width=7)

    @pytest.mark.parametrize("level", [-1, 32])
    def test_level_out_of_range(self, level) -> None:
        with pytest.raises(ValidationError):
            FlashParams(rate=3, level=level, width=7)

    @pytest.mark.parametrize("width", [-1, 16])
    def test_width_out_of_range(self, width) -> None:
        with pytest.raises(ValidationError):
            FlashParams(rate=3, level=15, width=width)


class TestStimParams:
    def test_valid_stim_params(self) -> None:
        StimParams(rate=3, level=128, mask=[True, False, True, False])

    def test_mask_must_have_4_elements(self) -> None:
        with pytest.raises(ValidationError):
            StimParams(rate=3, level=128, mask=[True, False, True])

    def test_mask_cannot_exceed_4(self) -> None:
        with pytest.raises(ValidationError):
            StimParams(rate=3, level=128, mask=[True, True, True, True, True])


# ===========================================================================
# QuaboUids validators
# ===========================================================================

class TestQuaboUids:
    def _make_module(self):
        return {
            "ip_addr": "192.168.3.200",
            "quabos": [{"uid": "AABB"}, {"uid": "CCDD"}, {"uid": "EEFF"}, {"uid": ""}]
        }

    def test_valid_quabo_uids(self) -> None:
        QuaboUids(domes=[{"modules": [self._make_module()]}])

    def test_module_must_have_four_quabos(self) -> None:
        with pytest.raises(ValidationError):
            QuaboUids(domes=[{"modules": [{"ip_addr": "1.1.1.1", "quabos": [{"uid": "X"}]}]}])

    def test_module_cannot_have_three_quabos(self) -> None:
        with pytest.raises(ValidationError):
            QuaboUids(domes=[{"modules": [{"ip_addr": "1.1.1.1", "quabos": [
                {"uid": "A"}, {"uid": "B"}, {"uid": "C"}
            ]}]}])

    def test_empty_uid_is_valid(self) -> None:
        """Empty string UID means quabo is offline."""
        mod = {
            "ip_addr": "192.168.3.200",
            "quabos": [{"uid": ""}, {"uid": ""}, {"uid": ""}, {"uid": ""}]
        }
        QuaboUids(domes=[{"modules": [mod]}])


# ===========================================================================
# FirmwareConfig (extra='allow')
# ===========================================================================

class TestFirmwareConfig:
    def test_arbitrary_hw_keys_allowed(self) -> None:
        """Firmware config accepts any hardware variant keys."""
        FirmwareConfig(bga="fw_bga_v2.bin", qfp="fw_qfp_v1.bin")  # type: ignore

    def test_empty_firmware_config_ok(self) -> None:
        FirmwareConfig()  # type: ignore


# ===========================================================================
# PortForwarding — grpc_port field
# ===========================================================================

class TestPortForwarding:
    _BASE: ClassVar[dict[str, Any]] = {"status": True, "gw_ip": "203.0.113.1"}

    def test_valid_without_grpc_port(self) -> None:
        """grpc_port is optional; omitting it means "not explicitly forwarded" (None), not a 50051 default.

        This distinguishes "operator set it" from "field default" so
        daq_grpc_endpoint() can fall through to a direct (env-resolved)
        connection instead of silently assuming every forwarded node's gRPC
        traffic goes through the gateway on 50051.
        """
        pf = PortForwarding(**self._BASE)
        assert pf.grpc_port is None

    def test_valid_grpc_port(self) -> None:
        """grpc_port in 1-65535 is accepted."""
        from random import randint
        for _ in range(10):
            grpc_port = 50051 + randint(-10000, 10000)
            pf = PortForwarding(**self._BASE, grpc_port=grpc_port)
            assert pf.grpc_port == grpc_port

    def test_grpc_port_zero_rejected(self) -> None:
        """grpc_port=0 is below the valid range."""
        with pytest.raises(ValidationError):
            PortForwarding(**self._BASE, grpc_port=0)  # type: ignore

    def test_grpc_port_too_large_rejected(self) -> None:
        """grpc_port above 65535 is rejected."""
        with pytest.raises(ValidationError):
            PortForwarding(**self._BASE, grpc_port=65536)  # type: ignore

    def test_grpc_port_max_valid(self) -> None:
        """grpc_port=65535 is the highest valid value."""
        pf = PortForwarding(**self._BASE, grpc_port=65535)  # type: ignore
        assert pf.grpc_port == 65535


# ===========================================================================
# NetworkConfig / NetworkDaqNode / NetworkHeadnode
# ===========================================================================

class TestNetworkHeadnode:
    def test_defaults_to_none_when_omitted(self) -> None:
        """None (not a 50051 literal) so callers can distinguish "operator
        set it in network_config.json" from "field default" -- see
        health.py's _check_grpc_headnode(), which passes this straight
        through as resolve_grpc_port()'s explicit override and only then
        falls through to HEADNODE_GRPC_PORT / the 50051 default.
        """
        cfg = NetworkConfig()
        assert cfg.headnode.grpc_port is None

    def test_explicit_value(self) -> None:
        cfg = NetworkConfig(headnode=NetworkHeadnode(grpc_port=60051))
        assert cfg.headnode.grpc_port == 60051

    def test_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NetworkHeadnode(grpc_port=70000)  # type: ignore


class TestNetworkDaqNode:
    """NetworkDaqNode.grpc_port is a sibling of port_forwarding -- used for
    direct connections when port_forwarding.status is False. Distinct from
    port_forwarding.grpc_port, which only applies when status is True.
    """

    def test_grpc_port_defaults_to_none_when_omitted(self) -> None:
        """None (not a 50051 literal) so attach_daq_config() can tell "not
        set in network_config.json" apart from "explicitly set" and only
        seed daq_config.json's DaqNode.grpc_port when it's itself unset.
        """
        node = NetworkDaqNode(
            ip_addr="192.168.0.10",  # type: ignore
            port_forwarding=PortForwarding(status=False, gw_ip="10.0.1.254"),
        )
        assert node.grpc_port is None

    def test_grpc_port_explicit_value(self) -> None:
        node = NetworkDaqNode(
            ip_addr="192.168.0.10",  # type: ignore
            grpc_port=50077,
            port_forwarding=PortForwarding(status=False, gw_ip="10.0.1.254"),
        )
        assert node.grpc_port == 50077

    def test_grpc_port_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NetworkDaqNode(
                ip_addr="192.168.0.10",  # type: ignore
                grpc_port=0,  # type: ignore
                port_forwarding=PortForwarding(status=False, gw_ip="10.0.1.254"),
            )

    def test_grpc_port_independent_of_port_forwarding_grpc_port(self) -> None:
        node = NetworkDaqNode(
            ip_addr="192.168.0.10",  # type: ignore
            grpc_port=50077,
            port_forwarding=PortForwarding(status=True, gw_ip="10.0.1.254", grpc_port=50099),
        )
        assert node.grpc_port == 50077
        assert node.port_forwarding.grpc_port == 50099
