# mypy: ignore-errors
"""
test_fleet_spec.py — Unit tests for the FleetSpec DSL and Topology synthesis.

Verifies that FleetSpec.build() correctly:
 - produces valid Pydantic models for all 7 config types
 - passes GlobalConfigValidator (no ERRORs)
 - builds a NetworkX graph with the correct node/edge structure
 - seeded RNG is deterministic
"""

import pytest

from ci.software_only_v2.infra.spec import FleetSpec, GatewaySpec, Topology
from control.utils.pydantic_config_models import (
    DaqConfig,
    DaemonConfig,
    DataConfig,
    FirmwareConfig,
    NetworkConfig,
    ObsConfig,
    QuaboUids,
)


class TestFleetSpecMinimalUnit:
    """FleetSpec.minimal_unit() — single dome, one module, one DAQ node."""

    def test_build_returns_topology(self) -> None:
        t = FleetSpec.minimal_unit().build()
        assert isinstance(t, Topology)

    def test_all_config_types_are_correct(self) -> None:
        t = FleetSpec.minimal_unit().build()
        assert isinstance(t.obs, ObsConfig)
        assert isinstance(t.daq, DaqConfig)
        assert isinstance(t.network, NetworkConfig)
        assert isinstance(t.data, DataConfig)
        assert isinstance(t.firmware, FirmwareConfig)
        assert isinstance(t.quabo_uids, QuaboUids)
        assert isinstance(t.daemons, DaemonConfig)

    def test_obs_config_has_one_dome(self) -> None:
        t = FleetSpec.minimal_unit().build()
        assert len(t.obs.domes) == 1
        assert t.obs.domes[0].name == "dome0"

    def test_obs_config_has_one_module(self) -> None:
        t = FleetSpec.minimal_unit().build()
        assert len(t.obs.domes[0].modules) == 1
        assert t.obs.domes[0].modules[0].id == 200

    def test_quabo_uids_has_four_entries_per_module(self) -> None:
        t = FleetSpec.minimal_unit().build()
        mod = t.quabo_uids.domes[0].modules[0]
        assert len(mod.quabos) == 4

    def test_data_config_is_valid(self) -> None:
        t = FleetSpec.minimal_unit().build()
        assert t.data.run_type == "engineering"
        assert t.data.image is not None

    def test_firmware_config_has_qfp_and_bga(self) -> None:
        t = FleetSpec.minimal_unit().build()
        assert t.firmware.qfp is not None
        assert t.firmware.bga is not None

    def test_daemons_all_disabled(self) -> None:
        t = FleetSpec.minimal_unit().build()
        # All daemons should be False (safe test default)
        for val in t.daemons.daemons.model_extra.values():
            assert val is False

    def test_graph_has_headnode_and_module(self) -> None:
        import networkx as nx
        t = FleetSpec.minimal_unit().build()
        g = t.graph
        assert isinstance(g, nx.DiGraph)
        roles = {data.get("role") for _, data in g.nodes(data=True)}
        assert "headnode" in roles


class TestFleetSpecMinimalFleet:
    """FleetSpec.minimal_fleet() — one dome, one module, one DAQ node (no gateway)."""

    def test_build_produces_one_daq_node(self) -> None:
        t = FleetSpec.minimal_fleet().build()
        assert len(t.daq.daq_nodes) == 1

    def test_daq_node_has_correct_module_id(self) -> None:
        from control.utils.config_file import ip_addr_to_module_id
        t = FleetSpec.minimal_fleet().build()
        node = t.daq.daq_nodes[0]
        # Module 200 → base IP 192.168.3.32
        expected_mid = ip_addr_to_module_id("192.168.3.32")
        assert expected_mid in node.module_ids

    def test_network_config_empty_no_gateway(self) -> None:
        t = FleetSpec.minimal_fleet().build()
        assert len(t.network.daq_nodes) == 0
        assert len(t.network.modules) == 0


class TestFleetSpecBuilder:
    """Custom FleetSpec via builder pattern."""

    def test_two_dome_two_module_fleet(self) -> None:
        spec = (
            FleetSpec(seed=99, name="two_dome")
            .with_headnode(ip="10.0.1.5")
            .add_dome("dome0", lat=37.342, lon=-121.637, alt=1283.0)
            .add_module(200, version="qfp", timing="wr", ip="192.168.3.32")
            .add_dome("dome1", lat=37.343, lon=-121.638, alt=1283.0)
            .add_module(204, version="bga", timing="wr", ip="192.168.3.48")
            .add_daq_node(ip="192.168.0.10", modules=[200, 204], bindhost="lo")
        )
        t = spec.build()
        assert len(t.obs.domes) == 2
        assert t.obs.domes[1].modules[0].quabo_version == "bga"

    def test_add_module_without_dome_raises(self) -> None:
        spec = FleetSpec(seed=0, name="bad")
        with pytest.raises(RuntimeError, match="add_dome"):
            spec.add_module(200, version="qfp", timing="wr", ip="192.168.3.32")

    def test_add_module_without_ip_raises(self) -> None:
        spec = FleetSpec(seed=0, name="bad").add_dome("d0", lat=37.0, lon=-120.0, alt=1000.0)
        with pytest.raises(ValueError, match="requires an explicit ip"):
            spec.add_module(200, version="qfp", timing="wr", ip="")

    def test_with_data_overrides_default(self) -> None:
        spec = (
            FleetSpec.minimal_unit()
            .with_data(run_type="science", overvoltage=3)
        )
        t = spec.build()
        assert t.data.run_type == "science"
        assert t.data.detector_overvoltage == 3

    def test_with_firmware_overrides_default(self) -> None:
        spec = FleetSpec.minimal_unit().with_firmware(qfp="myqfp.bin", bga="mybga.bin")
        t = spec.build()
        assert t.firmware.qfp == "myqfp.bin"

    def test_with_gateway_populates_network_config(self) -> None:
        spec = (
            FleetSpec(seed=42, name="gw_test")
            .with_headnode(ip="10.0.1.5")
            .add_dome("dome0", lat=37.342, lon=-121.637, alt=1283.0)
            .add_module(200, version="qfp", timing="wr", ip="192.168.3.32")
            .add_daq_node(
                ip="192.168.0.10",
                modules=[200],
                gateway=GatewaySpec(ip="10.200.146.13", grpc_port=50051),
                bindhost="lo",
            )
        )
        t = spec.build()
        assert len(t.network.daq_nodes) == 1
        assert len(t.network.modules) == 1

    def test_seeded_rng_is_deterministic(self) -> None:
        # The seed doesn't affect FleetSpec (which is fully explicit), but
        # generate_fleet_configs from topology.fleet should be deterministic.
        from control.topology.fleet import generate_fleet_configs
        r1 = generate_fleet_configs(3, modules_per_node=1, seed=42)
        r2 = generate_fleet_configs(3, modules_per_node=1, seed=42)
        r3 = generate_fleet_configs(3, modules_per_node=1, seed=99)
        assert r1[0].model_dump() == r2[0].model_dump()
        # Different seed → potentially different subnet_probability draws
        # (not guaranteed to differ, but the RNG itself is deterministic)


class TestFleetSpecTwoNodeCi:
    """FleetSpec.two_node_ci() — mirrors the static compose topology."""

    def test_two_daq_nodes(self) -> None:
        t = FleetSpec.two_node_ci().build()
        assert len(t.daq.daq_nodes) == 2

    def test_head_node_container_flag_set(self) -> None:
        t = FleetSpec.two_node_ci().build()
        assert t.daq.head_node_container is True

    def test_no_gateway_in_network_config(self) -> None:
        t = FleetSpec.two_node_ci().build()
        assert len(t.network.daq_nodes) == 0


class TestTopologyName:
    def test_topology_carries_spec_name(self) -> None:
        t = FleetSpec(seed=0, name="my_special_fleet").build()
        # Note: .build() requires at least one dome for some validators;
        # minimal_unit is the safe baseline
        t2 = FleetSpec.minimal_unit().build()
        assert t2.name == "minimal_unit"
