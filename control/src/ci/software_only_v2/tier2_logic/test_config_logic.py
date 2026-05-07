"""
test_config_logic.py — Cross-config invariant tests for GlobalConfigValidator.

Ported from ci/software_only/tier2_logic/test_config_logic.py.

Uses:
 - generate_palomar_topology() for realistic multi-site scenarios
 - FleetSpec-built topologies for targeted mutation tests

All tests exercise GlobalConfigValidator.validate_all_rules() directly so
error-case assertions can inspect individual rule results without triggering
the ValueError that validate_all() raises.
"""

from __future__ import annotations

import pytest

from control.topology.fleet import generate_palomar_topology
from control.utils.global_validator import GlobalConfigValidator

from ci.software_only_v2.infra.spec import FleetSpec


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def palomar():
    """Return (daq, uids, net, obs) for a realistic 4-site Palomar topology."""
    return generate_palomar_topology()


def _validator(daq=None, uids=None, net=None, obs=None, data=None) -> GlobalConfigValidator:
    return GlobalConfigValidator({
        "obs": obs, "data": data, "daq": daq,
        "network": net, "firmware": None, "uids": uids,
    })


# ---------------------------------------------------------------------------
# Palomar realistic topology
# ---------------------------------------------------------------------------

class TestPalomarTopology:
    """GlobalConfigValidator on the realistic Palomar 4-site topology."""

    def test_palomar_passes_all_invariants(self, palomar) -> None:
        daq, uids, net, obs = palomar
        v = _validator(daq=daq, uids=uids, net=net, obs=obs)
        assert v.validate_all_rules() is True

    def test_palomar_no_error_rules(self, palomar) -> None:
        daq, uids, net, obs = palomar
        v = _validator(daq=daq, uids=uids, net=net, obs=obs)
        v.validate_all_rules()
        errors = [t for t in v.report.tests if t["status"] == "ERROR"]
        assert not errors, f"Unexpected ERRORs: {errors}"


# ---------------------------------------------------------------------------
# Module ID / BOARDLOC collision
# ---------------------------------------------------------------------------

class TestModuleIdCollision:
    """BOARDLOC uniqueness must be enforced across domes."""

    def test_duplicate_module_ip_across_domes_rejected(self, palomar) -> None:
        daq, uids, net, obs = palomar
        # Inject a second module in PTI dome with the same IP as Gattini → same module_id
        gattini_ip = obs.domes[0].modules[0].ip_addr
        dup_mod = obs.domes[3].modules[0].model_copy(update={"ip_addr": gattini_ip})
        obs.domes[3].modules.append(dup_mod)
        uids.domes[0].modules.append(uids.domes[0].modules[0].model_copy())

        v = _validator(daq=daq, uids=uids, net=net, obs=obs)
        assert v.validate_all_rules() is False
        assert any(
            "collision" in t["info"].lower() or "module id" in t["info"].lower()
            for t in v.report.tests if t["status"] == "ERROR"
        )

    def test_fleetspec_two_modules_same_id_rejected(self) -> None:
        """Two modules with the same IP (same module_id) in different domes must fail validation."""
        from ipaddress import IPv4Address
        from control.utils.pydantic_config_models import (
            DaqConfig, DaqNode, NetworkConfig, ObsConfig, ObsDomeConfig,
            ObsModuleConfig, QuaboUidDome, QuaboUidEntry, QuaboUidModule, QuaboUids,
        )
        shared_ip = IPv4Address("192.168.3.32")
        obs = ObsConfig(
            name="collision_test",
            domes=[
                ObsDomeConfig(name="d0", obslat=37.0, obslon=-121.0, obsalt=1000.0,
                              modules=[ObsModuleConfig(mobo_serialno="A", quabo_version="qfp",
                                                       ip_addr=shared_ip, id=200)]),
                # Same IP → same module_id in d1 as d0
                ObsDomeConfig(name="d1", obslat=37.0, obslon=-121.001, obsalt=1000.0,
                              modules=[ObsModuleConfig(mobo_serialno="B", quabo_version="qfp",
                                                       ip_addr=shared_ip, id=200)]),
            ],
            detector_overvoltage=2,
        )
        obs.wps = {"url": "http://192.168.1.1", "quabo_socket": 4}  # type: ignore[assignment]
        daq = DaqConfig(
            head_node_data_dir="/data/head",
            head_node_ip_addr=IPv4Address("10.0.1.5"),
            head_node_container=True,
            daq_nodes=[DaqNode(username="u", data_dir="/d",
                               ip_addr=IPv4Address("192.168.0.10"), module_ids=[200])],
        )
        uids = QuaboUids(domes=[
            QuaboUidDome(num=0, modules=[
                QuaboUidModule(ip_addr=shared_ip,
                               quabos=[QuaboUidEntry(uid=f"q_{i}") for i in range(4)],
                               id=200),
            ])
        ])
        net = NetworkConfig()
        v = _validator(daq=daq, uids=uids, net=net, obs=obs)
        assert v.validate_all_rules() is False
        assert any(
            "collision" in t["info"].lower() or "module id" in t["info"].lower()
            for t in v.report.tests if t["status"] == "ERROR"
        )


# ---------------------------------------------------------------------------
# DAQ node overlap
# ---------------------------------------------------------------------------

class TestDaqNodeOverlap:
    """Two DAQ nodes cannot claim the same module_id."""

    def test_duplicate_module_id_across_daqnodes_rejected(self, palomar) -> None:
        daq, uids, net, obs = palomar
        # Gattini (node 0) handles module 1; make Winter (node 1) also claim it.
        daq.daq_nodes[1].module_ids.append(1)
        v = _validator(daq=daq, uids=uids, net=net, obs=obs)
        assert v.validate_all_rules() is False
        assert any(
            "multiple" in t["info"].lower() or "assigned" in t["info"].lower()
            for t in v.report.tests if t["status"] == "ERROR"
        )

    def test_fleetspec_no_overlap_by_construction(self) -> None:
        """FleetSpec.two_node_ci() topology must pass overlap validation."""
        from control.utils.global_validator import validate_all
        # FleetSpec.build() calls validate_all() internally; if it returns a Topology,
        # it passed — no further assertion needed beyond the call not raising.
        t = FleetSpec.two_node_ci().build()
        # Confirm the two nodes really have different module_ids
        ids0 = set(t.daq.daq_nodes[0].module_ids)
        ids1 = set(t.daq.daq_nodes[1].module_ids)
        assert ids0.isdisjoint(ids1)


# ---------------------------------------------------------------------------
# FleetSpec-built topology invariants
# ---------------------------------------------------------------------------

class TestFleetSpecTopologyInvariants:
    """Validate that FleetSpec-built topologies are clean by construction."""

    def test_minimal_unit_passes_validator(self) -> None:
        from control.utils.pydantic_config_models import DaqConfig, NetworkConfig, ObsConfig, QuaboUids
        t = FleetSpec.minimal_unit().build()
        assert isinstance(t.obs, ObsConfig)
        assert isinstance(t.daq, DaqConfig)
        assert isinstance(t.network, NetworkConfig)
        assert isinstance(t.quabo_uids, QuaboUids)

    def test_two_node_ci_no_errors(self) -> None:
        import copy
        t = FleetSpec.two_node_ci().build()
        # Build a fresh validator over the configs (deepcopy to avoid associate() mutation)
        v = _validator(
            daq=copy.deepcopy(t.daq),
            uids=copy.deepcopy(t.quabo_uids),
            net=copy.deepcopy(t.network),
            obs=copy.deepcopy(t.obs),
        )
        v.validate_all_rules()
        errors = [t for t in v.report.tests if t["status"] == "ERROR"]
        assert not errors, f"two_node_ci topology has ERRORs: {errors}"

    def test_gateway_topology_network_routing_is_valid(self) -> None:
        """A FleetSpec with a gateway must pass network tunneling validation."""
        from ci.software_only_v2.infra.spec import GatewaySpec
        import copy
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
        v = _validator(
            daq=copy.deepcopy(t.daq),
            uids=copy.deepcopy(t.quabo_uids),
            net=copy.deepcopy(t.network),
            obs=copy.deepcopy(t.obs),
        )
        v.validate_all_rules()
        errors = [r for r in v.report.tests if r["status"] == "ERROR"]
        assert not errors, f"Gateway topology has ERRORs: {errors}"
