# mypy: ignore-errors
"""
test_config_validation.py — Config validation tests for v2.

Ported from ci/software_only/tier2_logic/test_config_validation.py.

Two complementary angles:
 1. pseti_workspace-generated configs — GlobalConfigValidator must pass against
    the Pydantic objects already validated at build() time. Tests confirm the
    JSON round-trip doesn't lose information.
 2. Static CI fixture configs (direct/gateway) — regression guard that the
    checked-in JSON files remain structurally valid as the code evolves.
"""

from __future__ import annotations

import copy
import json

import pytest

from ci.paths import PanoPathsTest
from ci.software_only_v2.infra.spec import FleetSpec, GatewaySpec
from ci.software_only_v2.infra.workspace import Workspace

INTEGRATION_CONFIGS = PanoPathsTest.integration_configs_root()


# ---------------------------------------------------------------------------
# Workspace-generated config validation
# ---------------------------------------------------------------------------

class TestWorkspaceConfigValidation:
    """Configs produced by pseti_workspace must pass GlobalConfigValidator."""

    def _validate_topology(self, workspace: Workspace) -> list:
        """Run GlobalConfigValidator over topology objects; return any ERRORs."""
        from control.utils.global_validator import GlobalConfigValidator
        t = workspace.topology
        v = GlobalConfigValidator({
            "obs": t.obs,
            "data": t.data,
            "daq": copy.deepcopy(t.daq),
            "network": t.network,
            "firmware": None,
            "uids": copy.deepcopy(t.quabo_uids),
        })
        v.validate_all_rules()
        return [r for r in v.report.tests if r["status"] == "ERROR"]

    def test_minimal_unit_workspace_configs_validate(
        self, pseti_workspace: Workspace
    ) -> None:
        from ci.software_only_v2.infra.parity import run_scenario
        run_scenario("config_validator_passes", topology=pseti_workspace.topology)

    def test_workspace_seven_config_files(
        self, pseti_workspace: Workspace
    ) -> None:
        from ci.software_only_v2.infra.parity import run_scenario
        expected = [
            "obs_config.json", "daq_config.json", "network_config.json",
            "data_config.json", "firmware.json", "quabo_uids.json", "daemons.json",
        ]
        run_scenario("workspace_seven_config_files", 
                     config_dir=pseti_workspace.config_dir, 
                     expected_files=expected)

    def test_workspace_json_files_all_readable(
        self, pseti_workspace: Workspace
    ) -> None:
        expected = [
            "obs_config.json", "daq_config.json", "network_config.json",
            "data_config.json", "firmware.json", "quabo_uids.json", "daemons.json",
        ]
        for fname in expected:
            path = pseti_workspace.config_dir / fname
            assert path.exists(), f"Missing config: {fname}"
            obj = json.loads(path.read_text())
            assert isinstance(obj, dict), f"{fname} must be a JSON object"

    @pytest.mark.parametrize(
        "pseti_workspace",
        [FleetSpec.two_node_ci()],
        indirect=True,
    )
    def test_two_node_ci_workspace_validates(
        self, pseti_workspace: Workspace
    ) -> None:
        errors = self._validate_topology(pseti_workspace)
        assert not errors, f"two_node_ci workspace configs have ERRORs: {errors}"

    @pytest.mark.parametrize(
        "pseti_workspace",
        [
            FleetSpec(seed=77, name="gw_val_test")
            .with_headnode(ip="10.0.1.5")
            .add_dome("d0", lat=37.342, lon=-121.637, alt=1283.0)
            .add_module(200, version="qfp", timing="wr", ip="192.168.3.32")
            .add_daq_node(
                ip="192.168.0.10",
                modules=[200],
                gateway=GatewaySpec(ip="10.200.146.13", grpc_port=50051),
                bindhost="lo",
            )
        ],
        indirect=True,
    )
    def test_gateway_workspace_network_config_has_daq_node(
        self, pseti_workspace: Workspace
    ) -> None:
        raw = pseti_workspace.config_as_dict("network_config.json")
        assert len(raw.get("daq_nodes", [])) == 1
        pf = raw["daq_nodes"][0]["port_forwarding"]
        assert pf["status"] is True
        assert 1 <= pf["grpc_port"] <= 65535

    def test_overvoltage_written_to_both_obs_and_data(
        self, pseti_workspace: Workspace
    ) -> None:
        obs_raw = pseti_workspace.config_as_dict("obs_config.json")
        data_raw = pseti_workspace.config_as_dict("data_config.json")
        assert obs_raw["detector_overvoltage"] == data_raw["detector_overvoltage"]

    def test_with_data_overrides_written_correctly(self) -> None:
        spec = FleetSpec.minimal_unit().with_data(run_type="science", overvoltage=3)
        t = spec.build()
        assert t.data.run_type == "science"
        assert t.data.detector_overvoltage == 3
        assert t.obs.detector_overvoltage == 3


# ---------------------------------------------------------------------------
# Static CI fixture configs (structural regression guard)
# ---------------------------------------------------------------------------

class TestStaticCiConfigs:
    """The checked-in CI fixture configs must remain structurally valid."""

    def test_direct_daq_config_parses(self) -> None:
        from control.utils.pydantic_config_models import DaqConfig
        dc = json.loads((INTEGRATION_CONFIGS / "direct" / "daq_config.json").read_text())
        model = DaqConfig(**dc)
        assert len(model.daq_nodes) >= 1

    def test_direct_daq_config_head_node_container(self) -> None:
        dc = json.loads((INTEGRATION_CONFIGS / "direct" / "daq_config.json").read_text())
        assert dc.get("head_node_container") is True

    def test_gateway_network_config_has_grpc_port(self) -> None:
        nc = json.loads(
            (INTEGRATION_CONFIGS / "gateway" / "network_config.json").read_text()
        )
        daq_nodes = nc.get("daq_nodes", [])
        assert daq_nodes, "Gateway network_config has no daq_nodes"
        pf = daq_nodes[0].get("port_forwarding", {})
        assert pf.get("status") is True
        assert "grpc_port" in pf, "gateway network_config must specify grpc_port"
        assert 1 <= pf["grpc_port"] <= 65535

    def test_gateway_network_config_parses(self) -> None:
        from control.utils.pydantic_config_models import NetworkConfig
        nc = json.loads(
            (INTEGRATION_CONFIGS / "gateway" / "network_config.json").read_text()
        )
        model = NetworkConfig(**nc)
        assert len(model.daq_nodes) >= 1
        assert len(model.modules) >= 1

    def test_direct_obs_config_parses(self) -> None:
        from control.utils.pydantic_config_models import ObsConfig
        oc = json.loads((INTEGRATION_CONFIGS / "direct" / "obs_config.json").read_text())
        model = ObsConfig(**oc)
        assert len(model.domes) >= 1
        assert len(model.domes[0].modules) >= 1
