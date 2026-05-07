"""
test_workspace.py — Unit tests for the pseti_workspace fixture.

Verifies that the fixture:
 - writes all 7 config files to PSETI_CONFIG
 - sets PSETI_* env vars pointing into tmp_path
 - creates the full state directory tree (runs/, locks/, transfer/queue/...)
 - exposes a valid Workspace handle with correct attributes
 - passes GlobalConfigValidator at build time
"""

import json
import os
import pathlib

import pytest

from ci.software_only_v2.infra.spec import FleetSpec
from ci.software_only_v2.infra.workspace import StateProbe, Workspace


EXPECTED_CONFIG_FILES = {
    "obs_config.json",
    "daq_config.json",
    "network_config.json",
    "data_config.json",
    "firmware.json",
    "quabo_uids.json",
    "daemons.json",
}


class TestWorkspaceSetup:
    """Verify pseti_workspace writes files and sets env correctly."""

    def test_workspace_is_workspace_instance(self, pseti_workspace: Workspace) -> None:
        assert isinstance(pseti_workspace, Workspace)

    def test_all_seven_config_files_exist(self, pseti_workspace: Workspace) -> None:
        config_dir = pseti_workspace.config_dir
        written = {f.name for f in config_dir.iterdir() if f.suffix == ".json"}
        assert EXPECTED_CONFIG_FILES.issubset(written)

    def test_config_files_are_valid_json(self, pseti_workspace: Workspace) -> None:
        config_dir = pseti_workspace.config_dir
        for fname in EXPECTED_CONFIG_FILES:
            path = config_dir / fname
            assert path.exists(), f"Missing: {fname}"
            obj = json.loads(path.read_text())
            assert isinstance(obj, dict), f"{fname} is not a JSON object"

    def test_pseti_config_env_points_into_tmp(
        self, pseti_workspace: Workspace, tmp_path: pathlib.Path
    ) -> None:
        config_env = pathlib.Path(os.environ["PSETI_CONFIG"])
        assert config_env == pseti_workspace.config_dir
        # Must be inside tmp_path
        assert str(config_env).startswith(str(tmp_path))

    def test_pseti_state_env_points_into_tmp(
        self, pseti_workspace: Workspace, tmp_path: pathlib.Path
    ) -> None:
        state_env = pathlib.Path(os.environ["PSETI_STATE"])
        assert str(state_env).startswith(str(tmp_path))

    def test_head_data_dir_env_set(
        self, pseti_workspace: Workspace, tmp_path: pathlib.Path
    ) -> None:
        head_data = pathlib.Path(os.environ["HEAD_DATA_DIR"])
        assert str(head_data).startswith(str(tmp_path))
        assert head_data.exists()

    def test_runs_dir_exists(self, pseti_workspace: Workspace) -> None:
        assert pseti_workspace.runs_dir.exists()

    def test_locks_dir_exists(self, pseti_workspace: Workspace) -> None:
        assert pseti_workspace.locks_dir.exists()

    def test_transfer_queue_dir_exists(self, pseti_workspace: Workspace) -> None:
        tq = pseti_workspace.transfer_queue_dir
        assert tq.exists()
        # All three sub-queues must be created
        for sub in ("pending", "active", "completed", "failed"):
            assert (tq / sub).exists(), f"Missing transfer queue subdir: {sub}"

    def test_topology_attached(self, pseti_workspace: Workspace) -> None:
        from ci.software_only_v2.infra.spec import Topology
        assert isinstance(pseti_workspace.topology, Topology)


class TestWorkspaceConfigContents:
    """Verify config file contents match the topology."""

    def test_obs_config_has_one_dome(self, pseti_workspace: Workspace) -> None:
        raw = pseti_workspace.config_as_dict("obs_config.json")
        assert len(raw["domes"]) == 1

    def test_daq_config_head_node_container_true(self, pseti_workspace: Workspace) -> None:
        raw = pseti_workspace.config_as_dict("daq_config.json")
        assert raw.get("head_node_container") is True

    def test_data_config_run_type(self, pseti_workspace: Workspace) -> None:
        raw = pseti_workspace.config_as_dict("data_config.json")
        assert raw.get("run_type") == "engineering"

    def test_firmware_config_has_qfp_and_bga(self, pseti_workspace: Workspace) -> None:
        raw = pseti_workspace.config_as_dict("firmware.json")
        assert raw.get("qfp") is not None
        assert raw.get("bga") is not None

    def test_daemons_all_disabled(self, pseti_workspace: Workspace) -> None:
        raw = pseti_workspace.config_as_dict("daemons.json")
        # All daemon flags should be False (safe test default)
        daemons = raw.get("daemons", {})
        for key, val in daemons.items():
            assert val is False, f"Daemon {key!r} is not disabled"

    def test_quabo_uids_has_four_quabos_per_module(self, pseti_workspace: Workspace) -> None:
        raw = pseti_workspace.config_as_dict("quabo_uids.json")
        first_dome = raw["domes"][0]
        first_module = first_dome["modules"][0]
        assert len(first_module["quabos"]) == 4


class TestStateProbe:
    """Verify StateProbe methods resolve correctly against the isolated workspace."""

    def test_ledger_status_is_none_initially(self, pseti_workspace: Workspace) -> None:
        assert pseti_workspace.state_probe.ledger_status() is None

    def test_no_pending_jobs_initially(self, pseti_workspace: Workspace) -> None:
        assert pseti_workspace.state_probe.pending_jobs() == []

    def test_no_completed_jobs_initially(self, pseti_workspace: Workspace) -> None:
        assert pseti_workspace.state_probe.completed_jobs() == []

    def test_assert_no_locks_passes_initially(self, pseti_workspace: Workspace) -> None:
        pseti_workspace.state_probe.assert_no_locks()


class TestWorkspaceParametric:
    """Verify pseti_workspace works with explicit FleetSpec via indirect param."""

    @pytest.mark.parametrize(
        "pseti_workspace",
        [FleetSpec.minimal_fleet()],
        indirect=True,
    )
    def test_minimal_fleet_has_one_daq_node(self, pseti_workspace: Workspace) -> None:
        raw = pseti_workspace.config_as_dict("daq_config.json")
        assert len(raw.get("daq_nodes", [])) == 1

    @pytest.mark.parametrize(
        "pseti_workspace",
        [FleetSpec.two_node_ci()],
        indirect=True,
    )
    def test_two_node_ci_has_two_daq_nodes(self, pseti_workspace: Workspace) -> None:
        raw = pseti_workspace.config_as_dict("daq_config.json")
        assert len(raw.get("daq_nodes", [])) == 2


class TestWorkspaceIsolation:
    """Verify different test invocations get isolated workspaces."""

    def test_config_dir_is_inside_root(self, pseti_workspace: Workspace) -> None:
        assert str(pseti_workspace.config_dir).startswith(str(pseti_workspace.root))

    def test_runs_dir_is_inside_root(self, pseti_workspace: Workspace) -> None:
        assert str(pseti_workspace.runs_dir).startswith(str(pseti_workspace.root))

    def test_reload_configs_does_not_raise(self, pseti_workspace: Workspace) -> None:
        # Should complete without error after env is set
        pseti_workspace.reload_configs()
