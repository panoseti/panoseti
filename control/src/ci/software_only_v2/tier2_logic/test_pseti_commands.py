# mypy: ignore-errors
"""
test_pseti_commands.py — CLI command smoke tests for 'pseti val', 'pseti show', etc.

Ported from ci/software_only/tier2_logic/test_pseti_commands.py.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from control.pseti import app
from ci.software_only_v2.infra.workspace import Workspace

@pytest.fixture
def runner():
    return CliRunner()


class TestPsetiVal:
    """Smoke tests for the 'pseti val' command group."""

    def test_pseti_val_basic(self, pseti_workspace: Workspace, runner: CliRunner) -> None:
        """Verify that pseti val runs without crashing and passes on a clean workspace."""
        # pseti_workspace ensures we have valid configs and isolation
        result = runner.invoke(app, ["val"])
        assert result.exit_code == 0
        assert "✔ Tier-1 File Syntax & Schema Validation Passed." in result.output
        assert "✔ Tier-2 Global Cross-Config Validation Passed." in result.output

    def test_pseti_val_graph(self, pseti_workspace: Workspace, runner: CliRunner) -> None:
        """Verify that pseti val graph generates the topology files."""
        result = runner.invoke(app, ["val", "graph"])
        assert result.exit_code == 0
        
        topology_json = pseti_workspace.root / "tmp" / "topology.json"
        topology_html = pseti_workspace.root / "tmp" / "topology.html"
        assert topology_json.exists()
        assert topology_html.exists()

    def test_pseti_val_all(self, pseti_workspace: Workspace, runner: CliRunner) -> None:
        """Verify pseti val all enables everything (schema, global, network, graph)."""
        # Note: network check will attempt to ping, but in the sim-only tier2
        # it should either be mocked or handled by the workspace.
        # GlobalConfigValidator.validate_all() has check_network=True for 'all'.
        result = runner.invoke(app, ["val", "all"])
        assert result.exit_code == 0


class TestPsetiShow:
    """Smoke tests for the 'pseti show' command group."""

    def test_pseti_show_paths(self, pseti_workspace: Workspace, runner: CliRunner) -> None:
        """Verify pseti show paths correctly reflects the isolated workspace."""
        result = runner.invoke(app, ["show", "paths"])
        assert result.exit_code == 0
        # The output should contain the temporary root of our isolated workspace
        assert str(pseti_workspace.root) in result.output
        assert "PSETI_CONFIG" in result.output
        assert "PSETI_STATE" in result.output

    def test_pseti_show_config(self, pseti_workspace: Workspace, runner: CliRunner) -> None:
        """Verify pseti show config displays the topology."""
        result = runner.invoke(app, ["cfg", "show"])
        assert result.exit_code == 0
        assert "module ID" in result.output
        # The minimal_unit has one module
        assert "module ID 0" in result.output
