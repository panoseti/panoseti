"""
test_pseti_commands.py — CLI command smoke tests for 'pseti val', 'pseti show', etc.

Ported from ci/software_only/tier2_logic/test_pseti_commands.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ci.software_only_v2.infra.workspace import Workspace
from control.pseti import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestPsetiVal:
    """Smoke tests for the 'pseti val' command group."""

    def test_when_pseti_val_runs_then_schema_and_global_checks_pass(
        self, pseti_workspace: Workspace, runner: CliRunner
    ) -> None:
        """pseti val: isolated workspace passes both schema and global validators."""
        result = runner.invoke(app, ["val"])
        assert result.exit_code == 0
        assert "✔ Tier-1 File Syntax & Schema Validation Passed." in result.output
        assert "✔ Tier-2 Global Cross-Config Validation Passed." in result.output

    def test_when_pseti_val_graph_runs_then_topology_files_are_written(
        self, pseti_workspace: Workspace, runner: CliRunner
    ) -> None:
        """pseti val graph: topology JSON and HTML files are written to PSETI_TMP."""
        result = runner.invoke(app, ["val", "graph"])
        assert result.exit_code == 0

        # pseti val graph writes to PSETI_TMP (not pseti_workspace.root/"tmp")
        pseti_tmp = Path(os.environ["PSETI_TMP"])
        assert (pseti_tmp / "topology.json").exists(), (
            f"topology.json missing from {pseti_tmp}"
        )
        assert (pseti_tmp / "topology.html").exists(), (
            f"topology.html missing from {pseti_tmp}"
        )

    def test_when_pseti_val_all_runs_then_all_checks_complete(
        self, pseti_workspace: Workspace, runner: CliRunner
    ) -> None:
        """pseti val all: schema, global, network, and graph checks all run."""
        from unittest.mock import AsyncMock, patch
        with (
            patch("control.utils.config_validator._check_reachability", return_value=(True, "OK")),
            patch(
                "control.start._quabo_reachability_report",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = runner.invoke(app, ["val", "all"])
            assert result.exit_code == 0


class TestPsetiShow:
    """Smoke tests for the 'pseti show' command group."""

    def test_when_pseti_show_paths_runs_then_env_keys_are_displayed(
        self, pseti_workspace: Workspace, runner: CliRunner
    ) -> None:
        """pseti show paths: PSETI_CONFIG and PSETI_STATE keys appear in output."""
        result = runner.invoke(app, ["show", "paths"])
        assert result.exit_code == 0
        # Check for env var key names — stable regardless of Rich line-wrapping
        assert "PSETI_CONFIG" in result.output
        assert "PSETI_STATE" in result.output

    def test_when_pseti_show_config_runs_then_topology_module_ids_are_shown(
        self, pseti_workspace: Workspace, runner: CliRunner, caplog: pytest.LogCaptureFixture
    ) -> None:
        """pseti cfg show: module ID from the topology appears in config output."""
        import logging
        with caplog.at_level(logging.INFO):
            result = runner.invoke(app, ["cfg", "show"])
        assert result.exit_code == 0
        # Derive the expected module ID from the materialized topology rather
        # than hard-coding it (minimal_unit uses module_id=200, not 0).
        expected_id = pseti_workspace.topology.daq.daq_nodes[0].module_ids[0]
        assert "module ID" in caplog.text
        assert f"module ID {expected_id}" in caplog.text
