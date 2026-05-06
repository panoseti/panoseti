"""
workspace.py — The unified pseti_workspace fixture for v2 tests.

Replaces all of v1's:
  - workspace_fixtures.mock_env
  - workspace_fixtures.mock_workspace
  - transfer_fixtures.isolated_transfer_env
  - chaos_fixtures.chaos_headnode_workspace
  - transfer_helpers.setup_isolated_integration_transfer_env

Usage in tests::

    def test_ledger(pseti_workspace):
        ws = pseti_workspace          # Workspace(root, topology, state_probe)
        ws.state_probe.assert_ledger_status("ARCHIVED", timeout=5)

Parametric usage (different topology per test)::

    @pytest.mark.parametrize(
        "pseti_workspace",
        [FleetSpec.minimal_unit(), FleetSpec.two_node_ci()],
        indirect=True,
    )
    def test_something(pseti_workspace):
        ...
"""

from __future__ import annotations

import importlib
import os
import pathlib
from collections.abc import Iterator
from typing import Any

import pytest

from ci.software_only_v2.infra.spec import FleetSpec, Topology
from ci.software_only_v2.infra.workspace import StateProbe, Workspace


def _setup_workspace(
    spec: FleetSpec,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Workspace:
    """Core implementation shared by function-scope and session-scope variants."""
    topology: Topology = spec.build()

    # 1. Create all subdirectories and set PSETI_* env overrides
    env_dirs: list[tuple[str, str]] = [
        ("PSETI_CONFIG", "configs"),
        ("PSETI_STATE", "state"),
        ("PSETI_TMP", "tmp"),
        ("PSETI_LOGS", "state/logs"),
        ("PSETI_QUABOS", "quabos"),
        ("PSETI_FIRMWARE", "firmware"),
        ("HEAD_DATA_DIR", "head_data"),
        ("DAQ_DATA_DIR", "daq_data"),
    ]
    for key, sub in env_dirs:
        path = tmp_path / sub
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o777)
        monkeypatch.setenv(key, str(path))

    # 2. Materialize all 7 config files into PSETI_CONFIG
    from control.utils.paths import PanoPaths
    from ci.software_only_v2.infra.materialize import write_all
    write_all(topology, PanoPaths.config_dir())

    # 3. Create the state/ directory tree
    PanoPaths.ensure_state_dirs()

    # 4. Reload config_file so cached loaders pick up the new PSETI_CONFIG
    from control.utils import config_file as _config_file_module
    importlib.reload(_config_file_module)

    return Workspace(
        root=tmp_path,
        topology=topology,
        state_probe=StateProbe(),
    )


@pytest.fixture
def pseti_workspace(
    request: pytest.FixtureRequest,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Workspace:
    """
    Function-scoped isolated test workspace.

    - Writes all 7 validated config files to a fresh tmp_path.
    - Sets PSETI_* env vars so PanoPaths resolves into that tmp_path.
    - Reloads config_file so all loaders use the new env.
    - Validates the topology via GlobalConfigValidator before yielding.

    Accepts an optional FleetSpec via indirect parametrization:
        @pytest.mark.parametrize("pseti_workspace", [FleetSpec(...)], indirect=True)
    Falls back to FleetSpec.minimal_unit() if not parametrized.
    """
    spec: FleetSpec = getattr(request, "param", FleetSpec.minimal_unit())
    return _setup_workspace(spec, tmp_path, monkeypatch)


@pytest.fixture(scope="session")
def pseti_workspace_session(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Workspace]:
    """
    Session-scoped shared workspace (read-only after setup).
    Use for session_fleet tests where the workspace must outlive individual tests.
    """
    spec: FleetSpec = getattr(request, "param", FleetSpec.minimal_fleet())
    tmp_path = tmp_path_factory.mktemp("session_workspace")

    # Session-scope can't use monkeypatch (function-scope), so we apply env overrides directly.
    env_dirs: list[tuple[str, str]] = [
        ("PSETI_CONFIG", "configs"),
        ("PSETI_STATE", "state"),
        ("PSETI_TMP", "tmp"),
        ("PSETI_LOGS", "state/logs"),
        ("PSETI_QUABOS", "quabos"),
        ("PSETI_FIRMWARE", "firmware"),
        ("HEAD_DATA_DIR", "head_data"),
        ("DAQ_DATA_DIR", "daq_data"),
    ]
    original_env: dict[str, str | None] = {}
    for key, sub in env_dirs:
        original_env[key] = os.environ.get(key)
        path = tmp_path / sub
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o777)
        os.environ[key] = str(path)

    topology = spec.build()

    from control.utils.paths import PanoPaths
    from ci.software_only_v2.infra.materialize import write_all
    from control.utils import config_file as _config_file_module
    write_all(topology, PanoPaths.config_dir())
    PanoPaths.ensure_state_dirs()
    importlib.reload(_config_file_module)

    workspace = Workspace(root=tmp_path, topology=topology, state_probe=StateProbe())

    yield workspace

    # Restore env
    for key, orig in original_env.items():
        if orig is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = orig
