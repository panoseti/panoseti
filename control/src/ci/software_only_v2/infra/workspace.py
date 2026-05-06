"""
workspace.py — Workspace dataclass and StateProbe for v2 tests.

Workspace is the unified handle that a test receives from pseti_workspace.
StateProbe provides assertion helpers over PanoPaths-resolved directories,
replacing the v1 state_probe.py which hard-coded env vars.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

from ci.software_only_v2.infra.spec import Topology


# ---------------------------------------------------------------------------
# StateProbe — rebuilt on PanoPaths (no hard-coded env vars)
# ---------------------------------------------------------------------------

class StateProbe:
    """
    Assertion helper for test state directories.

    All paths are resolved dynamically through PanoPaths class-methods, which
    respect the PSETI_* env overrides set by pseti_workspace. Import PanoPaths
    lazily so the fixture's monkeypatched env is already active when paths are
    resolved.
    """

    def ledger_path(self) -> pathlib.Path:
        from control.utils.paths import PanoPaths
        return PanoPaths.runs_dir() / "ledger.toml"

    def ledger_status(self) -> str | None:
        """Read the RunStatus from ledger.toml, or None if the file doesn't exist."""
        import tomllib
        path = self.ledger_path()
        if not path.exists():
            return None
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        return raw.get("status")

    def current_run_name(self) -> str | None:
        """Return the current run_name from ledger.toml."""
        import tomllib
        path = self.ledger_path()
        if not path.exists():
            return None
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        return raw.get("run_name")

    def any_pff_files(self, run_name: str, *, head: bool = True) -> bool:
        """Return True if at least one .pff file exists under the run directory."""
        from control.utils.paths import PanoPaths
        import os
        base = pathlib.Path(os.environ.get("HEAD_DATA_DIR", str(PanoPaths.state_dir()))) if head \
            else pathlib.Path(os.environ.get("DAQ_DATA_DIR", "/data"))
        run_dir = base / run_name
        if not run_dir.exists():
            return False
        return any(run_dir.rglob("*.pff"))

    def transfer_queue_dir(self) -> pathlib.Path:
        from control.utils.paths import PanoPaths
        return PanoPaths.transfer_queue_dir()

    def transfer_manifests_dir(self) -> pathlib.Path:
        from control.utils.paths import PanoPaths
        return PanoPaths.transfer_manifests_dir()

    def pending_jobs(self) -> list[pathlib.Path]:
        return list((self.transfer_queue_dir() / "pending").glob("*.job.toml"))

    def completed_jobs(self) -> list[pathlib.Path]:
        return list((self.transfer_queue_dir() / "completed").glob("*.job.toml"))

    def failed_jobs(self) -> list[pathlib.Path]:
        return list((self.transfer_queue_dir() / "failed").glob("*.job.toml"))

    def config_dir(self) -> pathlib.Path:
        from control.utils.paths import PanoPaths
        return PanoPaths.config_dir()

    def runs_dir(self) -> pathlib.Path:
        from control.utils.paths import PanoPaths
        return PanoPaths.runs_dir()

    def locks_dir(self) -> pathlib.Path:
        from control.utils.paths import PanoPaths
        return PanoPaths.locks_dir()

    def assert_ledger_status(self, expected: str, *, timeout: float = 0.0) -> None:
        """Assert ledger status equals expected; optionally poll for up to timeout seconds."""
        import time
        if timeout > 0:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.ledger_status() == expected:
                    return
                time.sleep(0.25)
        actual = self.ledger_status()
        assert actual == expected, f"Ledger status: expected {expected!r}, got {actual!r}"

    def assert_no_locks(self) -> None:
        """Assert no stale lock files remain under the locks directory."""
        locks = list(self.locks_dir().glob("*.lock"))
        assert not locks, f"Unexpected lock files: {locks}"


# ---------------------------------------------------------------------------
# Workspace — the fixture handle
# ---------------------------------------------------------------------------

@dataclass
class Workspace:
    """
    Unified test workspace handle, yielded by the pseti_workspace fixture.

    Attributes:
        root:        The tmp_path root for this test.
        topology:    The validated Topology from FleetSpec.build().
        state_probe: Assertion helper backed by PanoPaths.
    """
    root: pathlib.Path
    topology: Topology
    state_probe: StateProbe

    @property
    def config_dir(self) -> pathlib.Path:
        return self.state_probe.config_dir()

    @property
    def runs_dir(self) -> pathlib.Path:
        return self.state_probe.runs_dir()

    @property
    def locks_dir(self) -> pathlib.Path:
        return self.state_probe.locks_dir()

    @property
    def transfer_queue_dir(self) -> pathlib.Path:
        return self.state_probe.transfer_queue_dir()

    @property
    def transfer_manifests_dir(self) -> pathlib.Path:
        return self.state_probe.transfer_manifests_dir()

    def reload_configs(self) -> None:
        """Force config_file module to re-read from disk (after env changes)."""
        import importlib
        from control.utils import config_file
        importlib.reload(config_file)

    def config_as_dict(self, filename: str) -> dict[str, Any]:
        """Read a config file back as a raw dict for assertion helpers."""
        import json
        path = self.config_dir / filename
        if not path.exists():
            return {}
        return json.loads(path.read_text())
