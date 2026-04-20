from __future__ import annotations

import pathlib

from control.utils.paths import PanoPaths


class PanoPathsTest(PanoPaths):
    """
    CI-specific path resolution for PANOSETI tests.
    Inherits production PanoPaths and adds test-only directory lookups.
    """

    @classmethod
    def ci_root(cls) -> pathlib.Path:
        """The control/ci directory."""
        return cls.base_dir() / "ci"

    @classmethod
    def integration_configs_root(cls) -> pathlib.Path:
        """Root for all integration test configurations."""
        return cls.ci_root() / "integration" / "configs"

    @classmethod
    def test_state_root(cls) -> pathlib.Path:
        """Root for transient test state (tmp, logs, etc)."""
        return cls.ci_root() / "test_state"

    @classmethod
    def integration_configs(cls, variant: str = "direct") -> pathlib.Path:
        """Path to specific integration config variant (e.g. 'direct', 'gateway')."""
        return cls.integration_configs_root() / variant
