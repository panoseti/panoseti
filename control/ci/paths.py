from __future__ import annotations

from pathlib import Path
from typing import Literal

from control.utils.paths import PanoPaths


class PanoPathsTest(PanoPaths):
    """
    CI-specific path resolution for PANOSETI tests.
    Inherits production PanoPaths and adds test-only directory lookups.
    """

    @classmethod
    def ci_root(cls) -> Path:
        """The control/ci directory."""
        return cls.base_dir() / "ci"

    @classmethod
    def integration_configs_root(cls) -> Path:
        """Root for all integration test configurations."""
        return cls.ci_root() / "fixtures" / "configs"

    @classmethod
    def test_state_root(cls) -> Path:
        """Root for transient test state (tmp, logs, etc)."""
        return cls.ci_root() / "test_state"

    @classmethod
    def integration_configs(cls, variant: str = "direct") -> Path:
        """Path to specific integration config variant (e.g. 'direct', 'gateway')."""
        return cls.integration_configs_root() / variant

    @classmethod
    def hw_sw_configs(cls) -> Path:
        """Path to hardware-software test configurations."""
        return cls.ci_root() / "hardware-software" / "configs"

    @classmethod
    def configs_root(cls) -> Path:
        """Production configs directory."""
        return cls.base_dir() / "configs"

    @classmethod
    def grpc_server_configs(cls, variant: Literal["headnode", "daqnode"]) -> Path:
        """Path to specific grpc server config variant (e.g. 'headnode', 'daqnode')."""
        return cls.integration_configs_root() / "grpc" / variant

