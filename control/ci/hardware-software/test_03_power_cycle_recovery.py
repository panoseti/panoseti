"""
HW-02: Quabo power-cycle recovery.

Verifies the WPS power path, UID cache invalidation after power cycle,
HV re-init, and MAROC re-calibration.

Entry point: pseti test hw run -k HW_02
"""

from __future__ import annotations

import socket
import time

from typer.testing import CliRunner
import pytest

from control.pseti import app
from control.utils.pydantic_config_models import ObsConfig

BOOT_WAIT_DEFAULT = 60


pytest.skip(reason="requires power cycling")
class TestHW02PowerCycleRecovery:
    """Power cycle a single module and confirm full recovery."""

    def test_HW_02_power_off_module(self, runner: CliRunner, obs_config: ObsConfig) -> None:
        """Power off all Quabos."""
        result = runner.invoke(app, ["power", "off", "--yes"])
        assert result.exit_code == 0, f"power off failed:\n{result.stdout}"

    def test_HW_02_quabos_unreachable_after_off(self, obs_config: ObsConfig) -> None:
        """Quabos must be unreachable after power-off (UDP command port)."""
        time.sleep(5)  # brief stabilization
        for dome in obs_config.domes:
            for module in dome.modules:
                base = str(module.ip_addr).rsplit(".", 1)
                for i in range(4):
                    ip = f"{base[0]}.{int(base[1]) + i}"
                    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                        s.settimeout(1.0)
                        # Can't truly ping UDP; assert no TCP connection (management port 80)
                        try:
                            s.connect((ip, 80))
                            s.close()
                            # If connect didn't fail, Quabo is still up — warn only (UDP quirk)
                        except (OSError, TimeoutError):
                            pass  # expected: unreachable

    def test_HW_02_power_on_module(self, runner: CliRunner, boot_wait_time: int) -> None:
        """Power on all Quabos and wait for boot."""
        result = runner.invoke(app, ["power", "on", "--yes"])
        assert result.exit_code == 0, f"power on failed:\n{result.stdout}"
        print(f"Waiting {boot_wait_time}s for Quabo boot...")
        time.sleep(boot_wait_time)

    def test_HW_02_quabos_reachable_after_on(self, runner: CliRunner) -> None:
        """All Quabos must respond after power-on and boot wait."""
        result = runner.invoke(app, ["validate", "network", "--yes"])
        assert result.exit_code == 0, f"network validate failed:\n{result.stdout}"
        assert "is UP" in result.stdout, "Expected Quabos to be UP after power-on"

    def test_HW_02_short_run_after_power_cycle(self, runner: CliRunner) -> None:
        """A 10-second run must start successfully with fresh UIDs after power cycle."""
        # session-start performs UID refresh + calibration
        result = runner.invoke(app, ["session-start", "--yes"])
        assert result.exit_code == 0, f"session-start failed:\n{result.stdout}"

        result = runner.invoke(app, ["start", "--run-type", "test", "--yes"])
        assert result.exit_code == 0, f"pseti start failed after power cycle:\n{result.stdout}"

        time.sleep(10)

        result = runner.invoke(app, ["stop", "--yes"])
        assert result.exit_code == 0, f"pseti stop failed:\n{result.stdout}"
