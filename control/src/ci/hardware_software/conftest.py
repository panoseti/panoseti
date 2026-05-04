"""
HITL conftest — session fixtures for hardware-in-the-loop tests.

Unlike software-only/conftest.py, this file NEVER rewrites PSETI_* paths;
production paths must be preserved so quabo_uids.json, firmware/, and
/mnt/panoseti-test/ remain accessible.
"""

# Exclude archived test suites from collection.
# Tests in _archived/ are preserved for future promotion but not collected.
collect_ignore_glob = ["suites/_archived/*"]

# Re-export topology so it's discoverable by both suites/ and core_happy_path/.
from ci.hardware_software.fixtures.quabo_fixtures import topology  # noqa: F401

import logging
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from control.pseti import app
from control.utils import config_file

logger = logging.getLogger(__name__)

# ── Configurable Environment Variables ───────────────────────────────────────

BOOT_WAIT = int(os.environ.get("HW_TEST_QUABO_BOOT_WAIT", 60))
MIN_DISK_GB = int(os.environ.get("HW_TEST_MIN_DISK_GB", 50))

# State file persists hardware state across test runs / crashes.
_STATE_FILE = Path.home() / ".pseti" / "hw_runtime_state.json"


@pytest.fixture(scope="session")
def runner():
    """Typer CLI runner for pseti commands."""
    return CliRunner()


@pytest.fixture(scope="session", autouse=True)
def hw_safety_net(runner):
    """
    Registers the SafetyManager before any test runs and validates configs.

    The SafetyManager installs atexit + SIGTERM/SIGINT handlers that drive
    hardware to UNPOWERED on any exit path, including panics and OOM kills.
    It operates via raw subprocess calls, independent of pytest fixtures.
    """
    from ci.hardware_software.hw_utils.safety import SafetyManager
    from ci.hardware_software.hw_utils.state_machine import HardwareStateMachine

    # ── Setup ────────────────────────────────────────────────────────────────
    logger.info("[SAFETY NET] Validating HITL configurations...")
    result = runner.invoke(app, ["val"])
    if result.exit_code != 0:
        pytest.exit(f"Configuration validation failed:\n{result.stdout}")

    keep_running = os.environ.get("HW_KEEP_RUNNING", "").lower() in ("1", "true", "yes")
    sm = HardwareStateMachine()
    safety = SafetyManager(sm, _STATE_FILE, keep_running=keep_running)
    safety.register()

    logger.info("[SAFETY NET] Starting Transfer Daemon...")
    runner.invoke(app, ["xfr", "start"])

    yield

    # ── Teardown (also covered by SafetyManager atexit) ──────────────────────
    logger.info("[SAFETY NET] Starting mandatory hardware teardown...")

    try:
        logger.info("[SAFETY NET] Stopping Transfer Daemon...")
        runner.invoke(app, ["xfr", "stop", "--timeout", "10"])
    except Exception as exc:
        logger.warning("[SAFETY NET] pseti xfr stop failed: %s", exc)

    try:
        logger.info("[SAFETY NET] Stopping active runs...")
        runner.invoke(app, ["stop", "--yes", "--force-cleanup"])
    except Exception as exc:
        logger.warning("[SAFETY NET] pseti stop failed: %s", exc)

    try:
        logger.info("[SAFETY NET] Powering off Quabos...")
        runner.invoke(app, ["power", "off", "--yes"])
    except Exception as exc:
        logger.warning("[SAFETY NET] pseti power off failed: %s", exc)

    try:
        logger.info("[SAFETY NET] Stopping redis daemons...")
        runner.invoke(app, ["cfg", "stop-redis-daemons"])
    except Exception as exc:
        logger.warning("[SAFETY NET] pseti stop-redis-daemons failed: %s", exc)

    logger.info("[SAFETY NET] Teardown complete.")


# ── Config fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def topology_reachable(obs_config, network_config):
    """
    Session-scoped reachability check. 
    Quickly verifies every quabo responds to UDP echo through the gateway.
    """
    from ci.hardware_software.hw_utils.topology import HwTopology
    from control.driver.quabo_driver import QUABO
    
    topo = HwTopology()
    addrs = topo.quabo_ips()
    
    errors = []
    for a in addrs:
        q = QUABO(a.real_ip, port=a.cmd_port)
        try:
            cmd = q.make_cmd(0x01)
            q.send(cmd)
            q.sock.settimeout(1.0)
            data, _ = q.sock.recvfrom(1024)
            if not data:
                errors.append(f"Quabo {a.ip} (loc={a.boardloc}) at {a.real_ip}:{a.cmd_port} returned no data")
        except (TimeoutError, OSError) as exc:
            errors.append(f"Quabo {a.ip} (loc={a.boardloc}) at {a.real_ip}:{a.cmd_port} unreachable: {exc}")
        finally:
            q.close()
            
    if errors:
        return False, "\n".join(errors)
    return True, "All quabos reachable"


@pytest.fixture(scope="session")
def daq_config():
    """Load the validated DAQ configuration model."""
    return config_file.get_daq_config()


@pytest.fixture(scope="session")
def obs_config():
    """Load the validated Observatory configuration model."""
    return config_file.get_obs_config()


@pytest.fixture(scope="session")
def network_config():
    """Load the validated Network configuration model."""
    return config_file.get_network_config()


@pytest.fixture(scope="session")
def quabo_uids():
    """Load the validated Quabo UIDs model."""
    return config_file.get_quabo_uids()


@pytest.fixture
def boot_wait_time():
    return BOOT_WAIT


@pytest.fixture
def min_disk_gb():
    return MIN_DISK_GB
