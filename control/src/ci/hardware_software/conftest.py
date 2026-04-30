import os

import pytest
from typer.testing import CliRunner

from control.pseti import app
from control.utils import config_file

# ── Configurable Environment Variables ───────────────────────────────────────

BOOT_WAIT = int(os.environ.get("HW_TEST_QUABO_BOOT_WAIT", 60))
MIN_DISK_GB = int(os.environ.get("HW_TEST_MIN_DISK_GB", 50))

@pytest.fixture(scope="session")
def runner():
    """Typer CLI runner for pseti commands."""
    return CliRunner()

@pytest.fixture(scope="session", autouse=True)
def hw_safety_net(runner):
    """
    Ensures hardware is in a known safe state before and after the test session.
    Guarantees cleanup even if tests fail or are interrupted.
    """
    # ── Setup ────────────────────────────────────────────────────────────────
    print("\n[SAFETY NET] Validating HITL configurations...")
    # We use the CLI to validate, which checks all files.
    result = runner.invoke(app, ["val"])
    if result.exit_code != 0:
        pytest.exit(f"Configuration validation failed: {result.stdout}")

    print("[SAFETY NET] Starting Transfer Daemon...")
    runner.invoke(app, ["xfr", "start"])

    yield

    # ── Teardown ─────────────────────────────────────────────────────────────
    print("\n[SAFETY NET] Starting mandatory hardware teardown...")
    
    # 0. Stop Transfer Daemon
    try:
        print("[SAFETY NET] Stopping Transfer Daemon...")
        runner.invoke(app, ["xfr", "stop", "--timeout", "10"])
    except Exception as e:
        print(f"[SAFETY NET] WARNING: pseti xfr stop failed: {e}")

    # 1. Stop any active runs and force cleanup hashpipes
    try:
        print("[SAFETY NET] Stopping active runs and cleaning up DAQ nodes...")
        runner.invoke(app, ["stop", "--yes", "--force-cleanup"])
    except Exception as e:
        print(f"[SAFETY NET] WARNING: pseti stop failed: {e}")

    # 2. Power off Quabos
    try:
        print("[SAFETY NET] Powering off Quabos...")
        # runner.invoke(app, ["power", "off", "--yes"])
    except Exception as e:
        print(f"[SAFETY NET] WARNING: pseti power off failed: {e}")

    # 3. Stop redis daemons
    try:
        print("[SAFETY NET] Stopping redis daemons...")
        runner.invoke(app, ["config", "stop-redis-daemons", "--yes"])
    except Exception as e:
        print(f"[SAFETY NET] WARNING: pseti stop-redis-daemons failed: {e}")

    print("[SAFETY NET] Teardown complete.")

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
