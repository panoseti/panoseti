import os
import time
import pytest
import json
import shutil
from pathlib import Path
from typer.testing import CliRunner
from control.pseti import app
from control.utils.paths import PanoPaths

# ── Configurable Environment Variables ───────────────────────────────────────

BOOT_WAIT = int(os.environ.get("HW_TEST_QUABO_BOOT_WAIT", 60))
MIN_DISK_GB = int(os.environ.get("HW_TEST_MIN_DISK_GB", 50))
HEADNODE_IP = "192.168.88.103"
DAQ_NODE_IP = "192.168.0.228"
GRPC_PORT = 50051

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
    result = runner.invoke(app, ["validate", "--yes"])
    if result.exit_code != 0:
        pytest.exit(f"Configuration validation failed: {result.stdout}")

    yield

    # ── Teardown ─────────────────────────────────────────────────────────────
    print("\n[SAFETY NET] Starting mandatory hardware teardown...")
    
    # 1. Stop any active runs and force cleanup hashpipes
    try:
        print("[SAFETY NET] Stopping active runs and cleaning up DAQ nodes...")
        runner.invoke(app, ["stop", "--yes", "--force-cleanup"])
    except Exception as e:
        print(f"[SAFETY NET] WARNING: pseti stop failed: {e}")

    # 2. Power off Quabos
    try:
        print("[SAFETY NET] Powering off Quabos...")
        runner.invoke(app, ["power", "off", "--yes"])
    except Exception as e:
        print(f"[SAFETY NET] WARNING: pseti power off failed: {e}")

    # 3. Stop redis daemons
    try:
        print("[SAFETY NET] Stopping redis daemons...")
        runner.invoke(app, ["config", "stop-redis-daemons", "--yes"])
    except Exception as e:
        print(f"[SAFETY NET] WARNING: pseti stop-redis-daemons failed: {e}")

    print("[SAFETY NET] Teardown complete.")

@pytest.fixture
def daq_config():
    """Load the current DAQ configuration."""
    cfg_path = PanoPaths.config_dir() / "daq_config.json"
    with open(cfg_path, "r") as f:
        return json.load(f)

@pytest.fixture
def boot_wait_time():
    return BOOT_WAIT

@pytest.fixture
def min_disk_gb():
    return MIN_DISK_GB
