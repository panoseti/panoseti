"""
conftest.py for hw4_telemetry.
Bootstraps the capture_hk.py daemon.
"""

from __future__ import annotations

import subprocess
import time
import os
import signal
import pytest
import logging

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module", autouse=True)
def capture_hk_daemon():
    """
    Spawn capture_hk.py as a subprocess for the duration of the telemetry suite.
    """
    from control.utils.paths import PanoPaths
    capture_script = PanoPaths.control_dir() / "src" / "control" / "daemons" / "capture_hk.py"
    
    if not capture_script.exists():
        pytest.skip(f"capture_hk.py not found at {capture_script}")

    logger.info("Starting capture_hk.py daemon...")
    proc = subprocess.Popen(
        ["python3", str(capture_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    
    # Wait for it to bind
    time.sleep(2.0)
    if proc.poll() is not None:
        stdout, stderr = proc.communicate()
        pytest.fail(f"capture_hk.py failed to start:\nSTDOUT: {stdout.decode()}\nSTDERR: {stderr.decode()}")

    yield proc

    logger.info("Stopping capture_hk.py daemon...")
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    proc.wait(timeout=5.0)
