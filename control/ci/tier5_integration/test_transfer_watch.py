"""Tier 5 (Integration): Transfer --watch smoke test.

Verifies:
- pseti transfer status --watch renders correctly.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time

import pytest


@pytest.mark.skipif(not os.environ.get("RUN_REAL_DATA_TESTS"), reason="RUN_REAL_DATA_TESTS not set")
def test_transfer_status_watch_smoke(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PSETI_STATE", str(tmp_path / "state"))
    # PSETI_TQ_DIR must point to the queue root, not state root
    tq_dir = tmp_path / "queue"
    monkeypatch.setenv("PSETI_TQ_DIR", str(tq_dir))
    
    active_d = tq_dir / "active"
    active_d.mkdir(parents=True)
    
    # Create a fake active job file
    (active_d / "r1.job.toml").write_text("run_name = 'r1'")
    
    # Create a fake sidecar
    progress_file = active_d / "r1.127.0.0.1.progress.json"
    with open(progress_file, "w") as f:
        json.dump({"bytes": 1000, "pct": 50, "speed": "10MB/s", "eta": "0:01"}, f)
    
    # Run the command with a short interval and timeout
    # We use subprocess to capture stdout of the watch loop
    cmd = ["pseti", "xfr", "stat", "--watch", "--interval", "0.5"]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=os.environ)
    
    # Give it some time to render a few frames
    time.sleep(2.0)
    proc.terminate()
    stdout, _stderr = proc.communicate()
    
    # Check if we saw the progress
    assert "r1" in stdout
    assert "50%" in stdout
    # Clean output might have clear codes, but the tokens should be there
