"""
test_transfer_queue_robustness.py — Unit tests for TransferQueue TOML serialization.
Ensures that multiline strings (like rsync errors) and special characters 
are correctly escaped and can be re-loaded without errors.
"""

import os
import pathlib
import tomllib
from datetime import datetime, UTC

import pytest

from control.transfer.queue import TransferQueue
from control.transfer.models import TransferJob, TransferNodeSpec


def test_transfer_queue_serialization_robustness(tmp_path: pathlib.Path) -> None:
    """
    Verify that the TransferQueue can serialize and deserialize a job
    containing complex strings (newlines, quotes, backslashes).
    """
    tq = TransferQueue(queue_dir=tmp_path)
    
    # 1. Create a job with "hostile" strings
    multiline_error = (
        "rsync: [sender] send_files failed to open \"/path/to/file\": Permission denied (13)\n"
        "rsync error: some files/attrs were not transferred (code 23) at main.c(1852)\n"
        "Backslash test: \\path\\to\\somewhere\n"
        "Quotes test: \"Hello World\""
    )
    
    node = TransferNodeSpec(
        ip_addr="192.168.0.1",
        username="panoseti",
        data_dir="/data",
        module_ids=[1, 2, 3]
    )
    
    job = TransferJob(
        run_name="test_robust_run",
        head_data_dir="/head/data",
        head_node_username="headuser",
        created_at=datetime.now(UTC),
        daq_nodes=[node],
        last_error=multiline_error
    )
    
    # 2. Enqueue (serializes to TOML)
    success = tq.enqueue(job)
    assert success is True
    
    # 3. Verify file content on disk (manual check for escapes)
    job_path = tq._job_path("pending", job.run_name)
    content = job_path.read_text()
    
    # Newlines should be escaped as \n
    assert "\\n" in content
    # Backslashes should be escaped as \\
    assert "\\\\" in content
    # Double quotes should be escaped as \"
    assert "\\\"Hello World\\\"" in content
    
    # 4. Claim (deserializes from TOML)
    claimed_job = tq.claim()
    assert claimed_job is not None
    assert claimed_job.run_name == job.run_name
    assert claimed_job.last_error == multiline_error
    
    # Double check it parses correctly with standard tomllib
    with open(tq._job_path("active", job.run_name), "rb") as f:
        data = tomllib.load(f)
        assert data["last_error"] == multiline_error
        assert data["daq_nodes"][0]["module_ids"] == [1, 2, 3]
