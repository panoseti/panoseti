"""
test_telemetry.py — Telemetry and logging subsystem logic.

Ported from ci/software_only/tier2_logic/test_telemetry.py.
"""

from __future__ import annotations

import json
import os
import pathlib

from panoseti_grpc.telemetry.logger import get_logger


def test_when_logger_called_then_jsonl_output_is_valid(tmp_path: pathlib.Path) -> None:
    """
    Verify that the unified logger produces correctly formatted JSONL.
    get_logger() appends a per-host subdirectory by default (Phase 1),
    so we glob for the file rather than hard-coding the path.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    logger = get_logger("telemetry_test", log_dir=log_dir)
    logger.info("Test message", extra={"run_id": "test_123"})

    # The file lands at log_dir/<hostname>/telemetry_test.jsonl
    matches = list(log_dir.rglob("telemetry_test.jsonl"))
    assert matches, f"telemetry_test.jsonl not found under {log_dir}"
    jsonl_path = matches[0]
    assert jsonl_path.exists()
    
    lines = jsonl_path.read_text().strip().split('\n')
    assert len(lines) == 1
    
    entry = json.loads(lines[0])
    assert entry["message"] == "Test message"
    assert entry["service"] == "telemetry_test"
    assert entry["run_id"] == "test_123"
    assert "timestamp" in entry

def test_when_auto_isolated_then_loki_tenant_id_injected() -> None:
    """
    Verify that the v2 root configuration correctly sets the Loki tenant ID.
    """
    tenant = os.environ.get("LOKI_TENANT_ID")
    assert tenant is not None
    assert tenant.startswith("v2_test_tenant_")
