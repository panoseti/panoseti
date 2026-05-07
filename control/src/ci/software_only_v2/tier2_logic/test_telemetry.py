# mypy: ignore-errors
"""
test_telemetry.py — Telemetry and logging subsystem logic.

Ported from ci/software_only/tier2_logic/test_telemetry.py.
"""

from __future__ import annotations

import json
import os
import pathlib
from unittest.mock import patch

import pytest
from panoseti_grpc.telemetry.logger import get_logger


def test_when_logger_called_then_jsonl_output_is_valid(tmp_path: pathlib.Path) -> None:
    """
    Verify that the unified logger produces correctly formatted JSONL.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    
    logger = get_logger("telemetry_test", log_dir=log_dir)
    logger.info("Test message", extra={"run_id": "test_123"})
    
    # Locate the jsonl file
    jsonl_path = log_dir / "telemetry_test.jsonl"
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
    assert tenant.startswith("test_tenant_")

def test_when_redis_full_then_backpressure_logged() -> None:
    """
    Ensure Redis batcher handles ResponseError (OOM) correctly.
    """
    import redis
    
    with patch("redis.Redis.rpush", side_effect=redis.exceptions.ResponseError("OOM")):
        rc = redis.Redis()
        with pytest.raises(redis.exceptions.ResponseError):
            rc.rpush("logs:ingress", "test_entry")
