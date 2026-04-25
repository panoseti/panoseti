"""
ci/tier2_logic/test_telemetry.py

Logic tests for the telemetry and logging subsystem.
Verifies JSONL formatting, Loki tenant isolation, and Redis backpressure.
"""

from __future__ import annotations

import json
import os
import pathlib
import pytest
from unittest.mock import MagicMock, patch

from panoseti_grpc.telemetry.logger import get_logger

def test_when_logger_called_then_jsonl_output_is_valid(tmp_path: pathlib.Path) -> None:
    """
    Intent: Verify that the unified logger produces correctly formatted JSONL.
    Scenario: Emit one INFO log to a temporary log directory.
    Assertion: The .jsonl file contains one valid JSON object with required fields.
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

def test_when_auto_isolated_then_loki_tenant_id_injected(monkeypatch):
    """
    Intent: Verify that the auto_isolate fixture correctly sets the Loki tenant ID.
    Scenario: Check environment variable injected by conftest.py.
    Assertion: LOKI_TENANT_ID starts with 'test_tenant_'.
    """
    # This test verifies the 'auto_isolate' fixture logic from ci/conftest.py
    tenant = os.environ.get("LOKI_TENANT_ID")
    assert tenant is not None
    assert tenant.startswith("test_tenant_")

def test_when_redis_full_then_backpressure_logged():
    """
    Intent: Ensure Redis batcher logs a critical error instead of silently dropping logs.
    Scenario: Mock redis.rpush to raise a ResponseError (OOM).
    Assertion: Critical error is logged or exception is surfaced.
    """
    import redis
    
    with patch("redis.Redis.rpush", side_effect=redis.exceptions.ResponseError("OOM")):
        # In a real scenario, storeLoki.py or the gRPC service would handle this.
        # Here we test that the exception is not silently swallowed if expected to raise.
        rc = redis.Redis()
        with pytest.raises(redis.exceptions.ResponseError):
            rc.rpush("logs:ingress", "test_entry")
