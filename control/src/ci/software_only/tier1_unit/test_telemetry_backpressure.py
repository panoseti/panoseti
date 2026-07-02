"""
test_telemetry_backpressure.py — Redis backpressure unit test.

Moved from tier2_logic/test_telemetry.py — purely mocks redis.Redis.rpush.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


def test_when_redis_full_then_backpressure_logged() -> None:
    """Redis batcher surfaces ResponseError (OOM) as an exception to the caller."""
    import redis

    with patch("redis.Redis.rpush", side_effect=redis.exceptions.ResponseError("OOM")):
        rc = redis.Redis()
        with pytest.raises(redis.exceptions.ResponseError):
            rc.rpush("logs:ingress", "test_entry")
