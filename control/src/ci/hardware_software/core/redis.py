"""
Redis HK helpers for HITL tests.

capture_hk.py writes quabo housekeeping data into Redis hashes keyed by
QUABO_{boardloc}.  These helpers let tests verify the daemon is running and
the hardware is emitting packets.
"""

from __future__ import annotations

import time

import redis


def _client() -> redis.Redis:
    return redis.Redis(host="127.0.0.1", port=6379, socket_timeout=2)


def is_available() -> bool:
    """Return True if Redis is up and accepting connections."""
    try:
        return _client().ping()
    except Exception:
        return False


def hk_hash(boardloc: int) -> dict[bytes, bytes]:
    """Fetch the QUABO_{boardloc} hash from Redis.

    Returns an empty dict if Redis is unavailable or the key does not exist.
    """
    try:
        return _client().hgetall(f"QUABO_{boardloc}")
    except Exception:
        return {}


def wait_for_keys(boardlocs: list[int], timeout: float = 30.0) -> dict[int, dict[bytes, bytes]]:
    """Poll Redis until all boardloc HASH keys are populated.

    Args:
        boardlocs: List of board locations to wait for.
        timeout: Maximum seconds to wait.

    Returns:
        Dict mapping boardloc → HK hash data.

    Raises:
        AssertionError: if any boardloc is still empty after timeout.
    """
    deadline = time.monotonic() + timeout
    missing: list[int] = list(boardlocs)
    result: dict[int, dict[bytes, bytes]] = {}

    while time.monotonic() < deadline and missing:
        still_missing = []
        for loc in missing:
            data = hk_hash(loc)
            if data:
                result[loc] = data
            else:
                still_missing.append(loc)
        missing = still_missing
        if missing:
            time.sleep(1.0)

    assert not missing, (
        f"Redis HK keys not populated within {timeout:.0f}s for boardloc(s): {missing}"
    )
    return result
