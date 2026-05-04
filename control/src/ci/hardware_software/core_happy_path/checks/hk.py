"""HK Redis assertions for happy-path tests."""

from __future__ import annotations

from ci.hardware_software.core import redis as _redis


def redis_populated(boardlocs: list[int], timeout: float = 30.0) -> None:
    """Assert all boardloc HK hashes are populated in Redis."""
    if not _redis.is_available():
        import pytest
        pytest.skip("Redis not available — skipping HK check")
    _redis.wait_for_keys(boardlocs, timeout=timeout)


def value_in_range(boardloc: int, field: bytes, lo: float, hi: float) -> None:
    """Assert a specific HK field for a boardloc is within [lo, hi]."""
    data = _redis.hk_hash(boardloc)
    assert field in data, f"QUABO_{boardloc}: field {field!r} not in Redis hash"
    try:
        val = float(data[field])
    except (ValueError, TypeError) as exc:
        raise AssertionError(
            f"QUABO_{boardloc}: field {field!r} is not numeric: {data[field]!r}"
        ) from exc
    assert lo <= val <= hi, (
        f"QUABO_{boardloc}: field {field!r} = {val} out of range [{lo}, {hi}]"
    )
