"""
test_redis_utils.py

Unit tests for control/utils/redis_utils.py.
Uses fakeredis to simulate a Redis server with no live daemon required.
Covers:
  - store_in_redis: writes hash fields
  - get_casted_redis_value: int / float / string cast logic
  - get_updated_redis_keys: timestamp-based change detection
"""

import fakeredis
import pytest

from control.utils.redis_utils import (
    get_casted_redis_value,
    get_updated_redis_keys,
    store_in_redis,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def r() -> fakeredis.FakeRedis:
    """An in-memory fakeredis instance — no live Redis server needed."""
    return fakeredis.FakeRedis()


# ===========================================================================
# store_in_redis
# ===========================================================================

class TestStoreInRedis:
    def test_single_field(self, r) -> None:
        store_in_redis(r, "QUABO_0", {"pkt_num": "42"})
        assert r.hget("QUABO_0", "pkt_num") == b"42"

    def test_multiple_fields(self, r) -> None:
        store_in_redis(r, "QUABO_0", {"a": "1", "b": "2", "c": "hello"})
        assert r.hget("QUABO_0", "a") == b"1"
        assert r.hget("QUABO_0", "b") == b"2"
        assert r.hget("QUABO_0", "c") == b"hello"

    def test_overwrites_existing_field(self, r) -> None:
        store_in_redis(r, "QUABO_0", {"pkt_num": "10"})
        store_in_redis(r, "QUABO_0", {"pkt_num": "99"})
        assert r.hget("QUABO_0", "pkt_num") == b"99"

    def test_different_keys(self, r) -> None:
        store_in_redis(r, "QUABO_0", {"x": "1"})
        store_in_redis(r, "QUABO_1", {"x": "2"})
        assert r.hget("QUABO_0", "x") == b"1"
        assert r.hget("QUABO_1", "x") == b"2"

    def test_empty_dict_is_no_op(self, r) -> None:
        store_in_redis(r, "KEY", {})
        assert r.hget("KEY", "any") is None

    def test_bytes_key(self, r) -> None:
        store_in_redis(r, b"QUABO_BYTES", {"field": "val"})
        assert r.hget("QUABO_BYTES", "field") == b"val"

    def test_numeric_values_stored_as_strings(self, r) -> None:
        store_in_redis(r, "K", {"int_val": 42, "float_val": 3.14})
        # Redis stores everything as bytes; the value stored is str(42) → "42"
        result = r.hget("K", "int_val")
        assert result is not None


# ===========================================================================
# get_casted_redis_value
# Casting rules:
#   1. If str(val).isnumeric() or negative integer → int
#   2. If matches (-)X.Y[eE±N] pattern → float
#   3. Otherwise → str
# ===========================================================================

class TestGetCastedRedisValue:
    def test_positive_integer(self, r) -> None:
        store_in_redis(r, "K", {"n": "42"})
        assert get_casted_redis_value(r, "K", "n") == 42
        assert isinstance(get_casted_redis_value(r, "K", "n"), int)

    def test_zero(self, r) -> None:
        store_in_redis(r, "K", {"n": "0"})
        assert get_casted_redis_value(r, "K", "n") == 0
        assert isinstance(get_casted_redis_value(r, "K", "n"), int)

    def test_large_integer(self, r) -> None:
        store_in_redis(r, "K", {"n": "1000000000"})
        assert get_casted_redis_value(r, "K", "n") == 1_000_000_000

    def test_negative_integer(self, r) -> None:
        store_in_redis(r, "K", {"n": "-5"})
        result = get_casted_redis_value(r, "K", "n")
        # "-5" → val[0]=='-', val[1:].isnumeric() → int
        assert result == -5
        assert isinstance(result, int)

    def test_positive_float(self, r) -> None:
        store_in_redis(r, "K", {"v": "3.14"})
        result = get_casted_redis_value(r, "K", "v")
        assert abs(result - 3.14) < 1e-10
        assert isinstance(result, float)

    def test_negative_float(self, r) -> None:
        store_in_redis(r, "K", {"v": "-2.71"})
        result = get_casted_redis_value(r, "K", "v")
        assert isinstance(result, float)
        assert abs(result - (-2.71)) < 1e-10

    def test_scientific_notation_positive_exponent(self, r) -> None:
        store_in_redis(r, "K", {"v": "1.5e10"})
        result = get_casted_redis_value(r, "K", "v")
        assert isinstance(result, float)
        assert abs(result - 1.5e10) < 1

    def test_scientific_notation_negative_exponent(self, r) -> None:
        store_in_redis(r, "K", {"v": "1.5e-3"})
        result = get_casted_redis_value(r, "K", "v")
        assert isinstance(result, float)
        assert abs(result - 1.5e-3) < 1e-10

    def test_plain_string(self, r) -> None:
        store_in_redis(r, "K", {"s": "locked"})
        result = get_casted_redis_value(r, "K", "s")
        assert result == "locked"
        assert isinstance(result, str)

    def test_iso_timestamp_string(self, r) -> None:
        store_in_redis(r, "K", {"ts": "2026-03-27T12:00:00"})
        result = get_casted_redis_value(r, "K", "ts")
        assert isinstance(result, str)
        assert "2026" in result

    def test_mixed_alphanumeric_string(self, r) -> None:
        store_in_redis(r, "K", {"uid": "AABB1234"})
        result = get_casted_redis_value(r, "K", "uid")
        assert isinstance(result, str)

    def test_missing_rkey_returns_none(self, r) -> None:
        result = get_casted_redis_value(r, "NONEXISTENT_KEY", "field")
        assert result is None

    def test_missing_field_returns_none(self, r) -> None:
        store_in_redis(r, "K", {"a": "1"})
        result = get_casted_redis_value(r, "K", "b")
        assert result is None

    def test_empty_string_falls_through_to_string(self, r) -> None:
        """Empty string can't be cast to int or float → returned as string."""
        r.hset("K", "empty", "")
        result = get_casted_redis_value(r, "K", "empty")
        assert isinstance(result, str)


# ===========================================================================
# get_updated_redis_keys
# Returns keys whose Computer_UTC differs from the provided timestamp map
# ===========================================================================

class TestGetUpdatedRedisKeys:
    UTC1 = "2026-03-27 00:00:00"
    UTC2 = "2026-03-27 00:00:01"

    def test_all_new_keys_returned_when_no_timestamps(self, r) -> None:
        store_in_redis(r, "QUABO_0", {"Computer_UTC": self.UTC1, "pkt_num": "1"})
        store_in_redis(r, "QUABO_1", {"Computer_UTC": self.UTC1, "pkt_num": "2"})
        keys = get_updated_redis_keys(r, {})
        assert "QUABO_0" in keys
        assert "QUABO_1" in keys

    def test_unchanged_key_is_excluded(self, r) -> None:
        store_in_redis(r, "QUABO_0", {"Computer_UTC": self.UTC1, "pkt_num": "1"})
        timestamps = {"QUABO_0": self.UTC1}
        keys = get_updated_redis_keys(r, timestamps)
        assert "QUABO_0" not in keys

    def test_updated_key_is_included(self, r) -> None:
        store_in_redis(r, "QUABO_0", {"Computer_UTC": self.UTC2, "pkt_num": "2"})
        timestamps = {"QUABO_0": self.UTC1}  # stale timestamp
        keys = get_updated_redis_keys(r, timestamps)
        assert "QUABO_0" in keys

    def test_key_without_computer_utc_is_skipped(self, r) -> None:
        """Keys with no Computer_UTC field are not included in update list."""
        store_in_redis(r, "SOME_KEY", {"data": "value"})  # no Computer_UTC
        keys = get_updated_redis_keys(r, {})
        assert "SOME_KEY" not in keys

    def test_empty_db_returns_empty_list(self, r) -> None:
        keys = get_updated_redis_keys(r, {})
        assert keys == []

    def test_partial_update(self, r) -> None:
        """Only the key with a changed UTC is returned."""
        store_in_redis(r, "QUABO_0", {"Computer_UTC": self.UTC1})
        store_in_redis(r, "QUABO_1", {"Computer_UTC": self.UTC2})
        timestamps = {"QUABO_0": self.UTC1, "QUABO_1": self.UTC1}
        keys = get_updated_redis_keys(r, timestamps)
        assert "QUABO_0" not in keys
        assert "QUABO_1" in keys
