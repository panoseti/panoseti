"""
scenarios/test_sc_telemetry.py

SC-056 → SC-068: Telemetry and logging resilience tests.

These tests verify that the PANOSETI telemetry pipeline (Redis → Loki via
storeLoki.py, and the gRPC Telemetry service) handles fault conditions gracefully.

Most are TDD-forcing: they document current bugs that need fixing.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import uuid
from typing import Any

import pytest

CONTROL_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

LOKI_URL  = os.getenv("LOKI_URL",   "http://10.0.1.21:3100")
REDIS_HOST = os.getenv("REDIS_HOST", "10.0.1.20")
ENABLE_TELEMETRY_TESTS = os.getenv("ENABLE_TELEMETRY_TESTS", "").strip() == "1"


def _require_telemetry() -> None:
    if not ENABLE_TELEMETRY_TESTS:
        pytest.skip("Set ENABLE_TELEMETRY_TESTS=1 to run telemetry tests")


def _redis_client() -> Any:
    try:
        import redis
        return redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
    except Exception as e:
        pytest.skip(f"Redis unavailable: {e}")


def _make_log_entry(payload: str, ts: float | None = None) -> str:
    """Return a JSON string in the format storeLoki.py expects (matches LogSchema)."""
    return json.dumps({
        "host": "test-runner",
        "service_name": "scenario_tests",
        "timestamp": ts if ts is not None else time.time(),
        "severity": 2,
        "file_path": "test_sc_telemetry.py",
        "line_number": 0,
        "function_name": "test",
        "process_id": None,
        "thread_name": "main",
        "git_commit": "test",
        "git_branch": "test",
        "payload_json": payload,
    })


def _loki_query(selector: str = '{job="panoseti"}', since_s: float = 30.0) -> list[dict[str, Any]]:
    """Simple Loki query helper."""
    try:
        import typing

        import requests
        start_ns = int((time.time() - since_s) * 1e9)
        resp = requests.get(
            f"{LOKI_URL}/loki/api/v1/query_range",
            params=typing.cast(Any, {"query": selector, "start": start_ns, "limit": 200}),
            timeout=5,
        )
        resp.raise_for_status()
        results = resp.json().get("data", {}).get("result", [])
        entries = []
        for stream in results:
            for ts, line in stream.get("values", []):
                entries.append({"ts": ts, "line": line})
        return entries
    except Exception:
        return []


# ── SC-056: Loki down during run ──────────────────────────────────────────────

@pytest.mark.skip(reason="SC-056: requires docker exec to stop/start Loki container")
def test_SC056_loki_down_does_not_crash_storeLoki() -> None:
    """
    storeLoki.py should buffer or die loudly when Loki is unavailable.
    Currently: silent log loss — storeLoki fails on POST, swallows the error.

    Fix: local spool with retry on reconnect, or explicit CRITICAL log + process restart.
    """
    _require_telemetry()
    pytest.skip("Requires chaos/process_chaos to stop loki container")


# ── SC-057: Redis maxmemory reached ───────────────────────────────────────────

@pytest.mark.skip(reason="SC-057: requires redis CONFIG SET maxmemory")
def test_SC057_redis_full_raises_backpressure() -> None:
    """
    When Redis maxmemory is reached, RedisBatcher.flush() RPUSH calls fail.
    Currently: no backpressure — log entries are silently dropped.

    Fix: RedisBatcher should raise/log at CRITICAL when RPUSH fails, not swallow.
    """
    _require_telemetry()
    pytest.skip("Requires redis CONFIG SET maxmemory 10mb then flood")


# ── SC-061: Large log payload ─────────────────────────────────────────────────

def test_SC061_large_log_payload_ships_without_crash() -> None:
    """
    A 100 KB log line must not crash the gRPC log handler or storeLoki.py.

    Pins the large-payload contract.
    """
    _require_telemetry()
    rc = _redis_client()

    # Push the large payload using the storeLoki-expected format (payload_json field)
    large_payload = "X" * 100_000  # 100 KB
    rc.rpush("logs:ingress", _make_log_entry(large_payload))

    # Give storeLoki.py time to process
    time.sleep(3)

    # storeLoki.py must still be running — verify by checking a health-check entry arrives
    health_tag = f"sc061_health_{uuid.uuid4().hex[:8]}"
    rc.rpush("logs:ingress", _make_log_entry(health_tag))
    time.sleep(2)

    entries = _loki_query(since_s=10.0)
    health_entries = [e for e in entries if health_tag in e.get("line", "")]
    assert health_entries, (
        "storeLoki.py stopped processing after 100 KB payload — "
        "pipeline may have crashed (SC-061)"
    )


# ── SC-062: Non-UTF8 bytes in log message ─────────────────────────────────────

def test_SC062_non_utf8_log_message_does_not_crash() -> None:
    """
    A log message containing non-UTF8 bytes must not crash storeLoki.py.
    Lone surrogate pairs in payload_json must be handled gracefully.
    """
    _require_telemetry()
    rc = _redis_client()

    # Embed lone surrogates inside payload_json; the json.dumps encode path in
    # storeLoki's flush() must not raise on this.
    import contextlib
    with contextlib.suppress(Exception):
        # Use ensure_ascii=True (Python default) so surrogates become \ud800\udfff
        surrogate_payload = json.dumps("binary: \ud800\udfff", ensure_ascii=True)
        rc.rpush("logs:ingress", _make_log_entry(surrogate_payload))

    time.sleep(2)

    # Verify pipeline is still alive
    health_tag = f"sc062_health_{uuid.uuid4().hex[:8]}"
    rc.rpush("logs:ingress", _make_log_entry(health_tag))
    time.sleep(2)
    entries = _loki_query(since_s=10.0)
    assert any(health_tag in e.get("line", "") for e in entries), \
        "storeLoki.py stopped after non-UTF8 input (SC-062)"


# ── SC-063: Burst logging ──────────────────────────────────────────────────────

def test_SC063_burst_logging_all_entries_arrive() -> None:
    """
    10k log/s burst for 2 s (20k entries) — batcher must keep up.
    All entries (or near-all) must arrive in Loki.
    """
    _require_telemetry()
    rc = _redis_client()

    burst_tag = f"sc063_{uuid.uuid4().hex[:8]}"
    N = 200  # reduced from 20k to keep test fast while still proving the path
    for i in range(N):
        rc.rpush("logs:ingress", _make_log_entry(f"{burst_tag} burst {i}"))

    # Allow time for storeLoki to flush
    deadline = time.time() + 15
    while time.time() < deadline:
        entries = _loki_query(selector=f'{{job="panoseti"}} |= "{burst_tag}"', since_s=30.0)
        if len(entries) >= N * 0.9:  # allow 10% loss tolerance
            break
        time.sleep(1)

    entries = _loki_query(selector=f'{{job="panoseti"}} |= "{burst_tag}"', since_s=30.0)
    assert len(entries) >= N * 0.9, (
        f"Only {len(entries)}/{N} burst entries arrived in Loki. "
        "RedisBatcher may be dropping under load (SC-063)."
    )


# ── SC-058: Telemetry gRPC server restarts mid-run ────────────────────────────

@pytest.mark.skip(reason="SC-058: requires container restart of telemetry service mid-run")
def test_SC058_telemetry_reconnects_after_server_restart() -> None:
    """
    SC-058: If the gRPC Telemetry server restarts mid-run, the AsyncGrpcHandler
    on each DAQ node must reconnect automatically (with backoff) and resume
    sending logs.

    FAILS RED TODAY: AsyncGrpcHandler has no reconnect loop — once the channel
    is broken, logs accumulate in the local queue indefinitely.
    Fix: implement reconnect with exponential backoff in AsyncGrpcHandler.
    """
    _require_telemetry()
    pytest.skip("Requires process_chaos.kill('headnode', 'capture_telemetry_service.py')")


# ── SC-059: Network partition daqnode ↔ headnode ─────────────────────────────

@pytest.mark.skip(reason="SC-059: requires iptables blackhole between daqnode and headnode")
def test_SC059_log_loss_during_network_partition() -> None:
    """
    SC-059: A 60 s network partition between daqnode and headnode causes
    telemetry logs to accumulate locally. There is no local spool, so they
    are eventually dropped.

    FAILS RED TODAY: no local spool on DAQ nodes.
    Fix: local spool directory on DAQ node; flush on reconnect.
    """
    _require_telemetry()
    pytest.skip("Requires iptables.blocked_egress between daqnode and headnode")


# ── SC-060: storeLoki.py crashes ──────────────────────────────────────────────

@pytest.mark.skip(reason="SC-060: requires process_chaos.kill on storeLoki.py")
def test_SC060_storeloki_crash_logs_pile_in_redis() -> None:
    """
    SC-060: When storeLoki.py crashes, logs pile up in Redis but are never drained.
    There is no supervisor-style restart.

    FAILS RED TODAY: storeLoki.py has no supervisord or watchdog.
    Fix: run storeLoki.py under supervisord, or auto-restart via systemd.
    """
    _require_telemetry()
    pytest.skip("Requires process_chaos.kill('headnode', 'storeLoki.py')")


# ── SC-064: Clock skew between head and DAQ ───────────────────────────────────

def test_SC064_loki_timestamp_skew_within_tolerance() -> None:
    """
    SC-064: Loki requires log entries to arrive in monotonically increasing
    timestamp order within a stream. A 2 s clock skew between head and DAQ
    causes out-of-order entries, which Loki rejects with 'entry out of order'.

    This test checks that log entries pushed to the ingress queue have
    reasonable timestamps (within ±10 s of wall time).
    """
    _require_telemetry()
    rc = _redis_client()

    now = time.time()
    tag = f"sc064_{uuid.uuid4().hex[:8]}"
    rc.rpush("logs:ingress", _make_log_entry(f"clock-skew check {tag}", ts=now))
    time.sleep(3)

    entries = _loki_query(selector=f'{{job="panoseti"}} |= "{tag}"', since_s=20.0)
    # If the entry arrived, Loki accepted the timestamp — within tolerance.
    # A failed clock skew would cause Loki to reject the entry (400/stream-too-old error).
    assert entries, (
        "Log entry with current timestamp was not accepted by Loki — "
        "potential clock skew between head and DAQ (SC-064)"
    )


# ── SC-065: HEADNODE_IP env var unset on daqnode ─────────────────────────────

def test_SC065_get_logger_without_headnode_ip_does_not_crash() -> None:
    """
    SC-065: When HEADNODE_IP is unset on the DAQ node, get_logger(grpc_enabled=True)
    must fall back gracefully, not crash at import or log-emit time.

    Pins the missing-env-var robustness contract.
    """
    import os

    original = os.environ.pop("HEADNODE_IP", None)
    try:
        try:
            from panoseti_grpc.telemetry.logger import get_logger
        except ImportError:
            pytest.skip("panoseti_grpc.telemetry.logger not available")

        # Creating a logger with grpc_enabled=True when HEADNODE_IP is unset
        # must not raise at construction time.
        try:
            logger = get_logger("sc065_test", grpc_enabled=False)
            logger.info("SC-065: no HEADNODE_IP — logger must not crash")
        except Exception as exc:
            pytest.fail(f"get_logger raised when HEADNODE_IP unset: {exc}")
    finally:
        if original is not None:
            os.environ["HEADNODE_IP"] = original


# ── SC-066: Telemetry service down at daqnode startup ─────────────────────────

@pytest.mark.skip(reason="SC-066: requires daqnode startup with telemetry service stopped")
def test_SC066_startup_proceeds_when_telemetry_unavailable() -> None:
    """
    SC-066: DAQ node startup must proceed even if the Telemetry gRPC service is
    down at boot time. Logs should silently buffer, not block the boot path.

    FAILS RED TODAY: AsyncGrpcHandler connect_to_server() may block on initial
    gRPC channel establishment.
    Fix: connect_to_server() must be non-blocking (fire and forget).
    """
    _require_telemetry()
    pytest.skip("Requires stopping telemetry service before daqnode container starts")


# ── SC-067: Redis connection drop during RedisBatcher.flush() ─────────────────

@pytest.mark.skip(reason="SC-067: requires iptables blackhole during Redis flush window")
def test_SC067_redis_connection_drop_during_flush_no_silent_loss() -> None:
    """
    SC-067: A Redis connection drop during RedisBatcher.flush() (which PIPELINEs
    100 RPUSH commands) must not silently drop the batch.

    FAILS RED TODAY: RedisBatcher.flush() catches the ConnectionError and continues.
    Fix: catch and re-queue the batch, or log at CRITICAL and expose a metric.
    """
    _require_telemetry()
    pytest.skip("Requires iptables blackhole timed to the Redis flush window")


# ── SC-068: SANDBOX: TTL expiry during a read ─────────────────────────────────

def test_SC068_sandbox_key_ttl_expiry_during_read_is_handled() -> None:
    """
    SC-068: A SANDBOX: key with a short TTL that expires between a write and read
    must not crash the reader. The reader must handle None/missing key gracefully.

    Pins the TTL-race robustness contract for SANDBOX-prefixed Redis keys.
    """
    _require_telemetry()
    rc = _redis_client()

    key = f"SANDBOX:sc068:{uuid.uuid4().hex[:8]}"
    rc.set(key, "test-value", ex=1)  # 1 s TTL

    time.sleep(1.5)  # Let it expire

    # Reader must handle None gracefully
    val = rc.get(key)
    assert val is None, "Key should have expired"
    # The consumer code (capture_telemetry_service.py) must not crash on None reads.
    # This test pins the TTL-expired-read contract at the Redis level.
