"""
Telemetry fixtures for HITL tests.
Provides HK socket, Redis client, and InfluxDB client.
"""

from __future__ import annotations

import socket

import pytest


@pytest.fixture
def hk_socket():
    """
    Bound UDP socket for receiving raw housekeeping packets (UDP/60002).
    Times out after 5 s; must be used in tests with hardware active.
    Sets SO_REUSEPORT to allow co-existence with capture_hk.py.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.settimeout(5.0)
        sock.bind(("", 60002))
        yield sock
    finally:
        sock.close()


@pytest.fixture(scope="session")
def redis_client():
    """Session-scoped Redis client (localhost, default port 6379)."""
    try:
        import redis
        client = redis.Redis(host="localhost", port=6379, db=0)
        client.ping()
        yield client
        client.close()
    except Exception as exc:
        pytest.skip(f"Redis not available: {exc}")


@pytest.fixture(scope="session")
def influx_client():
    """Session-scoped InfluxDB client (localhost, default port 8086)."""
    try:
        from influxdb import InfluxDBClient  # type: ignore[import-untyped]
        client = InfluxDBClient(host="localhost", port=8086, database="metadata")
        client.ping()
        yield client
        client.close()
    except Exception as exc:
        pytest.skip(f"InfluxDB not available: {exc}")
