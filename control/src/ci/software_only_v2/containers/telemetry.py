"""
containers/telemetry.py — Redis (+ optional Loki) sidecar containers.

TelemetryStack groups the sidecars needed by the headnode telemetry service:
- Redis (mandatory) — hot storage for device status + log queue
- Loki (optional) — log aggregation backend

In most tier-3 tests only Redis is needed.  Loki is reserved for tier-4/5
tests that assert on structured log content.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

from testcontainers.core.container import DockerContainer

_REDIS_IMAGE = "redis:alpine"
_LOKI_IMAGE = "grafana/loki:latest"


class TelemetryStack:
    """Optional sidecar containers for the telemetry pipeline.

    Usage::

        stack = TelemetryStack(network=shared_net, enable_loki=False)
        stack.start()
        ...env vars available via stack.redis_host / stack.redis_port...
        stack.stop()
    """

    def __init__(
        self,
        *,
        name_prefix: str = "pseti-v2",
        network: Any | None = None,
        enable_loki: bool = False,
    ) -> None:
        self._prefix = name_prefix
        self._network = network
        self._enable_loki = enable_loki
        self._redis: DockerContainer | None = None
        self._loki: DockerContainer | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> TelemetryStack:
        self._redis = DockerContainer(_REDIS_IMAGE)
        self._redis.with_name(f"{self._prefix}-redis")
        self._redis.with_exposed_ports(6379)
        if self._network:
            self._redis.with_network(self._network)
        self._redis.start()

        if self._enable_loki:
            self._loki = DockerContainer(_LOKI_IMAGE)
            self._loki.with_name(f"{self._prefix}-loki")
            self._loki.with_exposed_ports(3100)
            if self._network:
                self._loki.with_network(self._network)
            self._loki.start()

        # Give Redis a moment to finish initialising
        time.sleep(1.0)
        return self

    def stop(self) -> None:
        for c in (self._redis, self._loki):
            if c is not None:
                with contextlib.suppress(Exception):
                    c.stop()
        self._redis = None
        self._loki = None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def redis_host(self) -> str:
        return "127.0.0.1"

    @property
    def redis_port(self) -> int:
        if self._redis is None:
            raise RuntimeError("TelemetryStack not started")
        return int(self._redis.get_exposed_port(6379))

    @property
    def loki_host(self) -> str:
        return "127.0.0.1"

    @property
    def loki_port(self) -> int | None:
        if self._loki is None:
            return None
        return int(self._loki.get_exposed_port(3100))

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> TelemetryStack:
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()
