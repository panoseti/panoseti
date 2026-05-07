"""
containers/base.py — PsetiContainer base class for all v2 test containers.

Wraps testcontainers DockerContainer with:
 - fluent env/volume helpers (consistent naming)
 - typed port discovery after start()
 - TCP + gRPC two-phase healthcheck (same as v1 fleet.py wait_healthy)
 - session-ID-namespaced container names (no xdist collisions)
"""

from __future__ import annotations

import os
import socket
import time
from typing import Any

from testcontainers.core.container import DockerContainer

GRPC_PORT = 50051
_DEFAULT_HEALTHCHECK_TIMEOUT = 90.0


def _tc_session_id() -> str:
    return os.environ.get("TC_SESSION_ID", "solo")


class PsetiContainer:
    """Thin wrapper around DockerContainer standardising PSETI test containers.

    Subclasses set `_IMAGE` and override `_configure()` to add their specific
    env vars, volumes, and command.  Callers use the lifecycle API:

        c = MyContainer(name="foo")
        c.start()
        c.wait_tcp(timeout=30)
        ...
        c.stop()
    """

    _IMAGE: str = ""
    _GRPC_PORT: int = GRPC_PORT

    def __init__(
        self,
        name: str | None = None,
        *,
        image: str | None = None,
        network: Any | None = None,
    ) -> None:
        img = image or self._IMAGE
        if not img:
            raise ValueError(f"{type(self).__name__} must set _IMAGE or pass image=")
        tc_id = _tc_session_id()
        self._name = name or f"pseti-v2-{type(self).__name__.lower()}-{tc_id}"
        self._container: DockerContainer = DockerContainer(img)
        self._container.with_name(self._name)
        if network is not None:
            self._container.with_network(network)
        self._mapped_grpc_port: int | None = None
        self._host_ip: str = "127.0.0.1"
        self._configure()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _configure(self) -> None:
        """Override in subclasses to add env vars, volumes, command, etc."""

    # ------------------------------------------------------------------
    # Fluent helpers (call from _configure())
    # ------------------------------------------------------------------

    def _env(self, key: str, value: str) -> "PsetiContainer":
        self._container.with_env(key, value)
        return self

    def _volume(self, host_path: str, container_path: str, mode: str = "rw") -> "PsetiContainer":
        self._container.with_volume_mapping(host_path, container_path, mode)
        return self

    def _expose(self, port: int) -> "PsetiContainer":
        self._container.with_exposed_ports(port)
        return self

    def _command(self, cmd: str) -> "PsetiContainer":
        self._container.with_command(cmd)
        return self

    def _kwargs(self, **kw: Any) -> "PsetiContainer":
        self._container.with_kwargs(**kw)
        return self

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "PsetiContainer":
        self._container.start()
        self._mapped_grpc_port = self._discover_port(self._GRPC_PORT)
        self._host_ip = self._discover_host_ip()
        return self

    def stop(self) -> None:
        import contextlib
        with contextlib.suppress(Exception):
            self._container.stop()

    def _discover_port(self, container_port: int) -> int | None:
        try:
            return int(self._container.get_exposed_port(container_port))
        except Exception:
            return None

    def _discover_host_ip(self) -> str:
        raw = self._container.get_container_host_ip()
        if raw in ("localhost", "127.0.0.1") or os.name == "nt":
            return "127.0.0.1"
        try:
            return socket.gethostbyname(raw)
        except socket.gaierror:
            return raw

    # ------------------------------------------------------------------
    # Port / IP accessors
    # ------------------------------------------------------------------

    @property
    def grpc_port(self) -> int:
        if self._mapped_grpc_port is None:
            raise RuntimeError(f"{self._name}: grpc_port unknown — call start() first")
        return self._mapped_grpc_port

    @property
    def grpc_host(self) -> str:
        return self._host_ip

    @property
    def grpc_addr(self) -> str:
        return f"{self._host_ip}:{self.grpc_port}"

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Healthchecks
    # ------------------------------------------------------------------

    def wait_tcp(self, *, timeout: float = _DEFAULT_HEALTHCHECK_TIMEOUT) -> None:
        """Block until the container's gRPC port accepts TCP connections."""
        if self._mapped_grpc_port is None:
            return
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"{self._name}: TCP not open on {self.grpc_addr} within {timeout}s"
                )
            try:
                with socket.create_connection(
                    (self._host_ip, self._mapped_grpc_port),
                    timeout=min(2.0, remaining),
                ):
                    return
            except OSError:
                time.sleep(0.5)

    def wait_grpc_ready(self, *, timeout: float = _DEFAULT_HEALTHCHECK_TIMEOUT) -> None:
        """Two-phase TCP + gRPC channel-READY healthcheck."""
        self.wait_tcp(timeout=timeout)
        if self._mapped_grpc_port is None:
            return
        import grpc as _grpc
        remaining = timeout - 0.5  # already used ~0.5s on TCP
        channel = _grpc.insecure_channel(self.grpc_addr)
        try:
            _grpc.channel_ready_future(channel).result(timeout=max(remaining, 1.0))
        except _grpc.FutureTimeoutError as exc:
            raise TimeoutError(
                f"{self._name}: gRPC channel on {self.grpc_addr} did not reach "
                f"READY within {timeout}s"
            ) from exc
        finally:
            channel.close()

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "PsetiContainer":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()
