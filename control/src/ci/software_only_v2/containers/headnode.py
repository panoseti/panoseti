"""
containers/headnode.py — Head-node container.

Runs the panoseti control-plane image (pseti-test-runner:latest) configured
as a head node.  In the smoke test it runs ``sleep infinity`` so it is alive
but doesn't need Redis/InfluxDB.  The pseti_workspace config files are
volume-mounted so control scripts running inside the container see a valid
config environment.
"""

from __future__ import annotations

import os
import pathlib

from ci.software_only_v2.containers.base import PsetiContainer

_HEADNODE_IMAGE = os.environ.get("PSETI_TEST_RUNNER_IMAGE", "pseti-test-runner:latest")
_GRPC_SRC = (pathlib.Path(__file__).parents[6] / "grpc" / "src" / "panoseti_grpc").resolve()


class HeadnodeContainer(PsetiContainer):
    """Control-plane headnode container.

    Two modes:
    - ``command="sleep infinity"`` (default) — inert container used as an
      addressable headnode in fleet topologies that don't need the telemetry
      gRPC service running.
    - ``command="pseti-grpc server --profile headnode"`` — starts the telemetry
      service (requires Redis to be reachable at REDIS_HOST:REDIS_PORT).
    """

    _IMAGE = _HEADNODE_IMAGE
    _GRPC_PORT = 50051

    def __init__(
        self,
        name: str,
        *,
        command: str = "sleep infinity",
        config_dir: pathlib.Path | None = None,
        state_dir: pathlib.Path | None = None,
        redis_host: str = "127.0.0.1",
        redis_port: int = 6379,
        grpc_port: int = 50051,
        network=None,
    ) -> None:
        self._cmd = command
        self._config_dir = config_dir
        self._state_dir = state_dir
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._GRPC_PORT = grpc_port
        super().__init__(name=name, network=network)

    def _configure(self) -> None:
        self._command(self._cmd)
        self._env("GRPC_PORT", str(self._GRPC_PORT))
        self._env("REDIS_HOST", self._redis_host)
        self._env("REDIS_PORT", str(self._redis_port))

        if self._config_dir and self._config_dir.exists():
            self._volume(str(self._config_dir), "/app/configs", "ro")
            self._env("PSETI_CONFIG", "/app/configs")

        if self._state_dir and self._state_dir.exists():
            self._volume(str(self._state_dir), "/app/state", "rw")
            self._env("PSETI_STATE", "/app/state")

        if _GRPC_SRC.exists():
            self._volume(str(_GRPC_SRC), "/grpc/src/panoseti_grpc", "rw")
            self._env("PYTHONPATH", "/grpc/src")

        # Only expose the gRPC port when we're running a real server
        if "sleep" not in self._cmd:
            self._expose(self._GRPC_PORT)
