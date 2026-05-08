"""
orchestrator/network.py — Docker network management for v2 test fleet.

Provides SharedNetwork (a persistent backbone that survives across xdist workers)
and per-worker subnet shifting so two parallel workers don't fight over the
same 192.168.x.x addresses.
"""

from __future__ import annotations

import contextlib
import os
import pathlib

# ---------------------------------------------------------------------------
# Docker host auto-detection
# ---------------------------------------------------------------------------

def setup_docker_host() -> None:
    """Configure DOCKER_HOST for the current platform if not already set.

    On macOS with Docker Desktop the socket lives at
    ~/.docker/run/docker.sock rather than the Linux default
    /var/run/docker.sock.
    """
    if os.environ.get("DOCKER_HOST"):
        return
    macos_socket = pathlib.Path.home() / ".docker" / "run" / "docker.sock"
    if macos_socket.exists():
        os.environ["DOCKER_HOST"] = f"unix://{macos_socket}"


# ---------------------------------------------------------------------------
# SharedNetwork
# ---------------------------------------------------------------------------

class SharedNetwork:
    """A persistent Docker network shared across containers in a test session.

    Multiple xdist workers create the network idempotently (check_duplicate=True).
    The network is never removed during tests; clean up manually or via
    ``docker network prune`` after a full test run.
    """

    def __init__(self, name: str = "pseti-v2-shared-net") -> None:
        self.name = name
        self._network = None

    def create(self) -> None:
        import docker
        import time
        client = docker.from_env()
        # Retry up to 3 times to handle potential race conditions during
        # parallel creation by multiple xdist workers.
        for _ in range(3):
            try:
                self._network = client.networks.get(self.name)
                return
            except docker.errors.NotFound:
                try:
                    self._network = client.networks.create(
                        self.name, check_duplicate=True
                    )
                    return
                except Exception:
                    # Possibly a race condition: someone else created it
                    # after our .get() failed but before our .create() finished.
                    time.sleep(0.5)
        
        # Final attempt without catching, let it raise if it still fails.
        self._network = client.networks.get(self.name)

    def remove(self) -> None:
        # Never remove the shared backbone during tests — other workers may
        # still be using it.  Prune via 'docker network prune' after the run.
        pass

    @property
    def id(self) -> str | None:
        return self._network.id if self._network else None


# ---------------------------------------------------------------------------
# Per-worker subnet shifting
# ---------------------------------------------------------------------------

def worker_subnet_offset() -> int:
    """Return an integer offset (0-63) derived from TC_SESSION_ID.

    Two xdist workers get different offsets so their quabo placeholder IPs
    don't collide.  Solo runs always return 0.

    The placeholder subnet for worker i is 192.168.(100 + i).x.
    """
    tc_id = os.environ.get("TC_SESSION_ID", "solo")
    if tc_id in ("solo", "master"):
        return 0
    digits = "".join(c for c in tc_id if c.isdigit())
    return int(digits) % 64 if digits else 0


def placeholder_subnet(offset: int | None = None) -> str:
    """Base subnet string for placeholder DAQ-node IPs.

    Example: '192.168.100' (for offset=0), '192.168.101' (for offset=1).
    """
    if offset is None:
        offset = worker_subnet_offset()
    return f"192.168.{100 + offset}"
