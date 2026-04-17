"""
chaos/process_chaos.py

Docker exec helpers for process-level fault injection.

All functions operate on a named container via the Docker SDK.
Every chaos action is designed to be reversible or self-terminating.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

_DOCKER_AVAILABLE = False
try:
    import docker as _docker_sdk  # noqa: F401
    _DOCKER_AVAILABLE = True
except ImportError:
    pass


def _get_client() -> Any:
    if not _DOCKER_AVAILABLE:
        raise RuntimeError("docker SDK not available — install 'docker' package")
    import docker
    return docker.from_env()


def _exec(container_name: str, cmd: str) -> tuple[int, str]:
    """Run a shell command inside a container. Returns (exit_code, output)."""
    client = _get_client()
    container = client.containers.get(container_name)
    result = container.exec_run(f"sh -c '{cmd}'")
    return result.exit_code, (result.output or b"").decode("utf-8", errors="replace")


def kill_process(container_name: str, process_name: str, sig: str = "KILL") -> None:
    """Send signal to the named process inside the container (by pidof)."""
    _exec(container_name, f"pkill -{sig} {process_name}")


def process_alive(container_name: str, process_name: str) -> bool:
    """Return True if a process with process_name is running in the container."""
    code, _ = _exec(container_name, f"pgrep -x {process_name}")
    return code == 0


def wait_for_process_death(
    container_name: str, process_name: str, timeout: float = 10.0
) -> bool:
    """Poll until process_name is gone in container or timeout. Returns True on success."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_alive(container_name, process_name):
            return True
        time.sleep(0.2)
    return False


def wait_for_process_start(
    container_name: str, process_name: str, timeout: float = 10.0
) -> bool:
    """Poll until process_name appears in container or timeout. Returns True on success."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process_alive(container_name, process_name):
            return True
        time.sleep(0.2)
    return False


def spawn_killer(
    container_name: str,
    process_name: str,
    delay_s: float = 0.0,
    sig: str = "KILL",
) -> threading.Timer:
    """Return a Timer that kills process_name after delay_s seconds.

    The returned Timer is already started.  Call .cancel() to abort it.
    """
    def _do_kill() -> None:
        kill_process(container_name, process_name, sig=sig)

    t = threading.Timer(delay_s, _do_kill)
    t.daemon = True
    t.start()
    return t


@contextmanager
def kill_after(
    container_name: str,
    process_name: str,
    delay_s: float = 0.0,
    sig: str = "KILL",
) -> Generator[None]:
    """Context manager that kills process_name in container after delay_s.

    Teardown cancels any pending kill timer (no-op if already fired).
    """
    timer = spawn_killer(container_name, process_name, delay_s=delay_s, sig=sig)
    try:
        yield
    finally:
        timer.cancel()


@contextmanager
def freeze_process(container_name: str, process_name: str) -> Generator[None]:
    """SIGSTOP a process for the duration of the block, then SIGCONT."""
    kill_process(container_name, process_name, sig="STOP")
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            kill_process(container_name, process_name, sig="CONT")
