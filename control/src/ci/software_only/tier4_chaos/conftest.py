"""conftest.py — Tier 4 chaos test configuration."""

import pytest


def _docker_available() -> bool:
    """Return True if the Docker daemon is reachable."""
    try:
        import docker  # type: ignore[import-untyped]
        docker.from_env(timeout=5).ping()
        return True
    except Exception:
        return False


requires_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon not available",
)
