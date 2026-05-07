"""conftest.py — Tier 3 fleet test configuration."""

from typing import Any

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


def make_startdaq_params(
    fleet: "Any",
    node_index: int,
    run_dir: str,
    *,
    max_file_size_mb: float = 1024.0,
    group_ph_frames: bool = False,
    obs: str = "engineering",
    force: bool = False,
) -> dict:
    """Build a complete StartDaqParameters dict from the fleet's live config.

    The fleet's live_daq_config carries daq_ip_addr, bindhost, and module_ids
    that the Pydantic client model requires but test authors often omit.
    """
    node_cfg = fleet.live_daq_config.daq_nodes[node_index]
    return {
        "data_dir": str(node_cfg.data_dir),
        "daq_ip_addr": str(node_cfg.ip_addr),
        "bindhost": node_cfg.bindhost or "lo",
        "max_file_size_mb": max_file_size_mb,
        "group_ph_frames": group_ph_frames,
        "run_dir": run_dir,
        "obs": obs,
        "module_id": list(node_cfg.module_ids),
        "force": force,
    }
