"""
Shared environment-variable construction for docker-compose.hw-sw.yml.

Every entry point that invokes `docker compose -f docker-compose.hw-sw.yml
...` must build its env through compose_env(), not reimplement it.
Historically this logic was duplicated in hw_utils/cli.py and
hw_ci_orchestrator.py independently, and the two drifted: at different
points each was missing PSETI_CORE_OBS_CONFIGS, which breaks CI with
"invalid spec: empty section between colons" (Compose collapses
${PSETI_CORE_OBS_CONFIGS}:/mnt/core_obs_configs:ro to an empty source
section when the var is unset, and it fails this way for *any* compose
subcommand against the file -- including `exec` against an
already-running container that doesn't need the volume at runtime,
since Compose parses and interpolates the whole file up front).
"""

from __future__ import annotations

import os
from pathlib import Path

from control.utils.paths import PanoPaths

# control/ root (contains pyproject.toml -- needed for `uv run`)
CONTROL_DIR: Path = PanoPaths.base_dir()
# panoseti-software/ root (needed for the compose build context)
PSETI_ROOT: Path = PanoPaths.software_root_dir()
# HITL test config directories
_HW_SW_DIR = CONTROL_DIR / "src" / "ci" / "hardware_software"
HW_CONFIGS_DIR: Path = _HW_SW_DIR / "configs"
HW_CORE_OBS_CONFIGS_DIR: Path = _HW_SW_DIR / "core_obs_configs"


def compose_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build the full env dict required by docker-compose.hw-sw.yml.

    Args:
        base_env: Starting environment to overlay onto. Defaults to a copy
            of the current process environment (``os.environ``).

    Returns:
        A new dict: *base_env* plus the HITL compose-required vars.
        ``PSETI_CONFIG`` is preserved if already present in *base_env*
        (some callers point it at a variant config on purpose); the rest
        are always set to their canonical values.
    """
    env = dict(base_env) if base_env is not None else os.environ.copy()

    uid = os.getuid()
    gid = os.getgid()
    if uid == 0:
        # Running under sudo: use the original user's IDs so files created
        # in the container are owned by them, not root.
        uid = int(os.environ.get("SUDO_UID", 0))
        gid = int(os.environ.get("SUDO_GID", 0))

    env["PSETI_ROOT_BUILD"] = str(PSETI_ROOT)
    env["PSETI_CONTROL_BUILD"] = str(CONTROL_DIR)
    env.setdefault("PSETI_CONFIG", str(HW_CONFIGS_DIR))
    env["PSETI_CORE_OBS_CONFIGS"] = str(HW_CORE_OBS_CONFIGS_DIR)
    env["HOST_UID"] = str(uid)
    env["HOST_GID"] = str(gid)
    return env
