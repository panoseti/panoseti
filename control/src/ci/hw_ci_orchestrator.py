"""
HITL CI Orchestrator for PANOSETI.
Provides a robust sequence for hardware-software integration tests.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Ensure we can import control modules if needed
_CONTROL_DIR = Path(__file__).parent.parent.parent
if str(_CONTROL_DIR / "src") not in sys.path:
    sys.path.insert(0, str(_CONTROL_DIR / "src"))

from ci.compose_env import compose_env  # noqa: E402


def run_cmd(cmd: list[str], cwd: Path | str = _CONTROL_DIR) -> int:
    print(f"\n>>> Running: {' '.join(cmd)}")
    # Use the current environment but ensure PSETI_CONTROL and IN_DOCKER_CI are not overriding host paths
    env = os.environ.copy()
    env.pop("PSETI_CONTROL", None)
    env.pop("PSETI_ROOT", None)
    env.pop("IN_DOCKER_CI", None)

    # compose_env() sets PSETI_ROOT_BUILD/PSETI_CONTROL_BUILD/PSETI_CONFIG/
    # PSETI_CORE_OBS_CONFIGS/HOST_UID/HOST_GID -- the single source of truth
    # for what docker-compose.hw-sw.yml needs, shared with hw_utils/cli.py.
    # Two independent reimplementations of this here and in hw_utils/cli.py
    # each drifted and forgot PSETI_CORE_OBS_CONFIGS at different points,
    # breaking CI identically both times.
    env = compose_env(env)

    return subprocess.run(cmd, cwd=cwd, env=env).returncode

def main():
    parser = argparse.ArgumentParser(description="PSETI HITL CI Orchestrator")
    parser.add_argument("--no-fail-fast", action="store_true", help="Continue even if steps fail.")
    args_parsed = parser.parse_args()

    steps = [
        # 1. Pre-deploy check
        ["uv", "run", "pseti", "test", "hw", "check-env", "--pre-deploy"],
        # 2. Teardown
        ["uv", "run", "pseti", "test", "hw", "down", "-v"],
        # 3. Build
        ["uv", "run", "pseti", "test", "hw", "build"],
        # 4. Deploy
        ["uv", "run", "pseti", "test", "hw", "deploy"],
        # 5. Post-deploy check
        ["uv", "run", "pseti", "test", "hw", "check-env", "--post-deploy"],
        # 6. Power off (Ensure clean state)
        ["docker", "compose", "-f", "src/ci/docker-compose.hw-sw.yml", "exec", "-T", "headnode-server", "pseti", "power", "off"],
        # 7. Run tests
        ["docker", "compose", "-f", "src/ci/docker-compose.hw-sw.yml", "exec", "-T", "headnode-server", "pseti", "test", "hw", "run", "--assume-state", "UNPOWERED", "-v", "--yes", "--", "-rs"],
    ]

    failed = False
    for i, step in enumerate(steps, 1):
        print(f"\n[Step {i}/{len(steps)}] {step[2] if 'pseti' in step else step[0]}")
        ret = run_cmd(step)
        if ret != 0:
            print(f"\n!!! Step {i} failed with exit code {ret}")
            failed = True
            if not args_parsed.no_fail_fast:
                break

    # Final cleanup sequence
    print("\n>>> Performing final safety teardown...")
    run_cmd(["docker", "compose", "-f", "src/ci/docker-compose.hw-sw.yml", "exec", "-T", "headnode-server", "pseti", "power", "off"])
    run_cmd(["uv", "run", "pseti", "test", "hw", "down", "-v"])
    
    if failed:
        print("\n>>> HITL Sequence FINISHED WITH ERRORS.")
        sys.exit(1)
    
    print("\nDONE.")

if __name__ == "__main__":
    main()
