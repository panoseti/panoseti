"""
HITL CI Orchestrator for PANOSETI.
Provides a robust sequence for hardware-software integration tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Ensure we can import control modules if needed
_CONTROL_DIR = Path(__file__).parent.parent.parent
if str(_CONTROL_DIR / "src") not in sys.path:
    sys.path.insert(0, str(_CONTROL_DIR / "src"))

def run_cmd(cmd: list[str], cwd: Path | str = _CONTROL_DIR) -> int:
    print(f"\n>>> Running: {' '.join(cmd)}")
    # Use the current environment but ensure PSETI_CONTROL and IN_DOCKER_CI are not overriding host paths
    env = os.environ.copy()
    env.pop("PSETI_CONTROL", None)
    env.pop("PSETI_ROOT", None)
    env.pop("IN_DOCKER_CI", None)
    
    # Ensure build variables are present for docker compose
    # Since we run from control/, PSETI_CONTROL_BUILD is . and ROOT is ..
    if "PSETI_CONTROL_BUILD" not in env:
        env["PSETI_CONTROL_BUILD"] = "."
    if "PSETI_ROOT_BUILD" not in env:
        env["PSETI_ROOT_BUILD"] = ".."
    
    # Inject PSETI_CONFIG for HITL environment so check-env passes on host
    if "PSETI_CONFIG" not in env:
        env["PSETI_CONFIG"] = str(_CONTROL_DIR / "src/ci/hardware_software/configs")

    return subprocess.run(cmd, cwd=cwd, env=env).returncode

def main():
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
        # 6. Power off
        ["docker", "compose", "-f", "src/ci/docker-compose.hw-sw.yml", "exec", "-T", "headnode-server", "pseti", "power", "off"],
        # 7. Run tests
        ["docker", "compose", "-f", "src/ci/docker-compose.hw-sw.yml", "exec", "-T", "headnode-server", "pseti", "test", "hw", "run", "--assume-state", "UNPOWERED", "-v"],
    ]

    for i, step in enumerate(steps, 1):
        print(f"\n[Step {i}/{len(steps)}] {step[2] if 'pseti' in step else step[0]}")
        ret = run_cmd(step)
        if ret != 0:
            print(f"\n!!! Step {i} failed with exit code {ret}")
            # Final cleanup attempt
            print("\n>>> Attempting final safety teardown...")
            run_cmd(["docker", "compose", "-f", "src/ci/docker-compose.hw-sw.yml", "exec", "-T", "headnode-server", "pseti", "power", "off"])
            run_cmd(["uv", "run", "pseti", "test", "hw", "down", "-v"])
            sys.exit(ret)

    # Success cleanup
    print("\n>>> All steps passed. Performing final safety teardown...")
    run_cmd(["docker", "compose", "-f", "src/ci/docker-compose.hw-sw.yml", "exec", "-T", "headnode-server", "pseti", "power", "off"])
    run_cmd(["uv", "run", "pseti", "test", "hw", "down", "-v"])
    print("\nDONE.")

if __name__ == "__main__":
    main()
