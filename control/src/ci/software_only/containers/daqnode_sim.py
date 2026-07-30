"""
containers/daqnode_sim.py — Simulated DAQ-node container.

Runs pseti-grpc server --profile daq_node (daq_data + daq_control) inside
the pseti-test-runner image.  hashpipe is NOT present; UdsStrategy from
panoseti_grpc.daq_data.simulate acts as the data-plane stand-in.

The server is started with HEADNODE_IP / HEADNODE_GRPC_PORT so gRPC log
forwarding is configured correctly (it fails gracefully if the headnode is
unreachable, which is acceptable for the smoke test).
"""

from __future__ import annotations

import os
import pathlib

from ci.software_only.containers.base import PsetiContainer

# Image built from Dockerfile.ci headnode stage (same as pseti-test-runner)
_DAQNODE_SIM_IMAGE = os.environ.get("PSETI_TEST_RUNNER_IMAGE", "pseti-test-runner:latest")


def _find_grpc_src() -> pathlib.Path:
    # Repo root is 5 levels up: containers/ → software_only/ → ci/ → src/ → control/ → repo_root/
    try:
        host_path = pathlib.Path(__file__).parents[5] / "grpc" / "src" / "panoseti_grpc"
        if (host_path / "__init__.py").exists():
            return host_path.resolve()
    except IndexError:
        pass
    # Fallback: standard container path — only valid if non-empty (has __init__.py)
    fallback = pathlib.Path("/grpc/src/panoseti_grpc")
    if (fallback / "__init__.py").exists():
        return fallback.resolve()
    return fallback


# grpc source on the host — volume-mounted so edits are visible immediately
_GRPC_SRC = _find_grpc_src()


class DaqNodeSimContainer(PsetiContainer):
    """A simulated DAQ node running pseti-grpc server --profile daq_node.

    The daq_data service starts with the bundled simulate_daq_cfg so
    UdsStrategy can replay PFF frames when a client calls init_sim().
    """

    _IMAGE = _DAQNODE_SIM_IMAGE
    _GRPC_PORT = 50051

    def __init__(
        self,
        name: str,
        *,
        module_ids: list[int] | None = None,
        data_dir: str = "/data",
        headnode_ip: str = "10.0.1.5",
        headnode_grpc_port: int = 50051,
        network=None,
    ) -> None:
        self._module_ids = module_ids or []
        self._data_dir = data_dir
        self._headnode_ip = headnode_ip
        self._headnode_grpc_port = headnode_grpc_port
        super().__init__(name=name, network=network)

    def _configure(self) -> None:
        # gRPC server on the standard port
        self._expose(self._GRPC_PORT)
        self._env("GRPC_PORT", str(self._GRPC_PORT))

        # Log forwarding goes to the headnode (fails gracefully if unreachable)
        self._env("HEADNODE_IP", self._headnode_ip)
        self._env("HEADNODE_GRPC_PORT", str(self._headnode_grpc_port))

        # Always use loopback for hashpipe net_thread in CI
        self._env("BINDHOST", "lo")

        # Data directory
        self._env("DATA_DIR", self._data_dir)

        # Mount fake hashpipe and configure daq_control to use it
        fake_hp_host = (
            pathlib.Path(__file__).parents[2] / "fixtures" / "build" / "fake_hashpipe.py"
        )
        self._volume(str(fake_hp_host.resolve()), "/usr/local/bin/fake_hashpipe.py", "ro")
        self._env("PSETI_DAQ_CONTROL_HASHPIPE_PATH", "/usr/local/bin/fake_hashpipe.py")
        self._env("PSETI_DAQ_CONTROL_HASHPIPE_NAME", "fake_hashpipe.py")

        # Mount live grpc source only if the host path is a valid Python package.
        # Checking __init__.py prevents mounting a stale empty directory that
        # would shadow the COPY'd source inside the image.
        if (_GRPC_SRC / "__init__.py").exists():
            self._volume(str(_GRPC_SRC), "/grpc/src/panoseti_grpc", "rw")
            self._env("PYTHONPATH", "/grpc/src")

        # Run the unified gRPC server in daq_node profile.
        # We wrap it in a shell loop so that 'chaos kill' tests don't kill the
        # container, and the process automatically restarts.
        self._command("/bin/sh -c 'while true; do pseti-grpc server --profile daq_node; sleep 1; done'")

        # Capabilities required by hashpipe (even in sim mode, the base image
        # requires IPC_LOCK for shared-memory setup).
        # init=True enables Docker's built-in tini as PID 1, which properly
        # reaps orphaned zombie processes (needed for process chaos tests).
        self._kwargs(
            cap_add=["IPC_LOCK", "SYS_NICE", "NET_ADMIN"],
            init=True,
        )
