"""
containers/daqnode_sim.py — Simulated DAQ-node container.

Runs panoseti-server --profile daq_node (daq_data + daq_control) inside
the pseti-test-runner image.  hashpipe is NOT present; UdsStrategy from
panoseti_grpc.daq_data.simulate acts as the data-plane stand-in.

The server is started with HEADNODE_IP / HEADNODE_GRPC_PORT so gRPC log
forwarding is configured correctly (it fails gracefully if the headnode is
unreachable, which is acceptable for the smoke test).
"""

from __future__ import annotations

import os
import pathlib

from ci.software_only_v2.containers.base import PsetiContainer

# Image built from Dockerfile.ci headnode stage (same as pseti-test-runner)
_DAQNODE_SIM_IMAGE = os.environ.get("PSETI_TEST_RUNNER_IMAGE", "pseti-test-runner:latest")

# grpc source on the host — volume-mounted so edits are visible immediately
_GRPC_SRC = (pathlib.Path(__file__).parents[6] / "grpc" / "src" / "panoseti_grpc").resolve()


class DaqNodeSimContainer(PsetiContainer):
    """A simulated DAQ node running panoseti-server --profile daq_node.

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

        # Mount live grpc source if it exists on the host (dev convenience)
        if _GRPC_SRC.exists():
            self._volume(str(_GRPC_SRC), "/grpc/src/panoseti_grpc", "rw")
            self._env("PYTHONPATH", "/grpc/src")

        # Run the unified gRPC server in daq_node profile
        self._command("pseti-grpc server --profile daq_node")

        # Capabilities required by hashpipe (even in sim mode, the base image
        # requires IPC_LOCK for shared-memory setup).
        # init=True enables Docker's built-in tini as PID 1, which properly
        # reaps orphaned zombie processes (needed for process chaos tests).
        self._kwargs(
            cap_add=["IPC_LOCK", "SYS_NICE"],
            init=True,
        )
