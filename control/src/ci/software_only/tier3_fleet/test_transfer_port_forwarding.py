# mypy: ignore-errors
"""
test_transfer_port_forwarding.py

Phase 3 integration tests: port-forwarding round-trip through TransferJob.

Requires Docker CI stack (IN_DOCKER_CI=1):
    pseti test sw integration -k transfer_port_forwarding

Critical regression: the port_forwarding block in daq_config.json must
survive the full stop→TOML→daemon→rsync pipeline without being dropped.

Topology:
    head (10.0.1.x) ──socat──→ gateway (10.0.1.254:22xx) ──→ daqnode (192.168.0.10:22)

The tests assert:
  1. TransferNodeSpec.port_forwarding round-trips through TOML serialization.
  2. build_rsync_cmd() uses the gateway address and SSH port when PF is active.
  3. _process_job() passes the PF-aware rsync command (captured via mock).
  4. With PF active, rsync is NOT called with the raw daqnode IP.
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import sys
import tomllib
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from control.transfer.daemon import _process_job
from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.queue import TransferQueue
from control.transfer.rsync import build_rsync_cmd
from control.utils import config_file
from control.utils.pydantic_config_models import PortForwarding
from control.utils.run_state import RunStateManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAQNODE_IP = os.environ.get("DAQNODE_DIRECT_HOST", "192.168.0.10")
GATEWAY_IP = os.environ.get("DAQNODE_GATEWAY_HOST", "10.0.1.254")
HEAD_DATA_DIR = pathlib.Path(os.environ.get("HEAD_DATA_DIR", "/data/head"))
DAQ_DATA_DIR = pathlib.Path(os.environ.get("DAQ_DATA_DIR", "/data"))
PF_PORT = int(os.environ.get("DAQNODE_PF_PORT", "2200"))


# ---------------------------------------------------------------------------
# gRPC stub injection
# ---------------------------------------------------------------------------

@contextmanager
def _mock_grpc(mock_client: MagicMock):
    stub_root = ModuleType("panoseti_grpc")
    stub_daq = ModuleType("panoseti_grpc.daq_control")
    stub_cm = ModuleType("panoseti_grpc.daq_control.client")
    stub_cm.AsyncDaqControlClient = MagicMock(return_value=mock_client)
    stub_root.daq_control = stub_daq
    stub_daq.client = stub_cm
    injected = {
        "panoseti_grpc": stub_root,
        "panoseti_grpc.daq_control": stub_daq,
        "panoseti_grpc.daq_control.client": stub_cm,
    }
    prev = {k: sys.modules.get(k) for k in injected}
    sys.modules.update(injected)
    try:
        yield stub_cm.AsyncDaqControlClient
    finally:
        for k, orig in prev.items():
            if orig is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = orig


def _grpc_client_ok() -> MagicMock:
    c = MagicMock()
    c.__aenter__ = AsyncMock(return_value=c)
    c.__aexit__ = AsyncMock(return_value=None)
    c.GenerateManifest = AsyncMock(return_value={"success": True, "file_count": 0})
    c.CleanupData = AsyncMock(return_value={"success": True, "deleted_count": 0})
    return c


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pf() -> PortForwarding:
    return PortForwarding(status=True, gw_ip=GATEWAY_IP, ssh_port=PF_PORT)


@pytest.fixture
def pf_node(pf: PortForwarding) -> TransferNodeSpec:
    return TransferNodeSpec(
        ip_addr=DAQNODE_IP,
        username="panoseti",
        data_dir=str(DAQ_DATA_DIR),
        module_ids=[200],
        port_forwarding=pf,
    )


@pytest.fixture
def run_name() -> str:
    return f"ci_transfer_pf_{uuid.uuid4().hex[:8]}.pffd"


@pytest.fixture
def run_dir(run_name: str, head_data_dir: pathlib.Path) -> pathlib.Path:
    d = head_data_dir / run_name
    d.mkdir(parents=True, exist_ok=True)
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def pf_job(run_name: str, pf_node: TransferNodeSpec, head_data_dir: pathlib.Path) -> TransferJob:
    return TransferJob(schema_version=1, 
        run_name=run_name,
        head_data_dir=str(head_data_dir),
        head_node_username="panoseti",
        created_at=datetime.now(UTC),
        no_collect=False,
        no_cleanup=True,
        daq_nodes=[pf_node],
    )


# ---------------------------------------------------------------------------
# 1. Round-trip: port_forwarding survives TOML serialization
# ---------------------------------------------------------------------------

class TestPortForwardingRoundTrip:
    """port_forwarding must survive the stop → TOML → daemon pipeline."""

    def test_pf_survives_toml_roundtrip(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        pf_job: TransferJob,
    ) -> None:
        """Enqueue → read TOML → model_validate must preserve port_forwarding."""
        queue_dir = tmp_path / "state" / "transfer" / "queue"
        tq = TransferQueue(queue_dir=queue_dir)
        tq.enqueue(pf_job)
        pending = queue_dir / "pending" / f"{run_name}.job.toml"
        assert pending.exists()
        with open(pending, "rb") as f:
            data = tomllib.load(f)
        reloaded = TransferJob.model_validate(data)
        node = reloaded.daq_nodes[0]
        assert node.port_forwarding is not None, (
            "port_forwarding was silently dropped — this is the core regression!"
        )
        assert str(node.port_forwarding.gw_ip) == GATEWAY_IP
        assert node.port_forwarding.ssh_port == PF_PORT
        assert node.port_forwarding.status is True

    def test_no_pf_survives_toml_roundtrip(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
    ) -> None:
        """A node with port_forwarding=None must also round-trip cleanly."""
        job = TransferJob(schema_version=1, 
            run_name=run_name,
            head_data_dir=str(HEAD_DATA_DIR),
            head_node_username="panoseti",
            created_at=datetime.now(UTC),
            no_collect=True,
            no_cleanup=True,
            daq_nodes=[
                TransferNodeSpec(
                    ip_addr=DAQNODE_IP,
                    username="panoseti",
                    data_dir=str(DAQ_DATA_DIR),
                    module_ids=[200],
                    port_forwarding=None,
                )
            ],
        )
        queue_dir = tmp_path / "state" / "transfer" / "queue"
        tq = TransferQueue(queue_dir=queue_dir)
        tq.enqueue(job)
        pending = queue_dir / "pending" / f"{run_name}.job.toml"
        with open(pending, "rb") as f:
            data = tomllib.load(f)
        reloaded = TransferJob.model_validate(data)
        assert reloaded.daq_nodes[0].port_forwarding is None


# ---------------------------------------------------------------------------
# 2. build_rsync_cmd uses gateway address when PF is active
# ---------------------------------------------------------------------------

class TestBuildRsyncCmd:
    """build_rsync_cmd() produces PF-aware ssh arguments."""

    def test_pf_cmd_uses_gateway_address(
        self,
        pf_node: TransferNodeSpec,
        run_name: str,
    ) -> None:
        cmd = build_rsync_cmd(pf_node, run_name, str(HEAD_DATA_DIR / run_name))
        cmd_str = " ".join(cmd)
        assert GATEWAY_IP in cmd_str, (
            f"rsync command must use gateway IP {GATEWAY_IP}, got: {cmd_str}"
        )
        assert DAQNODE_IP not in cmd_str, (
            f"rsync must NOT use raw daqnode IP {DAQNODE_IP} when PF is active"
        )
        assert f"-p {PF_PORT}" in cmd_str or f"-p{PF_PORT}" in cmd_str, (
            f"rsync command must include SSH port {PF_PORT}, got: {cmd_str}"
        )

    def test_no_pf_cmd_uses_daqnode_address(
        self,
        run_name: str,
    ) -> None:
        node = TransferNodeSpec(
            ip_addr=DAQNODE_IP,
            username="panoseti",
            data_dir=str(DAQ_DATA_DIR),
            module_ids=[200],
            port_forwarding=None,
        )
        cmd = build_rsync_cmd(node, run_name, str(HEAD_DATA_DIR / run_name))
        cmd_str = " ".join(cmd)
        assert DAQNODE_IP in cmd_str
        assert GATEWAY_IP not in cmd_str

    def test_pf_disabled_falls_back_to_direct(
        self,
        run_name: str,
    ) -> None:
        """If port_forwarding.status=False, rsync uses the direct IP."""
        node = TransferNodeSpec(
            ip_addr=DAQNODE_IP,
            username="panoseti",
            data_dir=str(DAQ_DATA_DIR),
            module_ids=[200],
            port_forwarding=PortForwarding(status=False, gw_ip=GATEWAY_IP, ssh_port=PF_PORT),
        )
        cmd = build_rsync_cmd(node, run_name, str(HEAD_DATA_DIR / run_name))
        cmd_str = " ".join(cmd)
        assert DAQNODE_IP in cmd_str
        assert GATEWAY_IP not in cmd_str


# ---------------------------------------------------------------------------
# 3. _process_job with PF active calls rsync with the gateway address
# ---------------------------------------------------------------------------

class TestProcessJobWithPortForwarding:
    """_process_job passes PF-aware rsync command to subprocess."""

    async def test_process_job_rsync_uses_gateway(
        self,
        tmp_path: pathlib.Path,
        run_name: str,
        run_dir: pathlib.Path,
        pf_job: TransferJob,
        head_data_dir: pathlib.Path,
    ) -> None:
        """subprocess.run is called with the gateway address, not the daqnode IP."""
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("PSETI_CONTROL", str(tmp_path))
        # Ensure head_node_data_dir in daq_config matches our isolated path
        # so validator passes.
        daq_cfg = config_file.get_daq_config()
        daq_cfg.head_node_data_dir = str(head_data_dir)
        client = _grpc_client_ok()
        rsync_calls: list[list[str]] = []

        def _capture_rsync(cmd, **kwargs):
            rsync_calls.append(cmd)
            return MagicMock(returncode=0, stderr="")

        with _mock_grpc(client), \
             patch("control.transfer.daemon.asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_sub:
            mock_sub.side_effect = await _capture_rsync_async(rsync_calls, head_data_dir, pf_job.run_name)
            result, _ = await _process_job(pf_job, asyncio.Event(), RunStateManager())

        assert result is True
        assert rsync_calls, "rsync must have been called"
        for cmd in rsync_calls:
            cmd_str = " ".join(cmd)
            assert GATEWAY_IP in cmd_str, (
                f"rsync command must reference gateway {GATEWAY_IP}, got: {cmd_str}"
            )
            assert DAQNODE_IP not in cmd_str, (
                f"rsync must NOT use raw daqnode IP {DAQNODE_IP} when PF is active"
            )
        monkeypatch.undo()

async def _mock_subprocess_ok(*args, **kwargs):
    proc = MagicMock()
    proc.returncode = 0
    proc.wait = AsyncMock(return_value=0)
    proc.communicate = AsyncMock(return_value=(b"", b""))
    proc.stdout.readline = AsyncMock(return_value=b"")
    proc.stderr.read = AsyncMock(return_value=b"")
    return proc

async def _capture_rsync_async(rsync_calls: list, head_data_dir: pathlib.Path, run_name: str):
    async def _mock(*args, **kwargs):
        rsync_calls.append(args) # args[0] is the cmd list
        (head_data_dir / run_name).mkdir(parents=True, exist_ok=True)
        (head_data_dir / run_name / "dp_manifest.node_test.algo_blake3.txt").touch()
        return await _mock_subprocess_ok()
    return _mock
