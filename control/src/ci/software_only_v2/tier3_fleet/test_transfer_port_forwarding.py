"""
tier3_fleet/test_transfer_port_forwarding.py

Port-forwarding round-trip through the TransferJob pipeline.

Verifies:
1. TransferNodeSpec.port_forwarding survives TOML serialization.
2. build_rsync_cmd() uses the gateway address and SSH port when PF is active.
3. _process_job() passes the PF-aware rsync command via mock_rsync_transfer.

No Docker is required for these tests.

Ported from software_only/tier3_fleet/test_transfer_port_forwarding.py.
"""

from __future__ import annotations

import asyncio
import pathlib
import tomllib
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ci.fixtures.rsync_fixtures import RsyncMock
from ci.software_only_v2.infra.spec import FleetSpec
from ci.software_only_v2.infra.workspace import Workspace
from control.transfer.daemon import _process_job
from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.queue import TransferQueue
from control.transfer.rsync import build_rsync_cmd
from control.utils.pydantic_config_models import PortForwarding
from control.utils.run_state import RunStateManager

pytestmark = pytest.mark.tier3

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_DAQNODE_IP = "192.168.0.10"
_GATEWAY_IP = "10.0.1.254"
_DAQ_DATA_DIR = "/data"
_PF_PORT = 2200  # SSH forwarded port


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pf_node() -> TransferNodeSpec:
    return TransferNodeSpec(
        ip_addr=_DAQNODE_IP,
        username="panoseti",
        data_dir=_DAQ_DATA_DIR,
        module_ids=[200],
        port_forwarding=PortForwarding(status=True, gw_ip=_GATEWAY_IP, port=_PF_PORT),
    )


def _grpc_client_ok() -> MagicMock:
    c = MagicMock()
    c.__aenter__ = AsyncMock(return_value=c)
    c.__aexit__ = AsyncMock(return_value=None)
    c.GenerateManifest = AsyncMock(return_value={"success": True, "file_count": 0})
    c.CleanupData = AsyncMock(return_value={"success": True, "deleted_count": 0})
    return c


# ---------------------------------------------------------------------------
# 1. Round-trip: port_forwarding survives TOML serialization
# ---------------------------------------------------------------------------

class TestPortForwardingRoundTrip:
    """port_forwarding must survive the stop → TOML → daemon pipeline."""

    def test_when_pf_active_then_survives_toml_roundtrip(
        self,
        transfer_queue: TransferQueue,
    ) -> None:
        """Enqueue → read TOML → model_validate preserves port_forwarding."""
        run_name = f"ci_pf_{uuid.uuid4().hex[:8]}.pffd"
        job = TransferJob(
            schema_version=1,
            run_name=run_name,
            head_data_dir="/data/head",
            head_node_username="panoseti",
            created_at=datetime.now(UTC),
            daq_nodes=[_pf_node()],
        )
        tq = transfer_queue
        tq.enqueue(job)

        pending = tq._queue / "pending" / f"{run_name}.job.toml"
        assert pending.exists()
        with open(pending, "rb") as f:
            data = tomllib.load(f)

        reloaded = TransferJob.model_validate(data)
        node = reloaded.daq_nodes[0]
        assert node.port_forwarding is not None, (
            "port_forwarding was silently dropped — core regression"
        )
        assert str(node.port_forwarding.gw_ip) == _GATEWAY_IP
        assert node.port_forwarding.port == _PF_PORT
        assert node.port_forwarding.status is True

    def test_when_pf_none_then_toml_roundtrip_clean(
        self,
        transfer_queue: TransferQueue,
    ) -> None:
        """A node with port_forwarding=None round-trips cleanly."""
        run_name = f"ci_nopf_{uuid.uuid4().hex[:8]}.pffd"
        node = TransferNodeSpec(
            ip_addr=_DAQNODE_IP,
            username="panoseti",
            data_dir=_DAQ_DATA_DIR,
            module_ids=[200],
            port_forwarding=None,
        )
        job = TransferJob(
            schema_version=1,
            run_name=run_name,
            head_data_dir="/data/head",
            head_node_username="panoseti",
            created_at=datetime.now(UTC),
            daq_nodes=[node],
        )
        tq = transfer_queue
        tq.enqueue(job)

        pending = tq._queue / "pending" / f"{run_name}.job.toml"
        with open(pending, "rb") as f:
            data = tomllib.load(f)

        reloaded = TransferJob.model_validate(data)
        assert reloaded.daq_nodes[0].port_forwarding is None


# ---------------------------------------------------------------------------
# 2. build_rsync_cmd uses gateway address when PF is active
# ---------------------------------------------------------------------------

class TestBuildRsyncCmd:
    """build_rsync_cmd() produces PF-aware ssh arguments."""

    def test_when_pf_active_then_uses_gateway_address(self) -> None:
        """rsync command targets gateway IP, not daqnode IP."""
        run_name = "pf_test.pffd"
        head_dir = "/data/head/pf_test.pffd"
        cmd_str = " ".join(build_rsync_cmd(_pf_node(), run_name, head_dir))

        assert _GATEWAY_IP in cmd_str, (
            f"rsync must reference gateway {_GATEWAY_IP}: {cmd_str}"
        )
        assert _DAQNODE_IP not in cmd_str, (
            f"rsync must NOT use raw daqnode IP when PF is active: {cmd_str}"
        )
        assert f"-p {_PF_PORT}" in cmd_str or f"-p{_PF_PORT}" in cmd_str, (
            f"rsync must include SSH port {_PF_PORT}: {cmd_str}"
        )

    def test_when_pf_none_then_uses_daqnode_address(self) -> None:
        """Without port-forwarding, rsync targets the daqnode directly."""
        node = TransferNodeSpec(
            ip_addr=_DAQNODE_IP,
            username="panoseti",
            data_dir=_DAQ_DATA_DIR,
            module_ids=[200],
            port_forwarding=None,
        )
        run_name = "direct.pffd"
        cmd_str = " ".join(build_rsync_cmd(node, run_name, "/data/head/direct.pffd"))

        assert _DAQNODE_IP in cmd_str
        assert _GATEWAY_IP not in cmd_str

    def test_when_pf_disabled_then_falls_back_to_direct(self) -> None:
        """port_forwarding.status=False → rsync uses the direct IP."""
        node = TransferNodeSpec(
            ip_addr=_DAQNODE_IP,
            username="panoseti",
            data_dir=_DAQ_DATA_DIR,
            module_ids=[200],
            port_forwarding=PortForwarding(
                status=False, gw_ip=_GATEWAY_IP, port=_PF_PORT
            ),
        )
        run_name = "disabled_pf.pffd"
        cmd_str = " ".join(build_rsync_cmd(node, run_name, "/data/head/disabled_pf.pffd"))

        assert _DAQNODE_IP in cmd_str
        assert _GATEWAY_IP not in cmd_str


# ---------------------------------------------------------------------------
# 3. _process_job passes PF-aware rsync command to subprocess
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pseti_workspace", [FleetSpec.minimal_unit()], indirect=True)
class TestProcessJobWithPortForwarding:
    """_process_job passes the gateway address to rsync when PF is active."""

    @pytest.mark.asyncio
    async def test_when_pf_active_then_rsync_targets_gateway(
        self,
        pseti_workspace: Workspace,
        mock_rsync_transfer: RsyncMock,
    ) -> None:
        """subprocess.run is called with the gateway address, not the daqnode IP."""
        head_data_dir = pseti_workspace.root / "head_data"
        run_name = f"pf_job_{uuid.uuid4().hex[:8]}.pffd"
        (head_data_dir / run_name).mkdir(parents=True, exist_ok=True)

        pf_job = TransferJob(
            schema_version=1,
            run_name=run_name,
            head_data_dir=str(head_data_dir),
            head_node_username="panoseti",
            created_at=datetime.now(UTC),
            daq_nodes=[_pf_node()],
        )

        rsync_calls: list[list[str]] = []

        def rsync_side_effect(*args: object, **kwargs: object) -> None:
            rsync_calls.append(list(args[0]))  # type: ignore[arg-type]
            (head_data_dir / run_name / "dp_manifest.node_test.algo_blake3.txt").touch()

        mock_rsync_transfer.side_effect = rsync_side_effect

        mock_client = _grpc_client_ok()

        with patch(
            "panoseti_grpc.daq_control.client.AsyncDaqControlClient",
            return_value=mock_client,
        ):
            result, _ = await _process_job(
                pf_job, asyncio.Event(), RunStateManager()
            )

        assert result is True
        assert rsync_calls, "rsync must have been called at least once"

        for cmd in rsync_calls:
            cmd_str = " ".join(cmd)
            assert _GATEWAY_IP in cmd_str, (
                f"rsync must reference gateway {_GATEWAY_IP}: {cmd_str}"
            )
            assert _DAQNODE_IP not in cmd_str, (
                f"rsync must NOT use raw daqnode IP when PF active: {cmd_str}"
            )
