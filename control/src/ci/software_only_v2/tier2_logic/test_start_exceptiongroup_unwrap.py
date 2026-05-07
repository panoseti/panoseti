"""
test_start_exceptiongroup_unwrap.py — ExceptionGroup handling in start.py.

Ported from ci/software_only/tier2_logic/test_start_exceptiongroup_unwrap.py.
"""

from __future__ import annotations

import json
import logging
import os
from unittest.mock import MagicMock, patch

import anyio
import pytest

from control.start import (
    StartTransaction,
    _check_quabo_reachability,
)
from control.utils.run_state import RunStateManager, ValidationError
from ci.software_only_v2.infra.spec import FleetSpec
from ci.software_only_v2.infra.workspace import Workspace


@pytest.fixture
def mock_quabo_uids() -> MagicMock:
    module = MagicMock()
    module.ip_addr = "192.168.1.10"
    quabo = MagicMock()
    quabo.uid = "q1"
    module.quabos = [quabo, MagicMock(uid=""), MagicMock(uid=""), MagicMock(uid="")]
    dome = MagicMock()
    dome.modules = [module]
    uids = MagicMock()
    uids.domes = [dome]
    return uids


@pytest.mark.asyncio
async def test_check_quabo_reachability_unwraps_exceptions(
    mock_quabo_uids: MagicMock,
) -> None:
    """_check_quabo_reachability collects per-quabo failures into a single ValidationError."""
    module2 = MagicMock()
    module2.ip_addr = "192.168.1.11"
    quabo2 = MagicMock()
    quabo2.uid = "q2"
    module2.quabos = [quabo2, MagicMock(uid=""), MagicMock(uid=""), MagicMock(uid="")]
    mock_quabo_uids.domes[0].modules.append(module2)

    network_config = MagicMock()

    with patch("control.utils.config_validator._check_reachability") as mock_reach:
        mock_reach.side_effect = [
            (False, "port closed"),
            (False, "UDP timeout"),
        ]

        with pytest.raises(ValidationError) as exc_info:
            await _check_quabo_reachability(mock_quabo_uids, network_config, lenient=False)

        msg = str(exc_info.value)
        assert "port closed" in msg
        assert "UDP timeout" in msg


@pytest.mark.parametrize(
    "pseti_workspace",
    [FleetSpec.minimal_unit()],
    indirect=True,
)
class TestStartTransactionExceptionGroupFormatting:
    @pytest.mark.asyncio
    async def test_start_transaction_aexit_formats_exception_group(
        self,
        pseti_workspace: Workspace,
        mock_quabo_uids: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """__aexit__ with ExceptionGroup logs each sub-exception and writes a JSON context file."""
        head_data_dir = str(pseti_workspace.root / "head_data")
        # Build a DaqConfig with head_node_data_dir pointing to a writable temp location.
        cfg = pseti_workspace.topology.daq.model_copy(
            update={"head_node_data_dir": head_data_dir}
        )

        try:
            raise ExceptionGroup("test group", [
                ValueError("sub1"),
                TypeError("sub2"),
            ])
        except ExceptionGroup as eg:
            exc_val = eg
            exc_tb = eg.__traceback__

        state_mgr = RunStateManager()
        tx = StartTransaction(state_mgr, "r1", cfg, mock_quabo_uids, MagicMock())

        with caplog.at_level(logging.ERROR):
            await tx.__aexit__(type(exc_val), exc_val, exc_tb)

        assert any("sub1" in record.message for record in caplog.records)
        assert any("sub2" in record.message for record in caplog.records)
        assert any("Traceback" in record.message for record in caplog.records)

        aborted_dir = pseti_workspace.root / "head_data" / "_aborted" / "r1"
        context_file = aborted_dir / "start_failure_context.json"
        assert context_file.exists()

        data = json.loads(await anyio.Path(context_file).read_text())
        assert "sub1" in data["traceback"]
        assert "sub2" in data["traceback"]
        assert "Traceback" in data["traceback"]
