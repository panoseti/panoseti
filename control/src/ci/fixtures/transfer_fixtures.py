"""
ci/fixtures/transfer_fixtures.py

Fixtures for managing TransferJob creation and TransferQueue environment setup.
Consolidates repetitive boilerplate found in transfer integration tests.
"""

from __future__ import annotations

import datetime
import pathlib
import uuid
from collections.abc import Callable
from typing import Any

import pytest

from control.transfer.models import TransferJob, TransferNodeSpec
from control.transfer.queue import TransferQueue
from control.utils.paths import PanoPaths


@pytest.fixture
def transfer_queue(mock_env) -> TransferQueue:
    """Provides a TransferQueue instance in an isolated environment.
    Ensures the queue directory structure is fully initialized.
    """
    queue_dir = PanoPaths.transfer_queue_dir()
    queue_dir.mkdir(parents=True, exist_ok=True)
    return TransferQueue(queue_dir=queue_dir)

@pytest.fixture
def transfer_job_factory(no_collect: bool=False, no_cleanup: bool=False) -> Callable[..., TransferJob]:
    """Returns a factory function for creating valid TransferJob objects.
    Defaults are derived from the active PSETI_CONFIG (set by mock_env or auto_isolate).
    """
    def _make(
        run_name: str | None = None,
        head_data_dir: str | pathlib.Path | None = None,
        no_collect: bool = no_collect,
        no_cleanup: bool = no_cleanup,
        daq_nodes: list[TransferNodeSpec] | None = None,
        bwlimit: int | None = None,
    ) -> TransferJob:
        from control.utils import config_file

        # 1. Resolve DAQ nodes from topology if not provided
        if daq_nodes is None:
            # We use the raw DaqConfig to get the most accurate PortForwarding etc.
            # but mapping it to TransferNodeSpec.
            daq_config = config_file.get_daq_config()
            daq_nodes = [
                TransferNodeSpec(
                    ip_addr=str(n.ip_addr),
                    username=n.username,
                    data_dir=str(n.data_dir),
                    module_ids=n.module_ids,
                    port_forwarding=n.port_forwarding
                )
                for n in daq_config.daq_nodes
            ]

        # 2. Resolve Head Node Data Dir
        if head_data_dir is None:
            head_data_dir = config_file.get_daq_config().head_node_data_dir

        return TransferJob(
            schema_version=1,
            run_name=run_name or f"ci_job_{uuid.uuid4().hex[:8]}.pffd",
            head_data_dir=str(head_data_dir),
            head_node_username="panoseti",
            created_at=datetime.datetime.now(datetime.UTC),
            daq_nodes=daq_nodes,
            no_collect=no_collect,
            no_cleanup=no_cleanup,
            bwlimit=bwlimit,
        )

    return _make

@pytest.fixture
def transfer_job(transfer_job_factory) -> TransferJob:
    """Provides a default TransferJob for the current topology."""
    return transfer_job_factory()

@pytest.fixture
def isolated_transfer_env(mock_workspace, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, session_fleet: Any):
    """Sets up an isolated environment for transfer tests, including head node data dir."""
    _fleet, daq_cfg_dict = session_fleet
    
    head_data_dir = tmp_path / "head_data"
    head_data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HEAD_DATA_DIR", str(head_data_dir))
    
    from control.utils import config_file
    daq_config = config_file.DaqConfig.model_validate(daq_cfg_dict)
    daq_config.head_node_data_dir = str(head_data_dir)
    
    config_dir = PanoPaths.config_dir()
    (config_dir / "daq_config.json").write_text(daq_config.model_dump_json())
    
    return head_data_dir, daq_config
