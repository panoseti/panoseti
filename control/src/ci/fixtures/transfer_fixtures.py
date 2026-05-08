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
def transfer_queue() -> TransferQueue:
    """Provides a TransferQueue instance in an isolated environment.
    Ensures the queue directory structure is fully initialized.
    """
    queue_dir = PanoPaths.transfer_queue_dir()
    queue_dir.mkdir(parents=True, exist_ok=True)
    return TransferQueue(queue_dir=queue_dir)

@pytest.fixture
def transfer_job_factory() -> Callable[..., TransferJob]:
    """Returns a factory function for creating valid TransferJob objects.
    Defaults are derived from the active PSETI_CONFIG.
    """
    def _make(
        run_name: str | None = None,
        head_data_dir: str | pathlib.Path | None = None,
        no_collect: bool = False,
        no_cleanup: bool = False,
        daq_nodes: list[TransferNodeSpec] | None = None,
        bwlimit: int | None = None,
        daq_config: Any | None = None,
    ) -> TransferJob:
        from control.utils import config_file

        if daq_nodes is None:
            # Use provided daq_config or load from disk
            cfg = daq_config or config_file.get_daq_config()
            daq_nodes = [
                TransferNodeSpec(
                    ip_addr=str(n.ip_addr),
                    username=n.username,
                    data_dir=str(n.data_dir),
                    module_ids=n.module_ids,
                    port_forwarding=n.port_forwarding
                )
                for n in cfg.daq_nodes
            ]

        # 2. Resolve Head Node Data Dir
        if head_data_dir is None:
            cfg = daq_config or config_file.get_daq_config()
            head_data_dir = cfg.head_node_data_dir

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
