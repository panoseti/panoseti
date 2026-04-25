"""
ci/fixtures/factories.py

Shared factory functions for creating mock observatory objects and configurations.
"""

from __future__ import annotations

import datetime
import os
import pathlib
import uuid
from typing import Any

from control.transfer.models import TransferJob, TransferNodeSpec
from control.utils.pydantic_config_models import DaqConfig, NetworkConfig, ObsConfig

def make_transfer_job(
    run_name: str | None = None,
    head_data_dir: str | pathlib.Path = "/data/head",
    daq_nodes: list[TransferNodeSpec] | None = None
) -> TransferJob:
    """Yields a valid TransferJob Pydantic model with randomized defaults."""
    return TransferJob(
        schema_version=1,
        run_name=run_name or f"test_run_{uuid.uuid4().hex[:8]}.pffd",
        head_data_dir=str(head_data_dir),
        head_node_username="panoseti",
        created_at=datetime.datetime.now(datetime.UTC),
        daq_nodes=daq_nodes or [
            TransferNodeSpec(
                ip_addr="192.168.0.10",
                username="root",
                data_dir="/data",
                module_ids=[201, 254]
            )
        ]
    )

def simulate_daq_filesystem(
    root: pathlib.Path,
    run_name: str,
    module_ids: list[int],
    pff_count: int = 2
) -> None:
    """
    Populates a directory structure mimicking a DAQ node after a run.
    Satisfies GenerateManifestModel validation.
    """
    # Root run dir (metadata)
    root_run = root / run_name
    root_run.mkdir(parents=True, exist_ok=True)
    
    # Module-specific science data
    for mid in module_ids:
        mod_run = root / f"module_{mid}" / run_name
        mod_run.mkdir(parents=True, exist_ok=True)
        for i in range(pff_count):
            (mod_run / f"data_seq{i}.pff").write_bytes(os.urandom(128))
        # Metadata files
        (mod_run / "meta.json").write_text("{}")

def make_mock_daq_config(
    head_dir: str = "/data/head",
    node_ips: list[str] | None = None
) -> DaqConfig:
    """Builds a valid DaqConfig with n nodes."""
    ips = node_ips or ["192.168.0.10"]
    return DaqConfig(
        head_node_data_dir=head_dir,
        head_node_ip_addr="10.0.1.5",
        head_node_container=True,
        daq_nodes=[
            {
                "username": "root",
                "data_dir": "/data",
                "ip_addr": ip,
                "module_ids": [201 + i, 254 + i],
                "bindhost": "lo"
            }
            for i, ip in enumerate(ips)
        ]
    )
