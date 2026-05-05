"""
ci/fixtures/data_fixtures.py

Fixtures for generating schema-valid PanoSETI data (PFF) and simulating
DAQ node filesystems.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from control.utils.pydantic_config_models import DaqConfig

@pytest.fixture
def dummy_data_generator():
    """Returns a function that populates a directory with valid PFF data."""
    def _generate(
        dest_dir: Path, 
        run_name: str, 
        module_ids: list[int], 
        pff_count: int = 2,
        frame_count: int = 3
    ) -> None:
        from ci.conftest import make_minimal_pff_bytes
        
        # 1. Root run dir
        run_root = dest_dir / run_name
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "meta.json").write_text(json.dumps({"test_run": True}))
        
        # 2. Module-specific data
        for mid in module_ids:
            mod_run = dest_dir / f"module_{mid}" / run_name
            mod_run.mkdir(parents=True, exist_ok=True)
            
            for i in range(pff_count):
                pff_name = f"start_2024-01-01T00:00:00Z.dp_img16.bpp_2.module_{mid}.seqno_{i}.pff"
                pff_bytes = make_minimal_pff_bytes(n_frames=frame_count)
                (mod_run / pff_name).write_bytes(pff_bytes)
                
            (mod_run / "meta.json").write_text(json.dumps({"module_id": mid}))

    return _generate

@pytest.fixture
def mock_daq_filesystem(dummy_data_generator) -> Callable[[Path, str, DaqConfig], None]:
    """Populates an isolated directory structure mimicking a full observatory capture."""
    def _simulate(root: Path, run_name: str, daq_config: DaqConfig) -> None:
        for node in daq_config.daq_nodes:
            # We assume root is the base for all node data in the test
            dummy_data_generator(root, run_name, node.module_ids)
            
    return _simulate
