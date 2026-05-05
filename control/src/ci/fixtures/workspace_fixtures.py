"""
ci/fixtures/workspace_fixtures.py

Fixtures for managing the filesystem workspace, PanoPaths environment variables,
and run directory structures.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import DaqConfig, DataConfig, NetworkConfig, ObsConfig, QuaboUids

logger = logging.getLogger(__name__)

@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Centralized environment variable overrides for testing.
    
    Ensures all PSETI_* variables point to isolated subdirectories within tmp_path.
    """
    state_tmp = tmp_path / "state"
    ctl_tmp = tmp_path / "control"
    cfg_tmp = ctl_tmp / "configs"
    tmp_tmp = tmp_path / "tmp"
    quabos_tmp = tmp_path / "quabos"
    logs_tmp = tmp_path / "logs"
    firmware_tmp = tmp_path / "firmware"
    
    for d in [state_tmp, ctl_tmp, cfg_tmp, tmp_tmp, quabos_tmp, logs_tmp, firmware_tmp]:
        d.mkdir(parents=True, exist_ok=True)
        
    monkeypatch.setenv("PSETI_STATE", str(state_tmp))
    monkeypatch.setenv("PSETI_CONTROL", str(ctl_tmp))
    monkeypatch.setenv("PSETI_CONFIG", str(cfg_tmp))
    monkeypatch.setenv("PSETI_TMP", str(tmp_tmp))
    monkeypatch.setenv("PSETI_QUABOS", str(quabos_tmp))
    monkeypatch.setenv("PSETI_LOGS", str(logs_tmp))
    monkeypatch.setenv("PSETI_FIRMWARE", str(firmware_tmp))
    
    # Initialize basic state directories (locks, runs, transfer queue, etc.)
    PanoPaths.ensure_dirs()
    
    return None

@pytest.fixture
def mock_workspace(mock_env, mock_obs_config, mock_daq_config, mock_network_config) -> Path:
    """Provides a complete, schema-valid temporary environment with default configs.
    
    Returns the path to the temporary configs directory.
    """
    cfg_dir = PanoPaths.config_dir()
    
    (cfg_dir / "obs_config.json").write_text(mock_obs_config.model_dump_json())
    (cfg_dir / "daq_config.json").write_text(mock_daq_config.model_dump_json())
    (cfg_dir / "network_config.json").write_text(mock_network_config.model_dump_json())
    
    # Default data config
    data_cfg = {
        "run_type": "sci",
        "image": {
            "integration_time_usec": 1000, 
            "pe_threshold": 1.0, 
            "quabo_sample_size": 16
        }
    }
    (cfg_dir / "data_config.json").write_text(json.dumps(data_cfg))
    
    # Default UIDs (empty)
    uids_cfg = {"domes": []}
    (PanoPaths.tmp_dir() / "quabo_uids.json").write_text(json.dumps(uids_cfg))
    
    return cfg_dir

@pytest.fixture
def setup_test_run_dir() -> Callable[[str, ObsConfig, DaqConfig, QuaboUids, DataConfig, NetworkConfig], None]:
    """Fixture to create hierarchical run directories using production logic.
    
    Returns the start.make_run_dirs function.
    """
    from control.start import make_run_dirs
    return make_run_dirs
