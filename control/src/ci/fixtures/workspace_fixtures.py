"""
ci/fixtures/workspace_fixtures.py

Fixtures for managing the filesystem workspace, PanoPaths environment variables,
and run directory structures.
"""

from __future__ import annotations

import os
import json
import logging
from collections.abc import Callable
from pathlib import Path
import contextlib
from ci.fixtures.fleet import Fleet

import pytest

from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import (
    DaqConfig,
    DataConfig,
    NetworkConfig,
    ObsConfig,
    QuaboUids,
)

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


def _prepare_container_dirs(fleet: Fleet, run_dir: str) -> None:
    """Create data directories in the ephemeral temp dirs used by containers."""
    for i, temp_dir in enumerate(fleet._temp_dirs):
        host_root = Path(temp_dir)
        spec = fleet.specs[i]

        # Root run dir (e.g. /data/ci_run_xxx.pffd/)
        main_run_dir = host_root / run_dir
        main_run_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(main_run_dir, 0o777)

        # Touch metadata logs to satisfy verification. Name must include node name.
        (main_run_dir / f"hp_stdout_{spec.name}.log").touch()
        (main_run_dir / "meta.json").write_text('{"test": true}')

        # Module subdirs (e.g. /data/module_250/ci_run_xxx.pffd/)
        for mid in spec.module_ids:
            mod_root = host_root / f"module_{mid}"
            mod_root.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(mod_root, 0o777)

            mod_run_dir = mod_root / run_dir
            mod_run_dir.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(mod_run_dir, 0o777)

            # Dummy data - name must match what GenerateManifest picks up
            f_path = mod_run_dir / f"data.module_{mid}.pff"
            f_path.write_bytes(b"synthetic data")
            with contextlib.suppress(OSError):
                os.chmod(f_path, 0o666)


def copy_run_dir(fleet: Fleet, run_dir: str, head_data_dir: Path) -> bool:
    """Mock rsync by copying from all isolated container volumes to head node.
    Simulates the inclusive rsync which pulls from both root and module directories.
    """
    dest_run = head_data_dir / run_dir
    dest_run.mkdir(parents=True, exist_ok=True)
    # Create a dummy manifest so the VERIFYING stage doesn't fail
    (dest_run / "dp_manifest.node_mock.algo_blake3.txt").write_text("")

    success = False

    import shutil
    for i, temp_dir in enumerate(fleet._temp_dirs):
        host_root = Path(temp_dir)
        spec = fleet.specs[i]
        # 1. Simulate root run dir transfer
        src_root = host_root / run_dir
        if src_root.is_dir():
            # Copy contents of root run dir into dest_run
            for item in src_root.iterdir():
                dest_path = dest_run / item.name
                if item.is_dir():
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(item, dest_path)
                else:
                    shutil.copy2(item, dest_path)
            success = True

        # 2. Simulate module run dir transfer (flattened)
        for mid in spec.module_ids:
            src_mod = host_root / f"module_{mid}" / run_dir
            if src_mod.is_dir():
                for item in src_mod.iterdir():
                    if item.is_file():
                        shutil.copy2(item, dest_run / item.name)
                success = True
    return success
