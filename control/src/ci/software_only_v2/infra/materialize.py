"""
materialize.py — Write a Topology's 7 Pydantic configs to a PSETI_CONFIG directory.

All writes are via model_dump_json() — never raw dicts. The output directory
is the same one PanoPaths.config_dir() resolves to.
"""

from __future__ import annotations

import pathlib

from ci.software_only_v2.infra.spec import Topology
from control.utils.paths import PanoPaths

# Mapping of filename → Topology attribute name
_CONFIG_FILES = {
    "obs_config.json":     "obs",
    "daq_config.json":     "daq",
    "network_config.json": "network",
    "data_config.json":    "data",
    "firmware.json":       "firmware",
    "quabo_uids.json":     "quabo_uids",
    "daemons.json":        "daemons",
}


def write_all(topology: Topology, config_dir: pathlib.Path) -> None:
    """
    Serialize all 7 Pydantic config models to JSON files under config_dir.

    Args:
        topology: A validated Topology (from FleetSpec.build()).
        config_dir: The directory to write config files into (should match
                    PanoPaths.config_dir() in the test's env).
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    for filename, attr in _CONFIG_FILES.items():
        model = getattr(topology, attr)
        dest = config_dir / filename
        content = model.model_dump_json(indent=2)  # type: ignore[union-attr]
        dest.write_text(content)

        # CRITICAL: get_quabo_uids() expects quabo_uids.json in PanoPaths.tmp_dir()
        if filename == "quabo_uids.json":
            tmp_dest = PanoPaths.tmp_dir() / filename
            tmp_dest.write_text(content)


def read_back(config_dir: pathlib.Path) -> dict[str, dict]:
    """
    Read the 7 config files back as raw dicts (for assertion helpers).
    """
    import json
    result = {}
    for filename in _CONFIG_FILES:
        path = config_dir / filename
        if path.exists():
            result[filename] = json.loads(path.read_text())
    return result
