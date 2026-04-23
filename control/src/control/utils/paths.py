from __future__ import annotations

import contextlib
import os
import pathlib


class PanoPaths:
    """
    Central utility for PSETI directory resolution.
    Supports environment variable overrides for custom workspace layouts.
    
    Overrideable Variables:
      PSETI_ROOT:        Root of the panoseti-software repository.
      PSETI_CONTROL:     Root of the control package (default: PSETI_ROOT/control).
      PSETI_CONFIG:      Directory for JSON configs (default: PSETI_CONTROL/configs).
      PSETI_TMP:         Directory for transient files (default: PSETI_CONTROL/tmp).
      PSETI_QUABOS:      Directory for Quabo metadata (default: PSETI_CONTROL/quabos).
      PSETI_LOGS:        Directory for system logs (default: PSETI_CONTROL/logs).
      PSETI_FIRMWARE:    Directory for firmware binaries (default: PSETI_CONTROL/firmware).
      PSETI_WR:          Directory for White Rabbit files (default: PSETI_CONTROL/wr).
      PSETI_DAQ_SCRIPTS: Directory for DAQ deployment scripts.
    """

    @classmethod
    def software_root_dir(cls) -> pathlib.Path:
        """The root panoseti-software directory."""
        # 1. Respect environment override
        override = os.environ.get("PSETI_ROOT")
        if override:
            return pathlib.Path(override).resolve()

        # 2. If PSETI_CONTROL is /app (Docker), root is /
        home = os.environ.get("PSETI_CONTROL")
        if home == "/app":
            return pathlib.Path("/")

        # 3. Default: File is at control/src/control/utils/paths.py
        # root is 5 levels up
        return pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve()

    @classmethod
    def base_dir(cls) -> pathlib.Path:
        """The control package root directory. Defaults to software_root / 'control'."""
        override = os.environ.get("PSETI_CONTROL")
        if override:
            with contextlib.suppress(Exception):
                return pathlib.Path(override).resolve()
        
        # In Docker, we often mount control at /app. 
        # If software_root is /, base_dir should be /app if it exists.
        root = cls.software_root_dir()
        if root == pathlib.Path("/") and pathlib.Path("/app").exists():
            return pathlib.Path("/app")

        return root / "control"

    @classmethod
    def config_dir(cls) -> pathlib.Path:
        """Directory containing observatory/daq/data JSON configs."""
        override = os.environ.get("PSETI_CONFIG")
        if override:
            return pathlib.Path(override).resolve()
        return cls.base_dir() / "configs"

    @classmethod
    def tmp_dir(cls) -> pathlib.Path:
        """Directory for transient files (locks, run state, UIDs)."""
        override = os.environ.get("PSETI_TMP")
        if override:
            return pathlib.Path(override).resolve()
        return cls.base_dir() / "tmp"

    @classmethod
    def quabos_dir(cls) -> pathlib.Path:
        """Directory for hardware-specific metadata (quabo_info, detector_info)."""
        override = os.environ.get("PSETI_QUABOS")
        if override:
            return pathlib.Path(override).resolve()
        return cls.base_dir() / "quabos"

    @classmethod
    def logs_dir(cls) -> pathlib.Path:
        """Directory for system log files."""
        override = os.environ.get("PSETI_LOGS")
        if override:
            return pathlib.Path(override).resolve()
        return cls.base_dir() / "logs"

    @classmethod
    def firmware_dir(cls) -> pathlib.Path:
        """Directory containing Quabo firmware binaries."""
        override = os.environ.get("PSETI_FIRMWARE")
        if override:
            return pathlib.Path(override).resolve()
        return cls.base_dir() / "firmware"

    @classmethod
    def wr_dir(cls) -> pathlib.Path:
        """Directory containing White Rabbit configuration and filesystem files."""
        override = os.environ.get("PSETI_WR")
        if override:
            return pathlib.Path(override).resolve()
        return cls.base_dir() / "wr"

    @classmethod
    def daq_scripts_dir(cls) -> pathlib.Path:
        """Directory containing scripts to be deployed to DAQ nodes."""
        override = os.environ.get("PSETI_DAQ_SCRIPTS")
        if override:
            return pathlib.Path(override).resolve()
        return cls.base_dir() / "src/control/daq_scripts"

    @classmethod
    def tools_dir(cls) -> pathlib.Path:
        """Directory containing control plane utility scripts."""
        return cls.base_dir() / "src/control/tools"

    @classmethod
    def daemons_dir(cls) -> pathlib.Path:
        """Directory containing background service daemons."""
        return cls.base_dir() / "src/control/daemons"

    @classmethod
    def ensure_dirs(cls) -> None:
        """Creates transient workspace directories if they do not exist."""
        for d in [
            cls.tmp_dir(),
            cls.logs_dir(),
        ]:
            d.mkdir(parents=True, exist_ok=True)
