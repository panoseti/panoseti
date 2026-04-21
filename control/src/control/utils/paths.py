from __future__ import annotations

import contextlib
import os
import pathlib


class PanoPaths:
    """
    Central utility for PANOSETI directory resolution.
    Supports environment variable overrides for custom workspace layouts.
    """

    @classmethod
    def software_root_dir(cls) -> pathlib.Path:
        """The root panoseti-software directory."""
        # 1. Respect environment override
        override = os.environ.get("PANOSETI_SOFTWARE_REPO_ROOT")
        if override:
            return pathlib.Path(override).resolve()

        # 2. If PANOSETI_CONTROL_ROOT is /app (Docker), root is /
        home = os.environ.get("PANOSETI_CONTROL_ROOT")
        if home == "/app":
            return pathlib.Path("/")

        # 3. Default: File is at control/src/control/utils/paths.py
        # root is 5 levels up
        return pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve()

    @classmethod
    def base_dir(cls) -> pathlib.Path:
        """The control package root directory. Defaults to software_root / 'control'."""
        override = os.environ.get("PANOSETI_CONTROL_ROOT")
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
        override = os.environ.get("PANOSETI_CONFIG_DIR")
        if override:
            return pathlib.Path(override).resolve()
        return cls.base_dir() / "configs"

    @classmethod
    def tmp_dir(cls) -> pathlib.Path:
        """Directory for transient files (locks, run state, UIDs)."""
        override = os.environ.get("PANOSETI_TMP_DIR")
        if override:
            return pathlib.Path(override).resolve()
        return cls.base_dir() / "tmp"

    @classmethod
    def quabos_dir(cls) -> pathlib.Path:
        """Directory for hardware-specific metadata (quabo_info, detector_info)."""
        override = os.environ.get("PANOSETI_QUABOS_DIR")
        if override:
            return pathlib.Path(override).resolve()
        return cls.base_dir() / "quabos"

    @classmethod
    def logs_dir(cls) -> pathlib.Path:
        """Directory for system log files."""
        override = os.environ.get("PANOSETI_LOGS_DIR")
        if override:
            return pathlib.Path(override).resolve()
        return cls.base_dir() / "logs"

    @classmethod
    def firmware_dir(cls) -> pathlib.Path:
        """Directory containing Quabo firmware binaries."""
        override = os.environ.get("PANOSETI_FIRMWARE_DIR")
        if override:
            return pathlib.Path(override).resolve()
        return cls.base_dir() / "firmware"

    @classmethod
    def wr_dir(cls) -> pathlib.Path:
        """Directory containing White Rabbit configuration and filesystem files."""
        override = os.environ.get("PANOSETI_WR_DIR")
        if override:
            return pathlib.Path(override).resolve()
        return cls.base_dir() / "wr"

    @classmethod
    def daq_scripts_dir(cls) -> pathlib.Path:
        """Directory containing scripts to be deployed to DAQ nodes."""
        override = os.environ.get("PANOSETI_DAQ_SCRIPTS_DIR")
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
