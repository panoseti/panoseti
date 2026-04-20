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
        # File is at control/src/control/utils/paths.py
        return pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve()

    @classmethod
    def base_dir(cls) -> pathlib.Path:
        """The control package root directory. Defaults to software_root / 'control'."""
        override = os.environ.get("PANOSETI_HOME")
        if override:
            with contextlib.suppress(Exception):
                return pathlib.Path(override).resolve()
        return cls.software_root_dir() / "control"

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
    def ensure_dirs(cls) -> None:
        """Creates transient workspace directories if they do not exist."""
        for d in [
            cls.tmp_dir(),
            cls.logs_dir(),
        ]:
            d.mkdir(parents=True, exist_ok=True)
