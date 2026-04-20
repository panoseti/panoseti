from __future__ import annotations

import os
import pathlib


class PanoPaths:
    """
    Central utility for PANOSETI directory resolution.
    Supports environment variable overrides for custom workspace layouts.
    """

    @classmethod
    def base_dir(cls) -> pathlib.Path:
        """The root PANOSETI workspace directory. Defaults to CWD."""
        return pathlib.Path(os.environ.get("PANOSETI_HOME", os.getcwd())).resolve()

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
    def ensure_dirs(cls) -> None:
        """Creates standard workspace directories if they do not exist."""
        for d in [cls.config_dir(), cls.tmp_dir(), cls.quabos_dir(), cls.logs_dir()]:
            d.mkdir(parents=True, exist_ok=True)
