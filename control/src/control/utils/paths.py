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
      PSETI_STATE:       Directory for state/ tree (default: PSETI_CONTROL/state).
      PSETI_QUABOS:      Directory for Quabo metadata (default: PSETI_CONTROL/quabos).
      PSETI_LOGS:        Directory for system logs (default: PSETI_CONTROL/logs).
      PSETI_FIRMWARE:    Directory for firmware binaries (default: PSETI_CONTROL/firmware).
      PSETI_WR:          Directory for White Rabbit files (default: PSETI_CONTROL/wr).
      PSETI_DAQ_SCRIPTS: Directory for DAQ deployment scripts.
      PSETI_LOCKS_DIR:   Directory for lock files (default: PSETI_STATE/locks).
      PSETI_RUNS_DIR:    Directory for run state (default: PSETI_STATE/runs).
      PSETI_TQ_DIR:      Directory for transfer queue (default: PSETI_STATE/transfer/queue).
      PSETI_TM_DIR:      Directory for transfer manifests (default: PSETI_STATE/transfer/manifests).
      PSETI_CALIB_DIR:   Directory for calibration artifacts (default: PSETI_STATE/calibration).
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
    def state_dir(cls) -> pathlib.Path:
        """Directory for state/ tree (locks, runs, transfer, calibration)."""
        override = os.environ.get("PSETI_STATE")
        if override:
            return pathlib.Path(override).resolve()
        return cls.base_dir() / "state"

    @classmethod
    def locks_dir(cls) -> pathlib.Path:
        """Directory for lock files."""
        override = os.environ.get("PSETI_LOCKS_DIR")
        if override:
            return pathlib.Path(override).resolve()
        return cls.state_dir() / "locks"

    @classmethod
    def runs_dir(cls) -> pathlib.Path:
        """Directory for run state files."""
        override = os.environ.get("PSETI_RUNS_DIR")
        if override:
            return pathlib.Path(override).resolve()
        return cls.state_dir() / "runs"

    @classmethod
    def transfer_queue_dir(cls) -> pathlib.Path:
        """Directory for transfer queue (pending, active, completed, failed)."""
        override = os.environ.get("PSETI_TQ_DIR")
        if override:
            return pathlib.Path(override).resolve()
        return cls.state_dir() / "transfer" / "queue"

    @classmethod
    def transfer_manifests_dir(cls) -> pathlib.Path:
        """Directory for transfer manifests."""
        override = os.environ.get("PSETI_TM_DIR")
        if override:
            return pathlib.Path(override).resolve()
        return cls.state_dir() / "transfer" / "manifests"

    @classmethod
    def calibration_dir(cls) -> pathlib.Path:
        """Directory for calibration artifacts."""
        override = os.environ.get("PSETI_CALIB_DIR")
        if override:
            return pathlib.Path(override).resolve()
        return cls.state_dir() / "calibration"

    @classmethod
    def snapshots_dir(cls, run_name: str) -> pathlib.Path:
        """Directory for run-specific snapshots (e.g., config/hk snapshots)."""
        return cls.state_dir() / "snapshots" / run_name

    @classmethod
    def daemon_logs_dir(cls, name: str) -> pathlib.Path:
        """Directory for a specific daemon's logs."""
        return cls.state_dir() / "logs" / name

    @classmethod
    def calibration_file(cls, filename: str) -> pathlib.Path:
        """Return a fully-qualified Path for a calibration artifact."""
        return cls.calibration_dir() / filename

    @classmethod
    def ensure_state_dirs(cls) -> None:
        """Creates all state/ subdirectories."""
        for d in [
            cls.locks_dir(),
            cls.runs_dir(),
            cls.transfer_queue_dir() / "pending",
            cls.transfer_queue_dir() / "active",
            cls.transfer_queue_dir() / "completed",
            cls.transfer_queue_dir() / "failed",
            cls.transfer_manifests_dir(),
            cls.calibration_dir(),
        ]:
            d.mkdir(parents=True, exist_ok=True)
        # Create snapshots parent
        (cls.state_dir() / "snapshots").mkdir(parents=True, exist_ok=True)

    @classmethod
    def ensure_dirs(cls) -> None:
        """Creates transient workspace directories if they do not exist."""
        for d in [
            cls.tmp_dir(),
            cls.logs_dir(),
        ]:
            d.mkdir(parents=True, exist_ok=True)
        cls.ensure_state_dirs()
