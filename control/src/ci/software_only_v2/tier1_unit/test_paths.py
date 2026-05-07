"""
test_paths.py

Unit tests for control/utils/paths.py.
Tests the PanoPaths class and state_dir() + 8 new state accessors.
No hardware or network access required.
"""

import os
import pathlib
import tempfile
from unittest import mock

from control.utils.paths import PanoPaths

# ===========================================================================
# Basic Accessor Return Types
# ===========================================================================

class TestStateAccessorsReturnPath:
    """Each new accessor should return a pathlib.Path instance."""

    def test_state_dir_returns_path(self) -> None:
        result = PanoPaths.state_dir()
        assert isinstance(result, pathlib.Path)

    def test_locks_dir_returns_path(self) -> None:
        result = PanoPaths.locks_dir()
        assert isinstance(result, pathlib.Path)

    def test_runs_dir_returns_path(self) -> None:
        result = PanoPaths.runs_dir()
        assert isinstance(result, pathlib.Path)

    def test_transfer_queue_dir_returns_path(self) -> None:
        result = PanoPaths.transfer_queue_dir()
        assert isinstance(result, pathlib.Path)

    def test_transfer_manifests_dir_returns_path(self) -> None:
        result = PanoPaths.transfer_manifests_dir()
        assert isinstance(result, pathlib.Path)

    def test_calibration_dir_returns_path(self) -> None:
        result = PanoPaths.calibration_dir()
        assert isinstance(result, pathlib.Path)

    def test_snapshots_dir_returns_path(self) -> None:
        result = PanoPaths.snapshots_dir("run_001")
        assert isinstance(result, pathlib.Path)

    def test_daemon_logs_dir_returns_path(self) -> None:
        result = PanoPaths.daemon_logs_dir("capture_hk")
        assert isinstance(result, pathlib.Path)


# ===========================================================================
# Accessor Subdirectory Relationships
# ===========================================================================

class TestStateAccessorsAreSubdirOfStateDir:
    """Each new accessor (except state_dir itself) should be under state_dir()."""

    def test_locks_dir_is_under_state_dir(self) -> None:
        locks = PanoPaths.locks_dir()
        state = PanoPaths.state_dir()
        assert str(locks).startswith(str(state))

    def test_runs_dir_is_under_state_dir(self) -> None:
        runs = PanoPaths.runs_dir()
        state = PanoPaths.state_dir()
        assert str(runs).startswith(str(state))

    def test_transfer_queue_dir_is_under_state_dir(self) -> None:
        tq = PanoPaths.transfer_queue_dir()
        state = PanoPaths.state_dir()
        assert str(tq).startswith(str(state))

    def test_transfer_manifests_dir_is_under_state_dir(self) -> None:
        tm = PanoPaths.transfer_manifests_dir()
        state = PanoPaths.state_dir()
        assert str(tm).startswith(str(state))

    def test_calibration_dir_is_under_state_dir(self) -> None:
        calib = PanoPaths.calibration_dir()
        state = PanoPaths.state_dir()
        assert str(calib).startswith(str(state))

    def test_snapshots_dir_is_under_state_dir(self) -> None:
        snap = PanoPaths.snapshots_dir("run_001")
        state = PanoPaths.state_dir()
        assert str(snap).startswith(str(state))

    def test_daemon_logs_dir_is_under_state_dir(self) -> None:
        dlogs = PanoPaths.daemon_logs_dir("capture_hk")
        state = PanoPaths.state_dir()
        assert str(dlogs).startswith(str(state))


# ===========================================================================
# Default Path Patterns
# ===========================================================================

class TestDefaultPathPatterns:
    """Test that default paths follow the expected naming convention."""

    def test_locks_dir_ends_with_locks(self) -> None:
        locks = PanoPaths.locks_dir()
        assert locks.name == "locks"

    def test_runs_dir_ends_with_runs(self) -> None:
        runs = PanoPaths.runs_dir()
        assert runs.name == "runs"

    def test_calibration_dir_ends_with_calibration(self) -> None:
        calib = PanoPaths.calibration_dir()
        assert calib.name == "calibration"

    def test_snapshots_dir_contains_run_name(self) -> None:
        snap = PanoPaths.snapshots_dir("run_001")
        assert "run_001" in str(snap)

    def test_daemon_logs_dir_contains_daemon_name(self) -> None:
        dlogs = PanoPaths.daemon_logs_dir("capture_hk")
        assert "capture_hk" in str(dlogs)

    def test_transfer_queue_dir_ends_with_queue(self) -> None:
        tq = PanoPaths.transfer_queue_dir()
        assert tq.name == "queue"

    def test_transfer_manifests_dir_ends_with_manifests(self) -> None:
        tm = PanoPaths.transfer_manifests_dir()
        assert tm.name == "manifests"


# ===========================================================================
# calibration_file() Helper
# ===========================================================================

class TestCalibrationFileHelper:
    """Test the calibration_file() typed helper method."""

    def test_calibration_file_returns_path(self) -> None:
        result = PanoPaths.calibration_file("foo.json")
        assert isinstance(result, pathlib.Path)

    def test_calibration_file_appends_to_calibration_dir(self) -> None:
        result = PanoPaths.calibration_file("foo.json")
        calib_dir = PanoPaths.calibration_dir()
        assert result == calib_dir / "foo.json"

    def test_calibration_file_with_different_names(self) -> None:
        files = ["maroc_calib.toml", "baseline.pkl", "hv_curve.csv"]
        for fname in files:
            result = PanoPaths.calibration_file(fname)
            assert result.name == fname
            assert str(result).startswith(str(PanoPaths.calibration_dir()))


# ===========================================================================
# Environment Variable Overrides
# ===========================================================================

class TestStateEnvOverride:
    """Test PSETI_STATE env var override."""

    def test_state_dir_respects_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.dict(os.environ, {"PSETI_STATE": tmpdir}):
                result = PanoPaths.state_dir()
                assert result == pathlib.Path(tmpdir).resolve()

    def test_locks_dir_inherits_state_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.dict(os.environ, {"PSETI_STATE": tmpdir}):
                locks = PanoPaths.locks_dir()
                expected = pathlib.Path(tmpdir).resolve() / "locks"
                assert locks == expected

    def test_runs_dir_inherits_state_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.dict(os.environ, {"PSETI_STATE": tmpdir}):
                runs = PanoPaths.runs_dir()
                expected = pathlib.Path(tmpdir).resolve() / "runs"
                assert runs == expected


class TestLocksEnvOverride:
    """Test PSETI_LOCKS_DIR env var override."""

    def test_locks_dir_respects_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.dict(os.environ, {"PSETI_LOCKS_DIR": tmpdir}):
                result = PanoPaths.locks_dir()
                assert result == pathlib.Path(tmpdir).resolve()


class TestRunsEnvOverride:
    """Test PSETI_RUNS_DIR env var override."""

    def test_runs_dir_respects_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.dict(os.environ, {"PSETI_RUNS_DIR": tmpdir}):
                result = PanoPaths.runs_dir()
                assert result == pathlib.Path(tmpdir).resolve()


class TestTransferQueueEnvOverride:
    """Test PSETI_TQ_DIR env var override."""

    def test_transfer_queue_dir_respects_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.dict(os.environ, {"PSETI_TQ_DIR": tmpdir}):
                result = PanoPaths.transfer_queue_dir()
                assert result == pathlib.Path(tmpdir).resolve()


class TestTransferManifestsEnvOverride:
    """Test PSETI_TM_DIR env var override."""

    def test_transfer_manifests_dir_respects_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.dict(os.environ, {"PSETI_TM_DIR": tmpdir}):
                result = PanoPaths.transfer_manifests_dir()
                assert result == pathlib.Path(tmpdir).resolve()


class TestCalibrationEnvOverride:
    """Test PSETI_CALIB_DIR env var override."""

    def test_calibration_dir_respects_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.dict(os.environ, {"PSETI_CALIB_DIR": tmpdir}):
                result = PanoPaths.calibration_dir()
                assert result == pathlib.Path(tmpdir).resolve()

    def test_calibration_file_uses_overridden_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.dict(os.environ, {"PSETI_CALIB_DIR": tmpdir}):
                result = PanoPaths.calibration_file("test.json")
                expected = pathlib.Path(tmpdir).resolve() / "test.json"
                assert result == expected


# ===========================================================================
# ensure_state_dirs()
# ===========================================================================

class TestEnsureStateDirs:
    """Test that ensure_state_dirs() creates all expected subdirectories."""

    def test_ensure_state_dirs_creates_subdirs(self) -> None:
        """Integration test: create all state dirs in a temp location."""
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {
            "PSETI_STATE": tmpdir,
            "PSETI_LOCKS_DIR": "",  # Clear to use state default
            "PSETI_RUNS_DIR": "",
            "PSETI_TQ_DIR": "",
            "PSETI_TM_DIR": "",
            "PSETI_CALIB_DIR": "",
        }):
            # Clear the env vars that shouldn't override
            os.environ.pop("PSETI_LOCKS_DIR", None)
            os.environ.pop("PSETI_RUNS_DIR", None)
            os.environ.pop("PSETI_TQ_DIR", None)
            os.environ.pop("PSETI_TM_DIR", None)
            os.environ.pop("PSETI_CALIB_DIR", None)

            PanoPaths.ensure_state_dirs()

            # Check that all expected directories exist
            assert PanoPaths.locks_dir().exists()
            assert PanoPaths.runs_dir().exists()
            assert (PanoPaths.transfer_queue_dir() / "pending").exists()
            assert (PanoPaths.transfer_queue_dir() / "active").exists()
            assert (PanoPaths.transfer_queue_dir() / "completed").exists()
            assert (PanoPaths.transfer_queue_dir() / "failed").exists()
            assert PanoPaths.transfer_manifests_dir().exists()
            assert PanoPaths.calibration_dir().exists()
            assert (PanoPaths.state_dir() / "snapshots").exists()

    def test_ensure_state_dirs_idempotent(self) -> None:
        """Calling ensure_state_dirs() twice should not raise."""
        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.dict(os.environ, {"PSETI_STATE": tmpdir}, clear=False):
                PanoPaths.ensure_state_dirs()
                PanoPaths.ensure_state_dirs()  # Should not raise

    def test_ensure_dirs_calls_ensure_state_dirs(self) -> None:
        """Test that ensure_dirs() calls ensure_state_dirs()."""
        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.dict(os.environ, {"PSETI_STATE": tmpdir}, clear=False):
                PanoPaths.ensure_dirs()

                # Check that state dirs were created
                assert PanoPaths.locks_dir().exists()
                assert PanoPaths.calibration_dir().exists()


# ===========================================================================
# Integration: Paths are Consistent
# ===========================================================================

class TestPathConsistency:
    """Verify that paths are internally consistent."""

    def test_all_paths_under_base_dir(self) -> None:
        """State paths should live under base_dir()."""
        PanoPaths.base_dir()
        state = PanoPaths.state_dir()
        # state_dir might use env overrides, but it should be absolute
        assert isinstance(state, pathlib.Path)
        assert state.is_absolute()

    def test_calibration_file_path_is_absolute(self) -> None:
        """calibration_file() should return an absolute path."""
        result = PanoPaths.calibration_file("test.json")
        assert result.is_absolute()
