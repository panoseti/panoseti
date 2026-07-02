"""Tests for ph_baseline_file_ok path resolution fix."""
import json
import os
import time

import pytest

import control.start as start_mod
from control.utils import config_file
from control.utils.paths import PanoPaths


def test_ph_baseline_file_ok_uses_calibration_dir(mock_workspace):
    """ph_baseline_file_ok() must probe calibration_dir, not CWD."""
    # File does NOT exist in calibration_dir → should return False
    result = start_mod.ph_baseline_file_ok()
    assert result is False

    # Create the file in calibration_dir → should return True
    calib_file = PanoPaths.calibration_file(config_file.quabo_ph_baseline_filename)
    test_data = {"date": "2024-01-01", "quabos": []}
    calib_file.write_text(json.dumps(test_data))

    result = start_mod.ph_baseline_file_ok()
    assert result is True


def test_ph_baseline_file_not_found_in_cwd(mock_workspace, tmp_path):
    """Regression: the file in CWD alone must NOT satisfy ph_baseline_file_ok()."""
    # Put file in a different (wrong) dir — simulates the old CWD bug
    wrong_dir = tmp_path / "wrong"
    wrong_dir.mkdir()
    cwd_file = wrong_dir / config_file.quabo_ph_baseline_filename
    test_data = {"date": "2024-01-01", "quabos": []}
    cwd_file.write_text(json.dumps(test_data))

    # Should still return False since calibration_dir doesn't have it
    result = start_mod.ph_baseline_file_ok()
    assert result is False


def test_ph_baseline_file_ok_checks_empty(mock_workspace):
    """Empty baseline file should return False."""
    # Create an empty file
    calib_file = PanoPaths.calibration_file(config_file.quabo_ph_baseline_filename)
    calib_file.write_text("")

    result = start_mod.ph_baseline_file_ok()
    assert result is False


def test_ph_baseline_file_ok_checks_age(mock_workspace):
    """Old baseline file (>24h) should return False."""
    # Create a valid file
    calib_file = PanoPaths.calibration_file(config_file.quabo_ph_baseline_filename)
    test_data = {"date": "2024-01-01", "quabos": []}
    calib_file.write_text(json.dumps(test_data))

    # Set mtime to 25 hours ago
    old_time = time.time() - (86400 + 3600)  # 25 hours
    os.utime(calib_file, (old_time, old_time))

    result = start_mod.ph_baseline_file_ok()
    assert result is False


def test_ph_baseline_file_ok_accepts_path_argument(mock_workspace, tmp_path):
    """ph_baseline_file_ok() should accept an explicit Path argument."""
    # Create a test file
    test_file = tmp_path / "test_baseline.json"
    test_data = {"date": "2024-01-01", "quabos": []}
    test_file.write_text(json.dumps(test_data))

    # Should work with explicit Path argument
    result = start_mod.ph_baseline_file_ok(test_file)
    assert result is True

    # Non-existent path should return False
    result = start_mod.ph_baseline_file_ok(tmp_path / "nonexistent.json")
    assert result is False


def test_get_quabo_ph_baselines_fallback(mock_workspace):
    """get_quabo_ph_baselines() should search calibration_dir then tmp_dir then config_dir."""
    # Set up directories
    calib_dir = PanoPaths.calibration_dir()
    tmp_dir = PanoPaths.tmp_dir()
    config_dir = PanoPaths.config_dir()

    # Test 1: File in tmp_dir (calibration_dir is empty)
    test_data = {"date": "2024-01-01", "quabos": [{"uid": "test", "coefs": [100] * 256}]}
    tmp_baseline = tmp_dir / config_file.quabo_ph_baseline_filename
    tmp_baseline.write_text(json.dumps(test_data))

    result = config_file.get_quabo_ph_baselines()
    assert result.date == test_data["date"]
    assert result.quabos[0].uid == test_data["quabos"][0]["uid"]
    assert result.quabos[0].coefs == test_data["quabos"][0]["coefs"]

    # Clean up
    tmp_baseline.unlink()

    # Test 2: File in calibration_dir (should prefer it over tmp_dir)
    calib_baseline = calib_dir / config_file.quabo_ph_baseline_filename
    calib_baseline.write_text(json.dumps(test_data))
    tmp_baseline.write_text(json.dumps({"date": "old", "quabos": []}))

    result = config_file.get_quabo_ph_baselines()
    assert result.date == test_data["date"]

    # Test 3: File not found anywhere should raise FileNotFoundError
    calib_baseline.unlink()
    tmp_baseline.unlink()

    with pytest.raises(FileNotFoundError):
        config_file.get_quabo_ph_baselines()
