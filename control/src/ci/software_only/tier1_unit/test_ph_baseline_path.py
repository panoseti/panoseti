"""Tests for ph_baseline_file_ok path resolution fix."""
import json
import time

import pytest


def test_ph_baseline_file_ok_uses_calibration_dir(tmp_path, monkeypatch):
    """ph_baseline_file_ok() must probe calibration_dir, not CWD."""
    monkeypatch.setenv("PSETI_CALIB_DIR", str(tmp_path))

    # Reload modules to pick up the environment variable
    import importlib

    import control.start as start_mod
    importlib.reload(start_mod)
    from control.utils import config_file

    # File does NOT exist in tmp_path → should return False
    result = start_mod.ph_baseline_file_ok()
    assert result is False

    # Create the file in calibration_dir → should return True
    calib_file = tmp_path / config_file.quabo_ph_baseline_filename
    test_data = {"date": "2024-01-01", "quabos": []}
    calib_file.write_text(json.dumps(test_data))

    # Reload to clear any cached state
    importlib.reload(start_mod)
    result = start_mod.ph_baseline_file_ok()
    assert result is True


def test_ph_baseline_file_not_found_in_cwd(tmp_path, monkeypatch):
    """Regression: the file in CWD alone must NOT satisfy ph_baseline_file_ok()."""
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    monkeypatch.setenv("PSETI_CALIB_DIR", str(calib_dir))

    import importlib

    import control.start as start_mod
    importlib.reload(start_mod)
    from control.utils import config_file

    # Put file in a different (wrong) dir — simulates the old CWD bug
    wrong_dir = tmp_path / "wrong"
    wrong_dir.mkdir()
    cwd_file = wrong_dir / config_file.quabo_ph_baseline_filename
    test_data = {"date": "2024-01-01", "quabos": []}
    cwd_file.write_text(json.dumps(test_data))

    # Should still return False since calibration_dir doesn't have it
    result = start_mod.ph_baseline_file_ok()
    assert result is False


def test_ph_baseline_file_ok_checks_empty(tmp_path, monkeypatch):
    """Empty baseline file should return False."""
    monkeypatch.setenv("PSETI_CALIB_DIR", str(tmp_path))

    import importlib

    import control.start as start_mod
    importlib.reload(start_mod)
    from control.utils import config_file

    # Create an empty file
    calib_file = tmp_path / config_file.quabo_ph_baseline_filename
    calib_file.write_text("")

    result = start_mod.ph_baseline_file_ok()
    assert result is False


def test_ph_baseline_file_ok_checks_age(tmp_path, monkeypatch):
    """Old baseline file (>24h) should return False."""
    monkeypatch.setenv("PSETI_CALIB_DIR", str(tmp_path))

    import importlib

    import control.start as start_mod
    importlib.reload(start_mod)
    from control.utils import config_file

    # Create a valid file
    calib_file = tmp_path / config_file.quabo_ph_baseline_filename
    test_data = {"date": "2024-01-01", "quabos": []}
    calib_file.write_text(json.dumps(test_data))

    # Set mtime to 25 hours ago
    old_time = time.time() - (86400 + 3600)  # 25 hours
    import os
    os.utime(calib_file, (old_time, old_time))

    result = start_mod.ph_baseline_file_ok()
    assert result is False


def test_ph_baseline_file_ok_accepts_path_argument(tmp_path):
    """ph_baseline_file_ok() should accept an explicit Path argument."""
    import importlib

    import control.start as start_mod
    importlib.reload(start_mod)

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


def test_get_quabo_ph_baselines_fallback(tmp_path, monkeypatch):
    """get_quabo_ph_baselines() should search calibration_dir then tmp_dir then config_dir."""
    from control.utils import config_file

    # Set up directories
    calib_dir = tmp_path / "calib"
    tmp_dir = tmp_path / "tmp_files"
    config_dir = tmp_path / "config"

    monkeypatch.setenv("PSETI_CALIB_DIR", str(calib_dir))
    monkeypatch.setenv("PSETI_TMP", str(tmp_dir))
    monkeypatch.setenv("PSETI_CONFIG", str(config_dir))

    # Create parent directories
    calib_dir.mkdir()
    tmp_dir.mkdir()
    config_dir.mkdir()

    import importlib
    importlib.reload(config_file)

    # Test 1: File in tmp_dir (calibration_dir is empty)
    test_data = {"date": "2024-01-01", "quabos": [{"uid": "test", "coefs": [100] * 256}]}
    (tmp_dir / config_file.quabo_ph_baseline_filename).write_text(json.dumps(test_data))

    result = config_file.get_quabo_ph_baselines()
    assert result.date == test_data["date"]
    assert result.quabos[0].uid == test_data["quabos"][0]["uid"]
    assert result.quabos[0].coefs == test_data["quabos"][0]["coefs"]

    # Clean up
    (tmp_dir / config_file.quabo_ph_baseline_filename).unlink()

    # Test 2: File in calibration_dir (should prefer it over tmp_dir)
    (calib_dir / config_file.quabo_ph_baseline_filename).write_text(json.dumps(test_data))
    (tmp_dir / config_file.quabo_ph_baseline_filename).write_text(json.dumps({"date": "old", "quabos": []}))

    result = config_file.get_quabo_ph_baselines()
    assert result.date == test_data["date"]

    # Test 3: File not found anywhere should raise FileNotFoundError
    (calib_dir / config_file.quabo_ph_baseline_filename).unlink()
    (tmp_dir / config_file.quabo_ph_baseline_filename).unlink()

    with pytest.raises(FileNotFoundError):
        config_file.get_quabo_ph_baselines()
