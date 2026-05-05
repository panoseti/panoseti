import json

import pytest
from typer.testing import CliRunner

from control.config import app
from control.utils import config_file
from control.utils.paths import PanoPaths

runner = CliRunner()

def test_calibrate_ph_strict_success(mock_workspace, monkeypatch):
    """calibrate-ph --strict should pass if values are in range."""
    from control import config
    
    # Create valid baseline file in the mock calibration directory
    valid_data = {
        "date": "2024-01-01T00:00:00",
        "quabos": [
            {"uid": "q1", "coefs": [700] * 256}
        ]
    }
    baseline_path = PanoPaths.calibration_file(config_file.quabo_ph_baseline_filename)
    baseline_path.write_text(json.dumps(valid_data))
    
    # Mock the hardware call part of do_calibrate_ph but let the validation run
    def mock_do_calibrate_ph(modules, quabo_uids, network_config, min_baseline=600, max_baseline=800, strict=False):
        # Already wrote the file above, now just run the validation part
        baselines = config_file.get_quabo_ph_baselines()
        config_file.validate_ph_baselines(baselines, min_val=min_baseline, max_val=max_baseline, raise_error=strict)
        
    monkeypatch.setattr(config, "do_calibrate_ph", mock_do_calibrate_ph)
    
    result = runner.invoke(app, ["calibrate-ph", "--strict"])
    assert result.exit_code == 0

def test_calibrate_ph_strict_failure(mock_workspace, monkeypatch):
    """calibrate-ph --strict should fail if values are out of range."""
    from control import config
    
    # Create invalid baseline file
    invalid_data = {
        "date": "2024-01-01T00:00:00",
        "quabos": [
            {"uid": "q1", "coefs": [500] * 256} # Out of range (default 600-800)
        ]
    }
    baseline_path = PanoPaths.calibration_file(config_file.quabo_ph_baseline_filename)
    baseline_path.write_text(json.dumps(invalid_data))
    
    def mock_do_calibrate_ph(modules, quabo_uids, network_config, min_baseline=600, max_baseline=800, strict=False):
        baselines = config_file.get_quabo_ph_baselines()
        config_file.validate_ph_baselines(baselines, min_val=min_baseline, max_val=max_baseline, raise_error=strict)
        
    monkeypatch.setattr(config, "do_calibrate_ph", mock_do_calibrate_ph)
    
    result = runner.invoke(app, ["calibrate-ph", "--strict"])
    assert result.exit_code != 0
    assert "PH baseline validation failed" in result.stdout or isinstance(result.exception, ValueError)

def test_global_validator_ph_baseline_check(mock_workspace):
    """Global validator should report status of PH baselines."""
    from control.utils.global_validator import GlobalConfigValidator
    from control.utils.pydantic_config_models import DataConfig, ObsConfig
    
    # 1. Missing file -> WARN
    # (mock_workspace starts with no baseline file)
    obs = ObsConfig(name="test", domes=[])
    data = DataConfig(run_type="sci")
    validator = GlobalConfigValidator({"obs": obs, "data": data})
    validator._check_ph_baselines()
    
    report = next(iter(t for t in validator.report.tests if t["name"] == "PH Baseline Calibration"))
    assert report["status"] == "WARN"
    assert "Baseline file missing" in report["info"]
    
    # 2. Invalid range -> WARN
    invalid_data = {
        "date": "2024-01-01T00:00:00",
        "quabos": [{"uid": "q1", "coefs": [500] * 256}]
    }
    PanoPaths.calibration_file(config_file.quabo_ph_baseline_filename).write_text(json.dumps(invalid_data))
    
    validator = GlobalConfigValidator({"obs": obs, "data": data})
    validator._check_ph_baselines()
    report = next(iter(t for t in validator.report.tests if t["name"] == "PH Baseline Calibration"))
    assert report["status"] == "WARN"
    assert "out of range" in report["info"]
    
    # 3. Valid -> PASS
    valid_data = {
        "date": "2024-01-01T00:00:00",
        "quabos": [{"uid": "q1", "coefs": [700] * 256}]
    }
    PanoPaths.calibration_file(config_file.quabo_ph_baseline_filename).write_text(json.dumps(valid_data))
    
    validator = GlobalConfigValidator({"obs": obs, "data": data})
    validator._check_ph_baselines()
    report = next(iter(t for t in validator.report.tests if t["name"] == "PH Baseline Calibration"))
    assert report["status"] == "PASS"

def test_ph_baseline_average_focus(mock_workspace):
    """Validate that we focus on average values, allowing individual pixel outliers."""
    # Data with average in range but some pixels out of range
    # Average: (700 * 250 + 400 * 6) / 256 = 692.9 (OK)
    mixed_coefs = [700] * 250 + [400] * 6
    mixed_data = {
        "date": "2024-01-01T00:00:00",
        "quabos": [{"uid": "q1", "coefs": mixed_coefs}]
    }
    PanoPaths.calibration_file(config_file.quabo_ph_baseline_filename).write_text(json.dumps(mixed_data))
    
    baselines = config_file.get_quabo_ph_baselines()
    # Should PASS because average is 692.9, even with 400s
    assert config_file.validate_ph_baselines(baselines, raise_error=True) is True
    
    # Data with average out of range
    # Average: 500 (Fail)
    bad_data = {
        "date": "2024-01-01T00:00:00",
        "quabos": [{"uid": "q1", "coefs": [500] * 256}]
    }
    PanoPaths.calibration_file(config_file.quabo_ph_baseline_filename).write_text(json.dumps(bad_data))
    
    baselines = config_file.get_quabo_ph_baselines()
    with pytest.raises(ValueError, match="average values out of range"):
        config_file.validate_ph_baselines(baselines, raise_error=True)
