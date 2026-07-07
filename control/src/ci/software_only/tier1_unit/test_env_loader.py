import os
from pathlib import Path
from unittest import mock

import pytest

from control.utils.env_loader import load_pseti_env

@pytest.fixture
def clean_env():
    """Fixture to ensure a clean os.environ for specific test keys."""
    keys_to_clean = ["PSETI_ENV_FILE", "TEST_PSETI_VAR_1", "TEST_PSETI_VAR_2"]
    original_env = {}
    for key in keys_to_clean:
        if key in os.environ:
            original_env[key] = os.environ.pop(key)
            
    yield
    
    # Restore
    for key in keys_to_clean:
        if key in os.environ:
            del os.environ[key]
    for key, val in original_env.items():
        os.environ[key] = val


def test_load_pseti_env_default(tmp_path: Path, clean_env):
    """Test loading from default .env file in the current working directory."""
    env_content = "TEST_PSETI_VAR_1=hello_default_env\n"
    
    # We must patch Path.is_file and load_dotenv because we can't easily 
    # change the actual CWD robustly without affecting other async tests.
    with mock.patch("control.utils.env_loader.Path.is_file") as mock_is_file, \
         mock.patch("control.utils.env_loader.load_dotenv") as mock_load_dotenv:
        
        mock_is_file.return_value = True
        
        load_pseti_env()
        
        mock_load_dotenv.assert_called_once_with(dotenv_path=Path(".env"), override=True)


def test_load_pseti_env_override_with_custom_file(tmp_path: Path, clean_env):
    """Test loading from a specific file set by PSETI_ENV_FILE."""
    custom_env = tmp_path / "custom.env"
    custom_env.write_text("TEST_PSETI_VAR_2=override_value\n")
    
    os.environ["PSETI_ENV_FILE"] = str(custom_env)
    
    # This time we actually let it run to verify it loads the real file
    load_pseti_env()
    
    assert os.environ.get("TEST_PSETI_VAR_2") == "override_value"


def test_load_pseti_env_overwrites_existing_vars(tmp_path: Path, clean_env):
    """Test that variables loaded from .env overwrite existing variables in os.environ."""
    # Pre-set the variable
    os.environ["TEST_PSETI_VAR_1"] = "original_value"
    
    custom_env = tmp_path / "override.env"
    custom_env.write_text("TEST_PSETI_VAR_1=new_overwritten_value\n")
    
    os.environ["PSETI_ENV_FILE"] = str(custom_env)
    
    load_pseti_env()
    
    # Verify that the value was overwritten (override=True)
    assert os.environ.get("TEST_PSETI_VAR_1") == "new_overwritten_value"
