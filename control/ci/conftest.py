"""
conftest.py — Shared pytest fixtures for the panoseti-control test suite.

sys.path is managed by pyproject.toml [tool.pytest.ini_options] pythonpath=["."],
which adds control/ to the path so "from utils.X import ..." works.

We also add control/utils/ for modules that use bare `import pff` style imports
(e.g. image_quantiles.py).
"""

import copy
import io
import json
import os
import pathlib
import struct
import tomllib
from collections.abc import Callable
from typing import Any

import pytest

from ci.paths import PanoPathsTest
from control.utils.pydantic_config_models import (
    DaqConfigValidator,
    NetworkConfigValidator,
    ObsConfigValidator,
)


def pytest_configure(config: Any) -> None:
    """
    Set environment variable overrides to isolate the test environment.
    This ensures PanoPaths resolves to test-specific directories instead of
    production code directories, preventing state leakage.
    """
    # 1. Route configs to the integration test configs (default to direct for unit tests)
    os.environ["PSETI_CONFIG"] = str(PanoPathsTest.integration_configs("direct"))

    # 2. Route state to isolated test directories
    os.environ["PSETI_TMP"] = str(PanoPathsTest.test_state_root() / "tmp")
    os.environ["PSETI_LOGS"] = str(PanoPathsTest.test_state_root() / "logs")
    os.environ["PSETI_QUABOS"] = str(PanoPathsTest.test_state_root() / "quabos")

    # 3. Ensure directories exist
    os.makedirs(os.environ["PSETI_TMP"], exist_ok=True)
    os.makedirs(os.environ["PSETI_LOGS"], exist_ok=True)
    os.makedirs(os.environ["PSETI_QUABOS"], exist_ok=True)
