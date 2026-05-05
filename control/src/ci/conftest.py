"""
conftest.py — Shared pytest fixtures for the panoseti-control test suite.

This file contains fixtures shared by both software-only and HITL tests.
It does NOT perform autouse isolation or session fleet setup, as those are
HITL-incompatible and reside in ci/software-only/conftest.py.
"""

import copy
import io
import json
import pathlib
import struct
import tomllib
from collections.abc import Callable
from typing import Any

import pytest

from ci.fixtures.factories import (
    make_mock_daq_config,
    make_transfer_job,
    simulate_daq_filesystem,
)
from control.utils.pydantic_config_models import (
    DaqConfig,
    NetworkConfig,
    ObsConfig,
)

pytest_plugins = [
    "ci.fixtures.workspace_fixtures",
    "ci.fixtures.network_fixtures",
    "ci.fixtures.data_fixtures",
]


def pytest_configure_node(node: Any) -> None:
    """Called by xdist controller to configure each worker before it starts."""
    if not hasattr(node.config, "_tc_run_uuid"):
        import uuid as _uuid
        node.config._tc_run_uuid = _uuid.uuid4().hex[:8]
    node.workerinput["tc_run_uuid"] = node.config._tc_run_uuid

@pytest.fixture(scope="session")
def worker_id(request: Any) -> str:
    """Returns the xdist worker ID or 'master' if not running in parallel."""
    if hasattr(request.config, "workerinput"):
        return request.config.workerinput["workerid"]
    return "master"

# ---------------------------------------------------------------------------
# Shared Factories & Mocks (Infrastructure)
# ---------------------------------------------------------------------------

@pytest.fixture
def transfer_job_factory():
    """Factory for creating valid TransferJob models."""
    return make_transfer_job

@pytest.fixture
def daq_fs_simulator():
    """Helper to populate a mock DAQ filesystem structure."""
    return simulate_daq_filesystem

@pytest.fixture
def daq_config_factory():
    """Factory for creating valid DaqConfig models."""
    return make_mock_daq_config

@pytest.fixture(scope="session")
def topology_templates() -> dict[str, dict[str, Any]]:
    """Loads all TOML topology templates from ci/test_topologies/."""
    templates = {}
    template_dir = pathlib.Path(__file__).parent / "test_topologies"
    if template_dir.exists():
        for toml_file in template_dir.glob("*.toml"):
            with open(toml_file, "rb") as f:
                templates[toml_file.stem] = tomllib.load(f)
    return templates


@pytest.fixture
def minimal_obs_config(topology_templates) -> dict[str, Any]:
    """Smallest valid obs_config dict: one dome, one module."""
    return copy.deepcopy(topology_templates.get("base_obs", {}))


@pytest.fixture
def two_dome_obs_config(topology_templates) -> dict[str, Any]:
    """Two-dome obs config for geospatial checks."""
    cfg = copy.deepcopy(topology_templates.get("base_obs", {}))
    # Add a second dome ~111 m apart — within 2 km baseline
    cfg["domes"].append({
        "name": "dome1",
        "obslat": 33.358,
        "obslon": -116.866,
        "obsalt": 1706.0,
        "modules": [
            {
                "mobo_serialno": "SN002",
                "quabo_version": "bga",
                "ip_addr": "192.168.3.204",
                "wps": "wps",
            }
        ],
    })
    return cfg


@pytest.fixture
def minimal_daq_config(topology_templates) -> dict[str, Any]:
    """Smallest valid daq_config dict: one DAQ node."""
    return copy.deepcopy(topology_templates.get("base_daq", {}))


@pytest.fixture
def minimal_data_config(topology_templates) -> dict[str, Any]:
    """Smallest valid data_config dict: image mode only."""
    return copy.deepcopy(topology_templates.get("base_data", {}))


@pytest.fixture
def minimal_firmware_config(topology_templates) -> dict[str, Any]:
    """Firmware config listing hardware variants."""
    return copy.deepcopy(topology_templates.get("base_firmware", {}))

@pytest.fixture
def mock_daq_config() -> DaqConfig:
    """Fully valid Pydantic model for DAQ configuration."""
    baseline = {
        "head_node_data_dir": "/data/head",
        "head_node_ip_addr": "10.0.0.1",
        "head_node_container": False,
        "daq_nodes": [
            {
                "ip_addr": "10.0.0.2",
                "data_dir": "/data",
                "username": "panoseti",
                "module_ids": [200],
                "bindhost": "lo"
            }
        ]
    }
    return DaqConfig(**baseline)

@pytest.fixture
def mock_network_config() -> NetworkConfig:
    """Fully valid Pydantic model for network configuration."""
    baseline: dict[str, Any] = {
        "modules": [
            {
                "ip_addr": "192.168.3.200",
                "port_forwarding": {
                    "status": False,
                    "gw_ip": "10.200.146.11",
                    "reboot_port": [60004, 60005, 60006, 60007],
                    "cmd_port": [60000, 60001, 60002, 60003]
                }
            }
        ],
        "daq_nodes": [
            {
                "ip_addr": "10.0.0.2",
                "port_forwarding": {
                    "status": False,
                    "gw_ip": "10.200.146.11"
                }
            }
        ]
    }
    return NetworkConfig(**baseline)

@pytest.fixture
def mock_obs_config() -> ObsConfig:
    """Fully valid Pydantic model for observatory configuration."""
    baseline = {
        "name": "test_obs",
        "comment": "Test Observatory",
        "wps": {
            "url": "http://192.168.1.2",
            "quabo_socket": 1
        },
        "wr_ip_addr": "192.168.1.254",
        "gps_port": "/dev/ttyUSB0",
        "detector_overvoltage": 2,
        "domes": [
            {
                "obslat": 33.3533,
                "obslon": -116.8622,
                "obsalt": 1693.0,
                "name": "dome0",
                "modules": [
                    {
                        "mobo_serialno": "M11",
                        "quabo_version": "qfp",
                        "ip_addr": "192.168.3.200",
                        "wps": "wps",
                        "timing_mode": "gnss",
                        "azimuth": 77.0,
                        "elevation": 77.0,
                        "position_angle": 77.0
                    }
                ]
            }
        ]
    }
    return ObsConfig(**baseline)


# ---------------------------------------------------------------------------
# PFF file helpers
# ---------------------------------------------------------------------------

def _make_pff_json_header(quabo_num: int = 0, pkt_num: int = 0,
                           pkt_tai: int = 613, pkt_nsec: int = 0,
                           tv_sec: int = 1_000_000, tv_usec: int = 0) -> bytes:
    """Build a PFF JSON header block (ends with \\n\\n)."""
    # For img16 / img8 style: use a quabo_0 sub-dict
    payload = {
        "quabo_0": {
            "quabo_num": quabo_num,
            "pkt_num": pkt_num,
            "pkt_tai": pkt_tai,
            "pkt_nsec": pkt_nsec,
            "tv_sec": tv_sec,
            "tv_usec": tv_usec,
        }
    }
    s = json.dumps(payload) + "\n\n"
    return s.encode()


def _make_pff_image_block_16bit(width: int = 32) -> bytes:
    """Build a 16-bit image block (32x32 pixels, all zeros)."""
    n = width * width
    return b"*" + struct.pack(f"{n}H", *([0] * n))


def make_minimal_pff_bytes(n_frames: int = 3, tv_sec_start: int = 1_000_000) -> bytes:
    """Return bytes of a minimal PFF file with n_frames of img16 data."""
    buf = io.BytesIO()
    for i in range(n_frames):
        header = _make_pff_json_header(tv_sec=tv_sec_start + i)
        buf.write(header)
        buf.write(_make_pff_image_block_16bit())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixed-size PFF file factory (for img_info / time_seek tests)
# ---------------------------------------------------------------------------

_FIXED_HEADER_JSON_LEN = 120  # pad all JSON to this many bytes before '\n\n'


def _make_fixed_header(tv_sec: int, pkt_num: int = 0, nested: bool = True) -> bytes:
    pkt_tai = (tv_sec + 37) % 1024
    inner = {
        "quabo_num": 0,
        "pkt_num": pkt_num,
        "pkt_tai": pkt_tai,
        "pkt_nsec": 0,
        "tv_sec": tv_sec,
        "tv_usec": 0,
    }
    payload = {"quabo_0": inner} if nested else inner
    s = json.dumps(payload)
    s = s + " " * max(0, _FIXED_HEADER_JSON_LEN - len(s))
    return (s + "\n\n").encode()


def make_pff_file(
    n_frames: int = 3,
    tv_sec_start: int = 1_000_000,
    tv_sec_values: list[int] | None = None,
    nested_header: bool = True,
    img_size: int = 32,
    bpp: int = 2,
) -> io.BytesIO:
    if tv_sec_values is None:
        tv_sec_values = [tv_sec_start + i for i in range(n_frames)]
    assert len(tv_sec_values) == n_frames, "tv_sec_values length must match n_frames"

    n_pixels = img_size * img_size
    fmt = f"{n_pixels}{'H' if bpp == 2 else 'B'}"
    image_bytes = b"*" + struct.pack(fmt, *([0] * n_pixels))

    buf = io.BytesIO()
    for i, tv_sec in enumerate(tv_sec_values):
        buf.write(_make_fixed_header(tv_sec, pkt_num=i, nested=nested_header))
        buf.write(image_bytes)
    buf.seek(0)
    return buf


@pytest.fixture
def pff_file_factory() -> Callable[..., io.BytesIO]:
    return make_pff_file


# ---------------------------------------------------------------------------
# Machine-readable test summary
# ---------------------------------------------------------------------------

def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:
    stats = terminalreporter.stats
    summary = {
        "passed": len(stats.get("passed", [])),
        "failed": len(stats.get("failed", [])),
        "skipped": len(stats.get("skipped", [])),
        "error": len(stats.get("error", [])),
        "xfail": len(stats.get("xfail", [])),
        "xpass": len(stats.get("xpass", [])),
    }
    print(f"\nTEST_METRICS_JSON: {json.dumps(summary)}")

@pytest.fixture(autouse=True)
def _enable_log_propagation_for_caplog(caplog):
    import logging
    saved = {}
    for name in ("PSETI.Start", "PSETI.Stop", "PSETI.Status",
                 "transfer_daemon", "PSETI.Config", "PSETI.Interleave",
                 "PSETI.Interface", "PSETI.storeInfluxDB"):
        lg = logging.getLogger(name)
        saved[name] = lg.propagate
        lg.propagate = True
    yield
    for name, p in saved.items():
        logging.getLogger(name).propagate = p
