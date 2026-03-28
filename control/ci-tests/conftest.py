"""
conftest.py — Shared pytest fixtures for the panoseti-control test suite.

sys.path is managed by pyproject.toml [tool.pytest.ini_options] pythonpath=["."],
which adds control/ to the path so "from utils.X import ..." works.

We also add control/utils/ for modules that use bare `import pff` style imports
(e.g. image_quantiles.py).
"""

import io
import json
import struct
import sys
import os
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub out system/hardware dependencies imported at module level in util.py
# but not available (or needed) in the test environment.
# ---------------------------------------------------------------------------

# Allow bare `import pff` as used inside image_quantiles.py
_utils_dir = os.path.join(os.path.dirname(__file__), "..", "utils")
if _utils_dir not in sys.path:
    sys.path.insert(0, _utils_dir)


# ---------------------------------------------------------------------------
# Minimal valid configuration dictionaries (no disk I/O required)
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_obs_config():
    """Smallest valid obs_config dict: one dome, one module."""
    return {
        "name": "test_obs",
        "wr_ip_addr": "192.168.1.254",
        "gps_port": "/dev/ttyUSB0",
        "detector_overvoltage": 3,
        "domes": [
            {
                "name": "dome0",
                "obslat": 33.357,
                "obslon": -116.865,
                "obsalt": 1706.0,
                "modules": [
                    {
                        "mobo_serialno": "SN001",
                        "quabo_version": "bga",
                        "ip_addr": "192.168.3.200",
                        "wps": "wps",
                        "timing_mode": "wr",
                    }
                ],
            }
        ],
        "wps": {"url": "http://192.168.1.2", "quabo_socket": 1},
    }


@pytest.fixture
def two_dome_obs_config():
    """Two-dome obs config for geospatial checks."""
    return {
        "name": "two_dome_obs",
        "wr_ip_addr": "192.168.1.254",
        "domes": [
            {
                "name": "dome0",
                "obslat": 33.357,
                "obslon": -116.865,
                "obsalt": 1706.0,
                "modules": [
                    {
                        "mobo_serialno": "SN001",
                        "quabo_version": "bga",
                        "ip_addr": "192.168.3.200",
                        "wps": "wps",
                    }
                ],
            },
            {
                "name": "dome1",
                "obslat": 33.358,   # ~111 m apart — within 2 km baseline
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
            },
        ],
        "wps": {"url": "http://192.168.1.2", "quabo_socket": 1},
    }


@pytest.fixture
def minimal_daq_config():
    """Smallest valid daq_config dict: one DAQ node."""
    return {
        "head_node_data_dir": "/data",
        "head_node_ip_addr": "10.0.0.1",
        "daq_nodes": [
            {
                "username": "panoseti",
                "data_dir": "/data",
                "ip_addr": "10.0.0.2",
                "module_ids": "224-225",
                "bindhost": "0.0.0.0",
            }
        ],
    }


@pytest.fixture
def minimal_data_config():
    """Smallest valid data_config dict: image mode only."""
    return {
        "run_type": "sci",
        "detector_overvoltage": 3,
        "image": {
            "integration_time_usec": 100000,
            "pe_threshold": 1.0,
            "quabo_sample_size": 16,
        },
    }


@pytest.fixture
def ph_only_data_config():
    """Pulse-height-only data config (no image)."""
    return {
        "run_type": "sci",
        "pulse_height": {
            "pe_threshold": 3.0,
        },
    }


@pytest.fixture
def minimal_firmware_config():
    """Firmware config listing the 'bga' hardware variant."""
    return {"bga": "firmware_bga_v2.bin"}


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
    """Build a 16-bit image block (32×32 pixels, all zeros)."""
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
# All frames have identical header sizes — required by img_info's frame_size math.
# pkt_tai is set so that d = (tv_sec - pkt_tai + 37) % 1024 == 0 for every frame.
# ---------------------------------------------------------------------------

_FIXED_HEADER_JSON_LEN = 120  # pad all JSON to this many bytes before '\n\n'


def _make_fixed_header(tv_sec: int, pkt_num: int = 0, nested: bool = True) -> bytes:
    """
    Build a PFF JSON header padded to _FIXED_HEADER_JSON_LEN + 2 bytes total.
    nested=True  → img16/img8 style   {"quabo_0": {...}}
    nested=False → ph256 style        {...}
    d=0 guaranteed: pkt_tai = (tv_sec + 37) % 1024.
    """
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
    # Pad with spaces so every header has the same byte length
    s = s + " " * max(0, _FIXED_HEADER_JSON_LEN - len(s))
    return (s + "\n\n").encode()


def make_pff_file(
    n_frames: int = 3,
    tv_sec_start: int = 1_000_000,
    tv_sec_values: list | None = None,
    nested_header: bool = True,
    img_size: int = 32,
    bpp: int = 2,
) -> io.BytesIO:
    """
    Write an in-memory PFF file and return a seeked-to-start BytesIO.

    All frames have identical fixed-size headers (padded to _FIXED_HEADER_JSON_LEN).
    tv_sec_values, if provided, overrides per-frame tv_sec; must be length n_frames.
    """
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
def pff_file_factory():
    """Fixture that returns the make_pff_file() helper."""
    return make_pff_file
