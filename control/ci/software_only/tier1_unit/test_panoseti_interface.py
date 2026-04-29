"""
test_panoseti_interface.py

Unit tests for control/utils/panoseti_interface.py.
Includes a PFF data simulator to generate test files.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from control.utils import panoseti_interface
from control.utils.panoseti_interface import PanosetiRun, PFFSequence, get_precise_time_ns

# ===========================================================================
# PFF Data Simulator
# ===========================================================================

class PFFSim:
    """Helper to generate valid PFF files for testing."""

    @staticmethod
    def create_header(dp_type, pkt_num=0, tv_sec=None, tv_usec=0, pkt_nsec=0, quabo_num=0):
        if tv_sec is None:
            tv_sec = int(time.time())
        
        pkt_tai = (tv_sec + 37) % 1024
        
        inner = {
            "pkt_num": pkt_num,
            "pkt_tai": pkt_tai,
            "pkt_nsec": pkt_nsec,
            "tv_sec": tv_sec,
            "tv_usec": tv_usec,
        }

        if dp_type == "ph256":
            header = {**inner, "quabo_num": quabo_num}
            # ph256 header is 124 bytes in docs, but we just need it to be consistent
            return header
        else:
            # img8, img16, ph1024 use nested quabo_0..3
            # For simplicity in tests, we just populate quabo_0
            return {
                "quabo_0": inner,
                "quabo_1": inner,
                "quabo_2": inner,
                "quabo_3": inner,
            }

    @staticmethod
    def write_frame(fh, header, dp_type, data=None, header_padding=512):
        header_json = json.dumps(header)
        # Pad header to be consistent within the file as required by PFF spec
        padding = " " * (header_padding - len(header_json))
        fh.write(header_json.encode() + padding.encode() + b"\n\n*")
        
        from typing import Any
        shape: tuple[int, int]
        dtype: Any
        if dp_type == "img8":
            shape = (32, 32)
            dtype = np.uint8
        elif dp_type == "img16":
            shape = (32, 32)
            dtype = np.uint16
        elif dp_type == "ph256":
            shape = (16, 16)
            dtype = np.int16
        elif dp_type == "ph1024":
            shape = (32, 32)
            dtype = np.int16
        else:
            raise ValueError(f"Unknown dp_type: {dp_type}")

        data = np.zeros(shape, dtype=dtype) if data is None else data.astype(dtype).reshape(shape)
        
        fh.write(data.tobytes())

    @classmethod
    def create_pff_file(cls, path, dp_type, n_frames=5, start_time=None, header_padding=512):
        if start_time is None:
            start_time = int(time.time())
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            for i in range(n_frames):
                header = cls.create_header(dp_type, pkt_num=i, tv_sec=start_time + i)
                cls.write_frame(f, header, dp_type, header_padding=header_padding)
        return path


# ===========================================================================
# Precise Timing Tests
# ===========================================================================

def test_get_precise_time_ns_sync() -> None:
    # d = (tv_sec - pkt_tai + 37) % 1024 == 0
    tv_sec = 1000
    pkt_tai = (tv_sec + 37) % 1024
    pkt_nsec = 123456789
    ts = get_precise_time_ns(tv_sec, 0, pkt_nsec, pkt_tai)
    assert ts == 1000 * 1_000_000_000 + pkt_nsec

def test_get_precise_time_ns_daq_ahead() -> None:
    # DAQ is 1s ahead: d=1
    tv_sec = 1000
    pkt_tai = (tv_sec + 37 - 1) % 1024
    pkt_nsec = 123456789
    ts = get_precise_time_ns(tv_sec, 0, pkt_nsec, pkt_tai)
    assert ts == 999 * 1_000_000_000 + pkt_nsec

def test_get_precise_time_ns_daq_behind() -> None:
    # DAQ is 1s behind: d=1023
    tv_sec = 1000
    pkt_tai = (tv_sec + 37 + 1) % 1024
    pkt_nsec = 123456789
    ts = get_precise_time_ns(tv_sec, 0, pkt_nsec, pkt_tai)
    assert ts == 1001 * 1_000_000_000 + pkt_nsec


# ===========================================================================
# PFFSequence Tests
# ===========================================================================

class TestPFFSequence:
    @pytest.fixture
    def img16_file(self, tmp_path) -> None:
        path = tmp_path / "start_2026-01-01T00:00:00Z.dp_img16.bpp_2.module_0.seqno_0.pff"
        return PFFSim.create_pff_file(path, "img16", n_frames=10, start_time=1704067200)

    def test_init_and_analyze(self, img16_file) -> None:
        seq = PFFSequence([img16_file])
        assert len(seq) == 10
        assert seq.frame_config is not None
        assert seq.frame_config.bytes_per_pixel == 2
        assert seq.frame_config.image_shape == (32, 32)
        assert seq.meta["dp"] == "img16"

    def test_get_frame(self, img16_file) -> None:
        seq = PFFSequence([img16_file])
        header, img = seq.get_frame(0)
        from typing import cast
        header = cast(panoseti_interface.ModuleHeader, header)
        assert img.shape == (32, 32)
        assert img.dtype == np.uint16
        assert header.quabo_0.tv_sec == 1704067200

    def test_index_timestamps(self, img16_file) -> None:
        seq = PFFSequence([img16_file])
        seq.index_timestamps()
        assert seq._timestamps is not None
        assert len(seq._timestamps) == 10
        assert seq._timestamps[0] == 1704067200 * 1_000_000_000

    def test_seek_time(self, img16_file) -> None:
        seq = PFFSequence([img16_file])
        # Force indexing
        seq.index_timestamps()
        
        idx = seq.seek_time(1704067205 * 1_000_000_000)
        assert idx == 5
        
        idx = seq.seek_time(1704067205 * 1_000_000_000 + 100)
        assert idx == 5

    def test_get_image_array(self, img16_file) -> None:
        seq = PFFSequence([img16_file])
        arr = seq.get_image_array(start=2, count=3)
        assert arr.shape == (3, 32, 32)
        assert arr.dtype == np.uint16

    def test_multiple_files(self, tmp_path) -> None:
        f1 = PFFSim.create_pff_file(tmp_path / "run.seqno_0.pff", "img16", n_frames=5, start_time=1000)
        f2 = PFFSim.create_pff_file(tmp_path / "run.seqno_1.pff", "img16", n_frames=5, start_time=1005)
        
        seq = PFFSequence([f1, f2])
        assert len(seq) == 10
        header, _ = seq.get_frame(7)
        from typing import cast
        header = cast(panoseti_interface.ModuleHeader, header)
        assert header.quabo_0.tv_sec == 1007


# ===========================================================================
# PanosetiRun Tests
# ===========================================================================

class TestPanosetiRun:
    @pytest.fixture
    def run_dir(self, tmp_path) -> None:
        run_path = tmp_path / "obs_test.start_2026.pffd"
        run_path.mkdir()
        
        # Create some PFF files
        PFFSim.create_pff_file(run_path / "start_2026.dp_img16.module_0.seqno_0.pff", "img16", n_frames=5)
        PFFSim.create_pff_file(run_path / "start_2026.dp_ph256.module_0.seqno_0.pff", "ph256", n_frames=5)
        
        # Create a config file
        config = {"run_type": "sci", "obs": "test"}
        (run_path / "data_config.json").write_text(json.dumps(config))
        
        return run_path

    def test_run_loading(self, run_dir) -> None:
        run = PanosetiRun(run_dir)
        assert "data_config" in run.configs
        assert len(run.products) == 2
        assert "dp_img16.module_0" in run.products
        assert "dp_ph256.module_0" in run.products

    def test_list_products(self, run_dir) -> None:
        run = PanosetiRun(run_dir)
        products = run.list_products()
        assert "dp_img16.module_0" in products
        assert "dp_ph256.module_0" in products

    def test_get_product(self, run_dir) -> None:
        run = PanosetiRun(run_dir)
        seq = run.get_product("dp_img16.module_0")
        assert isinstance(seq, PFFSequence)
        assert len(seq) == 5

    def test_show_smoke(self, run_dir, capsys) -> None:
        """Smoke test for the rich-based show() method."""
        run = PanosetiRun(run_dir)
        run.show()
        captured = capsys.readouterr()
        assert "Run: " in captured.out
        assert "Data Products" in captured.out
