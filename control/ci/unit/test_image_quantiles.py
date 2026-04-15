"""
test_image_quantiles.py

Unit tests for control/utils/image_quantiles.py.
Creates synthetic PFF files in a temporary directory to test
get_values() and get_quantiles() without hardware.

Note: image_quantiles.py uses `import pff` (bare import).
The conftest.py adds control/utils/ to sys.path so this resolves correctly.
"""

import json
import struct

# These imports rely on conftest.py adding control/utils/ to sys.path
import image_quantiles
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_pff_frame(f, pixel_values, tv_sec=1_000_000):
    """Write one JSON header + 32x32 16-bit image block to file f."""
    header = {
        "quabo_0": {
            "quabo_num": 0, "pkt_num": 0,
            "pkt_tai": 613, "pkt_nsec": 0,
            "tv_sec": tv_sec, "tv_usec": 0,
        }
    }
    hdr_bytes = (json.dumps(header) + "\n\n").encode()
    f.write(hdr_bytes)
    f.write(b"*")
    f.write(struct.pack(f"{len(pixel_values)}H", *pixel_values))


def _make_pff_file(tmp_path, frames, fill_value=None):
    """
    Write a PFF file to tmp_path/test.pff.

    Args:
        frames: list of pixel-value lists, one per frame
        fill_value: if given, all frames are filled with this value
    Returns: str path to the file
    """
    fpath = str(tmp_path / "test.pff")
    with open(fpath, "wb") as f:
        for i, frame_pixels in enumerate(frames):
            vals = [fill_value] * 1024 if fill_value is not None else frame_pixels
            _write_pff_frame(f, vals, tv_sec=1_000_000 + i)
    return fpath


# ===========================================================================
# get_values
# ===========================================================================

class TestGetValues:
    def test_returns_correct_number_of_values(self, tmp_path):
        """3 frames × 1024 pixels = 3072 values."""
        frames = [[100] * 1024 for _ in range(3)]
        fpath = _make_pff_file(tmp_path, frames)
        vals = image_quantiles.get_values(fpath, 32, 2, nframes=3)
        assert len(vals) == 3072

    def test_respects_nframes_limit(self, tmp_path):
        """Even with 5 frames on disk, nframes=2 returns only 2048 values."""
        frames = [[0] * 1024 for _ in range(5)]
        fpath = _make_pff_file(tmp_path, frames)
        vals = image_quantiles.get_values(fpath, 32, 2, nframes=2)
        assert len(vals) == 2048

    def test_all_values_match_fill(self, tmp_path):
        """All pixels filled with 42; all returned values must be 42."""
        fpath = _make_pff_file(tmp_path, [None], fill_value=42)
        vals = image_quantiles.get_values(fpath, 32, 2, nframes=1)
        assert all(v == 42 for v in vals)

    def test_zero_fill(self, tmp_path):
        fpath = _make_pff_file(tmp_path, [None], fill_value=0)
        vals = image_quantiles.get_values(fpath, 32, 2, nframes=1)
        assert all(v == 0 for v in vals)

    def test_max_fill_16bit(self, tmp_path):
        fpath = _make_pff_file(tmp_path, [None], fill_value=65535)
        vals = image_quantiles.get_values(fpath, 32, 2, nframes=1)
        assert all(v == 65535 for v in vals)

    def test_stops_at_eof_gracefully(self, tmp_path):
        """Requesting more frames than exist stops at EOF without error."""
        frames = [[50] * 1024 for _ in range(2)]
        fpath = _make_pff_file(tmp_path, frames)
        vals = image_quantiles.get_values(fpath, 32, 2, nframes=100)
        # Should return at most 2 * 1024 values
        assert len(vals) == 2048

    def test_mixed_pixel_values(self, tmp_path):
        """Frame with pixel values 0..1023 — all values should appear exactly once."""
        frame_pixels = list(range(1024))
        fpath = _make_pff_file(tmp_path, [frame_pixels])
        vals = image_quantiles.get_values(fpath, 32, 2, nframes=1)
        assert sorted(vals) == list(range(1024))


# ===========================================================================
# get_quantiles
# ===========================================================================

class TestGetQuantiles:
    def test_uniform_image_quantiles(self, tmp_path):
        """All pixels = 100 → all quantiles equal 100."""
        fpath = _make_pff_file(tmp_path, [None], fill_value=100)
        low, high = image_quantiles.get_quantiles(fpath, 32, 2, 0.1)
        assert low == 100
        assert high == 100

    def test_quantile_ordering(self, tmp_path):
        """Low quantile ≤ high quantile for any valid x < 0.5."""
        frame_pixels = list(range(1024))  # 0..1023
        fpath = _make_pff_file(tmp_path, [frame_pixels])
        for x in [0.01, 0.1, 0.25, 0.49]:
            low, high = image_quantiles.get_quantiles(fpath, 32, 2, x)
            assert low <= high, f"Failed for x={x}: low={low}, high={high}"

    def test_x_equals_zero_raises_index_error(self, tmp_path):
        """x=0 → int(n*(1-0))=int(1024)=1024 which is out of range → IndexError."""
        frame_pixels = list(range(1024))
        fpath = _make_pff_file(tmp_path, [frame_pixels])
        # get_quantiles: return [values[int(n*x)], values[int(n*(1-x))]]
        # For x=0.0: values[int(1024*1.0)] = values[1024] → IndexError
        with pytest.raises(IndexError):
            image_quantiles.get_quantiles(fpath, 32, 2, 0.0)

    def test_x_equals_0_01(self, tmp_path):
        """x=0.01 → 1% and 99% quantiles of a uniform 0..1023 distribution."""
        frame_pixels = list(range(1024))
        fpath = _make_pff_file(tmp_path, [frame_pixels])
        low, high = image_quantiles.get_quantiles(fpath, 32, 2, 0.01)
        # int(1024 * 0.01) = int(10.24) = 10 → values[10] = 10
        # int(1024 * 0.99) = int(1013.76) = 1013 → values[1013] = 1013
        assert low == 10
        assert high == 1013

    def test_x_equals_0_5_returns_median(self, tmp_path):
        """x=0.5 → both halves should be equal to the median value."""
        frame_pixels = list(range(1024))
        fpath = _make_pff_file(tmp_path, [frame_pixels])
        # int(1024 * 0.5) = 512 → values[512] = 512
        # int(1024 * 0.5) = 512 → values[512] = 512 (same index, since 1-0.5=0.5)
        low, high = image_quantiles.get_quantiles(fpath, 32, 2, 0.5)
        assert low == high == 512

    def test_two_frames(self, tmp_path):
        """Works correctly when reading multiple frames."""
        frames = [list(range(1024)), list(range(1024))]
        fpath = _make_pff_file(tmp_path, frames)
        vals = image_quantiles.get_values(fpath, 32, 2, nframes=2)
        assert len(vals) == 2048

    def test_returns_list_of_two(self, tmp_path):
        fpath = _make_pff_file(tmp_path, [[42] * 1024])
        result = image_quantiles.get_quantiles(fpath, 32, 2, 0.1)
        assert isinstance(result, list)
        assert len(result) == 2
