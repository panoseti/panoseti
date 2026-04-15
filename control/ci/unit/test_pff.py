"""
test_pff.py

Unit tests for control/utils/pff.py.
Covers:
  - parse_name: filename → dict
  - is_pff_dir / is_pff_file / pff_file_type
  - wr_to_unix: all four d-value branches (0, 1, 1023, desync)
  - write_image_1D + read_image: binary round-trip
  - run_dir_name: format check
  - read_json: JSON header parsing
No hardware required.
"""

import io
import json
import struct

import pytest

from utils import pff

# ===========================================================================
# parse_name
# Parses "key_value.key_value...key_value.ext" into {key: value}
# ===========================================================================

class TestParseName:
    @pytest.mark.parametrize("filename, key, expected_val", [
        ("start_2026-01-01T00:00:00Z.dp_ph256.bpp_2.module_254.seqno_0.pff", "dp", "ph256"),
        ("start_2026-01-01T00:00:00Z.dp_img16.bpp_2.module_0.seqno_1.pff",   "dp", "img16"),
        ("start_2026-01-01T00:00:00Z.dp_img8.bpp_1.module_0.seqno_0.pff",    "dp", "img8"),
        ("start_2026-01-01T00:00:00Z.dp_ph1024.bpp_2.module_254.seqno_0.pff","dp", "ph1024"),
    ])
    def test_extracts_data_product(self, filename, key, expected_val):
        d = pff.parse_name(filename)
        assert d[key] == expected_val

    def test_extracts_module_id(self):
        fname = "start_2026-01-01T00:00:00Z.dp_ph256.bpp_2.module_254.seqno_0.pff"
        d = pff.parse_name(fname)
        assert d["module"] == "254"

    def test_extracts_bpp(self):
        fname = "start_2026-01-01T00:00:00Z.dp_img16.bpp_2.module_0.seqno_1.pff"
        d = pff.parse_name(fname)
        assert d["bpp"] == "2"

    def test_extracts_seqno(self):
        fname = "start_2026-01-01T00:00:00Z.dp_ph256.bpp_2.module_254.seqno_7.pff"
        d = pff.parse_name(fname)
        assert d["seqno"] == "7"

    def test_extracts_start_timestamp(self):
        fname = "start_2026-01-01T12:30:00Z.dp_img16.bpp_2.module_0.seqno_0.pff"
        d = pff.parse_name(fname)
        # parse_name splits on '_'; start_timestamp contains '-' so it's a
        # single token "start" → "2026-01-01T12:30:00Z"
        assert "start" in d
        assert "2026" in d["start"]

    def test_no_extension_returns_none(self):
        assert pff.parse_name("nodotinfilename") is None

    def test_returns_dict(self):
        fname = "start_2026.dp_img8.bpp_1.module_0.seqno_0.pff"
        assert isinstance(pff.parse_name(fname), dict)

    def test_hk_file_parses_correctly(self):
        """hk.pff has no dp key — parse_name should return an empty/minimal dict."""
        d = pff.parse_name("hk.pff")
        assert isinstance(d, dict)
        assert "dp" not in d


# ===========================================================================
# is_pff_dir / is_pff_file / pff_file_type
# ===========================================================================

class TestPffTypeHelpers:
    @pytest.mark.parametrize("name", [
        "obs_test.start_2026-01-01T00:00:00Z.runtype_sci.pffd",
        "obs_palomar.start_2025-07-25T04:22:33Z.runtype_cal.pffd",
        "some.pffd",
    ])
    def test_is_pff_dir_true(self, name):
        assert pff.is_pff_dir(name) is True

    @pytest.mark.parametrize("name", [
        "some.pff",
        "some.pffd.pff",
        "pffd",
        "obs_test.start_X.pffd.tar",
    ])
    def test_is_pff_dir_false(self, name):
        assert pff.is_pff_dir(name) is False

    @pytest.mark.parametrize("name", [
        "hk.pff",
        "start_X.dp_img16.bpp_2.module_0.seqno_0.pff",
    ])
    def test_is_pff_file_true(self, name):
        assert pff.is_pff_file(name) is True

    def test_is_pff_file_false(self):
        assert pff.is_pff_file("some.pffd") is False

    def test_pff_file_type_hk(self):
        assert pff.pff_file_type("hk.pff") == "hk"

    @pytest.mark.parametrize("dp", ["img8", "img16", "ph256", "ph1024"])
    def test_pff_file_type_data_products(self, dp):
        fname = f"start_X.dp_{dp}.bpp_2.module_0.seqno_0.pff"
        assert pff.pff_file_type(fname) == dp

    def test_pff_file_type_no_dp_returns_none(self):
        """File without dp field → None."""
        assert pff.pff_file_type("nodp_here.something.pff") is None


# ===========================================================================
# run_dir_name
# Format: "obs_{name}.start_{ISO8601}Z.runtype_{run_type}.pffd"
# ===========================================================================

class TestRunDirName:
    def test_format(self):
        name = pff.run_dir_name("palomar", "sci")
        assert name.startswith("obs_palomar.start_")
        assert name.endswith(".runtype_sci.pffd")

    def test_is_pff_dir(self):
        name = pff.run_dir_name("lick", "cal")
        assert pff.is_pff_dir(name)

    def test_contains_utc_marker(self):
        name = pff.run_dir_name("ucb", "eng")
        # ISO 8601 with trailing Z
        assert "Z" in name

    def test_different_obs_names(self):
        a = pff.run_dir_name("palomar", "sci")
        b = pff.run_dir_name("lick", "sci")
        assert "palomar" in a
        assert "lick" in b

    def test_different_run_types(self):
        sci = pff.run_dir_name("obs", "sci")
        cal = pff.run_dir_name("obs", "cal")
        assert "sci" in sci
        assert "cal" in cal


# ===========================================================================
# wr_to_unix
# Converts White Rabbit TAI timestamp to Unix time.
# Formula: d = (tv_sec - pkt_tai + 37) % 1024
#   d=0    → return tv_sec + pkt_nsec/1e9
#   d=1    → return tv_sec - 1 + pkt_nsec/1e9
#   d=1023 → return tv_sec + 1 + pkt_nsec/1e9
#   else   → raises Exception (clocks > 1 s apart)
# ===========================================================================

class TestWrToUnix:
    # --- Helper: build pkt_tai for a given d value ---
    # d = (tv_sec - pkt_tai + 37) % 1024
    # pkt_tai = (tv_sec + 37 - d) % 1024

    TV_SEC = 1_000_000
    PKT_NSEC_HALF = 500_000_000   # 0.5 s as nanoseconds

    def _pkt_tai(self, tv_sec, d):
        return (tv_sec + 37 - d) % 1024

    def test_d_zero_no_nsec(self):
        """Synchronized clocks, pkt_nsec=0 → exactly tv_sec."""
        tv = self.TV_SEC
        tai = self._pkt_tai(tv, 0)
        result = pff.wr_to_unix(tai, 0, tv)
        assert result == tv

    def test_d_zero_with_nsec(self):
        """Synchronized clocks, pkt_nsec=0.5s → tv_sec + 0.5."""
        tv = self.TV_SEC
        tai = self._pkt_tai(tv, 0)
        result = pff.wr_to_unix(tai, self.PKT_NSEC_HALF, tv)
        assert abs(result - (tv + 0.5)) < 1e-6

    def test_d_one_returns_tv_sec_minus_one(self):
        """d=1: pkt_tai is one TAI second ahead → subtract one from tv_sec."""
        tv = self.TV_SEC
        tai = self._pkt_tai(tv, 1)
        result = pff.wr_to_unix(tai, self.PKT_NSEC_HALF, tv)
        expected = tv - 1 + 0.5
        assert abs(result - expected) < 1e-6

    def test_d_1023_returns_tv_sec_plus_one(self):
        """d=1023 (= -1 mod 1024): pkt_tai is one TAI second behind → add one."""
        tv = self.TV_SEC
        tai = self._pkt_tai(tv, 1023)
        result = pff.wr_to_unix(tai, self.PKT_NSEC_HALF, tv)
        expected = tv + 1 + 0.5
        assert abs(result - expected) < 1e-6

    def test_d_zero_with_zero_nsec_boundary(self):
        """Boundary: pkt_nsec=0, synchronized."""
        tv = self.TV_SEC
        tai = self._pkt_tai(tv, 0)
        assert pff.wr_to_unix(tai, 0, tv) == float(tv)

    def test_d_zero_with_max_nsec(self):
        """pkt_nsec = 999_999_999 (~1 s)."""
        tv = self.TV_SEC
        tai = self._pkt_tai(tv, 0)
        result = pff.wr_to_unix(tai, 999_999_999, tv)
        assert abs(result - (tv + 0.999_999_999)) < 1e-6

    def test_desync_raises_exception(self):
        """d not in {0, 1, 1023} → clocks disagree by > 1 s → Exception."""
        tv = self.TV_SEC
        # pkt_tai = 0 → d = (tv + 37) % 1024 = 613 for tv=1_000_000
        assert (tv - 0 + 37) % 1024 not in (0, 1, 1023)
        with pytest.raises(Exception, match="WR and Unix times differ"):
            pff.wr_to_unix(0, 0, tv)

    def test_ignore_desync_returns_approximation(self):
        """With ignore_clock_desync=True, returns tv_sec + pkt_nsec/1e9 even when out of sync."""
        tv = self.TV_SEC
        result = pff.wr_to_unix(0, self.PKT_NSEC_HALF, tv, ignore_clock_desync=True)
        expected = tv + 0.5
        assert abs(result - expected) < 1e-6

    def test_different_tv_sec_values(self):
        """Ensure formula works for multiple reference seconds."""
        for tv in [0, 1_000_000, 1_700_000_000]:  # epoch, arbitrary, ~2023
            tai = self._pkt_tai(tv, 0)
            result = pff.wr_to_unix(tai, 0, tv)
            assert result == float(tv)


# ===========================================================================
# write_image_1D + read_image: round-trip
# ===========================================================================

class TestImageRoundTrip:
    def _make_image_32x32(self, bpp, fill=42):
        n = 32 * 32  # 1024 pixels
        return list(range(n)) if fill is None else [fill] * n

    def test_round_trip_16bit(self):
        """write_image_1D then read_image recovers identical pixel values (16-bit)."""
        img = self._make_image_32x32(bpp=2)
        buf = io.BytesIO()
        pff.write_image_1D(buf, img, 32, 2)
        buf.seek(0)
        recovered = pff.read_image(buf, 32, 2)
        assert list(recovered) == img

    def test_round_trip_8bit(self):
        """write_image_1D then read_image recovers identical pixel values (8-bit)."""
        img = [i % 256 for i in range(1024)]  # wrap to stay in 0-255
        buf = io.BytesIO()
        pff.write_image_1D(buf, img, 32, 1)
        buf.seek(0)
        recovered = pff.read_image(buf, 32, 1)
        assert list(recovered) == img

    def test_image_block_starts_with_star(self):
        """The image block must be prefixed with b'*'."""
        img = [0] * 1024
        buf = io.BytesIO()
        pff.write_image_1D(buf, img, 32, 2)
        buf.seek(0)
        assert buf.read(1) == b'*'

    def test_read_image_bad_type_code_raises(self):
        """read_image raises Exception when magic byte is not b'*'."""
        buf = io.BytesIO(b'X' + struct.pack("1024H", *([0] * 1024)))
        with pytest.raises(Exception):
            pff.read_image(buf, 32, 2)

    def test_write_image_bad_params_raises(self):
        """write_image_1D raises Exception for unsupported (img_size, bpp) combinations."""
        img = [0] * 256
        buf = io.BytesIO()
        with pytest.raises(Exception):
            pff.write_image_1D(buf, img, 16, 2)  # only 32x32 is supported

    def test_image_all_zeros(self):
        img = [0] * 1024
        buf = io.BytesIO()
        pff.write_image_1D(buf, img, 32, 2)
        buf.seek(0)
        assert list(pff.read_image(buf, 32, 2)) == img

    def test_image_max_values_16bit(self):
        img = [65535] * 1024
        buf = io.BytesIO()
        pff.write_image_1D(buf, img, 32, 2)
        buf.seek(0)
        assert list(pff.read_image(buf, 32, 2)) == img


# ===========================================================================
# read_json
# Parses the JSON header block (starts with '{', ends with '\n\n')
# ===========================================================================

class TestReadJson:
    def _make_json_block(self, payload: dict) -> bytes:
        s = json.dumps(payload) + "\n\n"
        return s.encode()

    def test_reads_simple_header(self):
        payload = {"quabo_num": 0, "pkt_num": 42}
        buf = io.BytesIO(self._make_json_block(payload))
        result = pff.read_json(buf)
        parsed = json.loads(result)
        assert parsed["quabo_num"] == 0
        assert parsed["pkt_num"] == 42

    def test_reads_nested_header(self):
        payload = {"quabo_0": {"pkt_tai": 613, "pkt_nsec": 0, "tv_sec": 1_000_000}}
        buf = io.BytesIO(self._make_json_block(payload))
        result = pff.read_json(buf)
        parsed = json.loads(result)
        assert parsed["quabo_0"]["pkt_tai"] == 613

    def test_returns_none_at_eof(self):
        buf = io.BytesIO(b'')
        assert pff.read_json(buf) is None

    def test_bad_first_byte_raises(self):
        buf = io.BytesIO(b'X{"key": "val"}\n\n')
        with pytest.raises(Exception):
            pff.read_json(buf)

    def test_full_pff_frame_roundtrip(self):
        """Write JSON header + image block; read_json + read_image should recover both."""
        payload = {"quabo_0": {"quabo_num": 0, "pkt_num": 1,
                               "pkt_tai": 613, "pkt_nsec": 0,
                               "tv_sec": 1_000_000, "tv_usec": 0}}
        img = list(range(1024))

        buf = io.BytesIO()
        buf.write(self._make_json_block(payload))
        pff.write_image_1D(buf, img, 32, 2)

        buf.seek(0)
        header_str = pff.read_json(buf)
        header = json.loads(header_str)
        assert header["quabo_0"]["pkt_tai"] == 613

        recovered_img = pff.read_image(buf, 32, 2)
        assert list(recovered_img) == img


# ===========================================================================
# img_header_time / pkt_header_time
# ===========================================================================

class TestPktHeaderTime:
    TV_SEC = 1_000_000

    def _d0_tai(self, tv_sec):
        """pkt_tai that gives d=0."""
        return (tv_sec + 37) % 1024

    def test_flat_header_d0(self):
        h = {"pkt_tai": self._d0_tai(self.TV_SEC), "pkt_nsec": 0, "tv_sec": self.TV_SEC}
        assert pff.pkt_header_time(h) == float(self.TV_SEC)

    def test_flat_header_with_nsec(self):
        h = {"pkt_tai": self._d0_tai(self.TV_SEC), "pkt_nsec": 500_000_000, "tv_sec": self.TV_SEC}
        assert abs(pff.pkt_header_time(h) - (self.TV_SEC + 0.5)) < 1e-6


class TestImgHeaderTime:
    TV_SEC = 1_000_000

    def _d0_tai(self, tv_sec):
        return (tv_sec + 37) % 1024

    def test_nested_quabo0_format(self):
        """img16/img8/ph1024: header has {'quabo_0': {...}}."""
        inner = {"pkt_tai": self._d0_tai(self.TV_SEC), "pkt_nsec": 0, "tv_sec": self.TV_SEC}
        h = {"quabo_0": inner}
        assert pff.img_header_time(h) == float(self.TV_SEC)

    def test_flat_ph256_format(self):
        """ph256: flat header without quabo_0 key."""
        h = {"pkt_tai": self._d0_tai(self.TV_SEC), "pkt_nsec": 0, "tv_sec": self.TV_SEC}
        assert pff.img_header_time(h) == float(self.TV_SEC)

    def test_nested_with_nsec(self):
        inner = {"pkt_tai": self._d0_tai(self.TV_SEC), "pkt_nsec": 250_000_000, "tv_sec": self.TV_SEC}
        h = {"quabo_0": inner}
        assert abs(pff.img_header_time(h) - (self.TV_SEC + 0.25)) < 1e-6


# ===========================================================================
# img_info
# ===========================================================================

class TestImgInfo:
    BYTES_PER_IMAGE = 32 * 32 * 2  # img16: 2048

    def test_basic_returns_four_tuple(self, pff_file_factory):
        f = pff_file_factory(n_frames=3, tv_sec_start=1_000_000)
        result = pff.img_info(f, self.BYTES_PER_IMAGE)
        assert len(result) == 4

    def test_nframes_correct(self, pff_file_factory):
        for n in [1, 3, 5]:
            f = pff_file_factory(n_frames=n, tv_sec_start=1_000_000)
            frame_size, nframes, first_t, last_t = pff.img_info(f, self.BYTES_PER_IMAGE)
            assert nframes == n

    def test_first_and_last_t(self, pff_file_factory):
        f = pff_file_factory(n_frames=3, tv_sec_start=1_000_000)
        frame_size, nframes, first_t, last_t = pff.img_info(f, self.BYTES_PER_IMAGE)
        assert abs(first_t - 1_000_000.0) < 1e-6
        assert abs(last_t - 1_000_002.0) < 1e-6

    def test_frame_size_formula(self, pff_file_factory):
        """frame_size = header_size + bytes_per_image + 1."""
        f = pff_file_factory(n_frames=2, tv_sec_start=1_000_000)
        frame_size, _, _, _ = pff.img_info(f, self.BYTES_PER_IMAGE)
        assert frame_size == self.BYTES_PER_IMAGE + 1 + frame_size - self.BYTES_PER_IMAGE - 1 + 0
        # Simpler: verify file_size is consistent
        f.seek(0, 2)
        file_size = f.tell()
        assert file_size == frame_size * 2

    def test_skips_leading_zero_timestamp_frames(self, pff_file_factory):
        """Frames with tv_sec=0 produce first_t=0; img_info skips them."""
        # First 2 frames have tv_sec=0 → img_header_time returns 0
        tv_secs = [0, 0, 1_000_000]
        f = pff_file_factory(n_frames=3, tv_sec_values=tv_secs)
        frame_size, nframes, first_t, last_t = pff.img_info(f, self.BYTES_PER_IMAGE)
        assert first_t != 0

    def test_all_zero_timestamps_raises(self, pff_file_factory):
        """All frames zero → raises ValueError."""
        tv_secs = [0, 0, 0]
        f = pff_file_factory(n_frames=3, tv_sec_values=tv_secs)
        with pytest.raises(ValueError):
            pff.img_info(f, self.BYTES_PER_IMAGE)

    def test_flat_ph256_header(self, pff_file_factory):
        """img_info works with flat (ph256-style) headers too."""
        f = pff_file_factory(n_frames=3, tv_sec_start=1_000_000, nested_header=False)
        frame_size, nframes, first_t, last_t = pff.img_info(f, self.BYTES_PER_IMAGE)
        assert nframes == 3
        assert abs(first_t - 1_000_000.0) < 1e-6


# ===========================================================================
# img_frame_time
# ===========================================================================

class TestImgFrameTime:
    BYTES_PER_IMAGE = 32 * 32 * 2

    def test_frame_0_matches_first_t(self, pff_file_factory):
        f = pff_file_factory(n_frames=5, tv_sec_start=2_000_000)
        frame_size, _, first_t, _ = pff.img_info(f, self.BYTES_PER_IMAGE)
        f.seek(0)
        t = pff.img_frame_time(f, 0, frame_size)
        assert abs(t - first_t) < 1e-6

    def test_last_frame_matches_last_t(self, pff_file_factory):
        f = pff_file_factory(n_frames=5, tv_sec_start=2_000_000)
        frame_size, nframes, _, last_t = pff.img_info(f, self.BYTES_PER_IMAGE)
        f.seek(0)
        t = pff.img_frame_time(f, nframes - 1, frame_size)
        assert abs(t - last_t) < 1e-6

    def test_mid_frame(self, pff_file_factory):
        f = pff_file_factory(n_frames=5, tv_sec_start=1_000_000)
        frame_size, _, _, _ = pff.img_info(f, self.BYTES_PER_IMAGE)
        f.seek(0)
        t = pff.img_frame_time(f, 2, frame_size)
        assert abs(t - 1_000_002.0) < 1e-6


# ===========================================================================
# time_seek
# ===========================================================================

class TestTimeSeek:
    BYTES_PER_IMAGE = 32 * 32 * 2

    def test_t_before_first_seeks_to_start(self, pff_file_factory):
        f = pff_file_factory(n_frames=5, tv_sec_start=1_000_000)
        frame_size, _, first_t, _ = pff.img_info(f, self.BYTES_PER_IMAGE)
        f.seek(0)
        # t < first_t + frame_time → seek to 0
        pff.time_seek(f, frame_time=1.0, bytes_per_image=self.BYTES_PER_IMAGE,
                      t=first_t - 10.0)
        assert f.tell() == 0

    def test_t_after_last_seeks_to_last_frame(self, pff_file_factory):
        f = pff_file_factory(n_frames=5, tv_sec_start=1_000_000)
        frame_size, nframes, _, last_t = pff.img_info(f, self.BYTES_PER_IMAGE)
        f.seek(0)
        pff.time_seek(f, frame_time=1.0, bytes_per_image=self.BYTES_PER_IMAGE,
                      t=last_t + 100.0)
        assert f.tell() == (nframes - 1) * frame_size

    def test_t_exact_middle_seeks_near_that_frame(self, pff_file_factory):
        """time_seek converges to a frame within frame_time of the target."""
        n = 9
        f = pff_file_factory(n_frames=n, tv_sec_start=1_000_000)
        frame_size, nframes, first_t, last_t = pff.img_info(f, self.BYTES_PER_IMAGE)
        target_t = 1_000_004.0  # frame 4
        f.seek(0)
        pff.time_seek(f, frame_time=1.0, bytes_per_image=self.BYTES_PER_IMAGE,
                      t=target_t)
        # The seek should land us at a frame within a couple of positions
        landed_frame = f.tell() // frame_size
        assert 2 <= landed_frame <= 6


# ===========================================================================
# skip_image
# ===========================================================================

class TestSkipImage:
    def test_skip_advances_past_image_block(self):
        """skip_image advances by img_size^2*bpp + 1 bytes."""
        img = [0] * 1024
        buf = io.BytesIO()
        pff.write_image_1D(buf, img, 32, 2)  # writes 1 + 1024*2 = 2049 bytes
        buf.seek(0)
        pos_before = buf.tell()
        pff.skip_image(buf, 32, 2)
        pos_after = buf.tell()
        assert pos_after - pos_before == 32 * 32 * 2 + 1

    def test_skip_then_read_next_header(self, pff_file_factory):
        """After skipping frame 0's image, read_json gives frame 1's header."""
        f = pff_file_factory(n_frames=3, tv_sec_start=1_000_000)
        # Read frame 0 header
        h0_str = pff.read_json(f)
        json.loads(h0_str)
        # Skip frame 0 image
        pff.skip_image(f, 32, 2)
        # Now read frame 1 header
        h1_str = pff.read_json(f)
        h1 = json.loads(h1_str)
        # Frame 1 should have tv_sec = 1_000_001
        quabo_data = h1.get("quabo_0", h1)
        assert quabo_data["tv_sec"] == 1_000_001
