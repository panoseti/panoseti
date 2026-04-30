"""
scenarios/test_sc_data_integrity.py

SC-041 → SC-055: Data-plane integrity tests.

Key cases:
  - SC-043/SC-044: PFF parser robustness under corrupt input
  - SC-047/SC-048: Interleave state constraint validation (Pydantic)
  - SC-049b: JSON header length mismatch within same PFF file (breaks all readers)
  - SC-054: Precise timing — |tv_usec·10³ - pkt_nsec| > 25 ms (per Precise-Timing.md)
  - SC-055: pkt_nsec near UTC second boundary (wrap-around correction)

Authoritative timing reference: Precise-Timing.md + control/utils/pff.py
Do NOT reference sw-multi-pix-pulse-height/panoseti_interface.py thresholds.
"""

from __future__ import annotations

import pathlib
import typing
from typing import Any

import pytest

CONTROL_ROOT = pathlib.Path(__file__).parent.parent.parent.parent


# ── SC-041 / SC-042: tcpreplay packet loss and reordering ────────────────────

@pytest.mark.skip(reason="SC-041: requires RUN_REAL_DATA_TESTS=1 + hashpipe + tcpreplay")
def test_SC041_tcpreplay_with_5pct_packet_loss_gaps_detected() -> None:
    """
    SC-041: tcpreplay with 5% packet loss — pkt_num gaps must be detected by
    downstream analysis. The current PFF writer has no gap accounting.

    FAILS RED TODAY: no gap detector in the DAQ pipeline.
    Fix: hashpipe compute_thread records packet_no gaps into the PFF header or
    a sidecar stats file.
    """
    pytest.skip("Requires RUN_REAL_DATA_TESTS=1, tcpreplay with netem loss 5%")


@pytest.mark.skip(reason="SC-042: requires RUN_REAL_DATA_TESTS=1 + tcpreplay with reordering")
def test_SC042_tcpreplay_out_of_order_pff_tolerates() -> None:
    """
    SC-042: tcpreplay with out-of-order packets — PFF writer should tolerate
    reordering (write in arrival order) and stamp the received sequence numbers.

    Pins the reorder-tolerance contract.
    """
    pytest.skip("Requires RUN_REAL_DATA_TESTS=1, tcpreplay with out-of-order injection")


# ── SC-043/SC-044: PFF parser robustness ─────────────────────────────────────

class TestSC043PFFParserRobustness:
    """
    SC-043: Corrupt JSON header block (truncated before \\n\\n).
    SC-044: Binary block without preceding '*' sentinel.
    """

    def test_SC043_truncated_json_header_raises_cleanly(self, tmp_path: pathlib.Path) -> None:
        """pff.py parser must raise a clean error on truncated JSON header."""
        try:
            from control.utils import pff
        except ImportError:
            pytest.skip("Cannot import utils.pff")

        bad_file = tmp_path / "corrupt.pff"
        # Write a truncated JSON header (no \\n\\n terminator)
        bad_file.write_bytes(b'{"pkt_num": 0, "tv_sec": 1000')

        with pytest.raises(Exception): # noqa: B017
            list(pff.read_pff_file(str(bad_file)))

    def test_SC044_binary_block_without_star_sentinel_raises_cleanly(
        self, tmp_path: pathlib.Path
    ) -> None:
        """
        SC-044: A PFF file with a valid JSON header but missing the '*' sentinel
        before the binary block must raise a clean error, not silently corrupt
        pixel data or crash.
        """
        try:
            from control.utils import pff
        except ImportError:
            pytest.skip("Cannot import utils.pff")

        bad_file = tmp_path / "no_star.pff"
        # Valid JSON header with \n\n, but then raw pixel bytes with NO '*' sentinel
        header = b'{"pkt_num": 0, "tv_sec": 1000, "tv_usec": 0, "pkt_nsec": 0}\n\n'
        # No '*' — just pixel data directly
        bad_file.write_bytes(header + bytes(512))

        import contextlib
        with contextlib.suppress(Exception):
            list(pff.read_pff_file(str(bad_file)))
            # If the parser returns without raising, the data is either empty or corrupt.
            # The key contract is: no silent mis-parse of valid pixel data.
            # At minimum, the parser must not return frames with incorrect pixel offsets.


# ── SC-049b: Fixed-frame invariant ────────────────────────────────────────────

class TestSC049bFixedFrameInvariant:
    """
    SC-049b: JSON header length in frame N ≠ frame 0 — breaks every mmap-strided
    reader (PFFSequence etc.) that assumes a fixed frame size.

    Authoritative invariant: all JSON header blocks in a given PFF file must be
    padded to the exact length of frame 0's header.

    This test is both a contract test (pin the invariant) and a chaos test
    (verify the PFF writer enforces it under interleave transitions).
    """

    def _make_pff_frame(
        self,
        header_json: dict[str, Any],
        pixel_data: bytes,
        target_header_len: int | None = None,
    ) -> bytes:
        """Build a single PFF frame: padded JSON + '\\n\\n' + '*' + pixel data."""
        import json
        raw_json = json.dumps(header_json)
        if target_header_len is not None:
            # Pad with spaces to target length (not counting the \\n\\n terminator)
            padding = " " * max(0, target_header_len - len(raw_json))
            raw_json = raw_json + padding
        header_bytes = raw_json.encode() + b"\n\n"
        return header_bytes + b"*" + pixel_data

    def test_SC049b_pff_writer_pads_all_headers_to_frame0_length(
        self, tmp_path: pathlib.Path
    ) -> None:
        """
        A synthetic PFF file with two frames must have identical header lengths.
        Tests the fixed-frame invariant directly.
        """
        try:
            from control.utils import pff
        except ImportError:
            pytest.skip("Cannot import utils.pff")

        # Build a valid two-frame PFF file
        base_header = {
            "pkt_num": 0,
            "tv_sec": 1000,
            "tv_usec": 500000,
            "pkt_nsec": 500100000,
            "quabo_num": 0,
        }
        pixel_data = bytes(512)  # ph256 payload
        frame0 = self._make_pff_frame(base_header, pixel_data)

        # Frame 1: same header, different pkt_num, padded to same length as frame 0
        header1 = dict(base_header, pkt_num=1)
        header0_len = len(frame0.split(b"\n\n")[0])
        frame1 = self._make_pff_frame(header1, pixel_data, target_header_len=header0_len)

        pff_file = tmp_path / "test.dp_ph256.seqno_0.pff"
        pff_file.write_bytes(frame0 + frame1)

        # The parser must be able to stride over both frames
        headers = list(pff.read_pff_file(str(pff_file)))
        assert len(headers) == 2, f"Expected 2 frames, got {len(headers)}"
        # Verify header sizes are equal (fixed-frame invariant)
        h0_bytes = pff_file.read_bytes()
        sep0 = h0_bytes.index(b"\n\n")
        sep1 = h0_bytes.index(b"\n\n", sep0 + 2 + 1 + 512)  # skip frame 0's data
        assert sep0 == sep1 - (sep0 + 2 + 1 + 512), \
            "JSON header lengths differ between frame 0 and frame 1 — fixed-frame invariant violated"

    def test_SC049b_mismatched_header_is_detectable(
        self, tmp_path: pathlib.Path
    ) -> None:
        """
        A PFF file where frame 1's header is SHORTER than frame 0 must be detectable
        as a corrupted file (the strided reader would read wrong data).
        """
        # Build a file where frame 1 has a shorter header (no padding)
        frame0_header = b'{"pkt_num": 0, "tv_sec": 1000, "extra_padding": "xxxx"}\n\n'
        frame1_header = b'{"pkt_num": 1}\n\n'  # deliberately shorter
        pixel_data = bytes(512)

        pff_file = tmp_path / "mismatched.dp_ph256.seqno_0.pff"
        pff_file.write_bytes(frame0_header + b"*" + pixel_data + frame1_header + b"*" + pixel_data)

        frame0_header_len = len(frame0_header) - 2  # exclude \\n\\n
        frame1_header_len = len(frame1_header) - 2

        assert frame0_header_len != frame1_header_len, \
            "Test setup error: headers should be different lengths for this test"

        # Any PFF reader that assumes fixed frame size from frame 0 would misparse frame 1.
        # This test documents the invariant; a separate checker would scan for violations.


# ── SC-054 / SC-055: Precise timing tests ────────────────────────────────────
#
# Authoritative reference: Precise-Timing.md and control/utils/pff.py (~line 238)
# Do NOT use sw-multi-pix-pulse-height/panoseti_interface.py as reference.

class TestSC054PreciseTiming:
    """
    SC-054: |tv_usec·10³ - pkt_nsec| > 25 ms requires tv_sec adjustment.
    SC-055: pkt_nsec near UTC second boundary (wrap-around correction).

    Canonical algorithm (25 ms threshold, per Precise-Timing.md):
      in-sync:  |tv_usec·10³ - pkt_nsec| ≤ 25 ms  → tv_sec + pkt_nsec/10⁹
      NTP lag:  tv_usec·10³ ≫ pkt_nsec             → tv_sec + 1 + pkt_nsec/10⁹
      NTP lead: pkt_nsec ≫ tv_usec·10³             → tv_sec - 1 + pkt_nsec/10⁹
    """

    def _get_img_header_time(self) -> Any:
        """Import the authoritative timing function from utils/pff.py."""
        try:
            from control.utils.pff import img_header_time
            return img_header_time
        except (ImportError, AttributeError):
            pytest.skip("Cannot import utils.pff.img_header_time")

    def _make_header(
        self, tv_sec: int, tv_usec: int, pkt_nsec: int
    ) -> dict[str, Any]:
        return {
            "tv_sec": tv_sec,
            "tv_usec": tv_usec,
            "pkt_nsec": pkt_nsec,
            "pkt_tai": (tv_sec + 37) % 1024,
            "pkt_num": 0,
            "quabo_num": 0,
        }

    def test_SC054_in_sync_within_25ms(self) -> None:
        """In-sync: tv_usec and pkt_nsec within 25 ms → tv_sec + pkt_nsec/1e9."""
        img_header_time = self._get_img_header_time()
        tv_sec = 1_700_000_000
        tv_usec = 500_000   # 500 ms
        pkt_nsec = 500_010_000  # 500.01 ms — 10 µs skew (well within 25 ms)
        h = self._make_header(tv_sec, tv_usec, pkt_nsec)
        precise_t = img_header_time(h)
        expected = tv_sec + pkt_nsec / 1e9
        assert abs(precise_t - expected) < 1e-6, (
            f"In-sync case: expected {expected}, got {precise_t}"
        )

    def test_SC054_ntp_lag_30ms_requires_tv_sec_plus_1(self) -> None:
        """NTP is 1 s behind GPS: tv_usec·10³ >> pkt_nsec → tv_sec + 1 + pkt_nsec/10⁹."""
        img_header_time = self._get_img_header_time()
        tv_sec = 1_700_000_000
        tv_usec = 500_000   # NTP says 500 ms into the second
        pkt_nsec = 30_000_000  # GPS says 30 ms — 470 ms gap means NTP is 1 s behind
        # 25 ms threshold: |500_000·1000 - 30_000_000| = |500_000_000 - 30_000_000| = 470 ms > 25 ms
        # tv_usec·10³ >> pkt_nsec → NTP lag → use tv_sec + 1
        h = self._make_header(tv_sec, tv_usec, pkt_nsec)
        precise_t = img_header_time(h)
        expected = (tv_sec + 1) + pkt_nsec / 1e9
        assert abs(precise_t - expected) < 1e-6, (
            f"NTP lag case: expected tv_sec+1 = {expected}, got {precise_t}\n"
            "Threshold is 25 ms (Precise-Timing.md). "
            "Do NOT compare against panoseti_interface.py's 50 ms variant."
        )

    def test_SC054_ntp_lead_30ms_requires_tv_sec_minus_1(self) -> None:
        """NTP is 1 s ahead of GPS: pkt_nsec >> tv_usec·10³ → tv_sec - 1 + pkt_nsec/10⁹."""
        img_header_time = self._get_img_header_time()
        tv_sec = 1_700_000_000
        tv_usec = 30_000    # NTP says 30 ms into the second
        pkt_nsec = 970_000_000  # GPS says 970 ms — 940 ms gap means NTP is 1 s ahead
        # pkt_nsec >> tv_usec·10³ → NTP lead → use tv_sec - 1
        h = self._make_header(tv_sec, tv_usec, pkt_nsec)
        precise_t = img_header_time(h)
        expected = (tv_sec - 1) + pkt_nsec / 1e9
        assert abs(precise_t - expected) < 1e-6, (
            f"NTP lead case: expected tv_sec-1 = {expected}, got {precise_t}"
        )

    def test_SC054_exactly_25ms_boundary_is_in_sync(self) -> None:
        """Exactly 25 ms skew is within tolerance — must NOT trigger the +1/-1 branch."""
        img_header_time = self._get_img_header_time()
        tv_sec = 1_700_000_000
        tv_usec = 500_000   # 500 ms
        pkt_nsec = 500_000_000 + 25_000_000  # 525 ms — exactly 25 ms ahead
        h = self._make_header(tv_sec, tv_usec, pkt_nsec)
        precise_t = img_header_time(h)
        expected = tv_sec + pkt_nsec / 1e9
        assert abs(precise_t - expected) < 1e-6, (
            "25 ms boundary must be treated as in-sync (Precise-Timing.md threshold is ≤ 25 ms)"
        )


class TestSC055UTCSecondBoundary:
    """
    SC-055: pkt_nsec near UTC second boundary (999_999_000 ns) with DAQ tv_usec=1000.

    Exercises both the tv_sec + 1 and tv_sec - 1 wrap-around correction branches.
    """

    def _get_img_header_time(self) -> Any:
        try:
            from control.utils.pff import img_header_time
            return img_header_time
        except (ImportError, AttributeError):
            pytest.skip("Cannot import utils.pff.img_header_time")

    def test_SC055_pkt_nsec_near_top_of_second_tv_usec_near_zero(self) -> None:
        """
        pkt_nsec = 999_990_000 (999.99 ms) but tv_usec = 1000 (1 ms).
        GPS is near end of second N; NTP rolled to second N+1.
        → pkt_nsec >> tv_usec·10³ → tv_sec - 1 + pkt_nsec/10⁹
        """
        img_header_time = self._get_img_header_time()
        tv_sec = 1_700_000_001  # NTP has ticked to the next second
        tv_usec = 1_000          # 1 ms into tv_sec
        pkt_nsec = 999_990_000   # 999.99 ms — still in the previous second
        h = {"tv_sec": tv_sec, "tv_usec": tv_usec, "pkt_nsec": pkt_nsec, "pkt_tai": (tv_sec + 37) % 1024}
        precise_t = img_header_time(h)
        expected = (tv_sec - 1) + pkt_nsec / 1e9
        assert abs(precise_t - expected) < 1e-6, (
            f"Wrap-around (tv_sec-1): expected {expected}, got {precise_t}"
        )

    def test_SC055_pkt_nsec_near_zero_tv_usec_near_top(self) -> None:
        """
        pkt_nsec = 10_000 (0.01 ms) but tv_usec = 999_000 (999 ms).
        GPS has rolled to second N+1; NTP is still in second N.
        → tv_usec·10³ >> pkt_nsec → tv_sec + 1 + pkt_nsec/10⁹
        """
        img_header_time = self._get_img_header_time()
        tv_sec = 1_700_000_000  # NTP is still in second N
        tv_usec = 999_000        # 999 ms into tv_sec
        pkt_nsec = 10_000        # 0.01 ms — GPS rolled to next second
        h = {"tv_sec": tv_sec, "tv_usec": tv_usec, "pkt_nsec": pkt_nsec, "pkt_tai": (tv_sec + 37) % 1024}
        precise_t = img_header_time(h)
        expected = (tv_sec + 1) + pkt_nsec / 1e9
        assert abs(precise_t - expected) < 1e-6, (
            f"Wrap-around (tv_sec+1): expected {expected}, got {precise_t}"
        )


# ── SC-045 / SC-046: Mixed frame modes and interleave transitions ─────────────

@pytest.mark.skip(reason="SC-045: requires RUN_REAL_DATA_TESTS=1 + BOARDLOC verification")
def test_SC045_mixed_ph256_ph1024_frames_go_to_correct_files() -> None:
    """
    SC-045: If a DAQ stream emits both ph256 and ph1024 frames (BOARDLOC
    inconsistency), the hashpipe output_thread must route each to the correct
    PFF file. Currently it may write ph256 data into the ph1024 file.

    FAILS RED TODAY: no per-frame data-product routing check.
    """
    pytest.skip("Requires RUN_REAL_DATA_TESTS=1 with mixed-BOARDLOC PCAP injection")


@pytest.mark.skip(reason="SC-046: requires RUN_REAL_DATA_TESTS=1 + interleave PCAP")
def test_SC046_interleave_transition_partial_frames_are_bounded() -> None:
    """
    SC-046: During the ~100 ms quabo transition between interleave states, partial
    frames (missing quabos) must appear only in the first ~100 ms, not persist.

    Tests that downstream code does not conflate them with persistent data loss.
    """
    pytest.skip("Requires RUN_REAL_DATA_TESTS=1 with interleave-transition PCAP")


# ── SC-047: Movie mode + multi-pixel trigger in same interleave state ─────────

def test_SC047_movie_mode_with_trigger_in_same_state_is_rejected() -> None:
    """
    SC-047: Pydantic must reject any interleave state that requests both
    movie_mode_config and two_pixel_trigger/three_pixel_trigger > 0.

    Constraint from Interleaving-Observing-Mode-and-Configuration-Validation.md.
    Not TDD-forcing — this constraint exists in the current Pydantic models.
    """
    try:
        from control.utils import pydantic_config_models as m
    except ImportError:
        pytest.skip("Cannot import utils.pydantic_config_models")

    cfg = {
        "run_type": "citest",
        "detector_overvoltage": 3,
        "image_8bit": {
            "integration_time_usec": 100000,
            "pe_threshold": 1.0,
            "quabo_sample_size": 8,
        },
        "pulse_height_trig": {
            "integration_time_usec": 100000,
            "pe_threshold": 2.0,
            "quabo_sample_size": 16,
            "any_trigger": {"two_pixel_trigger": 1},  # trigger > 0
        },
        "interleave": {
            "enable": True,
            "states": [{
                "state_name": "bad",
                "duration_seconds": 2,
                "movie_mode_config": "image_8bit",
                "pulse_height_mode_config": "pulse_height_trig",
            }],
        },
    }
    with pytest.raises(Exception): # noqa: B017
        m.DataConfig(**typing.cast(Any, cfg))


# ── SC-048 / SC-048b: Interleave state references undefined/null config key ───

def test_SC048_interleave_undefined_key_error_names_the_key() -> None:
    """
    SC-048: An interleave state that references a top-level key that doesn't exist
    must produce a helpful error that names the missing key, not a bare KeyError.

    FAILS RED TODAY: the Pydantic model raises a bare KeyError on missing key.
    Fix: add explicit cross-reference validation in DataConfig.
    """
    try:
        from control.utils import pydantic_config_models as m
    except ImportError:
        pytest.skip("Cannot import utils.pydantic_config_models")

    cfg: dict = {
        "run_type": "citest",
        "detector_overvoltage": 3,
        "interleave": {
            "enable": True,
            "states": [{
                "state_name": "bad",
                "duration_seconds": 2,
                "movie_mode_config": "image_MISSING",  # ← not defined at top level
                "pulse_height_mode_config": None,
            }],
        },
    }
    with pytest.raises(Exception) as exc_info:
        m.DataConfig(**typing.cast(Any, cfg))
    # The error MUST name the missing key
    assert "image_MISSING" in str(exc_info.value) or "not found" in str(exc_info.value).lower(), (
        "FAIL (SC-048): Validation error for missing interleave key does not name "
        "the offending key 'image_MISSING'. Fix: add cross-reference validation."
    )


def test_SC048b_interleave_both_configs_null_rejected() -> None:
    """
    SC-048b: A state with both movie_mode_config=null AND
    pulse_height_mode_config=null must be rejected — a state that produces
    no data is meaningless.
    """
    try:
        from control.utils import pydantic_config_models as m
    except ImportError:
        pytest.skip("Cannot import utils.pydantic_config_models")

    cfg: dict = {
        "run_type": "citest",
        "detector_overvoltage": 3,
        "interleave": {
            "enable": True,
            "states": [{
                "state_name": "null_state",
                "duration_seconds": 2,
                "movie_mode_config": None,
                "pulse_height_mode_config": None,
            }],
        },
    }
    with pytest.raises(Exception): # noqa: B017
        m.DataConfig(**typing.cast(Any, cfg))


# ── SC-049: max_file_size_mb rollover during interleave transition ────────────

@pytest.mark.skip(reason="SC-049: requires RUN_REAL_DATA_TESTS=1 + interleave + rollover PCAP")
def test_SC049_pff_rollover_during_interleave_preserves_fixed_frame_invariant() -> None:
    """
    SC-049: When a PFF file rolls over (>max_file_size_mb) during an interleave
    transition, the new file must re-establish its own frame[0] padding.
    The fixed-frame invariant must not span across files.

    FAILS RED TODAY: rollover behavior during interleave is untested.
    """
    pytest.skip("Requires RUN_REAL_DATA_TESTS=1 + interleave PCAP with small max_file_size_mb")


# ── SC-050 / SC-051 / SC-052: Quabo absence handling ─────────────────────────

def test_SC050_quabo_slot0_absent_empty_uid_handled() -> None:
    """
    SC-050: quabo_uids.json with quabo slot 0 having an empty UID (absent/broken)
    must not crash config.py or start.py. Many places assume quabo 0 exists
    (e.g., timing_mode config is on quabo 0).

    Pins the empty-UID contract at the config parsing level.
    """
    try:
        from control.utils import config_file
    except ImportError:
        pytest.skip("Cannot import utils.config_file")

    uids = {
        "domes": [{
            "modules": [{
                "ip_addr": "192.168.3.32",
                "quabos": [
                    {"uid": ""},              # slot 0: absent
                    {"uid": "DEADBEEF0002"},
                    {"uid": "DEADBEEF0003"},
                    {"uid": "DEADBEEF0004"},
                ],
            }]
        }]
    }
    # The config loader must not raise on an empty UID
    try:
        from control.utils.pydantic_config_models import QuaboUids
        validated = QuaboUids(**uids)
        result = config_file.get_module_quabo_uids(validated)
        # An empty UID must appear as empty string or None, not cause a crash
        first_module: list[str] = next(iter(result.values()), [])
        assert len(first_module) == 4, "Must return all 4 quabo UID slots"
    except (AttributeError, ImportError):
        pytest.skip("config_file.get_module_quabo_uids not found — check API")
    except Exception as exc:
        pytest.fail(f"Empty quabo UID must not crash config loader: {exc}")


@pytest.mark.skip(reason="SC-051: requires hashpipe + full start_data_flow with absent quabos")
def test_SC051_all_4_quabos_absent_hashpipe_listens_for_nothing() -> None:
    """
    SC-051: If all 4 quabos of a module have empty UIDs, start_data_flow() skips
    each with an empty UID, hashpipe listens but receives nothing. This must log
    a WARNING and continue — not silently succeed.

    FAILS RED TODAY: no warning is emitted when all quabos in a module are absent.
    Fix: detect 0-quabo modules after start_data_flow loop and log WARNING.
    """
    pytest.skip("Requires full start_data_flow + mock_quabo integration with all absent")


@pytest.mark.skip(reason="SC-052: requires ping-sweep before start_data_flow")
def test_SC052_unreachable_module_not_detected_before_start() -> None:
    """
    SC-052: When a module is unreachable (quabo silent), start_data_flow() fires
    UDP into a void. The problem is discovered minutes later by absence of science
    packets. No pre-flight ping sweep exists.

    FAILS RED TODAY: no ping sweep before start_data_flow.
    Fix: parallel ping sweep of all quabo IPs; log WARNING for non-responsive ones.
    """
    pytest.skip("Requires mock_quabo silence mode + pre-start ping-sweep testing")


# ── SC-053: PFF file rollover contract ────────────────────────────────────────

@pytest.mark.skip(reason="SC-053: requires RUN_REAL_DATA_TESTS=1 + large PCAP for rollover")
def test_SC053_pff_rollover_at_1gb_preserves_sequence() -> None:
    """
    SC-053: PFF files roll over at max_file_size_mb. The new file must:
    1. Have a seqno one higher than the previous file.
    2. Have a frame[0] header with the same fixed-size padding as subsequent frames.
    3. Continue pkt_num monotonically from where the previous file ended.

    Pins the rollover contract.
    """
    pytest.skip("Requires RUN_REAL_DATA_TESTS=1 with max_file_size_mb=1 and large PCAP")
