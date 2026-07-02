"""
tier1_unit/test_corpus.py — Unit tests for PFFCorpus synthetic generation.

Validates the round-trip:
  make_synthetic() → writes PFF files with fixed-width JSON headers
  validate()       → opens via pypff.io2.PanosetiRun, asserts monotone pkt_num

No Docker, no gRPC, no hardware required.
"""

from __future__ import annotations

import pathlib

import pytest

pytestmark = pytest.mark.tier1


@pytest.fixture
def synth_run(tmp_path: pathlib.Path):
    """Generate a minimal 2-module synthetic corpus and return the run_dir."""
    from ci.software_only.infra.corpus import PFFCorpus
    corpus = PFFCorpus.__new__(PFFCorpus)
    corpus.root = tmp_path
    corpus._run_dir = None
    run_dir = corpus.make_synthetic(
        n_frames=5,
        run_name="test_run",
        module_ids=[200, 201],
        dest=tmp_path,
        data_products=["img16", "ph256"],
    )
    return run_dir


class TestMakeSynthetic:
    def test_run_dir_created(self, synth_run: pathlib.Path) -> None:
        assert synth_run.exists()
        assert synth_run.suffix == ".pffd"

    def test_recording_ended_marker(self, synth_run: pathlib.Path) -> None:
        assert (synth_run / "recording_ended").exists()

    def test_img16_files_present(self, synth_run: pathlib.Path) -> None:
        img16_files = list(synth_run.glob("*.dp_img16.*.pff"))
        # One per module_id x one seqno
        assert len(img16_files) == 2, f"Expected 2 img16 files, got {img16_files}"

    def test_ph256_files_present(self, synth_run: pathlib.Path) -> None:
        ph256_files = list(synth_run.glob("*.dp_ph256.*.pff"))
        assert len(ph256_files) == 2, f"Expected 2 ph256 files, got {ph256_files}"

    def test_pff_files_readable_via_panoseti_util(self, synth_run: pathlib.Path) -> None:
        """Each PFF file must be parseable by the legacy pff module."""
        from panoseti_grpc.panoseti_util import pff as pff_util

        for pff_file in sorted(synth_run.glob("*.pff")):
            with open(pff_file, "rb") as f:
                frame_count = 0
                while True:
                    header = pff_util.read_json(f)
                    if header is None:
                        break
                    import json
                    parsed = json.loads(header)
                    assert isinstance(parsed, dict), f"Header not a dict in {pff_file.name}"
                    # Skip image payload
                    byte = f.read(1)
                    if byte == b"":
                        break
                    assert byte == b"*", f"Expected '*' payload marker, got {byte!r}"
                    # img16: 32*32*2=2048 bytes; ph256: 16*16*2=512 bytes
                    if "dp_img16" in pff_file.name:
                        f.read(2048)
                    elif "dp_ph256" in pff_file.name:
                        f.read(512)
                    frame_count += 1
            assert frame_count == 5, f"Expected 5 frames in {pff_file.name}, got {frame_count}"

    def test_pkt_num_byte_offset_stable_across_frames(self, synth_run: pathlib.Path) -> None:
        """pkt_num byte offset must be identical across all frames (required by pypff.io2)."""
        from ci.software_only.infra.corpus import _HEADER_SIZE, _PKT_NUM_W

        pff_file = sorted(synth_run.glob("*.dp_img16.*.pff"))[0]
        raw = pff_file.read_bytes()

        # Find the byte offset of the pkt_num value in the first frame header
        first_header = raw[: _HEADER_SIZE]
        tag = b'"pkt_num": '
        offset_in_first = first_header.index(tag) + len(tag)

        # frame_size = header + \n\n (2) + * (1) + 32*32*2 (2048) = _HEADER_SIZE+2051
        frame_size = _HEADER_SIZE + 2 + 1 + 2048

        for frame_idx in range(5):
            frame_start = frame_idx * frame_size
            header_start = frame_start
            value_bytes = raw[header_start + offset_in_first : header_start + offset_in_first + _PKT_NUM_W]
            value_str = value_bytes.decode("ascii").strip()
            assert value_str.isdigit(), (
                f"Frame {frame_idx}: pkt_num value region is not numeric: {value_bytes!r}"
            )
            assert int(value_str) == frame_idx, (
                f"Frame {frame_idx}: expected pkt_num={frame_idx}, got {value_str!r}"
            )


class TestValidate:
    def test_validate_passes_for_synthetic(self, synth_run: pathlib.Path) -> None:
        """validate() must not raise for correctly generated synthetic data."""
        from ci.software_only.infra.corpus import PFFCorpus
        corpus = PFFCorpus.__new__(PFFCorpus)
        corpus.root = synth_run.parent
        corpus._run_dir = synth_run
        pytest.importorskip("pypff", reason="pypff not installed")
        corpus.validate(synth_run)

    def test_validate_detects_non_monotone_pkt_num(self, tmp_path: pathlib.Path) -> None:
        """validate() must raise AssertionError when pkt_num decreases."""
        import struct

        from ci.software_only.infra.corpus import _IMG16_PIXELS, PFFCorpus, _img16_header
        pytest.importorskip("pypff", reason="pypff not installed")

        run_dir = tmp_path / "bad_run.pffd"
        run_dir.mkdir()

        # Write 3 frames with pkt_num: 0, 5, 3 (non-monotone at frame 3)
        import io as _io
        buf = _io.BytesIO()
        for pkt_num in [0, 5, 3]:
            buf.write(_img16_header(pkt_num, 1_000_000 + pkt_num, 0))
            buf.write(b"*")
            buf.write(struct.pack(f"{_IMG16_PIXELS}H", *([0] * _IMG16_PIXELS)))

        (run_dir / "start_bad.dp_img16.bpp_2.dome_0.module_200.seqno_0.pff").write_bytes(
            buf.getvalue()
        )
        (run_dir / "recording_ended").touch()

        corpus = PFFCorpus.__new__(PFFCorpus)
        corpus.root = tmp_path
        corpus._run_dir = run_dir
        with pytest.raises(AssertionError, match="monotone"):
            corpus.validate(run_dir)
