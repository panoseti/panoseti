"""
corpus.py — PFFCorpus: access to the realistic PFF data corpus.

The primary corpus is the real Lick observatory data bundled inside the
panoseti_grpc package as package data (grpc/pyproject.toml line 86):

    [tool.setuptools.package-data]
    "panoseti_grpc.daq_data" = ["simulated_data_dir/**/*"]

This corpus ships with img16 + ph256 PFF files for modules 1 and 3, plus
the supporting config files (obs_config.json, daq_config.json, etc.) and
a 'recording_ended' marker.

Override the corpus root via qa.toml [corpus] path = "..." or the env var
PSETI_V2_CORPUS_PATH.

Synthetic generation:
  PFFCorpus.make_synthetic() writes fixed-width padded JSON headers so that
  pypff.io2.PFFSequence._analyze_offsets() can extract stable byte offsets for
  vectorized metadata extraction.  Use PFFCorpus.validate(run_dir) to verify.
"""

from __future__ import annotations

import importlib.resources
import io
import os
import pathlib
import struct
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Fixed-width JSON field widths for pypff.io2 compatibility.
# Each integer value is right-aligned in a field of this many characters so
# that byte offsets within the header are stable across all frames.
# ---------------------------------------------------------------------------
_HEADER_SIZE = 640   # total content bytes before \n\n (covers 4-quabo module)
_QUABO_NUM_W = 2     # 0–3
_PKT_NUM_W = 10      # up to 9,999,999,999
_PKT_TAI_W = 6       # 0–1023
_PKT_NSEC_W = 12     # nanoseconds (0–999,999,999)
_TV_SEC_W = 12       # Unix timestamp
_TV_USEC_W = 8       # microseconds (0–999,999)

_IMG16_PIXELS = 1024  # 32 × 32
_PH256_PIXELS = 256   # 16 × 16


def _quabo_field_json(q: int, pkt_num: int, tv_sec: int, tv_usec: int) -> str:
    """Build one quabo sub-dict with fixed-width padded integers."""
    pkt_tai = (tv_sec + 37) % 1024
    return (
        f'"quabo_{q}": {{'
        f'"quabo_num": {q:{_QUABO_NUM_W}d}, '
        f'"pkt_num": {pkt_num:{_PKT_NUM_W}d}, '
        f'"pkt_tai": {pkt_tai:{_PKT_TAI_W}d}, '
        f'"pkt_nsec": {0:{_PKT_NSEC_W}d}, '
        f'"tv_sec": {tv_sec:{_TV_SEC_W}d}, '
        f'"tv_usec": {tv_usec:{_TV_USEC_W}d}'
        f'}}'
    )


def _build_header_bytes(quabo_fields: list[str]) -> bytes:
    """Assemble a fixed-size PFF JSON header (ends in \\n\\n)."""
    content = "{" + ", ".join(quabo_fields) + "}"
    if len(content) > _HEADER_SIZE:
        raise RuntimeError(
            f"PFF header content ({len(content)} chars) exceeds _HEADER_SIZE "
            f"({_HEADER_SIZE}); increase _HEADER_SIZE in corpus.py"
        )
    padded = content.ljust(_HEADER_SIZE)
    return (padded + "\n\n").encode("ascii")


def _img16_header(pkt_num: int, tv_sec: int, tv_usec: int) -> bytes:
    """4-quabo module header for img16 / img8 data products."""
    fields = [_quabo_field_json(q, pkt_num, tv_sec, tv_usec) for q in range(4)]
    return _build_header_bytes(fields)


def _ph256_header(pkt_num: int, tv_sec: int, tv_usec: int) -> bytes:
    """Single-quabo header for ph256 data products."""
    return _build_header_bytes([_quabo_field_json(0, pkt_num, tv_sec, tv_usec)])


# ---------------------------------------------------------------------------
# Corpus root resolution
# ---------------------------------------------------------------------------

def _default_corpus_root() -> pathlib.Path:
    """Resolve the corpus root from env / qa.toml / package data, in that order."""
    override = os.environ.get("PSETI_V2_CORPUS_PATH", "").strip()
    if override:
        return pathlib.Path(override)

    try:
        from ci.software_only_v2._qa_config import corpus_path as _cfg_path
        if _cfg_path:
            return pathlib.Path(_cfg_path)
    except ImportError:
        pass

    try:
        pkg_ref = importlib.resources.files("panoseti_grpc.daq_data") / "simulated_data_dir"
        return pathlib.Path(str(pkg_ref))
    except Exception:
        raise RuntimeError(
            "PFFCorpus: cannot locate simulated_data_dir. "
            "Install panoseti_grpc with 'pip install -e .[dev]' from the grpc/ directory, "
            "or set PSETI_V2_CORPUS_PATH to an existing PFF corpus directory."
        )


# ---------------------------------------------------------------------------
# ModuleCorpus
# ---------------------------------------------------------------------------

@dataclass
class ModuleCorpus:
    """PFF files available for a single module in the corpus."""
    module_id: int
    run_dir: pathlib.Path

    def _module_glob(self, pattern: str) -> list[pathlib.Path]:
        module_tag = f".module_{self.module_id}."
        return sorted(
            p for p in self.run_dir.rglob(pattern) if module_tag in p.name
        )

    def img16_files(self) -> list[pathlib.Path]:
        return self._module_glob("*.dp_img16.*.pff")

    def ph256_files(self) -> list[pathlib.Path]:
        return self._module_glob("*.dp_ph256.*.pff")

    def all_pff_files(self) -> list[pathlib.Path]:
        return self._module_glob("*.pff")

    def config_files(self) -> dict[str, pathlib.Path]:
        return {p.name: p for p in self.run_dir.glob("*.json")}

    def has_recording_ended_marker(self) -> bool:
        return (self.run_dir / "recording_ended").exists()


# ---------------------------------------------------------------------------
# PFFCorpus
# ---------------------------------------------------------------------------

class PFFCorpus:
    """
    Access to the realistic PFF data corpus from the panoseti_grpc package.

    The corpus contains real Lick observatory data:
    - Module 1: img16 + ph256 PFF files
    - Module 3: img16 + ph256 PFF files

    For synthetic parametric data, use make_synthetic() / validate().
    """

    def __init__(self, root: pathlib.Path | None = None) -> None:
        self.root = root if root is not None else _default_corpus_root()
        self._run_dir = self._discover_run_dir()

    def _discover_run_dir(self) -> pathlib.Path | None:
        if not self.root.exists():
            return None
        for item in self.root.iterdir():
            if item.is_dir() and item.suffix == ".pffd":
                return item
        return None

    @property
    def run_dir(self) -> pathlib.Path:
        if self._run_dir is None:
            raise RuntimeError(f"No .pffd run directory found under corpus root: {self.root}")
        return self._run_dir

    def for_module(self, module_id: int) -> ModuleCorpus:
        return ModuleCorpus(module_id=module_id, run_dir=self.run_dir)

    def available_modules(self) -> list[int]:
        modules: list[int] = []
        if self._run_dir is None:
            return modules
        for pff in self.run_dir.rglob("*.pff"):
            for part in pff.name.split("."):
                if part.startswith("module_"):
                    try:
                        mid = int(part.split("_")[1])
                        if mid not in modules:
                            modules.append(mid)
                    except (ValueError, IndexError):
                        pass
        return sorted(modules)

    def supporting_configs(self) -> dict[str, Any]:
        import json
        configs: dict[str, Any] = {}
        for path in self.run_dir.glob("*.json"):
            try:
                configs[path.name] = json.loads(path.read_text())
            except Exception:
                pass
        return configs

    # ------------------------------------------------------------------
    # Synthetic data generation
    # ------------------------------------------------------------------

    def make_synthetic(
        self,
        *,
        n_frames: int = 10,
        run_name: str = "synthetic_run",
        module_ids: list[int] | None = None,
        dest: pathlib.Path,
        tv_sec_start: int = 1_000_000,
        data_products: list[str] | None = None,
    ) -> pathlib.Path:
        """Generate a minimal synthetic PFF corpus under dest/{run_name}.pffd/.

        Produces both img16 and ph256 PFF files by default (both are required
        by UdsStrategy._load_source_data).

        Headers use fixed-width padded integer values (see _QUABO_NUM_W etc.)
        so that pypff.io2.PFFSequence._analyze_offsets() can derive stable byte
        offsets and get_metadata_arrays() returns correct int64 arrays.

        PFF files are placed directly in the run_dir (flat structure) so that
        pypff.io2.PanosetiRun can discover them via glob("*.pff").

        img16 payloads are written via panoseti_grpc.panoseti_util.pff.write_image_1D.
        ph256 payloads are inlined (write_image_1D only supports 32×32 images).

        Returns the run directory path.
        """
        from panoseti_grpc.panoseti_util import pff as pff_util

        if module_ids is None:
            module_ids = [200]
        if data_products is None:
            data_products = ["img16", "ph256"]

        run_dir = dest / f"{run_name}.pffd"
        run_dir.mkdir(parents=True, exist_ok=True)

        _zero_img16 = [0] * _IMG16_PIXELS
        _zero_ph256_bytes = struct.pack(f"{_PH256_PIXELS}H", *([0] * _PH256_PIXELS))

        for mid in module_ids:
            for dp in data_products:
                buf = io.BytesIO()

                for i in range(n_frames):
                    tv_sec = tv_sec_start + i
                    tv_usec = (i * 100_000) % 1_000_000

                    if dp in ("img16", "img8"):
                        buf.write(_img16_header(i, tv_sec, tv_usec))
                        # pff.write_image_1D handles 32×32, bpp=2 (img16) or bpp=1 (img8)
                        bpp = 2 if dp == "img16" else 1
                        pff_util.write_image_1D(buf, _zero_img16, 32, bpp)

                    elif dp == "ph256":
                        buf.write(_ph256_header(i, tv_sec, tv_usec))
                        # write_image_1D only supports 32×32; ph256 is 16×16
                        buf.write(b"*")
                        buf.write(_zero_ph256_bytes)

                    else:
                        continue

                bpp_map = {"img16": 2, "img8": 1, "ph256": 2}
                bpp = bpp_map.get(dp, 2)
                pff_name = (
                    f"start_{run_name}"
                    f".dp_{dp}"
                    f".bpp_{bpp}"
                    f".dome_0"
                    f".module_{mid}"
                    f".seqno_0"
                    f".pff"
                )
                (run_dir / pff_name).write_bytes(buf.getvalue())

        (run_dir / "recording_ended").touch()
        return run_dir

    # ------------------------------------------------------------------
    # Validation via pypff.io2
    # ------------------------------------------------------------------

    def validate(self, run_dir: pathlib.Path) -> None:
        """Validate PFF files in run_dir using pypff.io2.PanosetiRun.

        Checks that:
        - At least one PFF product is discoverable.
        - pkt_num values are monotonically non-decreasing across all frames
          for every product.

        Raises:
            ImportError: if pypff is not installed.
            AssertionError: if validation fails.
        """
        try:
            from pypff import io2
        except ImportError as exc:
            raise ImportError(
                "pypff is not installed. Install with: "
                "pip install -e /path/to/panoseti/pypff"
            ) from exc

        run = io2.PanosetiRun(run_dir)
        assert run.products, (
            f"pypff.io2.PanosetiRun found no PFF products in {run_dir}. "
            "Check that PFF files are in the run_dir root (not in subdirectories)."
        )

        for product_key, seq in run.products.items():
            if len(seq) == 0:
                continue
            arrays = seq.get_metadata_arrays(["pkt_num"])
            pkt_nums = arrays.get("pkt_num")
            if pkt_nums is None or len(pkt_nums) < 2:
                continue
            diffs = pkt_nums[1:] - pkt_nums[:-1]
            assert (diffs >= 0).all(), (
                f"pkt_num is not monotone in product '{product_key}': "
                f"first decrease at frame index {int((diffs < 0).argmax()) + 1}"
            )
