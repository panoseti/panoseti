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
"""

from __future__ import annotations

import importlib.resources
import os
import pathlib
from dataclasses import dataclass
from typing import Any


def _default_corpus_root() -> pathlib.Path:
    """Resolve the corpus root from env / qa.toml / package data, in that order."""
    # 1. Env override
    override = os.environ.get("PSETI_V2_CORPUS_PATH", "").strip()
    if override:
        return pathlib.Path(override)

    # 2. qa.toml override
    try:
        from ci.software_only_v2._qa_config import corpus_path as _cfg_path
        if _cfg_path:
            return pathlib.Path(_cfg_path)
    except ImportError:
        pass

    # 3. Default: importlib.resources from the installed panoseti_grpc package
    try:
        pkg_ref = importlib.resources.files("panoseti_grpc.daq_data") / "simulated_data_dir"
        # Traverse into the first run directory if present
        return pathlib.Path(str(pkg_ref))
    except Exception:
        raise RuntimeError(
            "PFFCorpus: cannot locate simulated_data_dir. "
            "Install panoseti_grpc with 'pip install -e .[dev]' from the grpc/ directory, "
            "or set PSETI_V2_CORPUS_PATH to an existing PFF corpus directory."
        )


@dataclass
class ModuleCorpus:
    """PFF files available for a single module in the corpus."""
    module_id: int
    run_dir: pathlib.Path

    def img16_files(self) -> list[pathlib.Path]:
        return sorted(self.run_dir.rglob("*.img16.*.pff"))

    def ph256_files(self) -> list[pathlib.Path]:
        return sorted(self.run_dir.rglob("*.ph256.*.pff"))

    def all_pff_files(self) -> list[pathlib.Path]:
        return sorted(self.run_dir.rglob("*.pff"))

    def config_files(self) -> dict[str, pathlib.Path]:
        """Returns a dict of config filename → path for JSON configs in the run dir."""
        return {p.name: p for p in self.run_dir.glob("*.json")}

    def has_recording_ended_marker(self) -> bool:
        return (self.run_dir / "recording_ended").exists()


class PFFCorpus:
    """
    Access to the realistic PFF data corpus from the panoseti_grpc package.

    The corpus contains real Lick observatory data:
    - Module 1: img16 + ph256 PFF files
    - Module 3: img16 + ph256 PFF files

    For synthetic parametric data, use make_synthetic().
    """

    def __init__(self, root: pathlib.Path | None = None) -> None:
        self.root = root if root is not None else _default_corpus_root()
        self._run_dir = self._discover_run_dir()

    def _discover_run_dir(self) -> pathlib.Path | None:
        """Find the first .pffd directory inside the corpus root."""
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
        """Return the corpus slice for the given module_id."""
        return ModuleCorpus(module_id=module_id, run_dir=self.run_dir)

    def available_modules(self) -> list[int]:
        """Return the list of module IDs available in the corpus."""
        from control.utils.config_file import ip_addr_to_module_id
        modules = []
        if self._run_dir is None:
            return modules
        # Scan PFF filenames for module IDs
        for pff in self.run_dir.rglob("*.pff"):
            # Filename convention: ...module_{N}.seqno_{M}.pff
            parts = pff.name.split(".")
            for part in parts:
                if part.startswith("module_"):
                    try:
                        mid = int(part.split("_")[1])
                        if mid not in modules:
                            modules.append(mid)
                    except (ValueError, IndexError):
                        pass
        return sorted(modules)

    def supporting_configs(self) -> dict[str, Any]:
        """Return parsed JSON configs that come with the corpus run."""
        import json
        configs: dict[str, Any] = {}
        for path in self.run_dir.glob("*.json"):
            try:
                configs[path.name] = json.loads(path.read_text())
            except Exception:
                pass
        return configs

    def make_synthetic(
        self,
        *,
        n_frames: int = 10,
        run_name: str = "synthetic_run",
        module_ids: list[int] | None = None,
        dest: pathlib.Path,
        tv_sec_start: int = 1_000_000,
    ) -> pathlib.Path:
        """
        Generate a minimal synthetic PFF corpus under dest/{run_name}.pffd/.

        Uses panoseti_grpc.panoseti_util.pff.write_image_1D for frame generation.
        Returns the run directory path.
        """
        import io
        import json
        import struct

        if module_ids is None:
            module_ids = [200]

        run_dir = dest / f"{run_name}.pffd"
        run_dir.mkdir(parents=True, exist_ok=True)

        for mid in module_ids:
            mod_dir = run_dir / f"module_{mid}"
            mod_dir.mkdir(exist_ok=True)

            # Build a minimal img16 PFF file
            buf = io.BytesIO()
            n_pixels = 32 * 32
            fmt = f"{n_pixels}H"
            image_bytes = b"*" + struct.pack(fmt, *([0] * n_pixels))

            for i in range(n_frames):
                tv_sec = tv_sec_start + i
                pkt_tai = (tv_sec + 37) % 1024
                header_dict = {
                    "quabo_0": {
                        "quabo_num": 0,
                        "pkt_num": i,
                        "pkt_tai": pkt_tai,
                        "pkt_nsec": 0,
                        "tv_sec": tv_sec,
                        "tv_usec": 0,
                    }
                }
                header_str = json.dumps(header_dict)
                # Pad to fixed width so pff.time_seek works
                padded = header_str + " " * max(0, 120 - len(header_str))
                buf.write((padded + "\n\n").encode())
                buf.write(image_bytes)

            pff_name = (
                f"start_{run_name}.dp_img16.bpp_2.dome_0.module_{mid}.seqno_0.pff"
            )
            (mod_dir / pff_name).write_bytes(buf.getvalue())

        # Write a recording_ended marker
        (run_dir / "recording_ended").touch()

        return run_dir
