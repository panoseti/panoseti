"""
HW-05: Manifest round-trip + corruption detection on real volumes.

After HW-01 leaves an ARCHIVED run:
  1. Read manifest.blake3 on both DAQ and head node.
  2. Verify root digests match.
  3. Manually corrupt one byte in one PFF file on the head node.
  4. Call verify_manifest directly and assert the mismatch is detected with
     the exact file named in the error list.

Entry point: pseti test hw run -k HW_05
"""

from __future__ import annotations

import os
import pathlib

import pytest

from control.transfer.verify import verify_manifest
from control.utils.run_state import RunStateManager

pytestmark = pytest.mark.skip(reason="Skipped as per user request")

HEAD_DATA_DIR = pathlib.Path(os.getenv("HEAD_DATA_DIR", "/data/head"))


class TestHW05ManifestCorruptionDetection:
    """Integrity check on a completed archive."""

    def _find_archived_run(self) -> pathlib.Path:
        """Locate the most recent ARCHIVED run directory on the head node."""
        mgr = RunStateManager()
        ledger = mgr.get_state()
        assert ledger, "No ledger state — run HW-01 first"
        assert ledger.get("status") == "ARCHIVED", (
            f"Expected ARCHIVED ledger, got {ledger.get('status')}"
        )
        run_name = ledger["run_name"]
        run_dir = HEAD_DATA_DIR / run_name
        assert run_dir.exists(), f"Head-node run dir missing: {run_dir}"
        return run_dir

    def test_HW_05_manifest_exists_on_head_node(self) -> None:
        """At least one manifest.* file must be present after a successful archive."""
        run_dir = self._find_archived_run()
        manifests = list(run_dir.rglob("manifest.*"))
        assert manifests, f"No manifest file under {run_dir}"

    def test_HW_05_manifest_internally_consistent(self) -> None:
        """Every file listed in the manifest must match its recorded digest."""
        run_dir = self._find_archived_run()
        manifests = list(run_dir.rglob("manifest.*"))

        for manifest_path in manifests:
            ok, errors = verify_manifest(manifest_path, manifest_path.parent)
            assert ok, (
                f"Manifest {manifest_path} has integrity errors:\n"
                + "\n".join(errors)
            )

    def test_HW_05_corruption_detected_by_verify(self, tmp_path: pathlib.Path) -> None:
        """
        Flip one byte in a real PFF file on the head node (in a temp copy),
        then verify_manifest must detect the mismatch and name the exact file.
        """
        import shutil

        run_dir = self._find_archived_run()
        manifests = list(run_dir.rglob("manifest.*"))
        assert manifests

        # Use the first manifest + its first entry
        manifest_path = manifests[0]
        parent_dir = manifest_path.parent

        # Find the first PFF file listed in the manifest
        target_relpath: str | None = None
        with open(manifest_path) as f:
            for line in f:
                parts = line.strip().split("  ", 3)
                relpath = parts[-1] if len(parts) >= 3 else None
                if relpath and relpath.endswith(".pff") and (parent_dir / relpath).exists():
                    target_relpath = relpath
                    break

        if target_relpath is None:
            pytest.skip("No .pff file found in manifest — skip corruption test")

        # Work in a temp copy so we don't damage the real archive
        tmp_run = tmp_path / "run_copy"
        shutil.copytree(parent_dir, tmp_run)

        # Copy original manifest
        tmp_manifest = tmp_run / manifest_path.name

        # Corrupt one byte in the file
        target_file = tmp_run / target_relpath
        data = bytearray(target_file.read_bytes())
        data[0] ^= 0xFF
        target_file.write_bytes(bytes(data))

        ok, errors = verify_manifest(tmp_manifest, tmp_run)
        assert not ok, "verify_manifest must detect the corruption"
        assert any(target_relpath in e for e in errors), (
            f"verify_manifest must name the corrupted file in errors. Got: {errors}"
        )
