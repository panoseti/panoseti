"""Head-side manifest IO and digest helpers."""
from __future__ import annotations

import hashlib
import pathlib


def read_manifest_digest(manifest_path: pathlib.Path) -> str:
    """Return SHA-256 hex digest of the manifest file content.

    Args:
        manifest_path: Path to the manifest file on the head node.

    Returns:
        Lowercase hex SHA-256 digest of the file's raw bytes.
    """
    h = hashlib.sha256()
    h.update(manifest_path.read_bytes())
    return h.hexdigest()


def read_manifest_lines(manifest_path: pathlib.Path) -> list[str]:
    """Return non-empty lines from a manifest file.

    Args:
        manifest_path: Path to the manifest file.

    Returns:
        List of non-empty, non-whitespace-only lines.
    """
    return [line for line in manifest_path.read_text().splitlines() if line.strip()]
