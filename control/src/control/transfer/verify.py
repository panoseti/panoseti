"""Manifest verification for transferred run data."""
from __future__ import annotations

import hashlib
import pathlib


def _compute_digest(data: bytes, algo: str) -> str:
    """Compute the hex digest of *data* using *algo*.

    Args:
        data: Raw file bytes.
        algo: One of ``"blake3"``, ``"xxh3_128"``, or ``"sha256"``.

    Returns:
        Lowercase hex digest string.

    Raises:
        ValueError: If the requested algorithm library is not installed.
    """
    if algo == "blake3":
        try:
            import blake3 as _blake3
            return str(_blake3.blake3(data).hexdigest())
        except ImportError as e:
            raise ValueError(
                "Algorithm 'blake3' requested but 'blake3' library is not installed."
            ) from e
    if algo in ("xxh3_128", "xxhash"):
        try:
            import xxhash
            return str(xxhash.xxh3_128(data).hexdigest())
        except ImportError as e:
            raise ValueError(
                "Algorithm 'xxh3_128' requested but 'xxhash' library is not installed."
            ) from e
    return hashlib.sha256(data).hexdigest()


def _algo_from_path(manifest_path: pathlib.Path) -> str:
    """Infer the hash algorithm from the manifest file name or suffix.

    Supports:
        1. Legacy format: ``manifest.<algo>``
        2. New format: ``dp_manifest.node_<hostname>.algo_<algo>.txt``

    Args:
        manifest_path: Path to the manifest file.

    Returns:
        Algorithm name string suitable for ``_compute_digest``.
    """
    name = manifest_path.name
    # 1. New format: algo_<algo>
    if ".algo_" in name:
        parts = name.split(".algo_")
        if len(parts) > 1:
            algo_part = parts[1].split(".")[0]
            if algo_part in {"blake3", "xxh3_128", "sha256"}:
                return algo_part

    # 2. Legacy format: manifest.<algo>
    suffix = manifest_path.suffix.lstrip(".")
    known = {"blake3", "xxh3_128", "sha256"}
    return suffix if suffix in known else "sha256"


def verify_manifest(
    manifest_path: pathlib.Path,
    data_dir: pathlib.Path,
) -> tuple[bool, list[str]]:
    """Verify files listed in a manifest against their on-disk digests.

    Reads a manifest file in 4-column format::

        <digest>  <size>  <mtime_ns>  <relpath>

    or 3-column format::

        <digest>  <size>  <relpath>

    and recomputes each file's digest, comparing it against the recorded
    value.  The hashing algorithm is inferred from the manifest file suffix
    (e.g. ``manifest.blake3`` -> blake3, ``manifest.sha256`` -> SHA-256).

    Args:
        manifest_path: Path to the manifest file to read.
        data_dir: Root directory on the head node against which relative paths
            in the manifest are resolved.

    Returns:
        A ``(all_ok, errors)`` tuple.  ``all_ok`` is True when every file
        passes; ``errors`` is a list of human-readable failure descriptions.
    """
    errors: list[str] = []

    if not manifest_path.exists():
        return False, [f"Manifest not found: {manifest_path}"]

    algo = _algo_from_path(manifest_path)

    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("  ", 3)
            if len(parts) == 4:
                expected_digest, _size_str, _mtime_str, relpath = parts
            elif len(parts) == 3:
                expected_digest, _size_str, relpath = parts
            else:
                continue

            filepath = data_dir / relpath
            if not filepath.exists():
                errors.append(f"Missing: {relpath}")
                continue

            raw = filepath.read_bytes()
            actual = _compute_digest(raw, algo)

            if actual != expected_digest:
                errors.append(f"Digest mismatch: {relpath}")

    return len(errors) == 0, errors
