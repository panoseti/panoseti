from __future__ import annotations

import hashlib
import pathlib


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
    value.  The hashing algorithm is inferred from the digest length: a
    32-character hex string triggers xxhash XXH3-128 (when ``xxhash`` is
    available), and a 64-character string triggers SHA-256.  Unknown lengths
    fall back to SHA-256.

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
                # Unrecognised format — skip silently (forward-compatibility).
                continue

            filepath = data_dir / relpath
            if not filepath.exists():
                errors.append(f"Missing: {relpath}")
                continue

            raw = filepath.read_bytes()
            if len(expected_digest) == 32:
                # Attempt XXH3-128 first; fall back to SHA-256 if unavailable.
                try:
                    import xxhash

                    actual = xxhash.xxh3_128(raw).hexdigest()
                except ImportError:
                    actual = hashlib.sha256(raw).hexdigest()
            else:
                actual = hashlib.sha256(raw).hexdigest()

            if actual != expected_digest:
                errors.append(f"Digest mismatch: {relpath}")

    return len(errors) == 0, errors
