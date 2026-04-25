"""Parse rsync --info=progress2 output for transfer progress reporting."""
from __future__ import annotations

import re

_PROGRESS_RE = re.compile(
    r"^\s*(?P<bytes>[\d,]+)\s+(?P<pct>\d+)%\s+(?P<speed>\S+)\s+(?P<eta>\S+)"
)


def parse_rsync_progress(line: str) -> dict | None:
    """Parse a single rsync --info=progress2 output line.

    Returns a dict with bytes, pct, speed, eta, or None if not a progress line.

    Args:
        line: A single line from rsync standard output.

    Returns:
        Dict with keys ``bytes`` (int), ``pct`` (int), ``speed`` (str),
        ``eta`` (str), or ``None`` if the line does not match the progress
        format.
    """
    m = _PROGRESS_RE.match(line)
    if not m:
        return None
    return {
        "bytes": int(m.group("bytes").replace(",", "")),
        "pct": int(m.group("pct")),
        "speed": m.group("speed"),
        "eta": m.group("eta"),
    }
