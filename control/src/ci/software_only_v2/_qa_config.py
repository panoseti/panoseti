"""Internal helper: load corpus path from qa.toml."""

from __future__ import annotations

import pathlib
import tomllib

_QA_TOML = pathlib.Path(__file__).parent / "qa.toml"

try:
    with open(_QA_TOML, "rb") as _fh:
        _raw = tomllib.load(_fh)
    corpus_path: str = _raw.get("corpus", {}).get("path", "")
except Exception:
    corpus_path = ""
