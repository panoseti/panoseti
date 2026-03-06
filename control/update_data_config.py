#!/usr/bin/env python3
"""
update_data_config.py

Update /home/obs/panoseti_mount/panoseti/control/configs/data_config.json
with pulse_height and/or image settings.

Rules:
- If --pulse-height is provided: set key "pulse_height" to that object.
  Else: remove "pulse_height" from config if present.
- If --image is provided: set key "image" to that object.
  Else: remove "image" from config if present.

Writes atomically (temp file + replace) and preserves other keys in the config.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG_PATH = "/home/obs/panoseti_mount/panoseti/control/configs/data_config.json"
CYLON_DEST = "cylon:/web/panoseti-palomar/current/data_config_current.json"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("Top-level JSON must be an object/dict.")
        return obj
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON from {path}: {e}") from e


def _parse_obj(s: Optional[str], flagname: str) -> Optional[Dict[str, Any]]:
    if s is None:
        return None
    try:
        obj = json.loads(s)
    except Exception as e:
        raise RuntimeError(f"Invalid JSON for {flagname}: {e}\nValue was: {s}") from e
    if not isinstance(obj, dict):
        raise RuntimeError(f"{flagname} must be a JSON object/dict, got {type(obj).__name__}")
    return obj


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(str(tmp_path), str(path))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _best_effort_copy_to_cylon(src_path: Path) -> None:
    # Best-effort only: do not raise, do not mkdir, do not print.
    try:
        subprocess.run(
            ["scp", "-q", str(src_path), CYLON_DEST],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except Exception:
        # Includes: cylon unreachable, scp not installed, auth issues, timeout, etc.
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Update data_config.json pulse_height/image blocks.")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to data_config.json")
    ap.add_argument("--pulse-height", default=None, help='JSON object for "pulse_height" (string)')
    ap.add_argument("--image", default=None, help='JSON object for "image" (string)')
    args = ap.parse_args()

    cfg_path = Path(args.config)

    pulse = _parse_obj(args.pulse_height, "--pulse-height")
    image = _parse_obj(args.image, "--image")

    cfg = _load_json(cfg_path)

    # Apply rule: missing flag => remove key
    if pulse is None:
        cfg.pop("pulse_height", None)
    else:
        cfg["pulse_height"] = pulse

    if image is None:
        cfg.pop("image", None)
    else:
        cfg["image"] = image

    _atomic_write_json(cfg_path, cfg)

    # Best-effort mirror copy to cylon (ignore any failure)
    _best_effort_copy_to_cylon(cfg_path)

    # Small stdout signal for logs
    print(f"OK updated {cfg_path} | pulse_height={'set' if pulse else 'removed'} | image={'set' if image else 'removed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

