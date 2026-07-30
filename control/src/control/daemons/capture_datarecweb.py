#!/usr/bin/env python3
"""
Continuously monitor datarec_YYYYMMDD.log and upload it to cylon as
datarec_current.log ONLY if the file exists and has been updated.

Source:
  /mnt/data11/data/palomar/L0/YYYYMMDD/obslogs/datarec_YYYYMMDD.log

Destination:
  panoseti@132.239.146.24:/web/panoseti-palomar/logs/datarec_current.log

Default behavior: INFINITE LOOP
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------- CONFIG ----------------
L0_ROOT = "/mnt/data11/data/palomar/L0"
OBSLOGS_SUBDIR = "obslogs"

REMOTE_SERVER = "panoseti@132.239.146.24"
REMOTE_DIR = "/web/panoseti-palomar/logs"
REMOTE_DIR2 = "/web/panoseti-palomar/logs"
REMOTE_NAME = "datarec_current.log"

BANDWIDTH_LIMIT = 40000  # kbit/s
INTERVAL = 60            # seconds

STATE_FILE = "/tmp/datarec_copy_state.json"
# ----------------------------------------


def run_cmd(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True)


def today_yyyymmdd_utc() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")


def resolve_yyyymmdd(arg: str | None) -> str:
    if arg is None:
        return today_yyyymmdd_utc()
    if not re.fullmatch(r"\d{8}", arg):
        raise ValueError(f"Invalid date: {arg}")
    return arg


def source_path(yyyymmdd: str) -> Path:
    return Path(L0_ROOT) / yyyymmdd / OBSLOGS_SUBDIR / f"datarec_{yyyymmdd}.log"


def file_signature(p: Path) -> dict[str, Any]:
    st = p.stat()
    return {
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def load_state() -> dict[str, Any]:
    try:
        return json.loads(Path(STATE_FILE).read_text())
    except Exception:
        return {}


def save_state(state: dict[str, Any]) -> None:
    Path(STATE_FILE).write_text(json.dumps(state, indent=2) + "\n")


def upload_to_cylon(src: Path) -> None:
    tmp = Path("/tmp") / src.name
    shutil.copy2(src, tmp)

    run_cmd(
        f"scp -l {BANDWIDTH_LIMIT} '{tmp}' "
        f"{REMOTE_SERVER}:{REMOTE_DIR}/{REMOTE_NAME}"
    )

    run_cmd(
        f"ssh {REMOTE_SERVER} "
        f"\"chmod 644 {REMOTE_DIR2}/{REMOTE_NAME} || true\""
    )


def process_once(yyyymmdd: str) -> None:
    src = source_path(yyyymmdd)

    if not src.exists():
        print(f"[datarec] missing: {src}")
        return

    sig = file_signature(src)
    state = load_state()
    key = f"datarec_{yyyymmdd}"

    if state.get(key) == sig:
        print(f"[datarec] unchanged ? skip ({src.name})")
        return

    print(f"[datarec] updated ? upload ({src.name})")
    upload_to_cylon(src)

    state[key] = sig
    state["last_upload_utc"] = datetime.now(UTC).isoformat()
    save_state(state)

    print("[datarec] upload complete")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYYMMDD (default: today UTC)")
    ap.add_argument("--once", action="store_true", help="Run once and exit")
    ap.add_argument("--interval", type=int, default=INTERVAL)
    args = ap.parse_args()

    print(f"[datarec] monitoring every {args.interval}s (Ctrl-C to stop)")

    while True:
        try:
            yyyymmdd = resolve_yyyymmdd(args.date)
            process_once(yyyymmdd)
        except Exception as e:
            print(f"[datarec] ERROR: {e}", file=sys.stderr)

        if args.once:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()

