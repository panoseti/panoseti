#!/usr/bin/env python3
"""
Continuously export latest QUABO HV/TEMP values to JSON (InfluxDB 1.x),
and upload the JSON to Synology cylon using the SAME SSH/SCP parameters
as the working allsky script (REMOTE_SERVER + scp -l + chmod via ssh).

- Loop every INTERVAL seconds
- Filters out stale points older than MAX_AGE_MIN minutes
- Handles Influx time returned as RFC3339 string OR epoch ns
- Writes local JSON, then uploads to:
    REMOTE_SERVER:/web/panoseti-palomar/logs/quabo_hvtemps_latest.json

Requires:
  pip install influxdb
"""

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from influxdb import InfluxDBClient

# ---------------- CONFIG ----------------
# Influx
INFLUX_HOST = "localhost"
INFLUX_PORT = 8086
INFLUX_USER = None
INFLUX_PASS = None
INFLUX_DB   = "metadata"

# Staleness + loop cadence
MAX_AGE_MIN = 100
INTERVAL    = 30   # seconds

# NEW: if newest Influx point is older than this, do NOT upload to cylon
MAX_UPLOAD_AGE_SECONDS = 20

# Output
OUT_JSON = "quabo_hvtemps_latest.json"

# Cylon upload config (copied style from your working script)
REMOTE_SERVER = "panoseti@132.239.146.24"
REMOTE_DIR = "/web/panoseti-palomar/logs"
REMOTE_DIR2 = "/web/panoseti-palomar/logs"   # Used only for chmod
BANDWIDTH_LIMIT = 40000  # kbit/s scp limit
# ----------------------------------------


QUABO_MEASUREMENTS = [
    "QUABO_1000", "QUABO_1001", "QUABO_1002", "QUABO_1003",
    "QUABO_1008", "QUABO_1009", "QUABO_1010", "QUABO_1011",
    "QUABO_1012", "QUABO_1013", "QUABO_1014", "QUABO_1015",
    "QUABO_1016", "QUABO_1017", "QUABO_1018", "QUABO_1019",
]

FIELDS = [
    "HVIMON0", "HVIMON1", "HVIMON2", "HVIMON3",
    "HVMON0", "HVMON1", "HVMON2", "HVMON3",
    "TEMP1", "TEMP2",
]


def run_cmd(cmd: str) -> None:
    """Run a shell command and show if error occurs."""
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"[!] Command failed: {cmd}")


def ns_to_iso_utc(ns: int) -> str:
    dt = datetime.fromtimestamp(ns / 1e9, tz=UTC)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def rfc3339_to_ns(ts: str) -> int:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1e9)


def parse_time_to_ns(t) -> int:
    """Accept int/float epoch (ns) or RFC3339 string; return epoch ns."""
    if isinstance(t, int):
        return t
    if isinstance(t, float):
        return int(t * 1e9) if t < 1e12 else int(t)
    if isinstance(t, str):
        s = t.strip()
        if s.isdigit():
            return int(s)
        return rfc3339_to_ns(s)
    raise ValueError(f"Unsupported time type: {type(t)} ({t!r})")


def latest_point(client: InfluxDBClient, measurement: str):
    field_list = ", ".join([f'"{f}"' for f in FIELDS])
    q = f'SELECT {field_list} FROM "{measurement}" ORDER BY time DESC LIMIT 1'
    res = client.query(q)
    pts = list(res.get_points())
    return pts[0] if pts else None


def upload_json_to_cylon(local_json_path: str) -> None:
    """
    Upload JSON to cylon using the same approach as your allsky script:
    - copy to /tmp first
    - scp with bandwidth limit
    - chmod via ssh
    """
    tmp = "/tmp/quabo_hvtemps_latest.json"

    # Copy file locally first
    run_cmd(f"cp '{local_json_path}' '{tmp}'")

    # Upload to cylon
    run_cmd(
        f"scp -l {BANDWIDTH_LIMIT} '{tmp}' "
        f"{REMOTE_SERVER}:{REMOTE_DIR}/quabo_hvtemps_latest.json"
    )

    # Fix permissions
    run_cmd(
        f'ssh {REMOTE_SERVER} "chmod 644 {REMOTE_DIR2}/quabo_hvtemps_latest.json || true"'
    )

    print("? Updated quabo_hvtemps_latest.json on cylon")


def main():
    client = InfluxDBClient(
        host=INFLUX_HOST,
        port=INFLUX_PORT,
        username=INFLUX_USER,
        password=INFLUX_PASS,
        database=INFLUX_DB,
        timeout=10,
    )

    print(f"Writing + uploading QUABO HV/TEMP JSON every {INTERVAL} seconds.")
    print("Press Ctrl-C to stop.")

    while True:
        now_ns = time.time_ns()
        max_age_ns = int(MAX_AGE_MIN * 60 * 1e9)

        summary_lists: dict[str, list[str]] = {"fresh": [], "stale": [], "missing": [], "errors": []}

        out: dict[str, Any] = {
            "generated_time_ns": now_ns,
            "generated_utc": ns_to_iso_utc(now_ns),
            "max_age_minutes": MAX_AGE_MIN,
            "database": INFLUX_DB,
            "host": f"{INFLUX_HOST}:{INFLUX_PORT}",
            "summary": {"counts": {}, "lists": summary_lists},
            "quabos": {},
        }

        # Track the newest (smallest age) timestamp we can parse across all QUABOs
        parsed_ages_seconds = []

        for meas in QUABO_MEASUREMENTS:
            try:
                p = latest_point(client, meas)
            except Exception as e:
                out["quabos"][meas] = {"ok": False, "reason": f"query_error: {e}"}
                summary_lists["errors"].append(meas)
                continue

            if not p:
                out["quabos"][meas] = {"ok": False, "reason": "no_points"}
                summary_lists["missing"].append(meas)
                continue

            raw_time = p.get("time")
            try:
                t_ns = parse_time_to_ns(raw_time)
            except Exception as e:
                out["quabos"][meas] = {
                    "ok": False,
                    "reason": f"time_parse_error: {e}",
                    "raw_time": raw_time,
                }
                summary_lists["errors"].append(meas)
                continue

            age_ns = now_ns - t_ns
            if age_ns < 0:
                out["quabos"][meas] = {
                    "ok": False,
                    "reason": "time_in_future",
                    "time_utc": ns_to_iso_utc(t_ns),
                }
                summary_lists["errors"].append(meas)
                continue

            age_seconds = round(age_ns / 1e9, 3)
            parsed_ages_seconds.append(age_seconds)

            if age_ns > max_age_ns:
                out["quabos"][meas] = {
                    "ok": False,
                    "reason": "stale",
                    "time_utc": ns_to_iso_utc(t_ns),
                    "age_seconds": age_seconds,
                }
                summary_lists["stale"].append(meas)
                continue

            out["quabos"][meas] = {
                "ok": True,
                "time_utc": ns_to_iso_utc(t_ns),
                "age_seconds": age_seconds,
                "values": {f: p.get(f) for f in FIELDS},
            }
            summary_lists["fresh"].append(meas)

        out["summary"]["counts"] = {k: len(v) for k, v in summary_lists.items()}
        out["summary"]["counts"]["total"] = len(QUABO_MEASUREMENTS)

        # Write JSON locally
        Path(OUT_JSON).write_text(json.dumps(out, indent=2) + "\n")

        # NEW: gate upload based on freshest parsed Influx timestamp
        if not parsed_ages_seconds:
            print("[!] No parsable Influx timestamps found; skipping upload to cylon.")
        else:
            min_age = min(parsed_ages_seconds)
            if min_age > MAX_UPLOAD_AGE_SECONDS:
                print(
                    f"[!] Freshest Influx point is {min_age:.3f}s old "
                    f"(> {MAX_UPLOAD_AGE_SECONDS}s); skipping upload to cylon."
                )
            else:
                # Upload to cylon
                try:
                    upload_json_to_cylon(OUT_JSON)
                except Exception as e:
                    print(f"[!] Failed to upload JSON to cylon: {e}")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()



