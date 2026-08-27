#!/usr/bin/env python3
"""
Gather latest commanded focus states from multiple DAQ nodes.

Each DAQ node is expected to have:

    ~/panoseti/Calibrations/focus_current_state.json

This script periodically SSHes into each DAQ node, reads that JSON file,
merges all telescope entries into one local JSON file, optionally records the
merged focus states into a SQLite history database, and optionally copies the
merged JSON and SQLite database to cylon.

Important:
    These are LAST COMMANDED focus values, not hardware readbacks.

Example:
    python3 permanent_focus_reading.py \
        --config ~/panoseti_mount/panoseti/control/src/control/daemons/capture_focus/focus_state_gather_config.json

Run once:
    python3 permanent_focus_reading.py \
        --config ~/panoseti_mount/panoseti/control/src/control/daemons/capture_focus/focus_state_gather_config.json \
        --once

Relevant optional config fields:
    "history_db_path": "/home/obs/.../capture_focus/focus_history.db",
    "cylon_history_db_destination": "panoseti@132.239.146.24:/volume1/web/panoseti-palomar/logs/focus_history.db"
"""

import argparse
import json
import os
import shlex
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Optional


DEFAULT_CONFIG = os.path.expanduser(
    "~/panoseti_mount/panoseti/control/src/control/daemons/capture_focus/focus_state_gather_config.json"
)


FOCUS_HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS focus_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_utc TEXT NOT NULL,
  telescope TEXT NOT NULL,
  focus_steps REAL,
  command_utc TEXT,
  gathered_utc TEXT,
  daq_node TEXT,
  daq_host TEXT,
  daq_user TEXT,
  ip TEXT,
  valid INTEGER DEFAULT 1,
  raw_json TEXT NOT NULL,
  inserted_utc TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  UNIQUE(snapshot_utc, telescope)
);

CREATE INDEX IF NOT EXISTS idx_focus_history_snapshot
ON focus_history(snapshot_utc);

CREATE INDEX IF NOT EXISTS idx_focus_history_date
ON focus_history(substr(snapshot_utc, 1, 10));

CREATE INDEX IF NOT EXISTS idx_focus_history_telescope
ON focus_history(telescope);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def load_json_file(path: str) -> dict[str, Any]:
    path = expand_path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"JSON file does not contain an object: {path}")

    return data


def atomic_write_json(path: str, data: dict[str, Any]) -> None:
    """
    Write JSON atomically so a webpage never reads a half-written file.
    """
    path = expand_path(path)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=".focus_all_",
        suffix=".tmp",
        dir=directory,
        text=True,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)

    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def run_command(
    cmd: list[str],
    timeout: float,
) -> tuple[int, str, str]:
    """
    Run command and return:
        returncode, stdout, stderr
    """
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr

    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        stderr += f"\nTIMEOUT after {timeout:g} seconds"
        return 124, stdout, stderr


def ssh_read_file(
    user: str,
    host: str,
    remote_path: str,
    ssh_options: list[str],
    timeout: float,
) -> tuple[bool, Optional[dict[str, Any]], str]:
    """
    SSH to node and read remote JSON file.

    Returns:
        success, data, error_message
    """
    remote = f"{user}@{host}"

    # Let the remote shell expand ~.
    # Quote the path safely except for a leading ~/.
    if remote_path.startswith("~/"):
        quoted_remote_path = "~/" + shlex.quote(remote_path[2:])
    else:
        quoted_remote_path = shlex.quote(remote_path)

    remote_cmd = f"cat {quoted_remote_path}"

    cmd = [
        "ssh",
        *ssh_options,
        remote,
        remote_cmd,
    ]

    returncode, stdout, stderr = run_command(cmd, timeout=timeout)

    if returncode != 0:
        msg = stderr.strip() or stdout.strip() or f"ssh exited with status {returncode}"
        return False, None, msg

    if not stdout.strip():
        return False, None, "remote file was empty or produced no output"

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return False, None, f"remote JSON parse error: {exc}"

    if not isinstance(data, dict):
        return False, None, "remote JSON is not an object"

    return True, data, ""


def merge_node_state(
    merged_focus: dict[str, Any],
    node_name: str,
    node_host: str,
    node_user: str,
    remote_data: dict[str, Any],
) -> None:
    """
    Merge entries from one DAQ node into the global focus dictionary.

    Remote file usually looks like:

        {
          "Fern": {
            "focus_steps": 2600,
            "timestamp_utc": "...",
            ...
          }
        }

    The merged file keeps telescope names as top-level keys.
    """
    for key, value in remote_data.items():
        if not isinstance(value, dict):
            merged_focus[key] = {
                "telescope": key,
                "error": "Remote entry was not a JSON object",
                "raw_value": value,
                "daq_node": node_name,
                "daq_host": node_host,
                "daq_user": node_user,
                "gathered_utc": utc_now_iso(),
                "valid": False,
            }
            continue

        entry = dict(value)
        entry["daq_node"] = node_name
        entry["daq_host"] = node_host
        entry["daq_user"] = node_user
        entry["gathered_utc"] = utc_now_iso()
        entry["valid"] = True

        # Make the operational meaning explicit for webpages.
        entry.setdefault(
            "note",
            "Last commanded focus value only; not a hardware readback.",
        )

        merged_focus[key] = entry


def gather_all_nodes(config: dict[str, Any]) -> dict[str, Any]:
    poll_time = utc_now_iso()

    remote_focus_state_path = config.get(
        "remote_focus_state_path",
        "~/panoseti/Calibrations/focus_current_state.json",
    )

    ssh_options = config.get(
        "ssh_options",
        [
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=8",
            "-o", "StrictHostKeyChecking=accept-new",
        ],
    )

    if not isinstance(ssh_options, list):
        raise ValueError("Config field ssh_options must be a list")

    ssh_timeout = float(config.get("ssh_timeout_seconds", 15.0))

    nodes = config.get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Config field nodes must be a non-empty list")

    merged_focus: dict[str, Any] = {}
    node_status: dict[str, Any] = {}

    for node in nodes:
        if not isinstance(node, dict):
            continue

        node_name = str(node.get("name", node.get("host", "unknown")))
        node_host = str(node.get("host", ""))
        node_user = str(node.get("user", os.getenv("USER", "panoseti")))

        if not node_host:
            node_status[node_name] = {
                "ok": False,
                "error": "Missing host in config",
                "gathered_utc": utc_now_iso(),
            }
            continue

        ok, remote_data, error = ssh_read_file(
            user=node_user,
            host=node_host,
            remote_path=remote_focus_state_path,
            ssh_options=ssh_options,
            timeout=ssh_timeout,
        )

        if ok and remote_data is not None:
            merge_node_state(
                merged_focus=merged_focus,
                node_name=node_name,
                node_host=node_host,
                node_user=node_user,
                remote_data=remote_data,
            )

            node_status[node_name] = {
                "ok": True,
                "host": node_host,
                "user": node_user,
                "remote_focus_state_path": remote_focus_state_path,
                "entries": list(remote_data.keys()),
                "gathered_utc": utc_now_iso(),
            }

        else:
            node_status[node_name] = {
                "ok": False,
                "host": node_host,
                "user": node_user,
                "remote_focus_state_path": remote_focus_state_path,
                "error": error,
                "gathered_utc": utc_now_iso(),
            }

    output = {
        "schema": "pano_focus_commanded_state_v1",
        "generated_utc": poll_time,
        "generated_by": socket.gethostname(),
        "meaning": "Last commanded focus values only; these are not hardware readbacks.",
        "focus": merged_focus,
        "nodes": node_status,
    }

    return output


def init_focus_history_db(db_path: str) -> None:
    """
    Create the SQLite focus history table if needed.
    """
    db_path = expand_path(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        # Keep a single-file SQLite database. This makes rsync to cylon simple.
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.executescript(FOCUS_HISTORY_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int_valid(value: Any) -> int:
    return 1 if bool(value) else 0


def write_focus_history(db_path: str, output: dict[str, Any]) -> int:
    """
    Append one snapshot of the merged focus state into SQLite.

    One row is stored per telescope/site per gather cycle.
    The focus.php calibration page can then select a UT date and display the
    latest snapshot available for that date.

    Returns:
        number of telescope rows written
    """
    db_path = expand_path(db_path)
    init_focus_history_db(db_path)

    snapshot_utc = str(output.get("generated_utc") or utc_now_iso())
    focus = output.get("focus", {})

    if not isinstance(focus, dict):
        return 0

    rows_written = 0

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.executescript(FOCUS_HISTORY_SCHEMA)

        for key, entry in focus.items():
            if not isinstance(entry, dict):
                entry = {
                    "telescope": str(key),
                    "raw_value": entry,
                    "valid": False,
                }

            telescope = str(entry.get("telescope") or key)
            focus_steps = _to_float_or_none(entry.get("focus_steps"))
            command_utc = entry.get("timestamp_utc") or entry.get("command_utc")
            gathered_utc = entry.get("gathered_utc")
            daq_node = entry.get("daq_node")
            daq_host = entry.get("daq_host")
            daq_user = entry.get("daq_user")
            ip = entry.get("ip")
            valid = _to_int_valid(entry.get("valid", True))
            raw_json = json.dumps(entry, sort_keys=True)

            conn.execute(
                """
                INSERT OR REPLACE INTO focus_history (
                    snapshot_utc,
                    telescope,
                    focus_steps,
                    command_utc,
                    gathered_utc,
                    daq_node,
                    daq_host,
                    daq_user,
                    ip,
                    valid,
                    raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_utc,
                    telescope,
                    focus_steps,
                    str(command_utc) if command_utc is not None else None,
                    str(gathered_utc) if gathered_utc is not None else None,
                    str(daq_node) if daq_node is not None else None,
                    str(daq_host) if daq_host is not None else None,
                    str(daq_user) if daq_user is not None else None,
                    str(ip) if ip is not None else None,
                    valid,
                    raw_json,
                ),
            )
            rows_written += 1

        conn.commit()
    finally:
        conn.close()

    return rows_written


def copy_to_cylon(local_output_path: str, cylon_destination: str, timeout: float = 30.0) -> tuple[bool, str]:
    """
    Copy a local file to cylon using rsync.

    JSON destination example:
        panoseti@132.239.146.24:/volume1/web/panoseti-palomar/logs/focus_current_state_all.json

    SQLite destination example:
        panoseti@132.239.146.24:/volume1/web/panoseti-palomar/logs/focus_history.db
    """
    if not cylon_destination:
        return True, "No cylon destination configured"

    local_output_path = expand_path(local_output_path)

    if not os.path.isfile(local_output_path):
        return False, f"Local file does not exist: {local_output_path}"

    cmd = [
        "rsync",
        "-av",
        "--chmod=F644",
        local_output_path,
        cylon_destination,
    ]

    returncode, stdout, stderr = run_command(cmd, timeout=timeout)

    if returncode != 0:
        msg = stderr.strip() or stdout.strip() or f"rsync exited with status {returncode}"
        return False, msg

    return True, stdout.strip()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gather latest commanded focus state from all DAQ nodes."
    )

    p.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to focus gather daemon config JSON",
    )

    p.add_argument(
        "--once",
        action="store_true",
        help="Run one gather/copy cycle and exit",
    )

    p.add_argument(
        "--no-cylon-copy",
        action="store_true",
        help="Do not copy the output file to cylon",
    )

    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print status messages",
    )

    return p.parse_args(sys.argv[1:] if argv is None else argv)


def run_once(config: dict[str, Any], no_cylon_copy: bool = False, verbose: bool = False) -> int:
    local_output_path = config.get(
        "local_output_path",
        "/tmp/focus_current_state_all.json",
    )

    cylon_destination = config.get(
        "cylon_destination",
        "",
    )

    history_db_path = str(config.get("history_db_path", "") or "")
    cylon_history_db_destination = str(config.get("cylon_history_db_destination", "") or "")

    cylon_timeout = float(config.get("cylon_timeout_seconds", 30.0))

    output = gather_all_nodes(config)

    atomic_write_json(local_output_path, output)

    if verbose:
        n_focus = len(output.get("focus", {}))
        n_nodes = len(output.get("nodes", {}))
        print(
            f"{utc_now_iso()} wrote {local_output_path} "
            f"with {n_focus} focus entries from {n_nodes} nodes"
        )

    if history_db_path:
        rows_written = write_focus_history(history_db_path, output)
        output["sqlite_history"] = {
            "ok": True,
            "db_path": expand_path(history_db_path),
            "rows_written": rows_written,
            "timestamp_utc": utc_now_iso(),
        }

        # Rewrite local output including SQLite history status.
        atomic_write_json(local_output_path, output)

        if verbose:
            print(
                f"{utc_now_iso()} wrote {rows_written} row(s) "
                f"to SQLite history: {history_db_path}"
            )

    if not no_cylon_copy and cylon_destination:
        ok, msg = copy_to_cylon(
            local_output_path=local_output_path,
            cylon_destination=cylon_destination,
            timeout=cylon_timeout,
        )

        output["cylon_copy"] = {
            "ok": ok,
            "destination": cylon_destination,
            "timestamp_utc": utc_now_iso(),
            "message": msg,
        }

        # Rewrite local output including cylon copy status.
        atomic_write_json(local_output_path, output)

        if verbose:
            if ok:
                print(f"{utc_now_iso()} copied to cylon: {cylon_destination}")
            else:
                print(f"{utc_now_iso()} ERROR copying to cylon: {msg}", file=sys.stderr)

        if not ok:
            return 1

    if not no_cylon_copy and history_db_path and cylon_history_db_destination:
        ok, msg = copy_to_cylon(
            local_output_path=history_db_path,
            cylon_destination=cylon_history_db_destination,
            timeout=cylon_timeout,
        )

        output["cylon_history_db_copy"] = {
            "ok": ok,
            "destination": cylon_history_db_destination,
            "timestamp_utc": utc_now_iso(),
            "message": msg,
        }

        # Rewrite local output including history DB copy status.
        atomic_write_json(local_output_path, output)

        if verbose:
            if ok:
                print(f"{utc_now_iso()} copied SQLite history DB to cylon: {cylon_history_db_destination}")
            else:
                print(f"{utc_now_iso()} ERROR copying SQLite history DB to cylon: {msg}", file=sys.stderr)

        if not ok:
            return 1

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    config_path = expand_path(args.config)

    try:
        config = load_json_file(config_path)
    except Exception as exc:
        print(f"ERROR: Could not read config file: {config_path}", file=sys.stderr)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    poll_seconds = float(config.get("poll_seconds", 120))

    if args.once:
        return run_once(
            config=config,
            no_cylon_copy=args.no_cylon_copy,
            verbose=True if args.verbose else False,
        )

    if args.verbose:
        print(f"{utc_now_iso()} starting focus gather daemon")
        print(f"Config: {config_path}")
        print(f"Poll interval: {poll_seconds:g} seconds")

    while True:
        try:
            rc = run_once(
                config=config,
                no_cylon_copy=args.no_cylon_copy,
                verbose=args.verbose,
            )

            if rc != 0:
                print(
                    f"{utc_now_iso()} WARNING: gather cycle completed with errors",
                    file=sys.stderr,
                )

        except Exception as exc:
            print(f"{utc_now_iso()} ERROR in gather cycle: {exc}", file=sys.stderr)

        time.sleep(poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

