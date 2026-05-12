#!/usr/bin/env python3
"""
fake_hashpipe.py — hashpipe stub for software-only CI.

Accepts the same CLI arguments as the real hashpipe binary:
    hashpipe -p <plugin.so> -I <instance> -o KEY=VALUE... <threads...>

Behaviour:
  1. Extracts RUNDIR from -o RUNDIR=... option
  2. Reads module IDs from module.config in cwd (written by daq_control server)
  3. Creates stub .pff data files in module_{id}/{run_dir}/ for rsync tests
  4. Stays alive until SIGINT or SIGTERM

For real hardware lab CI: mount the actual hashpipe binary at /usr/local/bin/hashpipe
via a Docker volume — no container changes needed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import signal
import sys
import time
from typing import Any


def _read_module_ids(config_path: pathlib.Path) -> list[int]:
    """Read module IDs from module.config written by the daq_control server."""
    if not config_path.exists():
        return []
    try:
        return [int(x) for x in config_path.read_text().split() if x.strip()]
    except (ValueError, OSError):
        return []


def _create_stub_data(cwd: pathlib.Path, run_dir: str, module_ids: list[int], args_data: dict[str, Any]) -> None:
    """Create stub .pff files and record arguments for verification."""
    for mid in module_ids:
        data_dir = cwd / f"module_{mid}" / run_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        stub = data_dir / f"data.module_{mid}.pff"
        stub.write_bytes(b"PFFSTUB\n" * 16)  # minimal content for rsync tests

        # Record arguments for verification
        args_file = data_dir / "fake_hashpipe_args.json"
        args_file.write_text(json.dumps(args_data, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-p", dest="plugin")
    p.add_argument("-I", dest="instance")
    p.add_argument("-o", action="append", default=[], dest="options")
    p.add_argument("threads", nargs="*")
    args, _ = p.parse_known_args()

    options: dict[str, str] = {}
    for o in args.options:
        if "=" in o:
            k, v = o.split("=", 1)
            options[k] = v

    cwd = pathlib.Path.cwd()
    run_dir = options.get("RUNDIR", "")
    config_fn = options.get("CONFIG", "module.config")
    
    # Handle both absolute and relative paths for CONFIG
    config_path = pathlib.Path(config_fn)
    if not config_path.is_absolute():
        config_path = cwd / config_path

    module_ids = _read_module_ids(config_path)

    args_data = {
        "plugin": args.plugin,
        "instance": args.instance,
        "options": options,
        "threads": args.threads,
    }

    if run_dir and module_ids:
        _create_stub_data(cwd, run_dir, module_ids, args_data)

    # Stay alive until SIGINT / SIGTERM
    running = True

    def _stop(signum: int, frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while running:
        time.sleep(0.5)

    sys.exit(0)


if __name__ == "__main__":
    main()
