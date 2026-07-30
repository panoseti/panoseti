#! /usr/bin/env python3
"""
Fallback DAQ stop: use when the new gRPC-based daq_control path
(`pseti stop`) is unavailable and an active recording needs to be torn
down. Reads the *same* live config the new stack uses (via PSETI_CONFIG
-- see _config_bridge.py) and performs the two actions verified against
real hardware:

  1. Stop hashpipe on every configured DAQ node, over SSH, using the
     old software's daq_scripts/stop_daq.py (must be staged first --
     see stage_daq_nodes.py). No gRPC involved.
  2. Stop Quabo data flow directly over UDP (control/legacy/driver/
     quabo_driver.py). No gRPC involved.

This intentionally does NOT power off hardware or touch HV -- it returns
the observatory to a session-start-like state (DAQ stopped, quabos idle)
so an operator can then bring up an observing session on either stack.

Usage (from anywhere, with PSETI_CONFIG set -- see README.md):
    python3 control/legacy/emergency_stop.py
    python3 control/legacy/emergency_stop.py --quabo-uids /path/to/quabo_uids.json
    python3 control/legacy/emergency_stop.py --skip-quabos   # DAQ/hashpipe only
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "driver"))

from _config_bridge import resolve_config_dir, resolve_quabo_uids_path
from utils import config_file, util
import stop as old_stop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quabo-uids", help="Explicit path to quabo_uids.json (auto-detected otherwise)")
    parser.add_argument("--skip-quabos", action="store_true", help="Only stop DAQ/hashpipe, don't touch quabos")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    logger = logging.getLogger("emergency_stop")

    # quabo_driver.py's create_logger() writes to a 'logs/...' path relative
    # to cwd -- pin cwd here so this works regardless of where the operator
    # invoked this script from.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("logs", exist_ok=True)

    config_dir = resolve_config_dir()
    daq_config = config_file.get_daq_config(dir=config_dir)
    network_config = config_file.get_network_config(dir=config_dir)
    util.attach_daq_config(daq_config, network_config)

    logger.info("Stopping DAQ (hashpipe) on %d node(s) via SSH...", len(daq_config["daq_nodes"]))
    try:
        old_stop.stop_recording(daq_config, run_dir="", verbose=args.verbose)
    except Exception as e:
        logger.error("stop_recording reported a failure: %s", e)
        logger.error("Check SSH connectivity and that stage_daq_nodes.py has been run for this node.")
        sys.exit(1)
    logger.info("DAQ stop complete.")

    if args.skip_quabos:
        return

    quabo_uids_path = args.quabo_uids or resolve_quabo_uids_path()
    config_file.quabo_uids_filename = quabo_uids_path
    quabo_uids = config_file.get_quabo_uids()
    config_file.associate(daq_config, quabo_uids)

    logger.info("Stopping Quabo data flow over UDP...")
    old_stop.stop_data_flow(quabo_uids, network_config)
    logger.info("Quabo data flow stopped. Observatory returned to session-start-like state.")


if __name__ == "__main__":
    main()
