#! /usr/bin/env python3
"""
One-time (re-run whenever daq_config.json's node list changes) setup step:
copies this fallback's DAQ-side scripts (start_daq.py, stop_daq.py,
status_daq.py, util.py, pff.py, video_daq.py) onto every configured DAQ
node's data_dir over SCP, exactly like the old software's own
`config.py --init_daq_nodes` did.

Without this, emergency_stop.py's SSH step has nothing to run on the DAQ
node -- these files don't ship as part of the new stack's containers.

Usage (from control/legacy/, with PSETI_CONFIG set -- see README.md):
    python3 stage_daq_nodes.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _config_bridge import resolve_config_dir
from utils import config_file, file_xfer, util


def main() -> None:
    # copy_daq_files() uses paths relative to cwd ('daq_scripts/start_daq.py',
    # ...) -- pin cwd to this file's directory so it works regardless of
    # where the operator invoked this script from.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    config_dir = resolve_config_dir()
    daq_config = config_file.get_daq_config(dir=config_dir)
    network_config = config_file.get_network_config(dir=config_dir)
    util.attach_daq_config(daq_config, network_config)
    print(f"Staging DAQ-stop fallback scripts onto {len(daq_config['daq_nodes'])} node(s)...")
    file_xfer.copy_daq_files(daq_config)
    print("Done. Verify with: ssh <user>@<daq-host> 'ls <data_dir>/stop_daq.py'")


if __name__ == "__main__":
    main()
