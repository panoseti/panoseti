#! /usr/bin/env python3

# stop_daq.py: stop a hashpipe process.
# This is called on a DAQ node via SSH by a daemon on the head node.
# It reads the PID of the hashpipe process from a file.
# It kills that process, then kills the HK recorder.
# Then kill any other hashpipe processes
#
# On success, print OK.  Otherwise print an error message

import contextlib
import os

from control.utils import util


def main() -> None:
    try:
        with open(util.daq_hashpipe_pid_filename) as f:
            pid = int(f.read())
        if not util.stop_hashpipe(pid):
            print("Couldn't stop hashpipe")
        os.unlink(util.daq_hashpipe_pid_filename)
    except Exception:
        pass

    util.kill_hashpipe()

    # if the HK recorder is running on a remote DAQ, we didn't start it.
    # But it shouldn't be there, so kill it
    util.kill_hk_recorder()

    with contextlib.suppress(Exception):
        os.unlink(util.daq_run_name_filename)

    print("stop_daq.py: OK")


if __name__ == "__main__":
    main()
