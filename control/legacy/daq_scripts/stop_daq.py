#! /usr/bin/env python3

# This script is run (remotely) on a DAQ node to stop recording
# - get the PID of the hashpipe process
# - send it a SIGINT signal
# - wait for it to exit
#
# Then kill any other hashpipe processes
#
# On success, print OK.  Otherwise print an error message

import os, signal, util

def main():
    try:
        f = open(util.daq_hashpipe_pid_filename, 'r')
    except:
        f = None
    if f:
        pid = int(f.read())
        f.close()
        if not util.stop_hashpipe(pid):
            print("Couldn't stop hashpipe")
        os.unlink(util.daq_hashpipe_pid_filename)

    util.kill_hashpipe()

    # if the HK recorder is running on a remote DAQ, we didn't start it.
    # But it shouldn't be there, so kill it
    util.kill_hk_recorder()

    try:
        os.unlink(util.daq_run_name_filename)
    except:
        pass

    print('stop_daq.py: OK')

main()
