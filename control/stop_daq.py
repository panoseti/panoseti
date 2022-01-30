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

pid_filename = 'daq_pid'

def main():
    try:
        f = open(pid_filename, 'r')
    except:
        f = None
    if f:
        pid = int(f.read())
        f.close()
        if not util.stop_hashpipe(pid):
            print("Couldn't top hashpipe")
        os.unlink(pid_filename)

    util.kill_hashpipe()
    util.kill_hk_recorder()

    print('OK')

main()
