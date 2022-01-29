#! /usr/bin/env python3

# stop_daq.py
# This script is run (remotely) on a DAQ node to stop recording
# - get the PID of the hashpipe process
# - send it a STOP signal
#
# Then kill any other hashpipe processes
#
# On success, print OK.  Otherwise print an error message

import os, signal, psutil

pid_filename = 'daq_pid'

def main():
    try:
        f = open(pid_filename, 'r')
    except:
        f = None
    if f:
        pid = int(f.read())
        f.close()
        try:
            os.kill(pid, signal.SIGKILL)
        except:
            pass
        os.unlink(pid_filename)

    for p in psutil.process_iter():
        if p.name() == "hashpipe":
            os.kill(p.pid(), signal.SIGKILL)

    print('OK')

main()
