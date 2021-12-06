#! /usr/bin/env python

# start_daq.py dirname
# This script is run (remotely) on a DAQ node to start recording
# - create the directory
# - start the Hashpipe process
# - record the PID and dirname in a file
#
# On success, print OK.  Otherwise print an error message

import sys, os, subprocess

pid_filename = 'daq_pid'
dirname_filename = 'daq_dirname'
    # stores name of current run
daq_log_filename = 'daq_log'

def main():
    if len(sys.argv) != 2:
        print("usage: start_daq.py dirname")
        return
    dirname = sys.argv[1]

    if os.path.exists(pid_filename):
        print("PID file exists; run stop_daq.py")
        #return;

    # make the run directory

    try:
        os.mkdir(dirname)
    except:
        print("can't create run directory")
        #return

    # record its name in a file

    try:
        f = open(dirname_filename, 'w')
    except Exception as e:
        print(e)
        print("can't create run name file")
        return
    f.write(dirname)
    f.close()

    # run the DAQ program

    try:
        process = subprocess.Popen(['daq', dirname, daq_log_filename])
    except:
        print("can't launch daq process")
        return

    # write its PID to a file

    try:
        f = open(pid_filename, 'w')
    except:
        print("can't create PID file")
        return
    f.write(str(process.pid))
    f.close()

    print('OK')

main()
