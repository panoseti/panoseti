#! /usr/bin/env python3

# start_daq.py --run_dir dirname --max_file_size N --module_id M1 ... --module_id Mn
#
# This script is run (remotely) on a DAQ node to start recording.
# The run directory already exists.
# - create a shell script to run hashpipe (see below)
# - create a module.config telling hashpipe what modules to listen for
# - run/detach the shell script to start hashpipe
# - use pgrep to find the PID of the hashpipe process
# - record the PID in a file, so that stop_daq.py can kill it later.
#
# This is arcane, but I can't find a simpler way.
# If we run hashpipe directly from this script, it dies when the script exits.
# If we run hashpipe via a shell script and kill the shell process,
# hashpipe keeps running.

import sys, os, subprocess, time

pid_filename = 'daq_pid'
dirname_filename = 'daq_dirname'
    # stores name of current run

def main():
    argv = sys.argv
    i = 1
    max_file_size = None
    run_dir = None
    module_ids = []
    while i<len(argv):
        if argv[i] == '--run_dir':
            i += 1
            run_dir = argv[i]
        elif argv[i] == '--max_file_size':
            i += 1
            max_file_size = int(argv[i])
        elif argv[i] == '--module_id':
            i += 1
            module_ids.append(int(argv[i]))
        i += 1
    if not run_dir:
        raise Exception('no run dir specified')
    if not max_file_size:
        raise Exception('no max file size specified')
    if len(module_ids) == 0:
        raise Exception('no module IDs specified')
    if not os.path.isdir(run_dir):
        raise Exception("run dir doesn't exist")
    if os.path.exists(pid_filename):
        raise Exception("PID file exists; run stop_daq.py")

    # record the run name in a file

    f = open(dirname_filename, 'w')
    f.write(run_dir)
    f.close()

    # create module.config

    f = open('module.config', 'w')
    for id in module_ids:
        f.write('%d\n'%id)
    f.close()

    # create the run script

    f = open('run_hashpipe.sh', 'w')
    f.write('hashpipe -p ./HSD_hashpipe.so -I 0 -o BINDHOST="0.0.0.0" -o MAXFILESIZE=%d -o SAVELOC="%s" -o CONFIG="./module.config" -o OBS="LICK" HSD_net_thread HSD_compute_thread  HSD_output_thread'%(max_file_size, run_dir))
    f.close()

    # run the script

    process = subprocess.Popen(
        ['bash', 'run_hashpipe.sh'], start_new_session=True,
        close_fds=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # use pgrep to find the PID of the hashpipe process

    pid = process.pid
    while True:
        result = subprocess.run(['pgrep', '-P', str(pid)], stdout=subprocess.PIPE)
        if result != '': break
        time.sleep(1)
    child_pid = int(result.stdout)

    # write it to a file

    f = open(pid_filename, 'w')
    f.write(str(child_pid))
    f.close()

main()
