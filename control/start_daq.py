#! /usr/bin/env python3

# start_daq.py
#   --run_dir dirname
#   --max_file_size_mb N
#   --daq_ip_addr a.b.c.d
#   --module_id M1 ...
#   --module_id Mn
#   [--bindhost x]
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
import util

def main():
    argv = sys.argv
    i = 1
    max_file_size_mb = -1
    group_frames = 0
    run_dir = None
    daq_ip_addr = None
    module_ids = []
    bindhost = "0.0.0.0"
    while i<len(argv):
        if argv[i] == '--run_dir':
            i += 1
            run_dir = argv[i]
        elif argv[i] == '--daq_ip_addr':
            i += 1
            daq_ip_addr = argv[i]
        elif argv[i] == '--max_file_size_mb':
            i += 1
            max_file_size_mb = int(argv[i])
        elif argv[i] == '--group_frames':
            i += 1
            group_frames = int(argv[i])
        elif argv[i] == '--module_id':
            i += 1
            module_ids.append(int(argv[i]))
        elif argv[i] == '--bindhost':
            i += 1
            bindhost = argv[i]
        i += 1
    if not run_dir:
        raise Exception('no run dir specified')
    if max_file_size_mb<0:
        raise Exception('no max file size specified')
    if len(module_ids) == 0:
        raise Exception('no module IDs specified')
    if not os.path.isdir(run_dir):
        raise Exception("run dir doesn't exist")
    if os.path.exists(util.daq_hashpipe_pid_filename):
        raise Exception("PID file exists; run stop_daq.py")

    if util.is_hashpipe_running():
        raise Exception("Hashpipe is already running")
             
    # record the run name in a file

    f = open(util.daq_run_name_filename, 'w')
    f.write(run_dir)
    f.close()

    # create module.config

    f = open('module.config', 'w')
    for id in module_ids:
        f.write('%d\n'%id)
    f.close()

    # create the run script

    f = open('run_hashpipe.sh', 'w')
    f.write('hashpipe -p ./hashpipe.so -I 0 -o BINDHOST="%s" -o MAXFILESIZE=%d -o GROUPFRAMES=%d -o RUNDIR="%s" -o CONFIG="./module.config" -o OBS="LICK" net_thread compute_thread  output_thread > %s/%s%s'%(bindhost, max_file_size_mb, group_frames, run_dir, run_dir, util.hp_stdout_prefix, daq_ip_addr))
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
        if result.stdout != '': break
        time.sleep(1)
    try:
        child_pid = int(result.stdout)
    except:
        raise Exception("can't get hashpipe PID; it may have crashed: %s"%result.stdout);

    # write it to a file

    f = open(util.daq_hashpipe_pid_filename, 'w')
    f.write(str(child_pid))
    f.close()

main()
