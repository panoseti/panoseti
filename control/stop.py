#! /usr/bin/env python3

# stop and finish a recording run if one is in progress.
# stop recording activities whether or not a run is in progress.
#
# - tell DAQs to stop recording
# - stop HK recorder process
# - tell quabos to stop sending data
# - if a run is in progress, copy data files to head and delete from DAQs
#
# options:
#   --verbose           print details
#   --no_collect        don't copy data files to head node
#   --no_cleanup        don't delete files from DAQ nodes
#   --run X             clean up run X (default: read from current_run)

import os, sys
import collect, quabo_driver
from util import *
sys.path.insert(0, '../util')
import pff, config_file

# write message to error log
#
def log_error(msg, run_dir):
    print(msg)
    log_path = '%s/stop_errors'%run_dir if run_dir else 'stop_errors'
    with open(log_path, 'a') as f:
        f.write('%s: %s\n'%(now_str(), msg))

# tell the quabos to stop sending data
#
def stop_data_flow(quabo_uids):
    daq_params = quabo_driver.DAQ_PARAMS(False, 0, False, False, False)
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            if 'daq_node' not in module:
                continue
            base_ip_addr = module['ip_addr']
            for i in range(4):
                quabo = module['quabos'][i]
                if quabo['uid'] == '':
                    continue
                ip_addr = config_file.quabo_ip_addr(base_ip_addr, i)
                quabo = quabo_driver.QUABO(ip_addr)
                quabo.send_daq_params(daq_params)
                quabo.close()

# tell all DAQ nodes to stop recording
#
def stop_recording(daq_config, run_dir, verbose):
    for node in daq_config['daq_nodes']:
        cmd = 'ssh %s@%s "cd %s; ./stop_daq.py"'%(
            node['username'], node['ip_addr'], node['data_dir']
        )
        if verbose:
            print(cmd)
        ret = os.system(cmd)
        if ret:
            msg = '%s returned %d'%(cmd, ret)
            log_error(msg, run_dir)
            raise Exception(msg)

# write a "complete file" in the run dir
#
def write_complete_file(run_dir, filename):
    path = '%s/%s'%(run_dir, filename)
    with open(path , 'w') as f:
        f.write(now_str())

def complete_file_exists(run_dir, filename):
    path = '%s/%s'%(run_dir, filename)
    return os.path.exists(path)

# make symlinks to the first nonempty image and ph files in that dir
#
def make_links(run_dir, verbose):
    if os.path.lexists(img_symlink):
        os.unlink(img_symlink)
    if os.path.lexists(ph_symlink):
        os.unlink(ph_symlink)
    if os.path.lexists(hk_symlink):
        os.unlink(hk_symlink)
    did_img = False
    did_ph = False
    did_hk = False
    for f in os.listdir(run_dir):
        path = '%s/%s'%(run_dir, f)
        if not pff.is_pff_file(path): continue
        if os.path.getsize(path) == 0: continue
        ftype = pff.pff_file_type(f)
        if not did_img and ftype in ['img16', 'img8']:
            os.symlink(path, img_symlink)
            did_img = True
            if verbose:
                print('linked %s to %s'%(img_symlink, f))
        elif not did_ph and ftype in ['ph256', 'ph1024']:
            os.symlink(path, ph_symlink)
            did_ph = True
            if verbose:
                print('linked %s to %s'%(ph_symlink, f))
        elif not did_hk and ftype == 'hk':
            os.symlink(path, hk_symlink)
            did_hk = True
            if verbose:
                print('linked %s to %s'%(hk_symlink, f))
        if did_img and did_ph and did_hk: break
    if not did_img:
        print('make_links(): No nonempty image file')
    if not did_ph:
        print('make_links(): No nonempty PH file')
    if not did_hk:
        print('make_links(): No nonempty housekeeping file')

def stop_run(
    daq_config, quabo_uids, verbose=False, no_cleanup=False, no_collect=False,
    run = None
):
    # convert head node name to IP address
    head_node_ip = socket.gethostbyname(daq_config['head_node_ip_addr'])
    if local_ip() != head_node_ip:
        raise Exception(
            'This computer (%s) is not the head node specified in daq_config.json (%s)'%(
                local_ip(), daq_config['head_node_ip_addr']
            )
        )

    if not run:
        run = read_run_name()
    data_dir = daq_config['head_node_data_dir']
    run_dir = '%s/%s'%(data_dir, run)
    if not os.path.exists(run_dir):
        run_dir = None

    # do things that don't depend on having a run dir

    print("stopping data recording")
    stop_recording(daq_config, run_dir, verbose)

    print("stopping HV updater")
    kill_hv_updater()

    print("stopping HK recording")
    kill_hk_recorder()

    print("stopping data generation")
    stop_data_flow(quabo_uids)

    if run_dir:
        if not complete_file_exists(run_dir, recording_ended_filename):
            write_complete_file(run_dir, recording_ended_filename)
        collect_error = ''
        if not no_collect and not complete_file_exists(run_dir, collect_complete_filename):
            print("collecting data from DAQ nodes")
            collect_error = collect.collect_data(daq_config, run, verbose)
            if collect_error == '':
                write_complete_file(run_dir, collect_complete_filename)
        if collect_error == '':
            if not no_cleanup:
                if verbose:
                    print("cleaning up DAQ nodes")
                error_msg = collect.cleanup_daq(daq_config, run, verbose)
                if error_msg != '':
                    log_error(error_msg, run_dir)
            make_links(run_dir, verbose)
            write_complete_file(run_dir, run_complete_filename)
            print('completed run %s'%run)
        else:
            log_error(collect_error, run_dir)
        remove_run_name()
    else:
        print("No run is in progress")

if __name__ == "__main__":
    i = 1;
    argv = sys.argv
    verbose = False
    no_cleanup = False
    no_collect = False
    run = None
    while i < len(argv):
        if argv[i] == '--verbose':
            verbose = True
        elif argv[i] == '--no_cleanup':
            no_cleanup = True
        elif argv[i] == '--no_collect':
            no_collect = True
        elif argv[i] == '--run':
            i += 1
            run = argv[i]
        else:
            raise Exception('bad arg %s'%argv[i])
        i += 1
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    config_file.associate(daq_config, quabo_uids)
    stop_run(daq_config, quabo_uids, verbose, no_cleanup, no_collect, run)
