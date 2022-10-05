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
#   --verbose         print details
#   --no_collect      don't copy files to head node
#   --no_cleanup      don't delete files from DAQ nodes

import os, sys
import collect, quabo_driver
from util import *
sys.path.insert(0, '../util')
import pff, config_file

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
def stop_recording(daq_config):
    for node in daq_config['daq_nodes']:
        cmd = 'ssh %s@%s "cd %s; ./stop_daq.py"'%(
            node['username'], node['ip_addr'], node['data_dir']
        )
        print(cmd)
        ret = os.system(cmd)
        if ret: raise Exception('%s returned %d'%(cmd, ret))

# write a "run complete" file in the current run dir,
# and make symlinks to the first nonempty image and ph files in that dir
#
def write_run_complete_file(daq_config, run_name):
    data_dir = daq_config['head_node_data_dir']
    path = '%s/%s/%s'%(data_dir, run_name, run_complete_file)
    with open(path, 'w') as f:
        f.write(now_str())

    if os.path.exists(img_symlink):
        os.unlink(img_symlink)
    if os.path.exists(ph_symlink):
        os.unlink(ph_symlink)
    did_img = False
    did_ph = False
    did_hk = False
    for f in os.listdir('%s/%s'%(data_dir, run_name)):
        path = '%s/%s/%s'%(data_dir, run_name, f)
        if not pff.is_pff_file(path): continue
        if os.path.getsize(path) == 0: continue
        if not did_img and pff.pff_file_type(path) in ['img16', 'img8']:
            os.symlink(path, img_symlink)
            did_img = True
            print('linked %s to %s'%(img_symlink, f))
        elif not did_ph and pff.pff_file_type(path)=='ph16':
            os.symlink(path, ph_symlink)
            did_ph = True
            print('linked %s to %s'%(ph_symlink, f))
        elif not did_hk and pff.pff_file_type(path)=='hk':
            os.symlink(path, hk_symlink)
            did_hk = True
            print('linked %s to %s'%(hk_symlink, f))
        if did_img and did_ph and did_hk: break
    if not did_img:
        print('No nonempty image file')
    if not did_ph:
        print('No nonempty PH file')
    if not did_hk:
        print('No nonempty housekeeping file')

def stop_run(
    daq_config, quabo_uids, verbose=False, no_cleanup=False, no_collect=False
):
    if local_ip() != daq_config['head_node_ip_addr']:
        raise Exception(
            'This computer (%s) is not the head node specified in daq_config.json (%s)'%(
                local_ip(), daq_config['head_node_ip_addr']
            )
        )

    print("stopping data recording")
    stop_recording(daq_config)

    print("stopping HV updater")
    kill_hv_updater()

    print("stopping HK recording")
    kill_hk_recorder()

    print("stopping data generation")
    stop_data_flow(quabo_uids)

    run_dir = read_run_name()
    if run_dir:
        success = True
        if not no_collect:
            print("collecting data from DAQ nodes")
            success = collect.collect_data(daq_config, run_dir, verbose)
        if not no_cleanup:
            print("cleaning up DAQ nodes")
            collect.cleanup_daq(daq_config, run_dir, verbose)
        if success:
            write_run_complete_file(daq_config, run_dir)
            print('completed run %s'%run_dir)
        else:
            print('Failed to collect data from DAQ nodes')
        remove_run_name()
    else:
        print("No run is in progress")

if __name__ == "__main__":
    i = 1;
    argv = sys.argv
    verbose = False
    no_cleanup = False
    no_collect = False
    while i < len(argv):
        if argv[i] == '--verbose':
            verbose = True
        elif argv[i] == '--no_cleanup':
            no_cleanup = True
        elif argv[i] == '--no_collect':
            no_collect = True
        else:
            raise Exception('bad arg %s'%argv[i])
        i += 1
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    config_file.associate(daq_config, quabo_uids)
    stop_run(daq_config, quabo_uids, verbose, no_cleanup, no_collect)
