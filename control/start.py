#! /usr/bin/env python3

# start.py [--no_hv] [--no_redis] [--no_data] [--verbose] [--help]
#          [--nsecs N] [--stop_session]
#
# start a recording run:
#
# - figure out association of quabos and DAQ nodes,
#   based on config files
# - create "run directories" on head node, DAQ nodes
# - start the HK recorder
# - start the HV updater
# - start the temperature monitor
# - start the flow of data: set DAQ mode and dest IP addr of quabos
# - send commands to DAQ nodes to start hashpipe program
#
# fail if a recording run is in progress,
# or if recording activities are active
#

# based on matlab/startmodules.m, startqNph.m, changepeq.m

import os, sys, traceback, shutil, time
from glob import glob
import util, file_xfer, quabo_driver, stop, session_stop
from sw_info import get_sw_info
import socket

sys.path.insert(0, '../util')

import pff, config_file

verbose = False

def help():
    print("--no_hv: don't run hv_updater.py")
    print("--no_redis: OK if redis daemons not running")
    print("--no_data: set up to record, but don't start data flow or record")
    print("--nsecs N: record for N seconds, then stop run")
    print("--stop_session: stop session at end of run (with --nsecs)")
    print("--verbose: print commands")

# check that PH calibration file is present, nonempty, and at most 24 hours old
#
def ph_baseline_file_ok():
    if not os.path.exists(config_file.quabo_ph_baseline_filename):
        print('quabo_ph_baseline.json not found.  Run config.py --calibrate_ph')
        return False
    if os.path.getmtime(config_file.quabo_ph_baseline_filename) < time.time() - 24*86400:
        print('quabo_ph_baseline.json is too old.  Run config.py --calibrate_ph')
        return False
    return True


# check validity of image params (rate, bpp)
#
def check_img_params(image_8bit, image_usec):
    if image_8bit:
        if image_usec < 20 or image_usec > 25:
            raise Exception('integration time must be 20-25 usec in 8 bit mode')
    else:
        if image_usec < 40:
            raise Exception('integration time must be >= 40 usec in 16 bit mode')

# parse the data config file to get DAQ params for quabos
#
def get_daq_params(data_config):
    do_image = False
    image_usec = 1
    image_8bit = False
    do_ph = False
    bl_subtract = True
    do_any_trigger = False
    group_ph_frames = False
    if 'image' in data_config:
        do_image = True
        image = data_config['image']
        if 'integration_time_usec' not in image:
            raise Exception('missing integration_time_usec in data_config.json')
        if image['quabo_sample_size'] == 8:
            image_8bit = True
        elif image['quabo_sample_size'] != 16:
            raise Exception('quabo_sample_size must be 8 or 16')
        image_usec = image['integration_time_usec']
        check_img_params(image_8bit, image_usec)
    if 'pulse_height' in data_config:
        do_ph = True
        if 'any_trigger' in data_config['pulse_height']:
            do_any_trigger = True
            any_trigger = data_config['pulse_height']['any_trigger']
            if 'group_ph_frames' not in any_trigger:
                raise Exception('missing "group_ph_frames" param for "any_trigger" in data_config.json')
            if any_trigger['group_ph_frames'] == 1:
                group_ph_frames = True
            elif any_trigger['group_ph_frames'] != 0:
                raise Exception('group_ph_frames for any_trigger in data_config.json must be 0 or 1.')
    daq_params = quabo_driver.DAQ_PARAMS(
        do_image, image_usec - 1, image_8bit, do_ph, bl_subtract, do_any_trigger, group_ph_frames
    )
    if 'flash_params' in data_config:
        fp = data_config['flash_params']
        daq_params.set_flash_params(fp['rate'], fp['level'], fp['width'])
    if 'stim_params' in data_config:
        sp = data_config['stim_params']
        daq_params.set_stim_params(sp['rate'], sp['level'])
    return daq_params

# Start data flow from the quabos.
# For each quabo:
# - tell it where to send HK packets
# - tell it where to send data packets
# - set its DAQ mode
#
def start_data_flow(quabo_uids, data_config, daq_config):
    daq_params = get_daq_params(data_config)        
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            if 'daq_node' not in module:
                continue
            base_ip_addr = module['ip_addr']
            module_id = config_file.ip_addr_to_module_id(base_ip_addr)
            daq_node = config_file.module_id_to_daq_node(daq_config, module_id)
            daq_node_ip_addr = daq_node['ip_addr']
            head_node_ip_addr = daq_config['head_node_ip_addr']
            for i in range(4):
                quabo = module['quabos'][i]
                if quabo['uid'] == '':
                    continue
                ip_addr = config_file.quabo_ip_addr(base_ip_addr, i)
                quabo = quabo_driver.QUABO(ip_addr)
                if verbose:
                    print('setting HK packet dest to %s on quabo %s'%(
                        head_node_ip_addr, ip_addr
                    ))
                quabo.hk_packet_destination(head_node_ip_addr)
                if verbose:
                    print('setting data packet dest to %s on quabo %s'%(
                        daq_node_ip_addr, ip_addr
                    ))
                quabo.data_packet_destination(daq_node_ip_addr)
                if verbose:
                    print('setting DAQ mode on quabo %s'%ip_addr)
                quabo.send_daq_params(daq_params)
                quabo.close()

# make run directories; copy config files to them
# on each DAQ node:
# data/
#     run/         config files go here
#     module_n
#         run/     .pff files go here
#
def make_run_dirs(run_name, daq_config):
    my_ip = util.local_ip()
    run_dir = '%s/%s'%(daq_config['head_node_data_dir'], run_name)
    os.mkdir(run_dir)

    # copy config files to run dir on this node
    local_data_dir = daq_config['head_node_data_dir']
    for f in config_file.config_file_names:
        files = glob(f)
        for file in files:
            shutil.copyfile(file, '%s/%s'%(run_dir, file))
    
     
    # make module and run directories on DAQ nodes
    #
    for node in daq_config['daq_nodes']:
        if not node['modules']:
            continue
        ip_addr = node['ip_addr']
        if ip_addr == my_ip:
            for module in node['modules']:
                cmd = 'mkdir -p %s/module_%d/%s'%(
                    daq_config['head_node_data_dir'],
                    module['id'], run_name
                )
                if verbose:
                    print(cmd)
                ret = os.system(cmd)
                if ret: raise Exception('%s returned %d'%(cmd, ret))
        else:
            username = node['username']
            data_dir = node['data_dir']
            rcmds = ['mkdir %s/%s'%(data_dir, run_name)]
            for module in node['modules']:
                rcmds.append('mkdir -p %s/module_%d/%s'%(
                    data_dir, module['id'], run_name
                ))
            # create process snapshot
            rcmds.append('cd %s/%s; ps -ux > pss_%s.log'%(data_dir,run_name, ip_addr))
            rcmd = ';'.join(rcmds)
            cmd = 'ssh %s@%s "%s"'%(username, ip_addr, rcmd)
            if verbose:
                print(cmd)
            ret = os.system(cmd)
            if ret: raise Exception('%s returned %d'%(cmd, ret))

    # copy config files to DAQ nodes
    file_xfer.copy_config_files(daq_config, run_name, verbose)

# start recording data
#   for each DAQ node that is getting data
#       create run directory
#       copy config files to run directory
#       start hashpipe program
#
def start_recording(data_config, daq_config, run_name, no_hv):
    my_ip = util.local_ip()

    # start recording HK data
    util.start_hk_recorder(daq_config, run_name)

    if not no_hv:
        # start high-voltage updater
        util.start_hv_updater()

        # start module temperature monitor
        util.start_module_temp_monitor()

    # start hashpipe on DAQ nodes

    if 'max_file_size_mb' in data_config.keys():
        max_file_size_mb = int(data_config['max_file_size_mb'])
    else:
        max_file_size_mb = util.default_max_file_size_mb
    daq_params = get_daq_params(data_config)
    for node in daq_config['daq_nodes']:
        if not node['modules']:
            continue
        username = node['username']
        data_dir = node['data_dir']
        remote_cmd = './start_daq.py --daq_ip_addr %s --run_dir %s --max_file_size_mb %d --group_ph_frames %d'%(
            node['ip_addr'], run_name, max_file_size_mb, daq_params.do_group_ph_frames
        )
        if 'bindhost' in node.keys():
            remote_cmd += ' --bindhost %s'%node['bindhost']
        for m in node['modules']:
            module_id = config_file.ip_addr_to_module_id(m['ip_addr'])
            remote_cmd += ' --module_id %d'%module_id
        cmd = 'ssh %s@%s "cd %s; %s"'%(
            username, node['ip_addr'], data_dir, remote_cmd
        )
        if verbose:
            print(cmd)
        ret = os.system(cmd)
        if ret: raise Exception('%s returned %d'%(cmd, ret))

def start_run(
    obs_config, daq_config, quabo_uids, data_config, no_hv, no_redis, no_data
):
    my_ip = util.local_ip()
    # convert head node name to IP address
    head_node_ip = socket.gethostbyname(daq_config['head_node_ip_addr'])
    if my_ip != head_node_ip:
        print('This node (%s) is not the head node specified in daq_config.json (%s)'%(my_ip, daq_config['head_node_ip_addr']))
        return False

    rn = util.read_run_name()
    if (rn):
        print('A run is already in progress: %s' %rn)
        print('Run stop.py, then try again.')
        return False

    if util.is_hk_recorder_running():
        print('The HK recorder is running.  Run stop.py, then try again.')
        return False
        
    if not no_redis:
        if not util.are_redis_daemons_running():
            print('Redis daemons are not running.  Run config.py --redis_daemons')
            util.show_redis_daemons()
            return False

    if not ph_baseline_file_ok():
        return False

    # get git commit info, and write the info into sw_info.json
    get_sw_info()
    # if head node is also DAQ node, make sure data dirs are the same
    #
    for node in daq_config['daq_nodes']:
        if my_ip == node['ip_addr'] and daq_config['head_node_data_dir'] != node['data_dir']:
            print("Head node data dir doesn't match DAQ node data dir")
            return False

    try:
        run_name = pff.run_dir_name(obs_config['name'], data_config['run_type'])
        config_file.associate(daq_config, quabo_uids)
        config_file.show_daq_assignments(quabo_uids)
        print('setting up run directories')
        make_run_dirs(run_name, daq_config)
        if not no_data:
            print('starting data flow from quabos')
            start_data_flow(quabo_uids, data_config, daq_config)
            print('starting recording')
            start_recording(data_config, daq_config, run_name, no_hv)
    except:
        print(traceback.format_exc())
        print("Couldn't start run.  Run stop.py, then try again.")
        print('If other users might be using the telescope, check with them;')
        print('running stop.py will kill their run.')
        return False
    util.write_run_name(daq_config, run_name)
    print('started run %s'%run_name)
    return True

if __name__ == "__main__":
    argv = sys.argv
    no_hv = False
    no_redis = False
    no_data = False
    nsecs = 0
    stop_session = False
    i = 1
    while i < len(argv):
        if argv[i] == '--no_hv':
            no_hv = True
        elif argv[i] == '--no_redis':
            no_redis = True
        elif argv[i] == '--no_data':
            no_data = True
        elif argv[i] == '--verbose':
            verbose = True
        elif argv[i] == '--nsecs':
            i += 1
            nsecs = int(argv[i])
        elif argv[i] == '--stop_session':
            stop_session = True
        elif argv[i] == '--help':
            help()
            quit()
        else:
            help()
            raise Exception('bad arg %s'%argv[i])
        i += 1

    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    start_run(
        obs_config, daq_config, quabo_uids, data_config,
        no_hv, no_redis, no_data
    )
    if nsecs:
        time.sleep(nsecs)
        stop.stop_run(daq_config, quabo_uids)
        if stop_session:
            session_stop.session_stop(obs_config)
