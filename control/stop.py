#! /usr/bin/env python3

# stop and finish a recording run if one is in progress.
# stop recording activities whether or not a run is in progress.
#
# - tell DAQs to stop recording
# - stop HK recorder process
# - tell quabos to stop sending data
# - if a run is in progress, collect data files

import os, sys
import util, config_file, collect, quabo_driver

# tell the quabos to stop sending data
#
def stop_data_flow(quabo_uids):
    daq_params = quabo_driver.DAQ_PARAMS(False, 0, False, False)
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            if 'daq_node' not in module:
                continue
            base_ip_addr = module['ip_addr']
            for i in range(4):
                quabo = module['quabos'][i]
                if quabo['uid'] == '':
                    continue
                ip_addr = util.quabo_ip_addr(base_ip_addr, i)
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

def stop_run(daq_config, quabo_uids):
    print("stopping data recording")
    stop_recording(daq_config)

    print("stopping HK recording")
    util.stop_hk_recorder()

    print("stopping data generation")
    stop_data_flow(quabo_uids)

    if util.local_ip() != daq_config['head_node_ip_addr']:
        raise Exception('This is not the head node specific in daq_config.json')
    print("collecting data from DAQ nodes")
    run_dir = util.read_run_name()
    if run_dir:
        collect.collect_data(daq_config, run_dir)
        util.run_complete(daq_config, run_dir)
    else:
        print("No run name found - can't collect data")

if __name__ == "__main__":
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    util.associate(daq_config, quabo_uids)
    stop_run(daq_config, quabo_uids)
