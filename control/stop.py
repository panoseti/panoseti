#! /usr/bin/env python

# stop a recording run
#
# - tell DAQs to stop recording
# - stop HK recorder process
# - tell quabos to stop sending data
# - collect data files

import os, sys
import util, config_file, collect

# tell the quabos to stop sending data
#
def stop_data_flow(quabo_uids):
    pass

# tell the DAQ nodes to stop recording
#
def stop_recording(daq_config):
    for node in daq_config['daq_nodes']:
        if len(node['modules']) > 0:
            cmd = 'ssh %s@%s "cd %s; start_hashpipe.py --stop"'%(
                node['username'], node['ip_addr'], node['dir']
            )
            print(cmd)
            #os.system(cmd)

def stop_run(daq_config, quabo_uids):
    print("stopping data recording")
    stop_recording(daq_config)

    print("stopping HK recording")
    util.stop_hk_recorder()

    print("stopping data generation")
    stop_data_flow(quabo_uids)

    print("collecting data from DAQ nodes")
    run_dir = util.read_run_name()
    if run_dir:
        collect.collect_data(daq_config, run_dir)
    else:
        print("No run name found - can't collect data")


if __name__ == "__main__":
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    util.associate(daq_config, quabo_uids)
    stop_run(daq_config, quabo_uids)
