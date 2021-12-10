#! /usr/bin/env python

# stop a recording run
#
# - tell DAQs to stop recording
# - stop HK recorder process
# - tell quabos to stop sending data
# - collect data files
# - remove .run_name and .hk_pid files

import os, sys
import util, config_file

def stop_data_flow(quabo_uids):
    pass

def stop_recording(daq_config):
    for node in daq_config['daq_nodes']:
        if len(node['modules']) > 0:
            cmd = 'ssh %s@%s "cd %s; start_hashpipe.py --stop"'%(
                node['username'], node['ip_addr'], node['dir']
            )
            print(cmd)
            #os.system(cmd)

def stop_run(daq_config, quabo_uids):
    stop_recording(daq_config)
    util.stop_hk_recorder()
    stop_data_flow(quabo_uids)
    util.remove_run_name()
    

if __name__ == "__main__":
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    util.associate(daq_config, quabo_uids)
    stop_recording(daq_config)
    stop_run(daq_config, quabo_uids)
