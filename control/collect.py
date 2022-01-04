#! /usr/bin/env python3

# collect files at the end of a recording run
#
# --run_dir X    specify run dir

import sys
import config_file, file_xfer, util

def collect_data(daq_config, run_dir):
    for node in daq_config['daq_nodes']:
        if len(node['modules']) > 0:
            file_xfer.copy_dir_from_node(run_dir, daq_config, node)

if __name__ == "__main__":
    i = 1
    run_dir = ''
    while i<len(sys.argv):
        if sys.argv[i] == '--run_dir':
            i += 1
            run_dir = sys.argv[i]
        i += 1
    if not run_dir:
        run_dir = util.read_run_name()
        if not run_dir:
            print("No run found")
            sys.exit()
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    util.associate(daq_config, quabo_uids)
    collect_data(daq_config, run_dir)
