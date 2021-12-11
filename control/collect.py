#! /usr/bin/env python

# collect files at the end of a recording run

import config_file, file_xfer, util

def collect(daq_config):
    run_name = util.read_run_name()
    if not run_name:
        print("No run found")
        sys.exit()

    for node in daq_config['daq_nodes']:
        if len(node['modules']) > 0:
            file_xfer.copy_dir_from_node(util.get_data_dir(), run_name, node)

if __name__ == "__main__":
    daq_config = config_file.get_daq_config();
    quabo_uids = config_file.get_quabo_uids();
    util.associate(daq_config, quabo_uids)
    collect(daq_config)
