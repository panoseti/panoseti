#! /usr/bin/env python

# collect files at the end of a recording run

import config_file, copy_to_daq_nodes

def main():
    run_name = run.get_name()
    if not run_name:
        print("No run found")
        sys.exit()

    copy_to_daq_nodes(copy_dir_from_node)

main()
