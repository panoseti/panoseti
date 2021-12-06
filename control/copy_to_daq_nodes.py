#! /usr/bin/env python

# copy files to daq nodes
#
# --config      copy data products config file
# --hashpipe    copy hashpipe executable (HSD_hashpipe.so)

import config_file, sys

def usage():
    print('''options:
--config: copy config files: obs_config.json, data_config.json
--hashpipe: copy hashpipe .so file
''')
    sys.exit()

def copy(file, node):
    cmd = 'scp %s %s@%s:%s'%(file, node['username'], node['ip_addr'], node['dir'])
    print(cmd)

def main():
    do_config = False
    do_hashpipe = False
    argv = sys.argv
    i = 1
    while i < len(argv):
        if argv[i] == '--config':
            do_config = True
        elif argv[i] == '--hashpipe':
            do_hashpipe = True
        else:
            usage()
        i += 1

    if not do_config and not do_hashpipe:
        usage()

    c = config_file.get_daq_config()
    for node in c['daq_nodes']:
        print(node['ip_addr'])
        if do_config:
            copy('data_config.json', node)
            copy('obs_config.json', node)
        if do_hashpipe:
            copy('HSD_hashpipe.so', node)

main()
