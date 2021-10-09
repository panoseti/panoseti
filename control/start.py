#! /usr/bin/env python

# start quabos
# based on matlab/startmodules.m, startqNph.m, changepeq.m
# options:
# --dome N      select dome
# --module N    select module
# --quabo N     select quabo

import config_file, sys

def do_op(quabos, op):
    for quabo in quabos:
        print(op, quabo['ip_addr'])
        if op == 'start':

if __name__ == "__main__":
    argv = sys.argv
    nops = 0
    nsel = 0
    dome = -1
    module = -1
    quabo = -1
    obs_config = config_file.get_obs_config()
    i = 1
    while i < len(argv):
        if argv[i] == '--start':
            nops += 1
            op = 'start'
        elif argv[i] == '--stop':
            nops += 1
            op = 'stop'
        elif argv[i] == '--dome':
            nsel += 1
            i += 1
            dome = int(argv[i])
        elif argv[i] == '--module':
            nsel += 1
            i += 1
            module = int(argv[i])
        elif argv[i] == '--quabo':
            nsel += 1
            i += 1
            quabo = int(argv[i])
        else:
            raise Exception('bad arg %s'%argv[i])
        i += 1

    if (nops == 0):
        raise Exception('no op specified')
    if (nops > 1):
        raise Exception('must specify a single op')
    if (nsel > 1):
        raise Exception('only one selector allowed')

    quabos = config_file.get_quabos(obs_config, dome, module, quabo)
    do_op(quabos, op)

