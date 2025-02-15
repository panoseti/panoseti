#! /usr/bin/env python3

# power.py [wps1] [on|off]
#
# on/off: turn a web power switch (WPS) on or off
# neither: query quabo power
# wps1 (or other name):
#   use the "wps1" element from obs_config.json
#   default is "wps"

# the Quabos are plugged into a particular socket in
# a Digital Loggers Inc LPC9 WPS.
# This function turns the power to this socket on or off.
# The IP addr of the WPS and the socket # come from a config file
# This can be used as a module or a script.

import sys, os
sys.path.insert(0, '../util')
import config_file

# turn power on or off
#
def quabo_power(wps, on):
    url = wps['url']
    socket = wps['quabo_socket']
    value = 'ON' if on else 'OFF'
    cmd = 'curl -s %s/outlet?%d=%s > /dev/null'%(url,socket,value)
    ret = os.system(cmd)
    if ret: raise Exception('%s returned %d'%(cmd, ret))


# return True if power is on
#
def quabo_power_query(wps):
    url = wps['url']
    socket = wps['quabo_socket']
    cmd = 'curl -s %s/status'%(url)
    out = os.popen(cmd).read()
    off = out.find('state">')
    off += len('state">')
    y = out[off:off+2]
    status = int(y, 16)
    if(status&(1<<(socket-1))):
        return 'true'

def do_wps(name, obs_config, op):
    wps = obs_config[name]
    if op == 'query':
        if quabo_power_query(wps):
            print("%s: power is on"%name)
        else:
            print("%s: power is off"%name)
    elif op == 'on':
        quabo_power(wps, True)
        print("%s: turned power on"%name)
    elif op == 'off':
        quabo_power(wps, False)
        print("%s: turned power off"%name)

def do_all(obs_config, op):
    for key in [k for k in obs_config.keys() if 'wps' in k.lower()]:
        do_wps(key, obs_config, op)

if __name__ == "__main__":
    op = 'query'
    wps_name = 'wps'
    i = 1
    while i<len(sys.argv):
        if sys.argv[i] == 'on':
            op = 'on'
        elif sys.argv[i] == 'off':
            op = 'off'
        else:
            raise Exception('usage: power.py [on|off]')
        i += 1

    c = config_file.get_obs_config()
    do_all(c, op)
