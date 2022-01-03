#! /usr/bin/env python

# the Quabos are plugged into a particular socket in
# a Digital Loggers Inc LPC9 UPS.
# This function turns the power to this socket on or off.
# The IP addr of the UPS and the socket # come from a config file
# This can be used as a module or a script.

import config_file, sys, os

# turn power on or off
#
def quabo_power(on):
    c = config_file.get_obs_config()
    ups = c['ups']
    url = ups['url']
    socket = ups['quabo_socket']
    value = 'ON' if on else 'OFF'
    cmd = 'curl -s %s/outlet?%d=%s > /dev/null'%(url,socket,value)
    ret = os.system(cmd)
    if ret: raise Exception('%s returned %d'%(cmd, ret))


# return True if power is on
#
def quabo_power_query():
    c = config_file.get_obs_config()
    ups = c['ups']
    url = ups['url']
    socket = ups['quabo_socket']
    cmd='curl -s %s/status'%(url)
    out = os.popen(cmd).read()
    off = out.find('state">')
    off += len('state">')
    y = out[off:off+2]
    status = int(y,16)
    if(status&(1<<(socket-1))):
        return 'true'

if __name__ == "__main__":
    if len(sys.argv) < 2:
        if quabo_power_query():
            print("Quabo power is on")
        else:
            print("Quabo power is off")
    elif sys.argv[1] == 'on':
        quabo_power(True)
    elif sys.argv[1] == 'off':
        quabo_power(False)
    else:
        raise Exception('bad arg')
