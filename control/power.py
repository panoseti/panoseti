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
    c = config_file.get_misc_config()
    ups = c['ups']
    url = ups['url']
    socket = ups['quabo_socket']
    value = 'true' if on else 'false'
    cmd = 'curl --silent -X PUT -H \'X-CSRF: x\' -H "Accept: application/json" --data \'value=%s\' --digest \'%s/restapi/relay/outlets/=%d/state/\''%(value, url, socket)
    os.system(cmd)

# return True if power is on
#
def quabo_power_query():
    c = config_file.get_misc_config()
    ups = c['ups']
    url = ups['url']
    socket = ups['quabo_socket']
    cmd = 'curl --silent -H "Accept: application/json" --digest \'%s/restapi/relay/outlets/=%d/state/\''%(url, socket)
    out = os.popen(cmd).read()
    return 'true' in out

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
