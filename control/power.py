# the Quabos are plugged into a particular socket in
# a Digital Loggers Inc LPC9 UPS.
# This function turns the power to this socket on or off.
# The IP addr of the UPS and the socket # come from a config file
# This can be used as a module or a script.

import config, sys, os

def quabo_power(on):
    c = config.get_misc_config()
    ups = c['ups']
    url = ups['url']
    socket = ups['quabo_socket']
    value = 'true' if on else 'false'
    cmd = 'curl -v -X PUT -H \'X-CSRF: x\' -H "Accept: application/json" --data \'value=%s\' --digest \'%s/restapi/relay/outlets/=%d/state/\''%(value, url, socket)
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    if sys.argv[1] == 'on':
        value = True
    elif sys.argv[1] == 'off':
        value = False
    else:
        raise Exception('bad arg')
    quabo_power(value)
