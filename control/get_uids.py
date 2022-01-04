#! /usr/bin/env python3

# scan possible quabo IP addrs.
# If they respond to ping, get their UID
# write these to quabo_uids.json
#
# --exclude N    exclude quabo N (0..3) from each module

import util, config_file, sys, struct
from panoseti_tftp import tftpw

# return quabo UID as hex string
#
def get_uid(ip_addr):
    print("get uid", ip_addr)
    x = tftpw(ip_addr)
    x.get_flashuid()
    with open('flashuid', 'rb') as f:
        i = struct.unpack('q', f.read(8))
        return "%x"%(i[0])

def get_uids(obs_config, exclude):
    f = open(config_file.quabo_uids_filename, 'w')
    f.write(
'''{
    "domes": [
''')
    dfirst = True
    for dome in obs_config['domes']:
        if not dfirst:
            f.write(',')
        dfirst = False
        f.write(
'''
        {
            "modules": [
''')
        mfirst = True
        for module in dome['modules']:
            if not mfirst:
                f.write(',')
            mfirst = False
            f.write(
'''
                {
                    "ip_addr": "%s",
                    "quabos": [
'''%(module['ip_addr']))
            for i in range(4):
                uid = ''
                if i not in exclude:
                    ip_addr = util.quabo_ip_addr(module['ip_addr'], i)
                    if util.ping(ip_addr):
                        uid = get_uid(ip_addr)
                        print("%s has UID %s"%(ip_addr, uid))
                    else:
                        print("Can't ping %s"%ip_addr)
                f.write(
'''
                        {
                            "uid": "%s"
                        }%s
'''%(uid, ('' if i==3 else ',')))
            f.write(
'''
                    ]
                }
''')
        f.write(
'''
            ]
        }
''')
    f.write(
'''
    ]
}
''')
    f.close()

def usage():
    print("usage: get_uids.py [--exclude N ...]")
    sys.exit()

def main():
    obs_config = config_file.get_obs_config()
    i = 1
    exclude = []
    while i < len(sys.argv):
        if sys.argv[i] == '--exclude':
            i += 1
            exclude.append(int(argv[i]))
        else:
            usage()
        i += 1
    get_uids(obs_config, exclude)

main()
