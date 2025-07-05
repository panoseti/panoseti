#! /usr/bin/env python3

# scan possible quabo IP addrs.
# If they respond to ping, get their UID
# write these to quabo_uids.json
#
# --exclude N    exclude quabo N (0..3) from each module

import util , sys, struct
from panoseti_tftp import tftpw
sys.path.insert(0, '../util')
import config_file
import json
import argparse
from argparse import ArgumentParser

# return quabo UID as hex string
#
def get_uid(ip_addr, port):
    x = tftpw(ip_addr, port)
    x.get_flashuid()
    with open('flashuid', 'rb') as f:
        i = struct.unpack('q', f.read(8))
        return "%x"%(i[0])

def get_uids(obs_config, network_config, exclude=[]):
    quabo_uids = {}
    quabo_uids['domes'] = []
    for d in obs_config['domes']:
        dome = {}
        dome['modules'] = []
        for m in d['modules']:
            module = {}
            module['ip_addr'] = m['ip_addr']
            module['quabos'] = []
            for i in range(4):
                uid = ''
                if i not in exclude:
                    ip_ports = util.get_quabo_ip_port(module['ip_addr'], i, network_config)
                    ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
                    real_ip = ip_ports['ip_addr']
                    port = ip_ports['reboot_port']
                    print("get uid", ip_addr)
                    # TODO: we need to ping the board before get_uid
                    uid = get_uid(real_ip, port)
                    print("%s has UID %s"%(ip_addr, uid))
                    quabo = {}
                    quabo['uid'] = uid
                    module['quabos'].append(quabo)
            dome['modules'].append(module)
        quabo_uids['domes'].append(dome)
    with open(config_file.quabo_uids_filename, "w", encoding="utf-8") as f:
        json.dump(quabo_uids, f, ensure_ascii=False, indent=4)

def check_range(val):
    ivalue = int(val)
    if ivalue < 0 or ivalue > 3:
        raise argparse.ArgumentTypeError(f"{val} is out of allowed range [0-3]")
    return ivalue
          
if __name__ == "__main__":
    parser = ArgumentParser(description="Usage for get_uids.py.")
    parser.add_argument('-e','--exclude', dest='exclude', type=check_range, nargs='+', help='List of excluded Quabos')
    args = parser.parse_args()
    obs_config = config_file.get_obs_config()
    network_config = config_file.get_network_config()
    if args.exclude == None:
        exclude = []
    else:
        exclude = args.exclude
    get_uids(obs_config, network_config, exclude)
