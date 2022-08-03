#! /usr/bin/env python3

"""
Generates cumulative pulse height distributions.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt


sys.path.append('../util')
import pff
sys.path.append('../control')
import config_file

# Default data
DATA_IN_DIR = '/Users/nico/Downloads/720_ph_12pe'
DATA_OUT_DIR = '.'

fname_a = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff' # astrograph 1
# start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_254.seqno_0.pff # astrograph 2
fname_b = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_3.seqno_0.pff' # nexdome

fpath_a = f'{DATA_IN_DIR}/{fname_a}'
fpath_b = f'{DATA_IN_DIR}/{fname_b}'
'''
def get_module_file(module_ip_addr):
    parsed = pff.parse_name(fname)
    mod_num = parsed['module']
'''


def process_file(fpath, collect: list):
    img_size = 16
    bytes_per_pixel = 2
    with open(fpath, 'rb') as f:
        frame_num = 0
        while True:
            j = None
            # Deal with EOF issue pff.read_json
            try:
                j = pff.read_json(f)
            except Exception as e:
                if repr(e)[:26] == "Exception('bad type code',":
                    print('\nreached EOF')
                    break
            if not j:
                print('\nreached EOF')
                break
            print(f'Processed up to frame {frame_num}.', end='\r')
            # show_pff.print_json(j.encode(), is_ph, verbose)
            c = json.loads(j.encode())
            img = pff.read_image(f, img_size, bytes_per_pixel)
            collect.append((frame_num, c))
            frame_num += 1


def get_timestamp_lists(a_path, b_path):
    a_list, b_list = list(), list()
    process_file(a_path, a_list)
    process_file(b_path, b_list)
    return a_list, b_list


def get_timestamp(j):
    return j[1]['pkt_utc'], j[1]['pkt_nsec']


def get_timestamp_ns_diff(a, b):
    """Returns a tuple (diff in pkt_utc, diff in pkt_nsec)."""
    a_time, b_time = get_timestamp(a), get_timestamp(b)
    return a_time[0] - b_time[0], a_time[1] - b_time[1]


def a_after_b(a, b):
    """Returns True iff the timestamp of a is greater than the timestamp of b."""
    diff = get_timestamp_ns_diff(a, b)
    return (diff[0] > 0) or (diff[0] == 0 and diff[1] >= 0)


def within_threshold(a, b, max_time_diff):
    """Returns True iff the absolute difference between the timestamps
    of a and b is less than or equal to max_time_diff """
    diff = get_timestamp_ns_diff(a, b)
    is_coincident = diff[0] == 0 and abs(diff[1]) <= max_time_diff
    return is_coincident


def do_search(a_path, b_path, max_time_diff, threshold_pe):
    """
    Identify ph frames in 2 ph files that have timestamps with a difference of no more than 100ns.
    Returns a list of [frame number], [json data] pairs
    """
    pairs = list()
    a_list, b_list = get_timestamp_lists(a_path, b_path)
    '''
    for x in a_list[:15]:
        print(x)
    print("*"*5)
    for x in b_list[:15]:
        print(x)
    '''
    left = 0
    for a_tuple in a_list:
        while left < len(b_list) and not within_threshold(a_tuple, b_list[left], max_time_diff) and a_after_b(a_tuple, b_list[left]):
            left += 1
        right = left
        while left < len(b_list) and within_threshold(a_tuple, b_list[right], max_time_diff):
            pairs.append((a, b_list[right]))
            right += 1
    return pairs

pairs = do_search(fpath_a, fpath_b, 1000, 0)
for pair in pairs:
    print(f'Module A: {pair[0]}\nModule B: {pair[1]}')
    input()

'''
def usage():
    msg = "usage: coincidence.py <options> [--use-dir dir] file \tprocess ph data from a .pff file"
    msg += "\noptions:"
    msg += "\n\t--show-data" + '\t' * 6 + 'list available data files'
    msg += "\n\t--set-threshold <integer 0..4095>" + '\t' * 3 + 'set the minimum pe threshold (default is 0)'
    print(msg)


def main():
    """Process CLI inputs and dispatch actions"""
    global DATA_IN_DIR
    cmds = ['process']
    cmd = None
    ops = {
        '--use-dir': None,
        '--set-threshold': None,
        '--show-data': False,
    }
    threshold_pe = 0
    fname = None
    fpath = None
    # Process CLI commands and options
    i = 1
    argv = sys.argv
    while i < len(argv):
        if argv[i] in cmds:
            if cmd:
                'more than one command given'
                usage()
                return
            cmd = argv[i]
        elif argv[i] in ops:
            if argv[i] == '--use-dir':
                i += 1
                if i >= len(argv):
                    print('must supply a directory')
                    usage()
                    return
                ops['--use-dir'] = argv[i]
            elif argv[i] == '--set-threshold':
                i += 1
                if i >= len(argv):
                    print('must supply a number')
                    usage()
                    return
                ops['--set-threshold'] = argv[i]
            else:
                ops[argv[i]] = True
        elif i == len(argv) - 1:
            fname = argv[i]
        else:
            print(f'unrecognized input: "{argv[i]}"')
            usage()
            return
        i += 1

    if cmd is None:
        if fname is not None:
            print(f'unrecognized command: {cmd}')
        usage()
        return

    # Use new directory
    new_dir = ops['--use-dir']
    if cmd == 'process' and new_dir:
        if not os.path.isdir(new_dir):
            print(f'{new_dir} may not be a valid directory, or has a bad path')
            usage()
            return
        DATA_IN_DIR = new_dir

    # Use new threshold
    new_threshold = ops['--set-threshold']
    if new_threshold:
        if new_threshold is not None and new_threshold.isnumeric():
            threshold_pe = int(new_threshold)
        else:
            print(f'"{new_threshold}" is not a valid integer')
            usage()
            return
    if cmd == 'test':
        #do_test(threshold_pe)
        return
    # Check and parse fname
    if fname is not None:
        fpath = f'{DATA_IN_DIR}/{fname}'
        if not os.path.isfile(fpath):
            print(f'{fname} may not be a valid file, or has a bad path')
            usage()
            return
        parsed = pff.parse_name(fname)
        mod_num = parsed['module']
    elif fname is None and not ops['--show-data']:
        usage()
        return

    if cmd == 'process':
        if ops['--show-data'] and not fname:
            for f in sorted(os.listdir(DATA_IN_DIR)):
                if f[-4:] == '.pff':
                    print(f'{f}')
            return
        # Get data mode
        if fname == 'img':
            dp = 'img16'
        elif fname == 'ph':
            dp = 'ph16'
        else:
            dp = parsed['dp']
        # Get file metadata
        if dp == 'img16' or dp == '1':
            image_size = 32
            bytes_per_pixel = 2
            is_ph = False
        elif dp == 'ph16' or dp == '3':
            image_size = 16
            bytes_per_pixel = 2
            is_ph = True
        else:
            raise Exception("bad data product %s" % dp)
        # Process the data if fname is a ph file.
        if is_ph:
            process_file(fpath, image_size, bytes_per_pixel, threshold_pe)
            #do_save_data(fname[:-4])
        else:
            raise Warning(f'{fname} is not a ph file')
'''

if __name__ == '__main__':
    #main()
    ...