#! /usr/bin/env python3

"""
A script for finding the pairs of pulse height frames from two modules that
differ by no more than a specified number of nanoseconds.
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
DATA_IN_DIR = '/Users/nico/Downloads/719_ph_12pe'
DATA_OUT_DIR = '.'


fname_a = 'start_2022-07-20T06_44_48Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff' # astrograph 1
fname_b = 'start_2022-07-20T06_44_48Z.dp_ph16.bpp_2.dome_0.module_254.seqno_0.pff' # astrograph 2
#fname_b = 'start_2022-07-20T06_44_48Z.dp_ph16.bpp_2.dome_0.module_3.seqno_0.pff' # nexdome

# July 20
#fname_a = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff' # astrograph 1
#fname_a = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_254.seqno_0.pff' # astrograph 2
#fname_b = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_3.seqno_0.pff' # nexdome

fpath_a = f'{DATA_IN_DIR}/{fname_a}'
fpath_b = f'{DATA_IN_DIR}/{fname_b}'


'''
def get_module_file(module_ip_addr):
    parsed = pff.parse_name(fname)
    mod_num = parsed['module']
'''


def process_file(fpath, collect: list):
    """Returns COLLECT, a list of tuples in the form: [frame number], [frame json data].
    Assumes that fpath is a ph file."""
    img_size = 16
    bytes_per_pixel = 2
    print(f'Reading {fpath}')
    with open(fpath, 'rb') as f:
        frame_num = 0
        while True:
            j = None
            # Deal with EOF issue pff.read_json
            try:
                j = pff.read_json(f)
            except Exception as e:
                if repr(e)[:26] == "Exception('bad type code',":
                    print('\n\tReached EOF.')
                    break
            if not j:
                print('\n\tReached EOF.')
                break
            print(f'\tProcessed up to frame {frame_num}.', end='\r')
            c = json.loads(j.encode())
            img = pff.read_image(f, img_size, bytes_per_pixel)
            collect.append((frame_num, c, img))
            frame_num += 1


def get_frames(a_path, b_path):
    a_list, b_list = list(), list()
    process_file(a_path, a_list)
    process_file(b_path, b_list)
    return a_list, b_list


def get_timestamp(j):
    """Returns a tuple of [pkt_utc], [pkt_nsec]."""
    pkt_utc = j[1]['pkt_utc']
    pkt_nsec = j[1]['pkt_nsec']
    return pkt_utc * 10**9 + pkt_nsec


def get_timestamp_ns_diff(a, b):
    """Returns a tuple (diff in pkt_utc, diff in pkt_nsec)."""
    a_time, b_time = get_timestamp(a), get_timestamp(b)
    return a_time - b_time


def is_coincident(a, b, max_time_diff):
    """Returns True iff the absolute difference between the timestamps of a and b is less than or equal to max_time_diff."""
    within_threshold = abs(get_timestamp_ns_diff(a, b)) <= max_time_diff
    return within_threshold


def a_after_b(a, b):
    """Returns True iff the timestamp of a is greater than the timestamp of b."""
    diff = get_timestamp_ns_diff(a, b)
    return diff > 0

def do_search(a_path, b_path, max_time_diff, threshold_pe):
    """
    Identify all pairs of frames from 2 ph files that have timestamps with a difference of no more than 100ns.
    Assumes that ph frames are in monotonically increasing chronological order.
    Returns a list of [frame number], [json data] pairs.
    """
    pairs = list()
    a_frames, b_frames = get_frames(a_path, b_path)
    print(f'a_frames={len(a_frames)}, a_frames={len(a_frames)}')
    left = 0
    for a_frame in a_frames:
        # Increment the left pointer in b_frames until a coincident frame is found.
        while left < len(b_frames) \
                and a_after_b(a_frame, b_frames[left]) \
                and is_coincident(a_frame, b_frames[left], max_time_diff):
            left += 1
        right = left
        while right < len(b_frames) and is_coincident(a_frame, b_frames[right], max_time_diff):
            frame_pair = a_frame, b_frames[right]
            if frame_pair in pairs:
                print(f'duplicate frame pair: \n\t{frame_pair[0]}\n\t{frame_pair[1]}')
            pairs.append(frame_pair)
            right += 1
    return pairs

def get_image_2D(image_1D):
    """Converts a 1x256 element array to a 16x16 array."""
    rect = np.zeros((16,16,))
    for row in range(16):
        for col in range(16):
            rect[row][col] = image_1D[16 * row + col]
    return rect


def style_fig(fig, fname_a, mod_num_a, fname_b, mod_num_b, max_time_diff):
    # Add a title to the plot
    title = f'Pulse Height Event from Module {mod_num_a} and Module {mod_num_b} within {max_time_diff} ns'
    title += f'\nModule {mod_num_a} from: {fname_a}'
    title += f'\nModule {mod_num_b} from: {fname_b}'
    fig.suptitle(title)
    fig.tight_layout()
    canvas = fig.canvas


def style_ax(fig, ax, frame, plot):
    ax.set_box_aspect(1)
    #ax.set_xlabel('Pixel')
    #ax.set_ylabel('Pixel')
    metadata_text = 'Mod {0}, Quabo {1}: pkt_num={2}, \npkt_utc={3}, pkt_nsec={4},\n tv_sec={5}, tv_usec={6}'.format(
        frame[1]['mod_num'], frame[1]['quabo_num'], frame[1]['pkt_num'], frame[1]['pkt_utc'],
        frame[1]['pkt_nsec'], frame[1]['tv_sec'], frame[1]['tv_usec']
    )
    ax.set_title(metadata_text)
    cbar = fig.colorbar(plot, ax=ax, fraction=0.035, pad=0.05)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Raw ADC', rotation=270)


def plot_frame(fig, ax, frame):
    frame_img = get_image_2D(frame[2])
    plot = ax.pcolormesh(np.arange(16), np.arange(16), frame_img)
    style_ax(fig, ax, frame, plot)


def plot_coincidence(a, b, max_time_diff):
    fig, axs = plt.subplots(1, 2, figsize=(12, 7))
    for ax, frame in zip(axs, [a, b]):
        plot_frame(fig, ax, frame)
    style_fig(fig, fname_a, a[1]['mod_num'], fname_b, b[1]['mod_num'], max_time_diff)
    plt.show()


def main():
    max_time_diff = 1_000_000_000
    pairs = sorted(do_search(fpath_a, fpath_b, max_time_diff, 0))
    if len(pairs) == 0:
        print('\nNo coincident pairs found.')
        sys.exit(0)
    do_plot = input(f'\nPlot {len(pairs)} figures? (Y/N): ')
    if do_plot.lower() == 'y':
        for pair in pairs:
            mod_a, mod_b = pair[0], pair[1]
            #print(f'Module A: {mod_a[1]}\nModule B: {mod_b[1]}')
            plot_coincidence(mod_a, mod_b, max_time_diff)


if __name__ == '__main__':
    main()