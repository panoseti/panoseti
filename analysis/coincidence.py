#! /usr/bin/env python3

"""
A script for finding the pairs of pulse height frames from two modules that
differ by no more than a specified number of nanoseconds.
"""

import sys
import os
import json
from collections import deque
import numpy as np
import matplotlib.pyplot as plt


sys.path.append('../util')
import pff
sys.path.append('../control')
import config_file

# Default data paths
DATA_IN_DIR = '/Users/nico/Downloads/720_ph_12pe'
DATA_OUT_DIR = '/Users/nico/panoseti/data_figures/coincidence/2022_07_20_100ns_1500pe_nexdome_and_astrograph'


#fname_a = 'start_2022-06-28T19_06_14Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff'
#fname_b = 'start_2022-06-28T19_06_14Z.dp_ph16.bpp_2.dome_0.module_254.seqno_0.pff'

# July 19
#fname_a = 'start_2022-07-20T06_44_48Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff' # astrograph 1
#fname_a = 'start_2022-07-20T06_44_48Z.dp_ph16.bpp_2.dome_0.module_254.seqno_0.pff' # astrograph 2
#fname_b = 'start_2022-07-20T06_44_48Z.dp_ph16.bpp_2.dome_0.module_3.seqno_0.pff' # nexdome

# July 20
fname_a = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff' # astrograph 1
fname_b = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff' # astrograph 1
fname_b = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_254.seqno_0.pff' # astrograph 2
#fname_b = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_3.seqno_0.pff' # nexdome

fpath_a = f'{DATA_IN_DIR}/{fname_a}'
fpath_b = f'{DATA_IN_DIR}/{fname_b}'


'''
def get_module_file(module_ip_addr):
    parsed = pff.parse_name(fname)
    mod_num = parsed['module']
'''


class QuaboFrame:
    def __init__(self, frame_num=None, json=None, img=None):
        self.frame_num = frame_num
        self.json = json
        if img:
            self.img = img
        else:
            self.img = np.zeros(256)
        self.group_num = None
        self.module_num = None
        self.quabo_num = None
        if json:
            self.module_num = json['mod_num']
            self.quabo_num = json['quabo_num']


    def __eq__(self, frame):
        if isinstance(frame, QuaboFrame):
            flag = True
            flag &= self.frame_num == frame.frame_num
            flag &= self.json == frame.json
            flag &= self.img == frame.img
            flag &= self.group_num == frame.group_num
            return flag
        else:
            return NotImplemented

    def get_quabo_name(self):
        return self.module_num * 4 + self.quabo_num

    def set_group_num(self, group_num):
        self.group_num = group_num

    def get_timestamp(self):
        """Returns a timestamp for frame."""
        tv_sec = self.json['tv_sec']
        pkt_nsec = self.json['pkt_nsec']
        return tv_sec * 10 ** 9 + pkt_nsec

    def get_timestamp_ns_diff(self, frame):
        """Returns the difference in timestamps between self and frame."""
        return self.get_timestamp() - frame.get_timestamp()

    def is_coincident(self, frame, max_time_diff):
        """Returns True iff the absolute difference between the timestamps
        of self and frame is less than or equal to max_time_diff."""
        return abs(self.get_timestamp_ns_diff(frame)) <= max_time_diff

    def a_after_b(self, frame):
        """Returns True iff the timestamp of self is greater than the timestamp of frame."""
        return self.get_timestamp_ns_diff(frame) > 0

    def get_max_pe(self):
        """Returns the maximum raw adc from the image in frame."""
        return max(self.img)

    def get_16x16_image(self):
        """Converts the 1x256 array to a 16x16 array."""
        img_16x16 = np.zeros((16, 16,))
        for row in range(16):
            for col in range(16):
                img_16x16[row][col] = self.img[16 * row + col]
        return img_16x16


class ModuleFrame:
    def __init__(self, group_num):
        self.group_num = group_num
        self.frames = [None] * 4
        self.module_num = None

    def add_quabo_frame(self, quabo_index, quabo_frame):
        assert isinstance(quabo_frame, QuaboFrame)
        if quabo_frame:
            self.module_num = quabo_frame.module_num
            quabo_frame.set_group_num(self.group_num)
            self.frames[quabo_index] = quabo_frame

    def get_32x32_image(self):
        """Return a 32x32 array image from the four 16x16 arrays fX.img:
         f0  |  f1
        ---- | ----
         f2  |  f3
        """
        imgs = list()
        for i in range(4):
            if self.frames[i]:
                imgs.append(self.frames[i].get_16x16_image())
            else:
                imgs.append(QuaboFrame().get_16x16_image())
        row_0 = np.append(imgs[0], imgs[1], axis=1)
        row_1 = np.append(imgs[2], imgs[3], axis=1)
        img_32x32 = np.append(row_0, row_1, axis=0)
        return img_32x32

    def get_max_pe(self):
        max_adc = -1
        for frame in self.frames:
            if frame:
                max_adc = max(max_adc, frame.get_max_pe())
        return max_adc


def get_next_frame(file_obj, frame_num):
    # Get the next frame from file_obj.
    j, img = None, None
    try:
        j = pff.read_json(file_obj)
        j = json.loads(j.encode())
        # Img size = 16 x 16 and bytes per pixel = 2.
        img = pff.read_image(file_obj, 16, 2)
    except Exception as e:
        # Deal with EOF issue in pff.read_json
        if repr(e)[:26] == "Exception('bad type code',":
            return None
    if not j or not img:
        return None
    frame = QuaboFrame(frame_num, j, img)
    return frame


def get_groups(path, max_group_time_diff=15):
    """Returns a dictionary of [frame_number]:[group_number] pairs where each frame is no more than
    max_group_time_diff from the others"""
    groups = dict()
    group_deque = deque()
    group_num = 0
    frame_num = 0
    def right_append(f_obj):
        nonlocal frame_num
        frame = get_next_frame(f, frame_num)
        if frame is not None:
            group_deque.append(frame)
            frame_num += 1
        return frame

    def add_entry_to_groups(frame):
        if frame.frame_num not in groups:
            groups[frame.frame_num] = group_num

    with open(path, 'rb') as f:
        prev_frame = right_append(f)
        while True:
            next_frame = right_append(f)
            if next_frame is None:
                break
            if prev_frame.is_coincident(next_frame, max_group_time_diff):
                add_entry_to_groups(prev_frame)
                add_entry_to_groups(next_frame)
            else:
                #if prev_frame.frame_num in groups:
                group_num += 1
                prev_frame = next_frame
                group_deque.popleft()
    return groups


def search_2_modules(a_path, a_groups, b_path, b_groups, max_time_diff, threshold_max_pe):
    """
    Identify all pairs of frames from the files a_path and b_path with timestamps that
    differ by no more than 100ns.
    Assumes that the timestamps in each ph file are monotonically increasing when read from top to bottom.
    Returns a list of coincident frame pairs.
    """
    pairs = list()
    a_frame_num, b_frame_num = 0, 0
    b_deque = deque()

    def b_deque_right_append_next_frame(b_file_obj):
        """Right append the next frame in b_file_obj to b_deque"""
        nonlocal b_frame_num
        b_frame = get_next_frame(b_file_obj, b_frame_num)
        if b_frame is not None:
            b_deque.append(b_frame)
            b_frame_num += 1
    with open(a_path, 'rb') as fa, open(b_path, 'rb') as fb:
        b_deque_right_append_next_frame(fb)
        while True:
            # Get the next frame for module A and check if we've reached EOF.
            a_frame = get_next_frame(fa, a_frame_num)
            print(f'Processed up to frame {a_frame_num:,}... ', end='')
            if a_frame is None:
                break
            elif a_frame.get_max_pe() < threshold_max_pe:
                print('\r', end='')
                a_frame_num += 1
                continue
            else:
                # Left pop b_deque until a coincident frame is found.
                while len(b_deque) > 0 and a_frame.a_after_b(b_deque[0]) \
                        and not a_frame.is_coincident(b_deque[0], max_time_diff):
                    b_deque.popleft()
                    # Right append frames if b_deque runs out of frames.
                    if len(b_deque) == 0:
                        b_deque_right_append_next_frame(fb)
                # Inspect every frame that appears after b_deque[0] until a non-coincident frame is found.
                right_index = 0
                while right_index < len(b_deque) and a_frame.is_coincident(b_deque[right_index], max_time_diff):
                    b_frame = b_deque[right_index]
                    if b_frame.get_max_pe() >= threshold_max_pe:
                        # Each coincident pair of frames is added to the list pairs.
                        if a_frame.frame_num in a_groups:
                            a_frame.set_group_num(a_groups[a_frame.frame_num])
                        if b_frame.frame_num in b_groups:
                            b_frame.set_group_num(b_groups[b_frame.frame_num])
                        frame_pair = a_frame, b_frame
                        if frame_pair in pairs:
                            print(f'duplicate frame pair: \n\t{frame_pair[0]}\n\t{frame_pair[1]}')
                        elif a_frame != b_frame:
                            pairs.append(frame_pair)
                    right_index += 1
                    if right_index >= len(b_deque):
                        b_deque_right_append_next_frame(fb)
            a_frame_num += 1
            print('\r', end='')
    print('Done!')
    return pairs


def style_fig(fig, fig_num, fname_a, fname_b, max_time_diff, threshold_max_pe, pair):
    # Style each figure
    parsed_a, parsed_b = pff.parse_name(fname_a), pff.parse_name(fname_b)
    title = "Pulse Height Event from Module {0} and Module {1} within {2:,} ns and max(pe) >= {3:,}.".format(
        parsed_a['module'], parsed_b['module'], max_time_diff, threshold_max_pe)
    title += "\nLeft: Dome: {0}, Module {1}; Start: {2}; Seq No: {3}, Group No: {4}".format(
        parsed_a['dome'], parsed_a['module'], parsed_a['start'], parsed_a['seqno'], pair[0].group_num
    )
    title += "\nRight: Dome: {0}, Module {1}; Start: {2}; Seq No: {3}, Group No: {4}".format(
        parsed_b['dome'], parsed_b['module'], parsed_b['start'], parsed_b['seqno'], pair[1].group_num
    )
    fig.suptitle(title)
    fig.tight_layout()
    canvas = fig.canvas
    canvas.manager.set_window_title(f'Figure {fig_num:,}')
    save_name = "fig-num_{0}.ns-diff_{1}.threshold-pe_{2}.left-module_{3}.right-module_{4}.{5}".format(
        fig_num, max_time_diff, threshold_max_pe, fname_a[:-4], fname_b[:-4], canvas.get_default_filetype()
    )
    canvas.get_default_filename = lambda: save_name


def style_ax(fig, ax, module_frame, plot):
    # Style each plot.
    ax.set_box_aspect(1)
    '''
    metadata_text = 'Mod {0}, Quabo {1}:, frame#{2:,},\npkt_num={3}, pkt_utc={4}, pkt_nsec={5},\n tv_sec={6}, tv_usec={7}'.format(
        module_frame.module_num, module_frame.get_quabo_name(), module_frame.frame_num, module_frame.json['pkt_num'],
        'N/A' if 'pkt_utc' not in module_frame.json else module_frame.json['pkt_utc'],
        module_frame.json['pkt_nsec'], module_frame.json['tv_sec'], module_frame.json['tv_usec'])
    '''
    metadata_text = 'Module '
    ax.set_title(metadata_text)
    cbar = fig.colorbar(plot, ax=ax, fraction=0.035, pad=0.05)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Photoelectrons (Raw ADC)', rotation=270)


def plot_quabo_frame(fig, ax, frame, vmax=None):
    # Draw the 2d image of a frame.
    frame_img = frame.get_16x16_image()
    plot = ax.pcolormesh(np.arange(16), np.arange(16), frame_img, vmin=0, vmax=vmax)
    style_ax(fig, ax, frame, plot)


def plot_module_frame(fig, ax, module_frame, vmax=None):
    # Draw the 2d image of a frame.
    frame_img = module_frame.get_32x32_image()
    plot = ax.pcolormesh(np.arange(32), np.arange(32), frame_img, vmin=0, vmax=module_frame.get_max_pe())
    style_ax(fig, ax, module_frame, plot)


def plot_coincident_quabos(fig_num, a, b, max_time_diff, threshold_max_pe, save_fig=False):
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    for ax, frame in zip(axs, [a, b]):
        plot_module_frame(fig, ax, frame)
    style_fig(fig, fig_num, fname_a, fname_b, max_time_diff, threshold_max_pe, a, b)
    #input(fig.canvas.get_default_filename())
    if save_fig:
        os.system(f'mkdir -p {DATA_OUT_DIR}')
        plt.savefig(f'{DATA_OUT_DIR}/{fig.canvas.get_default_filename()}')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_coincident_modules(fig_num, pair, max_time_diff, threshold_max_pe, save_fig=False):
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    for ax, module_frame in zip(axs, pair):
        plot_module_frame(fig, ax, module_frame)
    style_fig(fig, fig_num, fname_a, fname_b, max_time_diff, threshold_max_pe, pair)
    #input(fig.canvas.get_default_filename())
    if save_fig:
        os.system(f'mkdir -p {DATA_OUT_DIR}')
        plt.savefig(f'{DATA_OUT_DIR}/{fig.canvas.get_default_filename()}')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def do_search_quabo(a_path, b_path, max_time_diff=500, threshold_max_pe=0, verbose=True):
    pairs = sorted(search_2_modules(a_path, b_path, max_time_diff, threshold_max_pe))
    if len(pairs) == 0:
        print(f'No coincident frames found within {max_time_diff:,} ns of each other and with max(pe) >= {threshold_max_pe}.')
        sys.exit(0)
    do_plot = input(f'Plot {len(pairs)} figures? (Y/N): ')
    if do_plot.lower() == 'y':
        for fig_num, pair in enumerate(pairs):
            if verbose: print(f'\nFigure {fig_num:,}:')
            mod_a, mod_b = pair[0], pair[1]
            if verbose: print(f'Left module: {mod_a[1]}\nRight module : {mod_b[1]}')
            plot_coincident_quabos(fig_num, mod_a, mod_b, max_time_diff, threshold_max_pe, save_fig=False)
            fig_num += 1


def get_module_frame_pairs(quabo_frame_pairs):
    module_frame_pairs = list()
    def get_module_frames(i):
        module_frames = dict()
        for pair in quabo_frame_pairs:
            group_num = pair[i].group_num
            if group_num not in module_frames:
                module_frame = ModuleFrame(group_num)
                module_frame.add_quabo_frame(pair[i].quabo_num, pair[i])
                module_frames[group_num] = module_frame
            else:
                module_frames[group_num].add_quabo_frame(pair[i].quabo_num, pair[i])
        return module_frames
    a_module_frames = get_module_frames(0)
    b_module_frames = get_module_frames(1)
    for pair in quabo_frame_pairs:
        mp = a_module_frames[pair[0].group_num], b_module_frames[pair[1].group_num]
        if mp not in module_frame_pairs:
            module_frame_pairs.append(mp)
    return module_frame_pairs


def do_search_module(a_path, b_path, max_time_diff=100, threshold_max_pe=1000, verbose=True):
    a_groups, b_groups = get_groups(a_path), get_groups(b_path)
    quabo_frame_pairs = search_2_modules(a_path, a_groups, b_path, b_groups, max_time_diff, threshold_max_pe)
    module_frame_pairs = get_module_frame_pairs(quabo_frame_pairs)
    if len(module_frame_pairs) == 0:
        print(f'No coincident frames found within {max_time_diff:,} ns of each other and with max(pe) >= {threshold_max_pe}.')
        sys.exit(0)
    do_plot = input(f'Plot {len(module_frame_pairs)} figures? (Y/N): ')
    if do_plot.lower() == 'y':
        for fig_num, pair in enumerate(module_frame_pairs):
            if verbose: print(f'\nFigure {fig_num:,}:')
            if verbose: print(f'Left module: {pair[0]}\nRight module : {pair[1]}')
            plot_coincident_modules(fig_num, pair, max_time_diff, threshold_max_pe, save_fig=False)
            fig_num += 1


if __name__ == '__main__':
    do_search_module(fpath_a, fpath_b)
    #check_order(fpath_a)