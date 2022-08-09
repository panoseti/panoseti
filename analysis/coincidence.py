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

# Obs_config_filename is hardcoded as a global variable, and we must change its path for get_obs_config to work.
config_file.obs_config_filename = f'../control/{config_file.obs_config_filename}'
#plt.rcParams['text.usetex'] = True

'''
def get_module_file(module_ip_addr):
    parsed = pff.parse_name(fname)
    mod_num = parsed['module']
'''


class QuaboFrame:
    """Abstraction of a ph data frame"""
    start_file_seconds = dict()
    max_file_seconds = dict()

    def __init__(self, frame_num, json, img):
        self.frame_num = frame_num
        self.img = img
        self.json = json
        self.module_num = json['mod_num']
        self.quabo_index = json['quabo_num']
        self.group_num = None
        self.file_second = None

    def __key(self):
        return self.frame_num, self.module_num, self.quabo_index

    def __hash__(self):
        return hash(self.__key())

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

    def __repr__(self):
        r = f'Quabo {self.get_boardloc()}:\t{self.json}'
        return r

    def __str__(self):
        j = self.json
        s = "Quabo {0}: tv_sec={1}, pkt_nsec={2}, file_sec={3}/{4}".format(
            self.get_boardloc(), j['tv_sec'], j['pkt_nsec'], self.file_second,
            self.max_file_seconds[self.module_num]
        )
        return s

    def set_file_second(self):
        if self.module_num not in self.start_file_seconds:
            self.start_file_seconds[self.module_num] = self.json['tv_sec']
        else:
            self.file_second = self.json['tv_sec'] - self.start_file_seconds[self.module_num]
            self.max_file_seconds[self.module_num] = self.file_second

    def set_group_num(self, group_num):
        self.group_num = group_num

    def get_boardloc(self):
        return self.module_num * 4 + self.quabo_index

    def get_timestamp(self):
        """Returns a timestamp for this quabo frame."""
        tv_sec = self.json['tv_sec']
        pkt_nsec = self.json['pkt_nsec']
        return tv_sec * 10 ** 9 + pkt_nsec

    def get_timestamp_ns_diff(self, other_qf):
        """Returns the difference in timestamps between self and other_qf."""
        return self.get_timestamp() - other_qf.get_timestamp()

    def is_coincident(self, other_qf, max_time_diff):
        """Returns True iff the absolute difference between the timestamps
        of self and frame is less than or equal to max_time_diff."""
        return abs(self.get_timestamp_ns_diff(other_qf)) <= max_time_diff

    def a_after_b(self, other_qf):
        """Returns True iff the timestamp of self is greater than the timestamp of other_qf."""
        return self.get_timestamp_ns_diff(other_qf) > 0

    def get_max_pe(self):
        """Returns the maximum raw adc from the image in frame."""
        return max(self.img)

    def get_16x16_image(self):
        """Converts the 1x256 array to a 16x16 array."""
        img_16x16 = np.zeros((16, 16,))
        for row in range(16):
            for col in range(16):
                img_16x16[row][col] = self.img[16 * row + col]
        # Rotate by 90, 180, 270 CW, depending on quabo index.
        rotated_16x16 = np.rot90(img_16x16, self.quabo_index, axes=(0,1))
        return img_16x16


class ModuleFrame:
    event_nums = dict()

    def __init__(self, group_num):
        self.group_num = group_num
        self.frames = [None] * 4
        self.module_num = None
        self.event_num = None
        self.dome_name = None
        # List of group numbers with which this module is paired.
        self.paired_mfs = list()

    def __key(self):
        return (self.group_num, self.module_num)

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        s = '{0}; Module {1}; Event#{2}/{3}:'.format(
            self.dome_name, self.module_num, self.event_num, self.event_nums[self.module_num]
        )
        for quabo_frame in self.frames:
            s += f'\n{quabo_frame}'
        return s

    def __repr__(self):
        r = f'Module {self.module_num}; Event# {self.event_num}:'
        if len(self.paired_mfs) > 1:
            r += '\nNOTE: This module is ' + self.get_group_list_str()
        for quabo_frame in self.frames:
            r += f'\n\t{repr(quabo_frame)}'
        return r

    def add_quabo_frame(self, quabo_frame):
        """Add a quabo frame to this module frame."""
        assert isinstance(quabo_frame, QuaboFrame)
        if quabo_frame:
            quabo_frame.set_group_num(self.group_num)
            self.module_num = quabo_frame.module_num
            self.frames[quabo_frame.quabo_index] = quabo_frame

    def update_paired_mfs(self, other_mf):
        assert other_mf.group_num not in self.paired_mfs
        self.paired_mfs.append(other_mf.group_num)

    def set_module_event_num(self):
        if self.module_num in ModuleFrame.event_nums:
            ModuleFrame.event_nums[self.module_num] += 1
        else:
            ModuleFrame.event_nums[self.module_num] = 0
        self.event_num = ModuleFrame.event_nums[self.module_num]

    def get_32x32_image(self):
        """Return a 32x32 array image from the four 16x16 arrays fX.img:
         f0  |  f1
        ---- | ----
         f3  |  f2
        """
        imgs = list()
        for i in range(4):
            if self.frames[i]:
                imgs.append(self.frames[i].get_16x16_image())
            else:
                imgs.append(np.zeros((16,16)))
        row_0 = np.append(imgs[0], imgs[1], axis=1)
        row_1 = np.append(imgs[3], imgs[2], axis=1)
        img_32x32 = np.append(row_0, row_1, axis=0)
        return img_32x32

    def get_max_pe(self):
        """Get the max raw adc among the 16x16 images in the quabo frames."""
        max_adc = -1
        for frame in self.frames:
            if frame:
                max_adc = max(max_adc, frame.get_max_pe())
        return max_adc

    def get_frame_names(self):
        names = list()
        for frame in self.frames:
            name = None
            if frame:
                name = f'Q#{frame.get_boardloc()}'
            else:
                name = 'None'
            names.append(name)
        return names

    def get_time_diff_str(self, other_module_frame):
        s = '\nTime differences in nanoseconds (left module - right module)\n'
        row_format = '{:<15}' * 5
        self_frame_names = self.get_frame_names()
        other_frame_names = other_module_frame.get_frame_names()
        s += row_format.format('', *other_frame_names)
        for i in range(4):
            row_diffs = list()
            frame = self.frames[i]
            for j in range(4):
                other_frame = other_module_frame.frames[j]
                if frame and other_frame:
                    diff = frame.get_timestamp() - other_frame.get_timestamp()
                else:
                    diff = ''
                row_diffs.append(diff)
            s += '\n' + row_format.format(self_frame_names[i], *row_diffs)
        return s

    def get_group_list_str(self):
        grps = 'grouped with '
        for group_num in self.paired_mfs:
            grps += str(group_num) + ', '
        grps = grps[:-2]
        return grps


# Coincidence searching


def get_next_frame(file_obj, frame_num):
    """Get the next frame from file_obj."""
    j, img = None, None
    start_timestamp = None
    try:
        j = pff.read_json(file_obj)
        j = json.loads(j.encode())
        # For ph files: img size = 16 x 16 and bytes per pixel = 2.
        img = pff.read_image(file_obj, 16, 2)
    except Exception as e:
        # Deal with EOF issue in pff.read_json
        if repr(e)[:26] == "Exception('bad type code',":
            return None
    if not j or not img:
        return None
    qf = QuaboFrame(frame_num, j, img)
    qf.set_file_second()
    return qf


def get_groups(path, max_group_time_diff=300):
    """Returns a dictionary of [frame_number]:[group_number] pairs. Within a group,
    frames are no more than max_group_time_diff from the others"""
    groups = dict()
    group_num = 0
    frame_num = 0

    def get_next(f_obj):
        nonlocal frame_num
        frame = get_next_frame(f, frame_num)
        if frame is not None:
            frame_num += 1
        return frame

    def add_entry_to_groups(frame):
        if frame.frame_num not in groups:
            groups[frame.frame_num] = group_num

    with open(path, 'rb') as f:
        prev_frame = get_next(f)
        while True:
            print(f'Processed {path} up to frame {frame_num:,}... ', end='')
            next_frame = get_next(f)
            if next_frame is None:
                break
            if prev_frame.is_coincident(next_frame, max_group_time_diff):
                add_entry_to_groups(prev_frame)
                add_entry_to_groups(next_frame)
            else:
                if prev_frame.frame_num in groups:
                    group_num += 1
                prev_frame = next_frame
            print('\r', end='')
    print('Done!')
    return groups


def search_2_modules(a_path, a_groups, b_path, b_groups, max_time_diff, threshold_max_pe):
    """
    Identify all pairs of frames from the files a_path and b_path with timestamps that
    differ by no more than 100ns.
    Assumes that the timestamps in each ph file are monotonically increasing when read from top to bottom.
    Returns a list of coincident frame pairs.
    """
    pairs = set()
    a_qf_num, b_qf_num = 0, 0
    none_counters = [max(a_groups) + 1, max(b_groups) + 1]
    b_deque = deque()

    def append_next_b_frame(b_file_obj):
        """Right append the next frame in b_file_obj to b_deque"""
        nonlocal b_qf_num
        b_qf = get_next_frame(b_file_obj, b_qf_num)
        if b_qf is not None:
            b_deque.append(b_qf)
            b_qf_num += 1

    def set_group_num(qf, module):
        # Set the group number.
        if module == 0:
            groups = a_groups
        else:
            groups = b_groups
        if qf.frame_num in groups:
            qf.set_group_num(groups[qf.frame_num])
        else:
            qf.set_group_num(none_counters[module])
            none_counters[module] += 1


    with open(a_path, 'rb') as fa, open(b_path, 'rb') as fb:
        append_next_b_frame(fb)
        while True:
            # Get the next frame for module A and check if we've reached EOF.
            a_qf = get_next_frame(fa, a_qf_num)
            print(f'Searched up to frame {a_qf_num:,}... ', end='')
            if a_qf is None:
                break
            elif a_qf.get_max_pe() < threshold_max_pe:
                print('\r', end='')
                a_qf_num += 1
                continue
            else:
                # Left pop b_deque until a coincident frame is found.
                while len(b_deque) > 0 and a_qf.a_after_b(b_deque[0]) \
                        and not a_qf.is_coincident(b_deque[0], max_time_diff):
                    b_deque.popleft()
                    # Right append frames if b_deque runs out of frames.
                    if len(b_deque) == 0:
                        append_next_b_frame(fb)
                # Inspect every frame that appears after b_deque[0] until a non-coincident frame is found.
                right_index = 0
                while right_index < len(b_deque) and a_qf.is_coincident(b_deque[right_index], max_time_diff):
                    b_frame = b_deque[right_index]
                    if b_frame.get_max_pe() >= threshold_max_pe:
                        set_group_num(a_qf, 0)
                        set_group_num(b_frame, 1)
                        # Each coincident pair of frames is added to the list pairs.
                        frame_pair = tuple((a_qf, b_frame))
                        if a_qf != b_frame:
                            pairs.add(frame_pair)
                    right_index += 1
                    if right_index >= len(b_deque):
                        append_next_b_frame(fb)
            a_qf_num += 1
            print('\r', end='')
    print('Done!')
    return pairs


# Matplotlib styling


def style_fig(fig, fig_num, a_file_name, b_file_name, max_time_diff, threshold_max_pe, pair):
    """Style each figure."""
    parsed_a, parsed_b = pff.parse_name(a_file_name), pff.parse_name(b_file_name)
    title = "Pulse Height Event from Module {0} and Module {1} within $\pm${2:,} ns and max(pe) $\geq$ {3:,}".format(
        parsed_a['module'], parsed_b['module'], max_time_diff, threshold_max_pe
    )
    time_diffs = f'{pair[0].get_time_diff_str(pair[1])}'
    fig.suptitle(title + time_diffs)
    fig.tight_layout()
    canvas = fig.canvas
    canvas.manager.set_window_title(f'Figure {fig_num:,}')
    save_name = "fig-num_{0}.ns-diff_{1}.threshold-pe_{2}.left-module_{3}.right-module_{4}.{5}".format(
        fig_num, max_time_diff, threshold_max_pe, a_file_name[:-4], b_file_name[:-4], canvas.get_default_filetype()
    )
    canvas.get_default_filename = lambda: save_name


def style_ax(fig, ax, module_frame, plot):
    """Style each plot."""
    ax.set_title(str(module_frame))
    cbar = fig.colorbar(plot, ax=ax, fraction=0.035, pad=0.05)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Photoelectrons (Raw ADC)', rotation=270)
    ax.invert_yaxis()
    #ax.set_axis_off()
    ax.set_box_aspect(1)


# Matplotlib plotting


def plot_module_frame(fig, ax, module_frame, max_pe):
    # Draw the 2d image of a frame.
    mf_img = module_frame.get_32x32_image()
    plot = ax.pcolormesh(np.arange(32), np.arange(32), mf_img, vmin=0, vmax=max_pe)
    style_ax(fig, ax, module_frame, plot)


def plot_coincident_modules(a_file_name, b_file_name, fig_num, pair, max_time_diff, threshold_max_pe, save_fig):
    obs_config = config_file.get_obs_config()
    parsed_a, parsed_b = pff.parse_name(a_file_name), pff.parse_name(b_file_name)
    max_pe = max(pair[0].get_max_pe(), pair[1].get_max_pe())
    fig, axs = plt.subplots(1, 2, figsize=(14, 10))
    for ax, module_frame in zip(axs, pair):
        dome_index = int(parsed_a['dome'])
        module_frame.dome_name = obs_config['domes'][dome_index]['name'].title()
        plot_module_frame(fig, ax, module_frame, max_pe)
    style_fig(fig, fig_num, a_file_name, b_file_name, max_time_diff, threshold_max_pe, pair)
    if save_fig:
        os.system(f'mkdir -p {DATA_OUT_DIR}')
        plt.savefig(f'{DATA_OUT_DIR}/{fig.canvas.get_default_filename()}')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def get_module_frame_pairs(quabo_frame_pairs):
    """For both modules, generate a collection of ModuleFrame objects for every frame group number.
    Then, join two ModuleFrame objects if at least one of each of their QuaboFrames appear as a pair in
    quabo_frame_pairs.
    """
    def get_module_frames(i):
        # Initialize Module Frames
        event_num = 0
        module_frames = dict()
        for pair in quabo_frame_pairs:
            qf = pair[i]
            if qf.group_num not in module_frames:
                mf = ModuleFrame(qf.group_num)
                mf.add_quabo_frame(pair[i])
                mf.set_module_event_num()
                module_frames[qf.group_num] = mf
            else:
                module_frames[qf.group_num].add_quabo_frame(pair[i])
        return module_frames
    a_mfs = get_module_frames(0)
    b_mfs = get_module_frames(1)
    mf_pairs = set()
    for pair in quabo_frame_pairs:
        mp = a_mfs[pair[0].group_num], b_mfs[pair[1].group_num]
        if mp not in mf_pairs:
            mp[0].update_paired_mfs(mp[1])
            mp[1].update_paired_mfs(mp[0])
            mf_pairs.add(mp)
    return mf_pairs


# Control


def get_file_path(file_name):
    return f'{DATA_IN_DIR}/{file_name}'


def do_coincidence_search(a_file_name, b_file_name, max_time_diff=500, threshold_max_pe=1000, verbose=True, save_fig=False):
    """
    Dispatch function for finding coincidences and plotting module frames.
    """
    a_path, b_path = get_file_path(a_file_name), get_file_path(b_file_name)
    a_groups, b_groups = get_groups(a_path), get_groups(b_path)
    qfps = search_2_modules(a_path, a_groups, b_path, b_groups, max_time_diff, threshold_max_pe)
    module_frame_pairs_unsorted = get_module_frame_pairs(sorted(qfps, key=lambda p: p[0].frame_num))
    module_frame_pairs = sorted(module_frame_pairs_unsorted, key=lambda mfp: mfp[0].event_num)
    if len(module_frame_pairs) == 0:
        print(f'No coincident frames found within {max_time_diff:,} ns of each other and with max(pe) >= {threshold_max_pe}.')
        sys.exit(0)
    do_plot = input(f'Plot {len(module_frame_pairs)} figures? (y/n): ')
    if do_plot.lower() == 'y':
        for fig_num, pair in enumerate(module_frame_pairs):
            if verbose:
                msg = '\n' + ' * ' * 3 + f' Figure {fig_num:,} ' + ' * ' * 3
                msg += f'\nLeft: {repr(pair[0])}\nRight: {repr(pair[1])}'
                msg += f'{pair[0].get_time_diff_str(pair[1])}'
                print(msg)
            plot_coincident_modules(a_file_name, b_file_name, fig_num, pair, max_time_diff, threshold_max_pe, save_fig)
            fig_num += 1



# Default data paths
DATA_IN_DIR = '/Users/nico/Downloads/720_ph_12pe'
DATA_OUT_DIR = '/Users/nico/panoseti/data_figures/coincidence/2022_07_20_500ns_0pe_nexdome_and_astrograph_254'


#a_fname = 'start_2022-06-28T19_06_14Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff'
#b_fname = 'start_2022-06-28T19_06_14Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff'
#b_fname = 'start_2022-06-28T19_06_14Z.dp_ph16.bpp_2.dome_0.module_254.seqno_0.pff'

# July 19
#a_fname = 'start_2022-07-20T06_44_48Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff' # astrograph 1
#a_fname = 'start_2022-07-20T06_44_48Z.dp_ph16.bpp_2.dome_0.module_254.seqno_0.pff' # astrograph 2
#b_fname = 'start_2022-07-20T06_44_48Z.dp_ph16.bpp_2.dome_0.module_3.seqno_0.pff' # nexdome

# July 20
#a_fname = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff' # astrograph 1
a_fname = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_254.seqno_0.pff' # astrograph 2
b_fname = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_3.seqno_0.pff' # nexdome

if __name__ == '__main__':
    do_coincidence_search(a_fname, b_fname, max_time_diff=500, threshold_max_pe=0, verbose=True, save_fig=False)
