#! /usr/bin/env python3

"""Utility functions for the program that finds coincident pulse-height events."""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../util')
import pff


# Matplotlib styling


def style_fig(fig, fig_num, right_ax, plot, a_file_name, b_file_name, max_time_diff, threshold_max_pe, pixel_distance, pair):
    """Style each figure."""
    parsed_a, parsed_b = pff.parse_name(a_file_name), pff.parse_name(b_file_name)
    title = "Pulse Height Event from Module {0} and Module {1} within $\pm${2:,} ns and max(pe) $\geq$ {3:,}".format(
        parsed_a['module'], parsed_b['module'], max_time_diff, threshold_max_pe
    )
    altitude_estimation = "\nApprox. (Pixel) Distance between Centroid of Event Maxima {0}".format(
        pixel_distance
    )
    time_diffs = f'{pair[0].get_time_diff_str(pair[1])}'
    fig.suptitle(title + altitude_estimation + time_diffs)
    canvas = fig.canvas
    canvas.manager.set_window_title(f'Figure {fig_num:,}')
    save_name = "event_{0}.{1}".format(
        fig_num, canvas.get_default_filetype()
    )
    canvas.get_default_filename = lambda: save_name

    cbar = fig.colorbar(plot, ax=right_ax, fraction=0.035, pad=0.05)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Raw ADC', rotation=270)


def style_ax(fig, ax, module_frame, plot):
    """Style each plot."""
    ax.set_title(str(module_frame))
    ax.invert_yaxis()
    ax.set_box_aspect(1)


# Matplotlib plotting


def plot_module_frame(fig, ax, module_frame, max_pe):
    """Plot one module frame."""
    mf_img = module_frame.get_32x32_image()
    plot = ax.pcolormesh(np.arange(32), np.arange(32), mf_img, vmin=0, vmax=max_pe)
    style_ax(fig, ax, module_frame, plot)
    return plot


def plot_coincident_modules(analysis_out_dir, obs_config, a_fname, b_fname, fig_num, mf_pair, max_time_diff, threshold_max_pe, save_fig):
    """Create a single figure displaying the 32x32 image in the coincident module frames in mf_pair."""
    parsed_a, parsed_b = pff.parse_name(a_fname), pff.parse_name(b_fname)
    max_pe = max(mf_pair[0].get_max_adc(), mf_pair[1].get_max_adc())
    pixel_distance = mf_pair[0].get_distance_between_max_adc(mf_pair[1])
    fig, axs = plt.subplots(1, 2, figsize=(14, 10), constrained_layout=True)
    plot = None
    for ax, module_frame in zip(axs, mf_pair):
        dome_index = int(parsed_a['dome'])
        module_frame.dome_name = obs_config['domes'][dome_index]['name'].title()
        plot = plot_module_frame(fig, ax, module_frame, max_pe)
    style_fig(fig, fig_num, axs[1], plot, a_fname, b_fname, max_time_diff, threshold_max_pe, pixel_distance, mf_pair)
    if save_fig:
        plt.savefig(f'{analysis_out_dir}/{fig.canvas.get_default_filename()}')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


class QuaboFrame:
    """Abstraction of a pulse height data frame."""
    start_file_seconds = dict()
    max_file_seconds = dict()

    def __init__(self, frame_num, j, img):
        self.frame_num = frame_num
        self.img = img
        self.json = j
        self.module_num = j['mod_num']
        self.quabo_index = j['quabo_num']
        self.group_num = None
        self.file_second = None
        self.img_16 = None
        self.set_file_second()

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
        s = "Quabo {0}: file_sec={1}/{2}, tv_sec={3}, pkt_nsec={4}".format(
            self.get_boardloc(), self.file_second, self.max_file_seconds[self.module_num],
            j['tv_sec'], j['pkt_nsec'],
        )
        return s

    def set_file_second(self):
        """Set the file_second field of this quabo frame and update the max file second
         for the module associated with this quabo frame."""
        if self.module_num not in self.start_file_seconds:
            self.start_file_seconds[self.module_num] = self.json['tv_sec']
        else:
            self.file_second = self.json['tv_sec'] - self.start_file_seconds[self.module_num]
            self.max_file_seconds[self.module_num] = self.file_second

    def get_boardloc(self):
        """Return the board loc of this quabo frame."""
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

    def get_max_adc(self):
        """Returns the maximum raw adc from the image in frame."""
        return max(self.img)

    def get_max_adc_pixel_offset(self):
        """Compute position offset from center of module."""
        max_adc = self.get_max_adc()
        if not self.img_16:
            self.get_16x16_image()
        pixel_index = np.nonzero(self.img_16 == max_adc)
        # The 0.5 offset makes x,y represent the coordinate of the center of the pixel rather than a vertex.
        x, y = np.mean(pixel_index[0]) + 0.5, np.mean(pixel_index[1]) + 0.5
        #print(max_adc)
        #print(x)
        #input(f'x: {x}, y: {y}')
        # Place pixel coordinate in correct quadrant.
        x_offset, y_offset = x, y
        if self.quabo_index in [0, 3]:
            x_offset = -x
        if self.quabo_index in [2, 3]:
            y_offset = -y
        #input(f'quabo_index = {self.quabo_index}, x_offset: {x_offset}, y_offset: {y_offset}')
        return x_offset, y_offset


    def get_16x16_image(self):
        """Converts the 1x256 array to a 16x16 array."""
        img_16x16 = np.zeros((16, 16,))
        for row in range(16):
            for col in range(16):
                img_16x16[row][col] = self.img[16 * row + col]
        # Rotate by 90, 180, 270 CW, depending on quabo index.
        rotated_16x16 = np.rot90(img_16x16, self.quabo_index, axes=(0,1))
        self.img_16 = img_16x16
        return img_16x16


class ModuleFrame:
    """Abstraction of a coincident event captured by 1-4 quabos in a given module."""
    event_nums = dict()

    def __init__(self, group_num):
        self.group_num = group_num
        self.frames = [None] * 4
        self.module_num = None
        self.event_num = None
        self.dome_name = None
        # List of group numbers with which this module is paired.
        self.paired_mfs = list()
        self.img = None
        self.max_adc_offset = None

    def __key(self):
        return self.group_num, self.module_num

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        s = '{0}; Module {1}; Event#{2:,}/{3:,}:'.format(
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
            quabo_frame.group_num = self.group_num
            self.module_num = quabo_frame.module_num
            self.frames[quabo_frame.quabo_index] = quabo_frame

    def update_paired_mfs(self, other_mf):
        """Record the group number of other_mf, a module frame plotted with this module frame."""
        assert other_mf.group_num not in self.paired_mfs
        self.paired_mfs.append(other_mf.group_num)

    def set_module_event_num(self):
        """Set the module event number of this module frame."""
        if self.module_num in ModuleFrame.event_nums:
            ModuleFrame.event_nums[self.module_num] += 1
        else:
            ModuleFrame.event_nums[self.module_num] = 1
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
        self.img = img_32x32
        return img_32x32

    def get_max_adc(self):
        """Get the max raw adc among the 16x16 images in the quabo frames."""
        max_adc = -1
        frame_with_max_adc = None
        for frame in self.frames:
            if frame and frame.get_max_adc() > max_adc:
                max_adc = frame.get_max_adc()
                frame_with_max_adc = frame
        self.max_adc_offset = frame_with_max_adc.get_max_adc_pixel_offset()
        return max_adc

    def get_distance_between_max_adc(self, other_mf):
        """Return the distance between the max adc pixel coord in self and other_mf."""
        if not self.max_adc_offset:
            self.get_max_adc()
        if not other_mf.max_adc_offset:
            other_mf.get_max_adc()
        x_diff = self.max_adc_offset[0] - other_mf.max_adc_offset[0]
        y_diff = self.max_adc_offset[1] - other_mf.max_adc_offset[1]
        return round((x_diff**2 + y_diff**2)**0.5, 3)

    def get_frame_names(self):
        """Returns a list of the quabo boardlocs associated with this module frame."""
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
        """Returns a formatted string displaying a 4x4 table of
         quabo frame timestamp differences in nanoseconds."""
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
        """Returns a list of the group numbers of module frames
        (from the other module) paired with this module"""
        grps = 'grouped with '
        for group_num in self.paired_mfs:
            grps += str(group_num) + ', '
        grps = grps[:-2]
        return grps


