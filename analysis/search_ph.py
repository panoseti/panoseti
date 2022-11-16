#! /usr/bin/env python3

"""
Find and plot the coincident pulse height events between two modules with the same orientation.
"""
import sys
import json
from collections import deque

from search_ph_utils import QuaboFrame, ModuleFrame, plot_coincident_modules

sys.path.append('../util')
import pff


# Coincidence searching

def get_next_frame(file_obj, frame_num):
    """Returns the next quabo frame from file_obj."""
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
    return qf


def get_groups(path, max_group_time_diff, verbose):
    """Returns a dictionary of [frame_number]:[group_number] pairs. Within a group,
    frames are no more than max_group_time_diff from the others"""
    parsed = pff.parse_name(path)
    groups = dict()
    group_num = 0
    frame_num = 0

    def get_next(f_obj):
        """Returns the next quabo frame and increments the frame count."""
        nonlocal frame_num
        frame = get_next_frame(f, frame_num)
        if frame is not None:
            frame_num += 1
        return frame

    def add_entry_to_groups(qf):
        """Adds a [frame_number]:[group_number] entry to the dict groups."""
        if qf.frame_num not in groups:
            groups[qf.frame_num] = group_num

    with open(path, 'rb') as f:
        prev_frame = get_next(f)
        while True:
            if verbose:
                print(f"Grouped coincident frames in Module {parsed['module']} up to frame {frame_num:,}... ", end='')
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
            if verbose:
                print('\r', end='')
    if verbose:
        print('Done!')
    return groups


def search_2_modules(a_path, a_groups, b_path, b_groups, max_time_diff, threshold_max_adc, verbose):
    """
    Identify all pairs of frames from the files a_path and b_path with timestamps that
    differ by no more than 100ns.
    Assumes that the timestamps in each ph file are monotonically increasing when read from top to bottom.
    Returns a list of sorted coincident quabo frame pairs.
    """
    qf_pairs = set()
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

    def set_group_num(qf, module_index):
        """Set the group number of the quabo frame qf."""
        if module_index == 0:
            groups = a_groups
        else:
            groups = b_groups
        if qf.frame_num in groups:
            qf.group_num = groups[qf.frame_num]
        else:
            qf.group_num = none_counters[module_index]
            none_counters[module_index] += 1

    with open(a_path, 'rb') as fa, open(b_path, 'rb') as fb:
        append_next_b_frame(fb)
        while True:
            # Get the next frame for module A and check if we've reached EOF.
            a_qf = get_next_frame(fa, a_qf_num)
            if verbose:
                print(f'Searched for coincident module events up to frame {a_qf_num:,}... ', end='')
            if a_qf is None:
                break
            elif a_qf.get_max_adc() < threshold_max_adc:
                if verbose:
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
                    if b_frame.get_max_adc() >= threshold_max_adc:
                        set_group_num(a_qf, 0)
                        set_group_num(b_frame, 1)
                        # Each coincident pair of frames is added to the list pairs.
                        frame_pair = tuple((a_qf, b_frame))
                        if a_qf != b_frame:
                            qf_pairs.add(frame_pair)
                    right_index += 1
                    if right_index >= len(b_deque):
                        append_next_b_frame(fb)
            a_qf_num += 1
            if verbose:
                print('\r', end='')
    qf_pairs_sorted = sorted(qf_pairs, key=lambda p: p[0].frame_num)
    if verbose:
        print('Done!')
    return qf_pairs_sorted


def get_module_frame_pairs(quabo_frame_pairs, verbose):
    """For both modules, generate a collection of ModuleFrame objects for every frame group number.
    Then, join two ModuleFrame objects if at least one of each of their QuaboFrames appear as a pair in
    quabo_frame_pairs.
    """
    if verbose:
        print('Generating event group pairs... ', end='')

    def get_module_frames(i):
        """Initialize the module frames for the module whose
         quabo frames are at index i in each quabo frame pair."""
        module_frames = dict()
        for qf_pair in quabo_frame_pairs:
            qf = qf_pair[i]
            if qf.group_num not in module_frames:
                mf = ModuleFrame(qf.group_num)
                mf.add_quabo_frame(qf_pair[i])
                mf.set_module_event_num()
                module_frames[qf.group_num] = mf
            else:
                module_frames[qf.group_num].add_quabo_frame(qf_pair[i])
        return module_frames
    a_mfs = get_module_frames(0)
    b_mfs = get_module_frames(1)
    mf_pairs = set()
    for qf_pair in quabo_frame_pairs:
        mp = a_mfs[qf_pair[0].group_num], b_mfs[qf_pair[1].group_num]
        if mp not in mf_pairs:
            mp[0].update_paired_mfs(mp[1])
            mp[1].update_paired_mfs(mp[0])
            mf_pairs.add(mp)
    mf_pairs_sorted = sorted(mf_pairs, key=lambda mfp: mfp[0].event_num)
    if verbose:
        print('Done!')
    return mf_pairs_sorted


def do_coincidence_search(analysis_out_dir,
                          obs_config,
                          a_fname,
                          a_path,
                          b_fname,
                          b_path,
                          bytes_per_pixel,
                          max_time_diff,
                          threshold_max_adc,
                          max_group_time_diff,
                          verbose,
                          save_fig):
    """Dispatch function for finding coincidences and plotting module frames."""
    a_groups, b_groups = get_groups(a_path, max_group_time_diff, verbose), get_groups(b_path, max_group_time_diff, verbose)
    qf_pairs = search_2_modules(a_path, a_groups, b_path, b_groups, max_time_diff, threshold_max_adc, verbose)
    module_frame_pairs = get_module_frame_pairs(qf_pairs, verbose)
    if len(module_frame_pairs) == 0:
        print(f'No coincident frames found within {max_time_diff:,} ns of each other and with max(pe) >= {threshold_max_adc}.')
        sys.exit(0)
    if verbose:
        do_plot = input(f'Plot {len(module_frame_pairs)} figures? (y/n): ').lower() == 'y'
    else:
        do_plot = True
    if do_plot:
        for fig_num, mf_pair in enumerate(module_frame_pairs):
            if verbose:
                msg = '\n' + ' * ' * 3 + f' Figure {fig_num:,} ' + ' * ' * 3
                msg += f'\nLeft: {repr(mf_pair[0])}\nRight: {repr(mf_pair[1])}'
                msg += f'{mf_pair[0].get_time_diff_str(mf_pair[1])}'
                print(msg)
            plot_coincident_modules(analysis_out_dir, obs_config, a_fname, b_fname, fig_num, mf_pair, max_time_diff, threshold_max_adc, save_fig)
