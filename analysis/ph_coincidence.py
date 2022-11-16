#! /usr/bin/env python3

"""

Dispatch script for pulse-height coincidence search program.

This module handles user input, creates a ph_coincidence analysis directory,
and calls the search routine.
"""
import os
import sys
import getpass
import itertools

from search_ph import do_coincidence_search
from analysis_util import make_dir, make_analysis_dir, write_summary, ANALYSIS_TYPE_PULSE_HEIGHT_COINCIDENCE

sys.path.append('../util')
import pff
import config_file

# Control

#a_fname = 'start_2022-06-28T19_06_14Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff'
#b_fname = 'start_2022-06-28T19_06_14Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff'
#a_fname = 'start_2022-06-28T19_06_14Z.dp_ph16.bpp_2.dome_0.module_254.seqno_0.pff'

# July 19
#a_fname = 'start_2022-07-20T06_44_48Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff' # astrograph 1
#a_fname = 'start_2022-07-20T06_44_48Z.dp_ph16.bpp_2.dome_0.module_254.seqno_0.pff' # astrograph 2
#b_fname = 'start_2022-07-20T06_44_48Z.dp_ph16.bpp_2.dome_0.module_3.seqno_0.pff' # nexdome

# July 20
#a_fname = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff' # astrograph 1
#b_fname = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_254.seqno_0.pff' # astrograph 2
#b_fname = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_3.seqno_0.pff' # nexdome


def check_pair(module_to_process, module_a, module_b):
    """Return True only if:
        - module_a and module_b are in different domes.
    """
    in_different_domes = module_to_process[module_a]['dome'] != module_to_process[module_b]['dome']
    return in_different_domes


def do_pair(run_path, analysis_dir, params, obs_config, modules_to_process, bytes_per_pixel, module_a, module_b):
    print(f'Processing modules {module_a} and {module_b}.')
    a_fname = modules_to_process[module_a]['fname']
    b_fname = modules_to_process[module_b]['fname']
    a_path = f'{run_path}/{modules_to_process[module_a]["fname"]}'
    b_path = f'{run_path}/{modules_to_process[module_b]["fname"]}'
    analysis_out_dir = make_dir(f'{analysis_dir}/module_{module_a}.module_{module_b}')
    #print('program will find coincidences between:'
    #      f'\n\t{a_fname} and {b_fname}'
    #      f'\nand store the result in {analysis_out_dir} using params:'
    #      f'\n\t{params}')

    do_coincidence_search(
        analysis_out_dir,
        obs_config,
        a_fname,
        a_path,
        b_fname,
        b_path,
        bytes_per_pixel,
        params['max_time_diff'],
        params['threshold_max_adc'],
        params['max_group_time_diff'],
        params['verbose'],
        params['save_fig']
    )


def do_run(vol, run, params, username):
    analysis_dir = make_analysis_dir(ANALYSIS_TYPE_PULSE_HEIGHT_COINCIDENCE, vol, run)
    run_path = f'{vol}/data/{run}'
    obs_config = config_file.get_obs_config(dir='../control')
    bytes_per_pixel = None

    # Get filepaths to each module specified by the user.
    modules_to_process = dict()
    for f in os.listdir(run_path):
        if not pff.is_pff_file(f): continue
        t = pff.pff_file_type(f)
        if t == 'ph16':
            if bytes_per_pixel and bytes_per_pixel != 2:
                raise Warning("Ph files with different bytes per pixel found.")
            bytes_per_pixel = 2
        elif t == 'ph8':
            if bytes_per_pixel and bytes_per_pixel != 1:
                raise Warning("Ph files with different bytes per pixel found.")
            bytes_per_pixel = 1
        else:
            continue
        file_path = f'{run_path}/{f}'
        if os.path.getsize(file_path) != 0:
            file_attrs = pff.parse_name(f)
            module = int(file_attrs['module'])
            dome = int(file_attrs['dome'])
            if module in modules_to_process:
                raise Warning(f'Expected exactly one ph file for module {module} but found more than one.')
            modules_to_process[module] = dict()
            if params['modules'] == 'all_modules':
                modules_to_process[module]['fname'] = f
                modules_to_process[module]['dome'] = dome
            else:
                if module in params['modules']:
                    modules_to_process[module]['fname'] = f
                    modules_to_process[module]['dome'] = dome
    if modules_to_process:
        # Process all distinct pairs of modules in modules_to_process
        module_pairs = list(itertools.combinations(modules_to_process.keys(), 2))
        for module_a, module_b in module_pairs:
            if check_pair(modules_to_process, module_a, module_b):
                do_pair(
                    run_path,
                    analysis_dir,
                    params,
                    obs_config,
                    modules_to_process,
                    bytes_per_pixel,
                    module_a,
                    module_b
                )
            else:
                if params['verbose']:
                    print(f"Modules {module_a} and {module_b} are in the same dome. Skipping this pair.")
    else:
        print('No usable ph files found.')
    write_summary(analysis_dir, params, username)


def main():
    params = {
        'modules': [1, 3],
        'max_time_diff': 500,
        'threshold_max_adc': 0,
        'max_group_time_diff': 300,
        'verbose': False,
        'save_fig': True,
    }
    run = None
    vol = None
    username = None
    argv = sys.argv
    i = 1
    while i < len(argv):
        option = argv[i].replace('--', '', 1)
        if option in params:
            i += 1
            val = argv[i]
            if option == 'modules' and val != 'all_modules':
                try:
                    val = list(map(int, val.split(',')))
                except Exception as e:
                    msg = f'Expected either "all_modules" or a comma separated ' \
                          f'list of numbers after --module, not the given input "{val}"'
                    print(msg)
                    raise e
            elif option == 'verbose' or option == 'save_fig':
                val = val.lower() == 'true'
            else:
                val = float(val)
            params[option] = val
        elif option == 'run':
            i += 1
            run = argv[i]
        elif option == 'vol':
            i += 1
            vol = argv[i]
        elif option == 'username':
            i += 1
            username = argv[i]
        else:
            raise Warning(f'unrecognized input: "--{option} {argv[i]}". Options have the form: "--[option]".')
        i += 1

    if not run:
        raise Exception('no run specified')
    if not vol:
        raise Exception('no volume specified')
    if not username:
        username = getpass.getuser()

    do_run(vol, run, params, username)


if __name__ == '__main__':
    print('RUNNING')
    main()
    print('DONE')
