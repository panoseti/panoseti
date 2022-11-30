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
import re
import pprint

from search_ph import do_coincidence_search
from analysis_util import make_dir, make_analysis_dir, write_summary, ANALYSIS_TYPE_PULSE_HEIGHT_COINCIDENCE

sys.path.append('../util')
import pff
import config_file

# Control


def do_pair(run_path, analysis_dir, params, obs_config, modules_to_process, bytes_per_pixel, module_a, module_b):
    dome_a = modules_to_process[module_a]['dome']
    dome_b = modules_to_process[module_b]['dome']
    print(f'Processing the pair: Module {module_a} (Dome {dome_a}) and Module {module_b} (Dome {dome_b}).')
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


def check_all_module_pairs(available_modules, module_pairs_to_process):
    """Return True only if:
        - module_a and module_b have valid ph files in the specified run directory.
        - module_a and module_b are in different domes.
    """
    no_ph_file_error_msg = 'Module {0} does not have a valid ph file in the specified run directory.'
    not_in_diff_domes_error_msg = "Modules pairs must contain modules from different domes. " \
                                  "Modules {0} and {1} are in the same dome."
    for module_a, module_b in module_pairs_to_process:
        # Valid ph files?
        if module_a not in available_modules:
            raise Warning(no_ph_file_error_msg.format(module_a))
        elif module_b not in available_modules:
            raise Warning(no_ph_file_error_msg.format(module_b))
        # In different domes?
        if available_modules[module_a]['dome'] == available_modules[module_b]['dome']:
            raise Warning(not_in_diff_domes_error_msg.format(module_a, module_b))
    return True


def get_module_to_dome_dict(obs_config):
    """Dictionary storing pairs of [module_id]:[Dome]"""
    module_to_dome = dict()
    for dome in obs_config['domes']:
        for module in dome['modules']:
            module_ip_addr = module['ip_addr']
            module_id = config_file.ip_addr_to_module_id(module_ip_addr)
            dome_num = dome['num']
            module_to_dome[module_id] = dome_num
    return module_to_dome

def do_run(vol, run, params, username):
    run_path = f'{vol}/data/{run}'
    obs_config = config_file.get_obs_config(dir=run_path)
    module_to_dome = get_module_to_dome_dict(obs_config)
    bytes_per_pixel = None

    # Get filepaths to each module specified by the user.
    available_modules = dict()
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
            dome = module_to_dome[module]
            if module in available_modules:
                raise Warning(f'Expected exactly one ph file for module {module} but found more than one.')
            available_modules[module] = {
                'fname': f, 'dome': dome
            }
    if available_modules:
        if params['modules'] == 'all_modules':
            # Process all distinct pairs of modules in modules_to_process
            module_pairs_to_process = []
            for module_a, module_b in list(itertools.combinations(available_modules.keys(), 2)):
                if available_modules[module_a]['dome'] != available_modules[module_b]['dome']:
                    module_pairs_to_process.append((module_a, module_b))
            if len(module_pairs_to_process) == 0:
                raise Warning(f'No usable ph files found in both domes. '
                              f'Only the following modules have usable ph files:'
                              f'\n{pprint.pformat(available_modules, width=1)}')
        else:
            module_pairs_to_process = params['modules']
            check_all_module_pairs(available_modules, module_pairs_to_process)
        analysis_dir = make_analysis_dir(ANALYSIS_TYPE_PULSE_HEIGHT_COINCIDENCE, vol, run)
        print('RUNNING')
        for module_a, module_b in module_pairs_to_process:
            do_pair(
                run_path,
                analysis_dir,
                params,
                obs_config,
                available_modules,
                bytes_per_pixel,
                module_a,
                module_b
            )
        write_summary(analysis_dir, params, username)
        print('DONE')
    else:
        print('No usable ph files found.')


def main():
    params = {
        'modules': [],
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
            if option == 'modules':
                # Since --modules receives user-input, we have to ensure the input is valid.
                module_pair_pattern = re.compile("^(\d+),(\d+)$")
                while True:
                    i += 1
                    if i >= len(argv):
                        break
                    match = module_pair_pattern.match(argv[i])
                    if '--' in argv[i]:
                        if params['modules']:
                            i -= 1
                            break
                        else:
                            raise Warning('After --module, Expected either "all_modules" or a list of space-separated,'
                                          'comma-separated module number pairs (e.g. "1,3 254,3")')
                    elif match is not None:
                        if params['modules'] == 'all_modules':
                            raise Warning(f'Cannot supply both all_modules and module pairs with --modules.')
                        else:
                            module_pair = int(match.group(1)), int(match.group(2))
                            params['modules'].append(module_pair)
                    elif argv[i] == 'all_modules':
                        if params['modules'] != 'all_modules' and len(params['modules']) == 0:
                            params['modules'] = 'all_modules'
                        else:
                            raise Warning(f'Cannot supply both all_modules and module pairs with --modules.')
                    else:
                        raise Warning('After --module, Expected either "all_modules" or a list of space-separated,'
                                      'comma-separated module number pairs (e.g. "1,3 254,3"), '
                                      f'not the given input: "{argv[i]}"')
            elif option in ('verbose', 'save_fig'):
                i += 1
                params[option] = argv[i].lower() == 'true'
            else:
                i += 1
                params[option] = float(argv[i])
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
            raise Warning(f'unrecognized input: "{option}". Options have the form: "--[option]".')
        i += 1
    if not run:
        raise Warning('no run specified')
    if not vol:
        raise Warning('no volume specified')
    if not username:
        username = getpass.getuser()
    if not params['modules']:
        raise Warning('no modules specified')
    do_run(vol, run, params, username)


if __name__ == '__main__':
    try:
        main()
    except Warning as w:
        print(w)

