#! /usr/bin/env python3

"""
Dispatch script for a signal injection and recovery program, which we call
'birdie injection' after a similarly named practice in radio astronomy.

This module handles user input, creates file structures for birdie
data, and calls the simulation routine for birdie injection.
"""
import sys
import os
import shutil
import cProfile

import birdie_utils
import birdie_simulation as birdie_sim
import analysis_util

sys.path.append('../util')
import pff
import config_file
sys.path.append('../control')


def do_file(data_dir, run_dir, birdie_dir, sequence_num, fin_name, params, verbose, plot_images):
    """Inject birdies into a single file."""
    print('\n* Injecting birdies into', fin_name)
    # Get config info.
    birdie_config = birdie_utils.get_birdie_config('birdie_config.json')
    obs_config = config_file.get_obs_config(f'{data_dir}/{run_dir}')
    file_attrs = pff.parse_name(fin_name)

    module_id = int(file_attrs['module'])

    # Create birdie log files.
    shutil.copy('birdie_config.json', f'{data_dir}/{birdie_dir}/birdie_config.json')
    birdie_log_path, birdie_sources_path = birdie_utils.make_birdie_log_files(data_dir, birdie_dir, module_id)

    # Input and output pff file paths
    fin_path = os.path.abspath(f'{data_dir}/{run_dir}/{fin_name}')
    fout_name = fin_name.replace('.pff', '') + f'.birdie_{sequence_num}.pff'
    fout_path = f'{data_dir}/{birdie_dir}/{fout_name}'

    # Note: we use a 1 byte char '*' to delimit the start of an image array.
    file_attrs = pff.parse_name(fin_name)
    bytes_per_pixel = int(file_attrs['bpp'])
    bytes_per_image = 1 + bytes_per_pixel * 1024

    # Get img info:
    with open(f'{data_dir}/{run_dir}/{fin_name}', 'rb') as fin:
        frame_size, nframes, first_unix_t, last_unix_t = pff.img_info(fin, bytes_per_image)
    # Get timing info
    start_t = first_unix_t
    end_t = min(first_unix_t + float(params['seconds']), last_unix_t)
    integration_time = birdie_utils.get_integration_time(data_dir, run_dir)
    if verbose: print(f'\tstart time (unix)={start_t}, end time (unix)={end_t},'
                      f' integration_time={integration_time} us')
    with open(fout_path, 'w+b') as fout:
        with open(fin_path, 'rb') as fin:
            # Create a copy of the file fin_name for birdie injection.
            shutil.copy(fin_path, fout_path)
        # Get nframes
        pff.time_seek(fout, integration_time * 1e-6, bytes_per_image, end_t)
        last_file_pos = fout.tell()
        fout.seek(0)
        pff.time_seek(fout, integration_time * 1e-6, bytes_per_image, start_t)
        first_file_pos = fout.tell()
        nframes = 1 + (last_file_pos - first_file_pos) / frame_size
        # Do simulation
        birdie_sim.do_simulation(
            data_dir,
            birdie_dir,
            birdie_log_path,
            birdie_sources_path,
            start_t,
            end_t,
            obs_config,
            birdie_config,
            module_id,
            bytes_per_pixel,
            fout,
            bytes_per_image,
            nframes,
            verbose,
            num_updates=20,
            plot_images=plot_images
        )


def do_run(data_dir, run_dir, params, verbose=False, plot_images=False):
    """Run birdie injection on a real or synthetic observing run."""
    if not pff.is_pff_dir(run_dir):
        print(f'"{run_dir}" is not a pff directory')
        return
    print('** Processing run', run_dir)
    files_to_process = []
    for fname in os.listdir(f'{data_dir}/{run_dir}'):
        if pff.is_pff_file(fname) and pff.pff_file_type(fname) in ('img16', 'img8'):
            files_to_process.append(fname)
    if birdie_utils.check_image_files(data_dir, run_dir, files_to_process):
        sequence_num = birdie_utils.get_birdie_sequence_num(data_dir, run_dir, verbose)
        birdie_dir = birdie_utils.make_birdie_dir(data_dir, run_dir, sequence_num)
        for fname in files_to_process:
            do_file(data_dir, run_dir, birdie_dir, sequence_num, fname, params, verbose, plot_images)
        print(f'Finished injecting birdies.')
        analysis_util.write_summary(f'{data_dir}/{birdie_dir}', params, 'TEST')
    else:
        print('No valid files for birdie injection found.')


def main():
    # Default parameters
    params = {
        'seconds': 1,
    }
    run = None
    vol = None
    argv = sys.argv
    i = 1
    while i < len(argv):
        option = argv[i].replace('--', '')
        if option in params:
            i += 1
            params[option] = argv[i]
        elif option == 'run':
            i += 1
            run = argv[i]
        elif option == 'vol':
            i += 1
            vol = argv[i]
        else:
            print(f'unrecognized input: "--{option} {argv[i]}". Options have the form: "--[option]".')
            return
        i += 1

    if not vol:
        raise Warning('no volume specified')
    if not run:
        raise Warning('no run specified')

    data_dir = f'{vol}/data'
    print("RUNNING")
    do_run(data_dir, run, params, verbose=True, plot_images=False)
    print("DONE")


# These file paths are hardcoded for program development. Will be changed later.
DATA_DIR = '/Users/nico/Downloads/test_data/obs_data'
RUN = 'obs_Lick.start_2022-10-26T20:01:33Z.runtype_eng.pffd'
#RUN = 'obs_Lick.start_2022-10-13T00:08:12Z.runtype_eng.pffd'

if __name__ == '__main__':
    #sys.argv = ['file', '--vol', DATA_DIR, '--run', RUN, '--seconds', '10']
    main()
    #cProfile.runctx('main()', globals(), locals(), sort='tottime')
