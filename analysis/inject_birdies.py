"""
Dispatch script for a signal injection and recovery program, which we call
'birdie injection' after a similarly named practice in radio astronomy.

This module handles user input, creates file structures for birdie
data, and calls the simulation routine for birdie injection.
"""
import sys
import os
import shutil
import numpy as np
import cProfile

import birdie_utils
import birdie_simulation as birdie_sim

sys.path.append('../util')
import pff
import config_file
sys.path.append('../control')

np.random.seed(301)


def test_simulation():
    start_utc = 1685417643
    end_utc = start_utc + 5
    # integration time in usec
    integration_time = 20
    birdie_config = birdie_utils.get_birdie_config('birdie_config.json')

    f = '''birdie_sim.do_simulation(
        start_utc, end_utc,
        birdie_config,
        integration_time,
        noise_mean=0,
        num_updates=30,
        plot_images=False,
    )'''
    cProfile.runctx(f, globals(), locals(), sort='tottime')


def do_file(data_dir, run_dir, fin_name, params, verbose):
    """Inject birdies into a single file."""
    print('* Injecting birdies into', fin_name)
    # Get sequence number for this run.
    sequence_num = birdie_utils.get_birdie_sequence_num(data_dir, run_dir, verbose)

    # Create birdie run directory.
    birdie_dir = birdie_utils.make_birdie_dir(data_dir, run_dir, sequence_num)
    shutil.copy('birdie_config.json', f'{data_dir}/{birdie_dir}/birdie_config.json')
    birdie_log_path, birdie_sources_path = birdie_utils.make_birdie_log_files(data_dir, birdie_dir)

    # Input and output pff file paths
    fin_path = os.path.abspath(f'{data_dir}/{run_dir}/{fin_name}')
    fout_name = fin_name.replace('.pff', '') + f'.birdie_{sequence_num}.pff'
    fout_path = f'{data_dir}/{birdie_dir}/{fout_name}'

    # Get config info.
    birdie_config = birdie_utils.get_birdie_config('birdie_config.json')
    obs_config = config_file.get_obs_config(f'{data_dir}/{run_dir}')
    file_attrs = pff.parse_name(fin_name)

    # Note: we use a 1 byte char '*' to delimit the start of an image array.
    bytes_per_pixel = int(file_attrs['bpp'])
    bytes_per_image = 1 + bytes_per_pixel * 1024

    # Get img info:
    with open(f'{data_dir}/{run_dir}/{fin_name}', 'rb') as fin:
        frame_size, nframes, first_t, last_t = pff.img_info(fin, bytes_per_image)

    # Get timing info; default is min and max filetimes.
    start_unix_t = first_t
    end_unix_t = last_t
    if 'start_t' in params:
        start_unix_t = params['start_t']
    if 'end_t' in params:
        end_unix_t = params['end_t']
    integration_time = birdie_utils.get_integration_time(data_dir, run_dir)
    if verbose: print(f'\tstart time (unix)={start_unix_t}, end time (unix)={end_unix_t},'
                      f' integration_time={integration_time} us')

    with open(fout_path, 'w+b') as fout:
        with open(fin_path, 'rb') as fin:
            # Create a copy of the file fin_name for birdie injection.
            if verbose: print(f'\tCopying:\n\t\tFrom:\t{fin_name}\n\t\tTo:\t\t{fout_name}')
            shutil.copy(fin_path, fout_path)
        # Move the file pointer to the frame closest to start_unix_t.
        pff.time_seek(fout, integration_time, bytes_per_image, start_unix_t)
        # Do simulation
        birdie_sim.do_simulation(
            data_dir,
            birdie_dir,
            birdie_log_path,
            birdie_sources_path,
            start_unix_t,
            end_unix_t,
            obs_config,
            birdie_config,
            bytes_per_pixel,
            integration_time,
            fout,
            bytes_per_image,
            nframes,
            verbose,
            module_id=int(file_attrs['module']),
            num_updates=20,
            plot_images=True
        )
    print(f'Finished injecting birdies.\n')

def do_run(data_dir, run_dir, fin_name, params, verbose=False):
    """Run birdie injection on a real observing run."""
    if not pff.is_pff_dir(run_dir):
        print(f'"{run_dir}" is not a pff directory')
        return
    print('** Processing run', run_dir, '\n')
    do_file(data_dir, run_dir, fin_name, params, verbose)


# These file paths are hardcoded for program development. Will be changed later.
DATA_DIR = '/Users/nico/Downloads/test_data/obs_data'
RUN = 'obs_Lick.start_2022-10-26T20:01:33Z.runtype_eng.pffd'
fname = 'start_2022-10-26T20_02_00Z.dp_img8.bpp_1.module_1.seqno_0.pff'


def main():
    params = {
        'seconds': 1
    }
    do_run(DATA_DIR, RUN, fname, params, verbose=False)


if __name__ == '__main__':
    print("RUNNING")
    main()
    #cProfile.runctx('main()', globals(), locals(), sort='tottime')
    print("DONE")
