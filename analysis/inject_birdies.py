"""
The program uses models of the modules in an observatory and the
celestial sphere to generate birdies and simulate image mode data for a single image file.

The output is a pff file of simulated image mode data containing birdies.

TODO:
    - File IO:
        - Add utility methods to import image mode files, read their metadata and image arrays, and write RAW + birdie frames.
        - Most important metadata:
            - Module ID, module orientation (alt-az + observatory GPS), integration time, start time, and end time.
    - Setup procedure:
        - Create or update birdie log file.
        - Create a file object for the simulated image file (birdies only).
    - Main loop
        - Check if we’ve reached EOF in any of the image mode files.

"""
import math
import time
import sys
import os
import shutil
import numpy as np
import cProfile

import birdie_utils
import birdie_simulation as birdie_sim
import analysis_util

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


def do_file(data_dir, run_dir, birdie_dir, fin_name, sequence_num, params):
    """Inject birdies into a single file."""
    print('Injecting birdies into', fin_name)
    print('Setup:')
    file_attrs = pff.parse_name(fin_name)
    birdie_config = birdie_utils.get_birdie_config('birdie_config.json')
    obs_config = config_file.get_obs_config(f'{data_dir}/{run_dir}')

    # Get start and end utc timestamps
    start_iso = pff.parse_name(run_dir)['start']  # use the timestamp from the run directory name.
    with open(f'{data_dir}/{run_dir}/run_complete') as f:
        end_iso = f.readline()  # use the timestamp in "data/$run/run_complete"

    # Get timing info
    start_utc = birdie_utils.iso_to_utc(start_iso)
    end_utc = birdie_utils.iso_to_utc(end_iso)
    integration_time = birdie_utils.get_integration_time(data_dir, run_dir)
    print(f'\tstart_utc={start_utc}, end_utc={end_utc}, integration_time={integration_time} us')
    dt = end_utc-start_utc
    print(f'\trecording_time = {int(dt // 60)}:{int(dt%60)}')

    # Input and output pff file paths
    fout_name = fin_name.replace('.pff', '') + f'.birdie_{sequence_num}.pff'
    fout_path = f'{data_dir}/{birdie_dir}/{fout_name}'
    fin_path = os.path.abspath(f'{data_dir}/{run_dir}/{fin_name}')

    #print(f"fin= {fin_path},\nfout={fout_path}")
    # Create a copy of the file fin_name for birdie injection.
    with open(fin_path, 'rb') as fin, open(fout_path, 'w+b') as fout:
        print(f'\tCopying:\n\t\tFrom:\t{fin_name}\n\t\tTo:\t\t{fout_name}')
        shutil.copy(fin_path, fout_path)
        # Do simulation
        birdie_sim.do_simulation(
                data_dir,
                birdie_dir,
                start_utc,
                end_utc,
                obs_config,
                birdie_config,
                integration_time,
                fin=fin,
                fout=fout,
                noise_mean=0,
                num_updates=20,
                module_id=int(file_attrs['module']),
                plot_images=False
            )


def do_run(data_dir, run_dir, fin_name, params):
    """Run birdie injection on a real observing run."""
    if not pff.is_pff_dir(run_dir):
        print(f'"{run_dir}" is not a pff directory')
        return
    print('Processing run', run_dir)
    sequence_num = birdie_utils.get_birdie_sequence_num(data_dir, run_dir)
    birdie_dir = birdie_utils.make_birdie_dir(data_dir, run_dir, sequence_num)
    print(f'Birdie sequence_num = {sequence_num}')
    for fname in os.listdir(f'{data_dir}/{run_dir}'):
        if not pff.is_pff_file(fname):
            # Create symlinks to config and metadata files used in original run.
            os.symlink(
                f'{data_dir}/{run_dir}/{fname}',
                f'{data_dir}/{birdie_dir}/{fname}'
            )
            continue
        if pff.pff_file_type(fname) not in ('img16', 'img8'):
            continue
        do_file(data_dir, run_dir, birdie_dir, fin_name, sequence_num, params)
    print(f'Finished injecting birdies.')



DATA_DIR = '/Users/nico/Downloads/test_data'
RUN = 'obs_Lick.start_2022-10-13T00:08:12Z.runtype_eng.pffd'
fname = 'start_2022-10-13T00:08:28Z.dp_img8.bpp_1.module_1.seqno_0.pff'


def main():
    params = {
        'seconds': 1
    }
    do_run(DATA_DIR, RUN, fname, params)
    #exec('test_simulation()')
    #test_simulation()


if __name__ == '__main__':
    print("RUNNING")
    main()
    #cProfile.runctx('main()', globals(), locals(), sort='tottime')
    print("DONE")
